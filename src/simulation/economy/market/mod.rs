pub(in crate::simulation::economy) mod feedback;
pub(in crate::simulation) mod indexing;
pub(in crate::simulation) mod orders;
pub(in crate::simulation::economy) mod pricing;
pub(in crate::simulation::economy) mod proposals;
pub(in crate::simulation) mod search;
pub(in crate::simulation) mod sorting;

use crate::simulation::economy::*;

#[derive(CubeType)]
pub(super) struct MarketControl<'a> {
    control: &'a [Atomic<u32>],
    parameters: &'a EconomyParameters,
}

#[cube]
impl<'a> MarketControl<'a> {
    pub(super) fn new(control: &'a [Atomic<u32>], parameters: &'a EconomyParameters) -> Self {
        MarketControl::<'a> {
            control,
            parameters,
        }
    }

    pub(super) fn is_open(&self) -> bool {
        self.control[LATCH_WORD].load() == 0
    }

    pub(super) fn boundary(&self, outcomes: &mut [u32], #[comptime] stage: EconomyStage, id: u32) {
        if id == 0 {
            self.control[LATCH_WORD]
                .store(self.control[crate::simulation::flags::FAULT_WORD].load());
            if comptime!(stage != EconomyStage::MarketBoundary) {
                self.control[CONSTRUCTED_EVENTS_WORD].store(0);
                self.control[RANGE_SLOTS_WORD].store(0);
                self.control[orders::SEARCHES_WORD].store(0);
                self.control[search::INPUT_MINIMUM_QUOTE_WORD].store(u32::MAX);
                if comptime!(matches!(self.parameters.market, MarketFlag::Input(_))) {
                    self.control[search::INPUT_RAIL_WORD].store(0);
                }
                if comptime!(
                    self.parameters.market
                        == MarketFlag::Input(crate::simulation::flags::InputMarket::Grain)
                ) {
                    self.control[search::GRAIN_FIRMS_WORD].store(0);
                }
                if comptime!(
                    self.parameters.empty_orders
                        == crate::simulation::parameters::EmptyOrderMode::FeedbackOnly
                ) {
                    self.control[orders::WIDE_ORDERS_WORD].store(0);
                }
            }
            if comptime!(matches!(
                stage,
                EconomyStage::InputMarketStart | EconomyStage::InputMarketPrepare
            )) {
                self.control[orders::PREPARATION_ACTORS_WORD].store(0);
            }
        }
        if comptime!(matches!(
            stage,
            EconomyStage::InputMarketStart | EconomyStage::InputMarketRepeat
        )) && self.control[crate::simulation::flags::FAULT_WORD].load() == 0
            && id < self.parameters.count
        {
            KernelEventStatus::new(comptime!(EventStatus::Inactive))
                .store(outcomes, 4 * self.parameters.count + id);
        }
    }

    pub(super) fn record_activity(&self, proposals: &[u32], actors: u32, id: u32) {
        let mut active = 0u32;
        if id < actors {
            active = u32::cast_from(
                proposals[id as usize * ProposalLayout::WORDS + ProposalLayout::ACTOR] != NO_ACTOR,
            );
        }
        let total = parallel::reduce::<u32>(active, comptime!(Reduction::Sum));
        if UNIT_POS == 0 && total != 0 {
            self.control[CONSTRUCTED_EVENTS_WORD].fetch_add(total);
        }
    }
}

pub(super) mod demand;
pub(in crate::simulation::economy) mod distribution;
pub(in crate::simulation) mod farm_offers;
pub(in crate::simulation::economy) mod feedback;
pub(in crate::simulation) mod indexing;
pub(in crate::simulation) mod orders;
pub(in crate::simulation::economy) mod pricing;
pub(in crate::simulation::economy) mod proposals;
pub(in crate::simulation) mod search;
pub(in crate::simulation) mod sorting;

use crate::simulation::economy::*;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(in crate::simulation) enum MarketStart {
    Standard,
    PrepareCandidates,
    ResetCandidatesAndOrders,
    ResetOrders,
}

impl MarketStart {
    pub(in crate::simulation) fn launch(
        self,
        buffers: crate::simulation::storage::buffers::EconomyBuffers<'_>,
        parameters: &EconomyParameters,
    ) {
        let crate::simulation::DeviceEconomy { device, config, .. } = buffers.economy;
        let width = device.workgroup;
        let count = match self {
            Self::ResetCandidatesAndOrders | Self::ResetOrders => {
                device.count(parameters.count as usize)
            }
            Self::Standard | Self::PrepareCandidates => CubeCount::Static(1, 1, 1),
        };
        start_market::launch(
            &device.client,
            count,
            CubeDim::new_1d(width),
            buffers.control.buffer_arg(),
            buffers.events.buffer_arg(),
            config.market_table.buffer_arg(),
            parameters.launch(),
            config.heating_per_person,
            self,
        );
    }
}

#[cube(launch)]
fn start_market(
    control: &[Atomic<u32>],
    outcomes: &mut [u32],
    clock: &mut [u32],
    parameters: EconomyParameters,
    heating: u32,
    #[comptime] start: MarketStart,
) {
    let id = ABSOLUTE_POS as u32;
    if id == 0 {
        parameters.write_clock(clock, heating);
        control[LATCH_WORD].store(control[crate::simulation::storage::flags::FAULT_WORD].load());
        control[CONSTRUCTED_EVENTS_WORD].store(0);
        control[RANGE_SLOTS_WORD].store(0);
        control[orders::SEARCHES_WORD].store(0);
        if comptime!(parameters.market != MarketFlag::Input(InputMarket::Grain))
            || parameters.matching_round == 0
        {
            control[search::INPUT_MINIMUM_QUOTE_WORD].store(u32::MAX);
        }
        if comptime!(matches!(parameters.market, MarketFlag::Input(_))) {
            control[search::INPUT_RAIL_WORD].store(0);
        }
        if comptime!(parameters.market == MarketFlag::Input(InputMarket::Grain))
            && parameters.matching_round == 0
        {
            control[search::GRAIN_FIRMS_WORD].store(0);
        }
        if comptime!(
            parameters.empty_orders
                == crate::simulation::economy::parameters::EmptyOrderMode::FeedbackOnly
        ) {
            control[orders::WIDE_ORDERS_WORD].store(0);
        }
        if comptime!(matches!(
            start,
            MarketStart::ResetCandidatesAndOrders | MarketStart::PrepareCandidates
        )) {
            control[orders::PREPARATION_ACTORS_WORD].store(0);
        }
    }
    if comptime!(matches!(
        start,
        MarketStart::ResetCandidatesAndOrders | MarketStart::ResetOrders
    )) && id < parameters.count
        && control[crate::simulation::storage::flags::FAULT_WORD].load() == 0
    {
        KernelEventStatus::new(comptime!(EventStatus::Inactive))
            .store(outcomes, 4 * parameters.count + id);
    }
}

#[derive(CubeType)]
pub(super) struct MarketControl<'a> {
    control: &'a [Atomic<u32>],
}

#[cube]
impl<'a> MarketControl<'a> {
    pub(super) fn new(control: &'a [Atomic<u32>]) -> Self {
        MarketControl::<'a> { control }
    }

    pub(super) fn is_open(&self) -> bool {
        self.control[LATCH_WORD].load() == 0
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

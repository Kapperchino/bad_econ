use super::*;

#[derive(CubeType, CubeLaunch)]
pub(in crate::simulation) struct EconomyInput {
    pub state: Box<[u32]>,
    pub actors: Box<[u32]>,
    pub config: EconomyConfig,
    pub spatial: Box<[u32]>,
    pub proposals: Box<[u32]>,
    pub bundles: Box<[u32]>,
    pub control: Box<[Atomic<u32>]>,
    pub parameters: EconomyParameters,
    #[cube(comptime)]
    pub stage: EconomyStage,
}

pub(in crate::simulation) struct EconomySystem;
pub(in crate::simulation) struct AdvanceEconomy;

impl System for EconomySystem {
    type Input = EconomyInput;
    type Passes = gpu_passes![AdvanceEconomy];
}

#[cube]
impl GpuPass<EconomyInput> for AdvanceEconomy {
    type Output = ();

    fn execute(mut input: EconomyInput) {
        input.parameters.refresh(&mut input.config);
        input.dispatch();
    }
}

#[cube]
impl EconomyInput {
    fn dispatch(&mut self) {
        let mut view = EconomyView::new(&mut self.state, &self.actors, &self.config);
        let economy = &mut view;
        let config = &self.config;
        let spatial: &mut [u32] = &mut self.spatial;
        let proposals: &mut [u32] = &mut self.proposals;
        let bundles: &mut [u32] = &mut self.bundles;
        let control: &[Atomic<u32>] = &self.control;
        let parameters = &self.parameters;

        let market = market::MarketControl::new(control);
        let id = ABSOLUTE_POS as u32;
        let guarded = comptime!(
            self.parameters.discovery == Discovery::Indexed && self.stage.requires_open_market()
        );
        if !guarded || market.is_open() {
            match comptime!(self.stage) {
                EconomyStage::InputQuotes => {
                    pricing::ReplacementMarket::new(&*spatial, control, parameters.day)
                        .refresh(economy, id)
                }
                EconomyStage::MarketBoundary => {
                    if id == 0 {
                        control[LATCH_WORD]
                            .store(control[crate::simulation::storage::flags::FAULT_WORD].load());
                    }
                }
                EconomyStage::ProposeCompact | EconomyStage::ProposeRetainedCompact => {
                    let actor = (id + parameters.settlement_rotation) % max(parameters.count, 1);
                    let peer = proposals::propose(
                        economy,
                        &*spatial,
                        proposals,
                        control,
                        parameters,
                        select(id < parameters.count, actor, config.actors),
                        comptime!(match self.stage {
                            EconomyStage::ProposeRetainedCompact =>
                                employment::LaborRound::Retained,
                            _ => employment::LaborRound::Vacancies,
                        }),
                    );
                    orders::compact_proposal(peer, bundles, parameters, id);
                }
                EconomyStage::GroupCompactFill => {
                    if control[crate::simulation::storage::flags::FAULT_WORD].load() == 0 {
                        orders::compact_fill(proposals, bundles, control, parameters, id);
                    }
                }
                EconomyStage::GroupRadixLocal => {
                    orders::radix_local(bundles, control, parameters, id)
                }
                EconomyStage::GroupRadixScatter => {
                    orders::radix_scatter(bundles, control, parameters, id)
                }
                EconomyStage::SupplierRanges => {
                    orders::OrderRanges::new(bundles, control, parameters)
                        .record_supplier(&*spatial, config, id)
                }
                EconomyStage::GroupRanges => {
                    if control[crate::simulation::storage::flags::FAULT_WORD].load() == 0 {
                        orders::OrderRanges::new(bundles, control, parameters)
                            .record_order(config, id);
                    }
                }
                EconomyStage::ClearFeedback => {
                    if id < control.len() as u32 {
                        control[id as usize].store(0);
                    }
                }
                EconomyStage::RecordFeedback => feedback::record_feedback(
                    &mut economy.ledger,
                    &*spatial,
                    proposals,
                    bundles,
                    control,
                    parameters,
                    id,
                ),
                EconomyStage::ApplyFeedback => {
                    feedback::apply_feedback(&mut economy.ledger, control, id)
                }
                EconomyStage::IndexCount => {
                    indexing::index_count(&*economy, spatial, bundles, parameters, id)
                }
                EconomyStage::IndexOffsets => {
                    indexing::index_offsets(config, spatial, bundles, parameters, id)
                }
                EconomyStage::IndexFill => {
                    indexing::index_fill(config, spatial, bundles, parameters, id)
                }
                EconomyStage::IndexPrices => indexing::index_prices(&economy.ledger, spatial),
                EconomyStage::FirmRent => rent::collect(economy, control, id, rent::Tenants::Firms),
                EconomyStage::HouseholdRent => {
                    rent::collect(economy, control, id, rent::Tenants::Households)
                }
                EconomyStage::Local => phases::ActorDay::new(id, parameters.day, control)
                    .advance(economy, comptime!(parameters.phase)),
                EconomyStage::Propose => {
                    proposals::propose(
                        economy,
                        &*spatial,
                        proposals,
                        control,
                        parameters,
                        id,
                        comptime!(employment::LaborRound::FromClock),
                    );
                }
                EconomyStage::MarketActivity => {
                    market.record_activity(&*proposals, config.actors, id)
                }
                EconomyStage::Construct => construct(
                    economy,
                    proposals,
                    bundles,
                    control,
                    parameters,
                    id,
                    comptime!(EventStorage::Buffer),
                ),
                EconomyStage::GroupInitialize
                | EconomyStage::GroupSort
                | EconomyStage::GroupMerge
                | EconomyStage::SettleInputTile
                | EconomyStage::GroupExecute
                | EconomyStage::GroupRemainder => {
                    let mut orders =
                        groups::GroupedOrders::new(proposals, bundles, control, parameters);
                    orders.dispatch(economy, comptime!(self.stage), id);
                }
                EconomyStage::GroupPrepare => actor_settlement(
                    economy,
                    proposals,
                    bundles,
                    control,
                    parameters,
                    id,
                    comptime!(Posting::CheckActor),
                ),
                EconomyStage::GroupFinalize | EconomyStage::FinalizeFixedWages => {
                    if comptime!(self.stage == EconomyStage::FinalizeFixedWages)
                        && id < orders::active(control, parameters)
                    {
                        let count = self.parameters.count;
                        let peer = bundles[(parameters.sort_source + id) as usize];
                        if id >= bundles[(7 * count + peer) as usize] {
                            let actor = bundles[(parameters.sort_source + count + id) as usize];
                            let prior = KernelEventStatus::load(bundles, 4 * count + actor);
                            let mut status =
                                KernelEventStatus::new(comptime!(EventStatus::Rejected));
                            if prior.contains(comptime!(EventStatus::Faulted)) {
                                status.set(comptime!(EventStatus::Faulted));
                            }
                            status.store(bundles, 4 * count + actor);
                            proposals[(actor + parameters.offset) as usize
                                * ProposalLayout::WORDS
                                + ProposalLayout::QUANTITY_HIGH] = 0;
                        }
                    }
                    actor_settlement(
                        economy,
                        proposals,
                        bundles,
                        control,
                        parameters,
                        id,
                        comptime!(Posting::ApplyActor),
                    );
                }
                EconomyStage::Summarize => economy.summarize(bundles, control, id),
                EconomyStage::Inspect => economy.ledger.inspect(proposals, parameters, id),
            }
        }
    }
}

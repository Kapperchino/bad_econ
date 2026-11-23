use super::demand::FoodSearchResult;
use crate::simulation::configuration::Configuration;
use crate::simulation::economy::*;
use crate::simulation::{
    flags::InputMarket,
    orders::{LATCH_WORD, SEARCHES_WORD},
};
mod candidates;
mod demand;
mod matching;
mod prepare;
mod ranking;
mod spatial;
use candidates::*;
pub(in crate::simulation::economy) use candidates::{
    OfferRequest, supplier_samples, supplier_stride, supply_score,
};
use demand::BuyerDemand;
use ranking::{CandidateRank, group_winner};
pub(super) use spatial::SearchAxis;
mod empty;
mod filter;
mod fuel;
pub(in crate::simulation) mod grain;
mod index;
mod road;
pub(in crate::simulation) use road::SIDE as ROAD_TILE_SIDE;

pub(in crate::simulation) const INPUT_MINIMUM_QUOTE_WORD: usize = CONTROL_WORDS + 7;
pub(in crate::simulation) const INPUT_RAIL_WORD: usize = orders::WIDE_ORDERS_WORD;
pub(in crate::simulation) const INPUT_BUYERS_FIELD: u32 = 12;
pub(in crate::simulation::economy) const GRAIN_FIRMS_WORD: usize = CONTROL_WORDS + 8;

#[cfg(test)]
mod tests;

#[derive(Clone, Copy)]
#[repr(u32)]
enum Preparation {
    Demand,
    Inventory,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PreparationRound {
    First,
    Capture,
    Selected,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(in crate::simulation) enum StatusReset {
    Full,
    Active,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(in crate::simulation) enum SearchStage {
    Index,
    EmptySort,
    EmptyBounds,
    Prepare,
    PrepareCandidates,
    PrepareSelected,
    Search,
    SearchIndexed,
    GrainFirms,
    GrainDealerTiles,
    SearchGrain,
    #[cfg_attr(not(test), allow(dead_code))]
    SearchFuel,
    #[cfg_attr(not(test), allow(dead_code))]
    SearchHousehold,
    SearchSpatialFuel,
    FilterEmpty,
    FilterBuyers,
    StockLimits,
    RoadCandidates,
    FinishFiltered,
    Compact,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(in crate::simulation::economy) enum SupplierSearch {
    Inactive,
    SingleCell { shipment_speed: u32 },
    Neighborhood { shipment_speed: u32 },
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(in crate::simulation) enum OrderCollection {
    Bitmap,
    InputTile,
    ConsumerMask,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(in crate::simulation) struct SearchProgram {
    collection: OrderCollection,
    stage: SearchStage,
    suppliers: SupplierSearch,
}

impl SearchProgram {
    pub(in crate::simulation) fn collecting(self, collection: OrderCollection) -> Self {
        Self { collection, ..self }
    }
}

impl SearchStage {
    pub(in crate::simulation) fn program(self, config: &Configuration) -> SearchProgram {
        let shipment_speed = config.shipment_speed;
        let suppliers = match self {
            Self::Search
            | Self::SearchGrain
            | Self::SearchIndexed
            | Self::SearchFuel
            | Self::SearchHousehold
            | Self::SearchSpatialFuel => {
                match config.spatial_side == 1 && config.map_size <= config.search_radius {
                    true => SupplierSearch::SingleCell { shipment_speed },
                    false => SupplierSearch::Neighborhood { shipment_speed },
                }
            }
            _ => SupplierSearch::Inactive,
        };
        SearchProgram {
            collection: OrderCollection::Bitmap,
            stage: self,
            suppliers,
        }
    }
}

pub(in crate::simulation) fn empty_bounds_invocations(
    members: usize,
    mode: parallel::ScanMode,
) -> usize {
    let lanes = match mode {
        parallel::ScanMode::Plane => 32,
        parallel::ScanMode::Shared => 1,
    };
    members
        .min(empty::LIMIT as usize)
        .div_ceil(empty::BLOCK as usize)
        * lanes
}

#[derive(CubeType, CubeLaunch)]
pub(in crate::simulation) struct SearchInput {
    pub state: Box<[u32]>,
    pub actors: Box<[u32]>,
    pub config: EconomyConfig,
    pub spatial: Box<[u32]>,
    pub proposals: Box<[u32]>,
    pub searches: Box<[u32]>,
    pub successes: Box<[Atomic<u32>]>,
    pub scratch: Box<[u32]>,
    pub control: Box<[Atomic<u32>]>,
    pub parameters: EconomyParameters,
    #[cube(comptime)]
    pub status_reset: StatusReset,
    #[cube(comptime)]
    pub program: SearchProgram,
}

pub(in crate::simulation) struct MarketSearch;
pub(in crate::simulation) struct Search;

impl System for MarketSearch {
    type Input = SearchInput;
    type Passes = gpu_passes![Search];
}

#[cube]
impl GpuPass<SearchInput> for Search {
    type Output = ();

    fn execute(mut input: SearchInput) {
        match comptime!(input.program.suppliers) {
            SupplierSearch::SingleCell { shipment_speed } => {
                input.config.spatial_side = 1;
                input.config.shipment_speed = shipment_speed;
            }
            SupplierSearch::Neighborhood { shipment_speed } => {
                input.config.shipment_speed = shipment_speed;
            }
            SupplierSearch::Inactive => (),
        }
        input.parameters.refresh(&mut input.config);
        if input.control[LATCH_WORD].load() == 0 {
            match comptime!(input.program.stage) {
                SearchStage::Index => input.build_supplier_index(),
                SearchStage::EmptySort => empty::sort(
                    &input.config,
                    &input.spatial,
                    &mut input.scratch,
                    &mut input.searches,
                    comptime!(input.parameters.scan),
                ),
                SearchStage::EmptyBounds => empty::bounds(
                    &ActorTable::new(&input.actors),
                    &input.config,
                    &input.spatial,
                    &mut input.scratch,
                    comptime!(input.parameters.scan),
                ),
                SearchStage::Prepare
                | SearchStage::PrepareCandidates
                | SearchStage::PrepareSelected => {
                    let mut group = CUBE_POS as u32;
                    let groups = match comptime!(input.program.stage) {
                        SearchStage::PrepareSelected => {
                            let clear = ceildiv(input.config.actors, 32)
                                + ceildiv(input.config.actors, orders::COMPACTION_BLOCK);
                            ceildiv(
                                max(clear, input.control[orders::PREPARATION_ACTORS_WORD].load()),
                                CUBE_DIM,
                            )
                        }
                        _ => CUBE_COUNT as u32,
                    };
                    while group < groups {
                        input.prepare_buyer(group * CUBE_DIM + UNIT_POS);
                        group += CUBE_COUNT as u32;
                        sync_cube();
                    }
                }
                SearchStage::GrainDealerTiles => grain::dealer_tiles(
                    &EconomyView::new(&mut input.state, &input.actors, &input.config),
                    &input.spatial,
                    &mut input.searches,
                    &input.control,
                    &input.parameters,
                ),
                SearchStage::GrainFirms => grain::collect_firms(
                    &EconomyView::new(&mut input.state, &input.actors, &input.config),
                    &input.spatial,
                    &mut input.searches,
                    &input.control,
                ),
                SearchStage::Search | SearchStage::SearchGrain | SearchStage::SearchIndexed => {
                    input.match_suppliers()
                }
                SearchStage::SearchFuel
                | SearchStage::SearchHousehold
                | SearchStage::SearchSpatialFuel => input.match_fuel(),
                SearchStage::RoadCandidates => road::prepare(
                    &ActorTable::new(&input.actors),
                    &input.config,
                    &input.spatial,
                    &input.scratch,
                    &mut input.searches,
                    &input.control,
                ),
                SearchStage::StockLimits => filter::stock_limits(
                    &Ledger::new(&mut input.state, &input.config),
                    &input.spatial,
                    &input.scratch,
                    &input.successes,
                ),
                SearchStage::FilterBuyers => filter::buyers(
                    &EconomyView::new(&mut input.state, &input.actors, &input.config),
                    &input.proposals,
                    &input.searches,
                    &input.successes,
                    &input.control,
                    &input.parameters,
                ),
                SearchStage::FilterEmpty => filter::orders(
                    &EconomyView::new(&mut input.state, &input.actors, &input.config),
                    &input.proposals,
                    &input.successes,
                    &input.control,
                    &input.parameters,
                ),
                SearchStage::FinishFiltered => filter::finish(
                    &input.config,
                    &input.spatial,
                    &input.successes,
                    &input.control,
                    &input.parameters,
                    comptime!(input.program.collection),
                ),
                SearchStage::Compact => {
                    let mut economy =
                        EconomyView::new(&mut input.state, &input.actors, &input.config);
                    if input.control[crate::simulation::flags::FAULT_WORD].load() == 0 {
                        compact(
                            &input.proposals,
                            &input.searches,
                            &input.successes,
                            &mut input.scratch,
                            &input.control,
                            &input.parameters,
                            ABSOLUTE_POS as u32,
                        );
                        if comptime!(matches!(
                            input.parameters.market,
                            MarketFlag::Consumer(_) | MarketFlag::Household
                        )) {
                            let job = ABSOLUTE_POS as u32;
                            if job < input.control[SEARCHES_WORD].load() {
                                let actor = input.searches[job as usize];
                                let peer = input.proposals[actor as usize * ProposalLayout::WORDS
                                    + ProposalLayout::ACTOR];
                                if peer != NO_ACTOR {
                                    let mut status = clearing::buyer_settlement(
                                        &mut economy,
                                        &mut input.proposals,
                                        &input.control,
                                        &input.parameters,
                                        actor,
                                        peer,
                                        comptime!(scheduler::Posting::CheckActor),
                                    );
                                    if comptime!(
                                        input.parameters.empty_orders
                                            == crate::simulation::parameters::EmptyOrderMode::FeedbackOnly
                                    ) {
                                        let position = (actor + input.parameters.count
                                            - input.parameters.settlement_rotation)
                                            % input.parameters.count;
                                        let kept = input.successes[(position / 32) as usize].load()
                                            & (1u32 << (position % 32));
                                        if kept == 0 {
                                            status.replace(clearing::empty_seller_settlement(
                                                &mut economy,
                                                &input.proposals,
                                                &input.control,
                                                &input.parameters,
                                                actor,
                                                peer,
                                                &status,
                                            ));
                                        }
                                    }
                                    status.store(
                                        &mut input.scratch,
                                        4 * input.parameters.count + actor,
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[cube]
pub(in crate::simulation::economy) fn record_success(
    successes: &[Atomic<u32>],
    parameters: &EconomyParameters,
    actor: u32,
) {
    let position = (actor + parameters.count - parameters.settlement_rotation) % parameters.count;
    let masks = ceildiv(parameters.count, 32);
    successes[(position / 32) as usize].fetch_or(1u32 << (position % 32));
    successes[(masks + position / orders::COMPACTION_BLOCK) as usize].fetch_add(1);
}

#[cube]
pub(in crate::simulation::economy) fn record_wide_order(
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
    quantity: Ticks,
) {
    if comptime!(
        parameters.empty_orders == crate::simulation::parameters::EmptyOrderMode::FeedbackOnly
    ) && (quantity.extract(HIGH_WORD) > 0 || equal(quantity, zero()))
    {
        control[orders::WIDE_ORDERS_WORD].fetch_or(select(
            quantity.extract(HIGH_WORD) > 0,
            1u32,
            2u32,
        ));
    }
}

#[cube]
pub(in crate::simulation::economy) fn compact(
    proposals: &[u32],
    searches: &[u32],
    successes: &[Atomic<u32>],
    scratch: &mut [u32],
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
    job: u32,
) {
    let count = parameters.count;
    let masks = ceildiv(count, 32);
    let blocks = ceildiv(count, orders::COMPACTION_BLOCK);
    if job == 0 {
        control[CONSTRUCTED_EVENTS_WORD].store(successes[(masks + blocks - 1) as usize].load());
    }
    if job < control[SEARCHES_WORD].load() {
        let actor = searches[job as usize];
        let peer = proposals[actor as usize * ProposalLayout::WORDS + ProposalLayout::ACTOR];
        if peer != NO_ACTOR {
            KernelEventStatus::new(comptime!(EventStatus::Inactive))
                .store(scratch, 4 * count + actor);
            let position = (actor + count - parameters.settlement_rotation) % count;
            let block = position / orders::COMPACTION_BLOCK;
            let word = position / 32;
            let kept = successes[word as usize].load();
            let bit = 1u32 << (position % 32);
            if kept & bit != 0 {
                let mut slot = 0u32;
                if block > 0 {
                    slot = successes[(masks + block - 1) as usize].load();
                }
                for preceding in block * (orders::COMPACTION_BLOCK / 32)..word {
                    slot += successes[preceding as usize].load().count_ones();
                }
                slot += (kept & (bit - 1)).count_ones();
                scratch[slot as usize] = peer;
                scratch[(count + slot) as usize] = actor;
            }
        }
    }
}

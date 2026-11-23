use super::{MarketRound, graphs::Preparation};
use crate::{
    domain::{Good, Market},
    simulation::{
        Simulation, Trace,
        economy::{
            EconomyStage,
            configuration::Feature,
            market::{
                indexing, orders,
                search::{
                    self, MarketSearch, OrderCollection, SearchInputLaunch, SearchStage,
                    StatusReset,
                },
            },
            parameters::EmptyOrderMode,
        },
        parallel::{self, ScanMode},
        storage::{
            flags,
            layout::{DISCOVERY_BLOCK, WORD_BYTES},
        },
        system::{AsBufferArg, Dispatch, System},
    },
};
use anyhow::Result;
use cubecl::{CubeCount, CubeDim, Device};
use std::time::Instant;

struct OrderSearch<'a> {
    round: MarketRound<'a>,
    collection: OrderCollection,
    members: usize,
}

#[derive(Clone, Copy)]
enum GrainSearch {
    Standard,
    DealerTiles,
}

impl MarketRound<'_> {
    pub(super) fn order_collection(&self, simulation: &Simulation) -> OrderCollection {
        let hardware = &simulation.device.client.properties().hardware;
        match (
            self.market,
            self.trace,
            &simulation.device.cube,
            self.parameters.scan,
            hardware.plane_size_min,
            hardware.plane_size_max,
        ) {
            (
                Market::Consumer(
                    Good::Linen | Good::Cloth | Good::Clothes | Good::Furniture | Good::Glass,
                ),
                Trace::Discard,
                Device::Metal(_),
                ScanMode::Plane,
                32,
                32,
            ) => OrderCollection::ConsumerMask,
            _ => OrderCollection::Bitmap,
        }
    }

    pub(super) fn prepare_orders(
        self,
        simulation: &mut Simulation,
        collection: OrderCollection,
    ) -> Result<()> {
        let search = OrderSearch::new(self, simulation, collection);
        let preparation = Preparation::new(self.market, self.index);
        search.refresh_index(simulation, preparation);
        let grain = search.grain_search(simulation);
        search.prepare_suppliers(simulation, grain)?;
        match self.market {
            Market::Labor | Market::Payroll | Market::Distribution => {
                self.prepare_proposals(simulation, preparation)
            }
            _ => search.collect(simulation, grain)?,
        }
        self.launch_stage(simulation, EconomyStage::MarketBoundary, 1);
        Ok(())
    }

    fn prepare_proposals(self, simulation: &Simulation, preparation: Preparation) {
        let stage = match preparation {
            Preparation::RetainedLabor => EconomyStage::ProposeRetainedCompact,
            _ => EconomyStage::ProposeCompact,
        };
        self.launch_stage(simulation, stage, self.count);
        parallel::scan(
            &simulation.device,
            &simulation.work.scratch,
            7 * self.count,
            self.count.div_ceil(simulation.device.workgroup as usize),
        );
        self.launch_stage(simulation, EconomyStage::GroupCompactFill, self.count);
    }

    pub(super) fn compact_sparse_input(self, simulation: &Simulation) {
        let search = OrderSearch::new(self, simulation, OrderCollection::Bitmap);
        search.scan_successes(simulation);
        search.dispatch(simulation, SearchStage::Compact);
    }
}

impl<'a> OrderSearch<'a> {
    fn new(round: MarketRound<'a>, simulation: &Simulation, collection: OrderCollection) -> Self {
        let members = match (round.market, simulation.config.home_mode) {
            (Market::Household | Market::Input(Good::Bread) | Market::Consumer(Good::Bread), _)
            | (Market::Input(Good::Grain) | Market::Consumer(Good::Grain), Feature::Enabled) => {
                round.count
            }
            _ => simulation.metadata.actors.firms(),
        };
        Self {
            round,
            collection,
            members,
        }
    }

    fn refresh_index(&self, simulation: &Simulation, preparation: Preparation) {
        if let (
            Market::Labor
            | Market::Household
            | Market::Input(Good::Bread)
            | Market::Consumer(Good::Bread),
            Preparation::First | Preparation::Repeat,
        )
        | (Market::Input(_) | Market::Consumer(_), Preparation::First) =
            (self.round.market, preparation)
        {
            let config = simulation.scenario.config();
            let side = config.map_size.div_ceil(config.search_radius);
            self.round
                .launch_stage(simulation, EconomyStage::IndexCount, self.members);
            parallel::scan(
                &simulation.device,
                &simulation.work.scratch,
                self.round.count,
                self.members.div_ceil(simulation.device.workgroup as usize),
            );
            indexing::MarketIndex::launch_elements(
                &simulation.device,
                self.members.max((side * side + 1) as usize),
                indexing::IndexInputLaunch {
                    config: simulation.config.launch(),
                    spatial: simulation.spatial.buffer_arg(),
                    scratch: simulation.work.scratch.buffer_arg(),
                    control: simulation.control.buffer_arg(),
                    parameters: self.round.parameters.launch(),
                },
            );
        }
    }

    fn grain_search(&self, simulation: &Simulation) -> GrainSearch {
        match (
            self.round.market,
            simulation.config.home_mode,
            simulation.config.spatial_side,
            self.round.count,
            self.round.parameters.scan,
            simulation
                .device
                .client
                .properties()
                .hardware
                .plane_size_min,
        ) {
            (Market::Input(Good::Grain), Feature::Enabled, 1, 4096.., ScanMode::Plane, 32) => {
                GrainSearch::DealerTiles
            }
            _ => GrainSearch::Standard,
        }
    }

    fn prepare_suppliers(&self, simulation: &mut Simulation, grain: GrainSearch) -> Result<()> {
        if let Market::Input(Good::Bread) | Market::Consumer(_) | Market::Household =
            self.round.market
        {
            self.launch(simulation, SearchStage::Index);
        }
        if let GrainSearch::DealerTiles = grain {
            self.launch(simulation, SearchStage::GrainFirms);
        }
        if let Market::Consumer(_) = self.round.market {
            self.launch(simulation, SearchStage::EmptySort);
            self.launch(simulation, SearchStage::EmptyBounds);
            self.report_supply(simulation)?;
        }
        Ok(())
    }

    fn collect(&self, simulation: &mut Simulation, grain: GrainSearch) -> Result<()> {
        use SearchStage::*;
        let preparation = match (self.round.market, self.round.index) {
            (Market::Input(_) | Market::Consumer(_), 0) => PrepareCandidates,
            (Market::Input(_) | Market::Consumer(_), _) => PrepareSelected,
            _ => Prepare,
        };
        self.launch(simulation, preparation);
        self.round
            .launch_stage(simulation, EconomyStage::MarketBoundary, 1);
        self.report_input_quotes(simulation)?;
        if let GrainSearch::DealerTiles = grain {
            self.launch(simulation, GrainDealerTiles);
        }
        if let (OrderCollection::ConsumerMask, 262_144.., 1) = (
            self.collection,
            self.round.count,
            simulation.config.spatial_side,
        ) {
            self.launch(simulation, RoadCandidates);
        }
        let stage = match (self.round.market, grain) {
            (Market::Consumer(_) | Market::Household, _) => SearchSpatialFuel,
            (Market::Input(Good::Bread), _) => SearchIndexed,
            (Market::Input(Good::Grain), GrainSearch::DealerTiles) => SearchGrain,
            _ => Search,
        };
        self.launch(simulation, stage);
        self.filter_orders(simulation);
        if self.collection != OrderCollection::InputTile {
            self.scan_successes(simulation);
            self.launch(simulation, Compact);
        }
        if let EmptyOrderMode::FeedbackOnly = self.round.parameters.empty_orders {
            self.launch(simulation, FinishFiltered);
        }
        self.report_orders(simulation)
    }

    fn filter_orders(&self, simulation: &Simulation) {
        match (self.round.parameters.empty_orders, self.collection) {
            (EmptyOrderMode::FeedbackOnly, OrderCollection::ConsumerMask) => {
                self.launch(simulation, SearchStage::StockLimits);
                self.launch(simulation, SearchStage::FilterBuyers);
            }
            (EmptyOrderMode::FeedbackOnly, _) => self.launch(simulation, SearchStage::FilterEmpty),
            (EmptyOrderMode::Settle, _) => (),
        }
    }

    fn scan_successes(&self, simulation: &Simulation) {
        parallel::scan(
            &simulation.device,
            &simulation.searches.successes,
            orders::search_counts_offset(self.round.count),
            orders::search_blocks(self.round.count),
        );
    }

    fn launch(&self, simulation: &Simulation, stage: SearchStage) {
        let started = Instant::now();
        let window = self.round.profile.start(&simulation.device);
        self.dispatch(simulation, stage);
        self.round
            .profile
            .end(&simulation.device, window, "SEARCH", stage);
        self.round
            .report_stage(simulation, "SEARCH", stage, started);
    }

    fn dispatch(&self, simulation: &Simulation, stage: SearchStage) {
        use SearchStage::*;
        let workgroup = match (stage, &simulation.device.cube) {
            (PrepareSelected, Device::Metal(_)) => 256,
            (Search | SearchGrain, _) => 32,
            (
                SearchIndexed | EmptySort | GrainDealerTiles | FilterBuyers | Prepare
                | PrepareCandidates,
                _,
            ) => 256,
            _ => simulation.device.workgroup,
        };
        let status_reset = match (self.round.market, self.round.trace) {
            (Market::Input(_) | Market::Consumer(Good::Coal | Good::Wood), _)
            | (_, Trace::Discard) => StatusReset::Active,
            (_, Trace::Record) => StatusReset::Full,
        };
        MarketSearch::launch(
            &simulation.device,
            self.dispatch_count(simulation, stage, workgroup),
            CubeDim::new_1d(workgroup),
            SearchInputLaunch {
                state: simulation.state.buffer_arg(),
                actors: simulation.actors.buffer_arg(),
                config: simulation.config.launch(),
                spatial: simulation.spatial.buffer_arg(),
                proposals: simulation.work.proposals.buffer_arg(),
                searches: simulation.searches.requests.buffer_arg(),
                successes: simulation.searches.successes.buffer_arg(),
                scratch: simulation.work.scratch.buffer_arg(),
                control: simulation.control.buffer_arg(),
                parameters: self.round.parameters.launch(),
                program: stage
                    .program(&simulation.config)
                    .collecting(self.collection),
                status_reset,
            },
        );
    }

    fn dispatch_count(
        &self,
        simulation: &Simulation,
        stage: SearchStage,
        workgroup: u32,
    ) -> Dispatch {
        use SearchStage::*;
        let device = &simulation.device;
        let count = self.round.count;
        let groups = count.div_ceil(workgroup as usize);
        let dispatches = &simulation.searches.dispatches;
        match (stage, &device.cube) {
            (SearchSpatialFuel | GrainFirms, _) => device.count(count).into(),
            (GrainDealerTiles, _) => CubeCount::Static(128, search::grain::DEALERS, 1).into(),
            (RoadCandidates, _) => CubeCount::Static(search::ROAD_TILE_SIDE.pow(2), 1, 1).into(),
            (EmptySort, _) => CubeCount::Static(1, 1, 1).into(),
            (EmptyBounds, _) => device
                .count(search::empty_bounds_invocations(
                    self.members,
                    self.round.parameters.scan,
                ))
                .into(),
            (FilterEmpty, _) => device.count(count.div_ceil(32)).into(),
            (FinishFiltered | StockLimits, _) => device.count(self.members).into(),
            (Index, _) => device
                .count(self.members.div_ceil(DISCOVERY_BLOCK as usize) * device.workgroup as usize)
                .into(),
            (Prepare | FilterBuyers | PrepareCandidates, _) => {
                cubecl::calculate_cube_count_elemwise(
                    &device.client,
                    count.max(1),
                    CubeDim::new_1d(workgroup),
                )
                .into()
            }
            (PrepareSelected, Device::Metal(_)) => {
                CubeCount::Static(groups.clamp(1, 512) as u32, 1, 1).into()
            }
            (PrepareSelected, _) => {
                orders::dispatch(dispatches, orders::PREPARATION_DISPATCH, groups)
            }
            (Search | SearchGrain | SearchIndexed, _) => {
                orders::job_dispatch(dispatches, orders::SEARCH_DISPATCH, count)
            }
            (SearchFuel | SearchHousehold | Compact, _) => {
                orders::dispatch(dispatches, orders::SEARCH_COMPACT_DISPATCH, groups)
            }
        }
    }

    fn report_supply(&self, simulation: &mut Simulation) -> Result<()> {
        if simulation.market_profiling("BAD_ECON_PROFILE_SUPPLY") {
            let side = simulation.config.spatial_side;
            let suppliers =
                simulation
                    .device
                    .read_range(&simulation.spatial, (side * side) as usize, 1)?[0];
            let records = simulation.device.read_range(
                &simulation.work.scratch,
                0,
                suppliers as usize * 4,
            )?;
            simulation.readback_bytes += ((records.len() + 1) * WORD_BYTES) as u64;
            let stocked = records
                .chunks_exact(4)
                .filter(|record| record[3] & 1 != 0)
                .count();
            let empty = records
                .chunks_exact(4)
                .filter(|record| record[3] & 3 == 2)
                .count();
            eprintln!(
                "SUPPLY {:?} {} suppliers={suppliers} stocked={stocked} empty={empty}",
                self.round.market, self.round.index,
            );
        }
        Ok(())
    }

    fn report_input_quotes(&self, simulation: &mut Simulation) -> Result<()> {
        if let (Market::Input(Good::Bread), true) = (
            self.round.market,
            simulation.market_profiling("BAD_ECON_PROFILE_SUPPLY"),
        ) {
            let count = self.round.count;
            let words = simulation.device.read_range(
                &simulation.control,
                0,
                search::INPUT_MINIMUM_QUOTE_WORD + 1,
            )?;
            let requests =
                simulation
                    .device
                    .read_range(&simulation.searches.requests, 0, 5 * count)?;
            simulation.readback_bytes += ((words.len() + requests.len()) * WORD_BYTES) as u64;
            let minimum_quote = u64::from(words[search::INPUT_MINIMUM_QUOTE_WORD]);
            let searches = words[orders::SEARCHES_WORD];
            let unaffordable = requests[..searches as usize]
                .iter()
                .filter(|&&buyer| {
                    let buyer = buyer as usize;
                    let cash = u64::from(requests[3 * count + buyer])
                        | (u64::from(requests[4 * count + buyer]) << 32);
                    cash < minimum_quote
                })
                .count();
            eprintln!(
                "INPUT_QUOTE {:?} {} minimum={minimum_quote} searches={searches} unaffordable={unaffordable}",
                self.round.market, self.round.index,
            );
        }
        Ok(())
    }

    fn report_orders(&self, simulation: &mut Simulation) -> Result<()> {
        if simulation.market_profiling("BAD_ECON_PROFILE_SUPPLY") {
            let words = simulation.device.read_range(
                &simulation.control,
                0,
                orders::WIDE_ORDERS_WORD + 1,
            )?;
            simulation.readback_bytes += (words.len() * WORD_BYTES) as u64;
            eprintln!(
                "ORDER_STATS {:?} {} mode={:?} fallback={} searches={} constructed={} completed={} frontier={}",
                self.round.market,
                self.round.index,
                self.round.parameters.empty_orders,
                words[orders::WIDE_ORDERS_WORD],
                words[orders::SEARCHES_WORD],
                words[flags::CONSTRUCTED_EVENTS_WORD],
                words[flags::COMPLETED_WORD],
                words[flags::PROGRESS_WORD],
            );
        }
        Ok(())
    }
}

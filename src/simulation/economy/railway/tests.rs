use super::*;
use crate::domain::{Experiment, Scenario};
use crate::simulation::Simulation;
use anyhow::Result;

#[derive(CubeType, CubeLaunch)]
struct QuoteCheckInput {
    actors: Box<[u32]>,
    config: EconomyConfig,
    mismatches: Box<[Atomic<u32>]>,
}

struct QuoteCheck;
struct CompareRouteDays;
struct CompareQuotes;

impl System for QuoteCheck {
    type Input = QuoteCheckInput;
    type Passes = gpu_passes![CompareRouteDays, CompareQuotes];
}

#[cube]
impl GpuPass<QuoteCheckInput> for CompareRouteDays {
    type Output = QuoteCheckInput;

    fn execute(input: QuoteCheckInput) -> QuoteCheckInput {
        let config = &input.config;
        let index = ABSOLUTE_POS as u32;
        let nodes = config.railway.nodes;
        if index < config.railway.company_count * nodes * nodes {
            let origin = index / nodes % nodes;
            let destination = index % nodes;
            let row = index - destination;
            let base = index as usize * ROUTE_WORDS;
            let mut rail_distance = select(origin == destination, 0, UNREACHABLE);
            if config.railway.routes[base + 2] < config.railway.company_count {
                rail_distance = 0;
                let mut cursor = destination;
                for _step in 0..nodes {
                    let edge = config.railway.routes[(row + cursor) as usize * ROUTE_WORDS + 1];
                    if cursor != origin && edge < config.railway.segment_count {
                        let segment = edge as usize * SEGMENT_WORDS;
                        rail_distance += config.railway.segments[segment + 3];
                        cursor = select(
                            config.railway.segments[segment] == cursor,
                            config.railway.segments[segment + 1],
                            config.railway.segments[segment],
                        );
                    }
                }
            }
            let days = max(
                ceildiv(
                    rail_distance,
                    config.shipment_speed * config.railway.speed_multiplier,
                ),
                1,
            );
            if days != config.railway.routes[base] {
                input.mismatches[0usize].fetch_add(1);
            }
        }
        input
    }
}

#[cube]
fn uncached_candidate(
    actors: &[u32],
    config: &EconomyConfig,
    buyer: u32,
    seller: u32,
    company: u32,
) -> FreightRoute {
    let mut result = animal_route(actors, config, buyer, seller);
    if comptime!(config.railway.enabled == Feature::Enabled)
        && config.railway.nodes > 0
        && company < config.railway.company_count
    {
        let origin = config.railway.access[seller as usize * 2];
        let destination = config.railway.access[buyer as usize * 2];
        let base = (((company * config.railway.nodes + origin) * config.railway.nodes + destination)
            as usize)
            * ROUTE_WORDS;
        let access = config.railway.access[seller as usize * 2 + 1]
            + config.railway.access[buyer as usize * 2 + 1];
        let days = ceildiv(access, config.shipment_speed) + config.railway.routes[base];
        let carrier = config.railway.routes[base + 2];
        let faster = carrier < config.railway.company_count && days < result.days;
        result.days = select(faster, days, result.days);
        result.fee = select(faster, config.railway.routes[base + 3], 0);
        result.origin = origin;
        result.destination = destination;
        result.carrier = select(faster, carrier, u32::MAX);
        result.foreign_distance = select(faster, config.railway.routes[base + 5], u32::MAX);
    }
    result
}

#[cube]
impl GpuPass<QuoteCheckInput> for CompareQuotes {
    type Output = ();

    fn execute(input: QuoteCheckInput) {
        let index = ABSOLUTE_POS as u32;
        let count = input.config.actors;
        if index < count * count {
            let buyer = index / count;
            let seller = index % count;
            if distance(&input.actors, buyer, seller) <= input.config.search_radius {
                let mut expected = animal_route(&input.actors, &input.config, buyer, seller);
                for company in 0..input.config.railway.company_count {
                    let path =
                        uncached_candidate(&input.actors, &input.config, buyer, seller, company);
                    let actual = candidate(&input.actors, &input.config, buyer, seller, company);
                    let equal = actual.days == path.days
                        && actual.fee == path.fee
                        && actual.origin == path.origin
                        && actual.destination == path.destination
                        && actual.carrier == path.carrier
                        && actual.foreign_distance == path.foreign_distance;
                    if !equal {
                        input.mismatches[0usize].fetch_add(1);
                    }
                    if path.preferred_to(&expected) {
                        expected.replace(&path);
                    }
                }
                let actual = route(&input.actors, &input.config, buyer, seller);
                let node = station(&input.actors, &input.config, buyer);
                let access = access_distance(&input.actors, &input.config, buyer, node);
                let equal = actual.days == expected.days
                    && actual.fee == expected.fee
                    && actual.carrier == expected.carrier
                    && actual.foreign_distance == expected.foreign_distance
                    && input.config.railway.access[buyer as usize * 2] == node
                    && input.config.railway.access[buyer as usize * 2 + 1] == access;
                if !equal {
                    input.mismatches[0usize].fetch_add(1);
                }
            }
        }
    }
}

#[test]
fn cached_quotes_match_exhaustive_carriers_on_gpu() -> Result<()> {
    let config = serde_json::from_str(include_str!("../../../../scenarios/railway.json"))?;
    let railway = Scenario::new(config)?;
    let mut slow = railway.config().clone();
    slow.shipment_speed = 16;
    slow.railway.speed_multiplier = 2;
    let scenarios = [
        railway,
        Scenario::new(slow)?,
        Scenario::industrial(193, 42, Experiment::Baseline)?,
        Scenario::industrial(193, 42, Experiment::Transport)?,
    ];
    for scenario in scenarios {
        let device = crate::simulation::tests::device(64)?;
        let mut simulation = Simulation::new(scenario, device.clone())?;
        for _ in 0..3 {
            simulation.advance()?;
            for stage in [RailStage::Access, RailStage::Routes, RailStage::Quotes] {
                simulation.railways(stage);
            }
            let mismatches = device.tensor(&[0]);
            let count = simulation.metadata.actors.len();
            let routes = simulation.config.railway.company_count as usize
                * simulation.config.railway.nodes as usize
                * simulation.config.railway.nodes as usize;
            QuoteCheck::launch(
                &device,
                device.count((count * count).max(routes)),
                CubeDim::new_1d(device.workgroup),
                QuoteCheckInputLaunch {
                    actors: simulation.actors.clone().into_buffer_arg(),
                    config: simulation.config.launch(),
                    mismatches: mismatches.clone().into_buffer_arg(),
                },
            );
            assert_eq!(device.read(&mismatches)?, [0]);
        }
    }
    Ok(())
}

#[derive(CubeType, CubeLaunch)]
struct PathCheckInput {
    config: EconomyConfig,
    results: Box<[u32]>,
}

struct PathCheck;
struct ComparePaths;

impl System for PathCheck {
    type Input = PathCheckInput;
    type Passes = gpu_passes![ComparePaths];
}

#[cube]
impl GpuPass<PathCheckInput> for ComparePaths {
    type Output = ();

    fn execute(mut input: PathCheckInput) {
        let config = &input.config;
        let mut actual = RequestLists::new(comptime!(config.railway.request_storage));
        actual.initialize(config);
        let mut expected = RequestLists::new(comptime!(RequestStorage::Shared));
        expected.initialize(config);
        if UNIT_POS == 0 {
            let regions = config.railway.region_count;
            let companies = config.railway.company_count;
            for carrier in 0..companies {
                for origin in 0..config.railway.nodes {
                    for destination in 0..config.railway.nodes {
                        for record in 0..regions {
                            expected.set_region(config, record as usize, 0);
                        }
                        expected.set_region(config, (2 * regions) as usize, 0);
                        for owner in 0..companies {
                            expected.set_access(config, owner as usize, 0);
                        }
                        expected.set_access(config, (2 * companies) as usize, 0);
                        let path = FreightRoute {
                            days: 0,
                            fee: 0,
                            origin,
                            destination,
                            carrier,
                            foreign_distance: 0,
                        };
                        traverse_with_requests(
                            config,
                            &path,
                            comptime!(Traversal::Request),
                            &mut expected,
                        );
                        prepare_request_with_lists(config, &path, &mut actual);
                        prepare_request_with_lists(config, &path, &mut actual);
                        for transition in 0..2u32 {
                            if transition == 1 {
                                let alternate = FreightRoute {
                                    days: 0,
                                    fee: 0,
                                    origin,
                                    destination: (destination + 1) % config.railway.nodes,
                                    carrier,
                                    foreign_distance: 0,
                                };
                                prepare_request_with_lists(config, &alternate, &mut actual);
                            }
                            prepare_request(config, &path);
                            for index in 0..regions {
                                input.results[0usize] += u32::cast_from(
                                    config.railway.requests[index as usize].load()
                                        != expected.region(config, index as usize),
                                );
                            }
                            for index in 0..companies {
                                input.results[0usize] += u32::cast_from(
                                    config.railway.access_requests[index as usize].load()
                                        != expected.access(config, index as usize),
                                );
                            }
                        }
                        actual.invalidate(config);
                        prepare_request_with_lists(config, &path, &mut actual);
                        let region_count = expected.region(config, (2 * regions) as usize);
                        let access_count = expected.access(config, (2 * companies) as usize);
                        let route = ((carrier * config.railway.nodes + origin)
                            * config.railway.nodes
                            + destination) as usize
                            * ROUTE_WORDS;
                        let valid = config.railway.routes[route + 2] < companies;
                        let cached = config.railway.routes[route + 4] > 0;
                        let cacheable =
                            valid && region_count + access_count <= REQUEST_PATH_ENTRIES as u32;
                        input.results[0usize] += u32::cast_from(cached != cacheable);
                        input.results[1usize] += u32::cast_from(cached);
                        input.results[2usize] += u32::cast_from(valid && !cached);
                        input.results[3usize] += access_count;
                        input.results[0usize] +=
                            u32::cast_from(actual.region_count(config) != region_count);
                        input.results[0usize] +=
                            u32::cast_from(actual.access_count(config) != access_count);
                        for index in 0..region_count {
                            let expected = expected.region_entry(config, index);
                            let actual = actual.region_entry(config, index);
                            input.results[0usize] += u32::cast_from(
                                actual.record != expected.record
                                    || actual.visits != expected.visits,
                            );
                        }
                        for index in 0..access_count {
                            let expected = expected.access_entry(config, index);
                            let actual = actual.access_entry(config, index);
                            input.results[0usize] += u32::cast_from(
                                actual.owner != expected.owner || actual.units != expected.units,
                            );
                        }
                        for record in 0..regions {
                            input.results[0usize] += u32::cast_from(
                                !cached
                                    && actual.region(config, record as usize)
                                        != expected.region(config, record as usize),
                            );
                        }
                        for owner in 0..companies {
                            input.results[0usize] += u32::cast_from(
                                !cached
                                    && actual.access(config, owner as usize)
                                        != expected.access(config, owner as usize),
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn cached_request_paths_preserve_order_access_and_long_route_fallback_on_gpu() -> Result<()> {
    use crate::domain::ScenarioConfig;
    use crate::simulation::encode;

    let device = crate::simulation::tests::device(64)?;
    for length_unit in [100, 3] {
        let mut config: ScenarioConfig =
            serde_json::from_str(include_str!("../../../../scenarios/railway.json"))?;
        config.railway.length_unit = length_unit;
        let scenario = Scenario::new(config)?;
        let simulation = Simulation::new(scenario.clone(), device.clone())?;
        assert!(matches!(
            simulation.config.railway.request_storage,
            RequestStorage::Cached { .. }
        ));
        assert!(simulation.config.railway.path_cache == Feature::Enabled);
        for operating in [1, 0, 1] {
            let mut state = scenario.initial()?;
            for actor in state.actors.firm_indices().collect::<Vec<_>>() {
                if state.actors.at(actor).kind == Good::Railway as u32 {
                    state.set(Field::RailOperating, actor, operating);
                }
            }
            device.write(
                &simulation.state,
                bytemuck::cast_slice(&encode(&state.values)),
            );
            simulation.railways(RailStage::Routes);
            let nodes = simulation.config.railway.nodes as usize;
            let route_words =
                simulation.config.railway.company_count as usize * nodes * nodes * ROUTE_WORDS;
            let routes = device.read_range(&simulation.config.railway.routes, 0, route_words)?;
            assert!(routes.chunks_exact(ROUTE_WORDS).all(|route| route[4] == 0));
            simulation.railways(RailStage::Quotes);
            let results = device.tensor(&[0; 4]);
            PathCheck::launch(
                &device,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(device.workgroup),
                PathCheckInputLaunch {
                    config: simulation.config.launch(),
                    results: results.clone().into_buffer_arg(),
                },
            );
            let result = device.read(&results)?;
            assert_eq!(result[0], 0, "length={length_unit}, operating={operating}");
            match operating {
                0 => assert_eq!(&result[1..], &[0, 0, 0]),
                _ => {
                    assert!(result[1] > 0);
                    assert_eq!(result[2] > 0, length_unit == 3);
                    assert!(result[3] > 0);
                }
            }
        }
    }
    Ok(())
}

#[derive(CubeType, CubeLaunch)]
struct RegionalCheckInput {
    state: Box<[u32]>,
    config: EconomyConfig,
    mismatches: Box<[u32]>,
}

struct RegionalCheck;
struct CompareRegionalState;

impl System for RegionalCheck {
    type Input = RegionalCheckInput;
    type Passes = gpu_passes![CompareRegionalState];
}

#[cube]
impl GpuPass<RegionalCheckInput> for CompareRegionalState {
    type Output = ();

    fn execute(mut input: RegionalCheckInput) {
        let config = &input.config;
        let mut requests = RequestLists::new(comptime!(config.railway.request_storage));
        requests.initialize(config);
        requests.refresh(&mut input.state, config);
        for step in 0..5u32 {
            if UNIT_POS == 0 {
                let amount = match step {
                    0 => ticks(u32::MAX - 1, 0),
                    1 => ticks(u32::MAX, 0),
                    2 => ticks(0, 1),
                    3 => ticks(u32::MAX, u32::MAX),
                    _ => ticks(7, 0),
                };
                for record in 0..config.railway.region_count {
                    #[unroll]
                    for _repeat in 0..2 {
                        requests.set_capacity_value(
                            &mut input.state,
                            config,
                            record,
                            amount,
                            comptime!(RegionField::Available),
                        );
                        requests.set_capacity_value(
                            &mut input.state,
                            config,
                            record,
                            amount,
                            comptime!(RegionField::Demand),
                        );
                    }
                    let available = requests.capacity_value(
                        &mut input.state,
                        config,
                        record,
                        comptime!(RegionField::Available),
                    );
                    let demand = requests.capacity_value(
                        &mut input.state,
                        config,
                        record,
                        comptime!(RegionField::Demand),
                    );
                    let stored_available = regional_value(
                        &mut input.state,
                        config,
                        record,
                        comptime!(RailCapacityField::Available),
                    );
                    let stored_demand = regional_value(
                        &mut input.state,
                        config,
                        record,
                        comptime!(RailCapacityField::Demand),
                    );
                    input.mismatches[0usize] += u32::cast_from(
                        !equal(available, amount)
                            || !equal(demand, amount)
                            || !equal(stored_available, amount)
                            || !equal(stored_demand, amount),
                    );
                    let external = ticks(record + step + 1, step % 2);
                    regional_set(
                        &mut input.state,
                        config,
                        record,
                        external,
                        comptime!(RailCapacityField::Available),
                    );
                    regional_set(
                        &mut input.state,
                        config,
                        record,
                        external,
                        comptime!(RailCapacityField::Demand),
                    );
                    regional_set(
                        &mut input.state,
                        config,
                        record,
                        ticks(record + 1, 99),
                        comptime!(RailCapacityField::Level),
                    );
                }
            }
            requests.refresh(&mut input.state, config);
            if UNIT_POS == 0 {
                for record in 0..config.railway.region_count {
                    let external = ticks(record + step + 1, step % 2);
                    let available = requests.capacity_value(
                        &mut input.state,
                        config,
                        record,
                        comptime!(RegionField::Available),
                    );
                    let demand = requests.capacity_value(
                        &mut input.state,
                        config,
                        record,
                        comptime!(RegionField::Demand),
                    );
                    let level = requests.capacity_level(&mut input.state, config, record);
                    input.mismatches[0usize] += u32::cast_from(
                        !equal(available, external)
                            || !equal(demand, external)
                            || level != record + 1,
                    );
                }
            }
        }
    }
}

#[test]
fn shared_regional_state_preserves_wide_values_and_fallback_refresh_on_gpu() -> Result<()> {
    use crate::domain::ScenarioConfig;

    let mut config: ScenarioConfig =
        serde_json::from_str(include_str!("../../../../scenarios/railway.json"))?;
    config.railway.length_unit = 3;
    let scenario = Scenario::new(config)?;
    let device = crate::simulation::tests::device(64)?;
    let simulation = Simulation::new(scenario, device.clone())?;
    assert!(matches!(
        simulation.config.railway.request_storage,
        RequestStorage::Cached { .. }
    ));
    let mismatches = device.tensor(&[0]);
    RegionalCheck::launch(
        &device,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(device.workgroup),
        RegionalCheckInputLaunch {
            state: simulation.state.clone().into_buffer_arg(),
            config: simulation.config.launch(),
            mismatches: mismatches.clone().into_buffer_arg(),
        },
    );
    assert_eq!(device.read(&mismatches)?, [0]);
    Ok(())
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum TradeExecution {
    Ledger,
    Direct,
    DeferredRejection,
}

#[derive(CubeType, CubeLaunch)]
struct TradeCheckInput {
    state: Box<[u32]>,
    actors: Box<[u32]>,
    config: EconomyConfig,
    proposals: Box<[u32]>,
    control: Box<[Atomic<u32>]>,
    status: Box<[u32]>,
    parameters: EconomyParameters,
    buyer: u32,
    seller: u32,
    #[cube(comptime)]
    execution: TradeExecution,
}

struct TradeCheck;
struct CompareTrades;

impl System for TradeCheck {
    type Input = TradeCheckInput;
    type Passes = gpu_passes![CompareTrades];
}

#[cube]
impl GpuPass<TradeCheckInput> for CompareTrades {
    type Output = ();

    fn execute(mut input: TradeCheckInput) {
        let buyer = input.buyer;
        let seller = input.seller;
        let quantity = ticks(
            input.proposals[buyer as usize * ProposalLayout::WORDS + ProposalLayout::QUANTITY_LOW],
            input.proposals[buyer as usize * ProposalLayout::WORDS + ProposalLayout::QUANTITY_HIGH],
        );
        let path = FreightRoute {
            days: 2,
            fee: 17,
            origin: 0,
            destination: 0,
            carrier: 0,
            foreign_distance: 0,
        };
        let funded = !lt(
            value(
                &mut input.state,
                &input.config,
                comptime!(Field::Cash as u32),
                seller,
            ),
            multiply(&input.control, quantity, path.fee),
        );
        match comptime!(input.execution) {
            TradeExecution::DeferredRejection => {
                let cash = value(
                    &mut input.state,
                    &input.config,
                    comptime!(Field::Cash as u32),
                    seller,
                );
                let sales = value(
                    &mut input.state,
                    &input.config,
                    comptime!(Field::Sales as u32),
                    seller,
                );
                reject_remainder(
                    &mut input.state,
                    &input.actors,
                    &input.config,
                    &input.parameters,
                    seller,
                    0,
                );
                set32(
                    &mut input.state,
                    &input.config,
                    comptime!(Field::Cash as u32),
                    seller,
                    0,
                );
                set32(
                    &mut input.state,
                    &input.config,
                    comptime!(Field::Sales as u32),
                    seller,
                    0,
                );
                finalize_rejection(
                    &mut input.state,
                    &input.actors,
                    &input.config,
                    &input.control,
                    &input.parameters,
                    &input.proposals,
                    &mut input.status,
                    buyer,
                    seller,
                    0,
                );
                let status =
                    KernelEventStatus::load(&mut input.status, 4 * input.parameters.count + buyer);
                status.store(&mut input.status, 0);
                set(
                    &mut input.state,
                    &input.config,
                    comptime!(Field::Cash as u32),
                    seller,
                    cash,
                );
                set(
                    &mut input.state,
                    &input.config,
                    comptime!(Field::Sales as u32),
                    seller,
                    sales,
                );
            }
            TradeExecution::Ledger => {
                let animal = animal_route(&input.actors, &input.config, buyer, seller);
                let mut event = Array::<u32>::new(EVENT_WORDS);
                construct_route(
                    &mut input.state,
                    &input.actors,
                    &input.config,
                    &mut input.proposals,
                    &mut event[..],
                    &input.control,
                    &input.parameters,
                    &animal,
                    buyer,
                    comptime!(EventStorage::Local),
                );
                let eligible = scheduler::settle(
                    &mut input.state,
                    &event[..],
                    &input.control,
                    0,
                    0,
                    1,
                    comptime!(Posting::CheckAll),
                );
                if eligible.contains(comptime!(EventStatus::Accepted)) && funded {
                    construct_route(
                        &mut input.state,
                        &input.actors,
                        &input.config,
                        &mut input.proposals,
                        &mut event[..],
                        &input.control,
                        &input.parameters,
                        &path,
                        buyer,
                        comptime!(EventStorage::Local),
                    );
                }
                apply(&mut input.state, &event[..], &input.control, 0).store(&mut input.status, 0);
            }
            TradeExecution::Direct => {
                let mut settlement = GoodsSettlement::new(
                    &mut input.state,
                    &input.actors,
                    &input.config,
                    &input.control,
                    &input.parameters,
                    buyer,
                    seller,
                    quantity,
                );
                if settlement.status.contains(comptime!(EventStatus::Accepted)) && funded {
                    settlement.reroute(
                        &mut input.state,
                        &input.config,
                        &input.control,
                        &input.parameters,
                        buyer,
                        quantity,
                        &path,
                    );
                }
                settlement.apply(
                    &mut input.state,
                    &input.config,
                    &input.parameters,
                    buyer,
                    seller,
                );
                settlement.status.store(&mut input.status, 0);
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum TradeCase {
    Ordinary,
    Wide,
    Unaffordable,
    Unstocked,
    UnstockedSellerOverflow,
    UnstockedSalesOverflow,
    UnstockedTransitOverflow,
    UnstockedPurchaseOverflow,
    UnfundedFreight,
    SellerOverflow,
    TransitOverflow,
    PurchaseOverflow,
    AnimalOverflow,
    RailOverflow,
    RetailCapacity,
    ExhaustedRetailer,
}

#[test]
fn direct_goods_settlement_preserves_ledger_balances_and_faults_on_gpu() -> Result<()> {
    use crate::simulation::{Market, encode};
    let scenario = Scenario::baseline(3, 42, Experiment::Concentrated)?;
    let device = crate::simulation::tests::device(64)?;
    let simulation = Simulation::new(scenario.clone(), device.clone())?;
    for market in [
        Market::Household,
        Market::Consumer(Good::Bread),
        Market::Input(Good::Bread),
    ] {
        let buyer = match market {
            Market::Input(_) => 4,
            _ => 0,
        };
        let seller = 5;
        for case in [
            TradeCase::Ordinary,
            TradeCase::Wide,
            TradeCase::Unaffordable,
            TradeCase::Unstocked,
            TradeCase::UnstockedSellerOverflow,
            TradeCase::UnstockedSalesOverflow,
            TradeCase::UnstockedTransitOverflow,
            TradeCase::UnstockedPurchaseOverflow,
            TradeCase::UnfundedFreight,
            TradeCase::SellerOverflow,
            TradeCase::TransitOverflow,
            TradeCase::PurchaseOverflow,
            TradeCase::AnimalOverflow,
            TradeCase::RailOverflow,
            TradeCase::RetailCapacity,
            TradeCase::ExhaustedRetailer,
        ] {
            let mut initial = scenario.initial()?;
            initial.set(Field::Cash, buyer, 1 << 48);
            initial.set(Field::Cash, seller, 1 << 48);
            initial.set(Field::Bread, seller, 1 << 48);
            initial.set(Field::Price, seller, 10);
            initial.set(Field::FirstInput, buyer, Good::Bread as u64);
            initial.set(Field::SecondInput, buyer, Good::Bread as u64);
            initial.set(Field::EnterpriseModel, buyer, 1);
            let quantity = match case {
                TradeCase::Wide => u64::from(u32::MAX) + 5,
                _ => 3,
            };
            match case {
                TradeCase::Unaffordable => initial.set(Field::Cash, buyer, 0),
                TradeCase::Unstocked => initial.set(Field::Bread, seller, 0),
                TradeCase::UnfundedFreight => initial.set(Field::Cash, seller, 0),
                TradeCase::SellerOverflow | TradeCase::UnstockedSellerOverflow => {
                    initial.set(Field::Cash, seller, u64::MAX)
                }
                TradeCase::UnstockedSalesOverflow => initial.set(Field::Sales, seller, u64::MAX),
                TradeCase::TransitOverflow | TradeCase::UnstockedTransitOverflow => {
                    initial.set(Field::Transit, buyer, u64::MAX)
                }
                TradeCase::PurchaseOverflow | TradeCase::UnstockedPurchaseOverflow => {
                    initial.set(Field::Purchased, buyer, u64::MAX)
                }
                TradeCase::AnimalOverflow | TradeCase::RailOverflow => {
                    let day = match case {
                        TradeCase::AnimalOverflow => 1,
                        _ => 0,
                    };
                    let cell =
                        initial.shipment_coordinate(Good::Bread, day, scenario.ring(), buyer);
                    initial.values[cell.0 as usize] = u64::MAX;
                }
                TradeCase::RetailCapacity | TradeCase::ExhaustedRetailer => {
                    initial.set(Field::EnterpriseModel, seller, 1);
                    initial.set(Field::RetailOpen, seller, 5);
                    initial.set(Field::Sales, seller, 4);
                }
                _ => (),
            }
            if let TradeCase::ExhaustedRetailer = case {
                initial.set(Field::Bread, seller, 200);
                initial.set(Field::Sales, seller, 5);
            }
            match case {
                TradeCase::UnstockedSellerOverflow
                | TradeCase::UnstockedSalesOverflow
                | TradeCase::UnstockedTransitOverflow
                | TradeCase::UnstockedPurchaseOverflow => initial.set(Field::Bread, seller, 0),
                _ => (),
            }
            let count = initial.actors.len();
            let mut proposals = vec![NO_ACTOR; count * ProposalLayout::WORDS];
            proposals[buyer * ProposalLayout::WORDS + ProposalLayout::ACTOR] = seller as u32;
            proposals[buyer * ProposalLayout::WORDS + ProposalLayout::QUANTITY_LOW] =
                quantity as u32;
            proposals[buyer * ProposalLayout::WORDS + ProposalLayout::QUANTITY_HIGH] =
                (quantity >> 32) as u32;
            let proposals = device.tensor(&proposals);
            let parameters = simulation.parameters(Phase::ProvisionPurchase, market, 0, count)?;
            let mut results = Vec::new();
            let executions: &[TradeExecution] = match (initial.get(Field::Bread, seller), case) {
                (0, _) | (_, TradeCase::ExhaustedRetailer) => &[
                    TradeExecution::Ledger,
                    TradeExecution::Direct,
                    TradeExecution::DeferredRejection,
                ],
                _ => &[TradeExecution::Ledger, TradeExecution::Direct],
            };
            for &execution in executions {
                let state = device.tensor(bytemuck::cast_slice(&encode(&initial.values)));
                let control = device.tensor(&[0; CONTROL_WORDS]);
                let status = device.tensor(&vec![0; count * 5 + 1]);
                TradeCheck::launch(
                    &device,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new_1d(1),
                    TradeCheckInputLaunch {
                        state: state.clone().into_buffer_arg(),
                        actors: simulation.actors.clone().into_buffer_arg(),
                        config: simulation.config.launch(),
                        proposals: proposals.clone().into_buffer_arg(),
                        control: control.clone().into_buffer_arg(),
                        status: status.clone().into_buffer_arg(),
                        parameters: parameters.launch(),
                        buyer: buyer as u32,
                        seller: seller as u32,
                        execution,
                    },
                );
                let mut result = device.read(&state)?;
                result.extend(device.read(&control)?);
                result.extend(device.read_range(&status, 0, 1)?);
                results.push(result);
            }
            for result in results.iter().skip(1) {
                assert_eq!(results[0], *result, "{market:?} {case:?}");
            }
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum SelectionCapacity {
    Current,
    Exhausted,
}

#[derive(CubeType, CubeLaunch)]
struct SelectionCheckInput {
    state: Box<[u32]>,
    actors: Box<[u32]>,
    config: EconomyConfig,
    control: Box<[Atomic<u32>]>,
    results: Box<[Atomic<u32>]>,
    #[cube(comptime)]
    capacity: SelectionCapacity,
}

struct SelectionCheck;
struct CompareSelections;

impl System for SelectionCheck {
    type Input = SelectionCheckInput;
    type Passes = gpu_passes![CompareSelections];
}

#[cube]
impl GpuPass<SelectionCheckInput> for CompareSelections {
    type Output = ();

    fn execute(mut input: SelectionCheckInput) {
        if comptime!(matches!(input.capacity, SelectionCapacity::Exhausted)) {
            for region in 0..input.config.railway.region_count {
                regional_set(
                    &mut input.state,
                    &input.config,
                    region,
                    zero(),
                    comptime!(RailCapacityField::Available),
                );
            }
            input.config.railway.requests[(2 * input.config.railway.region_count + 1) as usize]
                .fetch_add(1);
        }
        for seller in 0..input.config.actors {
            let saved = value(
                &mut input.state,
                &input.config,
                comptime!(Field::Cash as u32),
                seller,
            );
            for buyer in 0..input.config.actors {
                let preferred = route(&input.actors, &input.config, buyer, seller);
                if preferred.carrier < input.config.railway.company_count {
                    for sample in 0..12u32 {
                        let quantity = ticks(
                            select(sample == 0, 7, 1 + sample % 3),
                            select(sample >= 9, 1u32, 0u32),
                        );
                        let mut cash = ticks(u32::MAX, u32::MAX);
                        if sample % 3 != 1 {
                            cash = ticks(sample % 3 * 17, 0);
                        }
                        set(
                            &mut input.state,
                            &input.config,
                            comptime!(Field::Cash as u32),
                            seller,
                            cash,
                        );
                        let actual = available_route(
                            &mut input.state,
                            &input.actors,
                            &input.config,
                            &input.control,
                            buyer,
                            seller,
                            quantity,
                            &preferred,
                        );
                        let mut expected =
                            animal_route(&input.actors, &input.config, buyer, seller);
                        for company in 0..input.config.railway.company_count {
                            let path =
                                candidate(&input.actors, &input.config, buyer, seller, company);
                            let affordable =
                                !lt(cash, multiply(&input.control, quantity, path.fee));
                            if path.carrier < input.config.railway.company_count
                                && path.preferred_to(&expected)
                                && affordable
                            {
                                if !lt(
                                    route_capacity(
                                        &mut input.state,
                                        &input.config,
                                        &input.control,
                                        &path,
                                        quantity,
                                        comptime!(DemandSignal::Ignore),
                                    ),
                                    quantity,
                                ) {
                                    expected.replace(&path);
                                }
                            }
                        }
                        let identical = actual.days == expected.days
                            && actual.fee == expected.fee
                            && actual.carrier == expected.carrier
                            && actual.foreign_distance == expected.foreign_distance;
                        input.results[0usize].fetch_add(u32::cast_from(!identical));
                        input.results[1usize].fetch_add(1);
                        if comptime!(matches!(input.capacity, SelectionCapacity::Exhausted))
                            && sample == 0
                        {
                            let animal = animal_route(&input.actors, &input.config, buyer, seller);
                            let access = ceildiv(
                                input.config.railway.access[seller as usize * 2 + 1]
                                    + input.config.railway.access[buyer as usize * 2 + 1],
                                input.config.shipment_speed,
                            );
                            let limit = min(
                                max(animal.days, access + 1) - access - 1,
                                input.config.ring - 1,
                            );
                            let quote = (preferred.origin * input.config.railway.nodes
                                + preferred.destination)
                                * input.config.ring
                                + limit;
                            let cached = selection_cache(&input.config, quote);
                            input.results[2usize].fetch_add(u32::cast_from(
                                input.config.railway.requests[cached + 1].load() != 1
                                    || input.config.railway.requests[cached + 2].load() != 0,
                            ));
                        }
                    }
                }
            }
            set(
                &mut input.state,
                &input.config,
                comptime!(Field::Cash as u32),
                seller,
                saved,
            );
        }
    }
}

#[test]
fn exhausted_route_cache_preserves_partially_funded_carrier_selection_on_gpu() -> Result<()> {
    let config = serde_json::from_str(include_str!("../../../../scenarios/railway.json"))?;
    let scenario = Scenario::new(config)?;
    let device = crate::simulation::tests::device(64)?;
    for capacity in [SelectionCapacity::Current, SelectionCapacity::Exhausted] {
        let simulation = Simulation::new(scenario.clone(), device.clone())?;
        let mut initial = scenario.initial()?;
        for region in 0..scenario.rail_geography().regions.len() {
            for field in [RailCapacityField::Level, RailCapacityField::Available] {
                let cell = scenario
                    .rail_geography()
                    .coordinate(&scenario, &initial, field, region);
                initial.values[cell] = match field {
                    RailCapacityField::Level => 1,
                    _ => 100,
                };
            }
        }
        for actor in 0..initial.actors.len() {
            if initial.actors.at(actor).kind == Good::Railway as u32 {
                initial.set(Field::RailBuilt, actor, 1);
                initial.set(Field::RailOperating, actor, 1);
            }
        }
        device.write(
            &simulation.state,
            bytemuck::cast_slice(&crate::simulation::encode(&initial.values)),
        );
        for stage in [RailStage::Access, RailStage::Routes, RailStage::Quotes] {
            simulation.railways(stage);
        }
        let results = device.tensor(&[0, 0, 0]);
        SelectionCheck::launch(
            &device,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            SelectionCheckInputLaunch {
                state: simulation.state.clone().into_buffer_arg(),
                actors: simulation.actors.clone().into_buffer_arg(),
                config: simulation.config.launch(),
                control: simulation.control.clone().into_buffer_arg(),
                results: results.clone().into_buffer_arg(),
                capacity,
            },
        );
        let results = device.read(&results)?;
        assert_eq!(results[0], 0);
        assert!(results[1] > 0);
        assert_eq!(results[2], 0);
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
enum DispatchCase {
    Ordinary,
    FundedBatch,
    PartialFreight,
    Exhausted,
    RetailCapacity,
    SellerOverflow,
    SalesOverflow,
    BuyerOverflow,
    RerouteOverflow,
    ZeroQuantity,
    WideQuantity,
    RailwayBuyer,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum CapacityCheckMode {
    Zero,
    Mixed,
    Restored,
    Stable,
    Malformed,
}

#[derive(CubeType, CubeLaunch)]
struct CapacityCheckInput {
    state: Box<[u32]>,
    actors: Box<[u32]>,
    config: EconomyConfig,
    control: Box<[Atomic<u32>]>,
    results: Box<[u32]>,
    #[cube(comptime)]
    mode: CapacityCheckMode,
}

struct CapacityCheck;
struct CheckCapacityEpochs;

impl System for CapacityCheck {
    type Input = CapacityCheckInput;
    type Passes = gpu_passes![CheckCapacityEpochs];
}

#[cube]
fn check_capacity_demands(state: &mut [u32], config: &EconomyConfig, quantity: u32) -> u32 {
    let regions = config.railway.region_count;
    let mut mismatches = 0u32;
    for index in 0..config.railway.requests[(2 * regions) as usize].load() {
        let record = config.railway.requests[(regions + index) as usize].load();
        let visits = config.railway.requests[record as usize].load();
        let stock = regional_value(
            state,
            config,
            record,
            comptime!(RailCapacityField::Available),
        );
        let demand = regional_value(state, config, record, comptime!(RailCapacityField::Demand));
        let expected = plus(
            minus(ticks(config.railway.units_per_level, 0), stock),
            times32(quantity, visits),
        )
        .value;
        mismatches += u32::cast_from(!equal(demand, expected));
    }
    mismatches
}

#[cube]
impl GpuPass<CapacityCheckInput> for CheckCapacityEpochs {
    type Output = ();

    fn execute(mut input: CapacityCheckInput) {
        let path = route(&input.actors, &input.config, 0, 1);
        input.results[0usize] = path.carrier;
        if path.carrier < input.config.railway.company_count {
            let epoch = (2 * input.config.railway.region_count + 1) as usize;
            let cache = epoch
                + 1
                + ((path.carrier * input.config.railway.nodes + path.origin)
                    * input.config.railway.nodes
                    + path.destination) as usize
                    * CAPACITY_WORDS;
            match comptime!(input.mode) {
                CapacityCheckMode::Malformed => {
                    prepare_request(&input.config, &path);
                    let regions = input.config.railway.region_count;
                    let count = input.config.railway.requests[(2 * regions) as usize].load();
                    for index in 0..count {
                        let record =
                            input.config.railway.requests[(regions + index) as usize].load();
                        let visits = input.config.railway.requests[record as usize].load();
                        regional_set(
                            &mut input.state,
                            &input.config,
                            record,
                            ticks(input.config.railway.units_per_level + visits, 0),
                            comptime!(RailCapacityField::Available),
                        );
                        regional_set(
                            &mut input.state,
                            &input.config,
                            record,
                            ticks(input.config.railway.units_per_level + 7 * visits, 0),
                            comptime!(RailCapacityField::Demand),
                        );
                    }
                    input.config.railway.requests[cache].store(0);
                    route_capacity(
                        &mut input.state,
                        &input.config,
                        &input.control,
                        &path,
                        ticks(1, 0),
                        comptime!(DemandSignal::Record),
                    );
                    input.results[1usize] = input.config.railway.requests
                        [cache + comptime!(CapacityWord::StableDemand as usize)]
                    .load();
                    input.results[2usize] =
                        input.control[crate::simulation::flags::FAULT_WORD].load();
                    input.control[crate::simulation::flags::FAULT_WORD].store(0);
                    input.config.railway.requests[epoch].fetch_add(1);
                    record_route_demand(
                        &mut input.state,
                        &input.config,
                        &input.control,
                        &path,
                        ticks(1, 0),
                    );
                    input.results[3usize] =
                        input.control[crate::simulation::flags::FAULT_WORD].load();
                }
                CapacityCheckMode::Stable => {
                    prepare_request(&input.config, &path);
                    let regions = input.config.railway.region_count;
                    let count = input.config.railway.requests[(2 * regions) as usize].load();
                    for index in 0..count {
                        let record =
                            input.config.railway.requests[(regions + index) as usize].load();
                        let visits = input.config.railway.requests[record as usize].load();
                        regional_set(
                            &mut input.state,
                            &input.config,
                            record,
                            ticks(10 * visits, 0),
                            comptime!(RailCapacityField::Available),
                        );
                        regional_set(
                            &mut input.state,
                            &input.config,
                            record,
                            ticks(input.config.railway.units_per_level + 7 * visits, 0),
                            comptime!(RailCapacityField::Demand),
                        );
                    }
                    input.config.railway.requests[cache].store(0);
                    let available = route_capacity(
                        &mut input.state,
                        &input.config,
                        &input.control,
                        &path,
                        ticks(1, 0),
                        comptime!(DemandSignal::Record),
                    );
                    input.results[1usize] = u32::cast_from(!lt(available, ticks(1, 0)));
                    input.results[2usize] = input.config.railway.requests
                        [cache + comptime!(CapacityWord::StableDemand as usize)]
                    .load();
                    input.config.railway.requests[epoch].fetch_add(1);
                    for index in 0..count {
                        let record =
                            input.config.railway.requests[(regions + index) as usize].load();
                        regional_set(
                            &mut input.state,
                            &input.config,
                            record,
                            zero(),
                            comptime!(RailCapacityField::Available),
                        );
                    }
                    input.results[3usize] = u32::cast_from(capacity_stable_demand_recorded(
                        &input.config,
                        cache,
                        ticks(7, 0),
                    ));
                    record_route_demand(
                        &mut input.state,
                        &input.config,
                        &input.control,
                        &path,
                        ticks(7, 0),
                    );
                    input.results[4usize] = input.config.railway.requests[cache].load();
                    input.results[5usize] =
                        check_capacity_demands(&mut input.state, &input.config, 7);
                    let available = route_capacity(
                        &mut input.state,
                        &input.config,
                        &input.control,
                        &path,
                        ticks(7, 0),
                        comptime!(DemandSignal::Ignore),
                    );
                    input.results[6usize] = u32::cast_from(!lt(available, ticks(7, 0)));
                    input.results[7usize] = input.config.railway.requests
                        [cache + comptime!(CapacityWord::StableDemand as usize)]
                    .load();
                    record_route_demand(
                        &mut input.state,
                        &input.config,
                        &input.control,
                        &path,
                        ticks(8, 0),
                    );
                    input.results[8usize] =
                        check_capacity_demands(&mut input.state, &input.config, 8);
                    input.results[9usize] = input.config.railway.requests
                        [cache + comptime!(CapacityWord::StableDemand as usize)]
                    .load();
                    input.results[10usize] = input.config.railway.requests[cache].load();
                    input.results[11usize] = input.config.railway.requests
                        [cache + comptime!(CapacityWord::Terminal as usize)]
                    .load();
                    input.results[12usize] = u32::cast_from(capacity_stable_demand_recorded(
                        &input.config,
                        cache,
                        zero(),
                    )) + u32::cast_from(capacity_stable_demand_recorded(
                        &input.config,
                        cache,
                        ticks(0, 1),
                    ));
                }
                CapacityCheckMode::Restored => {
                    input.results[1usize] = input.config.railway.requests[cache].load();
                    input.results[2usize] = u32::cast_from(capacity_stable_demand_recorded(
                        &input.config,
                        cache,
                        ticks(1, 0),
                    ));
                    input.results[3usize] = u32::cast_from(!lt(
                        route_capacity(
                            &mut input.state,
                            &input.config,
                            &input.control,
                            &path,
                            ticks(1, 0),
                            comptime!(DemandSignal::Ignore),
                        ),
                        ticks(1, 0),
                    ));
                    input.results[4usize] = input.config.railway.requests
                        [cache + comptime!(CapacityWord::Terminal as usize)]
                    .load();
                }
                _ => {
                    prepare_request(&input.config, &path);
                    let regions = input.config.railway.region_count;
                    input.results[21usize] =
                        input.config.railway.requests[(2 * regions) as usize].load();
                    let first = input.config.railway.requests[regions as usize].load();
                    if comptime!(input.mode == CapacityCheckMode::Mixed) {
                        regional_set(
                            &mut input.state,
                            &input.config,
                            first,
                            ticks(10, 0),
                            comptime!(RailCapacityField::Available),
                        );
                    }
                    for sample in 0..9u32 {
                        if sample == 1 || sample == 3 || sample == 5 || sample == 6 || sample == 8 {
                            input.config.railway.requests[epoch].fetch_add(1);
                        }
                        if sample == 7 {
                            input.config.railway.requests[cache].store(0);
                        }
                        let mut quantity = ticks(8, 0);
                        match sample {
                            0 => quantity = ticks(3, 0),
                            1 => quantity = ticks(4, 0),
                            2 | 3 => quantity = ticks(5, 0),
                            4 => quantity = ticks(7, 0),
                            5 => quantity = ticks(0, 1),
                            6 => quantity = zero(),
                            _ => (),
                        }
                        if sample == 3 {
                            if comptime!(input.mode == CapacityCheckMode::Mixed) {
                                regional_set(
                                    &mut input.state,
                                    &input.config,
                                    first,
                                    ticks(9, 0),
                                    comptime!(RailCapacityField::Available),
                                );
                            }
                            input.results[19usize] = u32::cast_from(capacity_demand_recorded(
                                &input.config,
                                cache,
                                quantity,
                                input.config.railway.requests[epoch].load(),
                            ));
                        }
                        let available = match sample {
                            2 | 3 | 4 | 8 => route_capacity(
                                &mut input.state,
                                &input.config,
                                &input.control,
                                &path,
                                quantity,
                                comptime!(DemandSignal::Record),
                            ),
                            _ => route_capacity(
                                &mut input.state,
                                &input.config,
                                &input.control,
                                &path,
                                quantity,
                                comptime!(DemandSignal::Ignore),
                            ),
                        };
                        input.results[(1 + sample * 2) as usize] =
                            u32::cast_from(!lt(available, quantity));
                        input.results[(2 + sample * 2) as usize] =
                            input.config.railway.requests[cache].load();
                        if sample == 3 {
                            input.results[24usize] =
                                check_capacity_demands(&mut input.state, &input.config, 5);
                        }
                    }
                    input.results[20usize] =
                        check_capacity_demands(&mut input.state, &input.config, 8);
                    input.results[22usize] = input.config.railway.requests
                        [cache + comptime!(CapacityWord::Terminal as usize)]
                    .load();
                    let next_epoch = input.config.railway.requests[epoch].load() + 1;
                    input.results[23usize] = u32::cast_from(capacity_demand_recorded(
                        &input.config,
                        cache,
                        zero(),
                        next_epoch,
                    )) + u32::cast_from(capacity_demand_recorded(
                        &input.config,
                        cache,
                        ticks(0, 1),
                        next_epoch,
                    ));
                }
            }
        }
    }
}

#[test]
fn insufficient_capacity_proofs_preserve_demand_and_epoch_guards_on_gpu() -> Result<()> {
    let mut config: crate::domain::ScenarioConfig =
        serde_json::from_str(include_str!("../../../../scenarios/railway.json"))?;
    config.railway.length_unit = 100;
    let scenario = Scenario::new(config)?;
    let device = crate::simulation::tests::device(64)?;
    for mode in [CapacityCheckMode::Zero, CapacityCheckMode::Mixed] {
        let simulation = Simulation::new(scenario.clone(), device.clone())?;
        let mut initial = scenario.initial()?;
        for region in 0..scenario.rail_geography().regions.len() {
            for field in [RailCapacityField::Level, RailCapacityField::Available] {
                let cell = scenario
                    .rail_geography()
                    .coordinate(&scenario, &initial, field, region);
                initial.values[cell] = match field {
                    RailCapacityField::Level => 1,
                    _ => 0,
                };
            }
        }
        for actor in 0..initial.actors.len() {
            if initial.actors.at(actor).kind == Good::Railway as u32 {
                initial.set(Field::RailBuilt, actor, 1);
                initial.set(Field::RailOperating, actor, 1);
            }
        }
        device.write(
            &simulation.state,
            bytemuck::cast_slice(&crate::simulation::encode(&initial.values)),
        );
        for stage in [RailStage::Access, RailStage::Routes, RailStage::Quotes] {
            simulation.railways(stage);
        }
        let results = device.tensor(&[0; 25]);
        CapacityCheck::launch(
            &device,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            CapacityCheckInputLaunch {
                state: simulation.state.clone().into_buffer_arg(),
                actors: simulation.actors.clone().into_buffer_arg(),
                config: simulation.config.launch(),
                control: simulation.control.clone().into_buffer_arg(),
                results: results.clone().into_buffer_arg(),
                mode,
            },
        );
        let output = device.read(&results)?;
        let (terminal, refreshed) = match mode {
            CapacityCheckMode::Zero => (1, 2),
            _ => (0, 3),
        };
        assert!(output[0] < scenario.config().railway.companies.len() as u32);
        assert_eq!(
            output[1..19],
            [0, 1, 0, 1, 0, 2, 0, refreshed, 0, 3, 0, 4, 1, 5, 0, 5, 0, 6],
            "{mode:?}"
        );
        assert_eq!(output[19], terminal, "{mode:?}");
        assert_eq!(output[20], 0, "{mode:?}");
        assert!(output[21] > 1);
        assert_eq!(output[22], terminal, "{mode:?}");
        assert_eq!(output[23..25], [0, 0], "{mode:?}");
        for region in 0..scenario.rail_geography().regions.len() {
            let cell = scenario.rail_geography().coordinate(
                &scenario,
                &initial,
                RailCapacityField::Available,
                region,
            );
            initial.values[cell] = 100;
        }
        device.write(
            &simulation.state,
            bytemuck::cast_slice(&crate::simulation::encode(&initial.values)),
        );
        for stage in [RailStage::Access, RailStage::Routes, RailStage::Quotes] {
            simulation.railways(stage);
        }
        CapacityCheck::launch(
            &device,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            CapacityCheckInputLaunch {
                state: simulation.state.clone().into_buffer_arg(),
                actors: simulation.actors.clone().into_buffer_arg(),
                config: simulation.config.launch(),
                control: simulation.control.clone().into_buffer_arg(),
                results: results.clone().into_buffer_arg(),
                mode: CapacityCheckMode::Restored,
            },
        );
        assert_eq!(device.read(&results)?[1..5], [0, 0, 1, 0], "{mode:?}");
        CapacityCheck::launch(
            &device,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            CapacityCheckInputLaunch {
                state: simulation.state.clone().into_buffer_arg(),
                actors: simulation.actors.clone().into_buffer_arg(),
                config: simulation.config.launch(),
                control: simulation.control.clone().into_buffer_arg(),
                results: results.clone().into_buffer_arg(),
                mode: CapacityCheckMode::Stable,
            },
        );
        assert_eq!(
            device.read(&results)?[1..13],
            [1, 7, 1, 1, 0, 0, 7, 0, 8, 2, 1, 0],
            "{mode:?}"
        );
        assert_eq!(
            device.read(&simulation.control)?[crate::simulation::flags::FAULT_WORD],
            0
        );
        CapacityCheck::launch(
            &device,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            CapacityCheckInputLaunch {
                state: simulation.state.clone().into_buffer_arg(),
                actors: simulation.actors.clone().into_buffer_arg(),
                config: simulation.config.launch(),
                control: simulation.control.clone().into_buffer_arg(),
                results: results.clone().into_buffer_arg(),
                mode: CapacityCheckMode::Malformed,
            },
        );
        assert_eq!(
            device.read(&results)?[1..4],
            [
                0,
                crate::simulation::flags::FaultFlags::ARITHMETIC_OVERFLOW.bits(),
                crate::simulation::flags::FaultFlags::ARITHMETIC_OVERFLOW.bits(),
            ],
            "{mode:?}"
        );
    }
    Ok(())
}

struct DispatchTrade {
    buyer: usize,
    seller: usize,
}

#[derive(Clone, Copy, Debug)]
enum DispatchLayout {
    Concentrated,
    Distributed,
    Independent,
}

#[derive(Clone, Copy, Debug)]
enum DispatchExecution {
    Default,
    Global,
    Reference,
}

#[test]
fn cooperative_rail_dispatch_preserves_seeded_order_capacity_and_faults_on_gpu() -> Result<()> {
    use crate::domain::{Market, ScenarioConfig};
    use crate::simulation::{decode, encode};

    for layout in [
        DispatchLayout::Concentrated,
        DispatchLayout::Distributed,
        DispatchLayout::Independent,
    ] {
        let additional = match layout {
            DispatchLayout::Concentrated | DispatchLayout::Independent => 1,
            DispatchLayout::Distributed => 65,
        };
        let mut config: ScenarioConfig =
            serde_json::from_str(include_str!("../../../../scenarios/railway.json"))?;
        config.households = 547;
        config.firms += additional;
        let population = config.population.as_mut().expect("explicit population");
        let household = population.households[0].clone();
        population.households.extend((8..547).map(|index| {
            let mut household = household.clone();
            household.identity = 1000 + index;
            household
        }));
        let source = population.firms[0].clone();
        population.firms.extend((0..additional).map(|index| {
            let mut seller = source.clone();
            seller.identity = 9000 + u64::from(index);
            seller
        }));
        let scenario = Scenario::new(config)?;
        let first = scenario.config().households as usize;
        let second = first + 9;
        let sellers = std::iter::once(first)
            .chain(second..scenario.actors())
            .collect::<Vec<_>>();

        let mut replay = Vec::new();
        for workgroup in [32, 64, 128, 256] {
            let device = crate::simulation::tests::device(workgroup)?;
            for (case_index, case) in [
                DispatchCase::Ordinary,
                DispatchCase::FundedBatch,
                DispatchCase::PartialFreight,
                DispatchCase::Exhausted,
                DispatchCase::RetailCapacity,
                DispatchCase::SellerOverflow,
                DispatchCase::SalesOverflow,
                DispatchCase::BuyerOverflow,
                DispatchCase::RerouteOverflow,
                DispatchCase::ZeroQuantity,
                DispatchCase::WideQuantity,
                DispatchCase::RailwayBuyer,
            ]
            .into_iter()
            .enumerate()
            {
                let mut initial = scenario.initial()?;
                let count = initial.actors.len();
                for buyer in 0..first {
                    let cash = match case {
                        DispatchCase::FundedBatch => 10_000,
                        _ => u64::from(buyer % 7 != 0) * 10_000,
                    };
                    initial.set(Field::Cash, buyer, cash);
                }
                for &seller in &sellers {
                    initial.set(Field::Cash, seller, 2000);
                    initial.set(Field::Bread, seller, 600);
                    initial.set(Field::Price, seller, 10);
                }
                for region in 0..scenario.rail_geography().regions.len() {
                    for field in [RailCapacityField::Level, RailCapacityField::Available] {
                        let cell = scenario
                            .rail_geography()
                            .coordinate(&scenario, &initial, field, region);
                        initial.values[cell] = match field {
                            RailCapacityField::Level => 1,
                            _ => 100,
                        };
                    }
                }
                for actor in initial.actors.firm_indices().collect::<Vec<_>>() {
                    if initial.actors.at(actor).kind == Good::Railway as u32 {
                        initial.set(Field::RailBuilt, actor, 1);
                        initial.set(Field::RailOperating, actor, 1);
                        initial.set(Field::Railway, actor, 1000);
                    }
                }
                match case {
                    DispatchCase::PartialFreight => {
                        for &seller in &sellers {
                            initial.set(Field::Cash, seller, 0);
                        }
                    }
                    DispatchCase::Exhausted => initial.set(Field::Bread, first, 11),
                    DispatchCase::RetailCapacity => {
                        initial.set(Field::EnterpriseModel, first, 1);
                        initial.set(Field::RetailOpen, first, 17);
                    }
                    DispatchCase::SellerOverflow => initial.set(Field::Cash, second, u64::MAX - 15),
                    DispatchCase::SalesOverflow => initial.set(Field::Sales, second, u64::MAX - 2),
                    DispatchCase::BuyerOverflow => initial.set(Field::Transit, 33, u64::MAX),
                    DispatchCase::RerouteOverflow => {
                        let field = BASE_FIELDS + (Good::Bread as usize - 1) * scenario.ring() + 1;
                        initial.values[field * count + 33] = u64::MAX;
                    }
                    _ => (),
                }
                let mut trades = (0..first)
                    .map(|buyer| DispatchTrade {
                        buyer,
                        seller: match layout {
                            DispatchLayout::Concentrated | DispatchLayout::Independent => {
                                match buyer < 273 {
                                    true => first,
                                    false => second,
                                }
                            }
                            DispatchLayout::Distributed => sellers[buyer % sellers.len()],
                        },
                    })
                    .collect::<Vec<_>>();
                match layout {
                    DispatchLayout::Independent => (),
                    _ => {
                        trades.insert(
                            273,
                            DispatchTrade {
                                buyer: second,
                                seller: first,
                            },
                        );
                        trades.push(DispatchTrade {
                            buyer: first,
                            seller: second,
                        });
                    }
                }
                if matches!(case, DispatchCase::RailwayBuyer) {
                    trades.insert(
                        17,
                        DispatchTrade {
                            buyer: first + 1,
                            seller: first,
                        },
                    );
                }
                trades.sort_by_key(|trade| trade.seller);
                let mut proposals = vec![NO_ACTOR; count * ProposalLayout::WORDS];
                let mut scratch = vec![0; 8 * count];
                scratch[5 * count..6 * count].fill(NO_ACTOR);
                let mut flags = vec![0; count * (REJECTION_WORDS + 1) + 1];
                for &seller in &sellers {
                    flags[seller] = Settlement::Serial as u32;
                    flags[count + 1 + seller * REJECTION_WORDS] = u32::MAX;
                }
                for (index, &DispatchTrade { buyer, seller }) in trades.iter().enumerate() {
                    let quantity = match (case, buyer) {
                        (DispatchCase::ZeroQuantity, 33) => 0,
                        (DispatchCase::WideQuantity, 33) => u64::from(u32::MAX) + 2,
                        _ => (buyer % 3 + 1) as u64,
                    };
                    proposals[buyer * ProposalLayout::WORDS + ProposalLayout::ACTOR] =
                        seller as u32;
                    proposals[buyer * ProposalLayout::WORDS + ProposalLayout::QUANTITY_LOW] =
                        quantity as u32;
                    proposals[buyer * ProposalLayout::WORDS + ProposalLayout::QUANTITY_HIGH] =
                        (quantity >> 32) as u32;
                    scratch[index] = seller as u32;
                    scratch[count + index] = buyer as u32;
                    scratch[7 * count + seller] = (index + 1) as u32;
                }
                let mut results = Vec::new();
                for execution in [
                    DispatchExecution::Default,
                    DispatchExecution::Global,
                    DispatchExecution::Reference,
                ] {
                    let mut simulation = Simulation::new(scenario.clone(), device.clone())?;
                    if matches!(
                        execution,
                        DispatchExecution::Global | DispatchExecution::Reference
                    ) {
                        simulation.config.railway.request_storage = RequestStorage::Global;
                    }
                    device.write(
                        &simulation.state,
                        bytemuck::cast_slice(&encode(&initial.values)),
                    );
                    device.write(&simulation.proposals, &proposals);
                    let mut buffers = vec![0; simulation.scratch.meta.num_elements()];
                    buffers[..scratch.len()].copy_from_slice(&scratch);
                    device.write(&simulation.scratch, &buffers);
                    flags[count] = match execution {
                        DispatchExecution::Reference => Settlement::Serial as u32,
                        _ => Settlement::Parallel as u32,
                    };
                    device.write(&simulation.config.railway.settlements, &flags);
                    let mut control = vec![0; simulation.control.meta.num_elements()];
                    control[CONSTRUCTED_EVENTS_WORD] = trades.len() as u32;
                    device.write(&simulation.control, &control);
                    for stage in [RailStage::Access, RailStage::Routes, RailStage::Quotes] {
                        simulation.railways(stage);
                    }
                    let mut parameters = simulation.parameters(
                        Phase::ProvisionPurchase,
                        Market::Household,
                        0,
                        count,
                    )?;
                    parameters.discovery = Discovery::Indexed;
                    for stage in [
                        EconomyStage::RailClassify,
                        EconomyStage::RailDependencies,
                        EconomyStage::GroupPrepare,
                        EconomyStage::GroupRanges,
                        EconomyStage::RailExecute,
                        EconomyStage::GroupPrepare,
                        EconomyStage::GroupExecute,
                        EconomyStage::GroupRemainder,
                        EconomyStage::GroupFinalize,
                    ] {
                        let invocations = match stage {
                            EconomyStage::RailExecute => 1,
                            _ => trades.len(),
                        };
                        if stage == EconomyStage::RailExecute {
                            markets::prepare_dispatch(
                                &device,
                                &simulation.control,
                                &simulation.searches.dispatches,
                                count,
                            );
                            planning::launch(
                                &device,
                                &simulation.state,
                                &simulation.actors,
                                &simulation.config,
                                &simulation.proposals,
                                &simulation.scratch,
                                &simulation.control,
                                &simulation.searches.dispatches,
                                &parameters,
                                planning::PlanningStage::Prepare,
                            );
                            if matches!(
                                (layout, case, execution),
                                (
                                    DispatchLayout::Independent,
                                    DispatchCase::Ordinary,
                                    DispatchExecution::Default | DispatchExecution::Global
                                )
                            ) {
                                assert!(
                                    planning::planned_sellers(&device, &simulation.config)? > 0
                                );
                            }
                        }
                        EconomySystem::launch(
                            &device,
                            device.count(invocations),
                            CubeDim::new_1d(workgroup),
                            EconomyInputLaunch {
                                state: simulation.state.clone().into_buffer_arg(),
                                actors: simulation.actors.clone().into_buffer_arg(),
                                config: simulation.config.launch(),
                                spatial: simulation.spatial.clone().into_buffer_arg(),
                                proposals: simulation.proposals.clone().into_buffer_arg(),
                                bundles: simulation.scratch.clone().into_buffer_arg(),
                                control: simulation.control.clone().into_buffer_arg(),
                                parameters: parameters.launch(),
                                stage,
                            },
                        );
                        if stage == EconomyStage::RailExecute {
                            planning::launch(
                                &device,
                                &simulation.state,
                                &simulation.actors,
                                &simulation.config,
                                &simulation.proposals,
                                &simulation.scratch,
                                &simulation.control,
                                &simulation.searches.dispatches,
                                &parameters,
                                planning::PlanningStage::Finish,
                            );
                        }
                    }
                    let state = device.read(&simulation.state)?;
                    if matches!(case, DispatchCase::Ordinary)
                        && !matches!(layout, DispatchLayout::Independent)
                    {
                        let values = decode(&state);
                        assert!(values[Field::RailRevenue as usize * count + first + 1] > 0);
                        assert!(values[Field::Purchased as usize * count + first] > 0);
                        assert!(values[Field::Purchased as usize * count + second] > 0);
                    }
                    let mut result = state;
                    result.extend(device.read(&simulation.control)?);
                    result.extend(device.read_range(&simulation.scratch, 4 * count, count)?);
                    results.push(result);
                }
                for (variant, expected) in results.iter().enumerate().skip(1) {
                    assert_eq!(results[0].len(), expected.len());
                    for (word, (actual, expected)) in results[0].iter().zip(expected).enumerate() {
                        assert_eq!(
                            actual, expected,
                            "{layout:?} {workgroup} {case:?} variant {variant} word {word}"
                        );
                    }
                }
                match workgroup {
                    32 => replay.push(results.remove(0)),
                    _ => assert_eq!(
                        replay[case_index], results[0],
                        "workgroup replay {workgroup} {case:?}"
                    ),
                }
            }
        }
    }
    Ok(())
}

use super::*;
use crate::domain::{Experiment, Scenario};
use crate::simulation::Simulation;
use anyhow::Result;

#[test]
fn incident_routes_preserve_original_parallel_segment_ties_on_gpu() -> Result<()> {
    use crate::domain::{Checkpoint, company::Company};

    for first in [0usize, 1] {
        let mut config: crate::domain::ScenarioConfig =
            serde_json::from_str(include_str!("../../../../../scenarios/railway.json"))?;
        let mut primary = config.railway.segments[1].clone();
        let mut secondary = primary.clone();
        primary.company = config.railway.companies[first].firm;
        secondary.company = config.railway.companies[1 - first].firm;
        let destination = primary.to;
        config.railway.segments.truncate(1);
        config.railway.segments.extend([primary, secondary]);
        let company = Company::new(Company {
            identity: config.railway.companies[0].firm,
            sites: config
                .railway
                .companies
                .iter()
                .map(|company| company.firm)
                .collect(),
        })?;
        for firm in &mut config
            .population
            .as_mut()
            .expect("explicit population")
            .firms
        {
            if company.sites.contains(&firm.identity) {
                firm.owner = 2;
            }
        }
        config.companies.push(company);
        let scenario = Scenario::new(config)?;
        let mut initial = scenario.initial()?;
        for company in &scenario.config().railway.companies {
            let firm = initial
                .actors
                .firm_indices()
                .find(|&index| initial.actors.at(index).identity == company.firm)
                .expect("railway operator");
            let built = scenario
                .config()
                .railway
                .segments
                .iter()
                .filter(|segment| segment.company == company.firm)
                .count() as u64;
            initial.set(Field::RailBuilt, firm, built);
            initial.set(Field::RailOperating, firm, built);
        }
        let simulation = Simulation::from_checkpoint(
            Checkpoint::new(&scenario, &initial)?,
            crate::simulation::tests::device(64)?,
        )?;
        simulation.railways(RailStage::Routes);
        let stations = scenario.config().railway.stations();
        let target = stations
            .iter()
            .position(|&position| position == destination)
            .expect("destination station");
        let route = target * ROUTE_WORDS;
        let routes = simulation.device.read(&simulation.config.railway.routes)?;
        assert_eq!(routes[route], 900);
        assert_eq!(routes[route + 1], 1);
        assert_eq!(routes[route + 2], 0);
        assert_eq!(routes[route + 5], 0);
    }
    Ok(())
}

#[test]
fn unchanged_railway_companies_reuse_buffers_and_refresh_remapped_sites_on_gpu() -> Result<()> {
    use crate::domain::Actor;

    let scenario = Scenario::new(serde_json::from_str(include_str!(
        "../../../../../scenarios/railway.json"
    ))?)?;
    let device = crate::simulation::tests::device(64)?;
    let mut state = scenario.initial()?;
    let original = Network::new(&device, &scenario, &state);
    let retained = original.updated(&device, &scenario, &state);
    state.actors.extend([Actor {
        identity: state.next_identity,
        position: state.actors.at(0).position,
        kind: Good::Bread as u32,
        owner: 0,
        capacity: 1,
    }]);
    state.next_identity += 1;
    let extended = retained.updated(&device, &scenario, &state);
    for cached in [&retained, &extended] {
        for (before, after) in [
            (&original.companies, &cached.companies),
            (&original.access_rights, &cached.access_rights),
        ] {
            assert!(crate::simulation::market_graphs::same_buffer(before, after));
        }
    }
    let operators = scenario
        .config()
        .railway
        .companies
        .iter()
        .map(|company| {
            state
                .actors
                .firm_indices()
                .find(|&index| state.actors.at(index).identity == company.firm)
                .expect("railway operator")
        })
        .collect::<Vec<_>>();
    let first = state.actors.at(operators[0]);
    let second = state.actors.at(operators[1]);
    state.actors.set(operators[0], second);
    state.actors.set(operators[1], first);
    let remapped = extended.updated(&device, &scenario, &state);
    let rebuilt = Network::new(&device, &scenario, &state);
    assert!(!crate::simulation::market_graphs::same_buffer(
        &extended.companies,
        &remapped.companies
    ));
    assert!(!crate::simulation::market_graphs::same_buffer(
        &extended.access_rights,
        &remapped.access_rights
    ));
    assert_eq!(
        device.read(&remapped.companies)?,
        device.read(&rebuilt.companies)?
    );
    assert_eq!(
        device.read(&remapped.access_rights)?,
        device.read(&rebuilt.access_rights)?
    );
    Ok(())
}

#[derive(Debug, PartialEq, Eq)]
struct RailTables {
    routes: Vec<u32>,
    quotes: Vec<u32>,
}

impl RailTables {
    fn read(device: &Device, network: &Network) -> Result<Self> {
        Ok(Self {
            routes: device.read(&network.routes)?,
            quotes: device.read(&network.quotes)?,
        })
    }

    fn rebuilt(simulation: &Simulation, state: &crate::domain::State) -> Result<Self> {
        let config = Configuration::new(&simulation.device, &simulation.scenario, state);
        for stage in [RailStage::Routes, RailStage::Quotes] {
            launch(
                DeviceEconomy {
                    config: &config,
                    ..simulation.gpu()
                },
                &simulation.control,
                stage,
            );
        }
        Self::read(&simulation.device, &config.railway)
    }
}

fn route_cache_states(simulation: &Simulation) -> Result<Vec<u32>> {
    simulation.device.read_range(
        &simulation.config.railway.companies,
        simulation.config.railway.company_count as usize * COMPANY_WORDS,
        2,
    )
}

#[test]
fn exact_route_cache_reuses_clean_tables_and_preserves_pending_quotes_on_gpu() -> Result<()> {
    use crate::domain::Checkpoint;
    use crate::simulation::encode;

    let scenario = Scenario::new(serde_json::from_str(include_str!(
        "../../../../../scenarios/railway.json"
    ))?)?;
    let mut state = scenario.initial()?;
    let operators = scenario
        .config()
        .railway
        .companies
        .iter()
        .map(|company| {
            state
                .actors
                .firm_indices()
                .find(|&index| state.actors.at(index).identity == company.firm)
                .expect("railway operator")
        })
        .collect::<Vec<_>>();
    for &firm in &operators {
        state.set(Field::RailBuilt, firm, 1);
        state.set(Field::RailOperating, firm, 1);
    }
    let simulation = Simulation::from_checkpoint(
        Checkpoint::new(&scenario, &state)?,
        crate::simulation::tests::device(64)?,
    )?;
    let unknown = RouteCacheState::Unknown as u32;
    let dirty = RouteCacheState::Dirty as u32;
    let clean = RouteCacheState::Clean as u32;
    assert_eq!(route_cache_states(&simulation)?, [unknown, unknown]);
    simulation.railways(RailStage::Routes);
    assert_eq!(route_cache_states(&simulation)?, [dirty, dirty]);
    simulation.railways(RailStage::Quotes);
    assert_eq!(route_cache_states(&simulation)?, [dirty, clean]);
    let original = RailTables::read(&simulation.device, &simulation.config.railway)?;
    let mut marked = RailTables {
        routes: original.routes.clone(),
        quotes: original.quotes.clone(),
    };
    marked.routes[0] = 0x1234_5678;
    marked.quotes[0] = 0x8765_4321;
    simulation
        .device
        .write(&simulation.config.railway.routes, &marked.routes);
    simulation
        .device
        .write(&simulation.config.railway.quotes, &marked.quotes);
    for stage in [RailStage::Routes, RailStage::Routes, RailStage::Quotes] {
        simulation.railways(stage);
    }
    assert_eq!(route_cache_states(&simulation)?, [clean, clean]);
    assert_eq!(
        RailTables::read(&simulation.device, &simulation.config.railway)?,
        marked
    );
    simulation
        .device
        .write(&simulation.config.railway.routes, &original.routes);
    simulation
        .device
        .write(&simulation.config.railway.quotes, &original.quotes);
    state.set(Field::RailOperating, operators[1], 0);
    simulation.device.write(
        &simulation.state,
        bytemuck::cast_slice(&encode(&state.values)),
    );
    simulation.railways(RailStage::Quotes);
    assert_eq!(
        RailTables::read(&simulation.device, &simulation.config.railway)?,
        original
    );
    simulation.railways(RailStage::Routes);
    assert_eq!(route_cache_states(&simulation)?, [dirty, dirty]);
    simulation.railways(RailStage::Routes);
    assert_eq!(route_cache_states(&simulation)?, [clean, dirty]);
    simulation.railways(RailStage::Quotes);
    assert_eq!(route_cache_states(&simulation)?, [clean, clean]);
    let suspended = RailTables::read(&simulation.device, &simulation.config.railway)?;
    assert_eq!(suspended, RailTables::rebuilt(&simulation, &state)?);
    assert_ne!(suspended, original);
    state.set(Field::RailOperating, operators[1], 1);
    simulation.device.write(
        &simulation.state,
        bytemuck::cast_slice(&encode(&state.values)),
    );
    simulation.railways(RailStage::Routes);
    simulation.railways(RailStage::Quotes);
    assert_eq!(
        RailTables::read(&simulation.device, &simulation.config.railway)?,
        original
    );
    assert_eq!(
        RailTables::read(&simulation.device, &simulation.config.railway)?,
        RailTables::rebuilt(&simulation, &state)?
    );
    Ok(())
}

#[test]
fn exact_route_cache_initializes_empty_service_and_invalidates_company_access_on_gpu() -> Result<()>
{
    use crate::domain::{Checkpoint, company::Company, railway::RailSharing};
    use crate::simulation::encode;

    let mut config: crate::domain::ScenarioConfig =
        serde_json::from_str(include_str!("../../../../../scenarios/railway.json"))?;
    for company in &mut config.railway.companies {
        company.sharing = RailSharing::Private;
    }
    let scenario = Scenario::new(config)?;
    let mut state = scenario.initial()?;
    let mut simulation = Simulation::from_checkpoint(
        Checkpoint::new(&scenario, &state)?,
        crate::simulation::tests::device(64)?,
    )?;
    simulation.railways(RailStage::Routes);
    simulation.railways(RailStage::Quotes);
    assert_eq!(
        RailTables::read(&simulation.device, &simulation.config.railway)?,
        RailTables::rebuilt(&simulation, &state)?
    );
    for company in &scenario.config().railway.companies {
        let firm = state
            .actors
            .firm_indices()
            .find(|&index| state.actors.at(index).identity == company.firm)
            .expect("railway operator");
        state.set(Field::RailBuilt, firm, 1);
        state.set(Field::RailOperating, firm, 1);
    }
    simulation.device.write(
        &simulation.state,
        bytemuck::cast_slice(&encode(&state.values)),
    );
    simulation.railways(RailStage::Routes);
    simulation.railways(RailStage::Quotes);
    simulation.railways(RailStage::Routes);
    let private = RailTables::read(&simulation.device, &simulation.config.railway)?;
    let previous = state.actors.clone();
    let company = Company::new(Company {
        identity: scenario.config().railway.companies[0].firm,
        sites: scenario
            .config()
            .railway
            .companies
            .iter()
            .map(|company| company.firm)
            .collect(),
    })?;
    company.initialize(&mut state)?;
    state.companies.push(company);
    let updated = simulation
        .config
        .updated(&simulation.device, &scenario, &previous, &state);
    let before = &simulation.config.railway;
    let after = &updated.railway;
    for (original, retained) in [
        (&before.stations, &after.stations),
        (&before.access, &after.access),
        (&before.station_companies, &after.station_companies),
        (&before.segments, &after.segments),
        (&before.materials, &after.materials),
        (&before.routes, &after.routes),
        (&before.quotes, &after.quotes),
        (&before.regions, &after.regions),
        (&before.segment_regions, &after.segment_regions),
    ] {
        assert!(crate::simulation::market_graphs::same_buffer(
            original, retained
        ));
    }
    assert_eq!(RailTables::read(&simulation.device, after)?, private);
    simulation.config = updated;
    assert_eq!(
        route_cache_states(&simulation)?,
        [RouteCacheState::Unknown as u32; 2]
    );
    let rebuilt = Configuration::new(&simulation.device, &scenario, &state);
    assert_eq!(
        simulation
            .device
            .read(&simulation.config.railway.companies)?,
        simulation.device.read(&rebuilt.railway.companies)?
    );
    assert_eq!(
        simulation
            .device
            .read(&simulation.config.railway.access_rights)?,
        simulation.device.read(&rebuilt.railway.access_rights)?
    );
    simulation.railways(RailStage::Routes);
    simulation.railways(RailStage::Quotes);
    let shared = RailTables::read(&simulation.device, &simulation.config.railway)?;
    assert_eq!(shared, RailTables::rebuilt(&simulation, &state)?);
    assert_ne!(shared, private);
    Ok(())
}

#[test]
fn carrier_route_cache_preserves_unaffected_rows_and_access_dependencies_on_gpu() -> Result<()> {
    use crate::domain::{
        Checkpoint,
        company::Company,
        railway::{RailAgreement, RailSharing},
    };
    use crate::simulation::encode;

    #[derive(Clone, Copy, Debug)]
    enum Access {
        Private,
        Shared,
        Agreement,
        Unified,
    }

    let clean = RouteCacheState::Clean as u32;
    let dirty = RouteCacheState::Dirty as u32;
    for workgroup in [32, 256] {
        for access in [
            Access::Private,
            Access::Shared,
            Access::Agreement,
            Access::Unified,
        ] {
            let mut config: crate::domain::ScenarioConfig =
                serde_json::from_str(include_str!("../../../../../scenarios/railway.json"))?;
            for company in &mut config.railway.companies {
                company.sharing = match access {
                    Access::Shared => RailSharing::Shared,
                    _ => RailSharing::Private,
                };
            }
            match access {
                Access::Agreement => config.railway.sharing_agreements.push(RailAgreement {
                    owner: config.railway.companies[1].firm,
                    carrier: config.railway.companies[0].firm,
                }),
                Access::Unified => {
                    let company = Company::new(Company {
                        identity: config.railway.companies[0].firm,
                        sites: config
                            .railway
                            .companies
                            .iter()
                            .map(|company| company.firm)
                            .collect(),
                    })?;
                    for firm in &mut config
                        .population
                        .as_mut()
                        .expect("explicit population")
                        .firms
                    {
                        if company.sites.contains(&firm.identity) {
                            firm.owner = 2;
                        }
                    }
                    config.companies.push(company);
                }
                _ => (),
            }
            let scenario = Scenario::new(config)?;
            let mut state = scenario.initial()?;
            let operators = scenario
                .config()
                .railway
                .companies
                .iter()
                .map(|company| {
                    state
                        .actors
                        .firm_indices()
                        .find(|&index| state.actors.at(index).identity == company.firm)
                        .expect("railway operator")
                })
                .collect::<Vec<_>>();
            for &firm in &operators {
                state.set(Field::RailBuilt, firm, 1);
                state.set(Field::RailOperating, firm, 1);
            }
            let simulation = Simulation::from_checkpoint(
                Checkpoint::new(&scenario, &state)?,
                crate::simulation::tests::device(workgroup)?,
            )?;
            for stage in [RailStage::Routes, RailStage::Quotes] {
                simulation.railways(stage);
            }
            let companies = operators.len();
            let cache = companies * COMPANY_WORDS + 2 + companies;
            assert_eq!(
                simulation.device.read_range(
                    &simulation.config.railway.companies,
                    cache,
                    companies
                )?,
                [dirty, dirty],
            );
            let row_words = (simulation.config.railway.nodes as usize).pow(2) * ROUTE_WORDS;
            for owner in [1usize, 0, 1] {
                let mut marked = simulation.device.read(&simulation.config.railway.routes)?;
                for carrier in 0..companies {
                    marked[carrier * row_words] = 0x1234_5678 + carrier as u32;
                }
                simulation
                    .device
                    .write(&simulation.config.railway.routes, &marked);
                let firm = operators[owner];
                state.set(
                    Field::RailOperating,
                    firm,
                    1 - state.get(Field::RailOperating, firm),
                );
                simulation.device.write(
                    &simulation.state,
                    bytemuck::cast_slice(&encode(&state.values)),
                );
                let expected = RailTables::rebuilt(&simulation, &state)?;
                simulation.railways(RailStage::Routes);
                let flags = match access {
                    Access::Private => match owner {
                        0 => [dirty, clean],
                        _ => [clean, dirty],
                    },
                    Access::Agreement => match owner {
                        0 => [dirty, clean],
                        _ => [dirty, dirty],
                    },
                    Access::Shared | Access::Unified => [dirty, dirty],
                };
                assert_eq!(
                    simulation.device.read_range(
                        &simulation.config.railway.companies,
                        cache,
                        companies
                    )?,
                    flags,
                    "workgroup {workgroup}, access {access:?}, owner {owner}",
                );
                let mut actual = simulation.device.read(&simulation.config.railway.routes)?;
                for carrier in 0..companies {
                    let rows = carrier * row_words..(carrier + 1) * row_words;
                    let reference = match flags[carrier] == dirty {
                        true => &expected.routes,
                        false => &marked,
                    };
                    assert_eq!(actual[rows.clone()], reference[rows], "carrier {carrier}");
                    actual[carrier * row_words] = expected.routes[carrier * row_words];
                }
                simulation
                    .device
                    .write(&simulation.config.railway.routes, &actual);
                simulation.railways(RailStage::Quotes);
                assert_eq!(
                    RailTables::read(&simulation.device, &simulation.config.railway)?,
                    expected
                );
                simulation.railways(RailStage::Routes);
                assert_eq!(
                    simulation.device.read_range(
                        &simulation.config.railway.companies,
                        cache,
                        companies
                    )?,
                    [clean, clean],
                );
            }
        }
    }
    Ok(())
}

#[derive(CubeType, CubeLaunch)]
struct QuoteCheckInput {
    actors: Box<[u32]>,
    config: EconomyConfig,
    mismatches: Box<[Atomic<u32>]>,
}

struct QuoteCheck;
struct CompareQuotes;

impl System for QuoteCheck {
    type Input = QuoteCheckInput;
    type Passes = gpu_passes![CompareQuotes];
}

#[cube]
impl GpuPass<QuoteCheckInput> for CompareQuotes {
    type Output = ();

    fn execute(input: QuoteCheckInput) {
        let actors = ActorTable::new(&input.actors);
        let index = ABSOLUTE_POS as u32;
        let count = input.config.actors;
        if index < count * count {
            let buyer = index / count;
            let seller = index % count;
            if actors.distance(buyer, seller) <= input.config.search_radius {
                let mut expected = FreightRoute::by_road(&actors, &input.config, buyer, seller);
                for company in 0..input.config.railway.company_count {
                    let path = FreightRequest::new(&actors, &input.config, buyer, seller)
                        .candidate(&input.config, company);
                    if path.preferred_to(&expected) {
                        expected.replace(&path);
                    }
                }
                let actual = FreightRequest::new(&actors, &input.config, buyer, seller)
                    .preferred(&input.config);
                let node = input.config.railway.nearest_station(&actors, buyer);
                let access = input.config.railway.station_distance(&actors, buyer, node);
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
    let config = serde_json::from_str(include_str!("../../../../../scenarios/railway.json"))?;
    let railway = Scenario::new(config)?;
    let mut slow = railway.config().clone();
    slow.shipment_speed = 16;
    slow.railway.speed_multiplier = 2;
    let scenarios = [
        railway,
        Scenario::new(slow)?,
        Scenario::industrial(17, 42, Experiment::Baseline)?,
        Scenario::industrial(17, 42, Experiment::Transport)?,
    ];
    for scenario in scenarios {
        let device = crate::simulation::tests::device(64)?;
        let mut simulation = Simulation::new(scenario, device.clone())?;
        for _ in 0..2 {
            for stage in [
                RailStage::Infrastructure,
                RailStage::Access,
                RailStage::Routes,
                RailStage::Quotes,
            ] {
                simulation.railways(stage);
            }
            simulation.fault_check()?;
            let mismatches = device.tensor(&[0]);
            let count = simulation.metadata.actors.len();
            QuoteCheck::launch_elements(
                &device,
                count * count,
                QuoteCheckInputLaunch {
                    actors: simulation.actors.buffer_arg(),
                    config: simulation.config.launch(),
                    mismatches: mismatches.buffer_arg(),
                },
            );
            assert_eq!(device.read(&mismatches)?, [0]);
        }
    }
    Ok(())
}

#[test]
fn repeated_input_initialization_resets_prior_suppliers_and_preserves_faults_on_gpu() -> Result<()>
{
    use crate::simulation::{Market, encode, flags::FAULT_WORD};

    #[derive(Clone, Copy, Debug)]
    enum ResetCase {
        Suppliers,
        Empty,
        Fault,
    }

    let workgroup = 64;
    let device = crate::simulation::tests::device(workgroup)?;
    for case in [ResetCase::Suppliers, ResetCase::Empty, ResetCase::Fault] {
        let scenario = Scenario::industrial(67, 42, Experiment::Transport)?;
        let mut initial = scenario.initial()?;
        initial.set(Field::Bread, 0, 0);
        let simulation = Simulation::new(scenario, device.clone())?;
        let count = initial.actors.len();
        let parameters = simulation.parameters(
            Phase::ProvisionPurchase,
            Market::Input(Good::Bread),
            0,
            count,
        )?;
        device.write(
            &simulation.state,
            bytemuck::cast_slice(&encode(&initial.values)),
        );
        let mut control = vec![0; simulation.control.meta.num_elements()];
        device.write(&simulation.control, &control);
        market::MarketStart::ResetCandidatesAndOrders
            .launch(simulation.market_buffers(), &parameters);
        let clean_statuses = device.read_range(&simulation.work.scratch, 4 * count, count)?;
        let cells = (simulation.config.spatial_side * simulation.config.spatial_side) as usize;
        let mut spatial = vec![0; simulation.spatial.meta.num_elements()];
        match case {
            ResetCase::Suppliers | ResetCase::Fault => {
                let suppliers = [0, 67, 68];
                spatial[cells] = suppliers.len() as u32;
                for (index, seller) in suppliers.into_iter().enumerate() {
                    spatial[cells + 1 + index] = seller as u32;
                }
            }
            ResetCase::Empty => (),
        }
        control[FAULT_WORD] = match case {
            ResetCase::Fault => FaultFlags::ARITHMETIC_OVERFLOW.bits(),
            _ => 0,
        };
        let sentinel = 0x1234_5678;
        device.write(&simulation.control, &control);
        device.write(&simulation.spatial, &spatial);
        device.write(
            &simulation.work.scratch,
            &vec![sentinel; simulation.work.scratch.meta.num_elements()],
        );
        market::MarketStart::ResetOrders.launch(simulation.market_buffers(), &parameters);
        let expected_statuses = match case {
            ResetCase::Fault => vec![sentinel; count],
            _ => clean_statuses,
        };
        assert_eq!(
            device.read_range(&simulation.work.scratch, 4 * count, count)?,
            expected_statuses,
            "{case:?}, workgroup {workgroup}, statuses"
        );
        let actual_control = device.read(&simulation.control)?;
        assert_eq!(actual_control[FAULT_WORD], control[FAULT_WORD]);
        assert_eq!(actual_control[LATCH_WORD], control[FAULT_WORD]);
    }
    Ok(())
}

#[test]
fn indexed_routes_match_dense_routes_quotes_and_destination_access_on_gpu() -> Result<()> {
    let scenario = Scenario::new(serde_json::from_str(include_str!(
        "../../../../../scenarios/railway.json"
    ))?)?;
    let mut state = scenario.initial()?;
    let network = &scenario.config().railway;
    for company in &network.companies {
        let firm = scenario.index_of(company.firm).unwrap();
        let built = network
            .segments
            .iter()
            .filter(|segment| segment.company == company.firm)
            .count() as u64;
        state.set(Field::RailBuilt, firm, built);
        state.set(Field::RailOperating, firm, built);
    }
    let device = crate::simulation::tests::device(64)?;
    let simulation = Simulation::from_state(scenario.clone(), state.clone(), device.clone())?;
    let stations = network.stations();
    let tables = |storage| {
        let mut config = Configuration::new(&device, &scenario, &state);
        config.state_stride = simulation.config.state_stride;
        config.railway = Network::with_stations(&device, &scenario, &state, &stations, storage);
        for stage in [RailStage::Routes, RailStage::Quotes] {
            launch(
                DeviceEconomy {
                    device: &device,
                    config: &config,
                    ..simulation.gpu()
                },
                &simulation.control,
                stage,
            );
        }
        config.railway
    };
    let dense = tables(RouteStorage::Dense);
    let indexed = tables(RouteStorage::Indexed);
    let lookup = device.read(&indexed.station_companies)?;
    let expected = RailTables::read(&device, &dense)?;
    let actual = RailTables::read(&device, &indexed)?;
    let nodes = stations.len();
    let row = nodes * ROUTE_WORDS;
    let missing = [UNREACHABLE, u32::MAX, u32::MAX, 0, 0, UNREACHABLE].repeat(nodes);
    assert!(indexed.route_rows < dense.route_rows);
    for i in 0..dense.route_rows as usize {
        let position = lookup[indexed.route_index as usize + i] as usize;
        let expected_row = match position == indexed.route_rows as usize {
            true => missing.as_slice(),
            false => &expected.routes[i * row..(i + 1) * row],
        };
        assert_eq!(
            &actual.routes[position * row..(position + 1) * row],
            expected_row
        );
    }
    assert_eq!(&actual.quotes[..expected.quotes.len()], &expected.quotes);
    for destination in 0..nodes {
        let connected = (0..nodes).any(|origin| {
            expected.quotes[(origin * nodes + destination + 1) * scenario.ring() - 1]
                < dense.company_count
        });
        assert_eq!(
            actual.quotes[expected.quotes.len() + destination],
            u32::from(connected)
        );
    }
    Ok(())
}

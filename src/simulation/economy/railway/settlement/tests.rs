use super::*;
use crate::domain::{Market, Scenario};
use crate::simulation::Simulation;
use anyhow::Result;

#[test]
fn freight_capacity_reuse_refreshes_claim_offsets_after_actor_growth_on_gpu() -> Result<()> {
    let device = crate::simulation::tests::device(64)?;
    let actors = 17;
    let regions = 3;
    let mut buffers = Buffers::new(&device, actors, regions);
    let capacity = buffers.accounts.meta.num_elements() / ACCOUNT_WORDS;
    assert_eq!(buffers.claim_capacity, 0);
    assert_eq!(buffers.claims.meta.num_elements(), capacity + regions + 1);
    buffers.claim_capacity = capacity * 16;
    let claim_words = buffers.claim_capacity
        * (FREIGHT_CLAIM_WORDS + crate::domain::railway::FREIGHT_CLAIM_SORT_WORDS);
    buffers.claims = device.empty(claim_words + actors + regions + 1);
    let updated = buffers.updated(&device, actors + 1, regions);
    assert!(crate::simulation::market_graphs::same_buffer(
        &buffers.resources,
        &updated.resources
    ));
    assert!(crate::simulation::market_graphs::same_buffer(
        &buffers.accounts,
        &updated.accounts
    ));
    assert!(!crate::simulation::market_graphs::same_buffer(
        &buffers.claims,
        &updated.claims
    ));
    assert!(updated.claims.meta.num_elements() >= claim_words + capacity + regions + 1);
    let retained = updated.updated(&device, capacity, regions);
    assert!(crate::simulation::market_graphs::same_buffer(
        &updated.claims,
        &retained.claims
    ));
    let grown = retained.updated(&device, capacity + 1, regions);
    assert!(!crate::simulation::market_graphs::same_buffer(
        &retained.accounts,
        &grown.accounts
    ));
    assert!(grown.accounts.meta.num_elements() >= (capacity + 1) * ACCOUNT_WORDS);
    assert!(grown.claim_capacity >= buffers.claim_capacity);
    Ok(())
}

#[test]
fn reused_freight_after_firm_entry_matches_rebuilt_markets_and_cash_on_gpu() -> Result<()> {
    use crate::simulation::{RegistryAudit, enterprise, registry};

    let mut config: crate::domain::ScenarioConfig =
        serde_json::from_str(include_str!("../../../../../scenarios/railway.json"))?;
    config.entrepreneurship.interested_per_mille = 0;
    let population = config.population.as_mut().expect("explicit population");
    population.households[0].cash = 50_000;
    population.households[0].inventories[Good::Tools as usize - 1] = 2;
    population.households[0].inventories[Good::Wood as usize - 1] = 20;
    for firm in &mut population.firms {
        if firm.good == Good::Railway {
            for good in [Good::Coal, Good::Steel, Good::Locomotives] {
                firm.inventories[good as usize - 1] = 100;
            }
        }
    }
    let scenario = Scenario::new(config)?;
    let device = crate::simulation::tests::device(64)?;
    let mut reused = Simulation::new(scenario.clone(), device.clone())?;
    let mut rebuilt = Simulation::new(scenario.clone(), device.clone())?;
    reused.advance()?;
    rebuilt.advance()?;
    assert_eq!(reused.snapshot, rebuilt.snapshot);
    let count = reused.metadata.actors.len();
    let mut requests = vec![0; count];
    requests[0] = Good::Bread as u32;
    let pending = registry::records::PendingRecords::from_dense(&requests, &vec![0; count])?;
    let entries = registry::entries::EntryRequests::new(&scenario, count, &pending.records)
        .expect("firm entry");
    let plan = registry::RegistryPlan::new(
        &scenario,
        &reused.metadata,
        &reused.migrations,
        &entries.transitions,
    )?
    .expect("entry transition");
    let buffers = reused
        .config
        .railway
        .freight
        .graph_buffers()
        .map(Clone::clone);
    reused.commit_entries(&entries, plan, &pending.records, RegistryAudit::Full)?;
    for (before, after) in buffers
        .iter()
        .zip(reused.config.railway.freight.graph_buffers())
    {
        assert!(crate::simulation::market_graphs::same_buffer(before, after));
    }
    reused.copy_audited(crate::simulation::CopyDirection::Save)?;
    let mut expected = rebuilt.download()?;
    enterprise::enter(&mut expected, &scenario, &requests)?;
    rebuilt.replace(scenario.clone(), expected)?;
    assert_eq!(reused.download()?, rebuilt.download()?);
    assert_eq!(reused.snapshot, rebuilt.snapshot);
    for _ in 0..3 {
        reused.advance()?;
        rebuilt.advance()?;
        assert_eq!(reused.snapshot, rebuilt.snapshot);
        assert_eq!(reused.download()?, rebuilt.download()?);
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum FitMethod {
    Tracked,
    Exhaustive,
    Unconstrained,
    Bounded,
}

#[derive(CubeType, CubeLaunch)]
struct FitCheckInput {
    freight: Input,
    quantity_low: u32,
    quantity_high: u32,
    workers: u32,
    #[cube(comptime)]
    method: FitMethod,
}

struct FitCheck;
struct CheckRouteFit;

impl System for FitCheck {
    type Input = FitCheckInput;
    type Passes = gpu_passes![CheckRouteFit];
}

#[cube]
impl GpuPass<FitCheckInput> for CheckRouteFit {
    type Output = ();

    fn execute(mut input: FitCheckInput) {
        let worker = ABSOLUTE_POS as u32;
        if worker < input.workers {
            let seller = input.freight.config.actors - 9;
            let path = candidate(&input.freight.actors, &input.freight.config, 0, seller, 0);
            let quantity = ticks(input.quantity_low, input.quantity_high);
            let allowed = match comptime!(input.method) {
                FitMethod::Tracked => fits_tracked(&input.freight, &path, quantity, seller),
                FitMethod::Exhaustive => fits_exhaustive(&input.freight, &path, quantity, seller),
                FitMethod::Bounded => fits(&input.freight, &path, quantity, seller),
                FitMethod::Unconstrained => {
                    fits_unconstrained(&input.freight, &path, quantity, seller)
                }
            };
            input.freight.bookings[(4 * worker) as usize] = u32::cast_from(allowed);
            let mut cursor = RegionCursor::new(&path);
            let mut record = cursor.next(&input.freight.config);
            let mut count = 0u32;
            let mut multiplicity = 0u32;
            while record < input.freight.config.railway.region_count {
                count += 1;
                multiplicity = max(multiplicity, visits(&input.freight.config, &path, record));
                record = cursor.next(&input.freight.config);
            }
            input.freight.bookings[(4 * worker + 1) as usize] = count;
            let nodes = input.freight.config.railway.nodes;
            let base = ((path.carrier * nodes + path.origin) * nodes + path.destination) as usize
                * ROUTE_WORDS;
            input.freight.bookings[(4 * worker + 2) as usize] =
                input.freight.config.railway.routes[base + 4];
            input.freight.bookings[(4 * worker + 3) as usize] = multiplicity;
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum FitRoute {
    Short,
    Repeated,
    ManyNodes,
    Property,
    Long,
}

impl FitRoute {
    fn scenario(self) -> Result<Scenario> {
        use crate::domain::Position;
        let mut config: crate::domain::ScenarioConfig =
            serde_json::from_str(include_str!("../../../../../scenarios/railway.json"))?;
        config.railway.length_unit = match self {
            Self::Short => 1000,
            Self::Repeated | Self::ManyNodes | Self::Property => 200,
            Self::Long => 20,
        };
        config.wage = 100;
        if matches!(self, Self::Repeated | Self::ManyNodes | Self::Property) {
            config.railway.segments[0].to = Position { x: 400, y: 400 };
            config.railway.segments[1].company = config.railway.companies[0].firm;
            config.railway.segments[1].from = Position { x: 900, y: 0 };
            config.railway.segments[1].to = Position { x: 400, y: 400 };
        }
        if let Self::Property = self {
            let xs = [0, 150, 550, 1000];
            let ys = [0, 250, 600, 1000];
            config.property = Some(crate::domain::property::PropertyConfig {
                regions: (0..3)
                    .flat_map(|x| {
                        (0..3).map(move |y| crate::domain::property::PropertyRegion {
                            lower: Position { x: xs[x], y: ys[y] },
                            upper: Position {
                                x: xs[x + 1],
                                y: ys[y + 1],
                            },
                            plots: 1_000_000,
                            landowner: 1,
                        })
                    })
                    .collect(),
                buildings: Vec::new(),
                orders: Vec::new(),
            });
        }
        if let Self::ManyNodes = self {
            let company = config.railway.companies[1].firm;
            config.railway.segments.extend((0..32).map(|index| {
                crate::domain::railway::RailSegment {
                    company,
                    from: Position {
                        x: 100 + 2 * index,
                        y: 900,
                    },
                    to: Position {
                        x: 101 + 2 * index,
                        y: 900,
                    },
                }
            }));
        }
        Scenario::new(config)
    }
}

#[derive(Clone, Copy, Debug)]
struct FitBudget {
    quantity: u64,
    regional_budget: u64,
    sender_budget: u64,
}

struct FitFixture {
    simulation: Simulation,
    parameters: crate::simulation::parameters::EconomyParameters,
    output: burn_wgpu::CubeTensor,
    workers: u32,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum GeometryCheck {
    Segments,
    Routes,
}

#[derive(CubeType, CubeLaunch)]
struct GeometryInput {
    config: EconomyConfig,
    failures: Box<[Atomic<u32>]>,
    #[cube(comptime)]
    check: GeometryCheck,
}

struct RouteGeometry;
struct CheckRouteGeometry;

impl System for RouteGeometry {
    type Input = GeometryInput;
    type Passes = gpu_passes![CheckRouteGeometry];
}

#[cube]
impl GpuPass<GeometryInput> for CheckRouteGeometry {
    type Output = ();

    fn execute(input: GeometryInput) {
        let index = ABSOLUTE_POS as u32;
        let config = &input.config;
        match comptime!(input.check) {
            GeometryCheck::Segments => {
                if index < config.railway.segment_count {
                    let base = index as usize * SEGMENT_WORDS;
                    let owner = config.railway.segments[base + 2];
                    let start = config.railway.segments[base + 5];
                    let count = config.railway.segments[base + 6];
                    for position in 0..count {
                        let record = config.railway.segment_regions[(start + position) as usize];
                        if config.railway.regions[record as usize * REGION_WORDS] != owner {
                            input.failures[0usize].fetch_add(1);
                        }
                        for earlier in 0..position {
                            if config.railway.segment_regions[(start + earlier) as usize] == record
                            {
                                input.failures[0usize].fetch_add(1);
                            }
                        }
                    }
                }
            }
            GeometryCheck::Routes => {
                let nodes = config.railway.nodes;
                if index < config.railway.company_count * nodes * nodes {
                    let base = index as usize * ROUTE_WORDS;
                    if config.railway.routes[base + 2] < config.railway.company_count {
                        let path = FreightRoute {
                            days: 0,
                            fee: config.railway.routes[base + 3],
                            origin: index / nodes % nodes,
                            destination: index % nodes,
                            carrier: index / (nodes * nodes),
                            foreign_distance: config.railway.routes[base + 5],
                        };
                        let mut cursor = RegionCursor::new(&path);
                        let mut record = cursor.next(config);
                        let mut count = 0u32;
                        let mut multiplicity = 0u32;
                        while record < config.railway.region_count {
                            count += 1;
                            multiplicity = max(multiplicity, visits(config, &path, record));
                            record = cursor.next(config);
                        }
                        let bound = config.railway.routes[base + 4];
                        let traversal = traverse(config, &path);
                        if bound == 0
                            || bound < multiplicity
                            || bound > max(count, 1)
                            || traversal.visits != count
                            || traversal.segments < multiplicity
                        {
                            input.failures[0usize].fetch_add(1);
                        }
                    }
                }
            }
        }
    }
}

impl FitFixture {
    fn new(route: FitRoute, workgroup: u32, workers: u32) -> Result<Self> {
        use crate::simulation::encode;
        let scenario = route.scenario()?;
        let device = crate::simulation::tests::device(workgroup)?;
        let simulation = Simulation::new(scenario.clone(), device.clone())?;
        let mut state = scenario.initial()?;
        for company in &scenario.config().railway.companies {
            let firm = state
                .actors
                .firm_indices()
                .find(|&index| state.actors.at(index).identity == company.firm)
                .expect("railway operator");
            let built = scenario
                .config()
                .railway
                .segments
                .iter()
                .filter(|segment| segment.company == company.firm)
                .count() as u64;
            state.set(Field::RailBuilt, firm, built);
            state.set(Field::RailOperating, firm, built);
        }
        device.write(
            &simulation.state,
            bytemuck::cast_slice(&encode(&state.values)),
        );
        for stage in [RailStage::Access, RailStage::Routes, RailStage::Quotes] {
            simulation.railways(stage);
        }
        let parameters = simulation.parameters(
            Phase::ProvisionPurchase,
            Market::Input(Good::Bread),
            0,
            state.actors.len(),
        )?;
        Ok(Self {
            simulation,
            parameters,
            output: device.empty(workers as usize * 4),
            workers,
        })
    }

    fn budgets(&self, case: FitBudget) {
        let simulation = &self.simulation;
        let device = &simulation.device;
        let buffers = &simulation.config.railway.freight;
        let mut resources = vec![0; buffers.resources.meta.num_elements()];
        for resource in resources.chunks_exact_mut(RESOURCE_WORDS) {
            resource[0] = case.regional_budget as u32;
            resource[1] = (case.regional_budget >> 32) as u32;
        }
        let seller = simulation.config.actors as usize - 9;
        resources[seller * RESOURCE_WORDS] = case.sender_budget as u32;
        resources[seller * RESOURCE_WORDS + 1] = (case.sender_budget >> 32) as u32;
        device.write(&buffers.resources, &resources);
        device.write(
            &simulation.control,
            &vec![0; simulation.control.meta.num_elements()],
        );
    }

    fn launch(&self, quantity: u64, method: FitMethod) {
        let simulation = &self.simulation;
        let device = &simulation.device;
        let buffers = &simulation.config.railway.freight;
        FitCheck::launch_elements(
            device,
            self.workers as usize,
            FitCheckInputLaunch {
                freight: InputLaunch {
                    state: simulation.state.buffer_arg(),
                    actors: simulation.actors.buffer_arg(),
                    spatial: simulation.spatial.buffer_arg(),
                    config: simulation.config.launch(),
                    proposals: simulation.proposals.buffer_arg(),
                    scratch: simulation.scratch.buffer_arg(),
                    control: simulation.control.buffer_arg(),
                    resources: buffers.resources.buffer_arg(),
                    accounts: buffers.accounts.buffer_arg(),
                    bookings: self.output.buffer_arg(),
                    queue: buffers.queue.buffer_arg(),
                    claims: buffers.claims.buffer_arg(),
                    claim_source: claims::settlement_source(buffers),
                    claim_offsets: claims::offsets_start(buffers.claim_capacity),
                    progress: buffers.progress.buffer_arg(),
                    parameters: self.parameters.launch(),
                    stage: Stage::Propose,
                    execution: Execution::Immediate(Wave::Initial),
                },
                quantity_low: quantity as u32,
                quantity_high: (quantity >> 32) as u32,
                workers: self.workers,
                method,
            },
        );
    }
}

fn check_fit_route(route: FitRoute) -> Result<()> {
    for workgroup in [32, 64, 256] {
        let fixture = FitFixture::new(route, workgroup, 1)?;
        for case in [
            FitBudget {
                quantity: 0,
                regional_budget: 0,
                sender_budget: 0,
            },
            FitBudget {
                quantity: 1,
                regional_budget: u64::MAX,
                sender_budget: u64::MAX,
            },
            FitBudget {
                quantity: 1,
                regional_budget: 1,
                sender_budget: u64::MAX,
            },
            FitBudget {
                quantity: 1,
                regional_budget: 2,
                sender_budget: u64::MAX,
            },
            FitBudget {
                quantity: 2,
                regional_budget: 1,
                sender_budget: u64::MAX,
            },
            FitBudget {
                quantity: 2,
                regional_budget: 3,
                sender_budget: u64::MAX,
            },
            FitBudget {
                quantity: 2,
                regional_budget: 6,
                sender_budget: u64::MAX,
            },
            FitBudget {
                quantity: 1,
                regional_budget: 8,
                sender_budget: u64::MAX,
            },
            FitBudget {
                quantity: 2,
                regional_budget: 16,
                sender_budget: u64::MAX,
            },
            FitBudget {
                quantity: u64::from(u32::MAX),
                regional_budget: u64::MAX,
                sender_budget: u64::MAX,
            },
            FitBudget {
                quantity: 1u64 << 32,
                regional_budget: u64::MAX,
                sender_budget: u64::MAX,
            },
            FitBudget {
                quantity: u64::MAX,
                regional_budget: u64::MAX,
                sender_budget: u64::MAX,
            },
            FitBudget {
                quantity: u64::MAX,
                regional_budget: 0,
                sender_budget: 0,
            },
        ] {
            let mut observations = Vec::new();
            for method in [
                FitMethod::Tracked,
                FitMethod::Exhaustive,
                FitMethod::Unconstrained,
                FitMethod::Bounded,
            ] {
                fixture.budgets(case);
                fixture.launch(case.quantity, method);
                let device = &fixture.simulation.device;
                let mut values = device.read_range(&fixture.output, 0, 4)?;
                match route {
                    FitRoute::Long => assert!(values[1] > 16),
                    FitRoute::Repeated | FitRoute::ManyNodes => assert_eq!(values[1], 8),
                    FitRoute::Property => assert_eq!(values[1], 4),
                    FitRoute::Short => assert_eq!(values[1], 1),
                }
                assert!(values[2] >= values[3]);
                assert!(values[2] <= values[1].max(1));
                match route {
                    FitRoute::Short => assert_eq!(values[2], 1),
                    _ => assert_eq!(values[2], 2),
                }
                values.push(
                    device.read(&fixture.simulation.control)?[crate::simulation::flags::FAULT_WORD],
                );
                observations.push(values);
            }
            assert_eq!(
                observations[0], observations[1],
                "{route:?}, workgroup {workgroup}, {case:?}"
            );
            assert_eq!(
                observations[0], observations[2],
                "{route:?}, workgroup {workgroup}, {case:?}"
            );
            assert_eq!(
                observations[0], observations[3],
                "bounded {route:?}, workgroup {workgroup}, {case:?}"
            );
        }
    }
    Ok(())
}

#[test]
fn monotone_segment_regions_bound_every_route_resource_multiplicity_on_gpu() -> Result<()> {
    for route in [
        FitRoute::Short,
        FitRoute::Repeated,
        FitRoute::ManyNodes,
        FitRoute::Property,
        FitRoute::Long,
    ] {
        let fixture = FitFixture::new(route, 64, 1)?;
        let simulation = &fixture.simulation;
        let device = &simulation.device;
        let failures = device.tensor(&[0]);
        for check in [GeometryCheck::Segments, GeometryCheck::Routes] {
            let count = match check {
                GeometryCheck::Segments => simulation.config.railway.segment_count,
                GeometryCheck::Routes => {
                    simulation.config.railway.company_count * simulation.config.railway.nodes.pow(2)
                }
            };
            RouteGeometry::launch_elements(
                device,
                count as usize,
                GeometryInputLaunch {
                    config: simulation.config.launch(),
                    failures: failures.buffer_arg(),
                    check,
                },
            );
            assert_eq!(device.read(&failures)?, [0], "{route:?}, {check:?}");
        }
    }
    Ok(())
}

#[test]
fn freight_fit_methods_agree_across_routes_and_wide_faults_on_gpu() -> Result<()> {
    [
        FitRoute::Short,
        FitRoute::Repeated,
        FitRoute::ManyNodes,
        FitRoute::Property,
        FitRoute::Long,
    ]
    .into_iter()
    .try_for_each(check_fit_route)
}

#[test]
fn route_visit_bounds_follow_cached_service_and_preserve_zero_region_fits_on_gpu() -> Result<()> {
    let mut fixture = FitFixture::new(FitRoute::Repeated, 64, 1)?;
    let simulation = &fixture.simulation;
    let device = &simulation.device;
    let access = device.read(&simulation.config.railway.access)?;
    let seller = simulation.config.actors as usize - 9;
    let nodes = simulation.config.railway.nodes as usize;
    let origin = access[2 * seller] as usize;
    let destination = access[0] as usize;
    let base = (origin * nodes + destination) * ROUTE_WORDS;
    let original = device.read(&simulation.config.railway.routes)?;
    assert_eq!(original[base + 4], 2);
    for _ in 0..2 {
        simulation.railways(RailStage::Routes);
        assert_eq!(device.read(&simulation.config.railway.routes)?, original);
    }
    let companies = device.read(&simulation.config.railway.companies)?;
    let operator = companies[0] as usize;
    let operating =
        (Field::RailOperating as usize * simulation.config.state_stride as usize + operator) * 2;
    let state = device.read(&simulation.state)?;
    let mut suspended = state.clone();
    suspended[operating] = 0;
    suspended[operating + 1] = 0;
    device.write(&simulation.state, &suspended);
    simulation.railways(RailStage::Routes);
    let stopped = device.read(&simulation.config.railway.routes)?;
    assert_eq!(stopped[base + 2], u32::MAX);
    assert_eq!(stopped[base + 4], 0);
    device.write(&simulation.state, &state);
    simulation.railways(RailStage::Routes);
    simulation.railways(RailStage::Quotes);
    assert_eq!(device.read(&simulation.config.railway.routes)?, original);
    fixture.simulation.config.railway.region_count = 0;
    let simulation = &fixture.simulation;
    let device = &simulation.device;
    let mut companies = device.read(&simulation.config.railway.companies)?;
    companies[simulation.config.railway.company_count as usize * COMPANY_WORDS] =
        RouteCacheState::Unknown as u32;
    device.write(&simulation.config.railway.companies, &companies);
    simulation.railways(RailStage::Routes);
    simulation.railways(RailStage::Quotes);
    assert_eq!(device.read(&simulation.config.railway.routes)?[base + 4], 1);
    for quantity in [0, 1, u64::from(u32::MAX), 1u64 << 32, u64::MAX] {
        let mut observations = Vec::new();
        for method in [FitMethod::Tracked, FitMethod::Bounded] {
            fixture.budgets(FitBudget {
                quantity,
                regional_budget: 0,
                sender_budget: u64::MAX,
            });
            fixture.launch(quantity, method);
            let mut values = device.read_range(&fixture.output, 0, 4)?;
            assert_eq!(values[1..], [0, 1, 0]);
            values.push(device.read(&simulation.control)?[crate::simulation::flags::FAULT_WORD]);
            observations.push(values);
        }
        assert_eq!(observations[0], observations[1], "quantity {quantity}");
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
enum Seller {
    Firm(Good),
    Household,
}

#[derive(Clone, Copy, Debug)]
enum FreightLoad {
    Available,
    Unavailable,
    Sparse,
    Constrained,
    SparseConstrained,
    CashConstrained,
    Cascading,
    Empty,
}

impl FreightLoad {
    fn accepts(self, buyer: usize) -> bool {
        match self {
            Self::Available
            | Self::Unavailable
            | Self::Constrained
            | Self::CashConstrained
            | Self::Cascading => true,
            Self::Sparse | Self::SparseConstrained => buyer % 3 == 0,
            Self::Empty => false,
        }
    }

    fn capacity(self) -> u64 {
        match self {
            Self::Unavailable => 0,
            Self::Constrained | Self::SparseConstrained => 7,
            Self::Cascading => 5,
            _ => 1000,
        }
    }

    fn budget(self) -> u64 {
        match self {
            Self::CashConstrained => 49,
            Self::Cascading => 7,
            _ => 10_000,
        }
    }

    fn sender(self, buyer: usize, first: usize) -> usize {
        first
            + match self {
                Self::Cascading => match buyer / 5 {
                    0 => 0,
                    1 => 3,
                    _ => 4,
                },
                _ => 0,
            }
    }
}

fn freight_execution(
    source: Seller,
    load: FreightLoad,
    buyers: usize,
    workgroup: u32,
    execution: Execution,
) -> Result<Vec<u32>> {
    use crate::domain::{HouseholdSeed, Position, ScenarioConfig};
    use crate::simulation::{Market, decode, encode};
    let mut config: ScenarioConfig =
        serde_json::from_str(include_str!("../../../../../scenarios/railway.json"))?;
    config.railway.units_per_level = load.capacity().max(1) as u32;
    let population = config.population.as_mut().unwrap();
    let template = population.households[0].clone();
    population
        .households
        .extend(
            (population.households.len()..buyers).map(|index| HouseholdSeed {
                identity: 100 + index as u64,
                ..template.clone()
            }),
        );
    if let FreightLoad::Cascading = load {
        population.households.truncate(15);
    }
    for household in &mut population.households {
        household.position = Position { x: 900, y: 0 };
    }
    config.households = population.households.len() as u32;
    let buyers = config.households as usize;
    let population = config.population.as_mut().unwrap();
    let good = match source {
        Seller::Firm(good) => {
            population.firms[0].good = good;
            good
        }
        Seller::Household => {
            population.households[buyers - 1].position = population.firms[0].position;
            Good::Bread
        }
    };
    let seller = match source {
        Seller::Firm(_) => buyers,
        Seller::Household => buyers - 1,
    };
    let shipments = buyers.min(seller);
    let scenario = Scenario::new(config)?;
    let device = crate::simulation::tests::device(workgroup)?;
    let mut simulation = Simulation::new(scenario.clone(), device.clone())?;
    let capacity = simulation
        .config
        .railway
        .freight
        .accounts
        .meta
        .num_elements()
        / ACCOUNT_WORDS;
    claim_work(&device, &mut simulation.config, (2 * capacity) as u64)?;
    let mut state = scenario.initial()?;
    state.set(Field::Cash, seller, load.budget());
    if let FreightLoad::Cascading = load {
        state.set(Field::Cash, seller + 3, load.budget());
        state.set(Field::Cash, seller + 4, load.budget());
    }
    let count = state.actors.len();
    for firm in [buyers + 1, buyers + 2] {
        state.set(Field::RailBuilt, firm, 1);
        state.set(Field::RailOperating, firm, 1);
        state.set(Field::Railway, firm, load.capacity());
    }
    for region in 0..scenario.rail_geography().regions.len() {
        for (field, amount) in [
            (RailCapacityField::Level, 1),
            (RailCapacityField::Available, load.capacity()),
        ] {
            let coordinate = scenario
                .rail_geography()
                .coordinate(&scenario, &state, field, region);
            state.values[coordinate] = amount;
        }
    }
    let mut parameters =
        simulation.parameters(Phase::ProvisionPurchase, Market::Input(good), 0, count)?;
    parameters.discovery = Discovery::Indexed;
    parameters.sort_count = shipments as u32;
    let mut proposals = vec![NO_ACTOR; count * ProposalLayout::WORDS];
    let mut scratch = vec![0; markets::scratch_words(count)];
    let mut control = vec![0; CONTROL_WORDS + 4];
    control[CONSTRUCTED_EVENTS_WORD] = shipments as u32;
    let accepted = (0..shipments)
        .filter(|&buyer| load.accepts(buyer))
        .collect::<Vec<_>>();
    let delivered_buyers = match load {
        FreightLoad::Cascading => vec![0, 5],
        _ => accepted
            .iter()
            .copied()
            .take((load.capacity() as usize).min(load.budget() as usize / 7))
            .collect(),
    };
    let delivered = delivered_buyers.len();
    for buyer in 0..shipments {
        let delivery = state.cartage_coordinate(good, 9, scenario.ring(), buyer);
        let amount = u64::from(load.accepts(buyer));
        state.values[delivery.0 as usize] = amount;
        state.set(Field::RoadFreight, buyer, amount);
        state.set(Field::CartagePayable, buyer, amount * 27);
        let sender = load.sender(buyer, seller);
        proposals[buyer * ProposalLayout::WORDS + ProposalLayout::ACTOR] = sender as u32;
        proposals[buyer * ProposalLayout::WORDS + ProposalLayout::QUANTITY_LOW] = 1;
        proposals[buyer * ProposalLayout::WORDS + ProposalLayout::QUANTITY_HIGH] = 0;
        scratch[buyer] = sender as u32;
        scratch[count + buyer] = buyer as u32;
        scratch[4 * count + buyer] = match load.accepts(buyer) {
            true => EventStatus::Accepted as u32,
            false => EventStatus::Rejected as u32,
        };
    }
    device.write(
        &simulation.state,
        bytemuck::cast_slice(&encode(&state.values)),
    );
    device.write(&simulation.proposals, &proposals);
    device.write(&simulation.scratch, &scratch);
    device.write(&simulation.control, &control);
    for stage in [RailStage::Access, RailStage::Routes, RailStage::Quotes] {
        simulation.railways(stage);
    }
    device.write(
        &simulation.config.railway.freight.resources,
        &vec![
            0;
            simulation
                .config
                .railway
                .freight
                .resources
                .meta
                .num_elements()
        ],
    );
    let wave = std::cell::Cell::new(execution);
    let advance = |stage| {
        launch_buffered(
            &device,
            &simulation.state,
            &simulation.actors,
            &simulation.spatial,
            &simulation.config,
            &simulation.proposals,
            &simulation.scratch,
            &simulation.control,
            &parameters,
            stage,
            wave.get(),
        );
    };
    advance(Stage::Snapshot);
    advance(Stage::Initialize);
    let buffers = &simulation.config.railway.freight;
    assert_eq!(
        device.read(&buffers.progress)?[Wave::Initial as usize] as usize,
        accepted.len()
    );
    let mut pending = 1;
    let mut waves = 0;
    while pending > 0 {
        wave.set(match (execution, waves) {
            (Execution::Immediate(_), 0) => Execution::Immediate(Wave::Initial),
            (Execution::Immediate(_), _) => Execution::Immediate(Wave::Retry),
            (_, 0) => Execution::Buffered(Wave::Initial),
            _ => Execution::Buffered(Wave::Retry),
        });
        advance(Stage::Queue);
        if waves > 0 {
            advance(Stage::Reset);
        }
        let bookings = device.read(&buffers.bookings)?;
        let resources = device.read(&buffers.resources)?;
        let previous_control = device.read(&simulation.control)?;
        advance(Stage::Propose);
        let expected_bookings = device.read(&buffers.bookings)?;
        let expected_resources = device.read(&buffers.resources)?;
        let expected_control = device.read(&simulation.control)?;
        let mut fresh = bookings;
        for booking in fresh
            .chunks_exact_mut(FREIGHT_BOOKING_WORDS)
            .take(shipments)
        {
            if booking[0] == BookingState::Retrying as u32 {
                booking[0] = BookingState::Pending as u32;
            }
        }
        device.write(&buffers.bookings, &fresh);
        device.write(&buffers.resources, &resources);
        device.write(&simulation.control, &previous_control);
        advance(Stage::Propose);
        assert_eq!(device.read(&buffers.bookings)?, expected_bookings);
        assert_eq!(device.read(&buffers.resources)?, expected_resources);
        assert_eq!(device.read(&simulation.control)?, expected_control);
        match waves {
            0 => {
                let active =
                    device.read_range(&buffers.progress, Wave::Initial as usize, 1)?[0] as usize;
                let queued = device.read_range(&buffers.queue, 0, active)?;
                let buyers = device.read_range(
                    &simulation.scratch,
                    parameters.sort_source as usize + count,
                    count,
                )?;
                for priority in queued {
                    let buyer = (buyers[priority as usize] + parameters.offset) as usize;
                    let booking = buyer * FREIGHT_BOOKING_WORDS;
                    assert_eq!(fresh[booking], BookingState::Pending as u32);
                    fresh[booking] = BookingState::Ready as u32;
                }
                device.write(&buffers.bookings, &fresh);
                device.write(&buffers.resources, &resources);
                device.write(&simulation.control, &previous_control);
                advance(Stage::Propose);
                assert_eq!(device.read(&buffers.bookings)?, expected_bookings);
                assert_eq!(device.read(&buffers.resources)?, expected_resources);
                assert_eq!(device.read(&simulation.control)?, expected_control);
            }
            _ => (),
        }
        for stage in [
            Stage::CountClaims,
            Stage::AdmitClaims,
            Stage::EmitClaims,
            Stage::SortClaims,
            Stage::ClaimOffsets,
            Stage::Resolve,
        ] {
            advance(stage);
        }
        assert_eq!(device.read(&buffers.resources)?, expected_resources);
        let counts = device.read(&buffers.progress)?;
        let claims = counts[CLAIM_COUNT];
        assert_eq!(counts[CLAIM_CURSOR], claims);
        match load {
            FreightLoad::Available
            | FreightLoad::Unavailable
            | FreightLoad::Sparse
            | FreightLoad::Empty => {
                assert_eq!(claims, 0)
            }
            _ => (),
        }
        advance(Stage::Commit);
        let committed = device.read(&buffers.resources)?;
        for (resource_id, resource) in committed
            .chunks_exact(RESOURCE_WORDS)
            .take(count + scenario.rail_geography().regions.len())
            .enumerate()
        {
            let ticks =
                |offset| u64::from(resource[offset]) | (u64::from(resource[offset + 1]) << 32);
            assert!(
                ticks(CONSUMED) <= ticks(0),
                "source {source:?}, load {load:?}, execution {execution:?}, wave {waves}, resource {resource_id}, consumed {}, budget {}, claims {claims}",
                ticks(CONSUMED),
                ticks(0),
            );
        }
        advance(Stage::Release);
        pending = match wave.get().wave() {
            Wave::Initial => device.read(&buffers.progress)?[Wave::Retry as usize],
            Wave::Retry => 0,
        };
        let bookings = device.read(&buffers.bookings)?;
        if waves == 0 {
            let first_wave = match load {
                FreightLoad::Cascading => 1,
                FreightLoad::Unavailable => accepted.len(),
                _ => delivered,
            };
            assert_eq!(pending as usize, accepted.len() - first_wave);
        }
        let mut queued = device.read_range(&buffers.queue, count, pending as usize)?;
        queued.sort_unstable();
        let unresolved = (0..shipments)
            .filter(|&buyer| {
                bookings[buyer * FREIGHT_BOOKING_WORDS] == BookingState::Retrying as u32
            })
            .map(|buyer| buyer as u32)
            .collect::<Vec<_>>();
        assert_eq!(queued, unresolved);
        waves += 1;
        assert!(waves <= 2);
    }
    let settled = device.read(&simulation.state)?;
    let accounts = device.read(&buffers.accounts)?;
    for _ in 0..8 {
        for stage in [
            Stage::Queue,
            Stage::Reset,
            Stage::Propose,
            Stage::CountClaims,
            Stage::AdmitClaims,
            Stage::EmitClaims,
            Stage::SortClaims,
            Stage::ClaimOffsets,
            Stage::Resolve,
            Stage::Commit,
            Stage::Release,
        ] {
            advance(stage);
        }
    }
    assert_eq!(device.read(&simulation.state)?, settled);
    assert_eq!(device.read(&buffers.accounts)?, accounts);
    assert_eq!(
        device.read(&buffers.progress)?[Wave::Initial as usize] as usize,
        accepted.len()
    );
    wave.set(execution);
    advance(Stage::Accounts);
    match load {
        FreightLoad::Constrained
        | FreightLoad::SparseConstrained
        | FreightLoad::CashConstrained
        | FreightLoad::Cascading => assert_eq!(waves, 2),
        _ => assert_eq!(waves, 1),
    }
    assert_eq!(
        device.read(&simulation.control)?[crate::simulation::flags::FAULT_WORD],
        0
    );
    let mut actual = state.clone();
    actual.values = decode(&device.read(&simulation.state)?);
    assert_eq!(
        actual.get(Field::Cash, seller),
        state.get(Field::Cash, seller)
            - delivered_buyers
                .iter()
                .filter(|&&buyer| load.sender(buyer, seller) == seller)
                .count() as u64
                * 7
    );
    assert_eq!(
        actual.get(Field::RailRevenue, buyers + 1),
        delivered as u64 * 2
    );
    assert_eq!(
        actual.get(Field::RailRevenue, buyers + 2),
        delivered as u64 * 5
    );
    assert_eq!(
        actual.get(Field::Railway, buyers + 2),
        load.capacity() - delivered as u64
    );
    for buyer in 0..shipments {
        let delivery = actual.shipment_coordinate(
            good,
            (parameters.day as usize + 1) % scenario.ring(),
            scenario.ring(),
            buyer,
        );
        assert_eq!(
            actual.values[delivery.0 as usize],
            u64::from(delivered_buyers.contains(&buyer))
        );
        let animal = actual.cartage_coordinate(good, 9, scenario.ring(), buyer);
        let waiting = u64::from(load.accepts(buyer) && !delivered_buyers.contains(&buyer));
        assert_eq!(actual.get(Field::RoadFreight, buyer), waiting);
        assert_eq!(actual.get(Field::CartagePayable, buyer), waiting * 27);
        assert_eq!(actual.values[animal.0 as usize], waiting);
    }
    device.read(&simulation.state)
}

#[test]
fn buffered_freight_matches_immediate_settlement_and_retries_on_gpu() -> Result<()> {
    for load in [
        FreightLoad::Available,
        FreightLoad::Unavailable,
        FreightLoad::Sparse,
        FreightLoad::Constrained,
        FreightLoad::SparseConstrained,
        FreightLoad::CashConstrained,
        FreightLoad::Cascading,
        FreightLoad::Empty,
    ] {
        for (source, workgroup) in [
            (Seller::Firm(Good::Bread), 32),
            (Seller::Firm(Good::Steel), 64),
            (Seller::Household, 256),
        ] {
            assert_eq!(
                freight_execution(
                    source,
                    load,
                    389,
                    workgroup,
                    Execution::Buffered(Wave::Initial)
                )?,
                freight_execution(
                    source,
                    load,
                    389,
                    workgroup,
                    Execution::Immediate(Wave::Initial)
                )?,
                "{source:?} {load:?} workgroup {workgroup}"
            );
        }
    }
    Ok(())
}

#[test]
fn retry_queue_preserves_booking_priority_at_cube_boundaries_on_gpu() -> Result<()> {
    for buyers in [63, 64, 65, 70, 71, 72] {
        freight_execution(
            Seller::Firm(Good::Bread),
            FreightLoad::Constrained,
            buyers,
            64,
            Execution::Buffered(Wave::Initial),
        )?;
    }
    Ok(())
}

#[test]
fn market_claim_preparation_reads_gpu_searches_and_propagates_prior_faults() -> Result<()> {
    let scenario = Scenario::new(serde_json::from_str(include_str!(
        "../../../../../scenarios/railway.json"
    ))?)?;
    let device = crate::simulation::tests::device(64)?;
    let mut simulation = Simulation::new(scenario, device.clone())?;
    let mut control = vec![0; CONTROL_WORDS + 4];
    control[markets::SEARCHES_WORD] = 17;
    device.write(&simulation.control, &control);
    let preparation = prepare_market_claims(&device, &mut simulation.config, &simulation.control)?;
    assert!(matches!(preparation.claims, ClaimWork::Empty));
    assert_eq!(preparation.searches, 17);
    device.write(
        &simulation.config.railway.freight.progress,
        &[0, 0, 1, 0, 0],
    );
    control[markets::SEARCHES_WORD] = 0;
    device.write(&simulation.control, &control);
    let preparation = prepare_market_claims(&device, &mut simulation.config, &simulation.control)?;
    assert!(matches!(preparation.claims, ClaimWork::Ready));
    assert_eq!(preparation.searches, 0);
    control[crate::simulation::flags::FAULT_WORD] = FaultFlags::ARITHMETIC_OVERFLOW.bits();
    device.write(&simulation.control, &control);
    let error = prepare_market_claims(&device, &mut simulation.config, &simulation.control)
        .err()
        .expect("the prior GPU fault must propagate");
    assert!(error.to_string().contains("ARITHMETIC_OVERFLOW"));
    Ok(())
}

#[test]
fn buffered_claim_admission_preserves_zero_wide_counts_and_prior_faults_on_gpu() -> Result<()> {
    #[derive(Clone, Copy)]
    enum AdmissionDispatch {
        Empty,
        Active,
    }

    struct AdmissionCase {
        count: u64,
        faults: FaultFlags,
        dispatch: AdmissionDispatch,
    }

    let scenario = Scenario::new(serde_json::from_str(include_str!(
        "../../../../../scenarios/railway.json"
    ))?)?;
    let device = crate::simulation::tests::device(64)?;
    let mut simulation = Simulation::new(scenario, device.clone())?;
    claim_work(&device, &mut simulation.config, 17)?;
    let buffers = &simulation.config.railway.freight;
    let capacity = buffers.claim_capacity as u64;
    let parameters = simulation.parameters(
        Phase::ProvisionPurchase,
        Market::Input(Good::Bread),
        0,
        simulation.config.actors as usize,
    )?;
    for case in [
        AdmissionCase {
            count: 0,
            faults: FaultFlags::empty(),
            dispatch: AdmissionDispatch::Empty,
        },
        AdmissionCase {
            count: capacity,
            faults: FaultFlags::empty(),
            dispatch: AdmissionDispatch::Active,
        },
        AdmissionCase {
            count: capacity + 1,
            faults: FaultFlags::empty(),
            dispatch: AdmissionDispatch::Empty,
        },
        AdmissionCase {
            count: u64::from(u32::MAX) + 17,
            faults: FaultFlags::empty(),
            dispatch: AdmissionDispatch::Empty,
        },
        AdmissionCase {
            count: capacity + 1,
            faults: FaultFlags::ARITHMETIC_OVERFLOW,
            dispatch: AdmissionDispatch::Empty,
        },
        AdmissionCase {
            count: capacity + 1,
            faults: FaultFlags::ARITHMETIC_OVERFLOW | FaultFlags::CLAIM_CAPACITY,
            dispatch: AdmissionDispatch::Empty,
        },
    ] {
        let mut control = vec![0; crate::simulation::flags::CLAIM_CAPACITY_HIGH_WORD + 1];
        control[crate::simulation::flags::FAULT_WORD] = case.faults.bits();
        let progress = [7, 11, case.count as u32, (case.count >> 32) as u32, 13];
        let mut dispatches = vec![37; FREIGHT_DISPATCH_WORDS];
        dispatches[..3].copy_from_slice(&[3, 2, 1]);
        dispatches[BOOKING_DISPATCH..BOOKING_DISPATCH + 3].copy_from_slice(&[5, 7, 1]);
        device.write(&simulation.control, &control);
        device.write(&buffers.progress, &progress);
        device.write(&buffers.dispatches, &dispatches);
        launch_buffered(
            &device,
            &simulation.state,
            &simulation.actors,
            &simulation.spatial,
            &simulation.config,
            &simulation.proposals,
            &simulation.scratch,
            &simulation.control,
            &parameters,
            Stage::AdmitClaims,
            Execution::Buffered(Wave::Initial),
        );
        let actual_control = device.read(&simulation.control)?;
        let actual_progress = device.read(&buffers.progress)?;
        let mut expected_progress = progress;
        match case.faults.is_empty() {
            true => {
                if case.count > capacity {
                    control[crate::simulation::flags::FAULT_WORD] =
                        FaultFlags::CLAIM_CAPACITY.bits();
                    control[crate::simulation::flags::CLAIM_CAPACITY_LOW_WORD] = case.count as u32;
                    control[crate::simulation::flags::CLAIM_CAPACITY_HIGH_WORD] =
                        (case.count >> 32) as u32;
                    expected_progress[CLAIM_COUNT] = 0;
                    expected_progress[CLAIM_COUNT + 1] = 0;
                }
            }
            false => (),
        }
        assert_eq!(actual_control[..control.len()], control);
        assert_eq!(actual_progress, expected_progress);
        let active = match case.dispatch {
            AdmissionDispatch::Empty => [0, 1, 1],
            AdmissionDispatch::Active => [5, 7, 1],
        };
        let resources = match case.dispatch {
            AdmissionDispatch::Empty => [0, 1, 1],
            AdmissionDispatch::Active => [3, 2, 1],
        };
        dispatches[CLAIM_ACTIVE_DISPATCH..CLAIM_ACTIVE_DISPATCH + 3].copy_from_slice(&active);
        dispatches[CLAIM_RESOURCE_DISPATCH..CLAIM_RESOURCE_DISPATCH + 3]
            .copy_from_slice(&resources);
        let count = match case.dispatch {
            AdmissionDispatch::Active => case.count as usize,
            AdmissionDispatch::Empty => 0,
        };
        for (offset, width) in [
            (BLOCK_SORT_DISPATCH, buffers.sorting.tile()),
            (MERGE_SORT_DISPATCH, device.workgroup as usize),
        ] {
            dispatches[offset..offset + 3].copy_from_slice(&[count.div_ceil(width) as u32, 1, 1]);
        }
        assert_eq!(device.read(&buffers.dispatches)?, dispatches);
    }
    let error = simulation
        .fault_check()
        .expect_err("the genuine compute fault must prevail");
    assert!(error.to_string().contains("ARITHMETIC_OVERFLOW"));
    assert!(
        error
            .downcast_ref::<crate::simulation::ClaimCapacityDemand>()
            .is_none()
    );
    Ok(())
}

#[test]
fn disabled_buffered_freight_retry_preserves_all_settlement_buffers_on_gpu() -> Result<()> {
    let scenario = Scenario::new(serde_json::from_str(include_str!(
        "../../../../../scenarios/railway.json"
    ))?)?;
    let device = crate::simulation::tests::device(64)?;
    let mut simulation = Simulation::new(scenario, device.clone())?;
    claim_work(&device, &mut simulation.config, 17)?;
    let buffers = &simulation.config.railway.freight;
    device.write(&buffers.dispatches, &vec![0; FREIGHT_DISPATCH_WORDS]);
    let parameters = simulation.parameters(
        Phase::ProvisionPurchase,
        Market::Input(Good::Bread),
        0,
        simulation.config.actors as usize,
    )?;
    launch(
        &device,
        &simulation.state,
        &simulation.actors,
        &simulation.spatial,
        &simulation.config,
        &simulation.proposals,
        &simulation.scratch,
        &simulation.control,
        &parameters,
        Stage::Snapshot,
    );
    let initial_dispatches = device.read(&buffers.dispatches)?;
    let tensors = [
        simulation.state.clone(),
        simulation.scratch.clone(),
        simulation.proposals.clone(),
        buffers.resources.clone(),
        buffers.accounts.clone(),
        buffers.bookings.clone(),
        buffers.queue.clone(),
        buffers.claims.clone(),
    ];
    for (retry_bookings, faults) in [
        (0, FaultFlags::empty()),
        (1, FaultFlags::ARITHMETIC_OVERFLOW),
    ] {
        let mut control = vec![0; crate::simulation::flags::CLAIM_CAPACITY_HIGH_WORD + 1];
        control[crate::simulation::flags::FAULT_WORD] = faults.bits();
        let progress = [7, retry_bookings, 17, 0, 13];
        let mut stale_dispatches = initial_dispatches.clone();
        stale_dispatches[CLAIM_ACTIVE_DISPATCH..CLAIM_ACTIVE_DISPATCH + 3]
            .copy_from_slice(&[11, 13, 1]);
        stale_dispatches[CLAIM_RESOURCE_DISPATCH..CLAIM_RESOURCE_DISPATCH + 3]
            .copy_from_slice(&[17, 19, 1]);
        device.write(&simulation.control, &control);
        device.write(&buffers.progress, &progress);
        device.write(&buffers.dispatches, &stale_dispatches);
        let expected = tensors
            .iter()
            .map(|tensor| device.read(tensor))
            .collect::<Result<Vec<_>>>()?;
        for stage in [
            Stage::Queue,
            Stage::Reset,
            Stage::Propose,
            Stage::CountClaims,
            Stage::AdmitClaims,
            Stage::EmitClaims,
            Stage::SortClaims,
            Stage::ClaimOffsets,
            Stage::Resolve,
            Stage::Commit,
            Stage::Release,
            Stage::Accounts,
        ] {
            launch_buffered(
                &device,
                &simulation.state,
                &simulation.actors,
                &simulation.spatial,
                &simulation.config,
                &simulation.proposals,
                &simulation.scratch,
                &simulation.control,
                &parameters,
                stage,
                Execution::Buffered(Wave::Retry),
            );
        }
        for (tensor, expected) in tensors.iter().zip(expected) {
            assert_eq!(device.read(tensor)?, expected);
        }
        assert_eq!(device.read(&buffers.progress)?, progress);
        assert_eq!(device.read(&simulation.control)?[..control.len()], control);
        let dispatches = device.read(&buffers.dispatches)?;
        assert_eq!(dispatches[BOOKING_DISPATCH], 0);
        assert_eq!(dispatches[BLOCK_SORT_DISPATCH], 0);
        assert_eq!(
            dispatches[..BOOKING_DISPATCH],
            initial_dispatches[..BOOKING_DISPATCH]
        );
        assert_eq!(
            dispatches[RETRY_RESOURCE_DISPATCH..RETRY_RESOURCE_DISPATCH + 3],
            [0, 1, 1],
        );
        assert_eq!(
            dispatches[CLAIM_ACTIVE_DISPATCH..CLAIM_ACTIVE_DISPATCH + 3],
            [0, 1, 1],
        );
        assert_eq!(
            dispatches[CLAIM_RESOURCE_DISPATCH..CLAIM_RESOURCE_DISPATCH + 3],
            [0, 1, 1],
        );
    }
    let mut dispatches = initial_dispatches;
    dispatches[..3].copy_from_slice(&[2, 3, 1]);
    device.write(&buffers.dispatches, &dispatches);
    device.write(
        &simulation.control,
        &vec![0; simulation.control.meta.num_elements()],
    );
    device.write(&buffers.progress, &[7, 11, 17, 0, 13]);
    launch_buffered(
        &device,
        &simulation.state,
        &simulation.actors,
        &simulation.spatial,
        &simulation.config,
        &simulation.proposals,
        &simulation.scratch,
        &simulation.control,
        &parameters,
        Stage::Queue,
        Execution::Buffered(Wave::Retry),
    );
    let actual = device.read(&buffers.dispatches)?;
    assert_eq!(actual[..BOOKING_DISPATCH], dispatches[..BOOKING_DISPATCH]);
    assert_eq!(
        actual[RETRY_RESOURCE_DISPATCH..RETRY_RESOURCE_DISPATCH + 3],
        dispatches[..3],
    );
    assert_eq!(device.read(&buffers.progress)?, [7, 11, 0, 0, 0]);
    Ok(())
}

#[test]
fn buffered_advance_recovers_claim_capacity_and_preserves_registry_growth_on_gpu() -> Result<()> {
    buffered_capacity_recovery(0, ClaimContention::Absent)?;
    buffered_capacity_recovery(0, ClaimContention::Present)?;
    buffered_capacity_recovery(1, ClaimContention::Present)
}

#[derive(Clone, Copy)]
enum ClaimContention {
    Absent,
    Present,
}

fn buffered_capacity_recovery(initial_capacity: usize, contention: ClaimContention) -> Result<()> {
    use crate::domain::{HouseholdSeed, Position};
    use crate::simulation::{CopyDirection, RegistryAudit, Trace, registry};

    fn enter_bakery(simulation: &mut Simulation) -> Result<()> {
        let count = simulation.metadata.actors.len();
        let mut requests = vec![0; count];
        requests[0] = Good::Bread as u32;
        let pending = registry::records::PendingRecords::from_dense(&requests, &vec![0; count])?;
        let entries =
            registry::entries::EntryRequests::new(&simulation.scenario, count, &pending.records)
                .expect("funded bakery entry");
        let plan = registry::RegistryPlan::new(
            &simulation.scenario,
            &simulation.metadata,
            &simulation.migrations,
            &entries.transitions,
        )?
        .expect("bakery registry plan");
        simulation.commit_entries(&entries, plan, &pending.records, RegistryAudit::Full)?;
        simulation.copy_audited(CopyDirection::Save)
    }

    let mut config: crate::domain::ScenarioConfig =
        serde_json::from_str(include_str!("../../../../../scenarios/railway.json"))?;
    config.railway.units_per_level = match contention {
        ClaimContention::Absent => 100_000,
        ClaimContention::Present => 64,
    };
    config.entrepreneurship.interested_per_mille = 0;
    let population = config
        .population
        .as_mut()
        .expect("explicit railway population");
    let template = population.households[0].clone();
    population
        .households
        .extend((0..24).map(|index| HouseholdSeed {
            identity: 100 + index,
            ..template.clone()
        }));
    for household in &mut population.households {
        household.cash = 50_000;
        household.position = Position { x: 900, y: 0 };
        household.inventories[Good::Bread as usize - 1] = 0;
    }
    population.households[0].inventories[Good::Tools as usize - 1] = 10;
    population.households[0].inventories[Good::Wood as usize - 1] = 100;
    for firm in &mut population.firms {
        firm.cash = match contention {
            ClaimContention::Absent => 1_000_000_000,
            ClaimContention::Present => firm.cash,
        };
        if firm.good == Good::Railway {
            for good in [Good::Coal, Good::Steel, Good::Locomotives] {
                firm.inventories[good as usize - 1] = 100;
            }
        }
    }
    config.households = population.households.len() as u32;
    let scenario = Scenario::new(config)?;
    let device = crate::simulation::tests::device(64)?;
    let mut warmup = Simulation::new(scenario.clone(), device.clone())?;
    warmup.advance()?;
    let mut state = warmup.download()?;
    let count = state.actors.len();
    let seller = scenario.config().households as usize;
    let mut returned = 0;
    for household in 0..seller {
        let incoming =
            state.values[(TRANSIT_FIELDS + Good::Bread as usize - 1) * count + household];
        returned += state.get(Field::Bread, household) + incoming;
        state.set(Field::Bread, household, 0);
        state.set(
            Field::Transit,
            household,
            state.get(Field::Transit, household) - incoming,
        );
        state.values[(TRANSIT_FIELDS + Good::Bread as usize - 1) * count + household] = 0;
        for slot in 0..scenario.ring() {
            let coordinate =
                state.shipment_coordinate(Good::Bread, slot, scenario.ring(), household);
            state.values[coordinate.0 as usize] = 0;
            let pickup = state.cartage_coordinate(Good::Bread, slot, scenario.ring(), household);
            let waiting = state.values[pickup.0 as usize];
            state.values[pickup.0 as usize] = 0;
            state.set(
                Field::RoadFreight,
                household,
                state.get(Field::RoadFreight, household) - waiting,
            );
            let fare = waiting
                * slot as u64
                * u64::from(
                    crate::domain::railway::CART_BILLABLE_MINUTES_PER_UNIT_DAY
                        * scenario.config().wage,
                );
            state.set(
                Field::CartagePayable,
                household,
                state.get(Field::CartagePayable, household) - fare,
            );
        }
    }
    state.set(
        Field::Bread,
        seller,
        state.get(Field::Bread, seller) + returned,
    );
    let mut buffered = Simulation::from_state(scenario.clone(), state.clone(), device.clone())?;
    let mut synchronous = Simulation::from_state(scenario, state, device.clone())?;
    let mut limited = device.clone();
    let requested = ClaimAllocation::new(initial_capacity as u64, &buffered.config)?;
    limited.memory_budget = crate::simulation::MemoryBudget::Dedicated {
        available_bytes: device.memory_usage().bytes_reserved + requested.bytes(),
    };
    reserve_claims_for_replay(&limited, &mut buffered.config, initial_capacity as u64)?;
    assert_eq!(
        buffered.config.railway.freight.claim_capacity,
        initial_capacity
    );
    buffered.advance()?;
    for phase in Phase::ALL {
        synchronous.phase(phase, Trace::Discard)?;
    }
    assert_eq!(buffered.download()?, synchronous.download()?);
    assert_eq!(buffered.snapshot, synchronous.snapshot);
    let grown_capacity = buffered.config.railway.freight.claim_capacity;
    let recovered = buffered.download()?;
    match contention {
        ClaimContention::Absent => {
            assert_eq!(grown_capacity, 0);
            assert!(buffered.snapshot.used[Good::Railway as usize - 1] > 0);
        }
        ClaimContention::Present => {
            assert!(grown_capacity >= buffered.config.railway.freight.sorting.tile());
            assert!(
                grown_capacity > 1,
                "day {}, activity {:?}, purchased {}, rail used {}, rail operating {:?}, rail stocks {:?}",
                recovered.day,
                buffered.freight_activity,
                (0..seller)
                    .map(|actor| recovered.get(Field::Purchased, actor))
                    .sum::<u64>(),
                buffered.snapshot.used[Good::Railway as usize - 1],
                [
                    recovered.get(Field::RailOperating, seller + 1),
                    recovered.get(Field::RailOperating, seller + 2)
                ],
                [
                    recovered.get(Field::Railway, seller + 1),
                    recovered.get(Field::Railway, seller + 2)
                ],
            );
        }
    }
    enter_bakery(&mut buffered)?;
    enter_bakery(&mut synchronous)?;
    assert!(buffered.config.railway.freight.claim_capacity >= grown_capacity);
    assert_eq!(buffered.download()?, synchronous.download()?);
    for _ in 0..2 {
        buffered.advance()?;
        for phase in Phase::ALL {
            synchronous.phase(phase, Trace::Discard)?;
        }
        assert_eq!(buffered.download()?, synchronous.download()?);
        assert_eq!(buffered.snapshot, synchronous.snapshot);
        assert!(buffered.config.railway.freight.claim_capacity >= grown_capacity);
    }
    assert_eq!(buffered.checkpoint()?, synchronous.checkpoint()?);
    Ok(())
}

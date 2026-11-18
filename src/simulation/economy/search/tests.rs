use super::*;
use crate::{
    domain::{
        Experiment, FirmSeed, HouseholdSeed, Phase, Population, Position, Scenario, State,
        Technique, enterprise::HouseholdProduction,
    },
    simulation::{Market, Simulation, Trace, encode, parallel::ScanMode, tests::device},
};
use anyhow::Result;
use burn_wgpu::CubeTensor;

const FUEL_ROTATION_WIDTH: usize = HOUSEHOLD_SEARCH_SAMPLES as usize;

#[derive(CubeType, CubeLaunch)]
struct FuelRotationInput {
    cases: Box<[u32]>,
    members: Box<[u32]>,
}

struct FuelRotationCheck;
struct CompareFuelRotation;

impl System for FuelRotationCheck {
    type Input = FuelRotationInput;
    type Passes = gpu_passes![CompareFuelRotation];
}

#[cube]
impl GpuPass<FuelRotationInput> for CompareFuelRotation {
    type Output = ();

    fn execute(mut input: FuelRotationInput) {
        let index = ABSOLUTE_POS;
        if index < input.cases.len() / 3 {
            let length = input.cases[index * 3];
            let day = input.cases[index * 3 + 1];
            let household = input.cases[index * 3 + 2];
            let rotation = consumption::FuelSearchRotation::new(length, day, household);
            let samples = min(length, HOUSEHOLD_SEARCH_SAMPLES);
            for sample in 0..samples {
                let offset = sample * (length / samples) + sample * (length % samples) / samples;
                let original = (offset + day + household) % length;
                input.members[index * (2 * FUEL_ROTATION_WIDTH) + sample as usize] = original;
                input.members
                    [index * (2 * FUEL_ROTATION_WIDTH) + FUEL_ROTATION_WIDTH + sample as usize] =
                    rotation.member(sample);
            }
        }
    }
}

#[test]
fn fuel_sample_rotation_preserves_non_power_supplier_counts_and_wrapping_on_gpu() -> Result<()> {
    let mut cases = Vec::new();
    for length in [
        0,
        1,
        2,
        3,
        7,
        8,
        9,
        15,
        17,
        31,
        193,
        257,
        65_537,
        u32::MAX / 2,
        u32::MAX / 2 + 1,
        u32::MAX - 31,
        u32::MAX,
    ] {
        for day in [
            0,
            1,
            u32::MAX / 2,
            u32::MAX - 255,
            u32::MAX - 15,
            u32::MAX - 7,
            u32::MAX - 1,
            u32::MAX,
        ] {
            for household in [0, 1, 7, 31, 193, u32::MAX / 2, u32::MAX - 1, u32::MAX] {
                cases.extend([length, day, household]);
            }
        }
    }
    for workgroup in [32, 256] {
        let device = device(workgroup)?;
        let count = cases.len() / 3;
        let cases_buffer = device.tensor(&cases);
        let members = device.tensor(&vec![u32::MAX; count * 2 * FUEL_ROTATION_WIDTH]);
        FuelRotationCheck::launch_elements(
            &device,
            count,
            FuelRotationInputLaunch {
                cases: cases_buffer.into_buffer_arg(),
                members: members.buffer_arg(),
            },
        );
        let members = device.read(&members)?;
        for (case, members) in cases
            .chunks_exact(3)
            .zip(members.chunks_exact(2 * FUEL_ROTATION_WIDTH))
        {
            assert_eq!(
                members[..FUEL_ROTATION_WIDTH],
                members[FUEL_ROTATION_WIDTH..],
                "{case:?}, WG{workgroup}"
            );
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum EmptyPreparation {
    Bounds,
    ScalarSort,
    BitonicSort,
}

#[derive(CubeType, CubeLaunch)]
struct EmptyPreparationInput {
    actors: Box<[u32]>,
    config: EconomyConfig,
    spatial: Box<[u32]>,
    scratch: Box<[u32]>,
    searches: Box<[u32]>,
    #[cube(comptime)]
    preparation: EmptyPreparation,
    #[cube(comptime)]
    mode: ScanMode,
}

struct EmptyPreparationCheck;
struct CheckEmptyPreparation;

impl System for EmptyPreparationCheck {
    type Input = EmptyPreparationInput;
    type Passes = gpu_passes![CheckEmptyPreparation];
}

#[cube]
impl GpuPass<EmptyPreparationInput> for CheckEmptyPreparation {
    type Output = ();

    fn execute(mut input: EmptyPreparationInput) {
        match comptime!(input.preparation) {
            EmptyPreparation::Bounds => empty::bounds(
                &input.actors,
                &input.config,
                &input.spatial,
                &mut input.scratch,
                comptime!(input.mode),
            ),
            EmptyPreparation::ScalarSort => empty::sort_scalar(
                &input.config,
                &input.spatial,
                &mut input.scratch,
                &mut input.searches,
            ),
            EmptyPreparation::BitonicSort => empty::sort(
                &input.config,
                &input.spatial,
                &mut input.scratch,
                &mut input.searches,
                comptime!(input.mode),
            ),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum EmptyKeys {
    Ties,
    Mixed,
}

#[derive(Debug, PartialEq, Eq)]
struct EmptyPreparationState {
    scratch: Vec<u32>,
    searches: Vec<u32>,
}

struct EmptyPreparationBuffers {
    spatial: CubeTensor,
    scratch: CubeTensor,
    searches: CubeTensor,
    count: usize,
}

impl EmptyPreparationBuffers {
    fn new(simulation: &Simulation, count: usize) -> Self {
        let device = &simulation.device;
        let capacity = simulation.config.actors as usize;
        let mut spatial = vec![0; count + 2];
        spatial[1] = count as u32;
        for member in 0..count {
            spatial[2 + member] = ((member * 73 + 19) % simulation.metadata.actors.len()) as u32;
        }
        Self {
            spatial: device.tensor(&spatial),
            scratch: device.empty(markets::scratch_words(capacity)),
            searches: device.empty(18 * capacity + 32),
            count,
        }
    }

    fn run(
        &self,
        simulation: &Simulation,
        keys: EmptyKeys,
        preparation: EmptyPreparation,
        mode: ScanMode,
    ) -> Result<EmptyPreparationState> {
        let device = &simulation.device;
        let capacity = simulation.config.actors as usize;
        let mut scratch = vec![u32::MAX; self.scratch.meta.num_elements()];
        for member in 0..self.count {
            scratch[4 * member] = match member % 7 {
                0 => 0,
                _ => (member as u32 * 17) % 1000 + 1,
            };
            scratch[4 * member + 1] = member as u32;
            scratch[4 * member + 2] = 0;
            scratch[4 * member + 3] = (member % 4) as u32;
            scratch[5 * capacity + member] = match (keys, member % 8) {
                (EmptyKeys::Ties, _) => 16_383,
                (EmptyKeys::Mixed, 0) => 16_383,
                _ => (member as u32 * 193) % 31,
            };
            scratch[7 * capacity + member] = (self.count - member - 1) as u32;
        }
        device.write(&self.scratch, &scratch);
        device.write(
            &self.searches,
            &vec![u32::MAX; self.searches.meta.num_elements()],
        );
        let invocations = match preparation {
            EmptyPreparation::Bounds => empty_bounds_invocations(capacity, mode),
            _ => self.count.max(1),
        };
        EmptyPreparationCheck::launch_elements(
            device,
            invocations,
            EmptyPreparationInputLaunch {
                actors: simulation.actors.buffer_arg(),
                config: simulation.config.launch(),
                spatial: self.spatial.buffer_arg(),
                scratch: self.scratch.buffer_arg(),
                searches: self.searches.buffer_arg(),
                preparation,
                mode,
            },
        );
        Ok(EmptyPreparationState {
            scratch: device.read(&self.scratch)?,
            searches: device.read(&self.searches)?,
        })
    }
}

#[test]
fn plane_empty_bounds_match_scalar_trees_flat_blocks_and_rail_access_on_gpu() -> Result<()> {
    let scenario = Scenario::industrial(8192, 42, Experiment::Transport)?;
    for workgroup in [32, 64, 256] {
        let mut simulation = Simulation::new(scenario.clone(), device(workgroup)?)?;
        simulation.railways(railway::RailStage::Access);
        simulation.config.spatial_side = 1;
        for capacity in [1024, 8192] {
            simulation.config.actors = capacity;
            for availability in [
                railway::RouteAvailability::Available,
                railway::RouteAvailability::Dormant,
            ] {
                simulation
                    .config
                    .railway
                    .replace_route_availability(availability);
                for count in [0, 1, 7, 8, 9, 31, 32, 33, 511, 513, 957, 2048, 2049]
                    .into_iter()
                    .filter(|&count| count <= capacity as usize)
                {
                    let buffers = EmptyPreparationBuffers::new(&simulation, count);
                    let scalar = buffers.run(
                        &simulation,
                        EmptyKeys::Mixed,
                        EmptyPreparation::Bounds,
                        ScanMode::Shared,
                    )?;
                    let plane = buffers.run(
                        &simulation,
                        EmptyKeys::Mixed,
                        EmptyPreparation::Bounds,
                        ScanMode::new(&simulation.device),
                    )?;
                    assert_eq!(
                        plane, scalar,
                        "workgroup {workgroup}, capacity {capacity}, count {count}, {availability:?}"
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
fn bitonic_empty_sort_matches_scalar_morton_ties_and_original_stocked_order_on_gpu() -> Result<()> {
    let scenario = Scenario::industrial(8192, 42, Experiment::Transport)?;
    for workgroup in [32, 64, 128, 256] {
        let mut simulation = Simulation::new(scenario.clone(), device(workgroup)?)?;
        simulation.config.spatial_side = 1;
        for keys in [EmptyKeys::Ties, EmptyKeys::Mixed] {
            for count in [
                0, 1, 7, 8, 31, 32, 33, 257, 957, 1023, 1024, 1025, 2047, 2048, 2049,
            ] {
                let buffers = EmptyPreparationBuffers::new(&simulation, count);
                let scalar = buffers.run(
                    &simulation,
                    keys,
                    EmptyPreparation::ScalarSort,
                    ScanMode::Shared,
                )?;
                for mode in [ScanMode::Shared, ScanMode::new(&simulation.device)] {
                    let bitonic =
                        buffers.run(&simulation, keys, EmptyPreparation::BitonicSort, mode)?;
                    assert_eq!(
                        bitonic, scalar,
                        "workgroup {workgroup}, count {count}, {keys:?}, {mode:?}"
                    );
                    if count > 0 && count <= empty::LIMIT as usize {
                        let capacity = simulation.config.actors as usize;
                        let expected = (0..count)
                            .filter(|member| member % 4 & 1 != 0)
                            .map(|member| member as u32)
                            .collect::<Vec<_>>();
                        assert_eq!(bitonic.scratch[7 * capacity + count], expected.len() as u32);
                        assert_eq!(
                            bitonic.searches[17 * capacity..17 * capacity + expected.len()],
                            expected
                        );
                    }
                }
            }
        }
    }
    Ok(())
}

const SUPPLIERS: u32 = 385;

#[derive(Clone, Copy, Debug)]
enum SupplyCase {
    Capped,
    Sparse,
    Empty,
    Unaffordable,
    MixedAffordability,
    Retail,
    RetailSold,
    Free,
    HighCash,
    DistantStock,
    DistantEmpty,
    LaborPrefix(u32),
}

#[derive(Clone, Copy, Debug)]
enum SearchExecution {
    Serial,
    Prepare,
    Shared,
    Plane,
    Cached,
    ScalarFuel,
    ScalarHousehold,
    SpatialFuel,
}

#[derive(Clone, Copy)]
enum CacheRefresh {
    Reset,
    Retain,
}

#[derive(Debug, PartialEq, Eq)]
struct SearchOutcome {
    proposal: Vec<u32>,
    state: Vec<u32>,
    faults: u32,
}

#[derive(Debug, PartialEq, Eq)]
struct PreparedRequests {
    actors: Vec<u32>,
    demand: Vec<u32>,
    candidates: Vec<u32>,
    statuses: Vec<u32>,
    faults: u32,
}

struct SearchProbe {
    simulation: Simulation,
    state: State,
    parameters: EconomyParameters,
}

impl SearchProbe {
    fn new(case: SupplyCase, day: u32, market: Market) -> Result<Self> {
        Self::with_suppliers(case, day, market, SUPPLIERS)
    }

    fn with_suppliers(case: SupplyCase, day: u32, market: Market, suppliers: u32) -> Result<Self> {
        let good = match market {
            Market::Input(good) | Market::Consumer(good) => good,
            _ => Good::Bread,
        };
        let mut config = Scenario::baseline(1, 42, Experiment::Concentrated)?
            .config()
            .clone();
        config.firms = suppliers;
        config.search_radius = config.map_size;
        config.household_production = HouseholdProduction::industrial();
        let position = match case {
            SupplyCase::DistantStock | SupplyCase::DistantEmpty => Position { x: 0, y: 0 },
            _ => Position {
                x: config.map_size / 2,
                y: config.map_size / 2,
            },
        };
        let input_ticks = match good.inputs()[0] {
            Some(_) => 1,
            None => 0,
        };
        config.population = Some(Population {
            households: vec![HouseholdSeed {
                identity: 1,
                position,
                cash: 5000,
                inventories: Good::food([0, 0, 100]),
                fatigue: 0,
            }],
            firms: (2..=u64::from(suppliers) + 1)
                .map(|identity| FirmSeed {
                    identity,
                    position: match case {
                        SupplyCase::DistantStock | SupplyCase::DistantEmpty => match identity {
                            2..=129 => Position {
                                x: config.map_size - 1,
                                y: config.map_size - 1,
                            },
                            130..=257 => Position {
                                x: config.map_size * 3 / 4,
                                y: 0,
                            },
                            _ => Position { x: 0, y: 1 },
                        },
                        _ => position,
                    },
                    owner: 1,
                    good,
                    cash: 5000,
                    inventories: [0; Good::COUNT],
                    capacity: 1,
                    technique: Technique::new(1, input_ticks, 1).expect("one-minute packet"),
                    wage: 1,
                    price: 1,
                })
                .collect(),
        });
        if let SupplyCase::LaborPrefix(_) = case {
            config.bargaining = Some(crate::domain::bargaining::BargainingPolicy {
                legal_minimum: 1,
                quantum: 1,
                daily_cash_cost: 1_000_000,
                funded_other: 0,
                participating_adults: 1,
                normal_day_minutes: 1440,
                work_cost_per_minute: 0,
                desired_wage: 1000,
                mandate: 500,
                withdrawal: 0,
                alternatives: 0,
                concentration: 0,
                replacements: 0,
            });
        }
        let scenario = Scenario::new(config)?;
        let mut state = scenario.initial()?;
        state.set(Field::FoodNeed, 0, 3);
        state.set(Field::LivingMembers, 0, 3);
        if let Market::Consumer(Good::Coal | Good::Wood) = market {
            state.set(Field::HeatingNeed, 0, 9);
        }
        let bread = match market {
            Market::Consumer(_) => {
                state.get(Field::FoodNeed, 0)
                    * (u64::from(scenario.config().coverage) + scenario.ring() as u64)
            }
            _ => 0,
        };
        state.set(Field::Bread, 0, bread);
        if let Market::Consumer(_) = market {
            state.set(Field::Price, 0, 1);
        }
        state.set(Field::Time, 0, u64::from(u32::MAX));
        if let SupplyCase::HighCash = case {
            state.set(Field::Cash, 0, u64::from(u32::MAX) + 5000);
        }
        for firm in 1..=suppliers as usize {
            let supply = match case {
                SupplyCase::Capped
                | SupplyCase::Unaffordable
                | SupplyCase::Retail
                | SupplyCase::RetailSold
                | SupplyCase::Free
                | SupplyCase::HighCash
                | SupplyCase::DistantStock => 100,
                SupplyCase::MixedAffordability => u64::from(firm > 1) * 100,
                SupplyCase::Sparse => u64::from(firm % 129 == 0) * 100,
                _ => 0,
            };
            let price = match case {
                SupplyCase::Capped => 1 + u64::from(firm <= 8) * 9,
                SupplyCase::Sparse | SupplyCase::Empty => 1 + (firm % 3) as u64,
                SupplyCase::Unaffordable => 5001,
                SupplyCase::MixedAffordability => u64::from(firm > 1) * 5001,
                SupplyCase::Retail | SupplyCase::RetailSold => 1 + (firm % 3) as u64,
                SupplyCase::Free => 0,
                SupplyCase::HighCash => 5001,
                SupplyCase::LaborPrefix(_)
                | SupplyCase::DistantStock
                | SupplyCase::DistantEmpty => 1,
            };
            state.set(Field::inventory(good), firm, supply);
            state.set(Field::Price, firm, price);
            state.set(Field::LaborMinutes, firm, 1);
            state.set(Field::InputTicks, firm, 0);
            state.set(Field::EquipmentCondition, firm, 1000);
            if let SupplyCase::Retail | SupplyCase::RetailSold = case {
                state.set(Field::EnterpriseModel, firm, 1);
                state.set(Field::RetailOpen, firm, u64::from(u32::MAX) + 10);
                let sales = match case {
                    SupplyCase::Retail => u64::from(u32::MAX) + firm as u64 % 10,
                    _ => u64::from(u32::MAX) + 11,
                };
                state.set(Field::Sales, firm, sales);
            }
            if let SupplyCase::LaborPrefix(available) = case {
                state.set(
                    Field::Slots,
                    firm,
                    u64::from(firm <= available as usize || firm == 9 || firm == 10),
                );
            }
        }
        if let SupplyCase::LaborPrefix(_) = case {
            state.set(Field::LaborMinutes, 10, u64::from(u32::MAX));
            state.set(Field::EquipmentCondition, 10, u64::from(u32::MAX));
        }
        let market = match case {
            SupplyCase::LaborPrefix(_) => Market::Labor,
            _ => market,
        };
        let simulation = Simulation::new(scenario, device(64)?)?;
        let mut parameters =
            simulation.parameters(Phase::ProvisionPurchase, market, 0, state.actors.len())?;
        parameters.day = match market {
            Market::Consumer(_) => 0,
            _ => day,
        };
        parameters.settlement_rotation = day;
        let mut spatial = simulation.device.read(&simulation.spatial)?;
        spatial.fill(0);
        spatial[1] = suppliers;
        spatial[2..suppliers as usize + 2].copy_from_slice(&(1..=suppliers).collect::<Vec<u32>>());
        let prices = 3 * (2 + state.actors.len());
        spatial[prices] = (1..=suppliers as usize)
            .map(|firm| state.get(Field::Price, firm) as u32)
            .min()
            .expect("suppliers");
        let query = prices + state.actors.len().div_ceil(DISCOVERY_BLOCK as usize) + 1;
        let mut queries = (0..state.actors.len() as u32).collect::<Vec<_>>();
        if let Market::Consumer(_) = market {
            queries.reverse();
        }
        spatial[query..query + state.actors.len()].copy_from_slice(&queries);
        simulation.device.write(&simulation.spatial, &spatial);
        Ok(Self {
            simulation,
            state,
            parameters,
        })
    }

    fn outcome(&self, execution: SearchExecution) -> Result<SearchOutcome> {
        self.outcome_with_workgroup(
            execution,
            match execution {
                SearchExecution::Cached => 256,
                _ => 32,
            },
        )
    }

    fn outcome_with_workgroup(
        &self,
        execution: SearchExecution,
        workgroup: u32,
    ) -> Result<SearchOutcome> {
        self.outcome_with_cache(execution, workgroup, CacheRefresh::Reset)
    }

    fn outcome_with_cache(
        &self,
        execution: SearchExecution,
        workgroup: u32,
        cache: CacheRefresh,
    ) -> Result<SearchOutcome> {
        self.actor_outcome(execution, workgroup, cache, 0)
    }

    fn actor_outcome(
        &self,
        execution: SearchExecution,
        workgroup: u32,
        cache: CacheRefresh,
        buyer: usize,
    ) -> Result<SearchOutcome> {
        let simulation = &self.simulation;
        let device = &simulation.device;
        let count = self.state.actors.len();
        let mut parameters = self.parameters;
        parameters.scan = match execution {
            SearchExecution::Shared => ScanMode::Shared,
            _ => ScanMode::new(device),
        };
        device.write(
            &simulation.state,
            bytemuck::cast_slice(&encode(&self.state.values)),
        );
        let mut control = vec![0; simulation.control.meta.num_elements()];
        control[INPUT_MINIMUM_QUOTE_WORD] = u32::MAX;
        control[SEARCHES_WORD] = 1;
        device.write(&simulation.control, &control);
        device.write(
            &simulation.proposals,
            &vec![NO_ACTOR; count * ProposalLayout::WORDS],
        );
        let mut searches = match cache {
            CacheRefresh::Reset => vec![0; simulation.searches.requests.meta.num_elements()],
            CacheRefresh::Retain => device.read(&simulation.searches.requests)?,
        };
        searches[..count * 6].fill(0);
        searches[0] = buyer as u32;
        let coverage = match parameters.market {
            MarketFlag::Input(_) if parameters.good == Good::Grain as u32 => {
                simulation.config.coverage + simulation.config.ring
            }
            _ => simulation.config.coverage + simulation.config.ring - 1,
        };
        let wanted = match parameters.market {
            MarketFlag::Consumer(_) if parameters.good == Good::Coal as u32 => {
                (self.state.get(Field::HeatingNeed, buyer) * u64::from(coverage)).div_ceil(2)
            }
            MarketFlag::Consumer(_) if parameters.good == Good::Wood as u32 => {
                self.state.get(Field::HeatingNeed, buyer) * u64::from(coverage)
            }
            MarketFlag::Consumer(_) => 1,
            _ => self.state.get(Field::FoodNeed, buyer) * u64::from(coverage),
        };
        searches[count + buyer] = wanted as u32;
        searches[2 * count + buyer] = (wanted >> 32) as u32;
        searches[3 * count + buyer] = self.state.get(Field::Cash, buyer) as u32;
        searches[4 * count + buyer] = (self.state.get(Field::Cash, buyer) >> u32::BITS) as u32;
        device.write(&simulation.searches.requests, &searches);
        device.write(
            &simulation.searches.successes,
            &vec![
                0;
                markets::search_counts_offset(count) + 2 * markets::search_blocks(count) + 512
            ],
        );
        if matches!(
            (parameters.market, execution),
            (MarketFlag::Consumer(_) | MarketFlag::Household, _) | (_, SearchExecution::Cached)
        ) {
            MarketSearch::launch_elements(
                device,
                count.div_ceil(DISCOVERY_BLOCK as usize) * device.workgroup as usize,
                SearchInputLaunch {
                    state: simulation.state.buffer_arg(),
                    actors: simulation.actors.buffer_arg(),
                    config: simulation.config.launch(),
                    spatial: simulation.spatial.buffer_arg(),
                    proposals: simulation.proposals.buffer_arg(),
                    searches: simulation.searches.requests.buffer_arg(),
                    successes: simulation.searches.successes.buffer_arg(),
                    scratch: simulation.scratch.buffer_arg(),
                    control: simulation.control.buffer_arg(),
                    parameters: parameters.launch(),
                    status_reset: StatusReset::Full,
                    program: SearchStage::Index.program(&simulation.config),
                },
            );
            if matches!(parameters.market, MarketFlag::Consumer(_)) {
                for stage in [SearchStage::EmptySort, SearchStage::EmptyBounds] {
                    let invocations = match stage {
                        SearchStage::EmptyBounds => {
                            empty_bounds_invocations(count, parameters.scan)
                        }
                        _ => count,
                    };
                    MarketSearch::launch_elements(
                        device,
                        invocations,
                        SearchInputLaunch {
                            state: simulation.state.buffer_arg(),
                            actors: simulation.actors.buffer_arg(),
                            config: simulation.config.launch(),
                            spatial: simulation.spatial.buffer_arg(),
                            proposals: simulation.proposals.buffer_arg(),
                            searches: simulation.searches.requests.buffer_arg(),
                            successes: simulation.searches.successes.buffer_arg(),
                            scratch: simulation.scratch.buffer_arg(),
                            control: simulation.control.buffer_arg(),
                            parameters: parameters.launch(),
                            status_reset: StatusReset::Full,
                            program: stage.program(&simulation.config),
                        },
                    );
                }
            }
        }
        let stage = match execution {
            SearchExecution::Prepare => SearchStage::Prepare,
            SearchExecution::ScalarFuel => SearchStage::SearchFuel,
            SearchExecution::ScalarHousehold => SearchStage::SearchHousehold,
            SearchExecution::SpatialFuel => SearchStage::SearchSpatialFuel,
            SearchExecution::Cached => SearchStage::SearchIndexed,
            _ => SearchStage::Search,
        };
        match execution {
            SearchExecution::Serial => EconomySystem::launch_single(
                device,
                EconomyInputLaunch {
                    state: simulation.state.buffer_arg(),
                    actors: simulation.actors.buffer_arg(),
                    config: simulation.config.launch(),
                    spatial: simulation.spatial.buffer_arg(),
                    proposals: simulation.proposals.buffer_arg(),
                    bundles: simulation.scratch.buffer_arg(),
                    control: simulation.control.buffer_arg(),
                    parameters: parameters.launch(),
                    stage: EconomyStage::Propose,
                },
            ),
            _ => MarketSearch::launch(
                device,
                match execution {
                    SearchExecution::SpatialFuel => {
                        CubeCount::Static(count.div_ceil(workgroup as usize) as u32, 1, 1)
                    }
                    _ => CubeCount::Static(1, 1, 1),
                },
                CubeDim::new_1d(workgroup),
                SearchInputLaunch {
                    state: simulation.state.buffer_arg(),
                    actors: simulation.actors.buffer_arg(),
                    config: simulation.config.launch(),
                    spatial: simulation.spatial.buffer_arg(),
                    proposals: simulation.proposals.buffer_arg(),
                    searches: simulation.searches.requests.buffer_arg(),
                    successes: simulation.searches.successes.buffer_arg(),
                    scratch: simulation.scratch.buffer_arg(),
                    control: simulation.control.buffer_arg(),
                    parameters: parameters.launch(),
                    status_reset: StatusReset::Full,
                    program: stage.program(&simulation.config),
                },
            ),
        }
        let proposals = device.read(&simulation.proposals)?;
        let mut proposal =
            proposals[buyer * ProposalLayout::WORDS..(buyer + 1) * ProposalLayout::WORDS].to_vec();
        match proposal[ProposalLayout::ACTOR] {
            NO_ACTOR => proposal[ProposalLayout::QUANTITY_LOW..].fill(0),
            _ => (),
        }
        Ok(SearchOutcome {
            proposal,
            state: device.read(&simulation.state)?,
            faults: device.read(&simulation.control)?[crate::simulation::flags::FAULT_WORD],
        })
    }
}

#[test]
fn funded_stock_search_preserves_wide_price_division_and_overflow_fallback_on_gpu() -> Result<()> {
    for price in [5_001, u32::MAX - 1] {
        let mut probe = SearchProbe::new(SupplyCase::HighCash, 0, Market::Input(Good::Bread))?;
        probe.state.set(Field::Cash, 0, u64::MAX);
        probe.state.set(Field::FoodNeed, 0, u64::from(u32::MAX));
        for supplier in 1..=SUPPLIERS as usize {
            probe.state.set(Field::Bread, supplier, u64::MAX);
            probe.state.set(Field::Price, supplier, u64::from(price));
        }
        let coverage = probe.simulation.config.coverage + probe.simulation.config.ring - 1;
        let expected = (u64::from(u32::MAX) * u64::from(coverage)).min(u64::MAX / u64::from(price));
        for execution in [
            SearchExecution::Plane,
            SearchExecution::Shared,
            SearchExecution::Cached,
        ] {
            let outcome = probe.outcome(execution)?;
            assert_ne!(outcome.proposal[ProposalLayout::ACTOR], NO_ACTOR);
            let quantity = u64::from(outcome.proposal[ProposalLayout::QUANTITY_LOW])
                | (u64::from(outcome.proposal[ProposalLayout::QUANTITY_HIGH]) << 32);
            assert_eq!(quantity, expected, "price {price}, execution {execution:?}");
            assert_eq!(outcome.faults, 0);
        }
    }
    Ok(())
}

#[test]
fn indexed_bread_minimum_quotes_preserve_zero_targets_empty_offers_and_cash_boundaries_on_gpu()
-> Result<()> {
    #[derive(Clone, Copy, Debug)]
    enum SupplyStock {
        Stocked,
        Empty,
        Mixed,
    }

    struct CashBoundary {
        quote: u32,
        cash: u64,
    }

    for stock in [SupplyStock::Stocked, SupplyStock::Empty, SupplyStock::Mixed] {
        for boundary in [
            CashBoundary { quote: 8, cash: 0 },
            CashBoundary { quote: 8, cash: 7 },
            CashBoundary { quote: 8, cash: 8 },
            CashBoundary {
                quote: 8,
                cash: u64::from(u32::MAX) + 8,
            },
            CashBoundary {
                quote: u32::MAX,
                cash: u64::from(u32::MAX) - 1,
            },
            CashBoundary {
                quote: u32::MAX,
                cash: u64::from(u32::MAX),
            },
            CashBoundary {
                quote: u32::MAX,
                cash: u64::MAX,
            },
        ] {
            for wanted in [3, u32::MAX] {
                let mut probe =
                    SearchProbe::new(SupplyCase::Capped, 127, Market::Input(Good::Bread))?;
                probe.state.set(Field::InputTicks, 1, 0);
                probe.state.set(Field::FoodNeed, 1, u64::from(wanted));
                probe.state.set(Field::Cash, 1, boundary.cash);
                for supplier in 1..=SUPPLIERS as usize {
                    let amount = match (stock, supplier) {
                        (_, 1) | (SupplyStock::Empty, _) | (SupplyStock::Mixed, 2) => 0,
                        _ => 100,
                    };
                    let quote = match (stock, supplier) {
                        (_, 1 | 3) => 0,
                        (SupplyStock::Mixed, 2) => boundary.quote,
                        (SupplyStock::Mixed, _) => boundary.quote.saturating_add(9),
                        _ => boundary.quote,
                    };
                    probe.state.set(Field::Bread, supplier, amount);
                    probe.state.set(Field::Price, supplier, u64::from(quote));
                }
                let expected =
                    probe.actor_outcome(SearchExecution::Plane, 32, CacheRefresh::Reset, 1)?;
                for workgroup in [32, 256] {
                    let actual = probe.actor_outcome(
                        SearchExecution::Cached,
                        workgroup,
                        CacheRefresh::Reset,
                        1,
                    )?;
                    assert_eq!(
                        actual, expected,
                        "stock {stock:?}, quote {}, cash {}, wanted {wanted}, workgroup {workgroup}",
                        boundary.quote, boundary.cash
                    );
                    assert_eq!(
                        probe.simulation.device.read_range(
                            &probe.simulation.control,
                            INPUT_MINIMUM_QUOTE_WORD,
                            1
                        )?[0],
                        boundary.quote,
                    );
                    if boundary.cash < u64::from(boundary.quote) {
                        assert_eq!(actual.proposal[ProposalLayout::ACTOR], NO_ACTOR);
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
fn indexed_partial_input_search_preserves_quantities_ties_and_wide_targets_on_gpu() -> Result<()> {
    #[derive(Clone, Copy, Debug)]
    enum PartialInputOffers {
        Ranked,
        Tied,
    }

    for offers in [PartialInputOffers::Ranked, PartialInputOffers::Tied] {
        for rotation in [0, 31, 128, SUPPLIERS - 1] {
            for input_ticks in [1000, u32::MAX] {
                let mut probe =
                    SearchProbe::new(SupplyCase::Capped, rotation, Market::Input(Good::Bread))?;
                probe.state.set(Field::FoodNeed, 1, u64::from(input_ticks));
                probe
                    .state
                    .set(Field::InputTicks, 1, u64::from(input_ticks));
                probe
                    .state
                    .set(Field::Cash, 1, u64::from(u32::MAX) + 10_000);
                let capacity = match input_ticks {
                    u32::MAX => 2,
                    _ => 1,
                };
                let mut actor = probe.state.actors.at(1);
                actor.capacity = capacity;
                probe.state.actors.set(1, actor);
                let mut actors = probe.simulation.device.read(&probe.simulation.actors)?;
                actors[ActorLayout::CAPACITY * probe.state.actors.len() + 1] = capacity;
                probe
                    .simulation
                    .device
                    .write(&probe.simulation.actors, &actors);
                for supplier in 1..=SUPPLIERS as usize {
                    let stock = match supplier {
                        8 | 97 | 257 => 200,
                        _ => (supplier % 7) as u64,
                    };
                    let quote = match offers {
                        PartialInputOffers::Ranked => (1 + supplier % 11) as u64,
                        PartialInputOffers::Tied => 1,
                    };
                    probe.state.set(Field::Bread, supplier, stock);
                    probe.state.set(Field::Price, supplier, quote);
                }
                let expected =
                    probe.actor_outcome(SearchExecution::Plane, 32, CacheRefresh::Reset, 1)?;
                let chosen = expected.proposal[ProposalLayout::ACTOR];
                match offers {
                    PartialInputOffers::Ranked => assert_eq!(chosen, 257),
                    PartialInputOffers::Tied => assert!([8, 97, 257].contains(&chosen)),
                }
                assert_eq!(expected.proposal[ProposalLayout::QUANTITY_LOW], 200);
                assert_eq!(expected.proposal[ProposalLayout::QUANTITY_HIGH], 0);
                for workgroup in [32, 64, 256] {
                    assert_eq!(
                        probe.actor_outcome(
                            SearchExecution::Cached,
                            workgroup,
                            CacheRefresh::Reset,
                            1
                        )?,
                        expected,
                        "offers {offers:?}, rotation {rotation}, input {input_ticks}, workgroup {workgroup}",
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
fn cached_bread_suppliers_preserve_rotations_stock_and_affordability_on_gpu() -> Result<()> {
    for (case, day) in [
        (SupplyCase::Retail, 0),
        (SupplyCase::Sparse, 1),
        (SupplyCase::DistantStock, 19),
        (SupplyCase::DistantEmpty, SUPPLIERS - 1),
    ] {
        let probe = SearchProbe::new(case, day, Market::Input(Good::Bread))?;
        assert_eq!(
            probe.outcome(SearchExecution::Cached)?,
            probe.outcome(SearchExecution::Plane)?,
            "{case:?} day {day}",
        );
    }
    Ok(())
}

#[test]
fn inactive_household_input_preparation_preserves_inventory_faults_on_gpu() -> Result<()> {
    for good in Good::ALL {
        let mut probe = SearchProbe::new(SupplyCase::Capped, 0, Market::Input(Good::Linen))?;
        probe.parameters.good = good as u32;
        probe.state.set(Field::LivingMembers, 0, 0);
        for inventory in [0, u64::MAX] {
            probe.state.set(Field::inventory(good), 0, inventory);
            let transit = TRANSIT_FIELDS + good as usize - 1;
            probe.state.values[transit * probe.state.actors.len()] = 1;
            let expected = probe.outcome(SearchExecution::Serial)?;
            assert_eq!(expected.faults == 0, inventory == 0, "{good:?}");
            assert_eq!(
                probe.outcome(SearchExecution::Prepare)?,
                expected,
                "{good:?}, inventory {inventory}"
            );
        }
    }
    Ok(())
}

#[test]
fn cooperative_market_search_preserves_serial_search_and_shared_fallback_on_gpu() -> Result<()> {
    let cases = [
        SupplyCase::Capped,
        SupplyCase::Sparse,
        SupplyCase::Empty,
        SupplyCase::Unaffordable,
        SupplyCase::MixedAffordability,
        SupplyCase::Retail,
        SupplyCase::RetailSold,
        SupplyCase::Free,
        SupplyCase::HighCash,
        SupplyCase::DistantStock,
        SupplyCase::DistantEmpty,
    ];
    for market in [
        Market::Household,
        Market::Input(Good::Grain),
        Market::Consumer(Good::Linen),
        Market::Consumer(Good::Glass),
        Market::Consumer(Good::Coal),
        Market::Consumer(Good::Wood),
    ] {
        for (index, case) in cases.into_iter().enumerate() {
            let suppliers = match (market, case) {
                (
                    Market::Household,
                    SupplyCase::Sparse | SupplyCase::Empty | SupplyCase::DistantEmpty,
                ) => 2177,
                _ => SUPPLIERS,
            };
            let day = [0, 128, suppliers - 1][index % 3];
            let probe = SearchProbe::with_suppliers(case, day, market, suppliers)?;
            let expected = probe.outcome(SearchExecution::Serial)?;
            if let SupplyCase::Capped = case {
                assert_ne!(expected.proposal[ProposalLayout::ACTOR], NO_ACTOR);
                assert!(expected.proposal[ProposalLayout::QUANTITY_LOW] > 0);
            }
            for execution in [SearchExecution::Shared, SearchExecution::Plane] {
                assert_eq!(
                    probe.outcome(execution)?,
                    expected,
                    "{case:?}, {market:?}, day {day}, {execution:?}"
                );
            }
            if let Market::Consumer(Good::Coal | Good::Wood) = market {
                for execution in [SearchExecution::ScalarFuel, SearchExecution::SpatialFuel] {
                    assert_eq!(
                        probe.outcome(execution)?,
                        expected,
                        "{case:?}, {market:?}, day {day}, {execution:?}"
                    );
                }
            }
            if let Market::Household = market {
                assert_eq!(
                    probe.outcome(SearchExecution::ScalarHousehold)?,
                    expected,
                    "{case:?}, {market:?}, day {day}, scalar household"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn bounded_household_search_rotates_samples_and_repeats_on_gpu() -> Result<()> {
    for day in [0, 1] {
        let mut probe = SearchProbe::new(SupplyCase::Empty, day, Market::Household)?;
        for firm in 1..=SUPPLIERS as usize {
            probe.state.set(Field::Price, firm, 0);
        }
        probe.state.set(Field::Bread, 10, 100);
        probe.state.set(Field::Price, 10, 1);
        let expected = match day {
            0 => NO_ACTOR,
            _ => 10,
        };
        for execution in [
            SearchExecution::Serial,
            SearchExecution::Shared,
            SearchExecution::Plane,
            SearchExecution::ScalarHousehold,
        ] {
            let first = probe.outcome(execution)?;
            assert_eq!(first.proposal[ProposalLayout::ACTOR], expected);
            assert_eq!(first, probe.outcome(execution)?);
        }
    }
    Ok(())
}

#[test]
fn cooperative_market_search_only_reports_faults_from_the_visited_prefix_on_gpu() -> Result<()> {
    for available in [7, 6] {
        let probe = SearchProbe::new(SupplyCase::LaborPrefix(available), 0, Market::Labor)?;
        let expected = probe.outcome(SearchExecution::Serial)?;
        assert_eq!(expected.faults == 0, available == 7);
        for execution in [SearchExecution::Shared, SearchExecution::Plane] {
            assert_eq!(
                probe.outcome(execution)?,
                expected,
                "eligible prefix {available}, {execution:?}"
            );
        }
    }
    Ok(())
}

#[test]
fn deferred_consumer_quantity_preserves_wide_budgets_and_empty_requests_on_gpu() -> Result<()> {
    for good in [Good::Coal, Good::Wood] {
        for case in [SupplyCase::HighCash, SupplyCase::Empty] {
            let mut probe = SearchProbe::new(case, 128, Market::Consumer(good))?;
            probe.state.set(Field::Cash, 0, u64::MAX / 1000);
            probe.state.set(Field::HeatingNeed, 0, u64::from(u32::MAX));
            for firm in 1..probe.state.actors.len() {
                let stock = match case {
                    SupplyCase::Empty => 0,
                    _ => u64::MAX / 1000,
                };
                probe.state.set(Field::inventory(good), firm, stock);
            }
            let expected = probe.outcome(SearchExecution::Serial)?;
            assert_ne!(expected.proposal[ProposalLayout::ACTOR], NO_ACTOR);
            assert!(expected.proposal[ProposalLayout::QUANTITY_HIGH] > 0);
            for execution in [
                SearchExecution::Shared,
                SearchExecution::Plane,
                SearchExecution::ScalarFuel,
                SearchExecution::SpatialFuel,
            ] {
                assert_eq!(
                    probe.outcome(execution)?,
                    expected,
                    "{good:?}, {case:?}, {execution:?}",
                );
            }
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum DeliveryScoreCase {
    Tied,
    QuoteWrap,
    PenaltyWrap,
    AccessWrap,
    ScaledSpeedWrap,
}

#[derive(CubeType, CubeLaunch)]
struct DeliveryScoreFixtureInput {
    config: EconomyConfig,
    buyer: u32,
    seller: u32,
    #[cube(comptime)]
    case: DeliveryScoreCase,
}

struct DeliveryScoreFixture;
struct ConfigureDeliveryScore;

impl System for DeliveryScoreFixture {
    type Input = DeliveryScoreFixtureInput;
    type Passes = gpu_passes![ConfigureDeliveryScore];
}

#[cube]
impl GpuPass<DeliveryScoreFixtureInput> for ConfigureDeliveryScore {
    type Output = ();

    fn execute(mut input: DeliveryScoreFixtureInput) {
        if ABSOLUTE_POS == 0 {
            match comptime!(input.case) {
                DeliveryScoreCase::QuoteWrap | DeliveryScoreCase::AccessWrap => {
                    for index in 0..input.config.railway.quotes.len() {
                        input.config.railway.quotes[index] = u32::MAX;
                    }
                }
                _ => (),
            }
            if comptime!(input.case == DeliveryScoreCase::AccessWrap) {
                for actor in 0..input.config.actors {
                    input.config.railway.access[actor as usize * 2] = 2;
                    input.config.railway.access[actor as usize * 2 + 1] = 0;
                }
                input.config.railway.access[input.buyer as usize * 2] = 0;
                input.config.railway.access[input.seller as usize * 2] = 1;
                input.config.railway.access[input.seller as usize * 2 + 1] = u32::MAX - 3;
                let nodes = input.config.railway.nodes;
                let base = nodes as usize * crate::domain::railway::RAIL_ROUTE_WORDS;
                input.config.railway.routes[base] = 50;
                input.config.railway.routes[base + 2] = 0;
                input.config.railway.quotes[(nodes * input.config.ring) as usize] = 0;
            }
        }
    }
}

fn delivery_score_probe(good: Good, rotation: u32, case: DeliveryScoreCase) -> Result<SearchProbe> {
    let mut config: crate::domain::ScenarioConfig =
        serde_json::from_str(include_str!("../../../../scenarios/railway.json"))?;
    let population = config.population.as_mut().expect("railway population");
    let first = population.households.len() + population.firms.len();
    let template = population.firms[3].clone();
    for index in 0..9 {
        let mut inventories = [0; Good::COUNT];
        inventories[good as usize - 1] = u64::MAX / 1000;
        population.firms.push(FirmSeed {
            identity: 100 + index,
            position: match index % 4 {
                0 => Position { x: 900, y: 200 },
                1 => Position { x: 0, y: 0 },
                2 => Position { x: 400, y: 50 },
                _ => Position { x: 900, y: 0 },
            },
            good,
            inventories,
            price: 10,
            ..template.clone()
        });
    }
    config.firms = population.firms.len() as u32;
    let scenario = Scenario::new(config)?;
    let mut state = scenario.initial()?;
    state.set(Field::Cash, 0, u64::MAX);
    state.set(Field::HeatingNeed, 0, u64::from(u32::MAX));
    for actor in [
        scenario.config().households as usize + 1,
        scenario.config().households as usize + 2,
    ] {
        state.set(Field::RailBuilt, actor, 1);
        state.set(Field::RailOperating, actor, 1);
    }
    match case {
        DeliveryScoreCase::QuoteWrap | DeliveryScoreCase::AccessWrap => {
            for actor in first..first + 9 {
                state.set(Field::Price, actor, 1000);
            }
            state.set(Field::Price, first, 1);
            state.set(
                Field::Price,
                first + 1,
                match case {
                    DeliveryScoreCase::QuoteWrap => u64::from(u32::MAX - 40),
                    _ => 5,
                },
            );
        }
        DeliveryScoreCase::PenaltyWrap => state.set(Field::Price, first, 1),
        _ => (),
    }
    let mut simulation = Simulation::new(scenario, device(64)?)?;
    simulation.device.write(
        &simulation.state,
        bytemuck::cast_slice(&encode(&state.values)),
    );
    for stage in [
        railway::RailStage::Access,
        railway::RailStage::Routes,
        railway::RailStage::Quotes,
    ] {
        simulation.railways(stage);
    }
    match case {
        DeliveryScoreCase::PenaltyWrap => simulation.config.delivery_penalty = u32::MAX,
        DeliveryScoreCase::AccessWrap => simulation.config.shipment_speed = 1,
        DeliveryScoreCase::ScaledSpeedWrap => simulation.config.shipment_speed = u32::MAX,
        _ => (),
    }
    DeliveryScoreFixture::launch_single(
        &simulation.device,
        DeliveryScoreFixtureInputLaunch {
            config: simulation.config.launch(),
            buyer: 0,
            seller: first as u32 + 1,
            case,
        },
    );
    let count = state.actors.len();
    let cells = (simulation.config.spatial_side * simulation.config.spatial_side) as usize;
    let segment = cells + 1 + count;
    let mut spatial = simulation.device.read(&simulation.spatial)?;
    spatial.fill(0);
    spatial[1] = 9;
    spatial[cells + 1..cells + 10]
        .copy_from_slice(&(first as u32..first as u32 + 9).collect::<Vec<_>>());
    let query = 3 * segment + count.div_ceil(DISCOVERY_BLOCK as usize) + 1;
    spatial[query..query + count].copy_from_slice(&(0..count as u32).collect::<Vec<_>>());
    simulation.device.write(&simulation.spatial, &spatial);
    let mut parameters =
        simulation.parameters(Phase::ProvisionPurchase, Market::Consumer(good), 0, count)?;
    parameters.settlement_rotation = rotation;
    Ok(SearchProbe {
        simulation,
        state,
        parameters,
    })
}

#[test]
fn consumer_score_bounds_preserve_eight_offer_quota_ties_wrapping_and_rail_access_on_gpu()
-> Result<()> {
    for good in [Good::Coal, Good::Wood] {
        let mut quota =
            SearchProbe::with_suppliers(SupplyCase::Capped, 0, Market::Consumer(good), 9)?;
        quota.state.set(Field::HeatingNeed, 0, u64::from(u32::MAX));
        quota.state.set(Field::Cash, 0, u64::MAX);
        for actor in 1..quota.state.actors.len() {
            quota
                .state
                .set(Field::inventory(good), actor, u64::MAX / 1000);
        }
        let expected = quota.outcome(SearchExecution::Plane)?;
        assert_eq!(expected.proposal[ProposalLayout::ACTOR], 1);
        assert!(expected.proposal[ProposalLayout::QUANTITY_HIGH] > 0);
        for workgroup in [32, 256] {
            assert_eq!(
                quota.outcome_with_workgroup(SearchExecution::SpatialFuel, workgroup)?,
                expected
            );
        }
        for case in [
            DeliveryScoreCase::Tied,
            DeliveryScoreCase::QuoteWrap,
            DeliveryScoreCase::PenaltyWrap,
            DeliveryScoreCase::AccessWrap,
            DeliveryScoreCase::ScaledSpeedWrap,
        ] {
            for rotation in [0, 5] {
                let probe = delivery_score_probe(good, rotation, case)?;
                let expected = probe.outcome(SearchExecution::Plane)?;
                if rotation == 0
                    && matches!(
                        case,
                        DeliveryScoreCase::QuoteWrap | DeliveryScoreCase::AccessWrap
                    )
                {
                    assert_eq!(
                        expected.proposal[ProposalLayout::ACTOR],
                        probe.state.actors.len() as u32 - 8
                    );
                }
                for workgroup in [32, 256] {
                    assert_eq!(
                        probe.outcome_with_workgroup(SearchExecution::SpatialFuel, workgroup)?,
                        expected,
                        "{good:?}, {case:?}, rotation {rotation}, workgroup {workgroup}",
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
fn scalar_fuel_search_preserves_partial_dispatch_buffers_and_rotated_suppliers_on_gpu() -> Result<()>
{
    for good in [Good::Coal, Good::Wood] {
        for case in [
            SupplyCase::Capped,
            SupplyCase::Sparse,
            SupplyCase::Empty,
            SupplyCase::Unaffordable,
        ] {
            let mut probe = SearchProbe::new(case, 128, Market::Consumer(good))?;
            let expected = probe.outcome(SearchExecution::Shared)?;
            let expected_proposals = probe.simulation.device.read(&probe.simulation.proposals)?;
            let expected_successes = probe
                .simulation
                .device
                .read(&probe.simulation.searches.successes)?;
            for workgroup in [32, 64, 128, 256] {
                probe.simulation.device.workgroup = workgroup;
                assert_eq!(
                    probe.outcome_with_workgroup(SearchExecution::ScalarFuel, workgroup)?,
                    expected,
                    "{good:?}, {case:?}, workgroup {workgroup}",
                );
                assert_eq!(
                    probe.simulation.device.read(&probe.simulation.proposals)?,
                    expected_proposals,
                    "{good:?}, {case:?}, workgroup {workgroup}, proposals",
                );
                assert_eq!(
                    probe
                        .simulation
                        .device
                        .read(&probe.simulation.searches.successes)?,
                    expected_successes,
                    "{good:?}, {case:?}, workgroup {workgroup}, successes",
                );
            }
        }
        for case in [SupplyCase::Sparse, SupplyCase::Empty] {
            for day in [31, 1024, 2176] {
                let probe = SearchProbe::with_suppliers(case, day, Market::Consumer(good), 2177)?;
                let expected = probe.outcome(SearchExecution::Serial)?;
                for execution in [
                    SearchExecution::Shared,
                    SearchExecution::Plane,
                    SearchExecution::ScalarFuel,
                    SearchExecution::SpatialFuel,
                ] {
                    assert_eq!(
                        probe.outcome(execution)?,
                        expected,
                        "{good:?}, {case:?}, day {day}, {execution:?}",
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
fn single_cell_search_geometry_preserves_inconsistent_radius_fallback_on_gpu() -> Result<()> {
    for good in [Good::Coal, Good::Linen] {
        let mut probe =
            SearchProbe::with_suppliers(SupplyCase::Capped, 0, Market::Consumer(good), 193)?;
        let count = probe.state.actors.len();
        let map = probe.simulation.config.map_size;
        let mut actors = probe.simulation.device.read(&probe.simulation.actors)?;
        for actor in 0..count {
            let mut record = probe.state.actors.at(actor);
            record.position = Position {
                x: map - 1,
                y: map - 1,
            };
            actors[ActorLayout::X * count + actor] = record.position.x;
            actors[ActorLayout::Y * count + actor] = record.position.y;
            probe.state.actors.set(actor, record);
            probe.state.set(Field::Price, actor, 1);
        }
        probe
            .simulation
            .device
            .write(&probe.simulation.actors, &actors);
        for radius in [map, map.div_ceil(4)] {
            probe.simulation.config.search_radius = radius;
            for rotation in [0, 31, 192] {
                probe.parameters.settlement_rotation = rotation;
                let expected = probe.outcome(SearchExecution::Serial)?;
                assert_eq!(
                    expected.proposal[ProposalLayout::ACTOR] == NO_ACTOR,
                    radius < map,
                );
                for execution in [
                    SearchExecution::Shared,
                    SearchExecution::Plane,
                    SearchExecution::ScalarFuel,
                    SearchExecution::SpatialFuel,
                ] {
                    assert_eq!(
                        probe.outcome(execution)?,
                        expected,
                        "{good:?}, radius {radius}, rotation {rotation}, {execution:?}",
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
fn indexed_empty_suppliers_preserve_rotated_ties_across_cells_on_gpu() -> Result<()> {
    for good in [Good::Coal, Good::Cloth] {
        let mut probe =
            SearchProbe::with_suppliers(SupplyCase::Empty, 0, Market::Consumer(good), 511)?;
        let count = probe.state.actors.len();
        let map = probe.simulation.config.map_size;
        let radius = map.div_ceil(3);
        let side = map.div_ceil(radius);
        let cells = (side * side) as usize;
        let mut actors = probe.simulation.device.read(&probe.simulation.actors)?;
        for actor in 0..count {
            let mut record = probe.state.actors.at(actor);
            record.position = match actor {
                0 => Position {
                    x: radius - 1,
                    y: radius - 1,
                },
                _ => Position {
                    x: (actor as u32 * 97) % map,
                    y: (actor as u32 * 193) % map,
                },
            };
            actors[ActorLayout::X * count + actor] = record.position.x;
            actors[ActorLayout::Y * count + actor] = record.position.y;
            probe.state.actors.set(actor, record);
            probe.state.set(Field::Price, actor, 1);
        }
        probe
            .simulation
            .device
            .write(&probe.simulation.actors, &actors);
        probe.simulation.config.search_radius = radius;
        probe.simulation.config.spatial_side = side;
        for active in [0usize, 1, 511] {
            let mut members = (1..=active).collect::<Vec<_>>();
            let cell = |actor| {
                let position = probe.state.actors.at(actor).position;
                (position.y / radius * side + position.x / radius) as usize
            };
            members.sort_by_key(|&actor| (cell(actor), actor));
            let segment = cells + 1 + count;
            let mut spatial =
                vec![0; 3 * segment + count.div_ceil(DISCOVERY_BLOCK as usize) + 1 + count];
            for (position, &actor) in members.iter().enumerate() {
                spatial[cell(actor) + 1] += 1;
                spatial[cells + 1 + position] = actor as u32;
            }
            for cell in 1..=cells {
                spatial[cell] += spatial[cell - 1];
            }
            let query = 3 * segment + count.div_ceil(DISCOVERY_BLOCK as usize) + 1;
            spatial[query..query + count]
                .copy_from_slice(&(0..count as u32).rev().collect::<Vec<_>>());
            probe.simulation.spatial = probe.simulation.device.tensor(&spatial);
            for rotation in [0, 31, 128, 510, 1024] {
                probe.parameters.settlement_rotation = rotation;
                let expected = probe.outcome(SearchExecution::Serial)?;
                for execution in [SearchExecution::ScalarFuel, SearchExecution::SpatialFuel] {
                    assert_eq!(
                        probe.outcome(execution)?,
                        expected,
                        "{good:?}, active {active}, rotation {rotation}, {execution:?}",
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
fn supplier_cache_capacity_preserves_rotated_ties_at_the_inline_boundary_on_gpu() -> Result<()> {
    for (good, suppliers, fields, cached) in [
        (Good::Linen, 256, 18, 1),
        (Good::Linen, 257, 19, 1),
        (Good::Grain, 257, 18, 0),
    ] {
        let mut probe =
            SearchProbe::with_suppliers(SupplyCase::Empty, 0, Market::Consumer(good), suppliers)?;
        let count = probe.state.actors.len();
        assert_eq!(
            probe.simulation.searches.requests.meta.num_elements(),
            count * fields
        );
        probe.state.set(Field::inventory(good), 0, 0);
        for firm in 1..count {
            probe.state.set(Field::Price, firm, 1);
        }
        for rotation in [0, suppliers - 1, suppliers] {
            probe.parameters.settlement_rotation = rotation;
            let expected = probe.outcome(SearchExecution::Serial)?;
            assert_eq!(
                expected.proposal[ProposalLayout::ACTOR],
                rotation % suppliers + 1
            );
            let actual =
                probe.outcome_with_cache(SearchExecution::ScalarFuel, 64, CacheRefresh::Retain)?;
            assert_eq!(actual, expected);
            let header = probe.simulation.device.read_range(
                &probe.simulation.searches.requests,
                14 * count,
                1,
            )?[0];
            assert_eq!(header & 1, cached);
        }
    }
    Ok(())
}

#[test]
fn empty_supplier_cache_summary_preserves_empty_single_word_and_cross_word_ties_on_gpu()
-> Result<()> {
    #[derive(Clone, Copy, Debug)]
    enum Ties {
        Empty,
        FirstWord,
        LastWord,
        CrossWord,
    }

    for suppliers in [33u32, 193, 2048] {
        for ties in [
            Ties::Empty,
            Ties::FirstWord,
            Ties::LastWord,
            Ties::CrossWord,
        ] {
            let mut probe = SearchProbe::with_suppliers(
                SupplyCase::Empty,
                0,
                Market::Consumer(Good::Linen),
                suppliers,
            )?;
            let count = probe.state.actors.len();
            let last_word = (suppliers - 1) / 32;
            let mut actors = probe.simulation.device.read(&probe.simulation.actors)?;
            for actor in 0..count {
                let x = match (ties, actor) {
                    (_, 0) => 0,
                    (Ties::Empty, _) => probe.simulation.config.map_size - 1,
                    _ => 0,
                };
                let mut record = probe.state.actors.at(actor);
                record.position = Position { x, y: 0 };
                actors[ActorLayout::X * count + actor] = x;
                actors[ActorLayout::Y * count + actor] = 0;
                probe.state.actors.set(actor, record);
                let member = actor.saturating_sub(1) as u32;
                let price = match ties {
                    Ties::Empty => 1,
                    Ties::FirstWord => match member {
                        1 | 7 | 31 => 1,
                        _ => 10_000,
                    },
                    Ties::LastWord => match member / 32 == last_word {
                        true => 1,
                        false => 10_000,
                    },
                    Ties::CrossWord => match [0, suppliers - 1].contains(&member) {
                        true => 1,
                        false => 10_000,
                    },
                };
                probe.state.set(Field::Price, actor, price);
            }
            probe.simulation.config.search_radius = 1;
            probe
                .simulation
                .device
                .write(&probe.simulation.actors, &actors);
            let cases = [5000, 5000, 10_000, 5000]
                .into_iter()
                .flat_map(|cash| {
                    [0, 1, 7, 31, 32, suppliers - 1, suppliers, u32::MAX]
                        .into_iter()
                        .map(move |rotation| (cash, rotation))
                })
                .collect::<Vec<_>>();
            let mut expected = Vec::new();
            for &(cash, rotation) in &cases {
                probe.state.set(Field::Cash, 0, cash);
                probe.parameters.settlement_rotation = rotation;
                expected.push(probe.outcome(SearchExecution::Serial)?);
            }
            for ((cash, rotation), expected) in cases.into_iter().zip(expected) {
                probe.state.set(Field::Cash, 0, cash);
                probe.parameters.settlement_rotation = rotation;
                let actual = probe.outcome_with_cache(
                    SearchExecution::ScalarFuel,
                    64,
                    CacheRefresh::Retain,
                )?;
                assert_eq!(
                    actual, expected,
                    "{suppliers}, {ties:?}, {cash}, {rotation}"
                );
                let header = probe.simulation.device.read_range(
                    &probe.simulation.searches.requests,
                    14 * count,
                    1,
                )?[0];
                let summary = match ties {
                    Ties::Empty => 0,
                    Ties::CrossWord => 2,
                    _ => 1,
                };
                assert_eq!(header & 0xffff, 1);
                assert_eq!(header >> 30, summary);
                if let Ties::LastWord = ties {
                    assert_eq!((header >> 16) & 63, last_word);
                }
            }
        }
    }
    Ok(())
}

#[test]
fn empty_supplier_cache_tracks_stock_decreases_before_reachable_restocking_on_gpu() -> Result<()> {
    struct SupplierStocks {
        distant: u64,
        unaffordable: u64,
        reachable: u64,
        captured: u32,
    }

    let cases = [
        SupplierStocks {
            distant: 1,
            unaffordable: 1,
            reachable: 0,
            captured: 33,
        },
        SupplierStocks {
            distant: 0,
            unaffordable: 1,
            reachable: 0,
            captured: 17,
        },
        SupplierStocks {
            distant: 0,
            unaffordable: 0,
            reachable: 0,
            captured: 1,
        },
        SupplierStocks {
            distant: 0,
            unaffordable: 0,
            reachable: 1,
            captured: 1,
        },
    ];
    let mut probe =
        SearchProbe::with_suppliers(SupplyCase::Empty, 0, Market::Consumer(Good::Linen), 511)?;
    let count = probe.state.actors.len();
    let mut actors = probe.simulation.device.read(&probe.simulation.actors)?;
    for actor in 0..count {
        let mut record = probe.state.actors.at(actor);
        record.position = match actor {
            510 => Position {
                x: probe.simulation.config.map_size - 1,
                y: probe.simulation.config.map_size - 1,
            },
            _ => Position { x: 0, y: 0 },
        };
        actors[ActorLayout::X * count + actor] = record.position.x;
        actors[ActorLayout::Y * count + actor] = record.position.y;
        probe.state.actors.set(actor, record);
        probe.state.set(
            Field::Price,
            actor,
            match actor {
                511 => 10_000,
                _ => 1,
            },
        );
    }
    probe
        .simulation
        .device
        .write(&probe.simulation.actors, &actors);
    let mut expected = Vec::new();
    for stocks in &cases {
        probe.state.set(Field::Linen, 510, stocks.distant);
        probe.state.set(Field::Linen, 511, stocks.unaffordable);
        probe.state.set(Field::Linen, 1, stocks.reachable);
        expected.push(probe.outcome(SearchExecution::Serial)?);
    }
    for (stocks, expected) in cases.into_iter().zip(expected) {
        probe.state.set(Field::Linen, 510, stocks.distant);
        probe.state.set(Field::Linen, 511, stocks.unaffordable);
        probe.state.set(Field::Linen, 1, stocks.reachable);
        let actual =
            probe.outcome_with_cache(SearchExecution::ScalarFuel, 64, CacheRefresh::Retain)?;
        assert_eq!(actual, expected);
        let requests = probe
            .simulation
            .device
            .read(&probe.simulation.searches.requests)?;
        assert_eq!(requests[14 * count] & 0xffff, stocks.captured);
        if stocks.reachable > 0 {
            assert_eq!(actual.proposal[ProposalLayout::ACTOR], 1);
            assert_eq!(actual.proposal[ProposalLayout::QUANTITY_LOW], 1);
        }
    }
    Ok(())
}

#[test]
fn compacted_stocked_suppliers_preserve_original_rotation_order_on_gpu() -> Result<()> {
    let mut probe =
        SearchProbe::with_suppliers(SupplyCase::Sparse, 0, Market::Consumer(Good::Coal), 511)?;
    let stocked = [1usize, 2, 32, 33, 65, 129, 130, 257, 383, 385];
    let count = probe.state.actors.len();
    let map = probe.simulation.config.map_size;
    let mut actors = probe.simulation.device.read(&probe.simulation.actors)?;
    for actor in 1..count {
        let available = match stocked.binary_search(&actor) {
            Ok(_) => 100,
            Err(_) => 0,
        };
        probe.state.set(Field::Coal, actor, available);
        probe
            .state
            .set(Field::Price, actor, 1 + (511 - actor) as u64);
        let mut record = probe.state.actors.at(actor);
        record.position = Position {
            x: (actor as u32 * 97) % map,
            y: (actor as u32 * 193) % map,
        };
        actors[ActorLayout::X * count + actor] = record.position.x;
        actors[ActorLayout::Y * count + actor] = record.position.y;
        probe.state.actors.set(actor, record);
    }
    probe
        .simulation
        .device
        .write(&probe.simulation.actors, &actors);
    for cash in [3, 5000] {
        probe.state.set(Field::Cash, 0, cash);
        for rotation in [0, 31, 128, 257, 384, 1024] {
            probe.parameters.settlement_rotation = rotation;
            let expected = probe.outcome(SearchExecution::Serial)?;
            for execution in [SearchExecution::ScalarFuel, SearchExecution::SpatialFuel] {
                let actual = probe.outcome(execution)?;
                assert_eq!(
                    actual.proposal, expected.proposal,
                    "cash {cash}, rotation {rotation}, {execution:?}"
                );
                assert_eq!(actual.faults, expected.faults);
                assert!(actual.state == expected.state);
            }
        }
    }
    Ok(())
}

#[test]
fn global_supplier_price_floor_preserves_poor_and_wide_budgets_on_gpu() -> Result<()> {
    for good in [Good::Coal, Good::Cloth] {
        for case in [SupplyCase::Capped, SupplyCase::Empty, SupplyCase::Free] {
            let mut probe = SearchProbe::with_suppliers(case, 0, Market::Consumer(good), 211)?;
            for actor in 1..probe.state.actors.len() {
                let price = match case {
                    SupplyCase::Free => 0,
                    _ => 30 + actor as u64 % 2,
                };
                probe.state.set(Field::Price, actor, price);
            }
            probe.state.set(Field::Price, 1, 0);
            for cash in [0, 1, 29, 30, u64::from(u32::MAX) + 5000] {
                probe.state.set(Field::Cash, 0, cash);
                let expected = probe.outcome(SearchExecution::Shared)?;
                for execution in [SearchExecution::ScalarFuel, SearchExecution::SpatialFuel] {
                    assert_eq!(
                        probe.outcome(execution)?,
                        expected,
                        "{good:?}, {case:?}, cash {cash}, {execution:?}",
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
fn consumer_preparation_clears_inactive_requests_and_preserves_captured_retries_on_gpu()
-> Result<()> {
    let scenario = Scenario::industrial(129, 42, Experiment::Baseline)?;
    let device = device(64)?;
    let simulation = Simulation::new(scenario.clone(), device.clone())?;
    let count = scenario.actors();
    for good in Good::ALL
        .into_iter()
        .filter(|good| good.consumption_interval() > 1 || matches!(good, Good::Coal | Good::Wood))
    {
        let mut state = scenario.initial()?;
        let substitute = match good {
            Good::Coal => Good::Wood,
            _ => Good::Coal,
        };
        let firms = state.actors.firm_indices().collect::<Vec<_>>();
        for actor in firms {
            state.set(Field::inventory(substitute), actor, 0);
        }
        let households = (0..count)
            .filter(|&actor| state.actors.at(actor).kind == 0)
            .collect::<Vec<_>>();
        for &actor in &households {
            state.set(Field::LivingMembers, actor, 3);
            state.set(Field::HeatingNeed, actor, 9);
            state.set(Field::Bread, actor, 100_000);
            state.set(Field::Cash, actor, 1_000_000);
            state.set(Field::Coal, actor, 0);
            state.set(Field::Wood, actor, 0);
            state.set(Field::Clothes, actor, 1);
            state.set(Field::Furniture, actor, 1);
            state.set(Field::inventory(good), actor, 0);
        }
        let mut remaining = state.clone();
        for actor in households.into_iter().filter(|actor| actor % 3 == 0) {
            remaining.set(Field::inventory(good), actor, 1_000_000);
        }
        let mut expected = Vec::new();
        for (discovery, preparation) in [
            (Discovery::Linear, markets::PreparationWorkgroup::Native),
            (Discovery::Indexed, markets::PreparationWorkgroup::Native),
            (Discovery::Indexed, markets::PreparationWorkgroup::Wide),
        ] {
            let mut parameters = simulation.parameters(
                Phase::ProvisionPurchase,
                Market::Consumer(good),
                0,
                count,
            )?;
            parameters.discovery = discovery;
            device.write(
                &simulation.state,
                bytemuck::cast_slice(&encode(&state.values)),
            );
            device.write(&simulation.control, &vec![0; CONTROL_WORDS + 4]);
            device.write(&simulation.searches.requests, &vec![37; 6 * count]);
            device.write(
                &simulation.scratch,
                &vec![u32::MAX; markets::scratch_words(count)],
            );
            let mut observations = Vec::new();
            for stage in [SearchStage::PrepareCandidates, SearchStage::PrepareSelected] {
                let workgroup = preparation.width(&device);
                let dispatch = match stage {
                    SearchStage::PrepareSelected => {
                        markets::prepare_dispatch_for(
                            &device,
                            &simulation.control,
                            &simulation.searches.dispatches,
                            count,
                            preparation,
                        );
                        markets::dispatch(
                            &simulation.searches.dispatches,
                            markets::PREPARATION_DISPATCH,
                            count.div_ceil(workgroup as usize),
                        )
                    }
                    _ => cubecl::calculate_cube_count_elemwise(
                        &device.client,
                        count,
                        CubeDim::new_1d(workgroup),
                    )
                    .into(),
                };
                let resets = markets::search_counts_offset(count) + markets::search_blocks(count);
                device.write(&simulation.searches.successes, &vec![37; resets]);
                MarketSearch::launch(
                    &device,
                    dispatch,
                    CubeDim::new_1d(workgroup),
                    SearchInputLaunch {
                        state: simulation.state.buffer_arg(),
                        actors: simulation.actors.buffer_arg(),
                        config: simulation.config.launch(),
                        spatial: simulation.spatial.buffer_arg(),
                        proposals: simulation.proposals.buffer_arg(),
                        searches: simulation.searches.requests.buffer_arg(),
                        successes: simulation.searches.successes.buffer_arg(),
                        scratch: simulation.scratch.buffer_arg(),
                        control: simulation.control.buffer_arg(),
                        parameters: parameters.launch(),
                        status_reset: StatusReset::Full,
                        program: stage.program(&simulation.config),
                    },
                );
                assert_eq!(
                    device.read_range(&simulation.searches.successes, 0, resets)?,
                    vec![0; resets],
                );
                let mut control = device.read(&simulation.control)?;
                let requests = device.read(&simulation.searches.requests)?;
                let mut actors = requests[..control[SEARCHES_WORD] as usize].to_vec();
                let mut candidates = requests
                    [5 * count..5 * count + control[markets::PREPARATION_ACTORS_WORD] as usize]
                    .to_vec();
                actors.sort_unstable();
                candidates.sort_unstable();
                for actor in 0..count {
                    match actors.binary_search(&(actor as u32)) {
                        Ok(_) => assert!(requests[count + actor] > 0),
                        Err(_) => {
                            assert_eq!(requests[count + actor], 0);
                            assert_eq!(requests[2 * count + actor], 0);
                        }
                    }
                }
                observations.push(PreparedRequests {
                    actors,
                    demand: requests[count..5 * count].to_vec(),
                    candidates,
                    statuses: device.read_range(&simulation.scratch, 4 * count, count)?,
                    faults: control[crate::simulation::flags::FAULT_WORD],
                });
                control[SEARCHES_WORD] = 0;
                device.write(&simulation.control, &control);
                device.write(
                    &simulation.state,
                    bytemuck::cast_slice(&encode(&remaining.values)),
                );
            }
            match discovery {
                Discovery::Linear => expected = observations,
                _ => assert_eq!(observations, expected, "{good:?}"),
            }
        }
        assert!(!expected[0].actors.is_empty());
        assert!(expected[1].actors.len() < expected[0].actors.len());
        assert_eq!(expected[1].candidates, expected[0].candidates);
    }
    Ok(())
}

#[derive(CubeType, CubeLaunch)]
struct DeliveryFloorInput {
    actors: Box<[u32]>,
    config: EconomyConfig,
    findings: Box<[Atomic<u32>]>,
}

struct DeliveryFloorCheck;
struct CompareDeliveryFloor;

impl System for DeliveryFloorCheck {
    type Input = DeliveryFloorInput;
    type Passes = gpu_passes![CompareDeliveryFloor];
}

#[cube]
impl GpuPass<DeliveryFloorInput> for CompareDeliveryFloor {
    type Output = ();

    fn execute(input: DeliveryFloorInput) {
        let index = ABSOLUTE_POS as u32;
        let count = input.config.actors;
        if index < count * count {
            let buyer = index / count;
            let seller = index % count;
            let trip = distance(&input.actors, buyer, seller);
            let mut access = 0u32;
            if comptime!(input.config.railway.routes_enabled == Feature::Enabled)
                && input.config.railway.nodes > 0
            {
                access = input.config.railway.access[buyer as usize * 2 + 1];
            }
            let floor = minimum_delivery_days(&input.config, access, seller, trip);
            let route = railway::route(&input.actors, &input.config, buyer, seller);
            let days = railway::delivery_days(
                &input.config,
                railway::DeliveryRequest {
                    buyer,
                    seller,
                    trip,
                    buyer_access: access,
                },
            );
            if route.days < max(ceildiv(trip, input.config.shipment_speed), 1) {
                input.findings[3usize].fetch_add(1);
            }
            if days != route.days {
                input.findings[2usize].fetch_add(1);
            }
            if floor > route.days {
                input.findings[0usize].fetch_add(1);
            }
            let previous = max(
                ceildiv(
                    trip,
                    input.config.shipment_speed * input.config.railway.speed_multiplier,
                ),
                1,
            );
            if floor > previous {
                input.findings[1usize].fetch_add(1);
            }
        }
    }
}

#[test]
fn station_access_floor_never_exceeds_selected_route_days_on_gpu() -> Result<()> {
    let mut rail: crate::domain::ScenarioConfig =
        serde_json::from_str(include_str!("../../../../scenarios/railway.json"))?;
    let build = rail.railway.construction.quantities();
    let population = rail
        .population
        .as_mut()
        .expect("explicit railway population");
    for firm in population
        .firms
        .iter_mut()
        .filter(|firm| firm.good == Good::Railway)
    {
        for (good, amount) in [Good::Coal, Good::Steel, Good::Locomotives]
            .into_iter()
            .zip(build)
        {
            firm.inventories[good as usize - 1] = u64::from(amount);
        }
    }
    let device = device(64)?;
    let mut improved = 0;
    let mut faster_routes = 0;
    for scenario in [
        Scenario::industrial(193, 42, Experiment::Baseline)?,
        Scenario::industrial(193, 42, Experiment::Transport)?,
        Scenario::new(rail)?,
    ] {
        let simulation = Simulation::new(scenario, device.clone())?;
        for stage in [
            railway::RailStage::Infrastructure,
            railway::RailStage::Access,
            railway::RailStage::Routes,
            railway::RailStage::Quotes,
        ] {
            simulation.railways(stage);
        }
        let findings = device.tensor(&[0; 4]);
        let count = simulation.metadata.actors.len();
        DeliveryFloorCheck::launch_elements(
            &device,
            count * count,
            DeliveryFloorInputLaunch {
                actors: simulation.actors.buffer_arg(),
                config: simulation.config.launch(),
                findings: findings.buffer_arg(),
            },
        );
        let findings = device.read(&findings)?;
        assert_eq!(findings[0], 0);
        assert_eq!(findings[2], 0);
        improved += findings[1];
        faster_routes += findings[3];
    }
    assert!(improved > 0);
    assert!(faster_routes > 0);
    Ok(())
}

#[test]
fn filtered_empty_consumer_orders_preserve_state_feedback_and_diagnostics_on_gpu() -> Result<()> {
    #[derive(Clone, Copy, Debug)]
    enum Orders {
        Empty,
        Mixed,
        Wide,
        BuyerOverflow,
        SellerOverflow,
    }

    let households = 137usize;
    let scenario = Scenario::industrial(households as u32, 42, Experiment::Baseline)?;
    let device = device(64)?;
    for orders in [
        Orders::Empty,
        Orders::Mixed,
        Orders::Wide,
        Orders::BuyerOverflow,
        Orders::SellerOverflow,
    ] {
        let good = match orders {
            Orders::Wide => Good::Coal,
            _ => Good::Linen,
        };
        let mut state = scenario.initial()?;
        for actor in 0..households {
            state.set(Field::LivingMembers, actor, 3);
            state.set(Field::FoodNeed, actor, 3);
            state.set(Field::NutritionNeed, actor, 3);
            state.set(Field::FoodPrice, actor, 1);
            state.set(Field::Bread, actor, 100_000);
            state.set(Field::inventory(good), actor, 0);
            state.set(Field::Wood, actor, 0);
            state.set(
                Field::Cash,
                actor,
                match orders {
                    Orders::Wide => u64::MAX,
                    _ => 1_000_000,
                },
            );
            state.set(
                Field::HeatingNeed,
                actor,
                match orders {
                    Orders::Wide => u64::from(u32::MAX),
                    _ => 9,
                },
            );
            state.set(
                Field::Purchased,
                actor,
                match orders {
                    Orders::BuyerOverflow => u64::MAX,
                    _ => 0,
                },
            );
        }
        let peers = state
            .actors
            .firm_indices()
            .filter(|&peer| state.actors.at(peer).kind == good as u32)
            .collect::<Vec<_>>();
        assert!(!peers.is_empty());
        for peer in households..state.actors.len() {
            state.set(Field::inventory(good), peer, 0);
            state.set(Field::Wood, peer, 0);
            state.set(Field::Price, peer, 1);
            state.set(Field::Sales, peer, 0);
            state.set(
                Field::Cash,
                peer,
                match orders {
                    Orders::SellerOverflow => u64::MAX,
                    _ => 1000,
                },
            );
        }
        if let Orders::Mixed = orders {
            state.set(Field::inventory(good), peers[0], 3);
        }
        let initialize = || -> Result<Simulation> {
            let simulation = Simulation::new(scenario.clone(), device.clone())?;
            simulation.device.write(
                &simulation.state,
                bytemuck::cast_slice(&encode(&state.values)),
            );
            Ok(simulation)
        };
        let mut full = initialize()?;
        let mut filtered = initialize()?;
        let mut removed = 0u32;
        for round in 0..4 {
            full.matching_round = round;
            filtered.matching_round = round;
            let mut full_evidence = crate::domain::PhaseEvidence::default();
            let mut filtered_evidence = crate::domain::PhaseEvidence::default();
            let reference = full
                .device_market(
                    Market::Consumer(good),
                    Phase::ProvisionPurchase,
                    Trace::Record,
                    &mut full_evidence,
                )
                .and_then(|_| full.fault_check());
            let actual = filtered
                .device_market(
                    Market::Consumer(good),
                    Phase::ProvisionPurchase,
                    Trace::Discard,
                    &mut filtered_evidence,
                )
                .and_then(|_| filtered.fault_check());
            assert_eq!(
                actual.is_err(),
                reference.is_err(),
                "{orders:?}, round {round}"
            );
            let expected = device.read_range(&full.control, 0, 4)?;
            let observed = device.read_range(&filtered.control, 0, 4)?;
            let faults = match orders {
                Orders::BuyerOverflow | Orders::SellerOverflow => {
                    FaultFlags::ARITHMETIC_OVERFLOW.bits()
                }
                _ => 0,
            };
            assert_eq!(
                expected[crate::simulation::flags::FAULT_WORD],
                faults,
                "{orders:?}, round {round}"
            );
            assert_eq!(reference.is_err(), faults != 0, "{orders:?}, round {round}");
            assert_eq!(&observed[..3], &expected[..3], "{orders:?}, round {round}");
            assert_eq!(
                device.read(&filtered.state)?,
                device.read(&full.state)?,
                "{orders:?}, round {round}"
            );
            let expected_feedback = device.read(&full.feedback)?;
            let actual_feedback = device.read(&filtered.feedback)?;
            assert_eq!(
                actual_feedback[crate::simulation::flags::FAULT_WORD],
                expected_feedback[crate::simulation::flags::FAULT_WORD]
            );
            assert_eq!(
                &actual_feedback[CONTROL_WORDS..],
                &expected_feedback[CONTROL_WORDS..],
                "{orders:?}, round {round}"
            );
            removed += expected[CONSTRUCTED_EVENTS_WORD] - observed[CONSTRUCTED_EVENTS_WORD];
            if let (Orders::Wide, 3) = (orders, round) {
                assert_eq!(
                    device.read_range(&filtered.control, markets::WIDE_ORDERS_WORD, 1)?,
                    [1]
                );
                assert_eq!(
                    observed[CONSTRUCTED_EVENTS_WORD],
                    expected[CONSTRUCTED_EVENTS_WORD]
                );
            }
        }
        if let Orders::Empty | Orders::Mixed = orders {
            assert!(removed > 0, "{orders:?}");
        }
    }
    Ok(())
}

#[derive(Debug, PartialEq, Eq)]
struct PreparedInputs {
    queue: Vec<u32>,
    payload: Vec<u32>,
    statuses: Vec<u32>,
    faults: u32,
}

impl SearchProbe {
    fn clear_preparation(&self) {
        let simulation = &self.simulation;
        for buffer in [
            &simulation.control,
            &simulation.searches.requests,
            &simulation.searches.successes,
        ] {
            simulation
                .device
                .write(buffer, &vec![0; buffer.meta.num_elements()]);
        }
        simulation.device.write(
            &simulation.scratch,
            &vec![u32::MAX; simulation.scratch.meta.num_elements()],
        );
    }

    fn prepared_inputs(
        &self,
        initialization: EconomyStage,
        stage: SearchStage,
    ) -> Result<PreparedInputs> {
        let simulation = &self.simulation;
        let device = &simulation.device;
        let count = self.state.actors.len();
        device.write(
            &simulation.state,
            bytemuck::cast_slice(&encode(&self.state.values)),
        );
        EconomySystem::launch_elements(
            device,
            count,
            EconomyInputLaunch {
                state: simulation.state.buffer_arg(),
                actors: simulation.actors.buffer_arg(),
                config: simulation.config.launch(),
                spatial: simulation.spatial.buffer_arg(),
                proposals: simulation.proposals.buffer_arg(),
                bundles: simulation.scratch.buffer_arg(),
                control: simulation.control.buffer_arg(),
                parameters: self.parameters.launch(),
                stage: initialization,
            },
        );
        let dispatch = match stage {
            SearchStage::PrepareSelected => markets::dispatch(
                &simulation.searches.dispatches,
                markets::PREPARATION_DISPATCH,
                count.div_ceil(device.workgroup as usize),
            ),
            _ => device.count(count).into(),
        };
        MarketSearch::launch(
            device,
            dispatch,
            CubeDim::new_1d(device.workgroup),
            SearchInputLaunch {
                state: simulation.state.buffer_arg(),
                actors: simulation.actors.buffer_arg(),
                config: simulation.config.launch(),
                spatial: simulation.spatial.buffer_arg(),
                proposals: simulation.proposals.buffer_arg(),
                searches: simulation.searches.requests.buffer_arg(),
                successes: simulation.searches.successes.buffer_arg(),
                scratch: simulation.scratch.buffer_arg(),
                control: simulation.control.buffer_arg(),
                parameters: self.parameters.launch(),
                status_reset: StatusReset::Active,
                program: stage.program(&simulation.config),
            },
        );
        markets::prepare_dispatch(
            device,
            &simulation.control,
            &simulation.searches.dispatches,
            count,
        );
        let control = device.read(&simulation.control)?;
        let requests = device.read(&simulation.searches.requests)?;
        let mut queue = requests[..control[SEARCHES_WORD] as usize].to_vec();
        queue.sort_unstable();
        Ok(PreparedInputs {
            queue,
            payload: requests[count..5 * count].to_vec(),
            statuses: device.read_range(&simulation.scratch, 4 * count, count)?,
            faults: control[crate::simulation::flags::FAULT_WORD],
        })
    }
}

#[derive(Clone, Copy, Debug)]
enum PreparationEligibility {
    InactiveSeller,
    GrainCashless,
    GrainStocked,
    Investment,
    Building,
    Distributor,
}

#[test]
fn sparse_input_preparation_preserves_stable_eligibility_and_all_round_requests_on_gpu()
-> Result<()> {
    let workgroup = 64;
    for case in [
        PreparationEligibility::InactiveSeller,
        PreparationEligibility::GrainCashless,
        PreparationEligibility::GrainStocked,
        PreparationEligibility::Investment,
        PreparationEligibility::Building,
        PreparationEligibility::Distributor,
    ] {
        let mut dense =
            SearchProbe::with_suppliers(SupplyCase::Capped, 0, Market::Input(Good::Linen), 65)?;
        let mut sparse =
            SearchProbe::with_suppliers(SupplyCase::Capped, 0, Market::Input(Good::Linen), 65)?;
        let good = match case {
            PreparationEligibility::InactiveSeller => Good::Bread,
            PreparationEligibility::GrainCashless | PreparationEligibility::GrainStocked => {
                Good::Grain
            }
            PreparationEligibility::Investment => Good::Tools,
            PreparationEligibility::Building => Good::Construction,
            PreparationEligibility::Distributor => Good::Linen,
        };
        for probe in [&mut dense, &mut sparse] {
            probe.simulation.device.workgroup = workgroup;
            probe.parameters.good = good as u32;
            probe.state.set(Field::Cash, 0, 0);
            match case {
                PreparationEligibility::InactiveSeller => {
                    probe.state.set(Field::LivingMembers, 0, 0);
                    probe.state.set(Field::Bread, 0, 100);
                }
                PreparationEligibility::GrainCashless | PreparationEligibility::GrainStocked => {
                    probe.state.set(Field::LandPlots, 0, 1)
                }
                PreparationEligibility::Investment => probe.state.set(Field::InvestmentGood, 0, 1),
                PreparationEligibility::Building => {
                    probe.state.set(
                        Field::ProjectStatus,
                        0,
                        crate::domain::property::ProjectStatus::Building as u64,
                    );
                    probe.state.set(Field::ConstructionRemaining, 0, 7);
                }
                PreparationEligibility::Distributor => {
                    probe.state.set(Field::EnterpriseModel, 0, 1)
                }
            }
            if let PreparationEligibility::GrainStocked = case {
                probe.state.set(Field::Grain, 0, 10_000);
                probe.state.set(Field::Cash, 0, 5_000);
            }
            probe.clear_preparation();
        }
        let expected =
            dense.prepared_inputs(EconomyStage::InputMarketStart, SearchStage::Prepare)?;
        assert_eq!(
            sparse.prepared_inputs(
                EconomyStage::InputMarketStart,
                SearchStage::PrepareCandidates
            )?,
            expected,
            "{case:?}, workgroup {workgroup}, first"
        );
        let count = sparse.state.actors.len();
        let control = sparse.simulation.device.read(&sparse.simulation.control)?;
        let requests = sparse
            .simulation
            .device
            .read(&sparse.simulation.searches.requests)?;
        let mut eligible = requests
            [5 * count..5 * count + control[markets::PREPARATION_ACTORS_WORD] as usize]
            .to_vec();
        eligible.sort_unstable();
        let first = match case {
            PreparationEligibility::InactiveSeller => 1,
            _ => 0,
        };
        assert_eq!(eligible, (first..count as u32).collect::<Vec<_>>());
        for round in 1..4 {
            for probe in [&mut dense, &mut sparse] {
                probe.state.set(Field::Cash, 0, 5_000);
                probe.state.set(Field::inventory(good), 0, 0);
            }
            let expected =
                dense.prepared_inputs(EconomyStage::InputMarketRepeat, SearchStage::Prepare)?;
            assert_eq!(
                sparse.prepared_inputs(
                    EconomyStage::InputMarketRepeat,
                    SearchStage::PrepareSelected
                )?,
                expected,
                "{case:?}, workgroup {workgroup}, round {round}"
            );
        }
        if !matches!(case, PreparationEligibility::InactiveSeller) {
            for probe in [&mut dense, &mut sparse] {
                probe.state.set(Field::inventory(good), 0, u64::MAX);
                probe.state.values[(TRANSIT_FIELDS + good as usize - 1) * count] = 1;
            }
            let expected =
                dense.prepared_inputs(EconomyStage::InputMarketRepeat, SearchStage::Prepare)?;
            assert_ne!(expected.faults, 0);
            assert_eq!(
                sparse.prepared_inputs(
                    EconomyStage::InputMarketRepeat,
                    SearchStage::PrepareSelected
                )?,
                expected,
                "{case:?}, workgroup {workgroup}, repeated demand overflow"
            );
        }
    }
    Ok(())
}

#[test]
fn grain_capture_skips_stable_household_nonbuyers_and_keeps_firm_retries_on_gpu() -> Result<()> {
    #[derive(Clone, Copy, Debug)]
    enum HouseholdDemand {
        FullStock,
        Cashless,
        Searching,
    }

    for workgroup in [64, 256] {
        for demand in [
            HouseholdDemand::FullStock,
            HouseholdDemand::Cashless,
            HouseholdDemand::Searching,
        ] {
            let mut dense =
                SearchProbe::with_suppliers(SupplyCase::Capped, 0, Market::Input(Good::Grain), 65)?;
            let mut sparse =
                SearchProbe::with_suppliers(SupplyCase::Capped, 0, Market::Input(Good::Grain), 65)?;
            for probe in [&mut dense, &mut sparse] {
                probe.simulation.device.workgroup = workgroup;
                probe.simulation.config.firm_rent = 0;
                probe.state.set(Field::LandPlots, 0, 1);
                probe.state.set(
                    Field::Grain,
                    0,
                    match demand {
                        HouseholdDemand::FullStock => 10_000,
                        _ => 0,
                    },
                );
                probe.state.set(
                    Field::Cash,
                    0,
                    match demand {
                        HouseholdDemand::Cashless => 0,
                        _ => 5000,
                    },
                );
                for (field, value) in [
                    (Field::EnterpriseModel, 1),
                    (Field::Demand, 10),
                    (Field::StockCoverage, 3),
                    (Field::Cash, 0),
                    (Field::Grain, 0),
                    (Field::Wage, 0),
                    (Field::LaborMinutes, 0),
                ] {
                    probe.state.set(field, 1, value);
                }
                probe.clear_preparation();
            }
            let expected =
                dense.prepared_inputs(EconomyStage::InputMarketStart, SearchStage::Prepare)?;
            assert_eq!(
                sparse.prepared_inputs(
                    EconomyStage::InputMarketStart,
                    SearchStage::PrepareCandidates,
                )?,
                expected,
            );
            let count = sparse.state.actors.len();
            let control = sparse.simulation.device.read(&sparse.simulation.control)?;
            let requests = sparse
                .simulation
                .device
                .read(&sparse.simulation.searches.requests)?;
            let mut captured = requests
                [5 * count..5 * count + control[markets::PREPARATION_ACTORS_WORD] as usize]
                .to_vec();
            captured.sort_unstable();
            let first = match demand {
                HouseholdDemand::Searching => 0,
                _ => 1,
            };
            assert_eq!(captured, (first..count as u32).collect::<Vec<_>>());
            for round in 1..4 {
                for probe in [&mut dense, &mut sparse] {
                    probe.state.set(Field::Cash, 1, 5000);
                    if let HouseholdDemand::Searching = demand {
                        let transit =
                            3 * u64::from(
                                probe.simulation.config.coverage + probe.simulation.config.ring,
                            ) * round
                                / 3;
                        probe.state.values[(TRANSIT_FIELDS + Good::Grain as usize - 1) * count] =
                            transit;
                        probe.state.set(Field::Transit, 0, transit);
                        probe.state.set(Field::Cash, 0, 5000 - 10 * transit);
                    }
                }
                let expected =
                    dense.prepared_inputs(EconomyStage::InputMarketRepeat, SearchStage::Prepare)?;
                assert!(expected.queue.contains(&1));
                assert_eq!(
                    sparse.prepared_inputs(
                        EconomyStage::InputMarketRepeat,
                        SearchStage::PrepareSelected,
                    )?,
                    expected,
                    "{demand:?}, WG{workgroup}, round {round}",
                );
            }
        }
    }
    Ok(())
}

#[test]
fn sparse_input_preparation_preserves_first_round_inventory_faults_on_gpu() -> Result<()> {
    let workgroup = 64;
    let mut probe =
        SearchProbe::with_suppliers(SupplyCase::Capped, 0, Market::Input(Good::Linen), 65)?;
    probe.simulation.device.workgroup = workgroup;
    probe.state.set(Field::LivingMembers, 0, 0);
    probe.state.set(Field::Linen, 0, u64::MAX);
    let count = probe.state.actors.len();
    probe.state.values[(TRANSIT_FIELDS + Good::Linen as usize - 1) * count] = 1;
    probe.clear_preparation();
    let first = probe.prepared_inputs(
        EconomyStage::InputMarketStart,
        SearchStage::PrepareCandidates,
    )?;
    assert_ne!(first.faults, 0);
    let sentinel = 0x1234_5678;
    probe.simulation.device.write(
        &probe.simulation.scratch,
        &vec![sentinel; probe.simulation.scratch.meta.num_elements()],
    );
    let repeated = probe.prepared_inputs(
        EconomyStage::InputMarketRepeat,
        SearchStage::PrepareSelected,
    )?;
    assert_eq!(repeated.faults, first.faults);
    assert_eq!(repeated.statuses, vec![sentinel; count]);
    assert!(repeated.queue.is_empty());
    Ok(())
}

#[test]
fn selected_fuel_retries_include_buyers_with_only_freezing_reserves_on_gpu() -> Result<()> {
    for good in [Good::Coal, Good::Wood] {
        let mut dense =
            SearchProbe::with_suppliers(SupplyCase::Capped, 0, Market::Consumer(good), 65)?;
        let mut selected =
            SearchProbe::with_suppliers(SupplyCase::Capped, 0, Market::Consumer(good), 65)?;
        let units = match good {
            Good::Coal => 2,
            _ => 1,
        };
        let horizon =
            u64::from(dense.simulation.config.coverage + dense.simulation.config.ring - 1);
        let freezing = 2 * horizon / units;
        let comfort = 4 * horizon / units;
        for probe in [&mut dense, &mut selected] {
            probe.simulation.config.food_income = crate::domain::food::FoodIncome::PurchasingPower;
            probe.simulation.config.heating_per_person = 4;
            probe.state.set(Field::HeatingNeed, 0, 4);
            probe.state.set(Field::LivingMembers, 0, 1);
            probe.state.set(Field::FoodNeed, 0, 1);
            probe.state.set(Field::Bread, 0, 100);
            probe.state.set(Field::Cash, 0, 5_000);
            probe.state.set(Field::Coal, 0, 0);
            probe.state.set(Field::Wood, 0, 0);
            probe.state.set(Field::inventory(good), 0, freezing / 2);
            let count = probe.state.actors.len();
            let delivery = freezing - freezing / 2;
            probe.state.values[(TRANSIT_FIELDS + good as usize - 1) * count] = delivery;
            probe.state.set(Field::Transit, 0, delivery);
            probe.clear_preparation();
        }
        let expected =
            dense.prepared_inputs(EconomyStage::InputMarketStart, SearchStage::Prepare)?;
        assert!(expected.queue.is_empty());
        assert_eq!(expected.faults, 0);
        assert_eq!(
            selected.prepared_inputs(
                EconomyStage::InputMarketStart,
                SearchStage::PrepareCandidates,
            )?,
            expected,
            "{good:?} initial freezing demand",
        );
        let count = selected.state.actors.len();
        let control = selected
            .simulation
            .device
            .read(&selected.simulation.control)?;
        assert_eq!(control[markets::PREPARATION_ACTORS_WORD], 1);
        assert_eq!(
            selected.simulation.device.read_range(
                &selected.simulation.searches.requests,
                5 * count,
                1,
            )?,
            [0],
        );
        for round in 1..MATCHING_ROUNDS {
            for probe in [&mut dense, &mut selected] {
                probe.parameters.matching_round = round as u32;
            }
            let expected =
                dense.prepared_inputs(EconomyStage::InputMarketRepeat, SearchStage::Prepare)?;
            let actual = selected.prepared_inputs(
                EconomyStage::InputMarketRepeat,
                SearchStage::PrepareSelected,
            )?;
            assert_eq!(actual, expected, "{good:?} round {round}");
            match round + 1 == MATCHING_ROUNDS {
                true => {
                    assert_eq!(actual.queue, [0]);
                    let wanted =
                        u64::from(actual.payload[0]) | (u64::from(actual.payload[count]) << 32);
                    assert_eq!(wanted, comfort - freezing);
                }
                false => assert!(actual.queue.is_empty()),
            }
        }
    }
    Ok(())
}

#[test]
fn selected_fuel_buyers_match_dense_requests_and_reset_trace_statuses_on_gpu() -> Result<()> {
    let households = 97usize;
    let scenario = Scenario::baseline(households as u32, 42, Experiment::Concentrated)?;
    for good in [Good::Coal, Good::Wood] {
        let units = match good {
            Good::Coal => 2,
            _ => 1,
        };
        let target = 3 * (u64::from(scenario.config().coverage) + scenario.ring() as u64 - 1);
        let reserve = target.div_ceil(units);
        let probe = || -> Result<SearchProbe> {
            let simulation = Simulation::new(scenario.clone(), device(64)?)?;
            let mut state = scenario.initial()?;
            for household in 0..households {
                state.set(Field::HeatingNeed, household, 3);
                state.set(Field::FoodNeed, household, 3);
                state.set(Field::LivingMembers, household, 3);
                state.set(
                    Field::Cash,
                    household,
                    match household % 4 {
                        1 => 0,
                        _ => 5_000,
                    },
                );
                state.set(
                    Field::Bread,
                    household,
                    match household % 4 {
                        3 => 0,
                        _ => 100,
                    },
                );
                state.set(
                    Field::inventory(good),
                    household,
                    match household % 4 {
                        2 => reserve,
                        _ => 0,
                    },
                );
            }
            for firm in households..state.actors.len() {
                state.set(Field::inventory(good), firm, 1_000);
            }
            let parameters = simulation.parameters(
                Phase::ProvisionPurchase,
                Market::Consumer(good),
                0,
                state.actors.len(),
            )?;
            let probe = SearchProbe {
                simulation,
                state,
                parameters,
            };
            probe.clear_preparation();
            Ok(probe)
        };
        let mut dense = probe()?;
        let mut selected = probe()?;
        let expected =
            dense.prepared_inputs(EconomyStage::InputMarketStart, SearchStage::Prepare)?;
        assert_eq!(
            selected.prepared_inputs(
                EconomyStage::InputMarketStart,
                SearchStage::PrepareCandidates,
            )?,
            expected,
        );
        let count = selected.state.actors.len();
        let control = selected
            .simulation
            .device
            .read(&selected.simulation.control)?;
        let requests = selected
            .simulation
            .device
            .read(&selected.simulation.searches.requests)?;
        let mut captured = requests
            [5 * count..5 * count + control[markets::PREPARATION_ACTORS_WORD] as usize]
            .to_vec();
        captured.sort_unstable();
        assert_eq!(
            captured,
            (0..households as u32).step_by(4).collect::<Vec<_>>()
        );
        for round in 1..4u64 {
            for probe in [&mut dense, &mut selected] {
                for household in (0..households).step_by(4) {
                    probe.state.set(
                        Field::Cash,
                        household,
                        match round {
                            3 => 0,
                            _ => 5_000 - round * 1_000,
                        },
                    );
                    let delivered = match round {
                        1 => reserve / 2,
                        _ => reserve,
                    };
                    probe.state.values[(TRANSIT_FIELDS + good as usize - 1) * count + household] =
                        delivered;
                    probe.state.set(Field::Transit, household, delivered);
                }
                for firm in households..count {
                    probe.state.set(Field::Cash, firm, 100_000 + round * 100);
                    probe.state.set(Field::Sales, firm, round * 10);
                    probe
                        .state
                        .set(Field::inventory(good), firm, 1_000 - round * 10);
                }
                let simulation = &probe.simulation;
                simulation.device.write(
                    &simulation.scratch,
                    &vec![u32::MAX; simulation.scratch.meta.num_elements()],
                );
                simulation.device.write(
                    &simulation.searches.successes,
                    &vec![u32::MAX; simulation.searches.successes.meta.num_elements()],
                );
            }
            let expected =
                dense.prepared_inputs(EconomyStage::InputMarketRepeat, SearchStage::Prepare)?;
            assert_eq!(
                selected.prepared_inputs(
                    EconomyStage::InputMarketRepeat,
                    SearchStage::PrepareSelected,
                )?,
                expected,
                "{good:?} round {round}",
            );
            let resets = count.div_ceil(32) + markets::search_blocks(count);
            assert_eq!(
                selected.simulation.device.read_range(
                    &selected.simulation.searches.successes,
                    0,
                    resets,
                )?,
                vec![0; resets],
            );
        }
    }
    Ok(())
}

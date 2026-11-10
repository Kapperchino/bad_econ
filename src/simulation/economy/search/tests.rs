use super::*;
use crate::{
    domain::{
        Experiment, FirmSeed, HouseholdSeed, Phase, Population, Position, Scenario, State,
        Technique, enterprise::HouseholdProduction,
    },
    simulation::{Market, Simulation, Trace, encode, parallel::ScanMode, tests::device},
};
use anyhow::Result;

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
    PrepareRepeat,
    Shared,
    Plane,
}

#[derive(Debug, PartialEq, Eq)]
struct SearchOutcome {
    proposal: Vec<u32>,
    state: Vec<u32>,
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
        simulation.device.write(&simulation.spatial, &spatial);
        Ok(Self {
            simulation,
            state,
            parameters,
        })
    }

    fn outcome(&self, execution: SearchExecution) -> Result<SearchOutcome> {
        self.outcome_with_workgroup(execution, 32)
    }

    fn outcome_with_workgroup(
        &self,
        execution: SearchExecution,
        workgroup: u32,
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
        let mut control = vec![0; CONTROL_WORDS + 3];
        control[SEARCHES_WORD] = 1;
        device.write(&simulation.control, &control);
        device.write(
            &simulation.proposals,
            &vec![NO_ACTOR; count * ProposalLayout::WORDS],
        );
        let mut searches = vec![0; count * 5];
        let coverage = match parameters.market {
            MarketFlag::Input(_) if parameters.good == Good::Grain as u32 => {
                simulation.config.coverage + simulation.config.ring
            }
            _ => simulation.config.coverage + simulation.config.ring - 1,
        };
        searches[count] = match parameters.market {
            MarketFlag::Consumer => 1,
            _ => (self.state.get(Field::FoodNeed, 0) * u64::from(coverage)) as u32,
        };
        searches[3 * count] = self.state.get(Field::Cash, 0) as u32;
        searches[4 * count] = (self.state.get(Field::Cash, 0) >> u32::BITS) as u32;
        device.write(&simulation.searches.requests, &searches);
        device.write(
            &simulation.searches.successes,
            &vec![
                0;
                markets::search_counts_offset(count) + 2 * markets::search_blocks(count) + 512
            ],
        );
        if matches!(
            parameters.market,
            MarketFlag::Consumer | MarketFlag::Household
        ) {
            MarketSearch::launch(
                device,
                device.count(count.div_ceil(DISCOVERY_BLOCK as usize) * device.workgroup as usize),
                CubeDim::new_1d(device.workgroup),
                SearchInputLaunch {
                    state: simulation.state.clone().into_buffer_arg(),
                    actors: simulation.actors.clone().into_buffer_arg(),
                    config: simulation.config.launch(),
                    spatial: simulation.spatial.clone().into_buffer_arg(),
                    proposals: simulation.proposals.clone().into_buffer_arg(),
                    searches: simulation.searches.requests.clone().into_buffer_arg(),
                    successes: simulation.searches.successes.clone().into_buffer_arg(),
                    scratch: simulation.scratch.clone().into_buffer_arg(),
                    control: simulation.control.clone().into_buffer_arg(),
                    parameters: parameters.launch(),
                    status_reset: StatusReset::Full,
                    stage: SearchStage::Index,
                },
            );
        }
        match execution {
            SearchExecution::Serial => EconomySystem::launch(
                device,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                EconomyInputLaunch {
                    state: simulation.state.clone().into_buffer_arg(),
                    actors: simulation.actors.clone().into_buffer_arg(),
                    config: simulation.config.launch(),
                    spatial: simulation.spatial.clone().into_buffer_arg(),
                    proposals: simulation.proposals.clone().into_buffer_arg(),
                    bundles: simulation.scratch.clone().into_buffer_arg(),
                    control: simulation.control.clone().into_buffer_arg(),
                    parameters: parameters.launch(),
                    stage: EconomyStage::Propose,
                },
            ),
            _ => MarketSearch::launch(
                device,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(workgroup),
                SearchInputLaunch {
                    state: simulation.state.clone().into_buffer_arg(),
                    actors: simulation.actors.clone().into_buffer_arg(),
                    config: simulation.config.launch(),
                    spatial: simulation.spatial.clone().into_buffer_arg(),
                    proposals: simulation.proposals.clone().into_buffer_arg(),
                    searches: simulation.searches.requests.clone().into_buffer_arg(),
                    successes: simulation.searches.successes.clone().into_buffer_arg(),
                    scratch: simulation.scratch.clone().into_buffer_arg(),
                    control: simulation.control.clone().into_buffer_arg(),
                    parameters: parameters.launch(),
                    status_reset: StatusReset::Full,
                    stage: match execution {
                        SearchExecution::Prepare => SearchStage::Prepare,
                        SearchExecution::PrepareRepeat => SearchStage::PrepareRepeat,
                        _ => SearchStage::Search,
                    },
                },
            ),
        }
        let mut proposal = device.read(&simulation.proposals)?[..ProposalLayout::WORDS].to_vec();
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
            for workgroup in [32, 64, 128, 256] {
                assert_eq!(
                    probe.outcome_with_workgroup(SearchExecution::Prepare, workgroup)?,
                    expected,
                    "{good:?}, inventory {inventory}, workgroup {workgroup}"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn repeat_input_preparation_preserves_inactive_inventory_and_demand_faults_on_gpu() -> Result<()> {
    for good in [
        Good::Grain,
        Good::Bread,
        Good::Tools,
        Good::Wood,
        Good::Construction,
        Good::Coal,
        Good::Linen,
    ] {
        let mut probe = SearchProbe::new(SupplyCase::Capped, 0, Market::Input(Good::Linen))?;
        probe.parameters.good = good as u32;
        probe.state.set(Field::LivingMembers, 0, 0);
        probe.state.set(Field::inventory(good), 0, 100);
        let transit = TRANSIT_FIELDS + good as usize - 1;
        probe.state.values[transit * probe.state.actors.len()] = 7;
        let first = probe.outcome(SearchExecution::Prepare)?;
        assert_eq!(first.faults, 0);
        if good == Good::Bread {
            probe.state.set(Field::Bread, 0, 0);
        }
        let expected = probe.outcome(SearchExecution::Prepare)?;
        for workgroup in [32, 64, 128, 256] {
            assert_eq!(
                probe.outcome_with_workgroup(SearchExecution::PrepareRepeat, workgroup)?,
                expected,
                "inactive {good:?}, workgroup {workgroup}"
            );
        }
        probe.state.set(Field::EnterpriseModel, 0, 1);
        probe.state.set(Field::inventory(good), 0, u64::MAX);
        let expected = probe.outcome(SearchExecution::Prepare)?;
        assert_ne!(expected.faults, 0);
        for workgroup in [32, 64, 128, 256] {
            assert_eq!(
                probe.outcome_with_workgroup(SearchExecution::PrepareRepeat, workgroup)?,
                expected,
                "demand {good:?}, workgroup {workgroup}"
            );
        }
    }
    Ok(())
}

#[test]
fn first_round_inventory_faults_gate_repeat_preparation_on_gpu() -> Result<()> {
    for workgroup in [32, 64, 128, 256] {
        let mut probe = SearchProbe::new(SupplyCase::Capped, 0, Market::Input(Good::Linen))?;
        probe.state.set(Field::LivingMembers, 0, 0);
        probe.state.set(Field::Linen, 0, u64::MAX);
        let count = probe.state.actors.len();
        let transit = TRANSIT_FIELDS + Good::Linen as usize - 1;
        probe.state.values[transit * count] = 1;
        let first = probe.outcome_with_workgroup(SearchExecution::Prepare, workgroup)?;
        assert_ne!(first.faults, 0);
        let simulation = &probe.simulation;
        let device = &simulation.device;
        markets::prepare_dispatch(
            device,
            &simulation.control,
            &simulation.searches.dispatches,
            count,
        );
        let sentinel = 0x1234_5678;
        let proposals = vec![sentinel; count * ProposalLayout::WORDS];
        device.write(&simulation.proposals, &proposals);
        MarketSearch::launch(
            device,
            CubeCount::Static(count.div_ceil(workgroup as usize) as u32, 1, 1),
            CubeDim::new_1d(workgroup),
            SearchInputLaunch {
                state: simulation.state.clone().into_buffer_arg(),
                actors: simulation.actors.clone().into_buffer_arg(),
                config: simulation.config.launch(),
                spatial: simulation.spatial.clone().into_buffer_arg(),
                proposals: simulation.proposals.clone().into_buffer_arg(),
                searches: simulation.searches.requests.clone().into_buffer_arg(),
                successes: simulation.searches.successes.clone().into_buffer_arg(),
                scratch: simulation.scratch.clone().into_buffer_arg(),
                control: simulation.control.clone().into_buffer_arg(),
                parameters: probe.parameters.launch(),
                status_reset: StatusReset::Full,
                stage: SearchStage::PrepareRepeat,
            },
        );
        assert_eq!(device.read(&simulation.proposals)?, proposals);
        let control = device.read(&simulation.control)?;
        assert_eq!(control[crate::simulation::flags::FAULT_WORD], first.faults);
        assert_eq!(control[LATCH_WORD], first.faults);
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
    ] {
        for case in cases {
            for day in [0, 7, 31, 127, 128, SUPPLIERS - 1] {
                let probe = SearchProbe::new(case, day, market)?;
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
fn large_search_preserves_rotated_candidates_and_empty_offer_ties_on_gpu() -> Result<()> {
    for case in [
        SupplyCase::Capped,
        SupplyCase::Sparse,
        SupplyCase::Empty,
        SupplyCase::Unaffordable,
        SupplyCase::MixedAffordability,
        SupplyCase::Free,
        SupplyCase::DistantStock,
        SupplyCase::DistantEmpty,
    ] {
        for day in [31, 1023, 1024, 2047, 2176] {
            let probe = SearchProbe::with_suppliers(case, day, Market::Household, 2177)?;
            let expected = probe.outcome(SearchExecution::Serial)?;
            for execution in [SearchExecution::Shared, SearchExecution::Plane] {
                assert_eq!(
                    probe.outcome(execution)?,
                    expected,
                    "{case:?}, day {day}, {execution:?}"
                );
            }
        }
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
            if comptime!(input.config.railway.enabled == Feature::Enabled)
                && input.config.railway.nodes > 0
            {
                access = input.config.railway.access[buyer as usize * 2 + 1];
            }
            let floor = minimum_delivery_days(&input.config, access, seller, trip);
            let route = railway::route(&input.actors, &input.config, buyer, seller);
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
    let mut improved = 0;
    for scenario in [
        Scenario::industrial(193, 42, Experiment::Baseline)?,
        Scenario::industrial(193, 42, Experiment::Transport)?,
        Scenario::new(serde_json::from_str(include_str!(
            "../../../../scenarios/railway.json"
        ))?)?,
    ] {
        let device = device(64)?;
        let mut simulation = Simulation::new(scenario, device.clone())?;
        simulation.advance()?;
        for stage in [
            railway::RailStage::Access,
            railway::RailStage::Routes,
            railway::RailStage::Quotes,
        ] {
            simulation.railways(stage);
        }
        let findings = device.tensor(&[0, 0]);
        let count = simulation.metadata.actors.len();
        DeliveryFloorCheck::launch(
            &device,
            device.count(count * count),
            CubeDim::new_1d(device.workgroup),
            DeliveryFloorInputLaunch {
                actors: simulation.actors.clone().into_buffer_arg(),
                config: simulation.config.launch(),
                findings: findings.clone().into_buffer_arg(),
            },
        );
        let findings = device.read(&findings)?;
        assert_eq!(findings[0], 0);
        improved += findings[1];
    }
    assert!(improved > 0);
    Ok(())
}

#[test]
fn active_market_statuses_preserve_traces_and_feedback_across_days_on_gpu() -> Result<()> {
    for experiment in [Experiment::Concentrated, Experiment::Transport] {
        let generated = Scenario::industrial(67, 42, experiment)?;
        let initial = generated.initial()?;
        let mut config = generated.config().clone();
        let households = config.households as usize;
        config.population = Some(Population {
            households: (0..households)
                .map(|household| {
                    let actor = initial.actors.at(household);
                    let mut inventories =
                        Good::ALL.map(|good| initial.get(Field::inventory(good), household));
                    inventories[Good::Bread as usize - 1] = match household % 3 {
                        0 => 500,
                        _ => 0,
                    };
                    HouseholdSeed {
                        identity: actor.identity,
                        position: actor.position,
                        cash: match household % 4 {
                            0 => 1,
                            _ => 5000,
                        },
                        inventories,
                        fatigue: initial.get(Field::Fatigue, household) as u32,
                    }
                })
                .collect(),
            firms: (households..initial.actors.len())
                .map(|firm| {
                    let actor = initial.actors.at(firm);
                    Ok(FirmSeed {
                        identity: actor.identity,
                        position: actor.position,
                        owner: initial.actors.at(actor.owner as usize).identity,
                        good: Good::from_index(actor.kind)?,
                        cash: initial.get(Field::Cash, firm),
                        inventories: Good::ALL
                            .map(|good| initial.get(Field::inventory(good), firm)),
                        capacity: actor.capacity,
                        technique: generated.technique(firm),
                        wage: initial.get(Field::Wage, firm) as u32,
                        price: initial.get(Field::Price, firm) as u32,
                    })
                })
                .collect::<Result<Vec<_>>>()?,
        });
        let scenario = Scenario::new(config)?;
        let mut expected = Simulation::new(scenario.clone(), device(64)?)?;
        let mut actual = Simulation::new(scenario, device(64)?)?;
        for day in 0..3 {
            let trace = match day {
                1 => Trace::Record,
                _ => Trace::Discard,
            };
            for phase in Phase::ALL {
                let reference = expected.phase(phase, Trace::Record)?;
                let result = actual.phase(phase, trace)?;
                assert_eq!(
                    result.events, reference.events,
                    "{experiment:?} {day} {phase:?}"
                );
                assert_eq!(
                    result.frontier_depth, reference.frontier_depth,
                    "{experiment:?} {day} {phase:?}"
                );
                if let Trace::Record = trace {
                    assert_eq!(
                        result.decisions, reference.decisions,
                        "{experiment:?} {day} {phase:?}"
                    );
                }
                assert_eq!(
                    actual.download()?,
                    expected.download()?,
                    "{experiment:?} {day} {phase:?}"
                );
                assert_eq!(actual.snapshot, expected.snapshot);
            }
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
        EconomySystem::launch(
            device,
            device.count(count),
            CubeDim::new_1d(device.workgroup),
            EconomyInputLaunch {
                state: simulation.state.clone().into_buffer_arg(),
                actors: simulation.actors.clone().into_buffer_arg(),
                config: simulation.config.launch(),
                spatial: simulation.spatial.clone().into_buffer_arg(),
                proposals: simulation.proposals.clone().into_buffer_arg(),
                bundles: simulation.scratch.clone().into_buffer_arg(),
                control: simulation.control.clone().into_buffer_arg(),
                parameters: self.parameters.launch(),
                stage: initialization,
            },
        );
        let dispatch = match stage {
            SearchStage::PrepareSelected => markets::dispatch(
                &simulation.searches.dispatches,
                markets::PREPARATION_DISPATCH,
            ),
            _ => device.count(count),
        };
        MarketSearch::launch(
            device,
            dispatch,
            CubeDim::new_1d(device.workgroup),
            SearchInputLaunch {
                state: simulation.state.clone().into_buffer_arg(),
                actors: simulation.actors.clone().into_buffer_arg(),
                config: simulation.config.launch(),
                spatial: simulation.spatial.clone().into_buffer_arg(),
                proposals: simulation.proposals.clone().into_buffer_arg(),
                searches: simulation.searches.requests.clone().into_buffer_arg(),
                successes: simulation.searches.successes.clone().into_buffer_arg(),
                scratch: simulation.scratch.clone().into_buffer_arg(),
                control: simulation.control.clone().into_buffer_arg(),
                parameters: self.parameters.launch(),
                status_reset: StatusReset::Active,
                stage,
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
    for workgroup in [32, 64, 128, 256] {
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
                    PreparationEligibility::GrainCashless
                    | PreparationEligibility::GrainStocked => {
                        probe.state.set(Field::LandPlots, 0, 1)
                    }
                    PreparationEligibility::Investment => {
                        probe.state.set(Field::InvestmentGood, 0, 1)
                    }
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
                let expected = dense
                    .prepared_inputs(EconomyStage::InputMarketRepeat, SearchStage::PrepareRepeat)?;
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
                let expected = dense
                    .prepared_inputs(EconomyStage::InputMarketRepeat, SearchStage::PrepareRepeat)?;
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
    }
    Ok(())
}

#[test]
fn sparse_input_preparation_preserves_first_round_inventory_faults_on_gpu() -> Result<()> {
    for workgroup in [32, 64, 128, 256] {
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
    }
    Ok(())
}

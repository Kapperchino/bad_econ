use super::*;
use crate::{
    domain::{
        Experiment, FirmSeed, HouseholdSeed, Population, Position, Scenario, State, Technique,
        enterprise::HouseholdProduction,
    },
    simulation::{Market, Simulation, encode, parallel::ScanMode, tests::device},
};
use anyhow::Result;

#[derive(Clone, Copy, Debug)]
enum SupplyCase {
    Capped,
    Sparse,
    Empty,
    Unaffordable,
    MixedAffordability,
    LaborPrefix(u32),
}

#[derive(Clone, Copy, Debug)]
enum SearchExecution {
    Serial,
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
    fn new(case: SupplyCase, day: u32) -> Result<Self> {
        let mut config = Scenario::baseline(1, 42, Experiment::Concentrated)?
            .config()
            .clone();
        config.firms = 40;
        config.search_radius = config.map_size;
        config.household_production = HouseholdProduction::industrial();
        config.population = Some(Population {
            households: vec![HouseholdSeed {
                identity: 1,
                position: Position { x: 500, y: 500 },
                cash: 5000,
                inventories: Good::food([0, 0, 100]),
                fatigue: 0,
            }],
            firms: (2..=41)
                .map(|identity| FirmSeed {
                    identity,
                    position: Position { x: 500, y: 500 },
                    owner: 1,
                    good: Good::Bread,
                    cash: 5000,
                    inventories: [0; Good::COUNT],
                    capacity: 1,
                    technique: Technique::new(1, 1, 1).expect("one-minute packet"),
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
        state.set(Field::Bread, 0, 0);
        state.set(Field::Time, 0, u64::from(u32::MAX));
        for firm in 1..=40 {
            let supply = match case {
                SupplyCase::Capped | SupplyCase::Unaffordable => 100,
                SupplyCase::MixedAffordability => u64::from(firm > 1) * 100,
                SupplyCase::Sparse => u64::from(firm % 5 == 0) * 100,
                _ => 0,
            };
            let price = match case {
                SupplyCase::Capped => 1 + u64::from(firm <= 8) * 9,
                SupplyCase::Sparse | SupplyCase::Empty => 1 + (firm % 3) as u64,
                SupplyCase::Unaffordable => 5001,
                SupplyCase::MixedAffordability => u64::from(firm > 1) * 5001,
                SupplyCase::LaborPrefix(_) => 1,
            };
            state.set(Field::Bread, firm, supply);
            state.set(Field::Price, firm, price);
            state.set(Field::LaborMinutes, firm, 1);
            state.set(Field::InputTicks, firm, 0);
            state.set(Field::EquipmentCondition, firm, 1000);
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
            _ => Market::Household,
        };
        let simulation = Simulation::new(scenario, device(64)?)?;
        let mut parameters =
            simulation.parameters(Phase::ProvisionPurchase, market, 0, state.actors.len())?;
        parameters.day = day;
        let mut spatial = simulation.device.read(&simulation.spatial)?;
        spatial.fill(0);
        spatial[1] = 40;
        spatial[2..42].copy_from_slice(&(1..=40).collect::<Vec<u32>>());
        let prices = 3 * (2 + state.actors.len());
        spatial[prices] = (1..=40)
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
        searches[count] = (self.state.get(Field::FoodNeed, 0)
            * u64::from(simulation.config.coverage + simulation.config.ring - 1))
            as u32;
        searches[3 * count] = self.state.get(Field::Cash, 0) as u32;
        device.write(&simulation.searches.requests, &searches);
        device.write(
            &simulation.searches.successes,
            &vec![
                0;
                markets::search_counts_offset(count) + 2 * markets::search_blocks(count) + 512
            ],
        );
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
                CubeDim::new_1d(32),
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
                    stage: SearchStage::Search,
                },
            ),
        }
        Ok(SearchOutcome {
            proposal: device.read(&simulation.proposals)?[..ProposalLayout::WORDS].to_vec(),
            state: device.read(&simulation.state)?,
            faults: device.read(&simulation.control)?[crate::simulation::flags::FAULT_WORD],
        })
    }
}

#[test]
fn cooperative_market_search_preserves_serial_search_and_shared_fallback_on_gpu() -> Result<()> {
    for case in [
        SupplyCase::Capped,
        SupplyCase::Sparse,
        SupplyCase::Empty,
        SupplyCase::Unaffordable,
        SupplyCase::MixedAffordability,
    ] {
        for day in [0, 7, 31, 39] {
            let probe = SearchProbe::new(case, day)?;
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

#[test]
fn cooperative_market_search_only_reports_faults_from_the_visited_prefix_on_gpu() -> Result<()> {
    for available in [7, 6] {
        let probe = SearchProbe::new(SupplyCase::LaborPrefix(available), 0)?;
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

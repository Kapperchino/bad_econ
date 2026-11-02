use super::*;

#[test]
fn industrial_recipes_match_coordinate_scheduler() -> Result<()> {
    let mut config = Scenario::industrial(64, 42, Experiment::Concentrated)?
        .config()
        .clone();
    config.batch_size = 13;
    for lab in &mut config.research.as_mut().expect("industrial research").labs {
        lab.discovered = 31;
        lab.mastered = 31;
        lab.adopted = 31;
    }
    compare_market_execution(Scenario::new(config)?, 4)
}

fn compare_market_execution(scenario: Scenario, days: usize) -> Result<()> {
    let initial = scenario.initial()?;
    compare_market_state(scenario, initial, days)
}

fn compare_market_state(scenario: Scenario, initial: State, days: usize) -> Result<()> {
    let mut grouped =
        Simulation::from_state(scenario.clone(), initial.clone(), Device::headless(128)?)?;
    let mut reference = Simulation::from_state(scenario, initial, Device::headless(64)?)?;
    reference.market_execution = MarketExecution::Reference;
    for day in 0..days {
        for phase in Phase::ALL {
            let expected = reference.phase(phase, Trace::Record)?;
            let actual = grouped.phase(phase, Trace::Record)?;
            assert_eq!(actual.events, expected.events, "day {day} {phase:?}");
            assert_eq!(actual.decisions, expected.decisions, "day {day} {phase:?}");
            assert_eq!(
                grouped.download()?,
                reference.download()?,
                "day {day} {phase:?}"
            );
        }
        assert_eq!(grouped.snapshot, reference.snapshot, "day {day}");
    }
    Ok(())
}

#[test]
fn price_blocks_preserve_rotated_search_and_unaffordable_food_diagnostics() -> Result<()> {
    let mut config = Scenario::baseline(769, 52, Experiment::Baseline)?
        .config()
        .clone();
    config.firms = 3;
    config.population = Some(Population {
        households: (1..=769)
            .map(|identity| HouseholdSeed {
                identity,
                position: match identity % 3 {
                    0 => Position { x: 999, y: 999 },
                    1 => Position { x: 0, y: 0 },
                    _ => Position { x: 500, y: 500 },
                },
                cash: match identity % 4 {
                    0 => 1,
                    1 => 35,
                    _ => 5000,
                },
                inventories: Good::food([0, 0, u64::from(identity > 256) * 100]),
                fatigue: 0,
            })
            .collect(),
        firms: (770..=772)
            .map(|identity| FirmSeed {
                identity,
                position: Position { x: 0, y: 0 },
                owner: 1,
                good: Good::Bread,
                cash: 0,
                inventories: [0; Good::COUNT],
                capacity: 1,
                technique: Technique {
                    labor_minutes: 60,
                    input_ticks: 1,
                    output_ticks: 1,
                },
                wage: 1,
                price: 10,
            })
            .collect(),
    });
    let scenario = Scenario::new(config)?;
    let mut initial = scenario.initial()?;
    for household in 0..769 {
        initial.set(Field::Price, household, 10 * (household as u64 / 128 + 1));
    }
    compare_market_state(scenario, initial, 4)
}

#[test]
fn grouped_markets_match_coordinate_scheduler_at_every_phase() -> Result<()> {
    for experiment in [
        Experiment::Baseline,
        Experiment::Shortage,
        Experiment::Transport,
        Experiment::Concentrated,
    ] {
        let mut config = Scenario::baseline(193, 42, experiment)?.config().clone();
        config.firms = 12;
        config.batch_size = 31;
        compare_market_execution(Scenario::new(config)?, 8)?;
    }
    Ok(())
}

#[test]
fn grouped_institutions_and_custom_packets_match_coordinate_scheduler() -> Result<()> {
    for scenario in [
        include_str!("../../scenarios/institutions.json"),
        include_str!("../../scenarios/custom-population.json"),
    ] {
        let mut config: ScenarioConfig = serde_json::from_str(scenario)?;
        config.batch_size = 1;
        compare_market_execution(Scenario::new(config)?, 8)?;
    }
    let mut config = Scenario::baseline(67, 93, Experiment::Shortage)?
        .config()
        .clone();
    config.household_cash = 1u64 << 40;
    config.firm_cash = 1u64 << 42;
    config.reserve = 1;
    config.batch_size = 13;
    compare_market_execution(Scenario::new(config)?, 8)
}

struct MarketFault {
    phase: Phase,
    field: Field,
    actor: usize,
}

#[test]
fn grouped_market_faults_match_reference_and_restore_the_audited_state() -> Result<()> {
    let cases = [
        MarketFault {
            phase: Phase::Commitment,
            field: Field::Due,
            actor: 0,
        },
        MarketFault {
            phase: Phase::Commitment,
            field: Field::Escrow,
            actor: 3,
        },
        MarketFault {
            phase: Phase::Commitment,
            field: Field::Pipeline,
            actor: 3,
        },
        MarketFault {
            phase: Phase::Commitment,
            field: Field::Used,
            actor: 4,
        },
        MarketFault {
            phase: Phase::ProvisionPurchase,
            field: Field::Sales,
            actor: 5,
        },
        MarketFault {
            phase: Phase::ProvisionPurchase,
            field: Field::Purchased,
            actor: 0,
        },
    ];
    for case in cases {
        let scenario = Scenario::baseline(3, 42, Experiment::Concentrated)?;
        let initial = scenario.initial()?;
        let mut messages = Vec::new();
        for execution in [MarketExecution::Grouped, MarketExecution::Reference] {
            let mut simulation = Simulation::new(scenario.clone(), Device::headless(64)?)?;
            simulation.market_execution = execution;
            for phase in Phase::ALL
                .into_iter()
                .take_while(|phase| *phase != case.phase)
            {
                simulation.phase(phase, Trace::Record)?;
            }
            let mut state = simulation.download()?;
            match case.phase {
                Phase::Commitment => {
                    for firm in 3..6 {
                        state.set(Field::Slots, firm, 1);
                    }
                    if case.actor == 4 {
                        state.set(Field::Slots, 3, 0);
                    }
                }
                _ => {
                    for household in 0..3 {
                        state.set(Field::Bread, household, 0);
                    }
                }
            }
            state.set(case.field, case.actor, u64::MAX);
            simulation.device.write(
                &simulation.state,
                bytemuck::cast_slice(&encode(&state.values)),
            );
            messages.push(
                simulation
                    .phase(case.phase, Trace::Record)
                    .unwrap_err()
                    .to_string(),
            );
            assert_eq!(simulation.download()?, initial);
        }
        assert_eq!(messages[0], messages[1]);
    }
    Ok(())
}

#[test]
fn gpu_registry_migration_births_and_estates_match_audited_rebuilds() -> Result<()> {
    let mut config = Scenario::baseline(300, 42, Experiment::Baseline)?
        .config()
        .clone();
    config.firms = 3;
    config.search_radius = 200;
    config.shipment_speed = 100;
    config.household_production = crate::domain::enterprise::HouseholdProduction::industrial();
    config.demography.annual_births_per_million_fertile_women = 1_000_000;
    config.demography.mortality.background_annual_ppm = [1_000_000; 5];
    config.demography.mortality.famine_daily_ppm_at_thirty_days = 0;
    config.population = Some(Population {
        households: (1..=300)
            .map(|identity| HouseholdSeed {
                identity,
                position: Position { x: 0, y: 0 },
                cash: 5000,
                inventories: [0; Good::COUNT],
                fatigue: 0,
            })
            .collect(),
        firms: [400, 700, 900]
            .into_iter()
            .enumerate()
            .map(|(index, x)| FirmSeed {
                identity: 1000 + index as u64,
                position: Position { x, y: 0 },
                owner: 1,
                good: Good::Bread,
                cash: 0,
                inventories: Good::food([0, 0, if index == 0 { 6000 } else { 0 }]),
                capacity: 1,
                technique: Technique::new(60, 1, 1).expect("basic bakery"),
                wage: 1,
                price: 10,
            })
            .collect(),
    });
    let scenario = Scenario::new(config)?;
    let mut gpu = Simulation::new(scenario.clone(), Device::headless(64)?)?;
    let mut reference = Simulation::new(scenario.clone(), Device::headless(128)?)?;
    reference.registry_execution = RegistryExecution::Reference;
    for day in 0..60 {
        if day == 12 {
            let mut state = gpu.download()?;
            let stock = state.get(Field::Bread, 300);
            state.set(Field::Bread, 300, 0);
            state.set(Field::Bread, 302, stock);
            let checkpoint = Checkpoint::new(&scenario, &state)?;
            gpu.command(SimulationCommand::Restore(Box::new(checkpoint.clone())))?;
            reference.command(SimulationCommand::Restore(Box::new(checkpoint)))?;
        }
        gpu.advance()?;
        reference.advance()?;
        assert_eq!(gpu.download()?, reference.download()?, "day {day}");
        assert_eq!(gpu.snapshot, reference.snapshot, "day {day}");
    }
    let state = gpu.download()?;
    assert!(
        state
            .actors
            .iter()
            .any(|actor| actor.kind == 0 && actor.position.x == 900)
    );
    assert!(gpu.snapshot.population.births > 0);
    assert!(gpu.snapshot.population.deaths[0] > 0);
    assert!((0..state.actors.len()).any(|actor| state.get(Field::EstateSettled, actor) == 2));
    gpu.checkpoint()?.restore()?;
    Ok(())
}

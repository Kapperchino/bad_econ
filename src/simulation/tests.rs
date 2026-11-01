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
    let mut grouped = Simulation::new(scenario.clone(), Device::headless(64)?)?;
    let mut reference = Simulation::new(scenario, Device::headless(64)?)?;
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

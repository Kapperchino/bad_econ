#[cfg(feature = "compute")]
mod support;
use anyhow::Result;
use bad_econ::domain::*;

fn scenario() -> Result<Scenario> {
    let mut config = Scenario::baseline(3, 19, Experiment::Baseline)?
        .config()
        .clone();
    config.distribution_rate = 0;
    config.behavior = BehaviorPolicy {
        health_recovery: 7,
        hunger_damage: 30,
        work_fatigue: 13,
        rest_recovery: 17,
        wage_increment: 0,
        price_response: 0,
    };
    config.population = Some(Population {
        households: [901, 101, 303]
            .into_iter()
            .map(|identity| HouseholdSeed {
                identity,
                position: Position { x: 0, y: 0 },
                cash: 1000,
                inventories: [0, 0, 10],
                health: 1000,
                fatigue: 0,
            })
            .collect(),
        firms: vec![
            FirmSeed {
                identity: 8000,
                owner: 303,
                good: Good::Bread,
                cash: 10_000,
                position: Position { x: 0, y: 0 },
                inventories: [0, 3, 0],
                capacity: 1,
                technique: Technique::new(240, 3, 2)?,
                wage: 3,
                price: 30,
            },
            FirmSeed {
                identity: 2000,
                owner: 901,
                good: Good::Grain,
                cash: 10_000,
                position: Position { x: 0, y: 0 },
                inventories: [0, 0, 0],
                capacity: 1,
                technique: Technique::new(120, 0, 5)?,
                wage: 1,
                price: 10,
            },
            FirmSeed {
                identity: 5000,
                owner: 101,
                good: Good::Flour,
                cash: 10_000,
                position: Position { x: 0, y: 0 },
                inventories: [2, 0, 0],
                capacity: 1,
                technique: Technique::new(180, 2, 3)?,
                wage: 2,
                price: 20,
            },
        ],
    });
    Scenario::new(config)
}

#[test]
fn population_admission_checks_identity_ownership_resources_and_policy() -> Result<()> {
    let scenario = scenario()?;
    let state = scenario.initial()?;
    assert_eq!(
        state
            .actors
            .iter()
            .map(|actor| actor.identity)
            .collect::<Vec<_>>(),
        [101, 303, 901, 2000, 5000, 8000]
    );
    assert_eq!(state.next_identity, 8001);
    assert_eq!(state.initial_cash, 33_000);
    assert_eq!(state.initial_goods, [2, 3, 30]);
    assert!(InspectionRequest::new(1, &scenario).is_err());
    assert!(InspectionRequest::new(5000, &scenario).is_ok());
    let mut config = scenario.config().clone();
    config.population.as_mut().unwrap().firms[0].owner = 2000;
    assert!(Scenario::new(config).is_err());
    let mut config = scenario.config().clone();
    config.population.as_mut().unwrap().firms[0].identity = 101;
    assert!(Scenario::new(config).is_err());
    let mut config = scenario.config().clone();
    config.population.as_mut().unwrap().households[0].cash = u64::MAX;
    assert!(Scenario::new(config).is_err());
    let mut config = scenario.config().clone();
    config.population.as_mut().unwrap().firms[1]
        .technique
        .input_ticks = 0;
    assert!(Scenario::new(config).is_err());
    let mut config = scenario.config().clone();
    config.behavior.price_response = 1001;
    assert!(Scenario::new(config).is_err());
    let checkpoint = Checkpoint::new(&scenario, &state)?;
    assert_eq!(checkpoint.restore()?.state, state);
    Ok(())
}

#[test]
fn checkpoint_admission_preserves_fixed_metadata_and_labor_budgets() -> Result<()> {
    let generated = Scenario::baseline(3, 42, Experiment::Baseline)?;
    let explicit = scenario()?;
    for scenario in [&generated, &explicit] {
        let state = scenario.initial()?;
        let mut changed = state.clone();
        changed.actors[3].owner = (changed.actors[3].owner + 1) % 3;
        assert!(Checkpoint::new(scenario, &changed).is_err());
        let mut changed = state.clone();
        changed.actors[3].capacity += 1;
        assert!(Checkpoint::new(scenario, &changed).is_err());
        let mut changed = state.clone();
        changed.actors[0].position.x =
            (changed.actors[0].position.x + 1) % scenario.config().map_size;
        assert!(Checkpoint::new(scenario, &changed).is_err());
        let mut changed = state.clone();
        changed.set(Field::Employed, 0, 1);
        changed.set(Field::Employer, 0, 2);
        assert!(Checkpoint::new(scenario, &changed).is_err());
        let mut changed = state.clone();
        changed.set(Field::Employed, 0, 1);
        changed.set(Field::Employer, 0, 4);
        changed.set(
            Field::Hours,
            0,
            u64::from(scenario.technique(3).labor_minutes),
        );
        changed.set(Field::Time, 0, u64::from(scenario.config().work_cap));
        assert!(Checkpoint::new(scenario, &changed).is_err());
        let mut changed = state.clone();
        changed.set(Field::Plan, 3, u64::from(changed.actors[3].capacity) + 1);
        assert!(Checkpoint::new(scenario, &changed).is_err());
        let mut changed = state.clone();
        changed.set(Field::Transit, 3, 1);
        let delivery = changed.ring_coordinate(0, 3).0 as usize;
        changed.values[delivery] = 1;
        assert!(Checkpoint::new(scenario, &changed).is_err());
        let mut changed = state.clone();
        changed.set(Field::Transit, 0, 1);
        changed.set(Field::Bread, 0, changed.get(Field::Bread, 0) - 1);
        let expired = changed.ring_coordinate(scenario.ring() - 1, 0).0 as usize;
        changed.values[expired] = 1;
        assert!(Checkpoint::new(scenario, &changed).is_err());
        assert_eq!(Checkpoint::new(scenario, &state)?.restore()?.state, state);
    }
    assert_eq!(generated.technique(3).input_ticks, 0);
    Ok(())
}

#[test]
fn state_digest_includes_metadata_and_initial_totals() -> Result<()> {
    let state = scenario()?.initial()?;
    let digest = state.digest();
    let mut changed = state.clone();
    changed.actors[0].identity += 1;
    assert_ne!(changed.digest(), digest);
    let mut changed = state.clone();
    changed.actors[3].owner = 0;
    assert_ne!(changed.digest(), digest);
    let mut changed = state.clone();
    changed.actors[3].capacity += 1;
    assert_ne!(changed.digest(), digest);
    let mut changed = state.clone();
    changed.actors[0].position.x += 1;
    assert_ne!(changed.digest(), digest);
    let mut changed = state.clone();
    changed.initial_cash += 1;
    assert_ne!(changed.digest(), digest);
    let mut changed = state.clone();
    changed.initial_goods[0] += 1;
    assert_ne!(changed.digest(), digest);
    assert_eq!(
        serde_json::from_str::<State>(&serde_json::to_string(&state)?)?.digest(),
        digest
    );
    Ok(())
}

#[cfg(feature = "compute")]
mod compute {
    use super::*;
    use bad_econ::{
        runtime::SimulationCommand,
        simulation::{Device, Simulation, Trace},
    };

    #[test]
    fn heterogeneous_packets_preserve_time_goods_ownership_and_replay() -> Result<()> {
        let scenario = scenario()?;
        let mut simulation = Simulation::new(scenario.clone(), support::device(64)?)?;
        simulation.advance()?;
        let state = simulation.download()?;
        assert_eq!(simulation.snapshot.produced, [5, 3, 2]);
        assert_eq!(simulation.snapshot.used, [2, 3, 0]);
        assert_eq!(simulation.snapshot.employed, 3);
        assert_eq!(state.get(Field::Hours, 0), 240);
        assert_eq!(state.get(Field::Time, 0), 240);
        assert_eq!(state.get(Field::Fatigue, 0), 13);
        assert_eq!(state.get(Field::Cash, 0), 1720);
        let household = simulation.inspect(InspectionRequest::new(101, &scenario)?)?;
        assert_eq!(household.employer, 8000);
        let firm = simulation.inspect(InspectionRequest::new(5000, &scenario)?)?;
        assert_eq!(firm.owner_identity, 101);
        assert_eq!(firm.price, 20);
        assert_eq!(
            simulation.snapshot,
            ObservationSnapshot::from_state(&scenario, &state)?
        );
        let checkpoint = simulation.checkpoint()?;
        simulation.advance()?;
        let expected = simulation.download()?;
        simulation.command(SimulationCommand::Restore(checkpoint.into()))?;
        simulation.advance()?;
        assert_eq!(simulation.download()?, expected);
        let mut config = scenario.config().clone();
        config.reserve = 0;
        config.distribution_rate = 1000;
        let mut distribution = Simulation::new(Scenario::new(config)?, support::device(64)?)?;
        distribution.advance()?;
        let state = distribution.download()?;
        assert_eq!(state.get(Field::Cash, 0), 1720 + 9640);
        assert_eq!(state.get(Field::Cash, 1), 1360 + 9280);
        assert_eq!(state.get(Field::Cash, 2), 1120 + 9880);
        Ok(())
    }

    #[test]
    fn configured_recovery_and_missing_markets_have_exact_summaries() -> Result<()> {
        let mut config = scenario()?.config().clone();
        let population = config.population.as_mut().unwrap();
        population
            .households
            .iter_mut()
            .enumerate()
            .for_each(|(index, seed)| {
                seed.health = 900;
                seed.fatigue = 50;
                if index == 0 {
                    seed.inventories[2] = 0;
                }
            });
        population.firms.iter_mut().for_each(|seed| {
            seed.good = Good::Grain;
            seed.technique.input_ticks = 0;
            seed.cash = 0;
        });
        let scenario = Scenario::new(config)?;
        let mut simulation = Simulation::new(scenario.clone(), support::device(64)?)?;
        simulation.advance()?;
        let state = simulation.download()?;
        assert_eq!(state.get(Field::Health, 0), 870);
        assert_eq!(state.get(Field::Health, 1), 907);
        assert_eq!(state.get(Field::Fatigue, 0), 33);
        assert_eq!(simulation.snapshot.markets[1], MarketObservation::default());
        assert_eq!(simulation.snapshot.markets[2], MarketObservation::default());
        assert_eq!(
            simulation.snapshot.markets[0].prices,
            QuoteDistribution {
                count: 3,
                minimum: 10,
                maximum: 30,
                total: 60,
                squares: 1400
            }
        );
        assert_eq!(simulation.snapshot.markets[0].prices.mean(), 20.0);
        assert!(
            (simulation.snapshot.markets[0].prices.deviation() - (200.0f64 / 3.0).sqrt()).abs()
                < 1e-12
        );
        assert_eq!(
            simulation.snapshot,
            ObservationSnapshot::from_state(&scenario, &state)?
        );
        Ok(())
    }

    #[test]
    fn custom_population_matches_cpu_at_every_phase_and_workgroup() -> Result<()> {
        let mut canonical = None;
        for workgroup in [32, 64, 128, 256, 64] {
            let scenario = scenario()?;
            let mut cpu = Simulation::new(scenario.clone(), Device::cpu(workgroup)?)?;
            let mut gpu = Simulation::new(scenario, Device::headless(workgroup)?)?;
            for _ in 0..4 {
                for phase in Phase::ALL {
                    assert_eq!(
                        gpu.phase(phase, Trace::Record)?.decisions,
                        cpu.phase(phase, Trace::Record)?.decisions
                    );
                    assert_eq!(gpu.download()?, cpu.download()?);
                }
                let state = cpu.download()?;
                assert_eq!(
                    gpu.snapshot,
                    ObservationSnapshot::from_state(&cpu.scenario, &state)?
                );
            }
            let state = gpu.download()?;
            if let Some(expected) = canonical.replace(state.clone()) {
                assert_eq!(state, expected);
            }
        }
        Ok(())
    }
}

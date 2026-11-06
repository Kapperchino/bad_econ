#[cfg(feature = "compute")]
use crate::support;
use anyhow::Result;
use bad_econ::domain::*;

#[cfg(feature = "compute")]
#[test]
fn industrial_household_liquidity_is_unequal_and_replays_on_gpu() -> Result<()> {
    use bad_econ::simulation::Simulation;

    let scenario = Scenario::industrial(1000, 42, Experiment::Baseline)?;
    let config = scenario.config();
    let initial = scenario.initial()?;
    let mut balances = (0..config.households as usize)
        .map(|household| initial.get(Field::Cash, household))
        .collect::<Vec<_>>();
    balances.sort_unstable();
    let daily_expenses = u64::from(config.household_need)
        * u64::from(config.demography.initial_household_members)
        * u64::from(config.prices[2])
        + u64::from(config.rent.map_or(0, |rent| rent.household_daily));
    let weekly = balances
        .iter()
        .filter(|&&cash| cash <= 7 * daily_expenses)
        .count();
    assert!((550..=650).contains(&weekly));
    assert!(balances[500] < 7 * daily_expenses);
    assert!(balances[950] > 28 * daily_expenses);
    assert_eq!(
        initial.initial_cash,
        balances.iter().sum::<u64>() + u64::from(config.firms) * config.firm_cash
    );
    let decoded = Scenario::new(serde_json::from_slice(&serde_json::to_vec(config)?)?)?;
    assert_eq!(decoded.initial()?, initial);
    let mut first = Simulation::new(scenario.clone(), support::device(64)?)?;
    let uploaded = first.download()?;
    for household in 0..config.households as usize {
        assert_eq!(
            uploaded.get(Field::Cash, household),
            initial.get(Field::Cash, household)
        );
    }
    for _ in 0..8 {
        first.advance()?;
        assert_eq!(first.snapshot.cash, initial.initial_cash);
    }
    let mut replay = Simulation::from_checkpoint(first.checkpoint()?, support::device(128)?)?;
    first.advance()?;
    replay.advance()?;
    assert_eq!(first.download()?, replay.download()?);
    let different = Scenario::industrial(1000, 43, Experiment::Baseline)?.initial()?;
    assert!(
        (0..1000).any(|index| initial.get(Field::Cash, index) != different.get(Field::Cash, index))
    );
    Ok(())
}

#[cfg(feature = "compute")]
#[test]
fn legacy_and_explicit_household_cash_remain_exact_on_gpu() -> Result<()> {
    use bad_econ::simulation::Simulation;

    let mut legacy =
        serde_json::to_value(Scenario::baseline(12, 42, Experiment::Baseline)?.config())?;
    legacy
        .as_object_mut()
        .unwrap()
        .remove("household_liquidity");
    let uniform = Scenario::new(serde_json::from_value(legacy)?)?;
    let mut explicit = scenario()?.config().clone();
    explicit.household_liquidity = HouseholdLiquidity::PaycheckToPaycheck;
    for scenario in [uniform, Scenario::new(explicit)?] {
        let expected = scenario.initial()?;
        let mut simulation = Simulation::new(scenario.clone(), support::device(64)?)?;
        let state = simulation.download()?;
        for household in 0..scenario.config().households as usize {
            let cash = scenario
                .config()
                .population
                .as_ref()
                .map_or(scenario.config().household_cash, |population| {
                    population.households[household].cash
                });
            assert_eq!(state.get(Field::Cash, household), cash);
        }
        simulation.advance()?;
        assert_eq!(simulation.snapshot.cash, expected.initial_cash);
        simulation.checkpoint()?.restore()?;
    }
    Ok(())
}

fn scenario() -> Result<Scenario> {
    let mut config = Scenario::baseline(3, 19, Experiment::Baseline)?
        .config()
        .clone();
    config.distribution_rate = 0;
    config.behavior = BehaviorPolicy {
        work_fatigue: 13,
        rest_recovery: 17,
        wage_increment: 0,
        price_response: 0,
    };
    config.resources = ResourceMap::Explicit {
        regions: vec![ResourceRegion::new(
            Good::Grain,
            Position { x: 0, y: 0 },
            Position {
                x: config.map_size,
                y: config.map_size,
            },
            config.map_size,
        )?],
    };
    config.population = Some(Population {
        households: [901, 101, 303]
            .into_iter()
            .map(|identity| HouseholdSeed {
                identity,
                position: Position { x: 0, y: 0 },
                cash: 1000,
                inventories: Good::food([0, 0, 10]),
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
                inventories: Good::food([0, 3, 0]),
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
                inventories: Good::food([0, 0, 0]),
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
                inventories: Good::food([2, 0, 0]),
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
fn checkpoint_admission_preserves_fixed_metadata_and_labor_budgets() -> Result<()> {
    let generated = Scenario::baseline(3, 42, Experiment::Baseline)?;
    let explicit = scenario()?;
    for scenario in [&generated, &explicit] {
        let state = scenario.initial()?;
        let mut changed = state.clone();
        let mut actor = changed.actors.at(3);
        actor.owner = (changed.actors.at(3).owner + 1) % 3;
        changed.actors.set(3, actor);
        assert!(Checkpoint::new(scenario, &changed).is_err());
        let mut changed = state.clone();
        let mut actor = changed.actors.at(3);
        actor.capacity += 1;
        changed.actors.set(3, actor);
        assert!(Checkpoint::new(scenario, &changed).is_err());
        let mut changed = state.clone();
        let mut actor = changed.actors.at(0);
        actor.position.x = (changed.actors.at(0).position.x + 1) % scenario.config().map_size;
        changed.actors.set(0, actor);
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
        changed.set(Field::Plan, 3, u64::from(changed.actors.at(3).capacity) + 1);
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
    let mut actor = changed.actors.at(0);
    actor.identity += 1;
    changed.actors.set(0, actor);
    assert_ne!(changed.digest(), digest);
    let mut changed = state.clone();
    let mut actor = changed.actors.at(3);
    actor.owner = 0;
    changed.actors.set(3, actor);
    assert_ne!(changed.digest(), digest);
    let mut changed = state.clone();
    let mut actor = changed.actors.at(3);
    actor.capacity += 1;
    changed.actors.set(3, actor);
    assert_ne!(changed.digest(), digest);
    let mut changed = state.clone();
    let mut actor = changed.actors.at(0);
    actor.position.x += 1;
    changed.actors.set(0, actor);
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
    use bad_econ::{runtime::SimulationCommand, simulation::Simulation};

    #[test]
    fn heterogeneous_packets_preserve_time_goods_ownership_and_replay() -> Result<()> {
        let scenario = scenario()?;
        let mut simulation = Simulation::new(scenario.clone(), support::device(64)?)?;
        simulation.advance()?;
        let state = simulation.download()?;
        assert_eq!(simulation.snapshot.produced, Good::food([5, 3, 2]));
        assert_eq!(simulation.snapshot.used, Good::food([2, 3, 0]));
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
}

#[cfg(feature = "compute")]
#[test]
fn actor_rows_round_trip_on_gpu_and_reject_mismatched_lengths() -> Result<()> {
    use bad_econ::simulation::Simulation;
    let scenario = scenario()?;
    let expected = scenario.initial()?;
    let mut simulation = Simulation::new(scenario.clone(), support::device(64)?)?;
    assert_eq!(simulation.download()?, expected);
    let encoded = serde_json::to_value(&expected)?;
    assert_eq!(
        encoded["actors"]["identity"],
        serde_json::json!([101, 303, 901, 2000, 5000, 8000])
    );
    for field in ["identity", "x", "y", "kind", "owner", "capacity"] {
        let mut invalid = encoded.clone();
        invalid["actors"][field].as_array_mut().unwrap().pop();
        assert!(serde_json::from_value::<State>(invalid).is_err());
    }
    for actor in expected.actors.iter() {
        let request = InspectionRequest::new(actor.identity, &scenario)?;
        assert_eq!(simulation.inspect(request.clone())?.actor, actor);
        assert_eq!(Inspection::from_state(&expected, &request)?.actor, actor);
    }
    let checkpoint = simulation.checkpoint()?;
    let mut old = serde_json::to_value(&checkpoint)?;
    old["version"] = serde_json::json!(2);
    assert!(serde_json::from_value::<Checkpoint>(old).is_err());
    Ok(())
}

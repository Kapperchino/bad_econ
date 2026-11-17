use super::support;
use anyhow::Result;
use bad_econ::{
    domain::{
        demography::{DeathCause, LivingState},
        *,
    },
    simulation::Simulation,
};

fn heating_scenario(households: u32) -> Result<Scenario> {
    let mut config = Scenario::baseline(households, 19, Experiment::Baseline)?
        .config()
        .clone();
    config.household_cash = 0;
    config.firm_cash = 0;
    config.demography.annual_births_per_million_fertile_women = 0;
    config.demography.mortality.background_annual_ppm = [0; 5];
    config.demography.mortality.occupational_annual_ppm = [0; Good::COUNT];
    config.demography.mortality.famine_daily_ppm_at_thirty_days = 0;
    config.demography.mortality.cold_daily_ppm_at_seven_days = 0;
    config.population = Some(Population {
        households: (1..=u64::from(households))
            .map(|identity| HouseholdSeed {
                identity,
                position: Position { x: 0, y: 0 },
                cash: 0,
                inventories: Good::food([0, 0, 4]),
                fatigue: 0,
            })
            .collect(),
        firms: [Good::Wood, Good::Coal, Good::Bread]
            .into_iter()
            .enumerate()
            .map(|(index, good)| {
                let mut inventories = [0; Good::COUNT];
                inventories[good as usize - 1] = 1000;
                Ok(FirmSeed {
                    identity: u64::from(households) + index as u64 + 1,
                    position: Position { x: 0, y: 0 },
                    owner: index as u64 % u64::from(households) + 1,
                    good,
                    cash: 0,
                    inventories,
                    capacity: 1,
                    technique: Technique::new(
                        60,
                        match good.inputs()[0] {
                            None => 0,
                            Some(_) => 1,
                        },
                        1,
                    )?,
                    wage: 1,
                    price: 10,
                })
            })
            .collect::<Result<Vec<_>>>()?,
    });
    Scenario::new(config)
}

fn fuel(state: &mut State, household: usize, good: Good, quantity: u64) {
    let supplier = state
        .actors
        .iter()
        .position(|actor| actor.kind == good as u32)
        .unwrap();
    let stock = state.get(Field::inventory(good), supplier);
    state.set(Field::inventory(good), supplier, stock - quantity);
    state.set(
        Field::inventory(good),
        household,
        state.get(Field::inventory(good), household) + quantity,
    );
}

#[test]
fn cold_households_burn_substitute_fuels_and_conserve_every_unit() -> Result<()> {
    let scenario = heating_scenario(5)?;
    let mut initial = scenario.initial()?;
    fuel(&mut initial, 0, Good::Wood, 3);
    fuel(&mut initial, 1, Good::Coal, 2);
    fuel(&mut initial, 2, Good::Coal, 1);
    fuel(&mut initial, 2, Good::Wood, 1);
    fuel(&mut initial, 4, Good::Wood, 1);
    let checkpoint = Checkpoint::new(&scenario, &initial)?;
    let mut simulation = Simulation::from_checkpoint(checkpoint, support::device(64)?)?;
    simulation.advance()?;
    let state = simulation.download()?;
    for household in 0..5 {
        assert_eq!(state.get(Field::HeatingNeed, household), 3);
    }
    assert_eq!(
        (0..5)
            .map(|actor| state.get(Field::HeatingServed, actor))
            .collect::<Vec<_>>(),
        [3, 3, 3, 0, 1]
    );
    assert_eq!(simulation.snapshot.consumption[Good::Coal as usize - 1], 3);
    assert_eq!(simulation.snapshot.consumption[Good::Wood as usize - 1], 5);
    assert_eq!(
        state
            .people
            .iter()
            .map(|person| person.cold_exposure)
            .collect::<Vec<_>>(),
        [0, 0, 0, 1000, 0]
    );
    assert_eq!(simulation.snapshot.population.cold_exposure, 1000);
    assert_eq!(
        simulation.snapshot,
        ObservationSnapshot::from_state(&scenario, &state)?
    );
    simulation.checkpoint()?.restore()?;
    Ok(())
}

#[test]
fn unheated_exposure_has_a_distinct_cause_and_checkpoint_replay() -> Result<()> {
    let mut config = heating_scenario(3)?.config().clone();
    config.demography.mortality.cold_daily_ppm_at_seven_days = 100_000;
    config.demography.mortality.cold_vulnerability_per_mille = [4000; 5];
    let scenario = Scenario::new(config)?;
    let mut initial = scenario.initial()?;
    for person in &mut initial.people {
        person.cold_exposure = 28_000;
    }
    let checkpoint = Checkpoint::new(&scenario, &initial)?;
    let mut original = Simulation::from_checkpoint(checkpoint.clone(), support::device(64)?)?;
    let mut replay = Simulation::from_checkpoint(checkpoint, support::device(64)?)?;
    original.advance()?;
    replay.advance()?;
    let state = original.download()?;
    assert_eq!(state, replay.download()?);
    assert_eq!(original.snapshot.population.deaths, [0, 0, 0, 3]);
    assert_eq!(original.snapshot.population.living, 0);
    assert!(state.people.iter().all(|person| matches!(
        person.state,
        LivingState::Deceased {
            day: 1,
            cause: DeathCause::Cold
        }
    )));
    assert_eq!(
        original.snapshot,
        ObservationSnapshot::from_state(&scenario, &state)?
    );
    original.checkpoint()?.restore()?;
    Ok(())
}

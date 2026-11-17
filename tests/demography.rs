#![cfg(feature = "compute")]
use super::support;
use anyhow::Result;
use bad_econ::{domain::*, runtime::SimulationCommand, simulation::Simulation};

#[test]
fn births_grow_the_person_registry_and_checkpoint_without_creating_money() -> Result<()> {
    let mut config = Scenario::baseline(2000, 19, Experiment::Concentrated)?
        .config()
        .clone();
    config.demography.annual_births_per_million_fertile_women = 1_000_000;
    config.demography.mortality.background_annual_ppm = [0; 5];
    config.demography.mortality.cold_daily_ppm_at_seven_days = 0;
    config.demography.mortality.occupational_annual_ppm = [0; Good::COUNT];
    let scenario = Scenario::new(config)?;
    let initial = scenario.initial()?;
    let mut original = Simulation::new(scenario.clone(), support::device(64)?)?;
    for _ in 0..7 {
        original.advance()?;
    }
    assert!(original.snapshot.population.births > 0);
    assert!(original.snapshot.population.dependents > 0);
    assert_eq!(original.snapshot.cash, initial.initial_cash);
    let state = original.download()?;
    assert_eq!(
        state.people.len() as u64,
        initial.initial_population + original.snapshot.population.births
    );
    assert!(
        state
            .people
            .iter()
            .filter(|person| person.age_days < 7)
            .all(|person| person.work_minutes == 0 && person.employer.is_none())
    );
    let checkpoint = original.checkpoint()?;
    let mut resumed = Simulation::new(scenario, support::device(64)?)?;
    resumed.command(SimulationCommand::Restore(Box::new(checkpoint)))?;
    original.advance()?;
    resumed.advance()?;
    assert_eq!(original.download()?, resumed.download()?);
    Ok(())
}

#[test]
fn household_formation_and_inheritance_preserve_assets_and_pending_deliveries() -> Result<()> {
    use bad_econ::domain::demography::{PersonSeed, Sex};
    use bad_econ::domain::enterprise::{EnterpriseState, HouseholdProduction};
    let mut config = Scenario::baseline(3, 3, Experiment::Concentrated)?
        .config()
        .clone();
    config.household_cash = 10_000;
    config.household_production = HouseholdProduction::industrial();
    config.shipment_speed = 500;
    config.demography.annual_births_per_million_fertile_women = 0;
    config.demography.mortality.background_annual_ppm = [0; 5];
    config.demography.mortality.cold_daily_ppm_at_seven_days = 0;
    config.demography.mortality.occupational_annual_ppm = [0; Good::COUNT];
    config.demography.mortality.famine_daily_ppm_at_thirty_days = 100_000;
    config.demography.mortality.famine_vulnerability_per_mille = [4000; 5];
    config.demography.people = vec![
        PersonSeed {
            identity: 100,
            household: 1,
            age_days: 30 * 365,
            sex: Sex::Female,
        },
        PersonSeed {
            identity: 365,
            household: 1,
            age_days: 18 * 365,
            sex: Sex::Male,
        },
        PersonSeed {
            identity: 200,
            household: 2,
            age_days: 30 * 365,
            sex: Sex::Male,
        },
        PersonSeed {
            identity: 300,
            household: 3,
            age_days: 30 * 365,
            sex: Sex::Female,
        },
    ];
    let scenario = Scenario::new(config)?;
    let mut simulation = Simulation::new(scenario.clone(), support::device(64)?)?;
    simulation.advance()?;
    let mut state = simulation.download()?;
    assert_eq!(state.actors.len(), scenario.actors() + 1);
    let child = scenario.actors();
    assert_eq!(state.get(Field::ParentHousehold, child), 1);
    assert_eq!(
        state
            .people
            .iter()
            .find(|person| person.identity == 365)
            .unwrap()
            .household,
        state.actors.at(child).identity
    );
    state
        .people
        .iter_mut()
        .find(|person| person.identity == 100)
        .unwrap()
        .famine_exposure = 90_000;
    let bread = state.get(Field::Bread, 0);
    assert!(bread > 0);
    state.set(Field::Bread, 0, bread - 1);
    let slot = ((state.day + 1) % scenario.ring() as u64) as usize;
    let delivery = state
        .shipment_coordinate(Good::Bread, slot, scenario.ring(), 0)
        .0 as usize;
    state.values[delivery] += 1;
    state.values[(TRANSIT_FIELDS + Good::Bread as usize - 1) * state.actors.len()] += 1;
    state.set(Field::Transit, 0, state.get(Field::Transit, 0) + 1);
    let prior_cash = state.initial_cash;
    state.set(Field::EnterpriseStatus, 3, EnterpriseState::Forming as u64);
    state.set(Field::ConstructionRemaining, 3, 960);
    let checkpoint = Checkpoint::new(&scenario, &state)?;
    simulation.command(SimulationCommand::Restore(Box::new(checkpoint)))?;
    simulation.advance()?;
    let inherited = simulation.download()?;
    assert_eq!(simulation.snapshot.population.deaths[2], 1);
    assert_eq!(inherited.get(Field::Cash, 0), 0);
    assert_eq!(inherited.get(Field::EstateSettled, 0), 1);
    assert_eq!(inherited.actors.at(3).owner, child as u32);
    assert_eq!(simulation.snapshot.cash, prior_cash);
    let inherited_delivery = inherited
        .shipment_coordinate(Good::Bread, slot, scenario.ring(), child)
        .0 as usize;
    assert!(inherited.values[inherited_delivery] > 0);
    assert_eq!(inherited.values[delivery], 0);
    simulation.checkpoint()?.restore()?;
    simulation.advance()?;
    let delivered = simulation.download()?;
    assert_eq!(delivered.values[inherited_delivery], 0);
    assert_eq!(delivered.get(Field::InvestmentFirm, child), 4);
    assert!(
        delivered.get(Field::ConstructionRemaining, 3)
            < inherited.get(Field::ConstructionRemaining, 3)
    );
    assert!(delivered.get(Field::GrossIncome, child) > 0);
    Ok(())
}

#[test]
fn exhausted_person_growth_budget_restores_the_complete_audited_day() -> Result<()> {
    use bad_econ::simulation::{Device, MemoryBudget, Trace};
    let mut config = Scenario::baseline(3000, 19, Experiment::Concentrated)?
        .config()
        .clone();
    config.demography.annual_births_per_million_fertile_women = 1_000_000;
    config.demography.mortality.background_annual_ppm = [0; 5];
    config.demography.mortality.cold_daily_ppm_at_seven_days = 0;
    let mut simulation = Simulation::new(Scenario::new(config)?, Device::headless(64)?)?;
    let initial = simulation.download()?;
    for phase in Phase::ALL
        .into_iter()
        .filter(|phase| *phase != Phase::Audit)
    {
        simulation.phase(phase, Trace::Record)?;
    }
    let prior_budget = simulation.device.memory_budget;
    simulation.device.memory_budget = MemoryBudget::Reported {
        available_bytes: simulation.memory_usage().bytes_in_use * 2,
    };
    let error = simulation.phase(Phase::Audit, Trace::Record).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("person registry capacity exhausted")
    );
    simulation.device.memory_budget = prior_budget;
    assert_eq!(simulation.download()?, initial);
    simulation.checkpoint()?.restore()?;
    Ok(())
}

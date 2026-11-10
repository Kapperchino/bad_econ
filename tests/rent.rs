#![cfg(feature = "compute")]
use crate::support;
use anyhow::Result;
use bad_econ::{
    domain::{enterprise::HouseholdProduction, *},
    simulation::{Simulation, Trace},
};

#[test]
fn households_buy_food_before_paying_rent_on_gpu() -> Result<()> {
    let mut config = Scenario::baseline(1, 42, Experiment::Concentrated)?
        .config()
        .clone();
    config.rent = Some(RentPolicy {
        household_daily: 60,
        firm_daily: 60,
    });
    config.distribution_rate = 0;
    config.forecast_weight = 0;
    config.behavior.price_response = 0;
    config.demography.annual_births_per_million_fertile_women = 0;
    config.demography.mortality.background_annual_ppm = [0; 5];
    config.demography.mortality.occupational_annual_ppm = [0; Good::COUNT];
    config.population = Some(Population {
        households: vec![HouseholdSeed {
            identity: 1,
            position: Position { x: 0, y: 0 },
            cash: 30,
            inventories: Good::food([0, 0, 1]),
            fatigue: 0,
        }],
        firms: [Good::Bread, Good::Grain, Good::Flour]
            .into_iter()
            .enumerate()
            .map(|(index, good)| {
                Ok(FirmSeed {
                    identity: 2 + index as u64,
                    position: Position { x: 0, y: 0 },
                    owner: 1,
                    good,
                    cash: 0,
                    inventories: match good {
                        Good::Bread => Good::food([0, 0, 100]),
                        _ => [0; Good::COUNT],
                    },
                    capacity: 1,
                    technique: Technique::new(60, u32::from(good != Good::Grain), 1)?,
                    wage: 1,
                    price: 10,
                })
            })
            .collect::<Result<_>>()?,
    });
    let scenario = Scenario::new(config)?;
    let initial = scenario.initial()?;
    let mut simulation = Simulation::new(scenario.clone(), support::device(64)?)?;
    simulation.advance()?;
    let state = simulation.download()?;
    assert_eq!(state.get(Field::Purchased, 0), 2);
    assert_eq!(state.get(Field::Transit, 0), 2);
    assert_eq!(state.get(Field::Cash, 0), 0);
    let landlord = state.get(Field::Landlord, 0) as usize - 1;
    assert_eq!(state.get(Field::RentalIncome, landlord), 10);
    assert_eq!(state.get(Field::Cash, landlord), 10);
    assert_eq!(state.get(Field::RentTreasury, 0), 10);
    assert_eq!(state.get(Field::RentArrears, 0), 50);
    assert_eq!(simulation.snapshot.shortages, 0);
    assert_eq!(simulation.snapshot.cash, initial.initial_cash);
    simulation.advance()?;
    assert_eq!(simulation.snapshot.shortages, 0);
    Ok(())
}

#[test]
fn daily_rent_conserves_cash_and_replays_arrears_on_gpu() -> Result<()> {
    let mut config = Scenario::baseline(6, 42, Experiment::Concentrated)?
        .config()
        .clone();
    config.rent = Some(RentPolicy {
        household_daily: 60,
        firm_daily: 60,
    });
    config.household_production = HouseholdProduction::industrial();
    config.household_cash = 30;
    config.firm_cash = 600;
    config.distribution_rate = 0;
    config.demography.annual_births_per_million_fertile_women = 0;
    config.demography.mortality.background_annual_ppm = [0; 5];
    config.demography.mortality.occupational_annual_ppm = [0; Good::COUNT];
    config.demography.mortality.famine_daily_ppm_at_thirty_days = 0;
    let scenario = Scenario::new(config)?;
    let mut initial = scenario.initial()?;
    for actor in 6..initial.actors.len() {
        initial.set(Field::Demand, actor, 0);
    }
    let checkpoint = Checkpoint::new(&scenario, &initial)?;
    let mut simulation = Simulation::from_checkpoint(checkpoint, support::device(64)?)?;
    for phase in [
        Phase::Opening,
        Phase::Observation,
        Phase::Proposal,
        Phase::Commitment,
    ] {
        simulation.phase(phase, Trace::Record)?;
    }
    let before = simulation.download()?;
    assert_eq!(before.get(Field::RentTreasury, 1), 0);
    simulation.phase(Phase::WorkPayment, Trace::Record)?;
    let paid = simulation.download()?;
    for household in 0..6 {
        assert_eq!(paid.get(Field::Cash, household), 30);
        assert_eq!(paid.get(Field::RentTreasury, household), 0);
        assert_eq!(paid.get(Field::RentArrears, household), 0);
    }
    for firm in
        (6..paid.actors.len()).filter(|&firm| paid.actors.at(firm).kind != Good::Housing as u32)
    {
        assert_eq!(
            paid.get(Field::Cash, firm),
            before.get(Field::Cash, firm) - 60
        );
        assert_eq!(paid.get(Field::RentTreasury, firm), 60);
        assert_eq!(paid.get(Field::RentArrears, firm), 0);
    }
    simulation.phase(Phase::ProvisionPurchase, Trace::Record)?;
    let purchased = simulation.download()?;
    for household in 0..6 {
        match initial.get(Field::LandPlots, household) {
            0 => {
                assert_eq!(purchased.get(Field::Cash, household), 0);
                assert_eq!(purchased.get(Field::RentTreasury, household), 30);
                assert_eq!(purchased.get(Field::RentArrears, household), 30);
            }
            _ => {
                assert_eq!(purchased.get(Field::Cash, household), 30);
                assert_eq!(purchased.get(Field::RentTreasury, household), 0);
                assert_eq!(purchased.get(Field::RentArrears, household), 0);
            }
        }
    }
    for phase in [Phase::PhysicalClose, Phase::AccountsDecisions, Phase::Audit] {
        simulation.phase(phase, Trace::Record)?;
    }
    assert_eq!(simulation.snapshot.rent_collected, 270);
    assert_eq!(simulation.snapshot.rent_arrears, 90);
    assert_eq!(simulation.snapshot.cash, initial.initial_cash);
    let state = simulation.download()?;
    assert_eq!(
        simulation.snapshot,
        ObservationSnapshot::from_state(&scenario, &state)?
    );
    let request = InspectionRequest::new(2, &scenario)?;
    assert_eq!(
        simulation.inspect(request.clone())?,
        Inspection::from_state(&state, &request)?
    );
    let encoded = serde_json::to_string(&simulation.checkpoint()?)?;
    let mut resumed =
        Simulation::from_checkpoint(serde_json::from_str(&encoded)?, support::device(128)?)?;
    for day in 2..=4 {
        simulation.advance()?;
        resumed.advance()?;
        assert_eq!(simulation.download()?, resumed.download()?);
        assert_eq!(simulation.snapshot, resumed.snapshot);
        assert_eq!(
            simulation.snapshot.rent_collected + simulation.snapshot.rent_arrears,
            360 + 180 * (day - 1)
        );
        assert_eq!(simulation.snapshot.cash, initial.initial_cash);
    }
    Ok(())
}

#[test]
fn industrial_paycheck_households_avoid_startup_famine_at_a_million_on_gpu() -> Result<()> {
    let mut config = Scenario::industrial(1_000_000, 42, Experiment::Baseline)?
        .config()
        .clone();
    config.shipment_speed = config.search_radius;
    let scenario = Scenario::new(config)?;
    assert_eq!(scenario.ring(), 2);
    let mut simulation = Simulation::new(scenario, support::device(128)?)?;
    let initial_cash = simulation.snapshot.cash;
    for day in 1..=120 {
        simulation.advance()?;
        assert_eq!(simulation.snapshot.cash, initial_cash);
        assert!(
            simulation.snapshot.shortages < 20_000,
            "day {day}: {} unmet bread units",
            simulation.snapshot.shortages
        );
    }
    assert!(simulation.snapshot.population.deaths[2] < 3_000);
    assert!(simulation.snapshot.employed > 400_000);
    assert!(simulation.snapshot.rent_collected > 0);
    assert!(simulation.snapshot.rent_arrears < simulation.snapshot.rent_collected / 5);
    Ok(())
}

#[test]
fn industrial_rent_defaults_and_custom_rates_run_on_gpu() -> Result<()> {
    let generated = Scenario::industrial(24, 42, Experiment::Concentrated)?;
    assert_eq!(generated.config().rent, Some(RentPolicy::default()));
    let mut config = generated.config().clone();
    config.rent = Some(RentPolicy {
        household_daily: 7,
        firm_daily: 11,
    });
    config.research = None;
    let scenario = Scenario::new(config)?;
    let initial = scenario.initial()?;
    let households = initial
        .actors
        .iter()
        .enumerate()
        .filter(|(actor, metadata)| {
            metadata.kind == 0 && initial.get(Field::LandPlots, *actor) == 0
        })
        .count() as u64;
    let firms = initial
        .actors
        .iter()
        .filter(|actor| actor.kind > 0 && actor.kind != Good::Housing as u32)
        .count() as u64;
    let mut simulation = Simulation::new(scenario.clone(), support::device(64)?)?;
    simulation.advance()?;
    assert_eq!(
        simulation.snapshot.rent_collected + simulation.snapshot.rent_arrears,
        households * 7 + firms * 11
    );
    let state = simulation.download()?;
    assert_eq!(
        simulation.snapshot,
        ObservationSnapshot::from_state(&scenario, &state)?
    );
    simulation.checkpoint()?.restore()?;
    Ok(())
}

#[test]
fn rent_overflow_restores_last_audited_state_on_gpu() -> Result<()> {
    let mut config = Scenario::baseline(6, 42, Experiment::Concentrated)?
        .config()
        .clone();
    config.rent = Some(RentPolicy::default());
    let scenario = Scenario::new(config)?;
    let mut initial = scenario.initial()?;
    initial.set(Field::RentArrears, 0, u64::MAX);
    let checkpoint = Checkpoint::new(&scenario, &initial)?;
    let mut simulation = Simulation::from_checkpoint(checkpoint, support::device(64)?)?;
    assert!(simulation.advance().is_err());
    assert_eq!(simulation.download()?, initial);
    assert_eq!(simulation.snapshot.day, 0);
    Ok(())
}

#[test]
fn firms_reserve_rent_before_hiring_on_gpu() -> Result<()> {
    for cash in [400, 420] {
        let mut config = Scenario::baseline(1, 42, Experiment::Concentrated)?
            .config()
            .clone();
        config.rent = Some(RentPolicy {
            household_daily: 0,
            firm_daily: 60,
        });
        config.firm_cash = cash;
        let scenario = Scenario::new(config)?;
        let mut initial = scenario.initial()?;
        initial.set(Field::Demand, 1, 1000);
        let mut simulation = Simulation::from_checkpoint(
            Checkpoint::new(&scenario, &initial)?,
            support::device(64)?,
        )?;
        for phase in [Phase::Opening, Phase::Observation, Phase::Proposal] {
            simulation.phase(phase, Trace::Record)?;
        }
        let planned = simulation.download()?;
        assert_eq!(planned.get(Field::Plan, 1), (cash - 60) / 360);
        simulation.phase(Phase::Commitment, Trace::Record)?;
        simulation.phase(Phase::WorkPayment, Trace::Record)?;
        let paid = simulation.download()?;
        for firm in
            (1..paid.actors.len()).filter(|&firm| paid.actors.at(firm).kind != Good::Housing as u32)
        {
            assert_eq!(paid.get(Field::RentTreasury, firm), 60);
            assert_eq!(paid.get(Field::RentArrears, firm), 0);
        }
    }
    Ok(())
}

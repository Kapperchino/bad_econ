use super::support;
use anyhow::Result;
use bad_econ::domain::*;

fn scenario() -> Result<Scenario> {
    let mut config = Scenario::baseline(3, 1, Experiment::Concentrated)?
        .config()
        .clone();
    config.payroll = Some(PayrollPolicy {
        base_rate: 100,
        employee_charge: 50,
        brackets: vec![TaxBracket {
            threshold: 100,
            marginal_rate: 300,
        }],
    });
    config.distribution_rate = 0;
    config.forecast_weight = 0;
    config.coverage = 30;
    config.behavior.work_fatigue = 0;
    config.behavior.wage_increment = 0;
    config.resources = ResourceMap::Explicit {
        regions: vec![ResourceRegion::new(
            Good::Wood,
            Position { x: 0, y: 0 },
            Position {
                x: config.map_size,
                y: config.map_size,
            },
            config.map_size,
        )?],
    };
    config.population = Some(Population {
        households: (1..=3)
            .map(|identity| HouseholdSeed {
                identity,
                position: Position { x: 0, y: 0 },
                cash: 0,
                inventories: Good::food([0; 3]),
                fatigue: 0,
            })
            .collect(),
        firms: (4..=6)
            .map(|identity| FirmSeed {
                identity,
                position: Position { x: 0, y: 0 },
                owner: 1,
                good: Good::Wood,
                cash: 100_000,
                inventories: Good::food([0; 3]),
                capacity: 1,
                technique: Technique {
                    labor_minutes: 7,
                    input_ticks: 0,
                    output_ticks: 1,
                },
                wage: 1,
                price: 1,
            })
            .collect(),
    });
    Scenario::new(config)
}

#[test]
fn graduated_payroll_conserves_fractional_claims_across_checkpoints() -> Result<()> {
    use bad_econ::{runtime::SimulationCommand, simulation::Simulation};
    let mut simulation = Simulation::new(scenario()?, support::device(64)?)?;
    let opening = simulation.snapshot.cash;
    let mut incumbents = Vec::new();
    for day in 1..=16 {
        let gross = 7u64 * day;
        let tax_ticks = gross.min(100) * 100 + gross.saturating_sub(100) * 300;
        let charge_ticks = gross * 50;
        let net = (gross * 1000 - tax_ticks - charge_ticks) / 1000;
        let tax = tax_ticks / 1000;
        let charge = charge_ticks / 1000;
        simulation.advance()?;
        let state = simulation.download()?;
        let employers: Vec<_> = (0..3).map(|row| state.get(Field::Employer, row)).collect();
        if day == 1 {
            incumbents = employers.clone();
        }
        assert_eq!(employers, incumbents);
        for household in 0..3 {
            assert_eq!(
                state.get(Field::JobDaysRemaining, household),
                (14 - day % 14) % 14
            );
            assert_eq!(state.get(Field::GrossIncome, household), gross);
            assert_eq!(state.get(Field::Cash, household), net);
            assert_eq!(state.get(Field::TaxTreasury, household), tax);
            assert_eq!(state.get(Field::ChargeTreasury, household), charge);
            assert_eq!(
                state.get(Field::PayrollReserve, household),
                gross - net - tax - charge
            );
        }
        assert_eq!(simulation.snapshot.income_tax, 3 * tax);
        assert_eq!(simulation.snapshot.employee_charges, 3 * charge);
        assert_eq!(simulation.snapshot.cash, opening);
        assert_eq!(
            simulation.snapshot,
            ObservationSnapshot::from_state(&simulation.scenario, &state)?
        );
    }
    let checkpoint = simulation.checkpoint()?;
    simulation.advance()?;
    let expected = simulation.download()?;
    simulation.command(SimulationCommand::Restore(checkpoint.into()))?;
    simulation.advance()?;
    assert_eq!(simulation.download()?, expected);
    Ok(())
}

#[test]
fn regular_hires_review_competing_offers_at_expiry_and_can_replace_lost_shifts_on_gpu() -> Result<()>
{
    use bad_econ::simulation::Simulation;
    let mut config = scenario()?.config().clone();
    config.economy = Economy::Industrial;
    config.search_radius = 50;
    let population = config.population.as_mut().unwrap();
    population.households[1].position.x = 50;
    population.households[2].position.x = 200;
    population.firms[1].wage = 3;
    population.firms[2].position.x = 100;
    population.firms[2].inventories[Good::Wood as usize - 1] = 1000;
    let scenario = Scenario::new(config)?;
    let mut state = scenario.initial()?;
    state.set(Field::Job, 0, 4);
    state.set(Field::JobDaysRemaining, 0, 1);
    state.set(Field::Job, 1, 6);
    state.set(Field::JobDaysRemaining, 1, 5);
    let device = support::device(64)?;
    let mut simulation =
        Simulation::from_checkpoint(Checkpoint::new(&scenario, &state)?, device.clone())?;
    simulation.advance()?;
    let last_day = simulation.download()?;
    assert_eq!(last_day.get(Field::Employer, 0), 4);
    assert_eq!(last_day.get(Field::JobDaysRemaining, 0), 0);
    assert_eq!(last_day.get(Field::JobDaysRemaining, 1), 4);
    assert_eq!(last_day.get(Field::Job, 1), 6);
    for field in [
        Field::Employer,
        Field::Employed,
        Field::Hours,
        Field::GrossIncome,
    ] {
        assert_eq!(last_day.get(field, 1), 0);
    }
    simulation.advance()?;
    let mut renewed = simulation.download()?;
    assert_eq!(renewed.get(Field::Employer, 0), 5);
    assert_eq!(renewed.get(Field::JobDaysRemaining, 0), 13);
    let cash = renewed.get(Field::Cash, 3) + renewed.get(Field::Cash, 4);
    renewed.set(Field::Cash, 3, cash);
    renewed.set(Field::Cash, 4, 0);
    let mut resumed = Simulation::from_checkpoint(Checkpoint::new(&scenario, &renewed)?, device)?;
    resumed.advance()?;
    let replacement = resumed.download()?;
    assert_eq!(replacement.get(Field::Employer, 0), 4);
    assert_eq!(replacement.get(Field::JobDaysRemaining, 0), 13);
    Ok(())
}

#[test]
fn farm_service_day_labour_and_colliery_hires_use_different_terms_on_gpu() -> Result<()> {
    use bad_econ::{
        domain::enterprise::{LandAccess, LandGrant},
        simulation::Simulation,
    };
    let mut config = scenario()?.config().clone();
    config.search_radius = 50;
    config.land_access = Some(LandAccess {
        total_plots: 1,
        allocations: vec![LandGrant {
            household: 2,
            plots: 1,
        }],
    });
    config.resources = ResourceMap::Explicit {
        regions: [Good::Grain, Good::Coal]
            .into_iter()
            .map(|good| {
                ResourceRegion::new(
                    good,
                    Position { x: 0, y: 0 },
                    Position {
                        x: config.map_size,
                        y: config.map_size,
                    },
                    config.map_size,
                )
            })
            .collect::<Result<_>>()?,
    };
    let population = config.population.as_mut().unwrap();
    for (row, good) in [Good::Grain, Good::Grain, Good::Coal]
        .into_iter()
        .enumerate()
    {
        let x = 100 * row as u32;
        population.households[row].position.x = x;
        population.firms[row].position.x = x;
        population.firms[row].good = good;
    }
    let scenario = Scenario::new(config)?;
    let mut state = scenario.initial()?;
    state.day = 100;
    let device = support::device(64)?;
    let mut simulation =
        Simulation::from_checkpoint(Checkpoint::new(&scenario, &state)?, device.clone())?;
    simulation.advance()?;
    let first = simulation.download()?;
    for (row, remaining) in [364, 0, 29].into_iter().enumerate() {
        assert_eq!(first.get(Field::Employer, row), row as u64 + 4);
        assert_eq!(first.get(Field::JobDaysRemaining, row), remaining);
        let request = InspectionRequest::new(row as u64 + 1, &scenario)?;
        assert_eq!(simulation.inspect(request)?.job_days_remaining, remaining);
    }
    assert_eq!(first.get(Field::Job, 1), 0);
    let mut invalid = first.clone();
    invalid.set(Field::JobDaysRemaining, 2, 365);
    assert!(Checkpoint::new(&scenario, &invalid).is_err());
    let mut restored = Simulation::from_checkpoint(simulation.checkpoint()?, device)?;
    restored.advance()?;
    let second = restored.download()?;
    assert_eq!(second.get(Field::JobDaysRemaining, 0), 363);
    assert_eq!(second.get(Field::JobDaysRemaining, 2), 28);
    Ok(())
}

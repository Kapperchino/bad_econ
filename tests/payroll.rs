#![cfg(feature = "compute")]
use anyhow::Result;
use bad_econ::domain::*;

fn policy() -> PayrollPolicy {
    PayrollPolicy {
        base_rate: 100,
        employee_charge: 50,
        brackets: vec![TaxBracket {
            threshold: 100,
            marginal_rate: 300,
        }],
    }
}

#[cfg(feature = "compute")]
use super::support;

#[cfg(feature = "compute")]
fn scenario(minutes: u32) -> Result<Scenario> {
    let mut config = Scenario::baseline(3, 1, Experiment::Concentrated)?
        .config()
        .clone();
    config.payroll = Some(policy());
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
                    labor_minutes: minutes,
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

#[cfg(feature = "compute")]
#[test]
fn payroll_is_atomic_device_accounting_and_survives_checkpoints() -> Result<()> {
    use bad_econ::{runtime::SimulationCommand, simulation::Simulation};
    let mut simulation = Simulation::new(scenario(80)?, support::device(64)?)?;
    let opening = simulation.snapshot.cash;
    simulation.advance()?;
    let state = simulation.download()?;
    for household in 0..3 {
        assert_eq!(state.get(Field::Cash, household), 68);
        assert_eq!(state.get(Field::GrossIncome, household), 80);
        assert_eq!(state.get(Field::TaxTreasury, household), 8);
        assert_eq!(state.get(Field::ChargeTreasury, household), 4);
        assert_eq!(state.get(Field::PayrollReserve, household), 0);
    }
    assert_eq!(simulation.snapshot.income_tax, 24);
    assert_eq!(simulation.snapshot.employee_charges, 12);
    assert_eq!(simulation.snapshot.cash, opening);
    let checkpoint = simulation.checkpoint()?;
    (0..4).try_for_each(|_| simulation.advance())?;
    let expected = simulation.download()?;
    for household in 0..3 {
        assert_eq!(expected.get(Field::GrossIncome, household), 400);
        assert_eq!(expected.get(Field::TaxTreasury, household), 100);
        assert_eq!(expected.get(Field::ChargeTreasury, household), 20);
        assert_eq!(expected.get(Field::Cash, household), 280);
    }
    simulation.command(SimulationCommand::Restore(checkpoint.into()))?;
    (0..4).try_for_each(|_| simulation.advance())?;
    assert_eq!(simulation.download()?, expected);
    assert_eq!(simulation.snapshot.cash, opening);
    Ok(())
}

#[cfg(feature = "compute")]
#[test]
fn fractional_payroll_claims_remain_funded_across_small_payments() -> Result<()> {
    use bad_econ::simulation::Simulation;
    let mut simulation = Simulation::new(scenario(1)?, support::device(64)?)?;
    for gross in 1..=20u64 {
        simulation.advance()?;
        let state = simulation.download()?;
        for household in 0..3 {
            let net = gross * 850 / 1000;
            let tax = gross * 100 / 1000;
            let charge = gross * 50 / 1000;
            let reserve = gross - net - tax - charge;
            assert_eq!(state.get(Field::GrossIncome, household), gross);
            assert_eq!(state.get(Field::Cash, household), net);
            assert_eq!(state.get(Field::TaxTreasury, household), tax);
            assert_eq!(state.get(Field::ChargeTreasury, household), charge);
            assert_eq!(state.get(Field::PayrollReserve, household), reserve);
        }
        assert_eq!(
            simulation.snapshot,
            ObservationSnapshot::from_state(&simulation.scenario, &state)?
        );
    }
    Ok(())
}

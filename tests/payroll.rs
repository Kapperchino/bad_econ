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
    for day in 1..=16 {
        let gross = 7u64 * day;
        let tax_ticks = gross.min(100) * 100 + gross.saturating_sub(100) * 300;
        let charge_ticks = gross * 50;
        let net = (gross * 1000 - tax_ticks - charge_ticks) / 1000;
        let tax = tax_ticks / 1000;
        let charge = charge_ticks / 1000;
        simulation.advance()?;
        let state = simulation.download()?;
        for household in 0..3 {
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

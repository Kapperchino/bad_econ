#![cfg(feature = "compute")]
use super::support;
use anyhow::Result;
use bad_econ::{
    domain::{enterprise::Distributor, *},
    simulation::Simulation,
};

fn economy() -> Result<Scenario> {
    let mut config = Scenario::baseline(3, 17, Experiment::Baseline)?
        .config()
        .clone();
    config.search_radius = 600;
    config.shipment_speed = 600;
    config.coverage = 2;
    config.distribution_rate = 0;
    config.forecast_weight = 0;
    config.behavior.price_response = 0;
    config.demography.annual_births_per_million_fertile_women = 0;
    config.demography.mortality.background_annual_ppm = [0; 5];
    config.demography.mortality.occupational_annual_ppm = [0; Good::COUNT];
    config.population = Some(Population {
        households: (1..=3)
            .map(|identity| HouseholdSeed {
                identity,
                position: Position { x: 999, y: 0 },
                cash: 10_000,
                inventories: [0; Good::COUNT],
                fatigue: 0,
            })
            .collect(),
        firms: vec![
            FirmSeed {
                identity: 4,
                position: Position { x: 0, y: 0 },
                owner: 1,
                good: Good::Bread,
                cash: 10_000,
                inventories: Good::food([0, 200, 200]),
                capacity: 2,
                technique: Technique::new(60, 10, 10)?,
                wage: 1,
                price: 10,
            },
            FirmSeed {
                identity: 5,
                position: Position { x: 500, y: 0 },
                owner: 2,
                good: Good::Bread,
                cash: 10_000,
                inventories: [0; Good::COUNT],
                capacity: 1,
                technique: Technique::new(60, 10, 10)?,
                wage: 1,
                price: 15,
            },
            FirmSeed {
                identity: 6,
                position: Position { x: 0, y: 0 },
                owner: 3,
                good: Good::Grain,
                cash: 0,
                inventories: [0; Good::COUNT],
                capacity: 1,
                technique: Technique::new(60, 0, 10)?,
                wage: 1,
                price: 5,
            },
        ],
    });
    config.distributors = vec![Distributor {
        firm: 5,
        good: Good::Bread,
        handling_minutes: 60,
        units_per_shift: 24,
        reserve_days: 3,
        margin_per_mille: 250,
    }];
    Scenario::new(config)
}

#[test]
fn grocery_restock_delivery_staffing_and_resale_conserve_food_and_cash() -> Result<()> {
    let scenario = economy()?;
    let mut simulation = Simulation::new(scenario.clone(), support::device(64)?)?;
    simulation.advance()?;
    let first = simulation.download()?;
    assert_eq!(first.get(Field::Bread, 4), 0);
    assert!(first.get(Field::Transit, 4) > 0);
    assert_eq!(first.get(Field::Sales, 4), 0);
    simulation.advance()?;
    let second = simulation.download()?;
    assert!(second.get(Field::Sales, 4) > 0);
    assert_eq!(second.get(Field::Produced, 4), 0);
    assert_eq!(simulation.snapshot.produced[Good::Bread as usize - 1], 0);
    assert!(second.get(Field::RetailOpen, 4) > 0);
    assert_eq!(second.get(Field::ReplacementCost, 4), 10);
    assert!(second.get(Field::Price, 4) > 10);
    simulation.advance()?;
    assert!(simulation.snapshot.consumed > 0);
    let final_state = simulation.download()?;
    assert_eq!(
        simulation.snapshot,
        ObservationSnapshot::from_state(&scenario, &final_state)?
    );
    Ok(())
}

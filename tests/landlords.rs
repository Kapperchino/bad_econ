#![cfg(feature = "compute")]
use super::support;
use anyhow::Result;
use bad_econ::{
    domain::{
        property::{ProjectStatus, PropertyConfig, PropertyRegion, PropertySnapshot},
        *,
    },
    simulation::Simulation,
};

fn rentals() -> Result<ScenarioConfig> {
    let mut config = Scenario::baseline(8, 42, Experiment::Concentrated)?
        .config()
        .clone();
    config.economy = Economy::Industrial;
    config.firms = 3;
    config.expansion = enterprise::ExpansionPolicy {
        small_business_per_mille: 1000,
        industrial_per_mille: 1000,
    };
    config.prices = [1; 3];
    config.coverage = 2;
    config.forecast_weight = 0;
    config.distribution_rate = 0;
    config.behavior.price_response = 0;
    config.entrepreneurship.interested_per_mille = 0;
    config.rent = Some(RentPolicy {
        household_daily: 10,
        firm_daily: 10,
    });
    config.demography.annual_births_per_million_fertile_women = 0;
    config.demography.mortality.background_annual_ppm = [0; 5];
    config.demography.mortality.occupational_annual_ppm = [0; Good::COUNT];
    let mut materials = [0; Good::COUNT];
    materials[Good::Construction as usize - 1] = 100;
    config.population = Some(Population {
        households: (1..=8)
            .map(|identity| HouseholdSeed {
                identity,
                position: Position { x: 250, y: 250 },
                cash: match identity {
                    1 => 3000,
                    _ => 1000,
                },
                inventories: Good::food([0, 0, 100]),
                fatigue: 0,
            })
            .collect(),
        firms: vec![
            FirmSeed {
                identity: 101,
                position: Position { x: 250, y: 250 },
                owner: 2,
                good: Good::Construction,
                cash: 1000,
                inventories: materials,
                capacity: 1,
                technique: Technique::new(60, 1, 1)?,
                wage: 1,
                price: Good::Construction.initial_price(),
            },
            FirmSeed {
                identity: 102,
                position: Position { x: 250, y: 250 },
                owner: 2,
                good: Good::Grain,
                cash: 1000,
                inventories: [0; Good::COUNT],
                capacity: 1,
                technique: Technique::new(60, 0, 1)?,
                wage: 1,
                price: 1,
            },
            FirmSeed {
                identity: 103,
                position: Position { x: 250, y: 250 },
                owner: 1,
                good: Good::Housing,
                cash: 20_000,
                inventories: [0; Good::COUNT],
                capacity: 0,
                technique: Technique::housing(8)?,
                wage: 1,
                price: 10,
            },
        ],
    });
    Ok(config)
}

fn region(plots: u32) -> PropertyConfig {
    PropertyConfig {
        regions: vec![PropertyRegion {
            lower: Position { x: 0, y: 0 },
            upper: Position { x: 1000, y: 1000 },
            plots,
            landowner: 1,
        }],
        buildings: vec![],
        orders: vec![],
    }
}

#[test]
fn indexed_rentals_preserve_nearest_buildings_and_cyclic_ties_on_gpu() -> Result<()> {
    let mut config = rentals()?;
    let population = config.population.as_mut().unwrap();
    let building = population.firms.pop().unwrap();
    population.firms.extend((0..17).map(|index| FirmSeed {
        identity: 103 + index,
        position: match index % 4 {
            0 => Position { x: 200, y: 250 },
            1 => Position { x: 300, y: 250 },
            2 => Position { x: 250, y: 200 },
            _ => Position { x: 250, y: 300 },
        },
        ..building.clone()
    }));
    for household in population.households.iter_mut().skip(4) {
        household.position = Position {
            x: household.identity as u32 * 71,
            y: 150 + household.identity as u32 * 21,
        };
    }
    config.firms = population.firms.len() as u32;
    let scenario = Scenario::new(config)?;
    let initial = scenario.initial()?;
    let buildings = initial
        .actors
        .iter()
        .enumerate()
        .filter(|(_, actor)| actor.kind == Good::Housing as u32)
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    let mut simulation = Simulation::new(scenario, support::device(64)?)?;
    simulation.advance()?;
    let state = simulation.download()?;
    for (index, actor) in initial
        .actors
        .iter()
        .enumerate()
        .filter(|(_, actor)| actor.kind != Good::Housing as u32)
    {
        let household = match actor.kind {
            0 => index,
            _ => actor.owner as usize,
        };
        let expected = buildings
            .iter()
            .copied()
            .cycle()
            .skip(household % buildings.len())
            .take(buildings.len())
            .min_by_key(|&building| {
                actor
                    .position
                    .distance(initial.actors.at(building).position)
            })
            .unwrap();
        assert_eq!(
            state.get(Field::Landlord, index),
            expected as u64 + 1,
            "tenant {index}"
        );
    }
    Ok(())
}

#[test]
fn rental_income_funds_paid_housing_and_replays_on_gpu() -> Result<()> {
    for property in [None, Some(region(14))] {
        let mut config = rentals()?;
        config.property = property;
        let scenario = Scenario::new(config)?;
        let initial = scenario.initial()?;
        let mut simulation = Simulation::new(scenario.clone(), support::device(64)?)?;
        simulation.advance()?;
        let ordered = simulation.download()?;
        assert_eq!(
            ordered.get(Field::RentalIncome, 10),
            100,
            "property {:?}, housing {}, plots {}, supply {}, household cash {}, bread {}, need {}, rent paid {}, arrears {}",
            scenario.config().property.is_some(),
            ordered.get(Field::Housing, 10),
            ordered.get(Field::OccupiedPlots, 10),
            ordered.get(Field::HousingAvailable, 0),
            ordered.get(Field::Cash, 0),
            ordered.get(Field::Bread, 0),
            ordered.get(Field::FoodNeed, 0),
            ordered.get(Field::RentTreasury, 0),
            ordered.get(Field::RentArrears, 0)
        );
        assert_eq!(
            ordered.get(Field::Cash, 10),
            initial.get(Field::Cash, 10) + 100 - 120
        );
        assert_eq!(ordered.get(Field::ConstructionRemaining, 10), 20);
        assert_eq!(ordered.get(Field::OutputTicks, 10), 8);
        assert_eq!(ordered.get(Field::RentalIncome, 1), 0);
        for actor in 0..ordered.actors.len() {
            assert_eq!(ordered.get(Field::RentPending, actor), 0);
            assert_eq!(
                ordered.get(Field::Landlord, actor),
                match actor {
                    10 => 0,
                    _ => 11,
                }
            );
        }
        let encoded = serde_json::to_string(&simulation.checkpoint()?)?;
        let mut resumed =
            Simulation::from_checkpoint(serde_json::from_str(&encoded)?, support::device(64)?)?;
        for _ in 0..3 {
            simulation.advance()?;
            resumed.advance()?;
            assert_eq!(simulation.download()?, resumed.download()?);
            assert_eq!(simulation.snapshot.cash, initial.initial_cash);
        }
        let completed = simulation.download()?;
        assert_eq!(
            completed.get(Field::ProjectStatus, 10),
            ProjectStatus::Completed as u64,
            "property {:?}, project {:?}, builder {:?}",
            scenario.config().property.is_some(),
            [
                Field::ConstructionRemaining,
                Field::Construction,
                Field::Transit,
                Field::Cash,
                Field::Purchased
            ]
            .map(|field| completed.get(field, 10)),
            [Field::Construction, Field::Price, Field::Sales].map(|field| completed.get(field, 8))
        );
        assert_eq!(completed.get(Field::ConstructionRemaining, 10), 0);
        assert_eq!(completed.get(Field::OccupiedPlots, 10), 2);
        assert_eq!(
            simulation.snapshot.used[Good::Construction as usize - 1],
            20
        );
        assert!(completed.get(Field::Cash, 8) > initial.get(Field::Cash, 8));
        assert_eq!(
            simulation.snapshot,
            ObservationSnapshot::from_state(&scenario, &completed)?
        );
        let inspection = InspectionRequest::new(103, &scenario)?;
        let inspected = simulation.inspect(inspection.clone())?;
        assert_eq!(
            inspected,
            Inspection::from_state(&scenario, &completed, &inspection)?
        );
        assert_eq!(inspected.rental_business.unwrap().homes, 16);
        if let Some(property) = PropertySnapshot::from_state(&scenario, &completed)? {
            assert_eq!(property.regions[0].housing, 2);
            assert_eq!(property.regions[0].reserved, 0);
        }
    }
    Ok(())
}

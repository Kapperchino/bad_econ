#![cfg(feature = "compute")]
use crate::support;
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
fn generated_rentals_house_local_renters_without_counting_subsistence_homes_on_gpu() -> Result<()> {
    let scenario = Scenario::industrial(10_000, 42, Experiment::Baseline)?;
    let initial = scenario.initial()?;
    let rentals = landlords::RentalRegistry::new(&scenario, &initial);
    for account in &rentals.accounts {
        let renters = account
            .tenants
            .iter()
            .filter(|&&tenant| {
                initial.actors.at(tenant).kind == 0 && initial.get(Field::LandPlots, tenant) == 0
            })
            .count() as u64;
        let units = initial.get(Field::OutputTicks, account.landlord);
        assert_eq!(
            initial.get(Field::OccupiedPlots, account.landlord),
            renters.div_ceil(units)
        );
        for &tenant in &account.tenants {
            let position = initial.actors.at(tenant).position;
            let distance = position.distance(initial.actors.at(account.landlord).position);
            assert!(rentals.accounts.iter().all(|other| {
                distance <= position.distance(initial.actors.at(other.landlord).position)
            }));
        }
    }
    let mut simulation = Simulation::new(scenario.clone(), support::device(64)?)?;
    simulation.advance()?;
    let state = simulation.download()?;
    for account in &rentals.accounts {
        let request =
            InspectionRequest::new(initial.actors.at(account.landlord).identity, &scenario)?;
        let inspected = simulation.inspect(request.clone())?;
        assert_eq!(inspected, Inspection::from_state(&state, &request)?);
        assert_eq!(
            inspected.business_scale,
            Some(enterprise::BusinessScale::Industrial)
        );
        assert_eq!(
            state.get(Field::ConstructionRemaining, account.landlord),
            20
        );
    }
    for _ in 0..3 {
        simulation.advance()?;
    }
    let built = simulation.download()?;
    assert!(rentals.accounts.iter().any(|account| {
        built.get(Field::OccupiedPlots, account.landlord)
            > initial.get(Field::OccupiedPlots, account.landlord)
    }));
    assert_eq!(simulation.snapshot.cash, initial.initial_cash);
    Ok(())
}

#[test]
fn urban_rental_shortages_fund_construction_with_default_expansion_on_gpu() -> Result<()> {
    let mut config = rentals()?;
    config.expansion = enterprise::ExpansionPolicy::default();
    config.households = 24;
    config.firms = 4;
    let population = config.population.as_mut().unwrap();
    let household = population.households[1].clone();
    population
        .households
        .extend((9..=24).map(|identity| HouseholdSeed {
            identity,
            ..household.clone()
        }));
    population.firms[2].owner = 3;
    population.firms[2].technique = Technique::housing(25)?;
    let mut distant = population.firms[2].clone();
    distant.identity = 104;
    distant.owner = 4;
    distant.position.x = 750;
    population.firms.push(distant);
    let scenario = Scenario::new(config)?;
    let mut simulation = Simulation::new(scenario.clone(), support::device(64)?)?;
    let cash = simulation.snapshot.cash;
    simulation.advance()?;
    let ordered = simulation.download()?;
    assert_eq!(ordered.get(Field::Demand, 26), 24);
    assert_eq!(ordered.get(Field::Demand, 27), 0);
    assert_eq!(ordered.get(Field::ConstructionRemaining, 26), 20);
    assert_eq!(ordered.get(Field::ConstructionRemaining, 27), 0);
    let mut replay = Simulation::from_checkpoint(simulation.checkpoint()?, support::device(32)?)?;
    for _ in 0..3 {
        simulation.advance()?;
        replay.advance()?;
        assert_eq!(simulation.download()?, replay.download()?);
    }
    let completed = simulation.download()?;
    assert_eq!(completed.get(Field::OccupiedPlots, 26), 2);
    assert_eq!(completed.get(Field::OccupiedPlots, 27), 1);
    assert_eq!(
        completed.get(Field::ProjectStatus, 26),
        ProjectStatus::Completed as u64
    );
    assert_eq!(
        simulation.snapshot.used[Good::Construction as usize - 1],
        20
    );
    assert_eq!(simulation.snapshot.cash, cash);
    let mut moving = completed;
    moving.set(Field::MoveDue, 0, 5);
    moving.set(Field::MoveX, 0, 750);
    moving.set(Field::MoveY, 0, 250);
    let mut simulation =
        Simulation::from_checkpoint(Checkpoint::new(&scenario, &moving)?, support::device(64)?)?;
    simulation.advance()?;
    let mut replay = Simulation::from_checkpoint(simulation.checkpoint()?, support::device(32)?)?;
    simulation.advance()?;
    replay.advance()?;
    let moved = simulation.download()?;
    assert_eq!(moved, replay.download()?);
    assert_eq!(moved.get(Field::Landlord, 0), 28);
    assert_eq!(moved.get(Field::Demand, 26), 23);
    assert_eq!(moved.get(Field::Demand, 27), 1);
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
            Simulation::from_checkpoint(serde_json::from_str(&encoded)?, support::device(32)?)?;
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
        assert_eq!(inspected, Inspection::from_state(&completed, &inspection)?);
        assert_eq!(inspected.rental_business.unwrap().homes, 16);
        if let Some(property) = PropertySnapshot::from_state(&scenario, &completed)? {
            assert_eq!(property.regions[0].housing, 2);
            assert_eq!(property.regions[0].reserved, 0);
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum Constraint {
    Vacancy,
    Savings,
    Supply,
    Land,
}

#[test]
fn rental_development_requires_demand_savings_construction_supply_and_land_on_gpu() -> Result<()> {
    for constraint in [
        Constraint::Vacancy,
        Constraint::Savings,
        Constraint::Supply,
        Constraint::Land,
    ] {
        let mut config = rentals()?;
        let expected = match constraint {
            Constraint::Vacancy => {
                config.population.as_mut().unwrap().firms[2].technique = Technique::housing(9)?;
                ProjectStatus::None
            }
            Constraint::Savings => {
                config.population.as_mut().unwrap().firms[2].cash = 0;
                ProjectStatus::None
            }
            Constraint::Supply => {
                config.population.as_mut().unwrap().firms[0].inventories
                    [Good::Construction as usize - 1] = 0;
                ProjectStatus::Building
            }
            Constraint::Land => {
                config.property = Some(region(3));
                ProjectStatus::Requested
            }
        };
        let scenario = Scenario::new(config)?;
        let mut simulation = Simulation::new(scenario, support::device(64)?)?;
        let cash = simulation.snapshot.cash;
        for _ in 0..4 {
            simulation.advance()?;
        }
        let state = simulation.download()?;
        assert_eq!(state.get(Field::ProjectStatus, 10), expected as u64);
        assert_eq!(state.get(Field::OccupiedPlots, 10), 1);
        assert_eq!(simulation.snapshot.used[Good::Construction as usize - 1], 0);
        assert_eq!(simulation.snapshot.cash, cash);
    }
    Ok(())
}

#[test]
fn small_owners_can_take_income_without_expanding_on_gpu() -> Result<()> {
    let mut config = rentals()?;
    config.expansion = enterprise::ExpansionPolicy {
        small_business_per_mille: 0,
        industrial_per_mille: 1000,
    };
    config.distribution_rate = 1000;
    config.reserve = 0;
    let scenario = Scenario::new(config)?;
    let mut simulation = Simulation::new(scenario.clone(), support::device(64)?)?;
    let initial = simulation.download()?;
    for _ in 0..3 {
        simulation.advance()?;
    }
    let state = simulation.download()?;
    assert_eq!(state.actors.at(10).capacity, 0);
    assert_eq!(state.get(Field::Plan, 10), 0);
    assert_eq!(state.get(Field::Slots, 10), 0);
    assert_eq!(state.get(Field::Hours, 10), 0);
    assert_eq!(
        state.get(Field::ProjectStatus, 10),
        ProjectStatus::None as u64
    );
    assert!(state.get(Field::Cash, 0) > initial.get(Field::Cash, 0));
    assert_eq!(simulation.snapshot.cash, initial.initial_cash);
    let policy = scenario.config().expansion;
    assert_eq!(
        policy.disposition(42, 1, enterprise::BusinessScale::SmallBusiness),
        enterprise::ExpansionDisposition::Maintain
    );
    assert_eq!(
        policy.disposition(42, 1, enterprise::BusinessScale::Industrial),
        enterprise::ExpansionDisposition::Expand
    );
    Ok(())
}

#[test]
fn large_landlords_expand_only_when_the_owner_wants_to_on_gpu() -> Result<()> {
    for preference in [
        enterprise::ExpansionDisposition::Maintain,
        enterprise::ExpansionDisposition::Expand,
    ] {
        let mut config = rentals()?;
        config.expansion = enterprise::ExpansionPolicy::default();
        config.distribution_rate = 1000;
        config.households = 24;
        let population = config.population.as_mut().unwrap();
        let household = population.households[1].clone();
        population
            .households
            .extend((9..=24).map(|identity| HouseholdSeed {
                identity,
                ..household.clone()
            }));
        population.firms[2].owner = match preference {
            enterprise::ExpansionDisposition::Maintain => 1,
            enterprise::ExpansionDisposition::Expand => 7,
        };
        population.firms[2].technique = Technique::housing(20)?;
        let scenario = Scenario::new(config)?;
        let mut simulation = Simulation::new(scenario.clone(), support::device(64)?)?;
        let initial = simulation.download()?;
        let landlord = 26;
        let owner = initial.actors.at(landlord).owner as usize;
        simulation.advance()?;
        let ordered = simulation.download()?;
        let inspection = simulation.inspect(InspectionRequest::new(103, &scenario)?)?;
        assert_eq!(
            inspection.business_scale,
            Some(enterprise::BusinessScale::Industrial)
        );
        assert_eq!(ordered.get(Field::Demand, landlord), 24);
        assert_eq!(ordered.get(Field::Sales, landlord), 20);
        assert_eq!(ordered.get(Field::Unfilled, landlord), 4);
        assert!(ordered.get(Field::Cash, owner) > initial.get(Field::Cash, owner));
        match preference {
            enterprise::ExpansionDisposition::Maintain => {
                assert_eq!(ordered.get(Field::ConstructionRemaining, landlord), 0);
                assert_eq!(ordered.get(Field::OperatingReserve, landlord), 120);
            }
            enterprise::ExpansionDisposition::Expand => {
                assert_eq!(ordered.get(Field::ConstructionRemaining, landlord), 20);
                assert_eq!(ordered.get(Field::OperatingReserve, landlord), 2520);
                assert_eq!(
                    ordered.get(Field::ProjectStatus, landlord),
                    ProjectStatus::Building as u64
                );
            }
        }
        let mut replay =
            Simulation::from_checkpoint(simulation.checkpoint()?, support::device(32)?)?;
        for _ in 0..3 {
            simulation.advance()?;
            replay.advance()?;
            assert_eq!(simulation.download()?, replay.download()?);
        }
        let state = simulation.download()?;
        match preference {
            enterprise::ExpansionDisposition::Maintain => {
                assert_eq!(state.get(Field::OccupiedPlots, landlord), 1);
                assert_eq!(
                    state.get(Field::ProjectStatus, landlord),
                    ProjectStatus::None as u64
                );
                assert_eq!(state.get(Field::InstalledCapital, landlord), 0);
            }
            enterprise::ExpansionDisposition::Expand => {
                assert!(state.get(Field::InstalledCapital, landlord) > 0);
                assert!(state.get(Field::ConstructionRemaining, landlord) < 20);
            }
        }
        assert_eq!(state.actors.at(landlord).capacity, 0);
        assert_eq!(simulation.snapshot.cash, initial.initial_cash);
    }
    Ok(())
}

#[test]
fn housing_needs_maintenance_and_never_hires_workers_on_gpu() -> Result<()> {
    let mut config = rentals()?;
    config.expansion = enterprise::ExpansionPolicy {
        small_business_per_mille: 0,
        industrial_per_mille: 0,
    };
    config.population.as_mut().unwrap().firms[0].inventories[Good::Construction as usize - 1] = 0;
    let scenario = Scenario::new(config)?;
    let mut initial = scenario.initial()?;
    initial.set(Field::MaintenanceDays, 10, 1);
    let mut simulation =
        Simulation::from_checkpoint(Checkpoint::new(&scenario, &initial)?, support::device(64)?)?;
    simulation.advance()?;
    assert_eq!(simulation.snapshot.produced[Good::Housing as usize - 1], 8);
    simulation.advance()?;
    let state = simulation.download()?;
    assert_eq!(simulation.snapshot.produced[Good::Housing as usize - 1], 8);
    assert_eq!(state.get(Field::MaintenanceDays, 10), 0);
    assert_eq!(state.get(Field::Slots, 10), 0);
    assert_eq!(state.get(Field::Plan, 10), 0);
    assert_eq!(
        state.get(Field::HousingStatus, 0),
        HousingState::Homeless as u64
    );
    assert_eq!(simulation.snapshot.cash, initial.initial_cash);
    Ok(())
}

#[test]
fn rental_capacity_and_regional_payments_are_separate_on_gpu() -> Result<()> {
    let mut config = rentals()?;
    config.expansion = enterprise::ExpansionPolicy {
        small_business_per_mille: 0,
        industrial_per_mille: 0,
    };
    config.firms = 4;
    let population = config.population.as_mut().unwrap();
    population.households[4..]
        .iter_mut()
        .for_each(|household| household.position.x = 750);
    population.firms[2].technique = Technique::housing(1)?;
    let mut second = population.firms[2].clone();
    second.identity = 104;
    second.owner = 5;
    second.position.x = 750;
    second.technique = Technique::housing(4)?;
    population.firms.push(second);
    config.property = Some(PropertyConfig {
        regions: vec![
            PropertyRegion {
                lower: Position { x: 0, y: 0 },
                upper: Position { x: 500, y: 1000 },
                plots: 7,
                landowner: 1,
            },
            PropertyRegion {
                lower: Position { x: 500, y: 0 },
                upper: Position { x: 1000, y: 1000 },
                plots: 5,
                landowner: 5,
            },
        ],
        buildings: vec![],
        orders: vec![],
    });
    let scenario = Scenario::new(config)?;
    let mut simulation = Simulation::new(scenario.clone(), support::device(64)?)?;
    let cash = simulation.snapshot.cash;
    simulation.advance()?;
    let state = simulation.download()?;
    assert_eq!(state.get(Field::RentalIncome, 10), 30);
    assert_eq!(state.get(Field::RentalIncome, 11), 40);
    assert_eq!(state.get(Field::Landlord, 3), 11);
    assert_eq!(state.get(Field::Landlord, 7), 12);
    assert_eq!(
        state.get(Field::HousingStatus, 0),
        HousingState::Housed as u64
    );
    assert_eq!(
        state.get(Field::HousingStatus, 1),
        HousingState::Homeless as u64
    );
    assert_eq!(simulation.snapshot.cash, cash);
    assert_eq!(
        simulation.snapshot,
        ObservationSnapshot::from_state(&scenario, &state)?
    );
    simulation.checkpoint()?.restore()?;
    let mut moving = state;
    moving.set(Field::MoveDue, 3, 2);
    moving.set(Field::MoveX, 3, 750);
    moving.set(Field::MoveY, 3, 250);
    let mut simulation =
        Simulation::from_checkpoint(Checkpoint::new(&scenario, &moving)?, support::device(64)?)?;
    simulation.advance()?;
    assert_eq!(simulation.actors().at(3).position.x, 750);
    let mut rebuilt = Simulation::from_checkpoint(simulation.checkpoint()?, support::device(32)?)?;
    simulation.advance()?;
    rebuilt.advance()?;
    let moved = simulation.download()?;
    assert_eq!(moved, rebuilt.download()?);
    assert_eq!(moved.get(Field::Landlord, 3), 12);
    assert_eq!(moved.get(Field::HousingAvailable, 3), 1);
    assert_eq!(moved.get(Field::HousingAvailable, 7), 0);
    Ok(())
}

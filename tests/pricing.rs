use super::support;
use anyhow::Result;
use bad_econ::{
    domain::{enterprise::*, food::FoodPolicy, *},
    simulation::Simulation,
};

#[test]
fn small_prices_accumulate_changes_and_replay_without_rounding_floors_on_gpu() -> Result<()> {
    let mut config = pricing_config()?;
    for firm in &mut config.population.as_mut().unwrap().firms {
        firm.price = 10;
        firm.cash = 1_000_000;
    }
    let mut simulation = Simulation::new(Scenario::new(config)?, support::device(64)?)?;
    for _ in 0..8 {
        simulation.advance()?;
    }
    let state = simulation.download()?;
    for firm in 1..4 {
        assert!((6..10).contains(&state.get(Field::Price, firm)));
    }
    let mut resumed = Simulation::from_checkpoint(simulation.checkpoint()?, support::device(64)?)?;
    simulation.advance()?;
    resumed.advance()?;
    assert_eq!(simulation.download()?, resumed.download()?);
    (9..80).try_for_each(|_| resumed.advance())?;
    let state = resumed.download()?;
    for firm in 1..4 {
        assert_eq!(state.get(Field::Price, firm), 1);
    }
    Ok(())
}

fn pricing_config() -> Result<ScenarioConfig> {
    let mut config = Scenario::baseline(1, 42, Experiment::Concentrated)?
        .config()
        .clone();
    config.food = Some(FoodPolicy::default());
    config.prices = FoodPolicy::PRICES;
    config.household_production = HouseholdProduction::industrial();
    config.entrepreneurship.interested_per_mille = 0;
    config.demography.annual_births_per_million_fertile_women = 0;
    config.distribution_rate = 0;
    config.forecast_weight = 0;
    config.behavior.wage_increment = 0;
    config.population = Some(Population {
        households: vec![HouseholdSeed {
            identity: 1,
            position: Position { x: 500, y: 500 },
            cash: 100_000,
            inventories: Good::food([0, 0, 1000]),
            fatigue: 0,
        }],
        firms: [Good::Grain, Good::Flour, Good::Bread]
            .into_iter()
            .enumerate()
            .map(|(index, good)| {
                let mut inventories = [0; Good::COUNT];
                inventories[good as usize - 1] = 10_000;
                Ok(FirmSeed {
                    identity: index as u64 + 2,
                    position: Position { x: 500, y: 500 },
                    owner: 1,
                    good,
                    cash: 0,
                    inventories,
                    capacity: 1,
                    technique: match good {
                        Good::Grain => Technique::new(20, 0, 20)?,
                        Good::Flour => Technique::new(20, 25, 20)?,
                        _ => Technique::new(20, 25, 64)?,
                    },
                    wage: 1000,
                    price: (index as u32 + 1) * 100,
                })
            })
            .collect::<Result<_>>()?,
    });
    Ok(config)
}

#[test]
fn input_quotes_ignore_empty_and_distant_suppliers_on_gpu() -> Result<()> {
    let mut config = pricing_config()?;
    config.search_radius = 100;
    config.shipment_speed = 100;
    config.resources = ResourceMap::Explicit {
        regions: vec![ResourceRegion::new(
            Good::Grain,
            Position { x: 0, y: 0 },
            Position { x: 1000, y: 1000 },
            1000,
        )?],
    };
    let firms = &mut config.population.as_mut().unwrap().firms;
    firms[0].capacity = 22;
    let mut empty = firms[0].clone();
    empty.identity = 5;
    empty.price = 1;
    empty.inventories = [0; Good::COUNT];
    let mut distant = firms[0].clone();
    distant.identity = 6;
    distant.price = 2;
    distant.position = Position { x: 900, y: 500 };
    firms.extend([empty, distant]);
    config.firms = 5;
    let scenario = Scenario::new(config)?;
    let mut simulation = Simulation::new(scenario, support::device(64)?)?;
    simulation.advance()?;
    let state = simulation.download()?;
    assert_eq!(state.get(Field::FirstInputCost, 2), 100);
    assert_eq!(state.get(Field::Plan, 2), 0);
    assert_eq!(state.get(Field::Wage, 2), 1000);
    Ok(())
}

#[test]
fn an_empty_bakery_reprices_and_restarts_from_zero_demand_on_gpu() -> Result<()> {
    let mut config = pricing_config()?;
    config.economy = Economy::Industrial;
    config.behavior.price_response = 1000;
    config.shipment_speed = config.map_size;
    let population = config.population.as_mut().unwrap();
    population.households[0].cash = 400;
    population.households[0].inventories = Good::food([0, 0, 2]);
    for firm in &mut population.firms {
        firm.cash = 1000;
        firm.wage = 2;
        firm.price = 2;
    }
    population.firms[2].inventories = [0; Good::COUNT];
    population.firms[2].price = 9000;
    let scenario = Scenario::new(config)?;
    let bakery = scenario.index_of(4).unwrap();
    let mut initial = scenario.initial()?;
    initial.set(Field::Demand, bakery, 0);
    let mut simulation =
        Simulation::from_checkpoint(Checkpoint::new(&scenario, &initial)?, support::device(64)?)?;
    simulation.advance()?;
    let opening = simulation.download()?;
    assert_eq!(opening.get(Field::Demand, bakery), 0);
    assert!(opening.get(Field::Price, bakery) < 400);
    assert_eq!(opening.get(Field::Transit, bakery), 25);
    let mut receipts = 0;
    for _ in 0..6 {
        simulation.advance()?;
        receipts += simulation.snapshot.economy.accounts[Good::Bread as usize - 1].revenue;
    }
    let state = simulation.download()?;
    assert!(state.get(Field::Produced, bakery) >= 64);
    assert!(receipts > 0);
    assert_eq!(simulation.snapshot.cash, initial.initial_cash);
    simulation.checkpoint()?.restore()?;
    Ok(())
}

#[test]
fn a_mill_with_surplus_flour_preserves_working_capital_on_gpu() -> Result<()> {
    let mut config = pricing_config()?;
    config.economy = Economy::Industrial;
    config.behavior.price_response = 1000;
    let population = config.population.as_mut().unwrap();
    population.firms[0].price = 31;
    population.firms[1].cash = 100_000;
    population.firms[1].wage = 2;
    population.firms[1].price = 12;
    population.firms[1].inventories[Good::Grain as usize - 1] = 25;
    let scenario = Scenario::new(config)?;
    let mill = scenario.index_of(3).unwrap();
    let bakery = scenario.index_of(4).unwrap();
    let mut initial = scenario.initial()?;
    initial.set(Field::Demand, mill, 20);
    initial.set(Field::FirstInputCost, mill, 80);
    initial.set(Field::Demand, bakery, 0);
    let mut simulation =
        Simulation::from_checkpoint(Checkpoint::new(&scenario, &initial)?, support::device(64)?)?;
    simulation.advance()?;
    let state = simulation.download()?;
    assert_eq!(state.get(Field::Price, mill), 12);
    assert_eq!(state.get(Field::Grain, mill), 25);
    assert_eq!(state.get(Field::FirstInputCost, mill), 80);
    assert_eq!(state.get(Field::Transit, mill), 0);
    assert_eq!(state.get(Field::Cash, mill), 100_000);
    Ok(())
}

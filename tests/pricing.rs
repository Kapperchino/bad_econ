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
fn a_bakery_marks_down_its_first_batch_only_after_a_sales_opportunity_on_gpu() -> Result<()> {
    let mut config = pricing_config()?;
    config.economy = Economy::Industrial;
    config.behavior.price_response = 1000;
    let bakery_seed = &mut config.population.as_mut().unwrap().firms[2];
    bakery_seed.cash = 100_000;
    bakery_seed.price = 40;
    bakery_seed.inventories = Good::food([0, 25, 0]);
    let opening_price = u64::from(bakery_seed.price);
    let scenario = Scenario::new(config)?;
    let bakery = scenario.index_of(4).unwrap();
    let mut initial = scenario.initial()?;
    initial.set(Field::Demand, bakery, 0);
    let mut simulation =
        Simulation::from_checkpoint(Checkpoint::new(&scenario, &initial)?, support::device(64)?)?;
    assert_eq!(initial.get(Field::Bread, bakery), 0);
    simulation.advance()?;
    let fresh = simulation.download()?;
    assert_eq!(fresh.get(Field::Produced, bakery), 64);
    assert_eq!(fresh.get(Field::Bread, bakery), 64);
    assert_eq!(fresh.get(Field::Sales, bakery), 0);
    assert_eq!(fresh.get(Field::Unfilled, bakery), 0);
    assert_eq!(fresh.get(Field::Price, bakery), opening_price);
    simulation.advance()?;
    let unsold = simulation.download()?;
    assert_eq!(unsold.get(Field::Produced, bakery), 64);
    assert_eq!(unsold.get(Field::Bread, bakery), 64);
    assert_eq!(unsold.get(Field::Sales, bakery), 0);
    assert_eq!(unsold.get(Field::Unfilled, bakery), 0);
    assert!(unsold.get(Field::Price, bakery) < fresh.get(Field::Price, bakery));
    Ok(())
}

#[test]
fn a_bakery_with_bread_orders_prices_from_the_local_delivered_flour_cost_on_gpu() -> Result<()> {
    let mut config = pricing_config()?;
    config.economy = Economy::Industrial;
    config.behavior.price_response = 1000;
    let population = config.population.as_mut().unwrap();
    population.households[0].inventories = [0; Good::COUNT];
    let mill = &mut population.firms[1];
    mill.price = 200;
    mill.position.x += 10;
    let delivered = u64::from(mill.price)
        + u64::from(config.wage) * u64::from(railway::CART_BILLABLE_MINUTES_PER_UNIT_DAY);
    let bakery_seed = &mut population.firms[2];
    bakery_seed.cash = 100_000;
    bakery_seed.wage = 2;
    bakery_seed.price = 10;
    bakery_seed.inventories = Good::food([0, 50, 1000]);
    let scenario = Scenario::new(config)?;
    let bakery = scenario.index_of(4).unwrap();
    let mut initial = scenario.initial()?;
    initial.set(Field::Demand, bakery, 0);
    initial.set(Field::FirstInputCost, bakery, 5);
    let mut simulation =
        Simulation::from_checkpoint(Checkpoint::new(&scenario, &initial)?, support::device(64)?)?;
    simulation.advance()?;
    let closing = simulation.download()?;
    assert!(closing.get(Field::Sales, bakery) > 0);
    assert_eq!(closing.get(Field::Flour, bakery), 50);
    assert_eq!(closing.get(Field::Transit, bakery), 0);
    assert_eq!(closing.get(Field::FirstInputCost, bakery), delivered);
    let batch_cost = closing.get(Field::LaborMinutes, bakery) * closing.get(Field::Wage, bakery)
        + closing.get(Field::InputTicks, bakery) * delivered;
    assert_eq!(
        closing.get(Field::Price, bakery),
        batch_cost.div_ceil(closing.get(Field::OutputTicks, bakery))
    );
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

#[test]
fn a_partial_steel_batch_marks_down_until_a_sale_restores_working_capital_protection_on_gpu()
-> Result<()> {
    let mut config = pricing_config()?;
    config.economy = Economy::Industrial;
    config.shipment_speed = config.map_size;
    let firms = &mut config.population.as_mut().unwrap().firms;
    firms[0].good = Good::Steel;
    firms[0].price = 100;
    firms[0].cash = 100_000;
    firms[0].technique = Technique::new(20, 1, 20)?;
    firms[0].inventories = [0; Good::COUNT];
    firms[0].inventories[Good::Steel as usize - 1] = 2;
    firms[1].good = Good::Locomotives;
    firms[1].cash = 114;
    firms[1].wage = 1;
    firms[1].technique = Technique::new(20, 1, 1)?;
    firms[1].inventories = [0; Good::COUNT];
    let scenario = Scenario::new(config)?;
    let steel = scenario.index_of(2).unwrap();
    let mut initial = scenario.initial()?;
    initial.set(Field::Demand, steel, 19);
    let buyer = scenario.index_of(3).unwrap();
    let mut simulation =
        Simulation::from_checkpoint(Checkpoint::new(&scenario, &initial)?, support::device(64)?)?;
    simulation.advance()?;
    let unsold = simulation.download()?;
    assert_eq!(unsold.get(Field::Sales, steel), 0);
    assert!(unsold.get(Field::Price, steel) < 100);
    simulation.advance()?;
    simulation.advance()?;
    let sold = simulation.download()?;
    assert_eq!(sold.get(Field::Sales, steel), 1);
    assert_eq!(sold.get(Field::Steel, steel), 1);
    assert!(sold.get(Field::Price, steel) > 100);
    simulation.advance()?;
    assert_eq!(simulation.download()?.get(Field::Steel, buyer), 1);
    Ok(())
}

#[test]
fn household_grain_floor_includes_paid_cartage_minutes_on_gpu() -> Result<()> {
    let mut config = pricing_config()?;
    config.behavior.price_response = 1000;
    let population = config.population.as_mut().unwrap();
    population.households[0].cash = 80;
    population.households[0].inventories = Good::food([100, 0, 0]);
    let bakery = &mut population.firms[2];
    bakery.price = 1;
    bakery.position.x += 10;
    let scenario = Scenario::new(config)?;
    let mut initial = scenario.initial()?;
    initial.day = 29;
    initial.set(Field::LandPlots, 0, 1);
    initial.set(Field::GrainQuote, 0, 15);
    let mut simulation =
        Simulation::from_checkpoint(Checkpoint::new(&scenario, &initial)?, support::device(64)?)?;
    simulation.advance()?;
    let after = simulation.download()?;
    assert_eq!(after.get(Field::Hours, 0), 0);
    let minutes = after.get(Field::CartageMinutes, 0);
    let earnings = after.get(Field::CartageIncome, 0);
    assert!(minutes > 0);
    assert!(earnings * 80 > minutes * 7 * 15);
    assert_eq!(after.get(Field::GrainSales, 0), 0);
    assert_eq!(after.get(Field::GrainQuote, 0), 15);
    Ok(())
}

use super::support;
use anyhow::Result;
use bad_econ::{
    domain::{
        enterprise::{Distributor, HouseholdProduction},
        *,
    },
    simulation::{Simulation, Trace},
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
fn producers_and_distributors_reserve_for_at_most_installed_capacity_on_gpu() -> Result<()> {
    let mut config = economy()?.config().clone();
    config.household_production = HouseholdProduction::industrial();
    config.behavior.wage_increment = 0;
    let scenario = Scenario::new(config)?;
    let bakery = scenario.index_of(4).unwrap();
    let dealer = scenario.index_of(5).unwrap();
    let mut initial = scenario.initial()?;
    initial.set(Field::Demand, bakery, 240);
    initial.set(Field::Demand, dealer, 240);
    let mut simulation =
        Simulation::from_checkpoint(Checkpoint::new(&scenario, &initial)?, support::device(64)?)?;
    simulation.advance()?;
    let state = simulation.download()?;
    let bakery_labor = state.get(Field::LaborMinutes, bakery) * state.get(Field::Wage, bakery);
    let bakery_flour =
        state.get(Field::InputTicks, bakery) * state.get(Field::FirstInputCost, bakery);
    let bakery_days = u64::from(scenario.config().coverage) + scenario.ring() as u64;
    let bakery_capacity = u64::from(state.actors.at(bakery).capacity);
    assert_eq!(
        state.get(Field::OperatingReserve, bakery),
        bakery_capacity * bakery_days * (bakery_labor + bakery_flour)
    );
    let dealer_labor = state.get(Field::LaborMinutes, dealer) * state.get(Field::Wage, dealer);
    let dealer_stock =
        state.get(Field::OutputTicks, dealer) * state.get(Field::ReplacementCost, dealer);
    let dealer_days = state.get(Field::StockCoverage, dealer) + scenario.ring() as u64;
    let dealer_capacity = u64::from(state.actors.at(dealer).capacity);
    assert_eq!(
        state.get(Field::OperatingReserve, dealer),
        dealer_capacity * dealer_days * (dealer_labor + dealer_stock)
    );
    Ok(())
}

#[test]
fn distributor_expansion_uses_combined_handling_demand_on_gpu() -> Result<()> {
    let mut config = economy()?.config().clone();
    config.household_production = HouseholdProduction::industrial();
    config.expansion = enterprise::ExpansionPolicy {
        small_business_per_mille: 1000,
        industrial_per_mille: 1000,
    };
    let population = config.population.as_mut().unwrap();
    let shop = &mut population.firms[1];
    shop.position = population.households[0].position;
    shop.cash = 100_000;
    shop.inventories = Good::food([24, 24, 0]);
    let scenario = Scenario::new(config)?;
    let dealer = scenario.index_of(5).unwrap();
    let demand_field =
        scenario.fields() - RetailAccount::FIELDS + RetailAccount::Demand as usize * Good::COUNT;
    let forecast = |good: Good| (demand_field + good as usize - 1) * scenario.actors() + dealer;
    let mut initial = scenario.initial()?;
    initial.set(Field::Demand, dealer, 0);
    initial.values[forecast(Good::Grain)] = 12;
    initial.values[forecast(Good::Flour)] = 13;
    let mut simulation =
        Simulation::from_checkpoint(Checkpoint::new(&scenario, &initial)?, support::device(64)?)?;
    simulation.advance()?;
    assert_eq!(
        simulation.download()?.get(Field::PendingCapacity, dealer),
        1
    );
    Ok(())
}

#[test]
fn buying_tools_preserves_the_delivered_cost_of_resale_stock_on_gpu() -> Result<()> {
    let mut config = economy()?.config().clone();
    config.household_production = HouseholdProduction::industrial();
    let firms = &mut config.population.as_mut().unwrap().firms;
    firms[2].good = Good::Tools;
    firms[2].technique = Technique::new(60, 1, 1)?;
    firms[2].price = 400;
    firms[2].position = firms[1].position;
    firms[2].inventories[Good::Tools as usize - 1] = 100;
    let scenario = Scenario::new(config)?;
    let dealer = scenario.index_of(5).unwrap();
    let device = support::device(64)?;
    for trace in [Trace::Discard, Trace::Record] {
        let mut simulation = Simulation::new(scenario.clone(), device.clone())?;
        for phase in Phase::ALL {
            simulation.phase(phase, trace)?;
        }
        let settled = simulation.download()?;
        let incoming = |good: Good| {
            settled.values[(TRANSIT_FIELDS + good as usize - 1) * settled.actors.len() + dealer]
        };
        assert!(incoming(Good::Bread) > 0);
        assert_eq!(incoming(Good::Tools), 2);
        assert_eq!(settled.get(Field::ReplacementCost, dealer), 13);
        simulation.checkpoint()?.restore()?;
    }
    Ok(())
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
    assert_eq!(second.get(Field::ReplacementCost, 4), 13);
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

#[test]
fn a_dealer_carries_a_secondary_quote_forward_while_its_primary_good_sells_on_gpu() -> Result<()> {
    let mut config = economy()?.config().clone();
    config.behavior.price_response = 1000;
    config.behavior.wage_increment = 0;
    let dealer_seed = &mut config.population.as_mut().unwrap().firms[1];
    dealer_seed.inventories[Good::Steel as usize - 1] = 5;
    dealer_seed.inventories[Good::Bread as usize - 1] = 50;
    let scenario = Scenario::new(config)?;
    let dealer = scenario.index_of(5).unwrap();
    let row = scenario.fields() - RetailAccount::FIELDS + Good::Steel as usize - 1;
    let mut initial = scenario.initial()?;
    let quote = row * initial.actors.len() + dealer;
    initial.values[quote] = 100;
    let mut simulation =
        Simulation::from_checkpoint(Checkpoint::new(&scenario, &initial)?, support::device(64)?)?;
    simulation.advance()?;
    let first = simulation.download()?;
    assert!(first.get(Field::Sales, dealer) > 0);
    assert_eq!(first.get(Field::Steel, dealer), 5);
    assert_eq!(first.values[quote], 95);
    simulation.advance()?;
    assert_eq!(simulation.download()?.values[quote], 90);
    Ok(())
}

#[test]
fn orders_restore_a_discounted_secondary_shelf_to_its_delivered_cost_on_gpu() -> Result<()> {
    let mut config = economy()?.config().clone();
    config.economy = Economy::Industrial;
    config.household_production = HouseholdProduction::industrial();
    config.behavior.price_response = 250;
    config.behavior.wage_increment = 0;
    let firms = &mut config.population.as_mut().unwrap().firms;
    firms[0].good = Good::Locomotives;
    firms[0].position = firms[1].position;
    firms[0].technique = Technique::new(60, 1, 1)?;
    firms[0].inventories = [0; Good::COUNT];
    firms[1].inventories[Good::Steel as usize - 1] = 1000;
    firms[1].inventories[Good::Bread as usize - 1] = 4;
    let scenario = Scenario::new(config)?;
    let dealer = scenario.index_of(5).unwrap();
    let mut initial = scenario.initial()?;
    let account = |good: Good, account: RetailAccount| {
        (scenario.fields() - RetailAccount::FIELDS
            + (account as usize) * Good::COUNT
            + good as usize
            - 1)
            * scenario.actors()
            + dealer
    };
    let quote = account(Good::Steel, RetailAccount::Quote);
    initial.values[quote] = 4;
    initial.values[account(Good::Steel, RetailAccount::Cost)] = 60;
    initial.set(Field::PrimaryRetailUnfilled, dealer, 1234);
    let mut simulation =
        Simulation::from_checkpoint(Checkpoint::new(&scenario, &initial)?, support::device(64)?)?;
    simulation.advance()?;
    let after = simulation.download()?;
    assert!(after.values[account(Good::Steel, RetailAccount::Orders)] > 0);
    assert!(after.get(Field::Steel, dealer) < 1000);
    assert!(after.values[quote] >= 60);
    let bread_sales = initial.get(Field::Bread, dealer) - after.get(Field::Bread, dealer);
    assert_eq!(
        after.get(Field::PrimaryRetailUnfilled, dealer),
        after.values[account(Good::Bread, RetailAccount::Orders)] - bread_sales
    );
    Ok(())
}

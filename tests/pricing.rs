#![cfg(feature = "compute")]

use super::support;
use anyhow::Result;
use bad_econ::{
    domain::{enterprise::*, food::FoodPolicy, *},
    simulation::Simulation,
};

fn pricing_scenario() -> Result<Scenario> {
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
    Scenario::new(config)
}

#[test]
fn surplus_prices_fall_below_costs_and_refresh_without_purchases_on_gpu() -> Result<()> {
    let scenario = pricing_scenario()?;
    let opening = scenario.initial()?;
    let mut simulation = Simulation::new(scenario, support::device(64)?)?;
    simulation.advance()?;
    let first = simulation.download()?;
    for firm in 1..4 {
        assert!(first.get(Field::Price, firm) < opening.get(Field::Price, firm));
        assert_eq!(first.get(Field::Purchased, firm), 0);
    }
    assert_eq!(first.get(Field::FirstInputCost, 2), 100);
    assert_eq!(first.get(Field::FirstInputCost, 3), 200);
    simulation.advance()?;
    let second = simulation.download()?;
    assert_eq!(
        second.get(Field::FirstInputCost, 2),
        first.get(Field::Price, 1)
    );
    assert_eq!(
        second.get(Field::FirstInputCost, 3),
        first.get(Field::Price, 2)
    );
    assert_eq!(simulation.snapshot.cash, opening.initial_cash);
    simulation.checkpoint()?.restore()?;
    Ok(())
}

#[test]
fn legacy_minimum_price_does_not_constrain_market_prices_on_gpu() -> Result<()> {
    let mut config = pricing_scenario()?.config().clone();
    config.minimum_price = 500;
    let scenario = Scenario::new(config)?;
    assert_eq!(scenario.config().minimum_price, 1);
    let mut simulation = Simulation::new(scenario, support::device(64)?)?;
    simulation.advance()?;
    assert!(simulation.snapshot.markets[0].prices.maximum < 100);
    Ok(())
}

#[test]
fn unfilled_orders_raise_prices_without_a_cost_floor_on_gpu() -> Result<()> {
    let mut config = pricing_scenario()?.config().clone();
    let population = config.population.as_mut().unwrap();
    population.households[0].inventories[Good::Bread as usize - 1] = 0;
    population.firms[2].inventories[Good::Bread as usize - 1] = 0;
    population.firms[0].inventories[Good::Bread as usize - 1] = 2000;
    let scenario = Scenario::new(config)?;
    let mut simulation = Simulation::new(scenario, support::device(64)?)?;
    simulation.advance()?;
    let state = simulation.download()?;
    assert!(state.get(Field::Unfilled, 3) > 0);
    assert!(state.get(Field::Price, 3) > 300);
    Ok(())
}

#[test]
fn retailer_prices_follow_stock_instead_of_replacement_markup_on_gpu() -> Result<()> {
    let mut config = pricing_scenario()?.config().clone();
    let firms = &mut config.population.as_mut().unwrap().firms;
    firms[1].good = Good::Bread;
    firms[1].inventories = Good::food([0, 0, 1000]);
    firms[1].technique = Technique::new(20, 25, 64)?;
    config.distributors = vec![Distributor {
        firm: 4,
        good: Good::Bread,
        handling_minutes: 20,
        units_per_shift: 24,
        reserve_days: 3,
        margin_per_mille: 1000,
    }];
    let scenario = Scenario::new(config)?;
    let mut simulation = Simulation::new(scenario, support::device(64)?)?;
    simulation.advance()?;
    let state = simulation.download()?;
    assert_eq!(state.get(Field::ReplacementCost, 3), 200);
    assert!(state.get(Field::Price, 3) < 300);
    assert_eq!(state.get(Field::Purchased, 3), 0);
    Ok(())
}

#[test]
fn closed_firms_do_not_anchor_input_quotes_or_market_statistics_on_gpu() -> Result<()> {
    let mut config = pricing_scenario()?.config().clone();
    let mut closed = config.population.as_ref().unwrap().firms[0].clone();
    closed.identity = 5;
    closed.price = 1;
    config.population.as_mut().unwrap().firms.push(closed);
    config.firms = 4;
    let scenario = Scenario::new(config)?;
    let mut state = scenario.initial()?;
    state.set(Field::EnterpriseStatus, 4, EnterpriseState::Closed as u64);
    let mut simulation =
        Simulation::from_checkpoint(Checkpoint::new(&scenario, &state)?, support::device(64)?)?;
    simulation.advance()?;
    let state = simulation.download()?;
    assert_eq!(state.get(Field::FirstInputCost, 2), 100);
    let quotes = simulation.snapshot.markets[Good::Grain as usize - 1].prices;
    assert_eq!(quotes.count, 1);
    assert_eq!(quotes.minimum, state.get(Field::Price, 1));
    assert_eq!(quotes.maximum, quotes.minimum);
    assert_eq!(simulation.snapshot.firms, 4);
    assert_eq!(
        simulation.snapshot,
        ObservationSnapshot::from_state(&simulation.scenario, &state)?,
    );
    Ok(())
}

#[test]
fn input_quotes_ignore_empty_and_distant_suppliers_on_gpu() -> Result<()> {
    let mut config = pricing_scenario()?.config().clone();
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
    Ok(())
}

#[test]
fn unstocked_and_unopened_firms_do_not_inflate_current_market_quotes_on_gpu() -> Result<()> {
    let mut config = pricing_scenario()?.config().clone();
    let mut startup = config.population.as_ref().unwrap().firms[0].clone();
    startup.identity = 5;
    startup.price = 9000;
    startup.inventories = [0; Good::COUNT];
    config.population.as_mut().unwrap().firms.push(startup);
    config.firms = 4;
    let scenario = Scenario::new(config)?;
    for status in [EnterpriseState::Forming, EnterpriseState::Operating] {
        let mut state = scenario.initial()?;
        state.set(Field::EnterpriseStatus, 4, status as u64);
        let construction = match status {
            EnterpriseState::Forming => 720,
            _ => 0,
        };
        state.set(Field::ConstructionRemaining, 4, construction);
        let mut simulation =
            Simulation::from_checkpoint(Checkpoint::new(&scenario, &state)?, support::device(64)?)?;
        simulation.advance()?;
        let state = simulation.download()?;
        let market = simulation.snapshot.markets[Good::Grain as usize - 1];
        assert_eq!(market.prices.count, 1);
        assert_eq!(market.prices.maximum, state.get(Field::Price, 1));
        let employers = match status {
            EnterpriseState::Forming => 1,
            _ => 2,
        };
        assert_eq!(market.wages.count, employers);
        assert_eq!(simulation.snapshot.firms, 4);
        assert_eq!(
            simulation.snapshot,
            ObservationSnapshot::from_state(&simulation.scenario, &state)?,
        );
    }
    Ok(())
}

#[test]
fn fully_closed_markets_have_empty_quote_statistics_on_gpu() -> Result<()> {
    let scenario = pricing_scenario()?;
    let mut state = scenario.initial()?;
    for firm in 1..4 {
        state.set(
            Field::EnterpriseStatus,
            firm,
            EnterpriseState::Closed as u64,
        );
    }
    let mut simulation =
        Simulation::from_checkpoint(Checkpoint::new(&scenario, &state)?, support::device(64)?)?;
    simulation.advance()?;
    assert_eq!(simulation.snapshot.minimum_price, 0);
    assert_eq!(simulation.snapshot.maximum_price, 0);
    assert!(simulation.snapshot.markets.iter().all(|market| {
        market.prices == QuoteDistribution::default()
            && market.wages == QuoteDistribution::default()
    }));
    let state = simulation.download()?;
    assert_eq!(
        simulation.snapshot,
        ObservationSnapshot::from_state(&simulation.scenario, &state)?,
    );
    Ok(())
}

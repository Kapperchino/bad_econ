#![cfg(feature = "compute")]
use crate::support;
use anyhow::Result;
use bad_econ::{
    domain::{enterprise::Distributor, *},
    runtime::SimulationCommand,
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

#[test]
fn grocery_checkpoint_and_workgroup_continuation_are_exact() -> Result<()> {
    let scenario = economy()?;
    let mut original = Simulation::new(scenario.clone(), support::device(32)?)?;
    original.advance()?;
    original.advance()?;
    let checkpoint = original.checkpoint()?;
    let mut resumed = Simulation::new(scenario, support::device(128)?)?;
    resumed.command(SimulationCommand::Restore(Box::new(checkpoint)))?;
    for _ in 0..5 {
        original.advance()?;
        resumed.advance()?;
        assert_eq!(original.download()?, resumed.download()?);
    }
    Ok(())
}

#[test]
fn grocery_restocking_keeps_the_next_shift_funded() -> Result<()> {
    let mut config = economy()?.config().clone();
    config.population.as_mut().unwrap().firms[1].cash = 300;
    let scenario = Scenario::new(config)?;
    let mut simulation = Simulation::new(scenario.clone(), support::device(64)?)?;
    simulation.advance()?;
    let restocked = simulation.download()?;
    assert_eq!(restocked.get(Field::Cash, 4), 60);
    assert_eq!(restocked.get(Field::Transit, 4), 24);
    simulation.advance()?;
    let open = simulation.download()?;
    assert_eq!(open.get(Field::RetailOpen, 4), 24);
    assert!(open.get(Field::Sales, 4) > 0);
    assert_eq!(
        simulation.snapshot,
        ObservationSnapshot::from_state(&scenario, &open)?
    );
    Ok(())
}

#[test]
fn grocery_distributions_preserve_stock_replacement_and_payroll_funds() -> Result<()> {
    let mut config = economy()?.config().clone();
    config.reserve = 0;
    config.distribution_rate = 1000;
    let scenario = Scenario::new(config)?;
    let mut simulation = Simulation::new(scenario.clone(), support::device(64)?)?;
    simulation.advance()?;
    let state = simulation.download()?;
    let replacement = state.get(Field::ReplacementCost, 4)
        * state.get(Field::OutputTicks, 4)
        * (state.get(Field::StockCoverage, 4) + scenario.ring() as u64);
    assert_eq!(state.get(Field::OperatingReserve, 4), replacement + 60);
    assert_eq!(
        state.get(Field::Cash, 4),
        state.get(Field::OperatingReserve, 4)
    );
    simulation.advance()?;
    let serving = simulation.download()?;
    assert!(serving.get(Field::RetailOpen, 4) > 0);
    assert!(serving.get(Field::Sales, 4) > 0);
    assert_eq!(
        simulation.snapshot,
        ObservationSnapshot::from_state(&scenario, &serving)?
    );
    Ok(())
}

#[test]
fn grocery_staffing_scales_with_demand_stock_and_site_capacity() -> Result<()> {
    let mut config = economy()?.config().clone();
    config.population.as_mut().unwrap().firms[1].capacity = 3;
    config.distributors[0].units_per_shift = 4;
    let scenario = Scenario::new(config)?;
    let mut simulation = Simulation::new(scenario.clone(), support::device(128)?)?;
    simulation.advance()?;
    simulation.advance()?;
    let state = simulation.download()?;
    assert_eq!(state.get(Field::Plan, 4), 3);
    assert_eq!(state.get(Field::Slots, 4), 0);
    assert_eq!(state.get(Field::RetailOpen, 4), 12);
    assert_eq!(simulation.snapshot.employed, 3);
    assert!(state.get(Field::Sales, 4) > 4);
    assert_eq!(simulation.snapshot.produced[Good::Bread as usize - 1], 0);
    assert_eq!(
        simulation.snapshot,
        ObservationSnapshot::from_state(&scenario, &state)?
    );
    Ok(())
}

#[test]
fn generated_groceries_stock_retail_goods_and_leave_production_capacity_to_bakeries() -> Result<()>
{
    let scenario = Scenario::industrial(1000, 42, Experiment::Baseline)?;
    let mut simulation = Simulation::new(scenario.clone(), support::device(64)?)?;
    let state = simulation.download()?;
    let mut bakery_capacity = 0;
    for (index, actor) in state
        .actors
        .iter()
        .enumerate()
        .filter(|(_, actor)| actor.kind == Good::Bread as u32)
    {
        match state.get(Field::EnterpriseModel, index) {
            1 => {
                assert_eq!(state.get(Field::Flour, index), 0);
                assert_eq!(
                    state.get(Field::Demand, index),
                    u64::from(actor.capacity) * state.get(Field::OutputTicks, index)
                );
                assert!(state.get(Field::Bread, index) > 0);
            }
            _ => {
                bakery_capacity += u64::from(actor.capacity) * state.get(Field::OutputTicks, index)
            }
        }
    }
    assert!(bakery_capacity >= 1200);
    assert_eq!(state.initial_goods, scenario.opening_totals().goods);
    assert_eq!(
        simulation.snapshot,
        ObservationSnapshot::from_state(&scenario, &state)?
    );
    Ok(())
}

#[test]
fn grain_distributor_can_operate_outside_agricultural_land() -> Result<()> {
    let mut config = economy()?.config().clone();
    config.resources = ResourceMap::Explicit {
        regions: Vec::new(),
    };
    let store = &mut config.population.as_mut().unwrap().firms[1];
    store.good = Good::Grain;
    store.technique = Technique::new(60, 0, 10)?;
    store.inventories[Good::Grain as usize - 1] = 24;
    config.distributors[0].good = Good::Grain;
    let scenario = Scenario::new(config)?;
    assert!(!scenario.has_resource(Good::Grain, Position { x: 500, y: 0 }));
    let mut simulation = Simulation::new(scenario.clone(), support::device(64)?)?;
    simulation.advance()?;
    let state = simulation.download()?;
    assert_eq!(state.get(Field::RetailOpen, 4), 24);
    assert_eq!(state.get(Field::Produced, 4), 0);
    assert_eq!(
        simulation.snapshot,
        ObservationSnapshot::from_state(&scenario, &state)?
    );
    Ok(())
}

#[test]
fn producer_input_orders_keep_a_production_shift_funded() -> Result<()> {
    let mut config = economy()?.config().clone();
    config.distributors.clear();
    config.household_production = bad_econ::domain::enterprise::HouseholdProduction::industrial();
    let firms = &mut config.population.as_mut().unwrap().firms;
    firms[0].position.x = 500;
    firms[0].cash = 300;
    firms[0].inventories = [0; Good::COUNT];
    firms[1].position.x = 0;
    firms[1].good = Good::Flour;
    firms[1].cash = 0;
    firms[1].price = 10;
    firms[1].inventories = Good::food([0, 100, 0]);
    let scenario = Scenario::new(config)?;
    let mut simulation = Simulation::new(scenario.clone(), support::device(64)?)?;
    simulation.advance()?;
    let ordered = simulation.download()?;
    assert_eq!(ordered.get(Field::Cash, 3), 60);
    assert_eq!(ordered.get(Field::Transit, 3), 24);
    simulation.advance()?;
    let produced = simulation.download()?;
    assert_eq!(produced.get(Field::Produced, 3), 10);
    assert_eq!(produced.get(Field::Slots, 3), 0);
    assert_eq!(
        simulation.snapshot,
        ObservationSnapshot::from_state(&scenario, &produced)?
    );
    Ok(())
}

#[test]
fn household_savings_open_a_grocery_that_connects_distant_food_supply_to_customers() -> Result<()> {
    let mut config = economy()?.config().clone();
    config.households = 30;
    config.distributors.clear();
    config.household_production = bad_econ::domain::enterprise::HouseholdProduction::industrial();
    config.population = Some(Population {
        households: (1..=30)
            .map(|identity| HouseholdSeed {
                identity,
                position: Position {
                    x: if identity == 1 { 500 } else { 999 },
                    y: 0,
                },
                cash: 12_000,
                inventories: match identity {
                    1 => Good::food([1000, 0, 20]),
                    _ => [0; Good::COUNT],
                },
                fatigue: 0,
            })
            .collect(),
        firms: [Good::Bread, Good::Tools, Good::Wood]
            .into_iter()
            .enumerate()
            .map(|(index, good)| {
                let mut inventories = [0; Good::COUNT];
                inventories[good as usize - 1] = 2000;
                FirmSeed {
                    identity: 31 + index as u64,
                    position: Position {
                        x: if index == 0 { 0 } else { 500 },
                        y: 0,
                    },
                    owner: 2,
                    good,
                    cash: 0,
                    inventories,
                    capacity: 1,
                    technique: Technique::new(60, u32::from(good.inputs()[0].is_some()), 10)
                        .unwrap(),
                    wage: 1,
                    price: 10,
                }
            })
            .collect(),
    });
    let scenario = Scenario::new(config)?;
    let mut simulation = Simulation::new(scenario.clone(), support::device(64)?)?;
    for _ in 0..20 {
        simulation.advance()?;
    }
    let state = simulation.download()?;
    let store = (scenario.actors()..state.actors.len())
        .find(|&index| state.get(Field::EnterpriseModel, index) == 1)
        .expect("household financed a grocery");
    assert_eq!(state.actors.at(store).owner, 0);
    assert_eq!(state.actors.at(store).position.x, 500);
    assert_eq!(state.get(Field::ConstructionRemaining, store), 0);
    assert_eq!(state.get(Field::InstalledCapital, store), 7);
    assert_eq!(state.get(Field::Produced, store), 0);
    assert!(state.get(Field::RetailOpen, store) > 0);
    assert!(state.get(Field::Sales, store) > 0);
    assert!(state.get(Field::Cash, 30) > 0);
    assert_eq!(
        simulation.snapshot,
        ObservationSnapshot::from_state(&scenario, &state)?
    );
    let mut resumed = Simulation::from_checkpoint(simulation.checkpoint()?, support::device(128)?)?;
    for _ in 0..3 {
        simulation.advance()?;
        resumed.advance()?;
        assert_eq!(simulation.download()?, resumed.download()?);
    }
    Ok(())
}

#[test]
fn affordable_missing_supply_reaches_producers_once_and_restarts_the_chain() -> Result<()> {
    let mut config = Scenario::baseline(30, 29, Experiment::Concentrated)?
        .config()
        .clone();
    config.buffer_days = 0;
    config.forecast_weight = 1000;
    config.distribution_rate = 0;
    config.behavior.price_response = 0;
    config.demography.annual_births_per_million_fertile_women = 0;
    config.demography.mortality.background_annual_ppm = [0; 5];
    let scenario = Scenario::new(config)?;
    let mut state = scenario.initial()?;
    for actor in scenario.config().households as usize..state.actors.len() {
        state.set(Field::Demand, actor, 0);
    }
    let checkpoint = Checkpoint::new(&scenario, &state)?;
    let mut simulation = Simulation::new(scenario.clone(), support::device(64)?)?;
    simulation.command(SimulationCommand::Restore(Box::new(checkpoint)))?;
    simulation.advance()?;
    let first = simulation.download()?;
    let bakery = first
        .actors
        .iter()
        .position(|actor| actor.kind == Good::Bread as u32)
        .unwrap();
    let order = u64::from(scenario.config().coverage) + scenario.ring() as u64 - 1;
    assert_eq!(first.get(Field::Unfilled, bakery), 30 * order);
    assert!(first.get(Field::Demand, bakery) > 0);
    assert_eq!(first.get(Field::FoodAccess, 0), 1);
    for _ in 0..14 {
        simulation.advance()?;
    }
    assert!(simulation.snapshot.produced[Good::Grain as usize - 1] > 0);
    assert!(simulation.snapshot.produced[Good::Flour as usize - 1] > 0);
    assert!(simulation.snapshot.produced[Good::Bread as usize - 1] > 0);
    assert!(simulation.snapshot.consumed > 0);
    Ok(())
}

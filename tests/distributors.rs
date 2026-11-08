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

fn competing_warehouses() -> Result<ScenarioConfig> {
    use bad_econ::domain::property::{Building, PropertyConfig, PropertyRegion};
    let mut config = economy()?.config().clone();
    config.households = 30;
    config.forecast_weight = 1000;
    config.entrepreneurship.interested_per_mille = 0;
    config.demography.mortality.famine_daily_ppm_at_thirty_days = 0;
    config.behavior.work_fatigue = 0;
    config.behavior.rest_recovery = 0;
    config.behavior.wage_increment = 0;
    config.rent = Some(RentPolicy {
        household_daily: 0,
        firm_daily: 10,
    });
    config.population = Some(Population {
        households: (1..=30)
            .map(|identity| HouseholdSeed {
                identity,
                position: Position { x: 999, y: 0 },
                cash: 100_000,
                inventories: [0; Good::COUNT],
                fatigue: 0,
            })
            .collect(),
        firms: [
            FirmSeed {
                identity: 31,
                position: Position { x: 0, y: 0 },
                owner: 1,
                good: Good::Bread,
                cash: 100_000,
                inventories: Good::food([0, 0, 100_000]),
                capacity: 1,
                technique: Technique::new(60, 1, 100)?,
                wage: 1,
                price: 10,
            },
            FirmSeed {
                identity: 32,
                position: Position { x: 500, y: 0 },
                owner: 2,
                good: Good::Bread,
                cash: 100_000,
                inventories: Good::food([0, 0, 500]),
                capacity: 1,
                technique: Technique::new(60, 10, 10)?,
                wage: 1,
                price: 16,
            },
            FirmSeed {
                identity: 33,
                position: Position { x: 500, y: 0 },
                owner: 3,
                good: Good::Bread,
                cash: 100_000,
                inventories: Good::food([0, 0, 500]),
                capacity: 1,
                technique: Technique::new(60, 10, 10)?,
                wage: 1,
                price: 14,
            },
        ]
        .into(),
    });
    config.distributors = [32, 33]
        .into_iter()
        .map(|firm| Distributor {
            firm,
            good: Good::Bread,
            handling_minutes: match firm {
                32 => 60,
                _ => 30,
            },
            units_per_shift: 24,
            reserve_days: 3,
            margin_per_mille: 250,
        })
        .collect();
    config.property = Some(PropertyConfig {
        regions: vec![PropertyRegion {
            lower: Position { x: 0, y: 0 },
            upper: Position { x: 1000, y: 1000 },
            plots: 100,
            landowner: 1,
        }],
        buildings: vec![Building { site: 33, plots: 4 }],
        orders: Vec::new(),
    });
    Ok(config)
}

#[test]
fn larger_competitor_drives_a_funded_warehouse_out_of_business() -> Result<()> {
    use bad_econ::domain::{enterprise::EnterpriseState, property::PropertyUse};
    let scenario = Scenario::new(competing_warehouses()?)?;
    let mut initial = scenario.initial()?;
    for firm in [31, 32] {
        initial.set(Field::ReplacementCost, firm, 10);
    }
    let mut simulation =
        Simulation::from_checkpoint(Checkpoint::new(&scenario, &initial)?, support::device(64)?)?;
    for _ in 0..29 {
        simulation.advance()?;
    }
    let struggling = simulation.download()?;
    assert!(struggling.get(Field::Cash, 31) > 90_000);
    assert!(struggling.get(Field::IdleDays, 31) > 20);
    assert_eq!(
        struggling.get(Field::PropertyUse, 31),
        PropertyUse::Store as u64
    );
    assert_eq!(struggling.get(Field::OccupiedPlots, 31), 1);
    assert_eq!(struggling.get(Field::OccupiedPlots, 32), 4);
    assert_eq!(struggling.get(Field::RentTreasury, 31), 290);
    assert_eq!(struggling.get(Field::RentArrears, 31), 0);
    assert_eq!(struggling.get(Field::Sales, 31), 0);
    assert!(struggling.get(Field::Sales, 32) > 0);
    let mut replay = Simulation::from_checkpoint(simulation.checkpoint()?, support::device(128)?)?;
    for _ in 0..6 {
        simulation.advance()?;
        replay.advance()?;
        assert_eq!(simulation.download()?, replay.download()?);
    }
    let closed = simulation.download()?;
    assert_eq!(
        closed.get(Field::EnterpriseStatus, 31),
        EnterpriseState::Closed as u64
    );
    assert_eq!(closed.get(Field::Cash, 31), 0);
    assert_eq!(closed.get(Field::Bread, 31), 0);
    assert_eq!(closed.get(Field::OccupiedPlots, 31), 1);
    assert_eq!(
        closed.get(Field::EnterpriseStatus, 32),
        EnterpriseState::Operating as u64
    );
    assert_eq!(simulation.snapshot.distributors, 1);
    assert_eq!(simulation.snapshot.cash, initial.initial_cash);
    assert_eq!(
        simulation.snapshot,
        ObservationSnapshot::from_state(&scenario, &closed)?
    );
    Ok(())
}

#[test]
fn profitable_small_warehouse_survives_and_recovers_from_a_sales_slump() -> Result<()> {
    use bad_econ::domain::enterprise::EnterpriseState;
    let mut config = competing_warehouses()?;
    config.distributors.pop();
    let competitor = &mut config.population.as_mut().unwrap().firms[2];
    competitor.good = Good::Grain;
    competitor.inventories = [0; Good::COUNT];
    competitor.technique = Technique::new(60, 0, 10)?;
    let scenario = Scenario::new(config)?;
    let mut initial = scenario.initial()?;
    initial.set(Field::ReplacementCost, 31, 10);
    initial.set(Field::IdleDays, 31, 29);
    let mut simulation =
        Simulation::from_checkpoint(Checkpoint::new(&scenario, &initial)?, support::device(64)?)?;
    simulation.advance()?;
    let recovered = simulation.download()?;
    assert!(recovered.get(Field::Sales, 31) > 0);
    assert_eq!(recovered.get(Field::IdleDays, 31), 0);
    for _ in 0..34 {
        simulation.advance()?;
    }
    let state = simulation.download()?;
    assert_eq!(state.get(Field::OccupiedPlots, 31), 1);
    assert_eq!(
        state.get(Field::EnterpriseStatus, 31),
        EnterpriseState::Operating as u64
    );
    assert_eq!(state.get(Field::RentTreasury, 31), 350);
    assert_eq!(simulation.snapshot.distributors, 1);
    assert_eq!(simulation.snapshot.cash, initial.initial_cash);
    assert_eq!(
        simulation.snapshot,
        ObservationSnapshot::from_state(&scenario, &state)?
    );
    Ok(())
}

#[test]
fn occasional_sales_do_not_save_a_warehouse_that_cannot_cover_costs() -> Result<()> {
    use bad_econ::domain::enterprise::EnterpriseState;
    let mut config = economy()?.config().clone();
    config.rent = Some(RentPolicy {
        household_daily: 0,
        firm_daily: 10,
    });
    config.behavior.wage_increment = 0;
    config.behavior.work_fatigue = 0;
    let store = &mut config.population.as_mut().unwrap().firms[1];
    store.cash = 100_000;
    store.inventories = Good::food([0, 0, 500]);
    let scenario = Scenario::new(config)?;
    let mut initial = scenario.initial()?;
    initial.set(Field::ReplacementCost, 4, 10);
    let mut simulation =
        Simulation::from_checkpoint(Checkpoint::new(&scenario, &initial)?, support::device(64)?)?;
    for _ in 0..29 {
        simulation.advance()?;
    }
    let struggling = simulation.download()?;
    assert!(struggling.get(Field::Cash, 4) > 90_000);
    assert!(struggling.get(Field::Sales, 4) > 0);
    assert!(struggling.get(Field::IdleDays, 4) > 20);
    assert!(
        struggling.get(Field::BusinessRevenue, 4)
            < struggling.get(Field::Sales, 4) * struggling.get(Field::ReplacementCost, 4) + 70
    );
    for _ in 0..6 {
        simulation.advance()?;
    }
    let closed = simulation.download()?;
    assert_eq!(
        closed.get(Field::EnterpriseStatus, 4),
        EnterpriseState::Closed as u64
    );
    assert_eq!(simulation.snapshot.distributors, 0);
    assert_eq!(simulation.snapshot.cash, initial.initial_cash);
    assert_eq!(
        simulation.snapshot,
        ObservationSnapshot::from_state(&scenario, &closed)?
    );
    Ok(())
}

#[test]
fn warehouse_construction_blocks_trading_but_still_requires_rent() -> Result<()> {
    use bad_econ::domain::{
        enterprise::EnterpriseState,
        property::{ConstructionOrder, Development, ProjectStatus},
    };
    let mut config = competing_warehouses()?;
    config.population.as_mut().unwrap().firms[1].owner = 1;
    config.property.as_mut().unwrap().orders = vec![ConstructionOrder {
        landowner: 1,
        site: 32,
        plots: 1,
        work: 20,
        development: Development::NewBuilding,
    }];
    let scenario = Scenario::new(config)?;
    let mut simulation = Simulation::new(scenario.clone(), support::device(64)?)?;
    for _ in 0..35 {
        simulation.advance()?;
    }
    let state = simulation.download()?;
    assert_eq!(state.get(Field::OccupiedPlots, 31), 0);
    assert_eq!(state.get(Field::ReservedPlots, 31), 1);
    assert_eq!(
        state.get(Field::ProjectStatus, 31),
        ProjectStatus::Building as u64
    );
    assert_eq!(
        state.get(Field::EnterpriseStatus, 31),
        EnterpriseState::Forming as u64
    );
    assert_eq!(state.get(Field::IdleDays, 31), 0);
    assert_eq!(state.get(Field::RetailOpen, 31), 0);
    assert_eq!(state.get(Field::Sales, 31), 0);
    assert_eq!(state.get(Field::RentTreasury, 31), 350);
    assert_eq!(
        simulation.snapshot,
        ObservationSnapshot::from_state(&scenario, &state)?
    );
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
fn entrepreneurial_household_can_open_a_grocery_without_guaranteed_customers() -> Result<()> {
    let mut config = economy()?.config().clone();
    config.entrepreneurship.interested_per_mille = 1000;
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
                cash: match identity {
                    1 => 12_000,
                    _ => 5000,
                },
                inventories: match identity {
                    1 => Good::food([1000, 0, 20]),
                    _ => Good::food([0, 0, 20]),
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
                    cash: 10_000,
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
    let mut initial = scenario.initial()?;
    initial.set(Field::InvestmentGood, 0, Good::Bread as u64);
    let mut simulation =
        Simulation::from_checkpoint(Checkpoint::new(&scenario, &initial)?, support::device(64)?)?;
    let mut staffed = 0;
    for _ in 0..20 {
        simulation.advance()?;
        let day = simulation.download()?;
        staffed += (scenario.actors()..day.actors.len())
            .filter(|&index| day.actors.at(index).owner == 0)
            .map(|index| day.get(Field::RetailOpen, index))
            .sum::<u64>();
    }
    let state = simulation.download()?;
    let store = (scenario.actors()..state.actors.len())
        .find(|&index| {
            state.get(Field::EnterpriseModel, index) == 1 && state.actors.at(index).owner == 0
        })
        .expect("household financed a grocery");
    assert_eq!(state.actors.at(store).owner, 0);
    assert_eq!(state.actors.at(store).position.x, 500);
    assert_eq!(state.get(Field::ConstructionRemaining, store), 0);
    assert_eq!(state.get(Field::InstalledCapital, store), 7);
    assert_eq!(state.get(Field::Produced, store), 0);
    assert!(staffed > 0);
    assert!(state.get(Field::Bread, store) > 0);
    assert_eq!(state.get(Field::Sales, store), 0);
    assert_eq!(
        state.get(Field::EnterpriseStatus, store),
        enterprise::EnterpriseState::Dormant as u64
    );
    assert_eq!(simulation.snapshot.cash, initial.initial_cash);
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

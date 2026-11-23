use super::support;
use anyhow::Result;
use bad_econ::{
    domain::{
        enterprise::{HouseholdProduction, Ownership, OwnershipShare},
        *,
    },
    simulation::Simulation,
};

#[test]
fn household_surplus_finances_construction_and_subsequent_paid_work() -> Result<()> {
    let mut config = Scenario::baseline(252, 73, Experiment::Concentrated)?
        .config()
        .clone();
    config.firms = 5;
    config.entrepreneurship.interested_per_mille = 1000;
    config.household_production = HouseholdProduction::industrial();
    config.land_access = Some(enterprise::LandAccess {
        total_plots: 1,
        allocations: vec![enterprise::LandGrant {
            household: 1,
            plots: 1,
        }],
    });
    config.demography.annual_births_per_million_fertile_women = 0;
    config.demography.mortality.background_annual_ppm = [0; 5];
    config.demography.mortality.occupational_annual_ppm = [0; Good::COUNT];
    config.distribution_rate = 0;
    config.forecast_weight = 0;
    config.behavior.price_response = 0;
    config.prices = [10, 15, 45];
    config.population = Some(Population {
        households: (1..=252)
            .map(|identity| HouseholdSeed {
                identity,
                position: Position { x: 500, y: 500 },
                cash: 5000,
                inventories: if identity == 1 {
                    Good::food([0, 1000, 4])
                } else {
                    Good::food([0, 0, 4])
                },
                fatigue: 0,
            })
            .collect(),
        firms: [
            Good::Tools,
            Good::Wood,
            Good::Bread,
            Good::Flour,
            Good::Bread,
        ]
        .into_iter()
        .enumerate()
        .map(|(index, good)| {
            let mut inventories = [0; Good::COUNT];
            inventories[good as usize - 1] = 2000;
            FirmSeed {
                identity: 1000 + index as u64,
                position: Position { x: 500, y: 500 },
                owner: 2,
                good,
                cash: match good {
                    Good::Tools | Good::Wood => 60,
                    _ => 0,
                },
                inventories,
                capacity: 1,
                technique: Technique {
                    labor_minutes: 60,
                    input_ticks: 10 * u32::from(good.inputs()[0].is_some()),
                    output_ticks: 10,
                },
                wage: 1,
                price: if good == Good::Bread { 60 } else { 10 },
            }
        })
        .collect(),
    });
    let shop = &mut config.population.as_mut().unwrap().firms[4];
    shop.cash = 100_000;
    shop.inventories[Good::Bread as usize - 1] = 0;
    config.distributors = vec![enterprise::Distributor {
        firm: shop.identity,
        good: Good::Bread,
        handling_minutes: 60,
        units_per_shift: 100,
        reserve_days: 3,
        margin_per_mille: 250,
    }];
    let scenario = Scenario::new(config)?;
    let mut initial = scenario.initial()?;
    initial.day = 120;
    let mut simulation =
        Simulation::from_checkpoint(Checkpoint::new(&scenario, &initial)?, support::device(64)?)?;
    let mut sales = 0;
    let mut saved = 0;
    for _ in 0..100 {
        simulation.advance()?;
        let state = simulation.download()?;
        sales += state.get(Field::Sales, 0);
        saved = saved.max(state.get(Field::Cash, 0));
    }
    assert!(sales > 0);
    assert!(saved >= 10_000);
    let state = simulation.download()?;
    assert!(simulation.snapshot.firm_entries > 0);
    let firm = state
        .actors
        .iter()
        .enumerate()
        .skip(scenario.actors())
        .find(|(_, actor)| actor.kind > 0 && actor.owner == 0)
        .map(|(index, _)| index)
        .expect("household funded a firm");
    assert_eq!(state.get(Field::ConstructionRemaining, firm), 0);
    assert!(state.get(Field::Produced, firm) > 0);
    assert_eq!(simulation.snapshot.cash, initial.initial_cash);
    assert_eq!(
        simulation.snapshot,
        ObservationSnapshot::from_state(&scenario, &state)?
    );
    let checkpoint = simulation.checkpoint()?;
    checkpoint.restore()?;
    let mut resumed = Simulation::new(scenario, support::device(64)?)?;
    resumed.command(bad_econ::runtime::SimulationCommand::Restore(Box::new(
        checkpoint,
    )))?;
    for _ in 0..3 {
        simulation.advance()?;
        resumed.advance()?;
        assert_eq!(simulation.download()?, resumed.download()?);
    }
    Ok(())
}

#[test]
fn explicit_partnership_shares_pay_exact_distributions_on_gpu() -> Result<()> {
    let mut config = Scenario::baseline(3, 7, Experiment::Concentrated)?
        .config()
        .clone();
    config.distribution_rate = 1000;
    config.reserve = config.firm_cash - 1000;
    config.ownership = vec![Ownership::new(Ownership {
        firm: 4,
        shares: vec![
            OwnershipShare {
                household: 1,
                units: 250_000,
            },
            OwnershipShare {
                household: 2,
                units: 750_000,
            },
        ],
    })?];
    let scenario = Scenario::new(config)?;
    let mut simulation = Simulation::new(scenario, support::device(64)?)?;
    simulation.advance()?;
    let state = simulation.download()?;
    assert_eq!(state.get(Field::Cash, 0), 5250);
    assert_eq!(state.get(Field::Cash, 1), 6750);
    assert_eq!(state.get(Field::Cash, 2), 6000);
    Ok(())
}

fn transition_economy() -> Result<Scenario> {
    let mut config = Scenario::baseline(3, 91, Experiment::Baseline)?
        .config()
        .clone();
    config.search_radius = 200;
    config.shipment_speed = 100;
    config.household_production = HouseholdProduction::industrial();
    config.distribution_rate = 0;
    config.demography.annual_births_per_million_fertile_women = 0;
    config.demography.mortality.background_annual_ppm = [0; 5];
    config.demography.mortality.occupational_annual_ppm = [0; Good::COUNT];
    config.demography.mortality.famine_daily_ppm_at_thirty_days = 0;
    config.population = Some(Population {
        households: (1..=3)
            .map(|identity| HouseholdSeed {
                identity,
                position: Position { x: 0, y: 0 },
                cash: 5000,
                inventories: [0; Good::COUNT],
                fatigue: 0,
            })
            .collect(),
        firms: [400, 700, 900]
            .into_iter()
            .enumerate()
            .map(|(index, x)| FirmSeed {
                identity: 4 + index as u64,
                position: Position { x, y: 0 },
                owner: 1,
                good: Good::Bread,
                cash: 0,
                inventories: Good::food([0, 0, if index == 0 { 200 } else { 0 }]),
                capacity: 1,
                technique: Technique {
                    labor_minutes: 60,
                    input_ticks: 1,
                    output_ticks: 1,
                },
                wage: 1,
                price: 10,
            })
            .collect(),
    });
    Scenario::new(config)
}

#[test]
fn repeated_migration_preserves_accounts_and_rebuilds_market_access() -> Result<()> {
    use bad_econ::runtime::SimulationCommand;
    let scenario = transition_economy()?;
    let mut simulation = Simulation::new(scenario.clone(), support::device(64)?)?;
    for _ in 0..7 {
        simulation.advance()?;
    }
    let mut state = simulation.download()?;
    assert_eq!(state.actors.at(0).position.x, 400);
    let bread = state.get(Field::Bread, 3);
    state.set(Field::Bread, 3, 0);
    state.set(Field::Bread, 5, bread);
    let cash = state.initial_cash;
    simulation.command(SimulationCommand::Restore(Box::new(Checkpoint::new(
        &scenario, &state,
    )?)))?;
    let mut observed_second_move = false;
    for _ in 0..25 {
        simulation.advance()?;
        let state = simulation.download()?;
        observed_second_move |= state.actors.at(0).position.x == 400
            && state.get(Field::MoveDue, 0) > 0
            && state.get(Field::MoveX, 0) == 900;
        if state.get(Field::MoveDue, 0) > 0 && state.get(Field::PreviousX, 0) == 400 {
            assert_eq!(state.get(Field::Hours, 0), 0);
        }
    }
    assert!(observed_second_move);
    assert_eq!(simulation.download()?.actors.at(0).position.x, 900);
    assert_eq!(simulation.snapshot.cash, cash);
    simulation.checkpoint()?.restore()?;
    Ok(())
}

#[test]
fn liquidation_distributes_assets_and_deliveries_by_shares_without_reusing_identity() -> Result<()>
{
    use bad_econ::{domain::enterprise::EnterpriseState, runtime::SimulationCommand};
    let mut config = transition_economy()?.config().clone();
    config.shipment_speed = 10;
    config.ownership = vec![Ownership::new(Ownership {
        firm: 4,
        shares: vec![
            OwnershipShare {
                household: 1,
                units: 500_000,
            },
            OwnershipShare {
                household: 2,
                units: 500_000,
            },
        ],
    })?];
    let population = config.population.as_mut().unwrap();
    for household in &mut population.households {
        household.cash = 0;
    }
    for (index, firm) in population.firms.iter_mut().enumerate() {
        firm.owner = index as u64 + 1;
    }
    population.firms[0].inventories = [0; Good::COUNT];
    population.firms[0].inventories[Good::Tools as usize - 1] = 5;
    population.firms[0].inventories[Good::Bread as usize - 1] = 7;
    let scenario = Scenario::new(config)?;
    let mut state = scenario.initial()?;
    state.day = 26;
    state.set(Field::Bread, 3, 0);
    let slot = 28 % scenario.ring();
    let delivery = state
        .shipment_coordinate(Good::Bread, slot, scenario.ring(), 3)
        .0 as usize;
    state.values[delivery] = 7;
    state.values[(TRANSIT_FIELDS + Good::Bread as usize - 1) * state.actors.len() + 3] = 7;
    state.set(Field::Transit, 3, 7);
    state.set(Field::EnterpriseStatus, 3, EnterpriseState::Dormant as u64);
    state.set(Field::IdleDays, 3, 89);
    let next_identity = state.next_identity;
    let mut simulation = Simulation::new(scenario.clone(), support::device(64)?)?;
    simulation.command(SimulationCommand::Restore(Box::new(Checkpoint::new(
        &scenario, &state,
    )?)))?;
    simulation.advance()?;
    let closed = simulation.download()?;
    assert_eq!(
        closed.get(Field::EnterpriseStatus, 3),
        EnterpriseState::Closed as u64
    );
    assert_eq!(closed.get(Field::Tools, 0), 2);
    assert_eq!(closed.get(Field::Tools, 1), 3);
    assert_eq!(closed.get(Field::Transit, 0), 3);
    assert_eq!(closed.get(Field::Transit, 1), 4);
    assert_eq!(closed.get(Field::Transit, 3), 0);
    assert_eq!(closed.next_identity, next_identity);
    simulation.checkpoint()?.restore()?;
    simulation.advance()?;
    assert_eq!(simulation.download()?.get(Field::InvestmentFirm, 0), 0);
    simulation.advance()?;
    assert_eq!(simulation.download()?.get(Field::Transit, 0), 0);
    Ok(())
}

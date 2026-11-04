#![cfg(feature = "compute")]
use crate::support;
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
    config.firms = 4;
    config.household_production = HouseholdProduction::industrial();
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
                    Good::food([1000, 0, 4])
                } else {
                    Good::food([0, 0, 4])
                },
                fatigue: 0,
            })
            .collect(),
        firms: [Good::Tools, Good::Wood, Good::Bread, Good::Flour]
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
                    cash: 0,
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
    let scenario = Scenario::new(config)?;
    let initial = scenario.initial()?;
    let mut simulation = Simulation::new(scenario.clone(), support::device(64)?)?;
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
    let mut resumed = Simulation::new(scenario, support::device(128)?)?;
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

fn capital_economy() -> Result<Scenario> {
    let mut config = Scenario::baseline(30, 9, Experiment::Concentrated)?
        .config()
        .clone();
    config.household_production = HouseholdProduction::industrial();
    config.distribution_rate = 0;
    config.forecast_weight = 0;
    config.demography.annual_births_per_million_fertile_women = 0;
    config.demography.mortality.background_annual_ppm = [0; 5];
    config.demography.mortality.occupational_annual_ppm = [0; Good::COUNT];
    config.payroll = Some(PayrollPolicy {
        base_rate: 100,
        employee_charge: 50,
        brackets: Vec::new(),
    });
    config.population = Some(Population {
        households: (1..=30)
            .map(|identity| HouseholdSeed {
                identity,
                position: Position { x: 500, y: 500 },
                cash: 50_000,
                inventories: Good::food([0, 0, 20]),
                fatigue: 0,
            })
            .collect(),
        firms: [Good::Bread, Good::Tools, Good::Wood]
            .into_iter()
            .enumerate()
            .map(|(index, good)| {
                let mut inventories = [0; Good::COUNT];
                if index == 0 {
                    inventories[Good::Flour as usize - 1] = 1000;
                    inventories[Good::Tools as usize - 1] = 2;
                    inventories[Good::Wood as usize - 1] = 5;
                }
                FirmSeed {
                    identity: 31 + index as u64,
                    position: Position { x: 500, y: 500 },
                    owner: 1 + index as u64,
                    good,
                    cash: 100_000,
                    inventories,
                    capacity: 1,
                    technique: Technique {
                        labor_minutes: 60,
                        input_ticks: u32::from(good.inputs()[0].is_some()),
                        output_ticks: 1,
                    },
                    wage: 1,
                    price: 450,
                }
            })
            .collect(),
    });
    Scenario::new(config)
}

#[test]
fn retained_earnings_build_capacity_after_paid_construction_and_payroll() -> Result<()> {
    use bad_econ::runtime::SimulationCommand;
    let scenario = capital_economy()?;
    let mut state = scenario.initial()?;
    state.set(Field::Demand, 30, 10);
    let initial_cash = state.initial_cash;
    let mut simulation = Simulation::new(scenario.clone(), support::device(64)?)?;
    simulation.command(SimulationCommand::Restore(Box::new(Checkpoint::new(
        &scenario, &state,
    )?)))?;
    simulation.advance()?;
    let forming = simulation.download()?;
    assert_eq!(forming.actors.at(30).capacity, 1);
    assert_eq!(forming.get(Field::PendingCapacity, 30), 1);
    assert!(forming.get(Field::ConstructionRemaining, 30) > 0);
    assert_eq!(forming.get(Field::InstalledCapital, 30), 7);
    assert_eq!(
        forming.get(Field::EnterpriseStatus, 30),
        bad_econ::domain::enterprise::EnterpriseState::Operating as u64
    );
    let mut production = forming.get(Field::Produced, 30);
    for _ in 0..3 {
        simulation.advance()?;
        let building = simulation.download()?;
        assert!(building.get(Field::Produced, 30) > production);
        production = building.get(Field::Produced, 30);
    }
    let completed = simulation.download()?;
    assert_eq!(completed.actors.at(30).capacity, 2);
    assert_eq!(completed.get(Field::ConstructionRemaining, 30), 0);
    assert!(completed.get(Field::GrossIncome, 0) >= 720);
    assert!(completed.get(Field::TaxTreasury, 0) >= 72);
    assert_eq!(simulation.snapshot.cash, initial_cash);
    simulation.checkpoint()?.restore()?;
    Ok(())
}

#[test]
fn multiple_owned_projects_rotate_and_pay_only_affordable_construction() -> Result<()> {
    use bad_econ::{domain::enterprise::EnterpriseState, simulation::Trace};
    let mut config = capital_economy()?.config().clone();
    let population = config.population.as_mut().unwrap();
    for firm in &mut population.firms {
        firm.owner = 1;
    }
    population.firms[0].cash = 60;
    population.firms[1].cash = 120;
    let scenario = Scenario::new(config)?;
    let mut state = scenario.initial()?;
    for firm in [30, 31] {
        state.set(
            Field::EnterpriseStatus,
            firm,
            EnterpriseState::Forming as u64,
        );
        state.set(Field::ConstructionRemaining, firm, 120);
    }
    assert_eq!(state.get(Field::InvestmentFirm, 0), 33);
    let checkpoint = Checkpoint::new(&scenario, &state)?;
    let mut simulation = Simulation::from_checkpoint(checkpoint.clone(), support::device(64)?)?;
    let mut replay = Simulation::from_checkpoint(checkpoint, support::device(128)?)?;
    for phase in Phase::ALL {
        simulation.phase(phase, Trace::Discard)?;
        replay.phase(phase, Trace::Discard)?;
        let current = simulation.download()?;
        assert_eq!(current, replay.download()?);
        if phase == Phase::WorkPayment {
            assert_eq!(current.get(Field::Cash, 30), 0);
            assert_eq!(current.get(Field::ConstructionRemaining, 30), 60);
            assert_eq!(current.get(Field::ConstructionRemaining, 31), 120);
            assert!(current.get(Field::GrossIncome, 0) >= 60);
        }
    }
    simulation.advance()?;
    replay.advance()?;
    let completed = simulation.download()?;
    assert_eq!(completed, replay.download()?);
    assert_eq!(completed.get(Field::ConstructionRemaining, 31), 0);
    assert_eq!(
        completed.get(Field::EnterpriseStatus, 31),
        EnterpriseState::Dormant as u64
    );
    assert_eq!(simulation.snapshot.cash, state.initial_cash);
    for (identity, status, minutes) in [
        (31, EnterpriseState::Forming, 60),
        (32, EnterpriseState::Dormant, 0),
    ] {
        let request = InspectionRequest::new(identity, &scenario)?;
        let inspection = simulation.inspect(request.clone())?;
        assert_eq!(inspection, Inspection::from_state(&completed, &request)?);
        assert_eq!(inspection.enterprise_state, Some(status));
        assert_eq!(inspection.construction_minutes_remaining, minutes);
    }
    simulation.checkpoint()?.restore()?;
    Ok(())
}

#[test]
fn outside_household_investment_issues_shares_and_transfers_savings() -> Result<()> {
    use bad_econ::runtime::SimulationCommand;
    let mut config = capital_economy()?.config().clone();
    let population = config.population.as_mut().unwrap();
    for household in &mut population.households {
        household.inventories = [0; Good::COUNT];
    }
    population.firms[0].inventories[Good::Bread as usize - 1] = 500;
    population.firms[0].cash = 100;
    let scenario = Scenario::new(config)?;
    let mut state = scenario.initial()?;
    state.day = 29;
    state.set(Field::Demand, 30, 500);
    let initial_cash = state.initial_cash;
    let mut simulation = Simulation::new(scenario.clone(), support::device(64)?)?;
    simulation.command(SimulationCommand::Restore(Box::new(Checkpoint::new(
        &scenario, &state,
    )?)))?;
    simulation.advance()?;
    let invested = simulation.download()?;
    let ownership = invested
        .ownership
        .iter()
        .find(|ownership| ownership.firm == 31)
        .unwrap();
    assert!(ownership.shares.len() > 1);
    assert_eq!(
        ownership
            .shares
            .iter()
            .map(|share| share.units)
            .sum::<u32>(),
        1_000_000
    );
    assert!((0..30).any(|household| invested.get(Field::EquityInvested, household) == 5000));
    assert_eq!(simulation.snapshot.cash, initial_cash);
    simulation.checkpoint()?.restore()?;
    Ok(())
}

#[test]
fn purchased_tools_fund_maintenance_and_safety_without_creating_equipment() -> Result<()> {
    use bad_econ::domain::enterprise::{FirmConditions, WorkingConditions};
    let mut config = capital_economy()?.config().clone();
    let population = config.population.as_mut().unwrap();
    population.firms[0].inventories[Good::Tools as usize - 1] = 0;
    population.firms[1].inventories[Good::Tools as usize - 1] = 100;
    config.working_conditions = vec![FirmConditions {
        firm: 31,
        conditions: WorkingConditions::new(WorkingConditions {
            equipment_condition: 400,
            safety_provision: 0,
        })?,
    }];
    let scenario = Scenario::new(config)?;
    let initial = scenario.initial()?;
    let mut simulation = Simulation::new(scenario, support::device(64)?)?;
    simulation.advance()?;
    let pending = simulation.download()?;
    assert!(pending.get(Field::Transit, 30) >= 2);
    assert_eq!(pending.get(Field::SafetyProvision, 30), 0);
    simulation.advance()?;
    let maintained = simulation.download()?;
    assert_eq!(maintained.get(Field::EquipmentCondition, 30), 1000);
    assert!(maintained.get(Field::SafetyProvision, 30) > 0);
    assert!(
        maintained.values[(USED_FIELDS + Good::Tools as usize - 1) * maintained.actors.len() + 30]
            >= 2
    );
    assert!(maintained.get(Field::Cash, 31) > initial.get(Field::Cash, 31));
    assert_eq!(simulation.snapshot.cash, initial.initial_cash);
    Ok(())
}

#[test]
fn observed_tool_demand_and_input_access_fund_a_nonfood_enterprise() -> Result<()> {
    let mut config = capital_economy()?.config().clone();
    config.behavior.price_response = 0;
    let population = config.population.as_mut().unwrap();
    for household in &mut population.households {
        household.inventories = [0; Good::COUNT];
    }
    let founder = &mut population.households[3];
    founder.inventories = Good::food([1000, 0, 100]);
    founder.inventories[Good::Tools as usize - 1] = 2;
    founder.inventories[Good::Wood as usize - 1] = 5;
    population.firms[0].good = Good::Iron;
    population.firms[0].technique = Technique::new(60, 0, 1)?;
    for (index, firm) in population.firms.iter_mut().enumerate() {
        firm.inventories = [0; Good::COUNT];
        firm.inventories[firm.good as usize - 1] = match index {
            1 => 2,
            _ => 100,
        };
        firm.price = match index {
            1 => 450,
            _ => 10,
        };
    }
    let scenario = Scenario::new(config)?;
    let mut simulation = Simulation::new(scenario.clone(), support::device(64)?)?;
    for _ in 0..30 {
        simulation.advance()?;
    }
    let state = simulation.download()?;
    let firm = (scenario.actors()..state.actors.len())
        .find(|&index| state.actors.at(index).kind == Good::Tools as u32)
        .expect("observed tool demand funded a tool workshop");
    assert_eq!(state.get(Field::FirstInput, firm), Good::Iron as u64);
    assert_eq!(state.get(Field::SecondInput, firm), Good::Wood as u64);
    assert_eq!(state.get(Field::ConstructionRemaining, firm), 0);
    assert!(state.get(Field::Produced, firm) > 0);
    assert_eq!(state.get(Field::EnterpriseModel, firm), 0);
    assert_eq!(
        simulation.snapshot,
        ObservationSnapshot::from_state(&scenario, &state)?
    );
    simulation.checkpoint()?.restore()?;
    Ok(())
}

#[test]
fn unfunded_food_firms_enter_liquidation_without_erasing_accounts() -> Result<()> {
    let mut config = transition_economy()?.config().clone();
    let population = config
        .population
        .as_mut()
        .expect("explicit transition economy");
    for household in &mut population.households {
        household.cash = 0;
        household.inventories = [0; Good::COUNT];
    }
    for firm in &mut population.firms {
        firm.cash = 0;
        firm.inventories = [0; Good::COUNT];
    }
    let scenario = Scenario::new(config)?;
    let initial = scenario.initial()?;
    let identities = initial
        .actors
        .iter()
        .map(|actor| actor.identity)
        .collect::<Vec<_>>();
    let mut simulation = Simulation::new(scenario.clone(), support::device(64)?)?;
    for _ in 0..90 {
        simulation.advance()?;
    }
    let liquidating = simulation.download()?;
    for firm in 3..6 {
        assert_eq!(
            liquidating.get(Field::EnterpriseStatus, firm),
            enterprise::EnterpriseState::Liquidating as u64
        );
    }
    for _ in 0..30 {
        simulation.advance()?;
    }
    assert_eq!(simulation.snapshot.firm_closures, 3);
    assert_eq!(simulation.snapshot.cash, initial.initial_cash);
    let state = simulation.download()?;
    assert_eq!(
        state
            .actors
            .iter()
            .map(|actor| actor.identity)
            .collect::<Vec<_>>(),
        identities
    );
    assert_eq!(
        simulation.snapshot,
        ObservationSnapshot::from_state(&scenario, &state)?
    );
    simulation.checkpoint()?.restore()?;
    Ok(())
}

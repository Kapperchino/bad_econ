#![cfg(feature = "compute")]
use super::support;
use anyhow::Result;
use bad_econ::{
    domain::{
        enterprise::{EntrepreneurialInterest, HouseholdProduction, Ownership, OwnershipShare},
        *,
    },
    simulation::Simulation,
};

#[test]
#[ignore = "extended scenario coverage; run cargo gpu-extended"]
fn industrial_transition_grows_work_and_output_with_supported_births_on_gpu() -> Result<()> {
    let scenario = Scenario::industrial(1000, 42, Experiment::Baseline)?;
    let mut simulation = Simulation::new(scenario, support::device(64)?)?;
    let initial = simulation.download()?;
    let cash = simulation.snapshot.cash;
    let mut employed = 0;
    let mut produced = 0;
    let mut first_annual = 0;
    let mut last_annual = 0;
    let mut rural = initial
        .actors
        .iter()
        .enumerate()
        .filter(|(household, actor)| {
            actor.kind == 0 && initial.get(Field::LandPlots, *household) > 0
        })
        .count();
    assert_eq!(rural, 900);
    for year in 0..3 {
        for _ in 0..365 {
            simulation.advance()?;
        }
        let state = simulation.download()?;
        let households = state
            .actors
            .iter()
            .enumerate()
            .filter(|(household, actor)| {
                actor.kind == 0 && state.get(Field::LandPlots, *household) > 0
            })
            .count();
        let bread = simulation.snapshot.produced[Good::Bread as usize - 1]
            - simulation.snapshot.household_production[Good::Bread as usize - 1];
        let population = &simulation.snapshot.population;
        assert_eq!(
            population.living + population.deaths.iter().sum::<u64>(),
            initial.initial_population + population.births
        );
        assert!(simulation.snapshot.employed > employed);
        last_annual = bread - produced;
        first_annual = match year {
            0 => last_annual,
            _ => first_annual,
        };
        assert!(households < rural);
        assert!(
            simulation
                .snapshot
                .economy
                .wage_employed(simulation.snapshot.employed)
                <= simulation.snapshot.economy.workforce
        );
        assert_eq!(simulation.snapshot.cash, cash);
        employed = simulation.snapshot.employed;
        produced = bread;
        rural = households;
    }
    assert!(last_annual > first_annual);
    simulation.checkpoint()?.restore()?;
    Ok(())
}

fn urbanization_scenario(wage: u32) -> Result<Scenario> {
    use bad_econ::domain::demography::{PersonSeed, Sex};
    let mut config = Scenario::baseline(2, 42, Experiment::Concentrated)?
        .config()
        .clone();
    config.economy = Economy::Industrial;
    config.firms = 4;
    config.household_production = HouseholdProduction::industrial();
    config.food = Some(food::FoodPolicy::default());
    config.prices = [9, 30, 20];
    config.search_radius = 1000;
    config.shipment_speed = 100;
    config.forecast_weight = 0;
    config.distribution_rate = 0;
    config.behavior.price_response = 0;
    config.behavior.wage_increment = 0;
    config.entrepreneurship.interested_per_mille = 0;
    config.expansion = enterprise::ExpansionPolicy {
        small_business_per_mille: 0,
        industrial_per_mille: 0,
    };
    config.demography.annual_births_per_million_fertile_women = 0;
    config.demography.people = (1..=2)
        .map(|household| PersonSeed {
            identity: 1000 + household,
            household,
            age_days: 30 * 365,
            sex: Sex::Male,
        })
        .collect();
    config.resources = ResourceMap::Explicit {
        regions: [Good::Wood, Good::Grain]
            .into_iter()
            .map(|good| {
                ResourceRegion::new(
                    good,
                    Position { x: 0, y: 0 },
                    Position { x: 1000, y: 1000 },
                    1000,
                )
            })
            .collect::<Result<_>>()?,
    };
    config.population = Some(Population {
        households: (1..=2)
            .map(|identity| {
                let mut inventories = Good::food([5, 0, 8]);
                inventories[Good::Tools as usize - 1] = 1;
                HouseholdSeed {
                    identity,
                    position: Position {
                        x: if identity == 1 { 0 } else { 700 },
                        y: 0,
                    },
                    cash: 10_000,
                    inventories,
                    fatigue: 0,
                }
            })
            .collect(),
        firms: [Good::Clothes, Good::Bread, Good::Wood, Good::Grain]
            .into_iter()
            .enumerate()
            .map(|(index, good)| {
                let mut inventories = [0; Good::COUNT];
                inventories[good as usize - 1] = 1000;
                inventories[Good::Cloth as usize - 1] = 1000;
                inventories[Good::Linen as usize - 1] = 1000;
                Ok(FirmSeed {
                    identity: 3 + index as u64,
                    position: Position { x: 700, y: 0 },
                    owner: 2,
                    good,
                    cash: match good {
                        Good::Clothes => 1_000_000,
                        _ => 0,
                    },
                    inventories,
                    capacity: 2,
                    technique: Technique::new(20, u32::from(good.inputs()[0].is_some()), 1)?,
                    wage,
                    price: match good {
                        Good::Bread => 20,
                        Good::Wood => 1,
                        _ => 100,
                    },
                })
            })
            .collect::<Result<_>>()?,
    });
    Scenario::new(config)
}

#[test]
fn sustained_urban_jobs_attract_farmers_before_food_shortages_on_gpu() -> Result<()> {
    let scenario = urbanization_scenario(12)?;
    let mut initial = scenario.initial()?;
    initial.day = 120;
    initial.set(Field::LandPlots, 0, 1);
    initial.set(Field::Demand, 2, 1000);
    let mut simulation =
        Simulation::from_checkpoint(Checkpoint::new(&scenario, &initial)?, support::device(64)?)?;
    simulation.advance()?;
    let first = simulation.download()?;
    assert_eq!(first.get(Field::Employed, 0), 1);
    assert!(
        simulation.snapshot.economy.workforce
            >= simulation
                .snapshot
                .economy
                .wage_employed(simulation.snapshot.employed)
    );
    assert_eq!(first.get(Field::MoveDue, 0), 0);
    let mut resumed = Simulation::from_checkpoint(simulation.checkpoint()?, support::device(64)?)?;
    let mut urban_work = 0;
    for _ in 1..40 {
        simulation.advance()?;
        resumed.advance()?;
        let state = simulation.download()?;
        assert_eq!(state, resumed.download()?);
        assert_eq!(simulation.snapshot.cash, initial.initial_cash);
        assert_eq!(state.get(Field::Shortage, 0), 0);
        urban_work += match state.actors.at(0).position.x {
            700 => state.get(Field::Employed, 0),
            _ => 0,
        };
    }
    let state = simulation.download()?;
    assert_eq!(state.actors.at(0).position.x, 700);
    assert_eq!(state.get(Field::LandPlots, 0), 0);
    assert_eq!(state.get(Field::CropWork, 0), 0);
    assert!(state.get(Field::IncomeForecast, 0) <= 240_000);
    assert!(urban_work > 0);
    Ok(())
}

#[test]
fn jobs_that_cannot_cover_city_living_do_not_displace_farmers_on_gpu() -> Result<()> {
    let scenario = urbanization_scenario(1)?;
    let mut initial = scenario.initial()?;
    initial.day = 120;
    initial.set(Field::LandPlots, 0, 1);
    initial.set(Field::Demand, 2, 1000);
    let mut simulation =
        Simulation::from_checkpoint(Checkpoint::new(&scenario, &initial)?, support::device(64)?)?;
    for _ in 0..40 {
        simulation.advance()?;
    }
    let state = simulation.download()?;
    assert_eq!(state.actors.at(0).position.x, 0);
    assert_eq!(state.get(Field::LandPlots, 0), 1);
    assert_eq!(state.get(Field::MoveDue, 0), 0);
    assert_eq!(simulation.snapshot.cash, initial.initial_cash);
    Ok(())
}

fn close_employers(state: &mut State) {
    for firm in 0..state.actors.len() {
        if state.actors.at(firm).kind > 0 {
            state.set(
                Field::EnterpriseStatus,
                firm,
                enterprise::EnterpriseState::Closed as u64,
            );
        }
    }
}

#[test]
fn temporary_rural_paid_work_does_not_reduce_farmer_population_on_gpu() -> Result<()> {
    use bad_econ::domain::demography::Occupation;

    let scenario = urbanization_scenario(1)?;
    let mut initial = scenario.initial()?;
    initial.day = 120;
    initial.set(Field::LandPlots, 0, 1);
    initial.set(Field::Demand, 2, 1000);
    let mut simulation =
        Simulation::from_checkpoint(Checkpoint::new(&scenario, &initial)?, support::device(64)?)?;
    simulation.advance()?;
    let mut state = simulation.download()?;
    let farmers = simulation.snapshot.population.occupations[Occupation::Farmer as usize];
    assert_eq!(state.get(Field::Employed, 0), 1);
    assert_eq!(farmers, 1);
    assert_eq!(simulation.snapshot.economy.rural_employed, 1);
    assert_eq!(simulation.snapshot.economy.subsistence, 1);
    assert_eq!(
        simulation.snapshot,
        ObservationSnapshot::from_state(&scenario, &state)?
    );
    close_employers(&mut state);
    let mut simulation =
        Simulation::from_checkpoint(Checkpoint::new(&scenario, &state)?, support::device(64)?)?;
    for _ in 0..4 {
        simulation.advance()?;
        let state = simulation.download()?;
        assert_eq!(state.get(Field::Employed, 0), 0);
        assert_eq!(state.get(Field::LandPlots, 0), 1);
        assert_eq!(simulation.snapshot.economy.rural_employed, 0);
        assert_eq!(simulation.snapshot.economy.subsistence, 1);
        assert_eq!(
            simulation.snapshot.population.occupations[Occupation::Farmer as usize],
            farmers
        );
        assert_eq!(
            simulation.snapshot,
            ObservationSnapshot::from_state(&scenario, &state)?
        );
    }
    Ok(())
}

#[test]
fn nearby_sustained_jobs_commit_farmers_to_wage_work_after_job_loss_on_gpu() -> Result<()> {
    use bad_econ::domain::demography::Occupation;

    let mut config = urbanization_scenario(12)?.config().clone();
    config.population.as_mut().unwrap().households[0].position = Position { x: 700, y: 0 };
    let scenario = Scenario::new(config)?;
    let mut initial = scenario.initial()?;
    initial.day = 120;
    initial.set(Field::LandPlots, 0, 1);
    initial.set(Field::Demand, 2, 1000);
    let mut simulation =
        Simulation::from_checkpoint(Checkpoint::new(&scenario, &initial)?, support::device(64)?)?;
    simulation.advance()?;
    assert_eq!(simulation.download()?.get(Field::LandPlots, 0), 1);
    let mut replay = Simulation::from_checkpoint(simulation.checkpoint()?, support::device(64)?)?;
    for _ in 1..32 {
        simulation.advance()?;
        replay.advance()?;
        let state = simulation.download()?;
        assert_eq!(state, replay.download()?);
        assert_eq!(simulation.snapshot.cash, initial.initial_cash);
        assert_eq!(state.get(Field::MoveDue, 0), 0);
        assert_eq!(state.actors.at(0).position, initial.actors.at(0).position);
        assert!(
            simulation
                .snapshot
                .economy
                .wage_employed(simulation.snapshot.employed)
                <= simulation.snapshot.economy.workforce
        );
        assert_eq!(
            simulation.snapshot,
            ObservationSnapshot::from_state(&scenario, &state)?
        );
    }
    let mut state = simulation.download()?;
    assert_eq!(state.get(Field::LandPlots, 0), 0);
    assert_eq!(state.get(Field::CropWork, 0), 0);
    assert_eq!(
        simulation.snapshot.population.occupations[Occupation::Farmer as usize],
        0
    );
    close_employers(&mut state);
    let mut simulation =
        Simulation::from_checkpoint(Checkpoint::new(&scenario, &state)?, support::device(64)?)?;
    for _ in 0..4 {
        simulation.advance()?;
        let state = simulation.download()?;
        assert_eq!(state.get(Field::Employed, 0), 0);
        assert_eq!(state.get(Field::LandPlots, 0), 0);
        assert_eq!(state.get(Field::CropWork, 0), 0);
        assert_eq!(
            simulation.snapshot.population.occupations[Occupation::Farmer as usize],
            0
        );
        assert!(simulation.snapshot.population.occupations[Occupation::Unemployed as usize] > 0);
        assert_eq!(
            simulation.snapshot,
            ObservationSnapshot::from_state(&scenario, &state)?
        );
    }
    Ok(())
}

#[test]
fn entrepreneurial_interest_limits_entry_even_with_abundant_savings_on_gpu() -> Result<()> {
    let mut config = Scenario::baseline(1000, 42, Experiment::Concentrated)?
        .config()
        .clone();
    config.household_production = HouseholdProduction::industrial();
    config.firms = 3;
    config.distribution_rate = 0;
    config.behavior.price_response = 0;
    config.demography.annual_births_per_million_fertile_women = 0;
    config.demography.mortality.background_annual_ppm = [0; 5];
    config.demography.mortality.occupational_annual_ppm = [0; Good::COUNT];
    config.population = Some(Population {
        households: (1..=1000)
            .map(|identity| {
                let mut inventories = Good::food([0, 0, 1000]);
                inventories[Good::Tools as usize - 1] = 2;
                inventories[Good::Wood as usize - 1] = 5;
                HouseholdSeed {
                    identity,
                    position: Position { x: 500, y: 500 },
                    cash: 50_000,
                    inventories,
                    fatigue: 0,
                }
            })
            .collect(),
        firms: [Good::Grain, Good::Flour, Good::Bread]
            .into_iter()
            .enumerate()
            .map(|(index, good)| FirmSeed {
                identity: 1001 + index as u64,
                position: Position { x: 500, y: 500 },
                owner: 1,
                good,
                cash: 100_000,
                inventories: Good::food([10_000, 10_000, 10_000]),
                capacity: 1,
                technique: Technique {
                    labor_minutes: 60,
                    input_ticks: 20 * u32::from(good.inputs()[0].is_some()),
                    output_ticks: 20,
                },
                wage: 1,
                price: [10, 15, 60][index],
            })
            .collect(),
    });
    let scenario = Scenario::new(config)?;
    let interested = (1..=1000)
        .filter(|&identity| {
            scenario.config().entrepreneurship.interest(42, identity)
                == EntrepreneurialInterest::Interested
        })
        .count();
    assert!((60..=100).contains(&interested));
    let mut initial = scenario.initial()?;
    initial.day = 120;
    let mut simulation =
        Simulation::from_checkpoint(Checkpoint::new(&scenario, &initial)?, support::device(64)?)?;
    for _ in 0..60 {
        simulation.advance()?;
        assert!(simulation.snapshot.firm_entries <= interested as u64);
        assert_eq!(simulation.snapshot.cash, initial.initial_cash);
    }
    assert!(simulation.snapshot.firm_entries > 0);
    let state = simulation.download()?;
    for firm in state.actors.iter().skip(initial.actors.len()) {
        let owner = state.actors.at(firm.owner as usize);
        assert_eq!(
            scenario
                .config()
                .entrepreneurship
                .interest(42, owner.identity),
            EntrepreneurialInterest::Interested
        );
    }
    for household in 0..1000 {
        if scenario
            .config()
            .entrepreneurship
            .interest(42, household as u64 + 1)
            == EntrepreneurialInterest::Uninterested
        {
            assert_eq!(state.get(Field::InvestmentGood, household), 0);
        }
    }
    let mut replay = Simulation::from_checkpoint(simulation.checkpoint()?, support::device(64)?)?;
    simulation.advance()?;
    replay.advance()?;
    assert_eq!(simulation.download()?, replay.download()?);
    Ok(())
}

#[test]
fn household_surplus_finances_construction_and_subsequent_paid_work() -> Result<()> {
    let mut config = Scenario::baseline(252, 73, Experiment::Concentrated)?
        .config()
        .clone();
    config.firms = 4;
    config.entrepreneurship.interested_per_mille = 1000;
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
#[ignore = "extended scenario coverage; run cargo gpu-extended"]
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
    config.expansion = enterprise::ExpansionPolicy {
        small_business_per_mille: 1000,
        industrial_per_mille: 1000,
    };
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
#[ignore = "extended scenario coverage; run cargo gpu-extended"]
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
#[ignore = "extended scenario coverage; run cargo gpu-extended"]
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
    let mut replay = Simulation::from_checkpoint(checkpoint, support::device(64)?)?;
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
        assert_eq!(
            inspection,
            Inspection::from_state(&scenario, &completed, &request)?
        );
        assert_eq!(inspection.enterprise_state, Some(status));
        assert_eq!(inspection.construction_minutes_remaining, minutes);
    }
    simulation.checkpoint()?.restore()?;
    Ok(())
}

#[test]
#[ignore = "extended scenario coverage; run cargo gpu-extended"]
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
#[ignore = "extended scenario coverage; run cargo gpu-extended"]
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
#[ignore = "extended scenario coverage; run cargo gpu-extended"]
fn local_tool_quotes_and_input_access_fund_a_nonfood_enterprise() -> Result<()> {
    let mut config = capital_economy()?.config().clone();
    config.entrepreneurship.interested_per_mille = 1000;
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
        .expect("an interested household funded a tool workshop");
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
fn unfunded_food_firms_close_without_erasing_accounts() -> Result<()> {
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
    let mut initial = scenario.initial()?;
    for firm in 3..6 {
        initial.set(Field::IdleDays, firm, 29);
        initial.set(
            Field::EnterpriseStatus,
            firm,
            enterprise::EnterpriseState::Dormant as u64,
        );
    }
    let identities = initial
        .actors
        .iter()
        .map(|actor| actor.identity)
        .collect::<Vec<_>>();
    let mut simulation =
        Simulation::from_checkpoint(Checkpoint::new(&scenario, &initial)?, support::device(64)?)?;
    simulation.advance()?;
    let closed = simulation.download()?;
    for firm in 3..6 {
        assert_eq!(
            closed.get(Field::EnterpriseStatus, firm),
            enterprise::EnterpriseState::Closed as u64
        );
    }
    simulation.advance()?;
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

#[test]
#[ignore = "extended scenario coverage; run cargo gpu-extended"]
fn industrial_owners_expand_more_readily_than_small_proprietors_on_gpu() -> Result<()> {
    for capacity in [1, 20] {
        let mut config = capital_economy()?.config().clone();
        config.expansion = enterprise::ExpansionPolicy {
            small_business_per_mille: 0,
            industrial_per_mille: 1000,
        };
        config.population.as_mut().unwrap().firms[0].capacity = capacity;
        let scenario = Scenario::new(config)?;
        let mut initial = scenario.initial()?;
        initial.set(Field::Demand, 30, 100);
        let mut simulation = Simulation::from_checkpoint(
            Checkpoint::new(&scenario, &initial)?,
            support::device(64)?,
        )?;
        simulation.advance()?;
        let state = simulation.download()?;
        assert_eq!(
            state.get(Field::PendingCapacity, 30),
            u64::from(capacity == 20)
        );
        assert_eq!(simulation.snapshot.cash, initial.initial_cash);
    }
    Ok(())
}

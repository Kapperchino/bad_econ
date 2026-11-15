#![cfg(feature = "compute")]

use super::support;
use anyhow::Result;
use bad_econ::{
    domain::{demography::*, enterprise::HouseholdProduction, food::FoodPolicy, *},
    simulation::Simulation,
};

#[test]
fn default_metric_food_recipes_and_opening_prices_on_gpu() -> Result<()> {
    let scenario = Scenario::industrial(96, 42, Experiment::Concentrated)?;
    assert_eq!(scenario.config().prices, [697, 1087, 468]);
    let mut simulation = Simulation::new(scenario.clone(), support::device(64)?)?;
    let opening = simulation.download()?;
    for (index, actor) in opening.actors.iter().enumerate() {
        let expected = match actor.kind {
            kind if kind == Good::Flour as u32 => Some(Technique::new(20, 25, 20)?),
            kind if kind == Good::Bread as u32
                && opening.get(Field::EnterpriseModel, index) == 0 =>
            {
                Some(Technique::new(20, 25, 64)?)
            }
            _ => None,
        };
        if let Some(expected) = expected {
            assert_eq!(scenario.technique(index), expected);
            for method in ProductionMethod::for_good(Good::from_index(actor.kind)?) {
                let upgraded = scenario.production_recipe(index, Some(method))?.technique;
                assert_eq!(
                    upgraded.input_ticks * expected.output_ticks,
                    upgraded.output_ticks * expected.input_ticks
                );
            }
        }
    }
    for _ in 0..3 {
        simulation.advance()?;
        assert_eq!(simulation.snapshot.cash, opening.initial_cash);
    }
    let state = simulation.download()?;
    for (index, actor) in state.actors.iter().enumerate() {
        match actor.kind {
            kind if kind == Good::Flour as u32 => {
                assert_eq!(
                    state.values
                        [(USED_FIELDS + Good::Grain as usize - 1) * state.actors.len() + index]
                        * 20,
                    state.get(Field::Produced, index) * 25
                );
            }
            kind if kind == Good::Bread as u32 && state.get(Field::EnterpriseModel, index) == 0 => {
                assert_eq!(
                    state.values
                        [(USED_FIELDS + Good::Flour as usize - 1) * state.actors.len() + index]
                        * 64,
                    state.get(Field::Produced, index) * 25
                );
            }
            _ => (),
        }
    }
    simulation.checkpoint()?.restore()?;
    Ok(())
}

fn diet_scenario() -> Result<Scenario> {
    let mut config = Scenario::baseline(3, 42, Experiment::Concentrated)?
        .config()
        .clone();
    config.food = Some(FoodPolicy::default());
    config.prices = FoodPolicy::PRICES;
    config.distribution_rate = 0;
    config.entrepreneurship.interested_per_mille = 0;
    config.behavior.wage_increment = 0;
    config.demography.annual_births_per_million_fertile_women = 0;
    config.demography.people = (1..=3)
        .map(|household| PersonSeed {
            identity: 100 + household,
            household,
            age_days: 30 * DAYS_PER_YEAR,
            sex: Sex::Male,
        })
        .collect();
    config.population = Some(Population {
        households: (1..=3)
            .map(|identity| HouseholdSeed {
                identity,
                position: Position { x: 500, y: 500 },
                cash: 100_000,
                inventories: Good::food([0, 0, 100]),
                fatigue: 0,
            })
            .collect(),
        firms: [Good::Grain, Good::Flour, Good::Bread]
            .into_iter()
            .enumerate()
            .map(|(index, good)| FirmSeed {
                identity: 4 + index as u64,
                position: Position { x: 500, y: 500 },
                owner: 1,
                good,
                cash: 0,
                inventories: [0; Good::COUNT],
                capacity: 1,
                technique: Technique {
                    labor_minutes: 20,
                    input_ticks: u32::from(good != Good::Grain),
                    output_ticks: 1,
                },
                wage: 1,
                price: 1000,
            })
            .collect(),
    });
    Scenario::new(config)
}

fn salary_scenario() -> Result<Scenario> {
    let mut config = diet_scenario()?.config().clone();
    for (firm, wage) in config
        .population
        .as_mut()
        .unwrap()
        .firms
        .iter_mut()
        .zip([1999, 2000, 10_000])
    {
        firm.good = Good::Bread;
        firm.technique = Technique::new(1, 1, 1)?;
        firm.cash = 1_000_000;
        firm.inventories[Good::Flour as usize - 1] = 100;
        firm.wage = wage;
    }
    Scenario::new(config)
}

#[derive(Debug, PartialEq, Eq)]
struct SalaryConsumption {
    salary: u64,
    portions: u64,
}

#[test]
fn salary_tiers_consume_one_two_or_ten_daily_bread_units_on_gpu() -> Result<()> {
    let scenario = salary_scenario()?;
    let mut simulation = Simulation::new(scenario, support::device(64)?)?;
    simulation.advance()?;
    let state = simulation.download()?;
    let mut outcomes = (0..3)
        .map(|household| SalaryConsumption {
            salary: state.get(Field::GrossIncome, household),
            portions: state.get(Field::Consumed, household),
        })
        .collect::<Vec<_>>();
    outcomes.sort_by_key(|outcome| outcome.salary);
    assert_eq!(
        outcomes,
        [
            SalaryConsumption {
                salary: 1999,
                portions: 1
            },
            SalaryConsumption {
                salary: 2000,
                portions: 2
            },
            SalaryConsumption {
                salary: 10_000,
                portions: 10
            },
        ]
    );
    for household in 0..3 {
        assert_eq!(state.get(Field::NutritionNeed, household), 1);
    }
    simulation.checkpoint()?.restore()?;
    Ok(())
}

#[test]
fn missing_rich_consumption_target_preserves_a_supplied_minimum_ration_on_gpu() -> Result<()> {
    let mut config = salary_scenario()?.config().clone();
    for firm in &mut config.population.as_mut().unwrap().firms {
        firm.wage = 10_000;
    }
    let scenario = Scenario::new(config)?;
    let mut state = scenario.initial()?;
    state.day = 100;
    let removed = state.get(Field::Bread, 2) - 1;
    state.set(Field::Bread, 2, 1);
    state.set(Field::Bread, 0, state.get(Field::Bread, 0) + removed);
    let mut simulation =
        Simulation::from_checkpoint(Checkpoint::new(&scenario, &state)?, support::device(64)?)?;
    simulation.advance()?;
    let state = simulation.download()?;
    assert_eq!(state.get(Field::FoodNeed, 2), 10);
    assert_eq!(state.get(Field::Consumed, 2), 1);
    assert_eq!(state.get(Field::Shortage, 2), 9);
    assert_eq!(state.people[2].famine_exposure, 0);
    simulation.checkpoint()?.restore()?;
    Ok(())
}

#[test]
fn household_metric_processing_keeps_fractional_output_on_gpu() -> Result<()> {
    let mut config = diet_scenario()?.config().clone();
    config.household_production = HouseholdProduction::industrial();
    config.coverage = 6;
    let households = &mut config.population.as_mut().unwrap().households;
    let removed = households[0].inventories[Good::Bread as usize - 1];
    households[0].inventories[Good::Bread as usize - 1] = 0;
    households[1].inventories[Good::Bread as usize - 1] += removed;
    households[0].inventories[Good::Grain as usize - 1] = 5;
    households[0].inventories[Good::Tools as usize - 1] = 1;
    let scenario = Scenario::new(config)?;
    let mut state = scenario.initial()?;
    state.set(Field::LandPlots, 0, 1);
    let mut simulation =
        Simulation::from_checkpoint(Checkpoint::new(&scenario, &state)?, support::device(64)?)?;
    simulation.advance()?;
    let state = simulation.download()?;
    let count = state.actors.len();
    assert_eq!(
        state.values[(PRODUCTION_FIELDS + Good::Flour as usize - 1) * count],
        4
    );
    assert_eq!(
        state.values[(PRODUCTION_FIELDS + Good::Bread as usize - 1) * count],
        10
    );
    assert_eq!(state.get(Field::MillingRemainder, 0), 0);
    assert_eq!(state.get(Field::BakingRemainder, 0), 6);
    simulation.checkpoint()?.restore()?;
    Ok(())
}

#[test]
fn base_wheat_harvest_respects_kg_per_hectare_on_gpu() -> Result<()> {
    let mut config = Scenario::industrial(96, 42, Experiment::Concentrated)?
        .config()
        .clone();
    config.research = None;
    let scenario = Scenario::new(config)?;
    let mut state = scenario.initial()?;
    state.day = 270;
    let farm = state
        .actors
        .iter()
        .position(|actor| actor.kind == Good::Grain as u32)
        .unwrap();
    state.set(Field::FarmHectares, farm, 1);
    state.set(Field::CropWork, farm, 10_000);
    let mut simulation =
        Simulation::from_checkpoint(Checkpoint::new(&scenario, &state)?, support::device(64)?)?;
    simulation.advance()?;
    let mut state = simulation.download()?;
    assert_eq!(state.get(Field::Produced, farm), 1601);
    state.set(Field::CropWork, farm, 10_000);
    let mut simulation =
        Simulation::from_checkpoint(Checkpoint::new(&scenario, &state)?, support::device(64)?)?;
    simulation.advance()?;
    assert_eq!(simulation.download()?.get(Field::Produced, farm), 1601);
    simulation.checkpoint()?.restore()?;
    Ok(())
}

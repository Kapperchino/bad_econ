use super::support;
use anyhow::Result;
use bad_econ::{
    domain::{demography::*, enterprise::HouseholdProduction, food::FoodPolicy, *},
    simulation::Simulation,
};

fn diet_config() -> Result<ScenarioConfig> {
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
    Ok(config)
}

fn salary_scenario() -> Result<Scenario> {
    let mut config = diet_config()?;
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
fn only_the_mill_makes_flour_and_a_farmer_buys_it_to_bake_on_gpu() -> Result<()> {
    let device = support::device(64)?;
    for economy in [Economy::Agrarian, Economy::Industrial] {
        let mut config = diet_config()?;
        config.economy = economy;
        config.household_production = HouseholdProduction::industrial();
        config.behavior.price_response = 0;
        config.coverage = 8;
        let population = config.population.as_mut().unwrap();
        for household in &mut population.households {
            household.inventories = Good::food([100, 0, 0]);
        }
        population.firms[2].good = Good::Grain;
        population.firms[2].technique = Technique::new(20, 0, 1)?;
        let mill = &mut population.firms[1];
        mill.inventories = Good::food([5, 0, 0]);
        mill.technique = Technique::new(20, 5, 4)?;
        mill.cash = 10_000;
        mill.price = 2;
        let scenario = Scenario::new(config)?;
        let mut initial = scenario.initial()?;
        let mill = scenario.index_of(5).unwrap();
        initial.set(Field::LandPlots, 0, 1);
        initial.set(Field::MillingRemainder, 0, 4);
        initial.set(Field::Demand, mill, 4);
        initial.set(Field::Job, 1, mill as u64 + 1);
        initial.set(Field::JobDaysRemaining, 1, 5);
        let mut simulation =
            Simulation::from_checkpoint(Checkpoint::new(&scenario, &initial)?, device.clone())?;
        simulation.advance()?;
        let milled = simulation.download()?;
        let count = milled.actors.len();
        let flour_output = (PRODUCTION_FIELDS + Good::Flour as usize - 1) * count;
        let bread_output = (PRODUCTION_FIELDS + Good::Bread as usize - 1) * count;
        assert_eq!(milled.get(Field::Produced, mill), 4);
        assert_eq!(milled.get(Field::Grain, mill), 0);
        assert_eq!(milled.values[bread_output], 0);
        simulation.advance()?;
        let traded = simulation.download()?;
        let flour_transit = (TRANSIT_FIELDS + Good::Flour as usize - 1) * count;
        assert_eq!(traded.values[flour_transit], 4);
        assert_eq!(traded.get(Field::Sales, mill), 4);
        simulation.advance()?;
        let baked = simulation.download()?;
        assert_eq!(baked.values[bread_output], 10);
        assert_eq!(baked.get(Field::BakingRemainder, 0), 6);
        assert_eq!(baked.get(Field::MillingRemainder, 0), 4);
        for household in 0..3 {
            assert_eq!(baked.values[flour_output + household], 0);
            assert_eq!(baked.get(Field::Grain, household), 100);
        }
        simulation.checkpoint()?.restore()?;
    }
    Ok(())
}

#[test]
fn landless_households_buy_finished_bread_and_cannot_process_stored_ingredients_on_gpu()
-> Result<()> {
    let device = support::device(64)?;
    for economy in [Economy::Agrarian, Economy::Industrial] {
        let mut config = diet_config()?;
        config.economy = economy;
        config.household_production = HouseholdProduction::industrial();
        let population = config.population.as_mut().unwrap();
        for household in &mut population.households {
            household.inventories = [0; Good::COUNT];
            household.cash = 50;
        }
        population.households[1].inventories = Good::food([5, 5, 0]);
        population.households[2].cash = 1000;
        population.firms[0].inventories = Good::food([1000, 0, 0]);
        population.firms[0].price = 2;
        population.firms[1].inventories = Good::food([0, 1000, 0]);
        population.firms[1].price = 2;
        population.firms[2].inventories = Good::food([0, 0, 1000]);
        population.firms[2].price = 100;
        let scenario = Scenario::new(config)?;
        let mut simulation = Simulation::new(scenario, device.clone())?;
        simulation.advance()?;
        let ordered = simulation.download()?;
        let count = ordered.actors.len();
        for household in 0..3 {
            assert_eq!(ordered.get(Field::LandPlots, household), 0);
            for good in [Good::Grain, Good::Flour] {
                assert_eq!(
                    ordered.values[(TRANSIT_FIELDS + good as usize - 1) * count + household],
                    0
                );
            }
        }
        assert!(ordered.values[(TRANSIT_FIELDS + Good::Bread as usize - 1) * count + 2] > 0);
        simulation.advance()?;
        let state = simulation.download()?;
        assert_eq!(state.get(Field::Grain, 1), 5);
        assert_eq!(state.get(Field::Flour, 1), 5);
        assert_eq!(state.get(Field::GrainReserve, 1), 0);
        for good in [Good::Flour, Good::Bread] {
            assert_eq!(
                state.values[(PRODUCTION_FIELDS + good as usize - 1) * count + 1],
                0
            );
        }
        assert_eq!(state.get(Field::Consumed, 1), 0);
        assert!(state.get(Field::Consumed, 2) >= state.get(Field::NutritionNeed, 2));
    }
    Ok(())
}

#[test]
fn farms_plant_to_fund_meals_and_reserve_only_seed_on_gpu() -> Result<()> {
    let mut config = diet_config()?;
    config.food = Some(FoodPolicy::industrial());
    config.prices[2] = 100;
    config.household_production = HouseholdProduction::industrial();
    let population = config.population.as_mut().unwrap();
    for (household, grain) in population.households.iter_mut().zip([92, 375]) {
        household.inventories[Good::Grain as usize - 1] = grain;
        household.inventories[Good::Tools as usize - 1] = 1;
    }
    let scenario = Scenario::new(config)?;
    let mut initial = scenario.initial()?;
    initial.day = 168;
    initial.set(Field::LandPlots, 0, 1);
    initial.set(Field::LandPlots, 1, 1);
    let mut simulation =
        Simulation::from_checkpoint(Checkpoint::new(&scenario, &initial)?, support::device(64)?)?;
    simulation.advance()?;
    let state = simulation.download()?;
    assert!(state.get(Field::FoodNeed, 0) > state.get(Field::NutritionNeed, 0));
    assert_eq!(state.get(Field::Consumed, 0), state.get(Field::FoodNeed, 0));
    assert_eq!(state.get(Field::GrainReserve, 0), 91);
    assert_eq!(state.get(Field::Consumed, 1), state.get(Field::FoodNeed, 1));
    assert!(state.get(Field::CropWork, 1) > 0);
    Ok(())
}

#[test]
fn households_with_bread_meet_the_calibrated_minimum_even_at_zero_income_on_gpu() -> Result<()> {
    let mut config = diet_config()?;
    config.food = Some(FoodPolicy::industrial());
    config.demography.people[0].age_days = DAYS_PER_YEAR;
    config.population.as_mut().unwrap().households[0].cash = 0;
    let mut simulation = Simulation::new(Scenario::new(config)?, support::device(64)?)?;
    simulation.advance()?;
    let state = simulation.download()?;
    assert_eq!(state.get(Field::NutritionNeed, 0), 1);
    assert_eq!(state.get(Field::Consumed, 0), 1);
    Ok(())
}

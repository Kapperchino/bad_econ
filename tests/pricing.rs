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

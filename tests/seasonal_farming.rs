#![cfg(feature = "compute")]

use super::support;
use anyhow::Result;
use bad_econ::{
    domain::{enterprise::HouseholdProduction, *},
    simulation::Simulation,
};

const CROPS: [Good; 4] = [Good::Grain, Good::Flax, Good::Cotton, Good::Dye];

fn farming_scenario(home: HouseholdProduction) -> Result<Scenario> {
    let mut config = Scenario::baseline(12, 42, Experiment::Concentrated)?
        .config()
        .clone();
    config.firms = 5;
    config.household_production = home;
    config.entrepreneurship.interested_per_mille = 0;
    config.distribution_rate = 0;
    config.forecast_weight = 0;
    config.behavior.price_response = 0;
    config.demography.annual_births_per_million_fertile_women = 0;
    config.demography.mortality.background_annual_ppm = [0; 5];
    config.demography.mortality.occupational_annual_ppm = [0; Good::COUNT];
    config.resources = ResourceMap::Explicit {
        regions: CROPS
            .into_iter()
            .chain([Good::Wood])
            .map(|good| {
                ResourceRegion::new(
                    good,
                    Position { x: 0, y: 0 },
                    Position { x: 1000, y: 1000 },
                    1000,
                )
            })
            .collect::<Result<Vec<_>>>()?,
    };
    config.population = Some(Population {
        households: (1..=12)
            .map(|identity| {
                let mut inventories = Good::food([0, 0, 100_000]);
                inventories[Good::Wood as usize - 1] = 100_000;
                inventories[Good::Tools as usize - 1] = 1;
                HouseholdSeed {
                    identity,
                    position: Position { x: 500, y: 500 },
                    cash: 10_000,
                    inventories,
                    fatigue: 0,
                }
            })
            .collect(),
        firms: CROPS
            .into_iter()
            .chain([Good::Wood])
            .enumerate()
            .map(|(index, good)| FirmSeed {
                identity: 13 + index as u64,
                position: Position { x: 500, y: 500 },
                owner: 1,
                good,
                cash: 1_000_000,
                inventories: [0; Good::COUNT],
                capacity: 1,
                technique: Technique {
                    labor_minutes: 60,
                    input_ticks: 0,
                    output_ticks: 10,
                },
                wage: 1,
                price: 10,
            })
            .collect(),
    });
    Scenario::new(config)
}

#[test]
fn crop_firms_accumulate_growing_work_and_release_only_at_harvest_on_gpu() -> Result<()> {
    let scenario = farming_scenario(HouseholdProduction::Unavailable)?;
    let mut initial = scenario.initial()?;
    initial.day = 269;
    for farm in 12..16 {
        initial.set(Field::CropWork, farm, 2110);
    }
    let checkpoint = Checkpoint::new(&scenario, &initial)?;
    let mut simulation = Simulation::from_checkpoint(checkpoint, support::device(64)?)?;
    simulation.advance()?;
    let growing = simulation.download()?;
    for farm in 12..16 {
        assert_eq!(growing.get(Field::Produced, farm), 0);
        assert_eq!(growing.get(Field::CropWork, farm), 2120);
    }
    assert_eq!(growing.get(Field::Produced, 16), 10);
    simulation.advance()?;
    let harvested = simulation.download()?;
    for (index, good) in CROPS.into_iter().enumerate() {
        let farm = 12 + index;
        let expected = growing.get(Field::CropWork, farm) * 365 / 211;
        assert_eq!(harvested.get(Field::Produced, farm), expected);
        assert_eq!(harvested.get(Field::inventory(good), farm), expected);
        assert_eq!(harvested.get(Field::CropWork, farm), 0);
    }
    simulation.advance()?;
    let following = simulation.download()?;
    for farm in 12..16 {
        assert_eq!(
            following.get(Field::Produced, farm),
            harvested.get(Field::Produced, farm)
        );
    }
    simulation.checkpoint()?.restore()?;
    Ok(())
}

#[test]
fn winter_farms_preserve_crops_without_hiring_or_producing_on_gpu() -> Result<()> {
    let scenario = farming_scenario(HouseholdProduction::Unavailable)?;
    let mut initial = scenario.initial()?;
    for farm in 12..16 {
        initial.set(Field::CropWork, farm, 2110);
    }
    let mut simulation =
        Simulation::from_checkpoint(Checkpoint::new(&scenario, &initial)?, support::device(64)?)?;
    simulation.advance()?;
    let winter = simulation.download()?;
    for farm in 12..16 {
        assert_eq!(winter.get(Field::Plan, farm), 0);
        assert_eq!(winter.get(Field::Produced, farm), 0);
        assert_eq!(winter.get(Field::CropWork, farm), 2110);
    }
    assert_eq!(winter.get(Field::Produced, 16), 10);
    Ok(())
}

#[test]
fn cashless_farms_keep_unharvested_crops_until_the_season_changes_on_gpu() -> Result<()> {
    let scenario = farming_scenario(HouseholdProduction::industrial())?;
    let mut initial = scenario.initial()?;
    initial.day = 269;
    let mut owner_cash = initial.get(Field::Cash, 0);
    for farm in 12..16 {
        owner_cash += initial.get(Field::Cash, farm);
        initial.set(Field::Cash, farm, 0);
        initial.set(Field::IdleDays, farm, 29);
        initial.set(Field::CropWork, farm, 2110);
    }
    initial.set(Field::Cash, 0, owner_cash);
    let mut simulation =
        Simulation::from_checkpoint(Checkpoint::new(&scenario, &initial)?, support::device(64)?)?;
    simulation.advance()?;
    let waiting = simulation.download()?;
    assert_eq!(waiting.actors.len(), initial.actors.len());
    for farm in 12..16 {
        assert_eq!(waiting.get(Field::CropWork, farm), 2110);
        assert_eq!(waiting.get(Field::IdleDays, farm), 0);
        assert_eq!(
            waiting.get(Field::EnterpriseStatus, farm),
            enterprise::EnterpriseState::Dormant as u64
        );
    }
    simulation.advance()?;
    let harvested = simulation.download()?;
    for farm in 12..16 {
        assert_eq!(harvested.get(Field::Produced, farm), 3650);
        assert_eq!(harvested.get(Field::CropWork, farm), 0);
    }
    Ok(())
}

#[test]
fn wrapped_household_harvest_runs_once_across_new_year_on_gpu() -> Result<()> {
    let scenario = farming_scenario(HouseholdProduction::Available {
        milling_minutes: 30,
        baking_minutes: 30,
        cultivation_minutes: 10,
        harvest_start: 350,
        harvest_days: 30,
        annual_yield_per_plot: 730,
    })?;
    let mut initial = scenario.initial()?;
    initial.day = 364;
    initial.set(Field::LandPlots, 0, 1);
    initial.set(Field::CropWork, 0, 100);
    let mut simulation =
        Simulation::from_checkpoint(Checkpoint::new(&scenario, &initial)?, support::device(64)?)?;
    simulation.advance()?;
    let mut harvested = simulation.download()?;
    assert_eq!(harvested.get(Field::Grain, 0), 10);
    harvested.set(Field::CropWork, 0, 100);
    let mut simulation = Simulation::from_checkpoint(
        Checkpoint::new(&scenario, &harvested)?,
        support::device(64)?,
    )?;
    simulation.advance()?;
    let mut january = simulation.download()?;
    assert_eq!(january.get(Field::Grain, 0), 10);
    assert_eq!(january.get(Field::CropWork, 0), 100);
    january.day = 715;
    let mut simulation =
        Simulation::from_checkpoint(Checkpoint::new(&scenario, &january)?, support::device(64)?)?;
    simulation.advance()?;
    let following = simulation.download()?;
    assert_eq!(following.get(Field::Grain, 0), 20);
    assert_eq!(following.get(Field::CropWork, 0), 0);
    Ok(())
}

#[test]
fn generated_crop_inventories_bridge_the_first_harvest_on_gpu() -> Result<()> {
    let scenario = Scenario::industrial(96, 42, Experiment::Baseline)?;
    let mut simulation = Simulation::new(scenario.clone(), support::device(64)?)?;
    let initial = simulation.download()?;
    for (index, actor) in initial.actors.iter().enumerate() {
        let crop = CROPS.iter().any(|good| actor.kind == *good as u32);
        if crop {
            let good = Good::from_index(actor.kind)?;
            assert!(initial.get(Field::inventory(good), index) >= 365);
        }
    }
    simulation.advance()?;
    assert_eq!(simulation.snapshot.cash, initial.initial_cash);
    simulation.checkpoint()?.restore()?;
    Ok(())
}

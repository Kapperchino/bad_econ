#![cfg(feature = "compute")]
use super::support;
use anyhow::Result;
use bad_econ::{
    domain::{
        enterprise::{EnterpriseState, HouseholdProduction, LandAccess, LandGrant},
        property::{
            ConstructionOrder, Development, ProjectStatus, PropertyConfig, PropertyRegion,
            PropertySnapshot,
        },
        *,
    },
    simulation::Simulation,
};

fn development() -> Result<ScenarioConfig> {
    let mut config = Scenario::baseline(8, 42, Experiment::Concentrated)?
        .config()
        .clone();
    config.economy = Economy::Industrial;
    config.firms = 4;
    config.household_production = HouseholdProduction::industrial();
    config.entrepreneurship.interested_per_mille = 0;
    config.coverage = 2;
    config.forecast_weight = 0;
    config.distribution_rate = 0;
    config.behavior = BehaviorPolicy {
        work_fatigue: 0,
        rest_recovery: 0,
        wage_increment: 0,
        price_response: 0,
    };
    config.demography.annual_births_per_million_fertile_women = 0;
    config.demography.mortality.background_annual_ppm = [0; 5];
    config.demography.mortality.occupational_annual_ppm = [0; Good::COUNT];
    config.resources = ResourceMap::Explicit {
        regions: [Good::Grain, Good::Wood]
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
        households: (1..=8)
            .map(|identity| HouseholdSeed {
                identity,
                position: Position {
                    x: match identity {
                        7..=8 => 750,
                        _ => 250,
                    },
                    y: 250,
                },
                cash: 100_000,
                inventories: Good::food([100, 0, 100]),
                fatigue: 0,
            })
            .collect(),
        firms: [Good::Construction, Good::Grain, Good::Furniture, Good::Wood]
            .into_iter()
            .enumerate()
            .map(|(index, good)| {
                let mut inventories = [0; Good::COUNT];
                inventories[Good::Tools as usize - 1] = 40;
                inventories[Good::Wood as usize - 1] = 40;
                FirmSeed {
                    identity: 101 + index as u64,
                    position: Position { x: 250, y: 250 },
                    owner: match index {
                        0 => 5,
                        _ => 1,
                    },
                    good,
                    cash: 100_000,
                    inventories,
                    capacity: 2,
                    technique: Technique {
                        labor_minutes: 60,
                        input_ticks: u32::from(good.inputs()[0].is_some()),
                        output_ticks: 2,
                    },
                    wage: 1,
                    price: 100,
                }
            })
            .collect(),
    });
    config.land_access = Some(LandAccess {
        total_plots: 9,
        allocations: [2, 3, 4, 7]
            .into_iter()
            .map(|household| LandGrant {
                household,
                plots: match household {
                    7 => 3,
                    _ => 2,
                },
            })
            .collect(),
    });
    config.property = Some(PropertyConfig {
        regions: vec![
            PropertyRegion {
                lower: Position { x: 0, y: 0 },
                upper: Position { x: 500, y: 1000 },
                plots: 14,
                landowner: 1,
            },
            PropertyRegion {
                lower: Position { x: 500, y: 0 },
                upper: Position { x: 1000, y: 1000 },
                plots: 5,
                landowner: 7,
            },
        ],
        buildings: vec![],
        orders: [1, 102, 103]
            .into_iter()
            .map(|site| ConstructionOrder {
                landowner: 1,
                site,
                plots: match site {
                    1 => 1,
                    _ => 2,
                },
                work: 6,
                development: match site {
                    1 => Development::Extension,
                    _ => Development::NewBuilding,
                },
            })
            .collect(),
    });
    Ok(config)
}

#[test]
fn contractors_build_housing_farms_and_factories_and_displace_only_local_farmers() -> Result<()> {
    let scenario = Scenario::new(development()?)?;
    let mut initial = scenario.initial()?;
    initial.day = 257;
    let mut simulation =
        Simulation::from_checkpoint(Checkpoint::new(&scenario, &initial)?, support::device(64)?)?;
    simulation.advance()?;
    let first = simulation.download()?;
    for farmer in [1, 2, 3] {
        assert_eq!(first.get(Field::LandPlots, farmer), 0);
        assert_eq!(first.get(Field::DisplacedPlots, farmer), 2);
        assert_eq!(first.get(Field::CropWork, farmer), 0);
        assert_eq!(first.get(Field::Grain, farmer), 0);
        assert_eq!(
            first.values[(USED_FIELDS + Good::Grain as usize - 1) * first.actors.len() + farmer],
            initial.get(Field::Grain, farmer)
        );
    }
    assert_eq!(first.get(Field::LandPlots, 6), 3);
    for household in [0, 4, 5, 6, 7] {
        assert_eq!(
            first.get(Field::Grain, household),
            initial.get(Field::Grain, household)
        );
    }
    assert!(simulation.snapshot.produced[Good::Construction as usize - 1] > 0);
    assert!(simulation.snapshot.used[Good::Wood as usize - 1] > 0);
    assert!(simulation.snapshot.used[Good::Tools as usize - 1] > 0);
    assert_eq!(first.get(Field::Produced, 9), 0);
    assert_eq!(first.get(Field::Produced, 10), 0);
    assert_eq!(first.get(Field::OccupiedPlots, 9), 0);
    let mut replay = Simulation::from_checkpoint(simulation.checkpoint()?, support::device(64)?)?;
    for _ in 0..12 {
        simulation.advance()?;
        replay.advance()?;
        assert_eq!(simulation.download()?, replay.download()?);
    }
    let completed = simulation.download()?;
    for site in [0, 9, 10] {
        assert_eq!(
            completed.get(Field::ProjectStatus, site),
            ProjectStatus::Completed as u64
        );
        assert_eq!(completed.get(Field::ConstructionRemaining, site), 0);
        assert_eq!(completed.get(Field::OccupiedPlots, site), 2);
    }
    assert_eq!(
        completed.get(Field::EnterpriseStatus, 9),
        EnterpriseState::Operating as u64
    );
    assert_eq!(completed.get(Field::Produced, 9), 0);
    assert!(completed.get(Field::CropWork, 9) > 0);
    assert!(completed.get(Field::Produced, 10) > 0);
    simulation.advance()?;
    replay.advance()?;
    let harvested = simulation.download()?;
    assert_eq!(harvested, replay.download()?);
    assert!(harvested.get(Field::Produced, 9) > 0);
    let land = PropertySnapshot::from_state(&scenario, &harvested)?.unwrap();
    assert_eq!(land.displaced_plots, 6);
    assert_eq!(land.regions[0].housing, 7);
    assert_eq!(land.regions[0].farms, 3);
    assert_eq!(land.regions[0].factories, 3);
    assert_eq!(land.regions[0].free, 1);
    assert_eq!(land.regions[1].subsistence, 3);
    assert_eq!(simulation.snapshot.cash, initial.initial_cash);
    assert_eq!(
        simulation.snapshot,
        ObservationSnapshot::from_state(&scenario, &harvested)?
    );
    simulation.checkpoint()?.restore()?;
    Ok(())
}

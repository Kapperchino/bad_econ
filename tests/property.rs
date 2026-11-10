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
    simulation::{Simulation, Trace},
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

fn commercial_expansion() -> Result<Scenario> {
    let mut config = development()?;
    config.property = None;
    config.search_radius = 200;
    config.expansion = enterprise::ExpansionPolicy {
        small_business_per_mille: 1000,
        industrial_per_mille: 1000,
    };
    for (index, firm) in config
        .population
        .as_mut()
        .unwrap()
        .firms
        .iter_mut()
        .enumerate()
    {
        firm.owner = match index {
            1 => 1,
            _ => 5 + index as u64,
        };
        firm.capacity = 1;
        firm.technique.output_ticks = 10;
    }
    Scenario::new(config)
}

#[test]
#[ignore = "extended scenario coverage; run cargo gpu-extended"]
fn completed_commercial_expansion_consolidates_local_land_and_releases_workers_on_gpu() -> Result<()>
{
    let scenario = commercial_expansion()?;
    let mut initial = scenario.initial()?;
    initial.set(Field::Demand, 9, 100);
    initial.set(Field::CropWork, 1, 100);
    let mut simulation =
        Simulation::from_checkpoint(Checkpoint::new(&scenario, &initial)?, support::device(64)?)?;
    simulation.advance()?;
    let building = simulation.download()?;
    assert_eq!(building.get(Field::PendingCapacity, 9), 1);
    assert!(building.get(Field::ConstructionRemaining, 9) > 0);
    assert_eq!(building.get(Field::LandPlots, 1), 2);
    assert_eq!(building.get(Field::DisplacedPlots, 1), 0);
    let mut replay = Simulation::from_checkpoint(simulation.checkpoint()?, support::device(64)?)?;
    let mut employed = 0;
    for _ in 0..8 {
        simulation.advance()?;
        replay.advance()?;
        let state = simulation.download()?;
        assert_eq!(state, replay.download()?);
        employed += state.get(Field::Employed, 1);
        let plots = state
            .actors
            .iter()
            .enumerate()
            .filter(|(_, actor)| actor.kind == 0)
            .map(|(farmer, _)| state.get(Field::LandPlots, farmer))
            .sum::<u64>();
        assert_eq!(plots + state.get(Field::OccupiedPlots, 9), 9);
        assert_eq!(state.get(Field::LandPlots, 6), 3);
        assert_eq!(state.get(Field::DisplacedPlots, 6), 0);
        assert_eq!(simulation.snapshot.cash, initial.initial_cash);
    }
    let state = simulation.download()?;
    assert_eq!(state.get(Field::LandPlots, 1), 0);
    assert_eq!(state.get(Field::DisplacedPlots, 1), 2);
    assert_eq!(state.get(Field::CropWork, 1), 0);
    assert_eq!(state.get(Field::Grain, 1), initial.get(Field::Grain, 1));
    assert_eq!(state.get(Field::OccupiedPlots, 9), 6);
    assert!(state.get(Field::Produced, 9) > state.get(Field::UnscaledProduced, 9));
    assert!(state.actors.at(9).capacity > initial.actors.at(9).capacity);
    assert!(employed > 0);
    assert_eq!(
        simulation.snapshot,
        ObservationSnapshot::from_state(&scenario, &state)?
    );
    simulation.checkpoint()?.restore()?;
    Ok(())
}

#[test]
#[ignore = "extended scenario coverage; run cargo gpu-extended"]
fn unfunded_farms_cannot_displace_subsistence_households_on_gpu() -> Result<()> {
    let scenario = commercial_expansion()?;
    let mut initial = scenario.initial()?;
    let cash = initial.get(Field::Cash, 9);
    initial.set(Field::Cash, 9, 0);
    initial.set(Field::Cash, 7, initial.get(Field::Cash, 7) + cash);
    initial.set(Field::Demand, 9, 100);
    let mut simulation =
        Simulation::from_checkpoint(Checkpoint::new(&scenario, &initial)?, support::device(64)?)?;
    for _ in 0..5 {
        simulation.advance()?;
    }
    let state = simulation.download()?;
    assert_eq!(state.get(Field::OccupiedPlots, 9), 0);
    for farmer in [1, 2, 3, 6] {
        assert_eq!(
            state.get(Field::LandPlots, farmer),
            initial.get(Field::LandPlots, farmer)
        );
        assert_eq!(state.get(Field::DisplacedPlots, farmer), 0);
    }
    simulation.checkpoint()?.restore()?;
    Ok(())
}

#[test]
#[ignore = "extended scenario coverage; run cargo gpu-extended"]
fn farm_expansion_without_a_productivity_advantage_leaves_subsistence_land_intact_on_gpu()
-> Result<()> {
    for technique in [Technique::new(10, 0, 2)?, Technique::new(240, 0, 10)?] {
        let mut config = commercial_expansion()?.config().clone();
        config.population.as_mut().unwrap().firms[1].technique = technique;
        let scenario = Scenario::new(config)?;
        let mut initial = scenario.initial()?;
        initial.set(Field::Demand, 9, 100);
        let mut simulation = Simulation::from_checkpoint(
            Checkpoint::new(&scenario, &initial)?,
            support::device(64)?,
        )?;
        for _ in 0..9 {
            simulation.advance()?;
        }
        let state = simulation.download()?;
        assert!(state.actors.at(9).capacity > initial.actors.at(9).capacity);
        assert_eq!(state.get(Field::OccupiedPlots, 9), 0);
        for farmer in [1, 2, 3, 6] {
            assert_eq!(
                state.get(Field::LandPlots, farmer),
                initial.get(Field::LandPlots, farmer)
            );
            assert_eq!(state.get(Field::DisplacedPlots, farmer), 0);
        }
        simulation.checkpoint()?.restore()?;
    }
    Ok(())
}

#[test]
fn contractors_build_housing_farms_and_factories_and_displace_only_local_farmers() -> Result<()> {
    let scenario = Scenario::new(development()?)?;
    let initial = scenario.initial()?;
    let mut simulation = Simulation::new(scenario.clone(), support::device(64)?)?;
    simulation.advance()?;
    let first = simulation.download()?;
    for farmer in [1, 2, 3] {
        assert_eq!(first.get(Field::LandPlots, farmer), 0);
        assert_eq!(first.get(Field::DisplacedPlots, farmer), 2);
        assert_eq!(first.get(Field::CropWork, farmer), 0);
    }
    assert_eq!(first.get(Field::LandPlots, 6), 3);
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
    assert!(completed.get(Field::Produced, 9) > 0);
    assert!(completed.get(Field::Produced, 10) > 0);
    let land = PropertySnapshot::from_state(&scenario, &completed)?.unwrap();
    assert_eq!(land.displaced_plots, 6);
    assert_eq!(land.regions[0].housing, 7);
    assert_eq!(land.regions[0].farms, 3);
    assert_eq!(land.regions[0].factories, 3);
    assert_eq!(land.regions[0].free, 1);
    assert_eq!(land.regions[1].subsistence, 3);
    assert_eq!(simulation.snapshot.cash, initial.initial_cash);
    assert_eq!(
        simulation.snapshot,
        ObservationSnapshot::from_state(&scenario, &completed)?
    );
    simulation.checkpoint()?.restore()?;
    Ok(())
}

#[test]
#[ignore = "extended scenario coverage; run cargo gpu-extended"]
fn scarce_space_reserves_projects_in_order_without_overallocating_the_region() -> Result<()> {
    let mut config = development()?;
    config.property.as_mut().unwrap().regions[0].plots = 11;
    let scenario = Scenario::new(config)?;
    let mut simulation = Simulation::new(scenario.clone(), support::device(64)?)?;
    simulation.phase(Phase::Opening, Trace::Discard)?;
    let state = simulation.download()?;
    assert_eq!(
        state.get(Field::ProjectStatus, 0),
        ProjectStatus::Building as u64
    );
    assert_eq!(
        state.get(Field::ProjectStatus, 9),
        ProjectStatus::Building as u64
    );
    assert_eq!(
        state.get(Field::ProjectStatus, 10),
        ProjectStatus::Requested as u64
    );
    assert_eq!(state.get(Field::ReservedPlots, 10), 0);
    let allocation = PropertySnapshot::from_state(&scenario, &state)?.unwrap();
    assert_eq!(allocation.regions[0].reserved, 3);
    assert_eq!(allocation.regions[0].free, 0);
    assert_eq!(allocation.regions[1].subsistence, 3);
    Ok(())
}

#[test]
#[ignore = "extended scenario coverage; run cargo gpu-extended"]
fn construction_stalls_without_materials_and_cannot_be_bypassed_by_owner_labor() -> Result<()> {
    let mut config = development()?;
    for firm in &mut config.population.as_mut().unwrap().firms {
        firm.inventories[Good::Tools as usize - 1] = 0;
    }
    let scenario = Scenario::new(config)?;
    let mut simulation = Simulation::new(scenario.clone(), support::device(64)?)?;
    for _ in 0..4 {
        simulation.advance()?;
    }
    let state = simulation.download()?;
    assert_eq!(
        simulation.snapshot.produced[Good::Construction as usize - 1],
        0
    );
    for site in [0, 9, 10] {
        assert_eq!(state.get(Field::ConstructionRemaining, site), 6);
        assert_eq!(
            state.get(Field::ProjectStatus, site),
            ProjectStatus::Building as u64
        );
    }
    assert_eq!(state.get(Field::Produced, 9), 0);
    assert_eq!(state.get(Field::Produced, 10), 0);
    simulation.checkpoint()?.restore()?;
    Ok(())
}

#[test]
fn invalid_regions_ownership_and_initial_overallocation_are_rejected_before_gpu_execution()
-> Result<()> {
    let mut config = development()?;
    config.property.as_mut().unwrap().regions[1].lower.x = 499;
    assert!(Scenario::new(config).is_err());
    let mut config = development()?;
    config.property.as_mut().unwrap().orders[0].landowner = 2;
    assert!(Simulation::new(Scenario::new(config)?, support::device(64)?).is_err());
    let mut config = development()?;
    config.property.as_mut().unwrap().regions[0].plots = 7;
    assert!(Simulation::new(Scenario::new(config)?, support::device(64)?).is_err());
    Ok(())
}

#[test]
#[ignore = "extended scenario coverage; run cargo gpu-extended"]
fn completed_extensions_add_capacity_and_keep_existing_production_operating() -> Result<()> {
    let mut config = development()?;
    config.property.as_mut().unwrap().orders = vec![ConstructionOrder {
        landowner: 1,
        site: 102,
        plots: 2,
        work: 6,
        development: Development::Extension,
    }];
    let scenario = Scenario::new(config)?;
    let mut simulation = Simulation::new(scenario.clone(), support::device(64)?)?;
    simulation.advance()?;
    assert!(simulation.download()?.get(Field::Produced, 9) > 0);
    assert_eq!(simulation.download()?.actors.at(9).capacity, 2);
    for _ in 0..7 {
        simulation.advance()?;
    }
    let state = simulation.download()?;
    assert_eq!(
        state.actors.at(9).capacity,
        6,
        "project {:?}; builder {:?}",
        [
            Field::ConstructionRemaining,
            Field::Construction,
            Field::ProjectStatus,
            Field::ReservedPlots,
            Field::PendingCapacity
        ]
        .map(|field| state.get(field, 9)),
        [
            Field::Produced,
            Field::Construction,
            Field::Sales,
            Field::Demand,
            Field::Slots
        ]
        .map(|field| state.get(field, 8))
    );
    assert_eq!(state.get(Field::OccupiedPlots, 9), 3);
    assert_eq!(state.get(Field::PendingCapacity, 9), 0);
    PropertySnapshot::from_state(&scenario, &state)?;
    simulation.checkpoint()?.restore()?;
    Ok(())
}

#[test]
#[ignore = "extended scenario coverage; run cargo gpu-extended"]
fn construction_methods_change_labor_and_output_on_gpu() -> Result<()> {
    use bad_econ::domain::research::*;
    let mut config = development()?;
    let technology = Technology::Mechanization as u32;
    config.research = Some(ResearchConfig {
        labs: vec![ResearchLab {
            owner: 101,
            researcher: 5,
            discovered: technology,
            mastered: technology,
            permitted: technology,
            adopted: technology,
            consent: ResearchConsent::Granted,
            production_requirements: 0,
            adoption: AdoptionPolicy::Manual,
            methods: vec![ProductionMethod::MechanizedConstruction],
            projects: vec![],
        }],
        ..Default::default()
    });
    let scenario = Scenario::new(config)?;
    let mut simulation = Simulation::new(scenario.clone(), support::device(64)?)?;
    simulation.advance()?;
    let state = simulation.download()?;
    assert_eq!(
        state.get(Field::ActiveMethod, 8),
        ProductionMethod::MechanizedConstruction as u64
    );
    assert_eq!(state.get(Field::LaborMinutes, 8), 40);
    assert_eq!(state.get(Field::OutputTicks, 8), 4);
    assert!(state.get(Field::Produced, 8) > 0);
    assert_eq!(
        simulation.snapshot,
        ObservationSnapshot::from_state(&scenario, &state)?
    );
    simulation.checkpoint()?.restore()?;
    Ok(())
}

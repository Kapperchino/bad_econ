use anyhow::Result;
use bad_econ::{arithmetic::DrawKey, domain::*};

#[test]
fn classification_preserves_good_ids_and_inventory_order() -> Result<()> {
    let natural = [
        Good::Grain,
        Good::Iron,
        Good::Coal,
        Good::Flax,
        Good::Wood,
        Good::Cotton,
        Good::Dye,
    ];
    for (index, good) in Good::ALL.into_iter().enumerate() {
        assert_eq!(good as usize, index + 1);
        assert_eq!(Good::from_index(index as u32 + 1)?, good);
        assert_eq!(Field::inventory(good) as u32, good as u32);
        assert_eq!(good.kind() == GoodKind::Natural, natural.contains(&good));
    }
    Ok(())
}

#[test]
fn resource_rectangles_use_half_open_bounds_and_combine_overlaps() -> Result<()> {
    let mut config = Scenario::baseline(3, 1, Experiment::Baseline)?
        .config()
        .clone();
    config.resources = ResourceMap::Explicit {
        regions: vec![
            ResourceRegion::new(
                Good::Grain,
                Position { x: 2, y: 3 },
                Position { x: 5, y: 7 },
                config.map_size,
            )?,
            ResourceRegion::new(
                Good::Grain,
                Position { x: 4, y: 6 },
                Position { x: 8, y: 9 },
                config.map_size,
            )?,
            ResourceRegion::new(
                Good::Iron,
                Position { x: 3, y: 4 },
                Position { x: 6, y: 8 },
                config.map_size,
            )?,
        ],
    };
    let scenario = Scenario::new(config)?;
    for position in [
        Position { x: 2, y: 3 },
        Position { x: 4, y: 6 },
        Position { x: 7, y: 8 },
    ] {
        assert!(scenario.has_resource(Good::Grain, position));
        assert_eq!(
            scenario.production_site(Good::Grain, position),
            ProductionSite::ResourceAvailable
        );
    }
    for position in [
        Position { x: 1, y: 3 },
        Position { x: 2, y: 2 },
        Position { x: 5, y: 3 },
        Position { x: 2, y: 7 },
        Position { x: 8, y: 8 },
        Position { x: 7, y: 9 },
    ] {
        assert!(!scenario.has_resource(Good::Grain, position));
        assert_eq!(
            scenario.production_site(Good::Grain, position),
            ProductionSite::ResourceMissing
        );
    }
    let overlap = Position { x: 4, y: 6 };
    assert!(scenario.has_resource(Good::Grain, overlap));
    assert!(scenario.has_resource(Good::Iron, overlap));
    assert!(!scenario.has_resource(Good::Coal, overlap));
    assert_eq!(
        scenario.production_site(Good::Steel, overlap),
        ProductionSite::UnrestrictedManufacturing
    );
    Ok(())
}

#[test]
fn invalid_regions_and_missing_generated_producer_resources_are_rejected() -> Result<()> {
    let origin = Position { x: 0, y: 0 };
    let edge = Position { x: 10, y: 10 };
    assert!(ResourceRegion::new(Good::Steel, origin, edge, 10).is_err());
    for upper in [
        origin,
        Position { x: 0, y: 10 },
        Position { x: 10, y: 0 },
        Position { x: 11, y: 10 },
        Position { x: 10, y: 11 },
    ] {
        assert!(ResourceRegion::new(Good::Grain, origin, upper, 10).is_err());
    }
    assert!(ResourceRegion::new(Good::Grain, edge, origin, 10).is_err());
    let mut config = Scenario::baseline(3, 1, Experiment::Baseline)?
        .config()
        .clone();
    config.resources = ResourceMap::Explicit { regions: vec![] };
    assert!(
        Scenario::new(config.clone())
            .unwrap_err()
            .to_string()
            .contains("Grain producer has no matching resource region")
    );
    config.resources = ResourceMap::Explicit {
        regions: vec![ResourceRegion::new(
            Good::Grain,
            origin,
            edge,
            config.map_size,
        )?],
    };
    let encoded = serde_json::to_value(&config)?;
    for invalid in [
        serde_json::json!({"good":"Steel", "lower":{"x":0,"y":0}, "upper":{"x":10,"y":10}}),
        serde_json::json!({"good":"Grain", "lower":{"x":10,"y":0}, "upper":{"x":10,"y":10}}),
        serde_json::json!({"good":"Grain", "lower":{"x":0,"y":0}, "upper":{"x":1001,"y":10}}),
    ] {
        let mut invalid_config = encoded.clone();
        invalid_config["resources"]["Explicit"]["regions"][0] = invalid;
        assert!(Scenario::new(serde_json::from_value(invalid_config)?).is_err());
    }
    config.economy = Economy::Industrial;
    config.firms = 16;
    assert!(
        Scenario::new(config)
            .unwrap_err()
            .to_string()
            .contains("Iron producer has no matching resource region")
    );
    Ok(())
}

#[test]
fn seeded_geography_and_placement_are_deterministic_even_on_tiny_maps() -> Result<()> {
    for map_size in [1, 2, 3, 4, 7, 1000] {
        for experiment in [Experiment::Baseline, Experiment::Concentrated] {
            let mut config = Scenario::baseline(12, 42, experiment)?.config().clone();
            config.economy = Economy::Industrial;
            config.firms = 64;
            config.map_size = map_size;
            config.search_radius = map_size;
            let scenario = Scenario::new(config.clone())?;
            assert_eq!(scenario, Scenario::new(config.clone())?);
            assert_eq!(scenario.resource_regions().len(), 28);
            let side = (map_size / 4).max(1);
            for good in Good::ALL
                .into_iter()
                .filter(|good| good.kind() == GoodKind::Natural)
            {
                assert_eq!(
                    scenario
                        .resource_regions()
                        .iter()
                        .filter(|region| region.good() == good)
                        .count(),
                    4
                );
            }
            for region in scenario.resource_regions() {
                assert_eq!(region.upper().x - region.lower().x, side);
                assert_eq!(region.upper().y - region.lower().y, side);
                assert!(region.upper().x <= map_size && region.upper().y <= map_size);
            }
            let state = scenario.initial()?;
            for actor in state.actors.iter() {
                let draw = DrawKey {
                    seed: 42,
                    day: 0,
                    phase: 0,
                    actor: actor.identity,
                    purpose: 0,
                    index: 0,
                }
                .draw();
                let original = match experiment {
                    Experiment::Concentrated => Position {
                        x: map_size / 2,
                        y: map_size / 2,
                    },
                    _ => Position {
                        x: draw as u32 % map_size,
                        y: (draw >> 32) as u32 % map_size,
                    },
                };
                match Good::from_index(actor.kind) {
                    Ok(good) if good.kind() == GoodKind::Natural => {
                        assert!(scenario.has_resource(good, actor.position));
                        if experiment == Experiment::Concentrated {
                            assert_eq!(actor.position, original);
                        }
                    }
                    _ => assert_eq!(actor.position, original),
                }
            }
            config.households = 48;
            config.firms = 80;
            assert_eq!(
                scenario.resource_regions(),
                Scenario::new(config.clone())?.resource_regions()
            );
            config.seed = 43;
            if map_size >= 4 {
                assert_ne!(
                    scenario.resource_regions(),
                    Scenario::new(config)?.resource_regions()
                );
            }
        }
    }
    Ok(())
}

#[test]
fn explicit_maps_replace_generated_maps_and_legacy_scenarios_default_to_generation() -> Result<()> {
    let baseline = Scenario::baseline(12, 42, Experiment::Baseline)?;
    let mut legacy = serde_json::to_value(baseline.config())?;
    legacy.as_object_mut().unwrap().remove("resources");
    assert_eq!(Scenario::new(serde_json::from_value(legacy)?)?, baseline);
    let mut config = baseline.config().clone();
    let only = ResourceRegion::new(
        Good::Grain,
        Position { x: 10, y: 20 },
        Position { x: 11, y: 21 },
        config.map_size,
    )?;
    config.resources = ResourceMap::Explicit {
        regions: vec![only],
    };
    let scenario = Scenario::new(config)?;
    assert_eq!(scenario.resource_regions(), &[only]);
    let generated = baseline.initial()?;
    for actor in scenario.initial()?.actors.iter() {
        match actor.kind {
            1 => assert_eq!(actor.position, only.lower()),
            _ => assert_eq!(actor, generated.actors.at(actor.identity as usize - 1)),
        }
    }
    let state = scenario.initial()?;
    let checkpoint = Checkpoint::new(&scenario, &state)?;
    assert_eq!(checkpoint.restore()?.scenario, scenario);
    let mut encoded = serde_json::to_value(checkpoint)?;
    assert_eq!(encoded["version"], 6);
    assert_eq!(encoded["scenario"]["version"], 2);
    assert_eq!(STATE_HASH_VERSION, "bad-econ/state/DeterministicV1/v7");
    encoded["version"] = serde_json::json!(4);
    assert!(serde_json::from_value::<Checkpoint>(encoded).is_err());
    Ok(())
}

#[cfg(feature = "compute")]
use crate::support;

#[cfg(feature = "compute")]
mod compute {
    use super::*;
    use bad_econ::{
        domain::research::*,
        runtime::SimulationCommand,
        simulation::{Simulation, Trace},
    };

    fn configuration(good: Good) -> Result<ScenarioConfig> {
        let mut config = Scenario::baseline(6, 42, Experiment::Baseline)?
            .config()
            .clone();
        config.coverage = 30;
        config.forecast_weight = 0;
        config.distribution_rate = 0;
        config.resources = ResourceMap::Explicit { regions: vec![] };
        config.behavior = BehaviorPolicy {
            work_fatigue: 0,
            rest_recovery: 0,
            wage_increment: 0,
            price_response: 0,
        };
        let position = Position { x: 10, y: 10 };
        let mut inventory = [0; Good::COUNT];
        for input in good.inputs().into_iter().flatten() {
            inventory[input as usize - 1] = 32;
        }
        config.population = Some(Population {
            households: (1..=6)
                .map(|identity| HouseholdSeed {
                    identity,
                    position,
                    cash: 0,
                    inventories: Good::food([0, 0, 30]),
                    fatigue: 0,
                })
                .collect(),
            firms: (0..3)
                .map(|index| FirmSeed {
                    identity: 100 + index,
                    position,
                    owner: index + 1,
                    good: match index {
                        0 => good,
                        _ => Good::Bread,
                    },
                    cash: match index {
                        0 => 10_000,
                        _ => 0,
                    },
                    inventories: match index {
                        0 => inventory,
                        _ => [0; Good::COUNT],
                    },
                    capacity: 2,
                    technique: Technique {
                        labor_minutes: 60,
                        input_ticks: match index {
                            0 if good.kind() == GoodKind::Natural => 0,
                            _ => 2,
                        },
                        output_ticks: 3,
                    },
                    wage: 1,
                    price: 10,
                })
                .collect(),
        });
        Ok(config)
    }

    fn local_resource(config: &mut ScenarioConfig, good: Good) -> Result<()> {
        config.resources = ResourceMap::Explicit {
            regions: vec![ResourceRegion::new(
                good,
                Position { x: 10, y: 10 },
                Position { x: 11, y: 11 },
                config.map_size,
            )?],
        };
        Ok(())
    }

    fn assert_conservation(simulation: &mut Simulation) -> Result<State> {
        let state = simulation.download()?;
        assert_eq!(
            simulation.snapshot,
            ObservationSnapshot::from_state(&simulation.scenario, &state)?
        );
        Ok(state)
    }

    #[test]
    fn generated_geography_preserves_custom_firm_positions_and_nominal_capacity() -> Result<()> {
        let mut config = configuration(Good::Grain)?;
        config.resources = ResourceMap::Generated;
        let geography = Scenario::new(config.clone())?;
        let map_size = config.map_size;
        let outside = (0..map_size)
            .flat_map(|y| (0..map_size).map(move |x| Position { x, y }))
            .find(|position| !geography.has_resource(Good::Grain, *position))
            .expect("four quarter-width patches cannot cover the map");
        let inside = geography
            .resource_regions()
            .iter()
            .find(|region| region.good() == Good::Grain)
            .unwrap()
            .lower();
        for position in [inside, outside] {
            config.population.as_mut().unwrap().firms[0].position = position;
            let scenario = Scenario::new(config.clone())?;
            let actor = scenario.initial()?.actors.at(6);
            assert_eq!(actor.position, position);
            assert_eq!(actor.capacity, 2);
            let expected = match position == inside {
                true => 2,
                false => 0,
            };
            assert_eq!(scenario.geographic_capacity(&actor), expected);
        }
        Ok(())
    }

    #[test]
    fn every_natural_good_requires_matching_geography_and_labor_on_gpu() -> Result<()> {
        struct Case {
            resource: Option<Good>,
            fatigue: u32,
            plan: u64,
            packets: u64,
        }
        for good in Good::ALL
            .into_iter()
            .filter(|good| good.kind() == GoodKind::Natural)
        {
            let wrong = match good {
                Good::Grain => Good::Iron,
                _ => Good::Grain,
            };
            for case in [
                Case {
                    resource: Some(good),
                    fatigue: 0,
                    plan: 2,
                    packets: 2,
                },
                Case {
                    resource: Some(good),
                    fatigue: 1000,
                    plan: 2,
                    packets: 0,
                },
                Case {
                    resource: Some(wrong),
                    fatigue: 0,
                    plan: 0,
                    packets: 0,
                },
                Case {
                    resource: None,
                    fatigue: 0,
                    plan: 0,
                    packets: 0,
                },
            ] {
                let mut config = configuration(good)?;
                if let Some(resource) = case.resource {
                    local_resource(&mut config, resource)?;
                }
                for household in &mut config.population.as_mut().unwrap().households {
                    household.fatigue = case.fatigue;
                }
                let scenario = Scenario::new(config)?;
                let initial = scenario.initial()?;
                let mut simulation = Simulation::new(scenario, support::device(64)?)?;
                simulation.advance()?;
                let state = assert_conservation(&mut simulation)?;
                assert_eq!(state.actors, initial.actors);
                assert_eq!(state.get(Field::Plan, 6), case.plan, "{good:?}");
                assert_eq!(state.get(Field::Produced, 6), case.packets * 3, "{good:?}");
                assert_eq!(
                    state.get(Field::Cash, 6),
                    10_000 - case.packets * 60,
                    "{good:?}"
                );
                assert_eq!(simulation.snapshot.employed, case.packets);
            }
        }
        Ok(())
    }

    #[test]
    fn manufacturers_need_labor_and_each_input_but_no_local_resource_on_gpu() -> Result<()> {
        enum Supply {
            Available,
            NoLabor,
            Missing(Good),
        }
        for good in Good::ALL
            .into_iter()
            .filter(|good| good.kind() == GoodKind::Manufactured)
        {
            let supplies = [Supply::Available, Supply::NoLabor]
                .into_iter()
                .chain(good.inputs().into_iter().flatten().map(Supply::Missing));
            for supply in supplies {
                let mut config = configuration(good)?;
                let population = config.population.as_mut().unwrap();
                let packets = match supply {
                    Supply::Available => 2,
                    Supply::NoLabor => {
                        for household in &mut population.households {
                            household.fatigue = 1000;
                        }
                        0
                    }
                    Supply::Missing(input) => {
                        population.firms[0].inventories[input as usize - 1] = 0;
                        0
                    }
                };
                let scenario = Scenario::new(config)?;
                assert_eq!(
                    scenario.production_site(good, Position { x: 10, y: 10 }),
                    ProductionSite::UnrestrictedManufacturing
                );
                let mut simulation = Simulation::new(scenario, support::device(64)?)?;
                simulation.advance()?;
                let state = assert_conservation(&mut simulation)?;
                assert_eq!(state.get(Field::Produced, 6), packets * 3, "{good:?}");
                assert_eq!(state.get(Field::Cash, 6), 10_000 - packets * 60, "{good:?}");
                for input in good.inputs().into_iter().flatten() {
                    assert_eq!(simulation.snapshot.used[input as usize - 1], packets * 2);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn unsuitable_producers_sell_inventory_without_production_wages_and_replay() -> Result<()> {
        let mut config = configuration(Good::Grain)?;
        let firms = &mut config.population.as_mut().unwrap().firms;
        firms[0].inventories[0] = 10;
        firms[1].good = Good::Flour;
        firms[1].cash = 10_000;
        let scenario = Scenario::new(config)?;
        let initial = scenario.initial()?;
        let mut canonical = None;
        for workgroup in [32, 64, 128, 256] {
            let mut simulation = Simulation::new(scenario.clone(), support::device(workgroup)?)?;
            simulation.advance()?;
            let state = assert_conservation(&mut simulation)?;
            assert!(state.get(Field::Sales, 6) > 0);
            assert_eq!(
                state.get(Field::Cash, 6),
                initial.get(Field::Cash, 6) + state.get(Field::Sales, 6) * 10
            );
            assert_eq!(state.get(Field::Produced, 6), 0);
            assert_eq!(state.get(Field::Plan, 6), 0);
            assert_eq!(simulation.snapshot.employed, 0);
            let checkpoint = simulation.checkpoint()?;
            simulation.advance()?;
            let expected = assert_conservation(&mut simulation)?;
            assert!(expected.get(Field::Produced, 7) > 0);
            assert_eq!(expected.get(Field::Produced, 6), 0);
            assert_eq!(expected.actors, initial.actors);
            match &canonical {
                None => canonical = Some(expected.clone()),
                Some(state) => assert_eq!(&expected, state),
            }
            simulation.command(SimulationCommand::Restore(checkpoint.into()))?;
            simulation.advance()?;
            assert_eq!(assert_conservation(&mut simulation)?, expected);
        }
        let mut invented = initial.clone();
        invented.set(Field::Plan, 6, 1);
        assert!(Checkpoint::new(&scenario, &invented).is_err());
        let mut invented = initial.clone();
        invented.set(Field::Produced, 6, 1);
        invented.set(Field::Grain, 6, initial.get(Field::Grain, 6) + 1);
        assert!(Checkpoint::new(&scenario, &invented).is_err());
        let mut invented = initial.clone();
        invented.set(Field::Employed, 0, 1);
        invented.set(Field::Employer, 0, 7);
        invented.set(Field::Hours, 0, 60);
        assert!(Checkpoint::new(&scenario, &invented).is_err());
        Ok(())
    }

    #[test]
    fn extraction_upgrades_keep_geography_and_tool_and_fuel_requirements() -> Result<()> {
        enum Supply {
            Available,
            NoResource,
            NoTools,
            NoFuel,
        }
        for good in Good::ALL
            .into_iter()
            .filter(|good| good.kind() == GoodKind::Natural)
        {
            for supply in [
                Supply::Available,
                Supply::NoResource,
                Supply::NoTools,
                Supply::NoFuel,
            ] {
                let mut config = configuration(good)?;
                let method = ProductionMethod::for_good(good)[0];
                let technology = method.technology() as u32;
                match supply {
                    Supply::NoResource => (),
                    _ => local_resource(&mut config, good)?,
                }
                let firm = &mut config.population.as_mut().unwrap().firms[0];
                firm.inventories[Good::Tools as usize - 1] = match supply {
                    Supply::NoTools => 0,
                    _ => 4,
                };
                firm.inventories[Good::Coal as usize - 1] = match supply {
                    Supply::NoFuel => 0,
                    _ => 4,
                };
                config.research = Some(ResearchConfig {
                    labs: vec![ResearchLab {
                        owner: 100,
                        researcher: 1,
                        discovered: technology,
                        mastered: technology,
                        permitted: technology,
                        adopted: technology,
                        consent: ResearchConsent::Granted,
                        production_requirements: 0,
                        adoption: AdoptionPolicy::Manual,
                        methods: vec![method],
                        projects: vec![],
                    }],
                    ..Default::default()
                });
                let packets = match supply {
                    Supply::Available => 2,
                    Supply::NoFuel if method != ProductionMethod::SteamMining => 2,
                    _ => 0,
                };
                let mut simulation = Simulation::new(Scenario::new(config)?, support::device(64)?)?;
                simulation.advance()?;
                let state = assert_conservation(&mut simulation)?;
                assert_eq!(good.kind(), GoodKind::Natural);
                assert_eq!(state.get(Field::ActiveMethod, 6), method as u64);
                assert_eq!(state.get(Field::Produced, 6), packets * 6, "{good:?}");
                assert_eq!(state.get(Field::Cash, 6), 10_000 - packets * 40, "{good:?}");
                assert_eq!(simulation.snapshot.used[Good::Tools as usize - 1], packets);
                if method == ProductionMethod::SteamMining {
                    assert_eq!(simulation.snapshot.used[Good::Coal as usize - 1], packets);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn research_equipment_and_stock_limits_only_reduce_geographic_capacity() -> Result<()> {
        for resource in [Some(Good::Grain), None] {
            let mut config = configuration(Good::Grain)?;
            if let Some(good) = resource {
                local_resource(&mut config, good)?;
            }
            config.research = Some(ResearchConfig {
                equipment: vec![Equipment {
                    firm: 100,
                    remaining_service: 1,
                    condition: 1000,
                    nameplate: 10,
                    downtime: 0,
                    wear_per_packet: 0,
                    maintenance_good: Good::Tools,
                    maintenance_ticks: 0,
                    restoration_per_tick: 0,
                }],
                natural_stocks: vec![NaturalStock {
                    firm: 100,
                    stock: 8,
                    capacity: 10,
                    renewal_rate: 0,
                    extraction_per_packet: 2,
                }],
                ..Default::default()
            });
            let scenario = Scenario::new(config)?;
            let packets = u64::from(scenario.has_resource(Good::Grain, Position { x: 10, y: 10 }));
            let mut simulation = Simulation::new(scenario, support::device(64)?)?;
            for phase in [Phase::Opening, Phase::Observation, Phase::Proposal] {
                simulation.phase(phase, Trace::Record)?;
            }
            assert_eq!(simulation.download()?.get(Field::Plan, 6), packets);
            for phase in Phase::ALL.into_iter().skip(3) {
                simulation.phase(phase, Trace::Record)?;
            }
            let state = assert_conservation(&mut simulation)?;
            assert_eq!(state.get(Field::Produced, 6), packets * 3);
            assert_eq!(simulation.snapshot.research.remaining_service, 1 - packets);
            assert_eq!(simulation.snapshot.research.natural_stock, 8 - packets * 2);
        }
        Ok(())
    }
}

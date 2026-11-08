#![cfg(feature = "compute")]
use crate::support;
use anyhow::Result;
use bad_econ::{
    domain::{railway::*, *},
    simulation::{Simulation, Trace},
};

fn network() -> Result<ScenarioConfig> {
    let mut config = Scenario::baseline(4, 42, Experiment::Transport)?
        .config()
        .clone();
    config.firms = 6;
    config.search_radius = 1000;
    config.shipment_speed = 100;
    config.coverage = 2;
    config.forecast_weight = 0;
    config.distribution_rate = 0;
    config.entrepreneurship.interested_per_mille = 0;
    config.behavior = BehaviorPolicy {
        work_fatigue: 0,
        rest_recovery: 0,
        wage_increment: 0,
        price_response: 0,
    };
    config.demography.annual_births_per_million_fertile_women = 0;
    config.demography.mortality.background_annual_ppm = [0; 5];
    config.demography.mortality.occupational_annual_ppm = [0; Good::COUNT];
    config.demography.mortality.famine_daily_ppm_at_thirty_days = 0;
    config.railway = RailwayConfig {
        speed_multiplier: 10,
        length_unit: 1000,
        units_per_level: 100,
        construction: RailMaterials {
            coal: 4,
            steel: 6,
            locomotives: 2,
        },
        maintenance: RailMaterials {
            coal: 1,
            steel: 1,
            locomotives: 1,
        },
        companies: vec![
            RailwayCompany {
                firm: 11,
                freight_rate: 2,
                access_fee: 3,
            },
            RailwayCompany {
                firm: 12,
                freight_rate: 3,
                access_fee: 5,
            },
        ],
        segments: vec![
            RailSegment {
                company: 11,
                from: Position { x: 0, y: 0 },
                to: Position { x: 400, y: 0 },
            },
            RailSegment {
                company: 12,
                from: Position { x: 400, y: 0 },
                to: Position { x: 900, y: 0 },
            },
        ],
    };
    config.population = Some(Population {
        households: (1..=4)
            .map(|identity| HouseholdSeed {
                identity,
                position: Position { x: 900, y: 0 },
                cash: match identity {
                    1 => 1000,
                    _ => 0,
                },
                inventories: match identity {
                    1 => [0; Good::COUNT],
                    _ => Good::food([0, 0, 100]),
                },
                fatigue: 0,
            })
            .collect(),
        firms: [
            Good::Bread,
            Good::Railway,
            Good::Railway,
            Good::Coal,
            Good::Steel,
            Good::Locomotives,
        ]
        .into_iter()
        .enumerate()
        .map(|(index, good)| {
            let mut inventories = [0; Good::COUNT];
            match good {
                Good::Bread => inventories[Good::Bread as usize - 1] = 1000,
                Good::Railway => {
                    inventories[Good::Coal as usize - 1] = 20;
                    inventories[Good::Steel as usize - 1] = 20;
                    inventories[Good::Locomotives as usize - 1] = 20;
                }
                _ => (),
            }
            FirmSeed {
                identity: 10 + index as u64,
                position: Position { x: 0, y: 0 },
                owner: (index % 4 + 1) as u64,
                good,
                cash: match good {
                    Good::Bread => 10_000,
                    _ => 0,
                },
                inventories,
                capacity: match good {
                    Good::Railway => 0,
                    _ => 1,
                },
                technique: match good {
                    Good::Railway => Technique::railway(),
                    _ => Technique::new(60, u32::from(good.inputs()[0].is_some()), 1).unwrap(),
                },
                wage: match good {
                    Good::Bread => 10_000,
                    _ => 1,
                },
                price: 10,
            }
        })
        .collect(),
    });
    Ok(config)
}

fn delivered(state: &State, scenario: &Scenario, day: usize) -> u64 {
    state.values[state
        .shipment_coordinate(Good::Bread, day % scenario.ring(), scenario.ring(), 0)
        .0 as usize]
}

#[test]
fn connected_railways_charge_freight_and_foreign_track_fees_on_gpu() -> Result<()> {
    let scenario = Scenario::new(network()?)?;
    let mut simulation = Simulation::new(scenario.clone(), support::device(64)?)?;
    simulation.advance()?;
    let state = simulation.download()?;
    let quantity = state.get(Field::Purchased, 0);
    assert!(quantity > 0);
    assert_eq!(delivered(&state, &scenario, 1), quantity);
    assert_eq!(state.get(Field::Cash, 0), 1000 - quantity * 10);
    assert_eq!(state.get(Field::Cash, 4), 10_000 + quantity * 3);
    assert_eq!(state.get(Field::Cash, 5), quantity * 2);
    assert_eq!(state.get(Field::Cash, 6), quantity * 5);
    assert_eq!(state.get(Field::BusinessRevenue, 5), quantity * 2);
    assert_eq!(state.get(Field::BusinessRevenue, 6), quantity * 5);
    assert_eq!(state.get(Field::FreightEscrow, 0), 0);
    assert_eq!(simulation.snapshot.used[Good::Coal as usize - 1], 8);
    assert_eq!(simulation.snapshot.used[Good::Steel as usize - 1], 12);
    assert_eq!(simulation.snapshot.used[Good::Locomotives as usize - 1], 4);
    let mut replay = Simulation::from_checkpoint(simulation.checkpoint()?, support::device(128)?)?;
    for _ in 0..4 {
        simulation.advance()?;
        replay.advance()?;
        assert_eq!(simulation.download()?, replay.download()?);
    }
    simulation.checkpoint()?.restore()?;
    Ok(())
}

#[test]
fn disconnected_or_unbuilt_tracks_use_animals_on_gpu() -> Result<()> {
    for missing in [
        None,
        Some(Good::Coal),
        Some(Good::Steel),
        Some(Good::Locomotives),
    ] {
        let mut config = network()?;
        match missing {
            None => config.railway.segments[1].from.x = 401,
            Some(good) => {
                config.population.as_mut().unwrap().firms[2].inventories[good as usize - 1] = 0
            }
        }
        let scenario = Scenario::new(config)?;
        let mut simulation = Simulation::new(scenario.clone(), support::device(64)?)?;
        simulation.advance()?;
        let state = simulation.download()?;
        let quantity = state.get(Field::Purchased, 0);
        assert!(quantity > 0);
        assert_eq!(delivered(&state, &scenario, 9), quantity);
        assert_eq!(delivered(&state, &scenario, 1), 0);
        assert_eq!(state.get(Field::Cash, 0), 1000 - quantity * 10);
        assert_eq!(state.get(Field::Cash, 5), 0);
        assert_eq!(state.get(Field::Cash, 6), 0);
        if missing.is_some() {
            assert_eq!(state.get(Field::RailBuilt, 6), 0);
            assert_eq!(state.get(Field::Used, 6), 0);
        }
    }
    Ok(())
}

#[test]
fn railways_suspend_without_upkeep_and_preserve_dispatched_deliveries_on_gpu() -> Result<()> {
    let mut config = network()?;
    for firm in &mut config.population.as_mut().unwrap().firms[1..=2] {
        for (good, quantity) in [(Good::Coal, 4), (Good::Steel, 6), (Good::Locomotives, 2)] {
            firm.inventories[good as usize - 1] = quantity;
        }
    }
    let mut simulation = Simulation::new(Scenario::new(config)?, support::device(64)?)?;
    simulation.advance()?;
    let first = simulation.download()?;
    assert_eq!(first.get(Field::RailOperating, 5), 1);
    let shipped = first.get(Field::Purchased, 0);
    simulation.advance()?;
    let second = simulation.download()?;
    for firm in [5, 6] {
        assert_eq!(second.get(Field::RailBuilt, firm), 1);
        assert_eq!(second.get(Field::RailOperating, firm), 0);
        assert_eq!(second.get(Field::Railway, firm), 0);
    }
    assert!(second.get(Field::Bread, 0) >= shipped - 1);
    simulation.checkpoint()?.restore()?;
    Ok(())
}

#[test]
fn senders_pay_rail_fares_and_reverse_routes_pay_the_other_owner_on_gpu() -> Result<()> {
    for reverse in [false, true] {
        let mut config = network()?;
        let population = config.population.as_mut().unwrap();
        population.households[0].cash = 10;
        if reverse {
            population.households[0].position.x = 0;
            population.firms[0].position.x = 900;
        }
        let mut simulation = Simulation::new(Scenario::new(config)?, support::device(64)?)?;
        simulation.advance()?;
        let state = simulation.download()?;
        assert_eq!(state.get(Field::Purchased, 0), 1);
        assert_eq!(state.get(Field::Cash, 0), 0);
        assert_eq!(
            state.get(Field::Cash, 5),
            match reverse {
                false => 2,
                true => 3,
            }
        );
        assert_eq!(
            state.get(Field::Cash, 6),
            match reverse {
                false => 5,
                true => 3,
            }
        );
    }
    Ok(())
}

#[test]
fn railroad_construction_waits_for_paid_material_deliveries_on_gpu() -> Result<()> {
    let mut config = network()?;
    let population = config.population.as_mut().unwrap();
    for firm in &mut population.firms[1..=2] {
        firm.cash = 1000;
        firm.inventories = [0; Good::COUNT];
    }
    for firm in &mut population.firms[3..] {
        firm.inventories[firm.good as usize - 1] = 1000;
    }
    let mut simulation = Simulation::new(Scenario::new(config)?, support::device(64)?)?;
    simulation.advance()?;
    let waiting = simulation.download()?;
    assert_eq!(waiting.get(Field::RailBuilt, 5), 0);
    assert!(waiting.get(Field::Transit, 5) > 0);
    assert!(waiting.get(Field::Cash, 5) < 1000);
    simulation.advance()?;
    let built = simulation.download()?;
    assert_eq!(built.get(Field::RailBuilt, 5), 1);
    assert_eq!(built.get(Field::RailBuilt, 6), 1);
    assert_eq!(built.get(Field::RailOperating, 5), 1);
    simulation.checkpoint()?.restore()?;
    Ok(())
}

#[test]
fn locomotives_require_steel_and_tools_on_gpu() -> Result<()> {
    for stock in [0, 2] {
        let mut config = network()?;
        config.railway = RailwayConfig::default();
        let factory = &mut config.population.as_mut().unwrap().firms[5];
        factory.cash = 1000;
        factory.inventories[Good::Steel as usize - 1] = 2;
        factory.inventories[Good::Tools as usize - 1] = stock;
        let scenario = Scenario::new(config)?;
        let mut state = scenario.initial()?;
        state.set(Field::Demand, 9, 2);
        let mut simulation =
            Simulation::from_checkpoint(Checkpoint::new(&scenario, &state)?, support::device(64)?)?;
        for phase in Phase::ALL {
            simulation.phase(phase, Trace::Discard)?;
        }
        assert_eq!(
            simulation.snapshot.produced[Good::Locomotives as usize - 1],
            stock.min(1)
        );
        assert_eq!(
            simulation.snapshot.used[Good::Tools as usize - 1],
            stock.min(1)
        );
    }
    Ok(())
}

#[test]
fn a_company_uses_its_other_sites_tracks_without_foreign_fees_on_gpu() -> Result<()> {
    let mut config = network()?;
    config.population.as_mut().unwrap().firms[2].owner = 2;
    config.companies.push(company::Company {
        identity: 11,
        sites: vec![11, 12],
    });
    let mut simulation = Simulation::new(Scenario::new(config)?, support::device(64)?)?;
    simulation.advance()?;
    let state = simulation.download()?;
    let quantity = state.get(Field::Purchased, 0);
    assert_eq!(state.get(Field::Cash, 0), 1000 - quantity * 10);
    assert_eq!(state.get(Field::Cash, 5), quantity * 2);
    assert_eq!(state.get(Field::Cash, 6), 0);
    Ok(())
}

#[test]
fn generated_industry_has_rail_plans_and_gpu_inspection_survives_checkpointing() -> Result<()> {
    let scenario = Scenario::industrial(48, 42, Experiment::Baseline)?;
    assert!(!scenario.config().railway.segments.is_empty());
    let identity = scenario.config().railway.companies[0].firm;
    let request = InspectionRequest::new(identity, &scenario)?;
    let mut simulation = Simulation::new(scenario, support::device(64)?)?;
    for _ in 0..8 {
        simulation.advance()?;
    }
    let inspection = simulation.inspect(request.clone())?;
    let state = simulation.download()?;
    assert_eq!(inspection, Inspection::from_state(&state, &request)?);
    assert!(inspection.railway_business.is_some());
    simulation.checkpoint()?.restore()?;
    Ok(())
}

#[test]
fn regional_freight_receipts_preserve_wide_cash_and_workgroup_replay_on_gpu() -> Result<()> {
    let mut config = network()?;
    config.household_need = 100;
    config.coverage = 3;
    config.railway.length_unit = 1;
    config.railway.units_per_level = 100_000;
    for company in &mut config.railway.companies {
        company.freight_rate = 1000;
        company.access_fee = 1000;
    }
    let population = config.population.as_mut().unwrap();
    for household in &mut population.households {
        household.cash = 10_000_000_000;
        household.inventories = [0; Good::COUNT];
    }
    population.firms[0].inventories[Good::Bread as usize - 1] = 20_000;
    population.firms[0].cash = 100_000_000_000;
    population.firms[0].capacity = 1;
    for firm in &mut population.firms[1..=2] {
        for good in [Good::Coal, Good::Steel, Good::Locomotives] {
            firm.inventories[good as usize - 1] = 10_000;
        }
    }
    let scenario = Scenario::new(config)?;
    let mut simulation = Simulation::new(scenario.clone(), support::device(64)?)?;
    let mut replay = Simulation::new(scenario, support::device(128)?)?;
    simulation.advance()?;
    replay.advance()?;
    let state = simulation.download()?;
    assert_eq!(state, replay.download()?);
    let shipped = (0..4)
        .map(|buyer| state.get(Field::Purchased, buyer))
        .sum::<u64>();
    assert!(shipped * 900_000 > u64::from(u32::MAX));
    assert_eq!(state.get(Field::Cash, 5), shipped * 900_000);
    assert_eq!(state.get(Field::Cash, 6), shipped * 500_000);
    assert_eq!(state.get(Field::BusinessRevenue, 5), shipped * 900_000);
    simulation.checkpoint()?.restore()?;
    Ok(())
}

#[test]
fn suspended_tracks_resume_after_paid_upkeep_arrives_on_gpu() -> Result<()> {
    let mut config = network()?;
    let population = config.population.as_mut().unwrap();
    for firm in &mut population.firms[1..=2] {
        firm.cash = 1000;
        for (good, quantity) in [(Good::Coal, 4), (Good::Steel, 6), (Good::Locomotives, 2)] {
            firm.inventories[good as usize - 1] = quantity;
        }
    }
    for firm in &mut population.firms[3..] {
        firm.position.x = 200;
        firm.inventories[firm.good as usize - 1] = 1000;
    }
    let mut simulation = Simulation::new(Scenario::new(config)?, support::device(64)?)?;
    for operating in [1, 0, 1] {
        simulation.advance()?;
        let state = simulation.download()?;
        assert_eq!(state.get(Field::RailBuilt, 5), 1);
        assert_eq!(state.get(Field::RailOperating, 5), operating);
    }
    Ok(())
}

#[test]
fn railway_example_runs_with_conserved_goods_and_cash_on_gpu() -> Result<()> {
    let config = serde_json::from_str(include_str!("../scenarios/railway.json"))?;
    let mut simulation = Simulation::new(Scenario::new(config)?, support::device(64)?)?;
    for _ in 0..14 {
        simulation.advance()?;
    }
    let state = simulation.download()?;
    assert_eq!(state.get(Field::RailBuilt, 9), 1);
    assert_eq!(state.get(Field::RailBuilt, 10), 1);
    simulation.checkpoint()?.restore()?;
    Ok(())
}

fn regional_network() -> Result<ScenarioConfig> {
    let mut config = network()?;
    config.railway.length_unit = 100;
    config.railway.units_per_level = 1;
    config.railway.companies.truncate(1);
    config.railway.segments = vec![RailSegment {
        company: 11,
        from: Position { x: 0, y: 0 },
        to: Position { x: 500, y: 0 },
    }];
    config.expansion = enterprise::ExpansionPolicy {
        small_business_per_mille: 0,
        industrial_per_mille: 0,
    };
    let population = config.population.as_mut().unwrap();
    population.households[0].position.x = 500;
    population.households[0].cash = 10;
    for good in [Good::Coal, Good::Steel, Good::Locomotives] {
        population.firms[1].inventories[good as usize - 1] = 1000;
    }
    Ok(config)
}

fn regional_value(
    scenario: &Scenario,
    state: &State,
    record: usize,
    field: RailCapacityField,
) -> u64 {
    state.values[scenario
        .rail_geography()
        .coordinate(scenario, state, field, record)]
}

#[test]
fn five_regions_consume_five_local_units_even_on_one_long_segment_on_gpu() -> Result<()> {
    let scenario = Scenario::new(regional_network()?)?;
    let mut simulation = Simulation::new(scenario.clone(), support::device(64)?)?;
    simulation.advance()?;
    let state = simulation.download()?;
    assert_eq!(state.get(Field::Purchased, 0), 1);
    assert_eq!(delivered(&state, &scenario, 1), 1);
    assert_eq!(state.get(Field::Cash, 0), 0);
    assert_eq!(state.get(Field::Cash, 4), 10_000);
    assert_eq!(state.get(Field::Cash, 5), 10);
    assert_eq!(
        state.values[(USED_FIELDS + Good::Railway as usize - 1) * state.actors.len() + 4],
        5
    );
    assert_eq!(state.get(Field::Produced, 5), 6);
    assert_eq!(state.get(Field::Railway, 5), 1);
    for record in 0..6 {
        assert_eq!(
            regional_value(&scenario, &state, record, RailCapacityField::Available),
            u64::from(record == 0)
        );
    }
    simulation.checkpoint()?.restore()?;
    Ok(())
}

#[test]
fn regional_bottleneck_uses_animals_without_partial_rail_payments_on_gpu() -> Result<()> {
    let mut config = regional_network()?;
    config.population.as_mut().unwrap().households[0].cash = 20;
    let scenario = Scenario::new(config)?;
    let mut initial = scenario.initial()?;
    initial.set(Field::RailBuilt, 5, 1);
    for record in 0..6 {
        let cell = scenario.rail_geography().coordinate(
            &scenario,
            &initial,
            RailCapacityField::Level,
            record,
        );
        initial.values[cell] = match record {
            3 => 1,
            _ => 2,
        };
    }
    let mut simulation =
        Simulation::from_checkpoint(Checkpoint::new(&scenario, &initial)?, support::device(64)?)?;
    simulation.advance()?;
    let state = simulation.download()?;
    assert_eq!(state.get(Field::Purchased, 0), 2);
    assert_eq!(delivered(&state, &scenario, 1), 0);
    assert_eq!(delivered(&state, &scenario, 5), 2);
    assert_eq!(state.get(Field::Cash, 4), 10_020);
    assert_eq!(state.get(Field::Cash, 5), 0);
    assert_eq!(
        state.values[(USED_FIELDS + Good::Railway as usize - 1) * state.actors.len() + 4],
        0
    );
    for record in 0..6 {
        assert_eq!(
            regional_value(&scenario, &state, record, RailCapacityField::Available),
            match record {
                3 => 1,
                _ => 2,
            }
        );
    }
    assert_eq!(
        regional_value(&scenario, &state, 3, RailCapacityField::Demand),
        2
    );
    simulation.checkpoint()?.restore()?;
    Ok(())
}

#[test]
fn shared_regions_do_not_oversell_and_replenish_next_day_on_gpu() -> Result<()> {
    let mut config = regional_network()?;
    let buyer = &mut config.population.as_mut().unwrap().households[1];
    buyer.cash = 10;
    buyer.inventories = [0; Good::COUNT];
    buyer.position.x = 500;
    let scenario = Scenario::new(config)?;
    let mut simulation = Simulation::new(scenario.clone(), support::device(64)?)?;
    simulation.advance()?;
    let state = simulation.download()?;
    let arrivals = |day| {
        (0..2)
            .map(|buyer| {
                state.values[state
                    .shipment_coordinate(Good::Bread, day, scenario.ring(), buyer)
                    .0 as usize]
            })
            .sum::<u64>()
    };
    assert_eq!(arrivals(1), 1);
    assert_eq!(arrivals(5), 1);
    assert_eq!(state.get(Field::Cash, 5), 10);
    assert_eq!(
        state.values[(USED_FIELDS + Good::Railway as usize - 1) * state.actors.len() + 4],
        5
    );
    let mut replay = Simulation::from_checkpoint(simulation.checkpoint()?, support::device(128)?)?;
    simulation.advance()?;
    replay.advance()?;
    let next = simulation.download()?;
    assert_eq!(next, replay.download()?);
    for record in 0..6 {
        assert_eq!(
            regional_value(&scenario, &next, record, RailCapacityField::Available),
            1
        );
    }
    simulation.checkpoint()?.restore()?;
    Ok(())
}

#[test]
fn only_expanding_owners_with_materials_upgrade_a_busy_region_on_gpu() -> Result<()> {
    for preference in [0, 1000] {
        for materials in [0, 1000] {
            let mut config = regional_network()?;
            config.expansion = enterprise::ExpansionPolicy {
                small_business_per_mille: preference,
                industrial_per_mille: preference,
            };
            for good in [Good::Coal, Good::Steel, Good::Locomotives] {
                config.population.as_mut().unwrap().firms[1].inventories[good as usize - 1] =
                    materials + 5;
            }
            let scenario = Scenario::new(config)?;
            let mut initial = scenario.initial()?;
            initial.set(Field::RailBuilt, 5, 1);
            for record in 0..6 {
                let cell = scenario.rail_geography().coordinate(
                    &scenario,
                    &initial,
                    RailCapacityField::Level,
                    record,
                );
                initial.values[cell] = 1;
            }
            let demand = scenario.rail_geography().coordinate(
                &scenario,
                &initial,
                RailCapacityField::Demand,
                3,
            );
            initial.values[demand] = 2;
            let mut simulation = Simulation::from_checkpoint(
                Checkpoint::new(&scenario, &initial)?,
                support::device(64)?,
            )?;
            simulation.advance()?;
            let state = simulation.download()?;
            assert_eq!(
                regional_value(&scenario, &state, 3, RailCapacityField::Level),
                1 + u64::from(preference == 1000 && materials == 1000)
            );
            assert_eq!(
                regional_value(&scenario, &state, 2, RailCapacityField::Level),
                1
            );
            simulation.checkpoint()?.restore()?;
        }
    }
    Ok(())
}

#[test]
fn splitting_segments_and_property_boundaries_preserve_regional_charges_on_gpu() -> Result<()> {
    for property in [None, Some(100), Some(200)] {
        let mut config = regional_network()?;
        config.railway.segments = [0, 80, 160, 300, 400, 500]
            .windows(2)
            .map(|pair| RailSegment {
                company: 11,
                from: Position { x: pair[0], y: 0 },
                to: Position { x: pair[1], y: 0 },
            })
            .collect();
        if let Some(width) = property {
            config.railway.length_unit = 1000;
            config.property = Some(property::PropertyConfig {
                regions: (0..1000 / width)
                    .map(|region| property::PropertyRegion {
                        lower: Position {
                            x: region * width,
                            y: 0,
                        },
                        upper: Position {
                            x: (region + 1) * width,
                            y: 1000,
                        },
                        plots: 100,
                        landowner: 1,
                    })
                    .collect(),
                buildings: vec![],
                orders: vec![],
            });
        }
        let scenario = Scenario::new(config)?;
        let mut initial = scenario.initial()?;
        initial.set(Field::RailBuilt, 5, 5);
        let mut simulation = Simulation::from_checkpoint(
            Checkpoint::new(&scenario, &initial)?,
            support::device(64)?,
        )?;
        simulation.advance()?;
        let state = simulation.download()?;
        let regions = match property {
            Some(200) => 2,
            _ => 5,
        };
        assert_eq!(delivered(&state, &scenario, 1), 1);
        assert_eq!(
            state.values[(USED_FIELDS + Good::Railway as usize - 1) * state.actors.len() + 4],
            regions
        );
        assert_eq!(state.get(Field::Cash, 5), regions * 2);
        simulation.checkpoint()?.restore()?;
    }
    Ok(())
}

#[test]
fn sender_without_freight_funds_uses_animals_and_keeps_all_rail_supply_on_gpu() -> Result<()> {
    let mut config = regional_network()?;
    config.population.as_mut().unwrap().firms[0].cash = 0;
    let scenario = Scenario::new(config)?;
    let mut simulation = Simulation::new(scenario.clone(), support::device(64)?)?;
    simulation.advance()?;
    let state = simulation.download()?;
    assert_eq!(delivered(&state, &scenario, 5), 1);
    assert_eq!(delivered(&state, &scenario, 1), 0);
    assert_eq!(state.get(Field::Cash, 4), 10);
    assert_eq!(state.get(Field::Cash, 5), 0);
    assert_eq!(state.get(Field::Railway, 5), 6);
    simulation.checkpoint()?.restore()?;
    Ok(())
}

#[test]
fn regional_expansion_waits_for_funded_material_deliveries_on_gpu() -> Result<()> {
    for cash in [0, 10_000] {
        let mut config = regional_network()?;
        config.expansion = enterprise::ExpansionPolicy {
            small_business_per_mille: 1000,
            industrial_per_mille: 1000,
        };
        let population = config.population.as_mut().unwrap();
        population.households[0].cash = 0;
        population.firms[1].cash = cash;
        for good in [Good::Coal, Good::Steel, Good::Locomotives] {
            population.firms[1].inventories[good as usize - 1] = 5;
        }
        for firm in &mut population.firms[3..] {
            firm.inventories[firm.good as usize - 1] = 1000;
        }
        let scenario = Scenario::new(config)?;
        let mut initial = scenario.initial()?;
        initial.set(Field::RailBuilt, 5, 1);
        for record in 0..6 {
            let cell = scenario.rail_geography().coordinate(
                &scenario,
                &initial,
                RailCapacityField::Level,
                record,
            );
            initial.values[cell] = 1;
        }
        let demand =
            scenario
                .rail_geography()
                .coordinate(&scenario, &initial, RailCapacityField::Demand, 3);
        initial.values[demand] = 2;
        let mut simulation = Simulation::from_checkpoint(
            Checkpoint::new(&scenario, &initial)?,
            support::device(64)?,
        )?;
        simulation.advance()?;
        let first = simulation.download()?;
        assert_eq!(
            regional_value(&scenario, &first, 3, RailCapacityField::Level),
            1
        );
        assert_eq!(first.get(Field::Transit, 5) > 0, cash > 0);
        simulation.advance()?;
        let second = simulation.download()?;
        assert_eq!(
            regional_value(&scenario, &second, 3, RailCapacityField::Level),
            1 + u64::from(cash > 0)
        );
        simulation.checkpoint()?.restore()?;
    }
    Ok(())
}

#[test]
fn regional_capacity_survives_actor_registry_growth_and_checkpoint_replay_on_gpu() -> Result<()> {
    use bad_econ::domain::demography::{PersonSeed, Sex};
    let mut config = regional_network()?;
    config.seed = 3;
    config.railway.length_unit = 10;
    config.household_production = enterprise::HouseholdProduction::industrial();
    for household in &mut config.population.as_mut().unwrap().households {
        household.cash = 10_000;
        household.inventories[Good::Bread as usize - 1] = 100;
    }
    config.demography.people = vec![
        PersonSeed {
            identity: 100,
            household: 1,
            age_days: 30 * 365,
            sex: Sex::Female,
        },
        PersonSeed {
            identity: 365,
            household: 1,
            age_days: 18 * 365,
            sex: Sex::Male,
        },
        PersonSeed {
            identity: 200,
            household: 2,
            age_days: 30 * 365,
            sex: Sex::Male,
        },
        PersonSeed {
            identity: 300,
            household: 3,
            age_days: 30 * 365,
            sex: Sex::Female,
        },
        PersonSeed {
            identity: 400,
            household: 4,
            age_days: 30 * 365,
            sex: Sex::Female,
        },
    ];
    let scenario = Scenario::new(config)?;
    let mut simulation = Simulation::new(scenario.clone(), support::device(64)?)?;
    simulation.advance()?;
    let state = simulation.download()?;
    assert_eq!(state.actors.len(), scenario.actors() + 1);
    assert_eq!(scenario.rail_geography().regions.len(), 51);
    for record in 0..51 {
        assert_eq!(
            regional_value(&scenario, &state, record, RailCapacityField::Available),
            1
        );
    }
    let mut replay = Simulation::from_checkpoint(simulation.checkpoint()?, support::device(128)?)?;
    simulation.advance()?;
    replay.advance()?;
    assert_eq!(simulation.download()?, replay.download()?);
    simulation.checkpoint()?.restore()?;
    Ok(())
}

#[test]
fn closing_businesses_preserves_regional_records_and_expires_rail_services_on_gpu() -> Result<()> {
    for firm in [4, 5] {
        let mut config = regional_network()?;
        config.population.as_mut().unwrap().households[0].cash = 0;
        let scenario = Scenario::new(config)?;
        let mut initial = scenario.initial()?;
        initial.set(Field::RailBuilt, 5, 1);
        initial.set(
            Field::EnterpriseStatus,
            firm,
            enterprise::EnterpriseState::Liquidating as u64,
        );
        for record in 0..6 {
            let cell = scenario.rail_geography().coordinate(
                &scenario,
                &initial,
                RailCapacityField::Level,
                record,
            );
            initial.values[cell] = 1;
        }
        let mut simulation = Simulation::from_checkpoint(
            Checkpoint::new(&scenario, &initial)?,
            support::device(64)?,
        )?;
        simulation.advance()?;
        let state = simulation.download()?;
        assert_eq!(
            state.get(Field::EnterpriseStatus, firm),
            enterprise::EnterpriseState::Closed as u64
        );
        for record in 0..6 {
            assert_eq!(
                regional_value(&scenario, &state, record, RailCapacityField::Level),
                1
            );
            assert_eq!(
                regional_value(&scenario, &state, record, RailCapacityField::Available),
                u64::from(firm == 4)
            );
        }
        simulation.checkpoint()?.restore()?;
    }
    Ok(())
}

#[test]
fn every_foreign_operator_is_paid_even_within_one_region_on_gpu() -> Result<()> {
    let mut config = network()?;
    config.population.as_mut().unwrap().households[0].cash = 10;
    let company = &mut config.population.as_mut().unwrap().firms[3];
    company.good = Good::Railway;
    company.capacity = 0;
    company.technique = Technique::railway();
    for good in [Good::Coal, Good::Steel, Good::Locomotives] {
        company.inventories[good as usize - 1] = 100;
    }
    config.railway.companies.push(RailwayCompany {
        firm: 13,
        freight_rate: 4,
        access_fee: 7,
    });
    config.railway.segments[1].to.x = 600;
    config.railway.segments.push(RailSegment {
        company: 13,
        from: Position { x: 600, y: 0 },
        to: Position { x: 900, y: 0 },
    });
    let scenario = Scenario::new(config)?;
    let mut simulation = Simulation::new(scenario.clone(), support::device(64)?)?;
    simulation.advance()?;
    let state = simulation.download()?;
    assert_eq!(delivered(&state, &scenario, 1), 1);
    assert_eq!(
        state.values[(USED_FIELDS + Good::Railway as usize - 1) * state.actors.len() + 4],
        1
    );
    assert_eq!(state.get(Field::Cash, 4), 10_000 + 10 - 2 - 5 - 7);
    for (firm, fee) in [(5, 2), (6, 5), (7, 7)] {
        assert_eq!(state.get(Field::Cash, firm), fee);
    }
    simulation.checkpoint()?.restore()?;
    Ok(())
}

#[test]
fn competing_companies_produce_separate_capacity_in_the_same_regions_on_gpu() -> Result<()> {
    let mut config = regional_network()?;
    config.railway.companies.push(RailwayCompany {
        firm: 12,
        freight_rate: 3,
        access_fee: 5,
    });
    config.railway.segments.push(RailSegment {
        company: 12,
        from: Position { x: 0, y: 0 },
        to: Position { x: 500, y: 0 },
    });
    let population = config.population.as_mut().unwrap();
    population.households[1].cash = 10;
    population.households[1].position.x = 500;
    population.households[1].inventories = [0; Good::COUNT];
    for good in [Good::Coal, Good::Steel, Good::Locomotives] {
        population.firms[2].inventories[good as usize - 1] = 1000;
    }
    let scenario = Scenario::new(config)?;
    let mut simulation = Simulation::new(scenario.clone(), support::device(64)?)?;
    simulation.advance()?;
    let state = simulation.download()?;
    assert_eq!(scenario.rail_geography().regions.len(), 12);
    let arrivals = (0..2)
        .map(|buyer| {
            state.values[state
                .shipment_coordinate(Good::Bread, 1, scenario.ring(), buyer)
                .0 as usize]
        })
        .sum::<u64>();
    assert_eq!(arrivals, 2);
    assert_eq!(state.get(Field::Cash, 5), 10);
    assert_eq!(state.get(Field::Cash, 6), 15);
    assert_eq!(state.get(Field::Railway, 5), 1);
    assert_eq!(state.get(Field::Railway, 6), 1);
    assert_eq!(
        state.values[(USED_FIELDS + Good::Railway as usize - 1) * state.actors.len() + 4],
        10
    );
    for record in 0..12 {
        assert_eq!(
            regional_value(&scenario, &state, record, RailCapacityField::Available),
            u64::from(record % 6 == 0)
        );
    }
    simulation.checkpoint()?.restore()?;
    Ok(())
}

#[test]
fn carriers_prefer_their_own_connected_tracks_over_a_foreign_shortcut_on_gpu() -> Result<()> {
    let mut config = network()?;
    config.population.as_mut().unwrap().households[0].cash = 10;
    let points = [(0, 0), (200, 0), (200, 200), (900, 200), (900, 0)];
    config.railway.segments = points
        .windows(2)
        .map(|pair| RailSegment {
            company: 11,
            from: Position {
                x: pair[0].0,
                y: pair[0].1,
            },
            to: Position {
                x: pair[1].0,
                y: pair[1].1,
            },
        })
        .collect();
    config.railway.segments.push(RailSegment {
        company: 12,
        from: Position { x: 200, y: 0 },
        to: Position { x: 900, y: 0 },
    });
    let scenario = Scenario::new(config)?;
    let mut initial = scenario.initial()?;
    initial.set(Field::RailBuilt, 5, 4);
    initial.set(Field::RailBuilt, 6, 1);
    let mut simulation =
        Simulation::from_checkpoint(Checkpoint::new(&scenario, &initial)?, support::device(64)?)?;
    simulation.advance()?;
    let state = simulation.download()?;
    assert_eq!(delivered(&state, &scenario, 1), 0);
    assert_eq!(delivered(&state, &scenario, 2), 1);
    assert_eq!(state.get(Field::Cash, 5), 2);
    assert_eq!(state.get(Field::Cash, 6), 0);
    assert_eq!(state.get(Field::Sales, 5), 1);
    assert_eq!(state.get(Field::Sales, 6), 0);
    simulation.checkpoint()?.restore()?;
    Ok(())
}

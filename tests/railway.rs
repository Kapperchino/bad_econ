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
                cash: 0,
                inventories,
                capacity: match good {
                    Good::Railway => 0,
                    _ => 1,
                },
                technique: match good {
                    Good::Railway => Technique::railway(),
                    _ => Technique::new(60, u32::from(good.inputs()[0].is_some()), 1).unwrap(),
                },
                wage: 1,
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
    assert_eq!(state.get(Field::Cash, 0), 1000 - quantity * 17);
    assert_eq!(state.get(Field::Cash, 4), quantity * 10);
    assert_eq!(state.get(Field::Cash, 5), quantity * 2);
    assert_eq!(state.get(Field::Cash, 6), quantity * 5);
    assert_eq!(state.get(Field::BusinessRevenue, 5), quantity * 7);
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
        assert_eq!(second.get(Field::Used, firm), first.get(Field::Used, firm));
    }
    assert!(second.get(Field::Bread, 0) >= shipped - 1);
    simulation.checkpoint()?.restore()?;
    Ok(())
}

#[test]
fn rail_fares_limit_affordability_and_reverse_routes_pay_the_other_owner_on_gpu() -> Result<()> {
    for reverse in [false, true] {
        let mut config = network()?;
        let population = config.population.as_mut().unwrap();
        population.households[0].cash = match reverse {
            false => 17,
            true => 16,
        };
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
    assert_eq!(state.get(Field::Cash, 0), 1000 - quantity * 12);
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
fn parallel_freight_receipts_preserve_wide_cash_and_workgroup_replay_on_gpu() -> Result<()> {
    let mut config = network()?;
    config.household_need = 100;
    config.coverage = 3;
    config.railway.length_unit = 1;
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
    assert_eq!(state.get(Field::BusinessRevenue, 5), shipped * 1_400_000);
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

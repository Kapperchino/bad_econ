#![cfg(feature = "compute")]
use super::support;
use anyhow::Result;
use bad_econ::{
    domain::{railway::*, *},
    simulation::Simulation,
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
        connections: Vec::new(),
        sharing_agreements: Vec::new(),
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
                sharing: RailSharing::Shared,
            },
            RailwayCompany {
                firm: 12,
                freight_rate: 3,
                access_fee: 5,
                sharing: RailSharing::Shared,
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
    let mut replay = Simulation::from_checkpoint(simulation.checkpoint()?, support::device(64)?)?;
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
fn private_tracks_require_a_directional_sharing_agreement_on_gpu() -> Result<()> {
    for agreements in [
        Vec::new(),
        vec![RailAgreement {
            owner: 12,
            carrier: 11,
        }],
        vec![RailAgreement {
            owner: 11,
            carrier: 12,
        }],
    ] {
        let shared = agreements
            .first()
            .is_some_and(|agreement| agreement.owner == 12);
        let mut config = network()?;
        config.railway.companies[1].sharing = RailSharing::Private;
        config.railway.sharing_agreements = agreements;
        let scenario = Scenario::new(config)?;
        let mut simulation = Simulation::new(scenario.clone(), support::device(64)?)?;
        simulation.advance()?;
        let state = simulation.download()?;
        let quantity = state.get(Field::Purchased, 0);
        assert!(quantity > 0);
        assert_eq!(
            delivered(&state, &scenario, 1),
            if shared { quantity } else { 0 }
        );
        assert_eq!(
            state.get(Field::RailRevenue, 6),
            if shared { quantity * 5 } else { 0 }
        );
        simulation.checkpoint()?.restore()?;
    }
    Ok(())
}

#[test]
fn concurrent_shared_freight_preserves_capacity_payments_and_checkpoints_on_gpu() -> Result<()> {
    struct FreightLoad {
        capacity: u32,
        budget: u64,
        delivered: u64,
    }
    for load in [
        FreightLoad {
            capacity: 7,
            budget: 10_000,
            delivered: 7,
        },
        FreightLoad {
            capacity: 1000,
            budget: 10_000,
            delivered: 261,
        },
        FreightLoad {
            capacity: 1000,
            budget: 49,
            delivered: 7,
        },
    ] {
        let mut config = network()?;
        config.railway.units_per_level = load.capacity;
        config.population.as_mut().unwrap().firms[0].cash = load.budget;
        let population = config.population.as_mut().unwrap();
        let template = population.households[0].clone();
        for household in &mut population.households {
            household.cash = 10;
            household.inventories = [0; Good::COUNT];
        }
        population
            .households
            .extend((0..257).map(|index| HouseholdSeed {
                identity: 100 + index,
                cash: 10,
                ..template.clone()
            }));
        config.households = population.households.len() as u32;
        let households = config.households as usize;
        let scenario = Scenario::new(config)?;
        let mut simulation = Simulation::new(scenario.clone(), support::device(64)?)?;
        simulation.advance()?;
        let state = simulation.download()?;
        let rail_units = (0..households)
            .map(|buyer| {
                state.values[state
                    .shipment_coordinate(Good::Bread, 1, scenario.ring(), buyer)
                    .0 as usize]
            })
            .sum::<u64>();
        assert_eq!(rail_units, load.delivered);
        assert_eq!(
            state.get(Field::RailRevenue, households + 1),
            rail_units * 2
        );
        assert_eq!(
            state.get(Field::RailRevenue, households + 2),
            rail_units * 5
        );
        assert_eq!(
            state.get(Field::Railway, households + 2),
            u64::from(load.capacity) - rail_units
        );
        simulation.checkpoint()?.restore()?;
    }
    Ok(())
}

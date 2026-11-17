use super::support;
use anyhow::Result;
use bad_econ::domain::*;

fn scenario() -> Result<Scenario> {
    let mut config = Scenario::baseline(3, 19, Experiment::Baseline)?
        .config()
        .clone();
    config.distribution_rate = 0;
    config.behavior = BehaviorPolicy {
        work_fatigue: 13,
        rest_recovery: 17,
        wage_increment: 0,
        price_response: 0,
    };
    config.resources = ResourceMap::Explicit {
        regions: vec![ResourceRegion::new(
            Good::Grain,
            Position { x: 0, y: 0 },
            Position {
                x: config.map_size,
                y: config.map_size,
            },
            config.map_size,
        )?],
    };
    config.population = Some(Population {
        households: [901, 101, 303]
            .into_iter()
            .map(|identity| HouseholdSeed {
                identity,
                position: Position { x: 0, y: 0 },
                cash: 1000,
                inventories: Good::food([0, 0, 10]),
                fatigue: 0,
            })
            .collect(),
        firms: vec![
            FirmSeed {
                identity: 8000,
                owner: 303,
                good: Good::Bread,
                cash: 10_000,
                position: Position { x: 0, y: 0 },
                inventories: Good::food([0, 3, 0]),
                capacity: 1,
                technique: Technique::new(240, 3, 2)?,
                wage: 3,
                price: 30,
            },
            FirmSeed {
                identity: 2000,
                owner: 901,
                good: Good::Grain,
                cash: 10_000,
                position: Position { x: 0, y: 0 },
                inventories: Good::food([0, 0, 0]),
                capacity: 1,
                technique: Technique::new(120, 0, 5)?,
                wage: 1,
                price: 10,
            },
            FirmSeed {
                identity: 5000,
                owner: 101,
                good: Good::Flour,
                cash: 10_000,
                position: Position { x: 0, y: 0 },
                inventories: Good::food([2, 0, 0]),
                capacity: 1,
                technique: Technique::new(180, 2, 3)?,
                wage: 2,
                price: 20,
            },
        ],
    });
    Scenario::new(config)
}

#[test]
fn actor_rows_round_trip_on_gpu_and_reject_mismatched_lengths() -> Result<()> {
    use bad_econ::simulation::Simulation;
    let scenario = scenario()?;
    let expected = scenario.initial()?;
    let device = support::device(64)?;
    let mut simulation = Simulation::new(scenario.clone(), device.clone())?;
    assert_eq!(simulation.download()?, expected);
    let encoded = serde_json::to_value(&expected)?;
    assert_eq!(
        encoded["actors"]["identity"],
        serde_json::json!([101, 303, 901, 2000, 5000, 8000])
    );
    for field in ["identity", "x", "y", "kind", "owner", "capacity"] {
        let mut invalid = encoded.clone();
        invalid["actors"][field].as_array_mut().unwrap().pop();
        assert!(serde_json::from_value::<State>(invalid).is_err());
    }
    for actor in expected.actors.iter() {
        let request = InspectionRequest::new(actor.identity, &scenario)?;
        assert_eq!(simulation.inspect(request.clone())?.actor, actor);
        assert_eq!(
            Inspection::from_state(&scenario, &expected, &request)?.actor,
            actor
        );
    }
    let checkpoint = simulation.checkpoint()?;
    let mut old = serde_json::to_value(&checkpoint)?;
    old["version"] = serde_json::json!(2);
    assert!(serde_json::from_value::<Checkpoint>(old).is_err());
    let check_rows = |rows: &ActorRows| -> Result<()> {
        let encoded = serde_json::to_value(rows)?;
        assert_eq!(encoded.as_object().unwrap().len(), 6);
        let decoded: ActorRows = serde_json::from_value(encoded)?;
        assert_eq!(&decoded, rows);
        assert_eq!(rows.iter().collect::<ActorRows>(), *rows);
        let words = |rows: &ActorRows| {
            rows.iter()
                .flat_map(|actor| {
                    [
                        actor.identity as u32,
                        (actor.identity >> 32) as u32,
                        actor.position.x,
                        actor.position.y,
                        actor.kind,
                        actor.owner,
                        actor.capacity,
                    ]
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(device.read(&device.tensor(&words(rows)))?, words(&decoded));
        Ok(())
    };
    let mut rows = expected.actors.clone();
    for (index, kind) in [
        (0, Good::Bread as u32),
        (3, 0),
        (0, 0),
        (4, 0),
        (5, 0),
        (2, Good::Bread as u32),
        (2, 0),
        (5, Good::Bread as u32),
        (4, Good::Bread as u32),
        (4, 0),
        (5, 0),
    ] {
        let mut actor = rows.at(index);
        actor.kind = kind;
        rows.set(index, actor);
        check_rows(&rows)?;
    }
    for kind in [0, Good::Bread as u32, 0, Good::Tools as u32] {
        let actor = Actor {
            identity: 10_000 + rows.len() as u64,
            position: Position { x: 1, y: 2 },
            kind,
            owner: 0,
            capacity: 1,
        };
        rows = rows.iter().chain(std::iter::once(actor)).collect();
        check_rows(&rows)?;
    }
    Ok(())
}

use super::*;
use crate::{
    domain::*,
    simulation::Device,
    viewer::{
        Chart, Dashboard,
        presentation::PresentationPlugin,
        ui::{FirmMarker, MapMarkers},
    },
};
use bevy::prelude::{App, Assets, ColorMaterial, Image, Mesh, MinimalPlugins};
use std::collections::VecDeque;

#[test]
fn closed_firms_leave_the_map_navigation_and_inspector_on_gpu() -> Result<()> {
    let mut config = Scenario::baseline(3, 91, Experiment::Baseline)?
        .config()
        .clone();
    config.entrepreneurship.interested_per_mille = 0;
    config.distribution_rate = 0;
    config.household_production = enterprise::HouseholdProduction::industrial();
    config.population = Some(Population {
        households: (1..=3)
            .map(|identity| HouseholdSeed {
                identity,
                position: Position { x: 0, y: 0 },
                cash: 5000,
                inventories: Good::food([0, 0, 100]),
                fatigue: 0,
            })
            .collect(),
        firms: (4..=6)
            .map(|identity| FirmSeed {
                identity,
                position: Position { x: 900, y: 0 },
                owner: 1,
                good: Good::Bread,
                cash: 0,
                inventories: [0; Good::COUNT],
                capacity: 1,
                technique: Technique {
                    labor_minutes: 60,
                    input_ticks: 1,
                    output_ticks: 1,
                },
                wage: 1,
                price: 10,
            })
            .collect(),
    });
    let scenario = Scenario::new(config)?;
    let mut state = scenario.initial()?;
    state.day = 26;
    state.set(Field::EnterpriseStatus, 3, EnterpriseState::Dormant as u64);
    state.set(Field::IdleDays, 3, 89);
    state.set(Field::EnterpriseStatus, 4, EnterpriseState::Forming as u64);
    let mut simulation =
        Simulation::from_checkpoint(Checkpoint::new(&scenario, &state)?, Device::headless(64)?)?;
    let publisher = Publisher {
        data: Arc::new(Mutex::new(Display {
            snapshot: ObservationSnapshot::default(),
            history: VecDeque::new(),
            comparison: None,
            inspection: None,
            firms: Vec::new(),
            identities: Vec::new(),
            status: DisplayState::Initializing,
            error: None,
            seed: scenario.config().seed,
            map_size: scenario.config().map_size,
            speed: Speed::Normal,
            revision: 0,
            observation_revision: 0,
            map_revision: 0,
            cues: VecDeque::new(),
        })),
        wake: || (),
    };
    let mut app = App::new();
    let (sender, _receiver) = crossbeam_channel::unbounded();
    app.insert_resource(Dashboard {
        sender,
        data: publisher.data.clone(),
        selected: 4,
        chart: Chart::Businesses,
        period: super::super::model::Period::default(),
        series: super::super::model::SeriesSelection::All,
    });
    app.add_plugins((MinimalPlugins, PresentationPlugin));
    app.init_resource::<Assets<Image>>()
        .init_resource::<Assets<Mesh>>()
        .init_resource::<Assets<ColorMaterial>>();
    app.world_mut().spawn(MapMarkers);
    publish(&mut simulation, &publisher, Publication::Restart, None)?;
    let request = InspectionRequest::new(4, &scenario)?;
    inspect(&mut simulation, &publisher, request.clone())?;
    app.update();
    assert_eq!(marker_identities(&mut app), vec![4, 5, 6]);
    assert_eq!(
        publisher
            .data
            .lock()
            .unwrap()
            .inspection
            .as_ref()
            .unwrap()
            .enterprise_state,
        Some(EnterpriseState::Dormant)
    );

    simulation.advance()?;
    assert_eq!(
        simulation.inspect(request.clone())?.enterprise_state,
        Some(EnterpriseState::Closed)
    );
    assert_eq!(simulation.actors().len(), 6);
    publish(&mut simulation, &publisher, Publication::Snapshot, None)?;
    app.update();
    assert_eq!(marker_identities(&mut app), vec![5, 6]);
    {
        let display = publisher.data.lock().unwrap();
        assert_eq!(display.identities, vec![1, 2, 3, 5, 6]);
        assert!(InspectionRequest::from_registry(4, &display.identities).is_err());
        assert!(display.inspection.is_none());
        assert_eq!(display.snapshot.firms, 2);
        assert_eq!(display.history.back().unwrap().firms, 2);
        assert_eq!(display.snapshot.firm_closures, 1);
    }
    inspect(&mut simulation, &publisher, request)?;
    assert!(publisher.data.lock().unwrap().inspection.is_none());

    let readback_bytes = simulation.readback_bytes;
    let map_revision = publisher.data.lock().unwrap().map_revision;
    let observation_revision = publisher.data.lock().unwrap().observation_revision;
    publish(&mut simulation, &publisher, Publication::Snapshot, None)?;
    assert_eq!(simulation.readback_bytes, readback_bytes);
    assert_eq!(publisher.data.lock().unwrap().map_revision, map_revision);
    assert_eq!(
        publisher.data.lock().unwrap().observation_revision,
        observation_revision
    );
    assert_eq!(publisher.data.lock().unwrap().snapshot.firms, 2);

    let checkpoint = simulation.checkpoint()?;
    simulation.command(SimulationCommand::Restore(Box::new(checkpoint)))?;
    publish(&mut simulation, &publisher, Publication::Restart, None)?;
    app.update();
    assert_eq!(marker_identities(&mut app), vec![5, 6]);

    simulation.command(SimulationCommand::Reset(scenario.into()))?;
    publish(&mut simulation, &publisher, Publication::Restart, None)?;
    app.update();
    assert_eq!(marker_identities(&mut app), vec![4, 5, 6]);
    assert_eq!(
        publisher.data.lock().unwrap().identities,
        vec![1, 2, 3, 4, 5, 6]
    );
    assert_eq!(publisher.data.lock().unwrap().snapshot.firms, 3);
    Ok(())
}

fn marker_identities(app: &mut App) -> Vec<u64> {
    let world = app.world_mut();
    let mut identities = world
        .query::<&FirmMarker>()
        .iter(world)
        .map(|marker| marker.identity)
        .collect::<Vec<_>>();
    identities.sort_unstable();
    identities
}

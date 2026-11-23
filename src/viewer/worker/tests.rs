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

#[test]
fn comparing_seeds_preserves_history_and_starts_a_new_run_on_gpu() -> Result<()> {
    let scenario = Scenario::baseline(8, 42, Experiment::Baseline)?;
    let mut simulation = Simulation::new(scenario.clone(), Device::headless(64)?)?;
    let publisher = Publisher {
        data: Arc::new(Mutex::new(Display::new(&scenario))),
        wake: || (),
    };
    publish(&mut simulation, &publisher, Publication::Restart, None)?;
    simulation.advance()?;
    publish(&mut simulation, &publisher, Publication::Snapshot, None)?;
    let history = publisher.data.lock().unwrap().history.clone();

    assert_eq!(
        compare(&mut simulation, &publisher)?,
        RunTarget::UntilDay(1)
    );
    {
        let display = publisher.data.lock().unwrap();
        let reference = display.comparison.as_ref().unwrap();
        assert_eq!(reference.seed, 42);
        assert_eq!(reference.horizon, 1);
        assert_eq!(reference.history, history);
        assert_eq!(display.seed, 43);
        assert_eq!(display.history.len(), 1);
        assert_eq!(display.history[0].day, 0);
    }
    simulation.command(SimulationCommand::Reset(scenario.into()))?;
    publish(&mut simulation, &publisher, Publication::Restart, None)?;
    assert!(publisher.data.lock().unwrap().comparison.is_none());
    Ok(())
}

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
        data: Arc::new(Mutex::new(Display::new(&scenario))),
        wake: || (),
    };
    let mut app = App::new();
    let (sender, _receiver) = crossbeam_channel::unbounded();
    app.insert_resource(Dashboard {
        sender,
        data: publisher.data.clone(),
        selected: 4,
        chart: Chart::Businesses,
        company_page: None,
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

#[test]
fn company_dashboard_ranks_pages_and_opens_site_accounts_on_gpu() -> Result<()> {
    use crate::viewer::{
        Launch,
        companies::{CompaniesPlugin, Direction, Ranking, panel},
        controls,
        model::{Menu, Period, SeriesSelection},
        ui::{ActorSelection, UiAction},
    };
    use bevy::prelude::{ButtonInput, Commands, KeyCode, Node, Startup, Text, Update};

    let mut config = Scenario::baseline(8, 2, Experiment::Concentrated)?
        .config()
        .clone();
    config.firms = 8;
    let scenario = Scenario::new(config)?;
    let mut state = scenario.initial()?;
    state.day = 1;
    for site in state.actors.firm_indices().collect::<Vec<_>>() {
        state.set(Field::BusinessOpeningCash, site, 11);
        state.set(Field::BusinessClosingCash, site, site as u64);
    }
    let mut sim =
        Simulation::from_checkpoint(Checkpoint::new(&scenario, &state)?, Device::headless(64)?)?;
    let publisher = Publisher {
        data: Arc::new(Mutex::new(Display::new(&scenario))),
        wake: || (),
    };
    publish(&mut sim, &publisher, Publication::Restart, None)?;
    let (sender, receiver) = crossbeam_channel::unbounded();
    let mut app = App::new();
    app.add_plugins((MinimalPlugins, CompaniesPlugin))
        .add_message::<UiAction>()
        .init_resource::<ButtonInput<KeyCode>>()
        .insert_resource(Launch {
            scenario,
            workgroup: 64,
        })
        .insert_resource(Dashboard {
            sender,
            data: publisher.data.clone(),
            selected: 1,
            chart: Chart::Employment,
            company_page: None,
            period: Period::default(),
            series: SeriesSelection::All,
        })
        .add_systems(Update, controls);
    app.add_systems(Startup, |mut commands: Commands| {
        commands.spawn(Node::default()).with_children(panel);
    });
    let show = |app: &mut App, action| {
        app.world_mut().write_message(action);
        app.update();
        let world = app.world_mut();
        world
            .query::<&Text>()
            .iter(world)
            .map(|label| label.0.clone())
            .collect::<Vec<_>>()
    };
    show(&mut app, UiAction::Companies);
    let first = show(&mut app, UiAction::Rank(Ranking::Profit));
    assert!(first.iter().any(|label| label.starts_with("Company #16 ")));
    assert!(!first.iter().any(|label| label.starts_with("Company #9 ")));
    let second = show(&mut app, UiAction::CompanyPage(Direction::Next));
    assert!(
        second
            .iter()
            .any(|label| label == "Day 1  ·  Companies 7–8 of 8")
    );
    assert!(second.iter().any(|label| label.starts_with("Company #9 ")));
    let losses = show(&mut app, UiAction::Rank(Ranking::Loss));
    assert!(losses.iter().any(|label| label == "Loss  -3"));
    assert!(
        losses
            .iter()
            .any(|label| label == "Day 1  ·  Companies 1–6 of 8")
    );
    show(&mut app, UiAction::Inspect(ActorSelection::Identity(9)));
    let request = match receiver.try_recv()? {
        Control::Inspect(request) => Ok(request),
        _ => Err(anyhow::anyhow!("expected a site inspection")),
    }?;
    inspect(&mut sim, &publisher, request)?;
    assert_eq!(
        publisher
            .data
            .lock()
            .unwrap()
            .inspection
            .as_ref()
            .unwrap()
            .actor
            .identity,
        9
    );
    show(&mut app, UiAction::Menu(Menu::Overview));
    assert!(app.world().resource::<Dashboard>().company_page.is_none());
    Ok(())
}

#[test]
#[ignore = "release benchmark for viewer day advancement"]
fn benchmark_million_people_viewer_days_on_gpu() -> Result<()> {
    let scenario = Scenario::for_people(
        1_000_000,
        CohortSizes::default(),
        Economy::Industrial,
        42,
        Experiment::Baseline,
    )?;
    let publisher = Publisher {
        data: Arc::new(Mutex::new(Display::new(&scenario))),
        wake: || (),
    };
    let mut simulation = Simulation::new(scenario, Device::headless(64)?)?;
    publish(&mut simulation, &publisher, Publication::Restart, None)?;
    let mut measured = Vec::new();
    for day in 1..=62 {
        let start = Instant::now();
        simulation.command(SimulationCommand::AdvanceDay)?;
        let advanced = start.elapsed();
        publish(
            &mut simulation,
            &publisher,
            Publication::Snapshot,
            Some(Cue::Stepped),
        )?;
        let elapsed = start.elapsed();
        eprintln!(
            "viewer day {day} {:.6} publish {:.6}",
            elapsed.as_secs_f64(),
            (elapsed - advanced).as_secs_f64(),
        );
        if day > 2 {
            measured.push(elapsed);
        }
        assert_eq!(publisher.data.lock().unwrap().snapshot.day, day);
    }
    measured.sort_unstable();
    let seconds = measured.iter().sum::<std::time::Duration>().as_secs_f64();
    let median = (measured[29] + measured[30]) / 2;
    let p90 = measured[53];
    let digest = simulation.download()?.digest();
    eprintln!("60 viewer days in {seconds:.3}s, median {median:?}, p90 {p90:?}, state {digest}");
    Ok(())
}

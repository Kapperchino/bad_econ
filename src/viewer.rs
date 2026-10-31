use crate::{
    domain::*,
    runtime::{RuntimeState, SimulationCommand},
    simulation::{Device, Simulation},
};
use anyhow::Result;
use bevy::{
    prelude::*,
    render::{
        RenderApp, RenderPlugin,
        renderer::{RenderAdapter, RenderDevice, RenderInstance, RenderQueue},
        settings::{RenderCreation, WgpuSettings},
    },
    sprite::Anchor,
};
use std::{
    collections::VecDeque,
    sync::{Arc, Mutex, mpsc},
    time::{Duration, Instant},
};

struct SharedGpu;
impl Plugin for SharedGpu {
    fn build(&self, _: &mut App) {}
    fn finish(&self, app: &mut App) {
        let instance = app
            .sub_app(RenderApp)
            .world()
            .resource::<RenderInstance>()
            .clone();
        app.insert_resource(instance);
    }
}

#[derive(Resource)]
struct Launch {
    scenario: Scenario,
    workgroup: u32,
}
#[derive(Clone, Copy)]
pub enum ViewMode {
    Interactive,
    Verify,
}
#[derive(Resource)]
struct Verification {
    state: VerificationState,
    deadline: Instant,
}
enum VerificationState {
    Start,
    AwaitDay,
    AwaitInspection,
    Frames(u32),
    Complete,
}
#[derive(Resource)]
struct Dashboard {
    sender: mpsc::Sender<Control>,
    data: Arc<Mutex<Display>>,
    selected: u64,
}
struct Display {
    snapshot: ObservationSnapshot,
    history: VecDeque<ObservationSnapshot>,
    comparison: Option<ObservationSnapshot>,
    inspection: Option<Inspection>,
    firms: Vec<Actor>,
    status: String,
    map_size: u32,
    speed: Speed,
}
enum Control {
    Simulation(Box<SimulationCommand>),
    Inspect(InspectionRequest),
    Speed(Speed),
    Compare,
}
impl Control {
    fn command(command: SimulationCommand) -> Self {
        Self::Simulation(Box::new(command))
    }
}
#[derive(Clone, Copy)]
enum Speed {
    Slow,
    Normal,
    Fast,
}
impl Speed {
    fn period(self) -> Duration {
        match self {
            Self::Slow => Duration::from_secs(2),
            Self::Normal => Duration::from_millis(500),
            Self::Fast => Duration::ZERO,
        }
    }
    fn label(self) -> &'static str {
        match self {
            Self::Slow => "0.5 days/s",
            Self::Normal => "2 days/s",
            Self::Fast => "maximum",
        }
    }
}
enum WorkerState {
    Active,
    Stopped,
}
#[derive(Component)]
struct Statistics;
#[derive(Component)]
struct Region(usize);

pub fn run(scenario: Scenario, workgroup: u32, mode: ViewMode) -> Result<()> {
    let settings = WgpuSettings {
        limits: burn_wgpu::wgpu::Limits {
            max_storage_buffer_binding_size: 1024 * 1024 * 1024,
            max_buffer_size: 2 * 1024 * 1024 * 1024,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut app = App::new();
    app.insert_resource(Launch {
        scenario,
        workgroup,
    })
    .insert_resource(ClearColor(Color::srgb(0.035, 0.045, 0.07)))
    .add_plugins(
        DefaultPlugins
            .set(WindowPlugin {
                primary_window: Some(Window {
                    title: "Spatial economy · grain / flour / bread".into(),
                    resolution: (1280, 840).into(),
                    ..Default::default()
                }),
                ..Default::default()
            })
            .set(RenderPlugin {
                render_creation: RenderCreation::Automatic(Box::new(settings)),
                ..Default::default()
            }),
    )
    .add_plugins(SharedGpu)
    .add_systems(Startup, setup)
    .add_systems(Update, (controls, draw));
    if let ViewMode::Verify = mode {
        app.insert_resource(Verification {
            state: VerificationState::Start,
            deadline: Instant::now() + Duration::from_secs(60),
        })
        .add_systems(Update, verify);
    }
    match app.run() {
        AppExit::Success => Ok(()),
        AppExit::Error(code) => Err(anyhow::anyhow!("viewer exited with {code}")),
    }
}

fn setup(
    mut commands: Commands,
    launch: Res<Launch>,
    instance: Res<RenderInstance>,
    adapter: Res<RenderAdapter>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    commands.spawn(Camera2d);
    let setup = burn_wgpu::WgpuSetup {
        instance: instance.as_ref().clone().into_inner(),
        adapter: adapter.as_ref().clone().into_inner(),
        device: device.wgpu_device().clone(),
        queue: queue.as_ref().clone().into_inner(),
        backend: adapter.get_info().backend,
    };
    let data = Arc::new(Mutex::new(Display {
        snapshot: ObservationSnapshot::default(),
        history: VecDeque::new(),
        comparison: None,
        inspection: None,
        firms: Vec::new(),
        status: "Initializing shared GPU…".into(),
        map_size: launch.scenario.config().map_size,
        speed: Speed::Normal,
    }));
    let (sender, receiver) = mpsc::channel();
    commands.insert_resource(Dashboard {
        sender,
        data: data.clone(),
        selected: 1,
    });
    let scenario = launch.scenario.clone();
    let workgroup = launch.workgroup;
    std::thread::spawn(move || {
        let result = Device::import(setup, workgroup)
            .and_then(|device| Simulation::new(scenario, device))
            .and_then(|simulation| worker(simulation, receiver, &data));
        match result {
            Ok(()) => (),
            Err(error) => {
                data.lock().expect("dashboard lock").status = format!("Fault: {error}");
            }
        }
    });
    (0..256).for_each(|i| {
        commands.spawn((
            Region(i),
            Sprite::from_color(Color::srgb(0.08, 0.16, 0.19), Vec2::splat(37.0)),
            Transform::from_xyz(
                -580.0 + (i % 16) as f32 * 39.0,
                305.0 - (i / 16) as f32 * 39.0,
                0.0,
            ),
        ));
    });
    commands.spawn((
        Text2d::new("SPATIAL ECONOMY"),
        TextFont {
            font_size: FontSize::Px(30.0),
            ..Default::default()
        },
        TextColor(Color::srgb(0.85, 0.92, 0.95)),
        Anchor::TOP_LEFT,
        Transform::from_xyz(-600.0, 390.0, 2.0),
    ));
    commands.spawn((
        Text2d::new("Regional shortages · amber grain / blue flour / green bread"),
        TextFont {
            font_size: FontSize::Px(15.0),
            ..Default::default()
        },
        Anchor::TOP_LEFT,
        Transform::from_xyz(-600.0, 345.0, 2.0),
    ));
    commands.spawn((
        Statistics,
        Text2d::new("Initializing…"),
        TextFont {
            font_size: FontSize::Px(17.0),
            ..Default::default()
        },
        Anchor::TOP_LEFT,
        Transform::from_xyz(75.0, 375.0, 2.0),
    ));
    commands.spawn((Text2d::new("SPACE pause/resume    N step    R reset    1/2/3 speed    H household    F firm    TAB next    C compare next seed"), TextFont { font_size: FontSize::Px(15.0), ..Default::default() }, Anchor::TOP_LEFT, Transform::from_xyz(-600.0, -365.0, 2.0)));
}

fn worker(
    mut simulation: Simulation,
    receiver: mpsc::Receiver<Control>,
    data: &Arc<Mutex<Display>>,
) -> Result<()> {
    let mut speed = Speed::Normal;
    let mut workflow = WorkerState::Active;
    let mut next_day = Instant::now();
    publish(&mut simulation, data)?;
    while matches!(workflow, WorkerState::Active) {
        let timeout = match simulation.status {
            RuntimeState::Running => next_day
                .saturating_duration_since(Instant::now())
                .min(Duration::from_millis(50)),
            _ => Duration::from_millis(50),
        };
        let result = match receiver.recv_timeout(timeout) {
            Ok(Control::Simulation(command)) => simulation
                .command(*command)
                .and_then(|()| publish(&mut simulation, data)),
            Ok(Control::Inspect(request)) => simulation.inspect(request).map(|inspection| {
                data.lock().expect("dashboard lock").inspection = Some(inspection)
            }),
            Ok(Control::Speed(value)) => {
                speed = value;
                data.lock().expect("dashboard lock").speed = speed;
                next_day = Instant::now();
                Ok(())
            }
            Ok(Control::Compare) => compare(&mut simulation, data),
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                workflow = WorkerState::Stopped;
                Ok(())
            }
            Err(mpsc::RecvTimeoutError::Timeout) => Ok(()),
        };
        if let Err(error) = result {
            data.lock().expect("dashboard lock").status =
                format!("{:?}: {error}", simulation.status);
        }
        if let (WorkerState::Active, RuntimeState::Running, true) =
            (&workflow, &simulation.status, Instant::now() >= next_day)
        {
            let result = simulation
                .advance()
                .and_then(|()| publish(&mut simulation, data));
            if let Err(error) = result {
                data.lock().expect("dashboard lock").status =
                    format!("{:?}: {error}", simulation.status);
            }
            next_day = Instant::now() + speed.period();
        }
    }
    Ok(())
}

fn compare(simulation: &mut Simulation, data: &Arc<Mutex<Display>>) -> Result<()> {
    let mut config = simulation.scenario.config().clone();
    config.seed = config
        .seed
        .checked_add(1)
        .ok_or_else(|| anyhow::anyhow!("seed overflow"))?;
    let scenario = Scenario::new(config)?;
    data.lock().expect("dashboard lock").comparison = Some(simulation.snapshot.clone());
    simulation.command(SimulationCommand::Reset(scenario))?;
    publish(simulation, data)
}

fn verify(
    mut verification: ResMut<Verification>,
    dashboard: Option<Res<Dashboard>>,
    launch: Res<Launch>,
    mut exit: MessageWriter<AppExit>,
) {
    if let Some(dashboard) = dashboard {
        let display = dashboard.data.lock().expect("dashboard lock");
        match (
            Instant::now() >= verification.deadline,
            display.status.starts_with("Fault"),
        ) {
            (true, _) | (_, true) => {
                eprintln!("viewer verification failed: {}", display.status);
                exit.write(AppExit::error());
                verification.state = VerificationState::Complete;
            }
            _ => {
                verification.state = match verification.state {
                    VerificationState::Start => {
                        let _ = dashboard
                            .sender
                            .send(Control::command(SimulationCommand::AdvanceDay));
                        VerificationState::AwaitDay
                    }
                    VerificationState::AwaitDay if display.snapshot.day == 1 => {
                        let request =
                            InspectionRequest::new(1, &launch.scenario).expect("first household");
                        let _ = dashboard.sender.send(Control::Inspect(request));
                        VerificationState::AwaitInspection
                    }
                    VerificationState::AwaitInspection if display.inspection.is_some() => {
                        VerificationState::Frames(3)
                    }
                    VerificationState::Frames(0) => {
                        eprintln!(
                            "viewer verification passed: shared device, audited day, inspection, rendered frames"
                        );
                        exit.write(AppExit::Success);
                        VerificationState::Complete
                    }
                    VerificationState::Frames(frames) => VerificationState::Frames(frames - 1),
                    ref state => match state {
                        VerificationState::AwaitDay => VerificationState::AwaitDay,
                        VerificationState::AwaitInspection => VerificationState::AwaitInspection,
                        _ => VerificationState::Complete,
                    },
                };
            }
        }
    }
}

fn publish(simulation: &mut Simulation, data: &Arc<Mutex<Display>>) -> Result<()> {
    let firms = match simulation.snapshot.day {
        0 => Some(
            simulation
                .actors()
                .iter()
                .filter(|actor| actor.kind > 0)
                .cloned()
                .collect(),
        ),
        _ => None,
    };
    let mut display = data.lock().expect("dashboard lock");
    match display.history.back().map(|s| s.day) {
        Some(day) if day == simulation.snapshot.day => (),
        _ => {
            display.history.push_back(simulation.snapshot.clone());
            if display.history.len() > 160 {
                display.history.pop_front();
            }
        }
    }
    if let Some(firms) = firms {
        display.firms = firms;
        display.history.clear();
        display.inspection = None;
    }
    display.snapshot = simulation.snapshot.clone();
    display.status = format!(
        "{:?} · seed {}",
        simulation.status,
        simulation.scenario.config().seed
    );
    Ok(())
}

fn controls(
    keys: Res<ButtonInput<KeyCode>>,
    launch: Res<Launch>,
    dashboard: Option<ResMut<Dashboard>>,
) {
    match dashboard {
        None => (),
        Some(mut dashboard) => {
            let running = dashboard
                .data
                .lock()
                .expect("dashboard lock")
                .status
                .starts_with("Running");
            let mut controls = Vec::new();
            if keys.just_pressed(KeyCode::Space) {
                controls.push(Control::command(match running {
                    true => SimulationCommand::Pause,
                    false => SimulationCommand::Resume,
                }));
            }
            if keys.just_pressed(KeyCode::KeyN) {
                controls.push(Control::command(SimulationCommand::AdvanceDay));
            }
            if keys.just_pressed(KeyCode::KeyR) {
                controls.push(Control::command(SimulationCommand::Reset(
                    launch.scenario.clone(),
                )));
            }
            if keys.just_pressed(KeyCode::KeyC) {
                controls.push(Control::Compare);
            }
            [
                (KeyCode::Digit1, Speed::Slow),
                (KeyCode::Digit2, Speed::Normal),
                (KeyCode::Digit3, Speed::Fast),
            ]
            .into_iter()
            .filter(|(key, _)| keys.just_pressed(*key))
            .for_each(|(_, speed)| controls.push(Control::Speed(speed)));
            if keys.just_pressed(KeyCode::KeyH) {
                dashboard.selected = 1;
            }
            if keys.just_pressed(KeyCode::KeyF) {
                dashboard.selected = u64::from(launch.scenario.config().households) + 1;
            }
            if keys.just_pressed(KeyCode::Tab) {
                dashboard.selected = dashboard.selected % launch.scenario.actors() as u64 + 1;
            }
            let inspection = [KeyCode::KeyH, KeyCode::KeyF, KeyCode::Tab]
                .into_iter()
                .any(|key| keys.just_pressed(key))
                .then(|| InspectionRequest::new(dashboard.selected, &launch.scenario));
            if let Some(Ok(request)) = inspection {
                controls.push(Control::Inspect(request));
            }
            controls.into_iter().for_each(|control| {
                let _ = dashboard.sender.send(control);
            });
        }
    }
}

fn draw(
    dashboard: Option<Res<Dashboard>>,
    mut labels: Query<&mut Text2d, With<Statistics>>,
    mut regions: Query<(&Region, &mut Sprite)>,
    mut gizmos: Gizmos,
) {
    match dashboard {
        None => (),
        Some(dashboard) => {
            let display = dashboard.data.lock().expect("dashboard lock");
            let s = &display.snapshot;
            let unemployment = 100.0 * (s.households.saturating_sub(s.employed)) as f64
                / s.households.max(1) as f64;
            let inspection = display
                .inspection
                .as_ref()
                .map(|i| {
                    let detail = match i.actor.kind {
                        0 => format!(
                            "Health {} · fatigue {} · labor {} min\nEmployer {}",
                            i.health, i.fatigue, i.available_minutes, i.employer
                        ),
                        _ => format!(
                            "Owner {} · wage {} · price {}\nCapacity {} · planned {} · open {}",
                            i.actor.owner + 1,
                            i.wage,
                            i.price,
                            i.actor.capacity,
                            i.planned_packets,
                            i.open_slots
                        ),
                    };
                    format!(
                        "Actor {} · day {} · kind {}\nCash {} · stock {:?} · transit {}\n{}",
                        i.actor.identity,
                        i.day,
                        i.actor.kind,
                        i.cash,
                        i.inventory,
                        i.transit,
                        detail
                    )
                })
                .unwrap_or_else(|| "H / F to inspect a household or firm".into());
            let comparison = display
                .comparison
                .as_ref()
                .map(|c| {
                    format!(
                        "Previous seed · day {}\nEmployed {} · shortage {}",
                        c.day, c.employed, c.shortages
                    )
                })
                .unwrap_or_default();
            let text = format!(
                "{}\nDay {} · target {}\n\nUnemployment  {:.1}%\nUnmet bread  {}\nCash  {}\n\nStocks G/F/B  {:?}\nIn transit  {:?}\nWages  {}–{} / minute\nPrices  {}–{}\n\n{}\n\n{}\n\nEmployment (green) / shortages (red)",
                display.status,
                s.day,
                display.speed.label(),
                unemployment,
                s.shortages,
                s.cash,
                s.inventories,
                s.transit,
                s.minimum_wage,
                s.maximum_wage,
                s.minimum_price,
                s.maximum_price,
                inspection,
                comparison
            );
            labels
                .iter_mut()
                .for_each(|mut label| label.0.clone_from(&text));
            let peak = s.regions.iter().copied().max().unwrap_or(1).max(1) as f32;
            regions.iter_mut().for_each(|(region, mut sprite)| {
                let heat = s.regions.get(region.0).copied().unwrap_or(0) as f32 / peak;
                sprite.color =
                    Color::srgb(0.08 + heat * 0.65, 0.16 - heat * 0.08, 0.19 - heat * 0.10);
            });
            display.firms.iter().for_each(|firm| {
                let color = match firm.kind {
                    1 => Color::srgb(0.95, 0.73, 0.25),
                    2 => Color::srgb(0.40, 0.70, 0.95),
                    _ => Color::srgb(0.45, 0.90, 0.65),
                };
                let point = Vec2::new(
                    -599.5 + firm.position.x as f32 / display.map_size as f32 * 624.0,
                    324.5 - firm.position.y as f32 / display.map_size as f32 * 624.0,
                );
                gizmos.circle_2d(point, 2.0, color);
            });
            display
                .history
                .iter()
                .zip(display.history.iter().skip(1))
                .enumerate()
                .for_each(|(i, (a, b))| {
                    let x = 80.0 + i as f32 * 3.0;
                    let h = s.households.max(1) as f32;
                    gizmos.line_2d(
                        Vec2::new(x, -300.0 + a.employed as f32 / h * 90.0),
                        Vec2::new(x + 3.0, -300.0 + b.employed as f32 / h * 90.0),
                        Color::srgb(0.45, 0.90, 0.65),
                    );
                    gizmos.line_2d(
                        Vec2::new(x, -300.0 + a.shortages as f32 / h * 90.0),
                        Vec2::new(x + 3.0, -300.0 + b.shortages as f32 / h * 90.0),
                        Color::srgb(0.95, 0.35, 0.3),
                    );
                });
        }
    }
}

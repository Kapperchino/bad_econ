mod model;
use crate::{
    domain::*,
    runtime::{RuntimeState, SimulationCommand},
    simulation::{Device, Simulation},
};
use anyhow::{Context, Result};
use bevy::{
    prelude::*,
    render::{
        RenderApp, RenderPlugin,
        renderer::{RenderAdapter, RenderDevice, RenderInstance, RenderQueue},
        settings::{RenderCreation, WgpuSettings},
        view::screenshot::{Screenshot, ScreenshotCaptured},
    },
    sprite::Anchor,
};
use model::{Chart, ChartRange, ComparisonRun, DisplayState, RunTarget};
use std::{
    collections::VecDeque,
    io::Write,
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
#[derive(Clone, Copy)]
enum VerificationState {
    Start,
    AwaitDay,
    AwaitInspection,
    AwaitComparison,
    AwaitFirmInspection,
    Frames(u32),
    AwaitCapture,
    Complete,
}
#[derive(Resource)]
struct Dashboard {
    sender: mpsc::Sender<Control>,
    data: Arc<Mutex<Display>>,
    selected: u64,
    chart: Chart,
}
struct Display {
    snapshot: ObservationSnapshot,
    history: VecDeque<ObservationSnapshot>,
    comparison: Option<ComparisonRun>,
    inspection: Option<Inspection>,
    firms: Vec<Actor>,
    status: DisplayState,
    error: Option<String>,
    seed: u64,
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
enum PanelLabel {
    Statistics,
    ChartTitle,
    ChartScale,
}
#[derive(Clone, Copy)]
enum Publication {
    Snapshot,
    Restart,
}
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
                    title: "Spatial economy | grain / flour / bread".into(),
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
        status: DisplayState::Initializing,
        error: None,
        seed: launch.scenario.config().seed,
        map_size: launch.scenario.config().map_size,
        speed: Speed::Normal,
    }));
    let (sender, receiver) = mpsc::channel();
    commands.insert_resource(Dashboard {
        sender,
        data: data.clone(),
        selected: launch.scenario.identity(0).expect("first household"),
        chart: Chart::Employment,
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
                data.lock().expect("dashboard lock").status =
                    DisplayState::Unavailable(error.to_string());
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
        Text2d::new("Regional shortages | amber grain / blue flour / green bread"),
        TextFont {
            font_size: FontSize::Px(15.0),
            ..Default::default()
        },
        Anchor::TOP_LEFT,
        Transform::from_xyz(-600.0, 345.0, 2.0),
    ));
    commands.spawn((
        PanelLabel::Statistics,
        Text2d::new("Initializing..."),
        TextFont {
            font_size: FontSize::Px(17.0),
            ..Default::default()
        },
        Anchor::TOP_LEFT,
        Transform::from_xyz(75.0, 375.0, 2.0),
    ));
    commands.spawn((
        PanelLabel::ChartTitle,
        Text2d::default(),
        TextFont {
            font_size: FontSize::Px(14.0),
            ..Default::default()
        },
        Anchor::TOP_LEFT,
        Transform::from_xyz(75.0, -170.0, 2.0),
    ));
    commands.spawn((
        PanelLabel::ChartScale,
        Text2d::default(),
        TextFont {
            font_size: FontSize::Px(13.0),
            ..Default::default()
        },
        Anchor::TOP_LEFT,
        Transform::from_xyz(75.0, -325.0, 2.0),
    ));
    commands.spawn((Text2d::new("SPACE pause/resume    N step    R reset    1/2/3 speed    H household    F firm    TAB next    C compare    G chart"), TextFont { font_size: FontSize::Px(15.0), ..Default::default() }, Anchor::TOP_LEFT, Transform::from_xyz(-600.0, -365.0, 2.0)));
}

fn worker(
    mut simulation: Simulation,
    receiver: mpsc::Receiver<Control>,
    data: &Arc<Mutex<Display>>,
) -> Result<()> {
    let mut speed = Speed::Normal;
    let mut workflow = WorkerState::Active;
    let mut next_day = Instant::now();
    let mut target = RunTarget::Open;
    publish(&mut simulation, data, Publication::Restart)?;
    while matches!(workflow, WorkerState::Active) {
        let timeout = match simulation.status {
            RuntimeState::Running => next_day
                .saturating_duration_since(Instant::now())
                .min(Duration::from_millis(50)),
            _ => Duration::from_millis(50),
        };
        let result = match receiver.recv_timeout(timeout) {
            Ok(Control::Simulation(command)) => {
                let publication = match &*command {
                    SimulationCommand::Reset(_) | SimulationCommand::Restore(_) => {
                        Publication::Restart
                    }
                    _ => Publication::Snapshot,
                };
                simulation.command(*command).and_then(|()| {
                    if let Publication::Restart = publication {
                        target = RunTarget::Open;
                        data.lock().expect("dashboard lock").comparison = None;
                    }
                    publish(&mut simulation, data, publication)
                })
            }
            Ok(Control::Inspect(request)) => simulation.inspect(request).map(|inspection| {
                data.lock().expect("dashboard lock").inspection = Some(inspection)
            }),
            Ok(Control::Speed(value)) => {
                speed = value;
                data.lock().expect("dashboard lock").speed = speed;
                next_day = Instant::now();
                Ok(())
            }
            Ok(Control::Compare) => compare(&mut simulation, data).map(|next| {
                target = next;
                next_day = Instant::now();
            }),
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                workflow = WorkerState::Stopped;
                Ok(())
            }
            Err(mpsc::RecvTimeoutError::Timeout) => Ok(()),
        };
        if let Err(error) = result {
            let mut display = data.lock().expect("dashboard lock");
            display.error = Some(error.to_string());
            display.status = DisplayState::Active(simulation.status.clone());
        }
        if target.reached(simulation.snapshot.day) {
            simulation.command(SimulationCommand::Pause)?;
            target = RunTarget::Open;
            publish(&mut simulation, data, Publication::Snapshot)?;
        }
        if let (WorkerState::Active, RuntimeState::Running, true) =
            (&workflow, &simulation.status, Instant::now() >= next_day)
        {
            let result = simulation.advance().and_then(|()| {
                if target.reached(simulation.snapshot.day) {
                    simulation.command(SimulationCommand::Pause)?;
                    target = RunTarget::Open;
                }
                publish(&mut simulation, data, Publication::Snapshot)
            });
            if let Err(error) = result {
                let mut display = data.lock().expect("dashboard lock");
                display.error = Some(error.to_string());
                display.status = DisplayState::Active(simulation.status.clone());
            }
            next_day = Instant::now() + speed.period();
        }
    }
    Ok(())
}

fn compare(simulation: &mut Simulation, data: &Arc<Mutex<Display>>) -> Result<RunTarget> {
    let mut config = simulation.scenario.config().clone();
    config.seed = config
        .seed
        .checked_add(1)
        .ok_or_else(|| anyhow::anyhow!("seed overflow"))?;
    let scenario = Scenario::new(config)?;
    let horizon = simulation.snapshot.day;
    let reference = ComparisonRun {
        seed: simulation.scenario.config().seed,
        horizon,
        history: data.lock().expect("dashboard lock").history.clone(),
    };
    simulation.command(SimulationCommand::Reset(scenario))?;
    data.lock().expect("dashboard lock").comparison = Some(reference);
    let target = match horizon {
        0 => RunTarget::Open,
        _ => {
            simulation.command(SimulationCommand::Resume)?;
            RunTarget::UntilDay(horizon)
        }
    };
    publish(simulation, data, Publication::Restart)?;
    Ok(target)
}

fn verify(
    mut verification: ResMut<Verification>,
    dashboard: Option<Res<Dashboard>>,
    launch: Res<Launch>,
    mut commands: Commands,
    mut exit: MessageWriter<AppExit>,
) {
    if let Some(dashboard) = dashboard {
        let display = dashboard.data.lock().expect("dashboard lock");
        match (
            Instant::now() >= verification.deadline,
            display.status.failed() || display.error.is_some(),
        ) {
            (true, _) | (_, true) => {
                eprintln!(
                    "viewer verification failed: {} {}",
                    display.status.label(),
                    display.error.as_deref().unwrap_or("")
                );
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
                        let request = InspectionRequest::new(
                            launch.scenario.identity(0).expect("first household"),
                            &launch.scenario,
                        )
                        .expect("first household");
                        let _ = dashboard.sender.send(Control::Inspect(request));
                        VerificationState::AwaitInspection
                    }
                    VerificationState::AwaitInspection if display.inspection.is_some() => {
                        let _ = dashboard.sender.send(Control::Compare);
                        VerificationState::AwaitComparison
                    }
                    VerificationState::AwaitComparison
                        if Some(display.seed) == launch.scenario.config().seed.checked_add(1)
                            && display.snapshot.day == 1
                            && matches!(
                                display.status,
                                DisplayState::Active(RuntimeState::Paused)
                            ) =>
                    {
                        let identity = launch
                            .scenario
                            .identity(launch.scenario.config().households as usize)
                            .expect("first firm");
                        let request =
                            InspectionRequest::new(identity, &launch.scenario).expect("first firm");
                        let _ = dashboard.sender.send(Control::Inspect(request));
                        VerificationState::AwaitFirmInspection
                    }
                    VerificationState::AwaitFirmInspection
                        if display
                            .inspection
                            .as_ref()
                            .is_some_and(|record| record.actor.kind > 0) =>
                    {
                        VerificationState::Frames(3)
                    }
                    VerificationState::Frames(0) => {
                        commands
                            .spawn(Screenshot::primary_window())
                            .observe(capture_verified_frame);
                        VerificationState::AwaitCapture
                    }
                    VerificationState::Frames(frames) => VerificationState::Frames(frames - 1),
                    state => state,
                };
            }
        }
    }
}

fn publish(
    simulation: &mut Simulation,
    data: &Arc<Mutex<Display>>,
    publication: Publication,
) -> Result<()> {
    let firms = match publication {
        Publication::Restart => Some(
            simulation
                .actors()
                .iter()
                .filter(|actor| actor.kind > 0)
                .cloned()
                .collect(),
        ),
        Publication::Snapshot => None,
    };
    let mut display = data.lock().expect("dashboard lock");
    if let Some(firms) = firms {
        display.firms = firms;
        display.map_size = simulation.scenario.config().map_size;
        display.history.clear();
        display.inspection = None;
    }
    match display.history.back().map(|s| s.day) {
        Some(day) if day == simulation.snapshot.day => (),
        _ => {
            display.history.push_back(simulation.snapshot.clone());
            if display.history.len() > 160 {
                display.history.pop_front();
            }
        }
    }
    display.snapshot = simulation.snapshot.clone();
    display.status = DisplayState::Active(simulation.status.clone());
    display.seed = simulation.scenario.config().seed;
    display.error = None;
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
            let running = matches!(
                dashboard.data.lock().expect("dashboard lock").status,
                DisplayState::Active(RuntimeState::Running)
            );
            if keys.just_pressed(KeyCode::KeyG) {
                dashboard.chart = dashboard.chart.next();
            }
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
                dashboard.selected = launch.scenario.identity(0).expect("first household");
            }
            if keys.just_pressed(KeyCode::KeyF) {
                dashboard.selected = launch
                    .scenario
                    .identity(launch.scenario.config().households as usize)
                    .expect("first firm");
            }
            if keys.just_pressed(KeyCode::Tab) {
                let index = launch.scenario.index_of(dashboard.selected).unwrap_or(0);
                dashboard.selected = launch
                    .scenario
                    .identity((index + 1) % launch.scenario.actors())
                    .expect("next actor");
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
    mut labels: Query<(&PanelLabel, &mut Text2d)>,
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
                            "Health {} | fatigue {} | labor {} min\nEmployer {}",
                            i.health, i.fatigue, i.available_minutes, i.employer
                        ),
                        _ => format!(
                            "Owner {} | wage {} | price {}\nCapacity {} | planned {} | open {}",
                            i.owner_identity,
                            i.wage,
                            i.price,
                            i.actor.capacity,
                            i.planned_packets,
                            i.open_slots
                        ),
                    };
                    format!(
                        "Actor {} | day {} | kind {}\nCash {} | stock {:?} | transit {}\n{}",
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
                .map(|reference| match reference.at_day(s.day) {
                    Some(prior) => format!(
                        "Seed {} | same day {}\nEmployment change {:+} | unmet bread change {:+}",
                        reference.seed,
                        s.day,
                        i128::from(s.employed) - i128::from(prior.employed),
                        i128::from(s.shortages) - i128::from(prior.shortages)
                    ),
                    None => format!(
                        "Comparing seed {} | target day {}",
                        reference.seed, reference.horizon
                    ),
                })
                .unwrap_or_default();
            let quotes = s
                .markets
                .iter()
                .zip(["Grain", "Flour", "Bread"])
                .map(|(market, name)| {
                    format!(
                        "{name}: {:.1} +/- {:.1} / {:.1} +/- {:.1}",
                        market.prices.mean(),
                        market.prices.deviation(),
                        market.wages.mean(),
                        market.wages.deviation()
                    )
                })
                .collect::<Vec<_>>()
                .join("\n");
            let text = format!(
                "{} | seed {}\nDay {} | target {}\nUnemployment {:.1}% | unmet bread {}\nCash {}\nStocks G/F/B {:?}\nIn transit {:?}\n\nPrice / wage | mean +/- standard deviation\n{}\n\n{}\n\n{}\n{}",
                display.status.label(),
                display.seed,
                s.day,
                display.speed.label(),
                unemployment,
                s.shortages,
                s.cash,
                s.inventories,
                s.transit,
                quotes,
                inspection,
                comparison,
                display.error.as_deref().unwrap_or("")
            );
            let range = ChartRange::new(
                dashboard.chart,
                s,
                &display.history,
                display.comparison.as_ref(),
            );
            let reference_label = match &display.comparison {
                Some(reference) => format!(
                    "Current seed {} | faint seed {}",
                    display.seed, reference.seed
                ),
                None => "Grain amber | flour blue | bread green".into(),
            };
            let title = format!("G: {}\n{}", dashboard.chart.label(), reference_label);
            labels.iter_mut().for_each(|(kind, mut label)| {
                label.0 = match kind {
                    PanelLabel::Statistics => text.clone(),
                    PanelLabel::ChartTitle => title.clone(),
                    PanelLabel::ChartScale => range.scale_label(dashboard.chart),
                };
            });
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
            draw_chart(&display, dashboard.chart, &range, &mut gizmos);
        }
    }
}

fn chart_color(chart: Chart, series: usize) -> Color {
    match (chart, series) {
        (Chart::Employment, 0) => Color::srgb(0.45, 0.90, 0.65),
        (Chart::Employment, _) => Color::srgb(0.95, 0.35, 0.30),
        (_, 0) => Color::srgb(0.95, 0.73, 0.25),
        (_, 1) => Color::srgb(0.40, 0.70, 0.95),
        _ => Color::srgb(0.45, 0.90, 0.65),
    }
}
fn draw_chart(display: &Display, chart: Chart, range: &ChartRange, gizmos: &mut Gizmos) {
    let baseline = -315.0;
    gizmos.line_2d(
        Vec2::new(80.0, baseline),
        Vec2::new(560.0, baseline),
        Color::srgb(0.3, 0.4, 0.45),
    );
    gizmos.line_2d(
        Vec2::new(80.0, baseline),
        Vec2::new(80.0, -215.0),
        Color::srgb(0.3, 0.4, 0.45),
    );
    match chart {
        Chart::Cash => {
            if let Some(reference) = display
                .comparison
                .as_ref()
                .and_then(|reference| reference.at_day(display.snapshot.day))
            {
                draw_cash(
                    reference,
                    range,
                    Color::srgba(0.95, 0.73, 0.25, 0.5),
                    gizmos,
                );
            }
            draw_cash(
                &display.snapshot,
                range,
                Color::srgb(0.4, 0.7, 0.95),
                gizmos,
            );
        }
        _ => {
            if let Some(reference) = &display.comparison {
                draw_history(
                    &reference.history,
                    chart,
                    range,
                    LineStyle::Reference,
                    gizmos,
                );
            }
            draw_history(&display.history, chart, range, LineStyle::Current, gizmos);
        }
    }
}

#[derive(Clone, Copy)]
enum LineStyle {
    Current,
    Reference,
}
fn draw_history(
    history: &VecDeque<ObservationSnapshot>,
    chart: Chart,
    range: &ChartRange,
    style: LineStyle,
    gizmos: &mut Gizmos,
) {
    let visible = history
        .iter()
        .filter(|snapshot| (range.first_day..=range.last_day).contains(&snapshot.day))
        .collect::<Vec<_>>();
    let span = range.last_day.saturating_sub(range.first_day).max(1) as f32;
    visible.windows(2).for_each(|pair| {
        let a = chart.values(pair[0]);
        let b = chart.values(pair[1]);
        let from_x = 80.0 + pair[0].day.saturating_sub(range.first_day) as f32 / span * 480.0;
        let to_x = 80.0 + pair[1].day.saturating_sub(range.first_day) as f32 / span * 480.0;
        a.iter()
            .zip(b)
            .enumerate()
            .for_each(|(series, (from, to))| {
                let color = match style {
                    LineStyle::Current => chart_color(chart, series),
                    LineStyle::Reference => chart_color(chart, series).with_alpha(0.35),
                };
                gizmos.line_2d(
                    Vec2::new(from_x, -315.0 + *from as f32 / range.maximum as f32 * 100.0),
                    Vec2::new(to_x, -315.0 + to as f32 / range.maximum as f32 * 100.0),
                    color,
                );
            });
    });
}
fn draw_cash(
    snapshot: &ObservationSnapshot,
    range: &ChartRange,
    color: Color,
    gizmos: &mut Gizmos,
) {
    let width = 480.0 / (range.last_bin + 1) as f32;
    snapshot
        .wealth_bins
        .iter()
        .take(range.last_bin + 1)
        .enumerate()
        .for_each(|(bin, count)| {
            let height = *count as f32 / range.maximum as f32 * 100.0;
            gizmos.rect_2d(
                Isometry2d::from_translation(Vec2::new(
                    80.0 + (bin as f32 + 0.5) * width,
                    -315.0 + height / 2.0,
                )),
                Vec2::new(width * 0.7, height),
                color,
            );
        });
}

fn capture_verified_frame(
    captured: On<ScreenshotCaptured>,
    mut dashboard: ResMut<Dashboard>,
    mut verification: ResMut<Verification>,
    mut exit: MessageWriter<AppExit>,
) {
    let chart = dashboard.chart;
    let result = save_verified_frame(&captured.image, chart);
    match result {
        Ok(path) => {
            eprintln!("viewer captured {}", path.display());
            match chart {
                Chart::Cash => {
                    eprintln!(
                        "viewer verification passed: shared device, audited day, household and firm inspection, same-day seed comparison, all charts captured"
                    );
                    verification.state = VerificationState::Complete;
                    exit.write(AppExit::Success);
                }
                _ => {
                    dashboard.chart = chart.next();
                    verification.state = VerificationState::Frames(3);
                }
            }
        }
        Err(error) => {
            eprintln!("viewer capture failed: {error}");
            verification.state = VerificationState::Complete;
            exit.write(AppExit::error());
        }
    }
}
fn save_verified_frame(image: &Image, chart: Chart) -> Result<std::path::PathBuf> {
    let pixels =
        Context::context(image.clone().try_into_dynamic(), "read rendered frame")?.into_rgb8();
    let first = Context::context(pixels.pixels().next(), "empty rendered frame")?;
    match pixels.pixels().any(|pixel| pixel != first) {
        true => Ok(()),
        false => Err(anyhow::anyhow!("rendered frame contains only a flat color")),
    }?;
    let path = std::env::temp_dir().join(format!("bad-econ-viewer-{}.ppm", chart.file_label()));
    let mut file = std::fs::File::create(&path)?;
    write!(file, "P6\n{} {}\n255\n", pixels.width(), pixels.height())?;
    file.write_all(pixels.as_raw())?;
    Ok(path)
}

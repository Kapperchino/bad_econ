mod feedback;
mod graph;
mod model;
mod pacing;
mod presentation;
mod ui;
mod worker;
use crate::{
    domain::*,
    runtime::{RuntimeState, SimulationCommand},
    simulation::{Device, Simulation},
};
use anyhow::{Context, Result};
use bevy::{
    ecs::system::SystemParam,
    input::{ButtonState, mouse::MouseButtonInput},
    prelude::*,
    render::{
        RenderApp, RenderPlugin,
        renderer::{RenderAdapter, RenderDevice, RenderInstance, RenderQueue},
        settings::{RenderCreation, WgpuSettings},
        view::screenshot::{Screenshot, ScreenshotCaptured},
    },
    window::{CursorMoved, PrimaryWindow, WindowEvent},
    winit::{EventLoopProxyWrapper, WinitUserEvent},
};
use crossbeam_channel::{Sender, unbounded};
use feedback::Cue;
use model::{Chart, ComparisonRun, DisplayState, RunTarget};
use std::{
    collections::VecDeque,
    io::Write,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use ui::{ActorSelection, UiAction};

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
    click: VerificationClick,
    capture_error: Option<String>,
}
#[derive(Clone, Copy, Debug)]
enum VerificationClick {
    Idle,
    Move(UiAction),
    Press(Entity),
    Release(Entity),
}
#[derive(Clone, Copy, Debug)]
enum VerificationState {
    Start,
    AwaitReset,
    AwaitDay,
    AwaitFeedbackCapture,
    InspectHousehold,
    AwaitInspection,
    AwaitInspectionRefresh,
    AwaitComparison,
    AwaitFirmInspection,
    AwaitMapInspection(u64),
    AwaitSpeed(RetainedPresentation),
    AwaitPrevious {
        identity: u64,
        presentation: RetainedPresentation,
    },
    AwaitNext {
        identity: u64,
        presentation: RetainedPresentation,
    },
    AwaitMenu(Chart),
    AwaitChart(Chart),
    Frames(u32),
    AwaitCapture,
    AwaitPreferences,
    Resize,
    CompactFrames(u32),
    AwaitCompactCapture,
    Complete,
}
#[derive(Clone, Copy, Debug)]
struct RetainedPresentation {
    plot: Entity,
    legend: Entity,
}

#[derive(SystemParam)]
struct VerificationPresentation<'w, 's> {
    plots: Query<'w, 's, &'static Children, With<ui::Plot>>,
    legends: Query<'w, 's, &'static Children, With<ui::Legend>>,
    meshes: Query<'w, 's, &'static Mesh2d>,
    windows: Query<'w, 's, &'static mut Window, With<PrimaryWindow>>,
}
impl VerificationPresentation<'_, '_> {
    fn retained(&self) -> RetainedPresentation {
        assert_eq!(self.meshes.iter().count(), 1, "charts share one GPU mesh");
        let children = self.plots.single().expect("plot children");
        assert!(
            children.len() <= 8,
            "plot UI must remain bounded independently of history"
        );
        RetainedPresentation {
            plot: children[0],
            legend: self.legends.single().expect("legend children")[0],
        }
    }
    fn assert_retained(&self, previous: RetainedPresentation) {
        let current = self.retained();
        assert_eq!(
            current.plot, previous.plot,
            "unrelated controls rebuilt the graph"
        );
        assert_eq!(
            current.legend, previous.legend,
            "unrelated controls rebuilt the legend"
        );
    }
}

#[derive(Resource)]
struct Dashboard {
    sender: Sender<Control>,
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
    identities: Vec<u64>,
    status: DisplayState,
    error: Option<String>,
    seed: u64,
    map_size: u32,
    speed: Speed,
    revision: u64,
    observation_revision: u64,
    map_revision: u64,
    cues: VecDeque<Cue>,
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
#[derive(Clone, Copy)]
enum Publication {
    Snapshot,
    Restart,
}

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
                    title: "Spatial industrial economy".into(),
                    resolution: (1600, 960).into(),
                    resize_constraints: bevy::window::WindowResizeConstraints {
                        min_width: 1280.0,
                        min_height: 800.0,
                        ..Default::default()
                    },
                    ..Default::default()
                }),
                ..Default::default()
            })
            .set(RenderPlugin {
                render_creation: RenderCreation::Automatic(Box::new(settings)),
                ..Default::default()
            }),
    )
    .add_plugins((
        SharedGpu,
        ui::UiPlugin,
        presentation::PresentationPlugin,
        pacing::PacingPlugin,
        feedback::FeedbackPlugin,
    ))
    .add_systems(Startup, setup)
    .add_systems(Update, controls);
    if let ViewMode::Verify = mode {
        app.insert_resource(Verification {
            state: VerificationState::Start,
            deadline: Instant::now() + Duration::from_secs(120),
            click: VerificationClick::Idle,
            capture_error: None,
        })
        .add_systems(
            Update,
            (verify, verify_clicks)
                .chain()
                .after(feedback::FeedbackSystems)
                .after(presentation::PresentationSystems),
        );
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
    event_loop: Res<EventLoopProxyWrapper>,
) {
    commands.spawn((Camera2d, IsDefaultUiCamera));
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
        identities: Vec::new(),
        status: DisplayState::Initializing,
        error: None,
        seed: launch.scenario.config().seed,
        map_size: launch.scenario.config().map_size,
        speed: Speed::Normal,
        revision: 0,
        observation_revision: 0,
        map_revision: 0,
        cues: VecDeque::new(),
    }));
    let (sender, receiver) = unbounded();
    commands.insert_resource(Dashboard {
        sender,
        data: data.clone(),
        selected: launch.scenario.identity(0).expect("first household"),
        chart: Chart::Employment,
    });
    let scenario = launch.scenario.clone();
    let workgroup = launch.workgroup;
    let wake = (**event_loop).clone();
    std::thread::spawn(move || {
        let worker_wake = wake.clone();
        let result = Device::import(setup, workgroup)
            .and_then(|device| Simulation::new(scenario, device))
            .and_then(|simulation| {
                worker::run(simulation, receiver, data.clone(), move || {
                    let _ = worker_wake.send_event(WinitUserEvent::WakeUp);
                })
            });
        match result {
            Ok(()) => (),
            Err(error) => {
                let mut display = data.lock().expect("dashboard lock");
                display.status = DisplayState::Unavailable(error.to_string());
                display.cues.push_back(Cue::Error);
                display.revision = display.revision.wrapping_add(1);
                let _ = wake.send_event(WinitUserEvent::WakeUp);
            }
        }
    });
}

fn verify(
    mut verification: ResMut<Verification>,
    dashboard: Option<Res<Dashboard>>,
    launch: Res<Launch>,
    feedback: Res<feedback::Feedback>,
    mut presentation: VerificationPresentation,
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
                    "viewer verification failed in {:?} / {:?}: {} {}",
                    verification.state,
                    verification.click,
                    display.status.label(),
                    display.error.as_deref().unwrap_or("")
                );
                exit.write(AppExit::error());
                verification.state = VerificationState::Complete;
            }
            _ => {
                verification.state = match verification.state {
                    VerificationState::Start => {
                        verification.click = VerificationClick::Move(UiAction::Reset);
                        VerificationState::AwaitReset
                    }
                    VerificationState::AwaitReset if feedback.current() == Some(Cue::Reset) => {
                        verification.click = VerificationClick::Move(UiAction::Step);
                        VerificationState::AwaitDay
                    }
                    VerificationState::AwaitDay if feedback.current() == Some(Cue::Stepped) => {
                        assert_eq!(display.snapshot.day, 1);
                        commands
                            .spawn(Screenshot::primary_window())
                            .observe(capture_step_feedback);
                        VerificationState::AwaitFeedbackCapture
                    }
                    VerificationState::InspectHousehold => {
                        verification.click =
                            VerificationClick::Move(UiAction::Inspect(ActorSelection::Household));
                        VerificationState::AwaitInspection
                    }
                    VerificationState::AwaitInspection if display.inspection.is_some() => {
                        verification.click = VerificationClick::Move(UiAction::Step);
                        VerificationState::AwaitInspectionRefresh
                    }
                    VerificationState::AwaitInspectionRefresh
                        if display.snapshot.day == 2
                            && display
                                .inspection
                                .as_ref()
                                .is_some_and(|record| record.day == 2) =>
                    {
                        verification.click = VerificationClick::Move(UiAction::Compare);
                        VerificationState::AwaitComparison
                    }
                    VerificationState::AwaitComparison
                        if Some(display.seed) == launch.scenario.config().seed.checked_add(1)
                            && display.snapshot.day == 2
                            && matches!(
                                display.status,
                                DisplayState::Active(RuntimeState::Paused)
                            )
                            && feedback.current() == Some(Cue::Compared) =>
                    {
                        verification.click =
                            VerificationClick::Move(UiAction::Inspect(ActorSelection::Firm));
                        VerificationState::AwaitFirmInspection
                    }
                    VerificationState::AwaitFirmInspection
                        if display
                            .inspection
                            .as_ref()
                            .is_some_and(|record| record.actor.kind > 0) =>
                    {
                        let identity = display.firms.last().expect("map firm").identity;
                        verification.click = VerificationClick::Move(UiAction::Inspect(
                            ActorSelection::Identity(identity),
                        ));
                        VerificationState::AwaitMapInspection(identity)
                    }
                    VerificationState::AwaitMapInspection(identity)
                        if display
                            .inspection
                            .as_ref()
                            .is_some_and(|record| record.actor.identity == identity) =>
                    {
                        verification.click = VerificationClick::Move(UiAction::Speed(Speed::Slow));
                        VerificationState::AwaitSpeed(presentation.retained())
                    }
                    VerificationState::AwaitSpeed(retained)
                        if display.speed == Speed::Slow
                            && matches!(verification.click, VerificationClick::Idle) =>
                    {
                        presentation.assert_retained(retained);
                        let index = display
                            .identities
                            .iter()
                            .position(|identity| *identity == dashboard.selected)
                            .expect("selected actor");
                        let previous = display.identities
                            [(index + display.identities.len() - 1) % display.identities.len()];
                        verification.click =
                            VerificationClick::Move(UiAction::Inspect(ActorSelection::Previous));
                        VerificationState::AwaitPrevious {
                            identity: previous,
                            presentation: retained,
                        }
                    }
                    VerificationState::AwaitPrevious {
                        identity,
                        presentation: retained,
                    } if display
                        .inspection
                        .as_ref()
                        .is_some_and(|record| record.actor.identity == identity)
                        && matches!(verification.click, VerificationClick::Idle) =>
                    {
                        presentation.assert_retained(retained);
                        let next = display.firms.last().expect("map firm").identity;
                        verification.click =
                            VerificationClick::Move(UiAction::Inspect(ActorSelection::Next));
                        VerificationState::AwaitNext {
                            identity: next,
                            presentation: retained,
                        }
                    }
                    VerificationState::AwaitNext {
                        identity,
                        presentation: retained,
                    } if display
                        .inspection
                        .as_ref()
                        .is_some_and(|record| record.actor.identity == identity)
                        && matches!(verification.click, VerificationClick::Idle) =>
                    {
                        presentation.assert_retained(retained);
                        VerificationState::Frames(3)
                    }
                    VerificationState::AwaitMenu(chart)
                        if dashboard.chart.menu() == chart.menu()
                            && matches!(verification.click, VerificationClick::Idle) =>
                    {
                        verification.click = VerificationClick::Move(UiAction::Chart(chart));
                        VerificationState::AwaitChart(chart)
                    }
                    VerificationState::AwaitChart(chart)
                        if dashboard.chart == chart
                            && matches!(verification.click, VerificationClick::Idle) =>
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
                    VerificationState::AwaitPreferences
                        if feedback.current() == Some(Cue::Preferences) =>
                    {
                        VerificationState::Resize
                    }
                    VerificationState::Resize => {
                        presentation
                            .windows
                            .single_mut()
                            .expect("primary window")
                            .resolution
                            .set(1280.0, 800.0);
                        VerificationState::CompactFrames(8)
                    }
                    VerificationState::CompactFrames(0) => {
                        commands
                            .spawn(Screenshot::primary_window())
                            .observe(capture_compact_frame);
                        VerificationState::AwaitCompactCapture
                    }
                    VerificationState::CompactFrames(frames) => {
                        VerificationState::CompactFrames(frames - 1)
                    }
                    state => state,
                };
            }
        }
    }
}

fn verify_clicks(
    mut verification: ResMut<Verification>,
    buttons: Query<(&UiAction, &ComputedNode, &UiGlobalTransform)>,
    windows: Query<(Entity, &Window), With<PrimaryWindow>>,
    mut input: MessageWriter<WindowEvent>,
) {
    verification.click = match verification.click {
        VerificationClick::Move(action) => match windows.single() {
            Ok((window, properties)) => match buttons.iter().find(|(candidate, node, _)| {
                **candidate == action && node.size().min_element() > 0.0
            }) {
                Some((_, _, transform)) => {
                    let position = transform.translation / properties.scale_factor();
                    assert!(
                        Rect::from_corners(Vec2::ZERO, properties.size()).contains(position),
                        "{action:?} lies outside the window at {position:?}"
                    );
                    input.write(WindowEvent::CursorMoved(CursorMoved {
                        window,
                        position,
                        delta: None,
                    }));
                    VerificationClick::Press(window)
                }
                None => VerificationClick::Move(action),
            },
            Err(_) => VerificationClick::Move(action),
        },
        VerificationClick::Press(window) => {
            input.write(WindowEvent::MouseButtonInput(MouseButtonInput {
                window,
                button: MouseButton::Left,
                state: ButtonState::Pressed,
            }));
            VerificationClick::Release(window)
        }
        VerificationClick::Release(window) => {
            input.write(WindowEvent::MouseButtonInput(MouseButtonInput {
                window,
                button: MouseButton::Left,
                state: ButtonState::Released,
            }));
            VerificationClick::Idle
        }
        VerificationClick::Idle => VerificationClick::Idle,
    };
}

fn controls(
    keys: Res<ButtonInput<KeyCode>>,
    launch: Res<Launch>,
    dashboard: Option<ResMut<Dashboard>>,
    mut actions: MessageReader<UiAction>,
) {
    let shortcuts = [
        (KeyCode::Space, UiAction::ToggleRun),
        (KeyCode::KeyN, UiAction::Step),
        (KeyCode::KeyR, UiAction::Reset),
        (KeyCode::KeyC, UiAction::Compare),
        (KeyCode::Digit1, UiAction::Speed(Speed::Slow)),
        (KeyCode::Digit2, UiAction::Speed(Speed::Normal)),
        (KeyCode::Digit3, UiAction::Speed(Speed::Fast)),
        (KeyCode::KeyH, UiAction::Inspect(ActorSelection::Household)),
        (KeyCode::KeyF, UiAction::Inspect(ActorSelection::Firm)),
        (
            KeyCode::Tab,
            UiAction::Inspect(
                match keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight) {
                    true => ActorSelection::Previous,
                    false => ActorSelection::Next,
                },
            ),
        ),
        (KeyCode::KeyG, UiAction::NextChart),
    ];
    let pending = actions
        .read()
        .copied()
        .chain(
            shortcuts
                .into_iter()
                .filter(|(key, _)| keys.just_pressed(*key))
                .map(|(_, action)| action),
        )
        .collect::<Vec<_>>();
    if let Some(mut dashboard) = dashboard {
        pending.into_iter().for_each(|action| {
            let control = match action {
                UiAction::Menu(menu) => {
                    dashboard.chart = menu.charts()[0];
                    None
                }
                UiAction::Chart(chart) => {
                    dashboard.chart = chart;
                    None
                }
                UiAction::NextChart => {
                    dashboard.chart = dashboard.chart.next();
                    None
                }
                UiAction::ToggleRun => Some(Control::command(
                    match dashboard.data.lock().expect("dashboard lock").status {
                        DisplayState::Active(RuntimeState::Running) => SimulationCommand::Pause,
                        _ => SimulationCommand::Resume,
                    },
                )),
                UiAction::Step => Some(Control::command(SimulationCommand::AdvanceDay)),
                UiAction::Reset => Some(Control::command(SimulationCommand::Reset(
                    launch.scenario.clone().into(),
                ))),
                UiAction::Compare => Some(Control::Compare),
                UiAction::Speed(speed) => Some(Control::Speed(speed)),
                UiAction::Inspect(selection) => {
                    let selected = {
                        let display = dashboard.data.lock().expect("dashboard lock");
                        match selection {
                            ActorSelection::Household => display.identities.first().copied(),
                            ActorSelection::Firm => {
                                display.firms.first().map(|actor| actor.identity)
                            }
                            ActorSelection::Identity(identity) => Some(identity),
                            ActorSelection::Next | ActorSelection::Previous => {
                                let index = display
                                    .identities
                                    .iter()
                                    .position(|&identity| identity == dashboard.selected)
                                    .unwrap_or(0);
                                let count = display.identities.len().max(1);
                                let next = match selection {
                                    ActorSelection::Previous => (index + count - 1) % count,
                                    _ => (index + 1) % count,
                                };
                                display.identities.get(next).copied()
                            }
                        }
                    };
                    selected.and_then(|identity| {
                        dashboard.selected = identity;
                        InspectionRequest::from_registry(
                            identity,
                            &dashboard.data.lock().expect("dashboard lock").identities,
                        )
                        .ok()
                        .map(Control::Inspect)
                    })
                }
                UiAction::Sound | UiAction::Motion => None,
            };
            if let Some(control) = control {
                let _ = dashboard.sender.send(control);
            }
        });
    }
}

impl Verification {
    fn capture(&mut self, image: &Image, label: &str) {
        match save_verified_frame(image, label) {
            Ok(path) => eprintln!("viewer captured {}", path.display()),
            Err(error) => {
                eprintln!("viewer {label} capture failed: {error}");
                self.capture_error.get_or_insert_with(|| error.to_string());
            }
        }
    }
}

fn capture_step_feedback(captured: On<ScreenshotCaptured>, mut verification: ResMut<Verification>) {
    verification.capture(&captured.image, "step-feedback");
    verification.state = VerificationState::InspectHousehold;
}

fn capture_verified_frame(
    captured: On<ScreenshotCaptured>,
    dashboard: Res<Dashboard>,
    mut verification: ResMut<Verification>,
) {
    let chart = dashboard.chart;
    verification.capture(&captured.image, chart.file_label());
    match chart {
        Chart::Research => {
            verification.click = VerificationClick::Move(UiAction::Sound);
            verification.state = VerificationState::AwaitPreferences;
        }
        _ => {
            let next = chart.next();
            verification.click = VerificationClick::Move(UiAction::Menu(next.menu()));
            verification.state = VerificationState::AwaitMenu(next);
        }
    }
}

fn capture_compact_frame(
    captured: On<ScreenshotCaptured>,
    buttons: Query<(&UiAction, &ComputedNode, &UiGlobalTransform)>,
    mut verification: ResMut<Verification>,
    mut exit: MessageWriter<AppExit>,
) {
    let viewport = Rect::from_corners(Vec2::ZERO, captured.image.size_f32());
    buttons
        .iter()
        .filter(|(_, node, _)| node.size().min_element() > 0.0)
        .for_each(|(action, node, transform)| {
            assert!(
                viewport.contains(transform.translation - node.size() / 2.0),
                "{action:?} begins outside the compact window"
            );
            assert!(
                viewport.contains(transform.translation + node.size() / 2.0),
                "{action:?} ends outside the compact window"
            );
        });
    verification.capture(&captured.image, "compact");
    eprintln!(
        "viewer interaction and layout verification passed: shared GPU, native mouse clicks, reset/step/comparison feedback, audited days, household and firm inspection, refreshed inspection, all five menus and ten charts, retained GPU chart on speed and inspection changes, previous/next navigation, footer controls, compact window"
    );
    match &verification.capture_error {
        None => {
            eprintln!("viewer screenshot verification passed");
            exit.write(AppExit::Success);
        }
        Some(error) => {
            eprintln!("viewer screenshot verification failed: {error}");
            exit.write(AppExit::error());
        }
    }
    verification.state = VerificationState::Complete;
}

fn save_verified_frame(image: &Image, label: &str) -> Result<std::path::PathBuf> {
    let pixels =
        Context::context(image.clone().try_into_dynamic(), "read rendered frame")?.into_rgb8();
    let first = Context::context(pixels.pixels().next(), "empty rendered frame")?;
    match pixels.pixels().any(|pixel| pixel != first) {
        true => Ok(()),
        false => Err(anyhow::anyhow!(
            "rendered frame contains only a flat color: {first:?}"
        )),
    }?;
    let path = std::env::temp_dir().join(format!("bad-econ-viewer-{label}.ppm"));
    let mut file = std::fs::File::create(&path)?;
    write!(file, "P6\n{} {}\n255\n", pixels.width(), pixels.height())?;
    file.write_all(pixels.as_raw())?;
    Ok(path)
}

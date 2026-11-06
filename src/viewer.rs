mod feedback;
mod model;
mod pacing;
mod presentation;
mod worker;
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
    AwaitReset,
    AwaitDay,
    AwaitFeedbackCapture,
    InspectHousehold,
    AwaitInspection,
    AwaitInspectionRefresh,
    AwaitComparison,
    AwaitFirmInspection,
    Frames(u32),
    AwaitCapture,
    Complete,
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
                    title: "Spatial industrial economy".into(),
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
    .add_plugins((
        SharedGpu,
        presentation::PresentationPlugin,
        pacing::PacingPlugin,
        feedback::FeedbackPlugin,
    ))
    .add_systems(Startup, setup)
    .add_systems(Update, controls);
    if let ViewMode::Verify = mode {
        app.insert_resource(Verification {
            state: VerificationState::Start,
            deadline: Instant::now() + Duration::from_secs(60),
        })
        .add_systems(Update, verify.after(feedback::FeedbackSystems));
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
        identities: Vec::new(),
        status: DisplayState::Initializing,
        error: None,
        seed: launch.scenario.config().seed,
        map_size: launch.scenario.config().map_size,
        speed: Speed::Normal,
        revision: 0,
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
        Text2d::new("Regional bread shortages | colored industry locations"),
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
            font_size: FontSize::Px(12.0),
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
    commands.spawn((Text2d::new("SPACE pause/resume    N step    R reset    1/2/3 speed    H household    F firm    TAB next    C compare    G chart"), TextFont { font_size: FontSize::Px(15.0), ..Default::default() }, Anchor::TOP_LEFT, Transform::from_xyz(-600.0, -360.0, 2.0)));
}

fn verify(
    mut verification: ResMut<Verification>,
    dashboard: Option<Res<Dashboard>>,
    launch: Res<Launch>,
    feedback: Res<feedback::Feedback>,
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
                            .send(Control::command(SimulationCommand::Reset(
                                launch.scenario.clone().into(),
                            )));
                        VerificationState::AwaitReset
                    }
                    VerificationState::AwaitReset if feedback.current() == Some(Cue::Reset) => {
                        let _ = dashboard
                            .sender
                            .send(Control::command(SimulationCommand::AdvanceDay));
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
                        let request = InspectionRequest::new(
                            launch.scenario.identity(0).expect("first household"),
                            &launch.scenario,
                        )
                        .expect("first household");
                        let _ = dashboard.sender.send(Control::Inspect(request));
                        VerificationState::AwaitInspection
                    }
                    VerificationState::AwaitInspection if display.inspection.is_some() => {
                        let _ = dashboard
                            .sender
                            .send(Control::command(SimulationCommand::AdvanceDay));
                        VerificationState::AwaitInspectionRefresh
                    }
                    VerificationState::AwaitInspectionRefresh
                        if display.snapshot.day == 2
                            && display
                                .inspection
                                .as_ref()
                                .is_some_and(|record| record.day == 2) =>
                    {
                        let _ = dashboard.sender.send(Control::Compare);
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
                    launch.scenario.clone().into(),
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
                let selected = {
                    let display = dashboard.data.lock().expect("dashboard lock");
                    let index = display
                        .identities
                        .iter()
                        .position(|&identity| identity == dashboard.selected)
                        .unwrap_or(0);
                    display
                        .identities
                        .get((index + 1) % display.identities.len().max(1))
                        .copied()
                        .unwrap_or(dashboard.selected)
                };
                dashboard.selected = selected;
            }
            let inspection = [KeyCode::KeyH, KeyCode::KeyF, KeyCode::Tab]
                .into_iter()
                .any(|key| keys.just_pressed(key))
                .then(|| {
                    InspectionRequest::from_registry(
                        dashboard.selected,
                        &dashboard.data.lock().expect("dashboard lock").identities,
                    )
                });
            if let Some(Ok(request)) = inspection {
                controls.push(Control::Inspect(request));
            }
            controls.into_iter().for_each(|control| {
                let _ = dashboard.sender.send(control);
            });
        }
    }
}

fn capture_step_feedback(
    captured: On<ScreenshotCaptured>,
    mut verification: ResMut<Verification>,
    mut exit: MessageWriter<AppExit>,
) {
    match save_verified_frame(&captured.image, "step-feedback") {
        Ok(path) => {
            eprintln!("viewer feedback captured {}", path.display());
            verification.state = VerificationState::InspectHousehold;
        }
        Err(error) => {
            eprintln!("viewer feedback capture failed: {error}");
            verification.state = VerificationState::Complete;
            exit.write(AppExit::error());
        }
    }
}

fn capture_verified_frame(
    captured: On<ScreenshotCaptured>,
    mut dashboard: ResMut<Dashboard>,
    mut verification: ResMut<Verification>,
    mut exit: MessageWriter<AppExit>,
) {
    let chart = dashboard.chart;
    let result = save_verified_frame(&captured.image, chart.file_label());
    match result {
        Ok(path) => {
            eprintln!("viewer captured {}", path.display());
            match chart {
                Chart::Cash => {
                    eprintln!(
                        "viewer verification passed: shared device, reset/step/comparison feedback, audited day, household and firm inspection, inspection refresh after stepping, same-day seed comparison, all charts captured"
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
fn save_verified_frame(image: &Image, label: &str) -> Result<std::path::PathBuf> {
    let pixels =
        Context::context(image.clone().try_into_dynamic(), "read rendered frame")?.into_rgb8();
    let first = Context::context(pixels.pixels().next(), "empty rendered frame")?;
    match pixels.pixels().any(|pixel| pixel != first) {
        true => Ok(()),
        false => Err(anyhow::anyhow!("rendered frame contains only a flat color")),
    }?;
    let path = std::env::temp_dir().join(format!("bad-econ-viewer-{label}.ppm"));
    let mut file = std::fs::File::create(&path)?;
    write!(file, "P6\n{} {}\n255\n", pixels.width(), pixels.height())?;
    file.write_all(pixels.as_raw())?;
    Ok(path)
}

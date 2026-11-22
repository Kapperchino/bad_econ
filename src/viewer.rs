mod companies;
mod feedback;
mod graph;
mod model;
mod pacing;
mod presentation;
mod ui;
mod verification;
mod worker;
use crate::{
    domain::*,
    runtime::{RuntimeState, SimulationCommand},
    simulation::{Device, Simulation},
};
use anyhow::Result;
use bevy::{
    prelude::*,
    winit::{EventLoopProxyWrapper, WinitUserEvent},
};
use crossbeam_channel::{Sender, unbounded};
use feedback::Cue;
use model::{Chart, ComparisonRun, DisplayState, Period, RunTarget, SeriesSelection};
use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
    time::Duration,
};
use ui::{ActorSelection, UiAction};

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
struct Dashboard {
    sender: Sender<Control>,
    data: Arc<Mutex<Display>>,
    selected: u64,
    chart: Chart,
    company_page: Option<companies::CompanyPage>,
    period: Period,
    series: SeriesSelection,
}
struct Display {
    snapshot: ObservationSnapshot,
    history: VecDeque<ObservationSnapshot>,
    comparison: Option<ComparisonRun>,
    inspection: Option<Inspection>,
    firms: Vec<Actor>,
    companies: Vec<crate::simulation::company_report::Business>,
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
impl Display {
    fn new(scenario: &Scenario) -> Self {
        Self {
            snapshot: ObservationSnapshot::default(),
            history: VecDeque::new(),
            comparison: None,
            inspection: None,
            firms: Vec::new(),
            companies: Vec::new(),
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
        }
    }
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
    Comparison { seed: u64, horizon: u64 },
}

pub fn run(scenario: Scenario, workgroup: u32, mode: ViewMode) -> Result<()> {
    let mut app = App::new();
    app.insert_resource(Launch {
        scenario,
        workgroup,
    })
    .insert_resource(ClearColor(Color::srgb(0.035, 0.045, 0.07)))
    .add_plugins(DefaultPlugins.set(WindowPlugin {
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
    }))
    .add_plugins((
        ui::UiPlugin,
        companies::CompaniesPlugin,
        presentation::PresentationPlugin,
        pacing::PacingPlugin,
        feedback::FeedbackPlugin,
    ))
    .add_systems(Startup, setup)
    .add_systems(Update, controls);
    if let ViewMode::Verify = mode {
        app.add_plugins(verification::VerificationPlugin);
    }
    match app.run() {
        AppExit::Success => Ok(()),
        AppExit::Error(code) => Err(anyhow::anyhow!("viewer exited with {code}")),
    }
}

fn setup(mut commands: Commands, launch: Res<Launch>, event_loop: Res<EventLoopProxyWrapper>) {
    commands.spawn((Camera2d, IsDefaultUiCamera));
    let data = Arc::new(Mutex::new(Display::new(&launch.scenario)));
    let (sender, receiver) = unbounded();
    commands.insert_resource(Dashboard {
        sender,
        data: data.clone(),
        selected: launch.scenario.identity(0).expect("first household"),
        chart: Chart::Employment,
        company_page: None,
        period: Period::default(),
        series: SeriesSelection::All,
    });
    let scenario = launch.scenario.clone();
    let workgroup = launch.workgroup;
    let wake = (**event_loop).clone();
    std::thread::spawn(move || {
        let worker_wake = wake.clone();
        let result = Device::headless(workgroup)
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
                UiAction::Companies => {
                    dashboard.company_page = Some(companies::CompanyPage::default());
                    None
                }
                UiAction::Rank(order) => {
                    dashboard.company_page = Some(companies::CompanyPage::new(order));
                    None
                }
                UiAction::CompanyPage(direction) => {
                    let count = dashboard
                        .data
                        .lock()
                        .expect("dashboard lock")
                        .companies
                        .len();
                    if let Some(page) = &mut dashboard.company_page {
                        page.turn(direction, count);
                    }
                    None
                }
                UiAction::Menu(menu) => {
                    dashboard.company_page = None;
                    dashboard.chart = menu.charts()[0];
                    dashboard.series = SeriesSelection::All;
                    None
                }
                UiAction::Chart(chart) => {
                    dashboard.company_page = None;
                    dashboard.chart = chart;
                    dashboard.series = SeriesSelection::All;
                    None
                }
                UiAction::Period(period) => {
                    dashboard.period = period;
                    None
                }
                UiAction::NextSeries => {
                    dashboard.series = dashboard.series.next(dashboard.chart.series().len());
                    None
                }
                UiAction::AllSeries => {
                    dashboard.series = SeriesSelection::All;
                    None
                }
                UiAction::NextChart => {
                    dashboard.company_page = None;
                    dashboard.chart = dashboard.chart.next();
                    dashboard.series = SeriesSelection::All;
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

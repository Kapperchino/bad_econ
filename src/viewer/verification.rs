use super::{
    ActorSelection, Chart, Cue, Dashboard, DisplayState, Launch, Period, SeriesSelection, Speed,
    UiAction, companies, feedback, model, presentation, ui,
};
use crate::runtime::RuntimeState;
use anyhow::{Context, Result};
use bevy::{
    ecs::system::SystemParam,
    input::{ButtonState, mouse::MouseButtonInput},
    prelude::*,
    render::view::screenshot::{Screenshot, ScreenshotCaptured},
    window::{CursorMoved, PrimaryWindow, WindowEvent},
};
use std::{
    io::Write,
    time::{Duration, Instant},
};

pub(super) struct VerificationPlugin;
impl Plugin for VerificationPlugin {
    fn build(&self, app: &mut App) {
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
}

#[derive(Resource)]
pub(super) struct Verification {
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
    AwaitCompanies,
    AwaitCompanyRanking,
    AwaitCompanyCapture,
    LeaveCompanies,
    AwaitOverview,
    AwaitMenu(Chart),
    AwaitChart(Chart),
    Frames(u32),
    AwaitCapture,
    AwaitPeriod(Period),
    AwaitFocus,
    AwaitAllSeries,
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
                        verification.click = VerificationClick::Move(UiAction::Companies);
                        VerificationState::AwaitCompanies
                    }
                    VerificationState::AwaitCompanies
                        if dashboard.company_page.is_some()
                            && matches!(verification.click, VerificationClick::Idle) =>
                    {
                        verification.click =
                            VerificationClick::Move(UiAction::Rank(companies::Ranking::Profit));
                        VerificationState::AwaitCompanyRanking
                    }
                    VerificationState::AwaitCompanyRanking
                        if dashboard
                            .company_page
                            .is_some_and(|page| page.ranking == companies::Ranking::Profit)
                            && matches!(verification.click, VerificationClick::Idle) =>
                    {
                        commands
                            .spawn(Screenshot::primary_window())
                            .observe(capture_companies);
                        VerificationState::AwaitCompanyCapture
                    }
                    VerificationState::LeaveCompanies => {
                        verification.click =
                            VerificationClick::Move(UiAction::Menu(model::Menu::Overview));
                        VerificationState::AwaitOverview
                    }
                    VerificationState::AwaitOverview
                        if dashboard.company_page.is_none()
                            && matches!(verification.click, VerificationClick::Idle) =>
                    {
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
                    VerificationState::AwaitPeriod(period)
                        if dashboard.period == period
                            && matches!(verification.click, VerificationClick::Idle) =>
                    {
                        match period {
                            Period::Month => {
                                verification.click =
                                    VerificationClick::Move(UiAction::Period(Period::Year));
                                VerificationState::AwaitPeriod(Period::Year)
                            }
                            Period::Year => {
                                verification.click =
                                    VerificationClick::Move(UiAction::Period(Period::FiveYears));
                                VerificationState::AwaitPeriod(Period::FiveYears)
                            }
                            Period::FiveYears => {
                                verification.click =
                                    VerificationClick::Move(UiAction::Period(Period::All));
                                VerificationState::AwaitPeriod(Period::All)
                            }
                            Period::All => {
                                verification.click = VerificationClick::Move(UiAction::NextSeries);
                                VerificationState::AwaitFocus
                            }
                        }
                    }
                    VerificationState::AwaitFocus
                        if dashboard.series == SeriesSelection::Only(0)
                            && matches!(verification.click, VerificationClick::Idle) =>
                    {
                        verification.click = VerificationClick::Move(UiAction::AllSeries);
                        VerificationState::AwaitAllSeries
                    }
                    VerificationState::AwaitAllSeries
                        if dashboard.series == SeriesSelection::All
                            && matches!(verification.click, VerificationClick::Idle) =>
                    {
                        verification.click = VerificationClick::Move(UiAction::Sound);
                        VerificationState::AwaitPreferences
                    }
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
        VerificationClick::Move(action) => windows
            .single()
            .ok()
            .and_then(|(window, properties)| {
                buttons
                    .iter()
                    .find(|(candidate, node, _)| {
                        **candidate == action && node.size().min_element() > 0.0
                    })
                    .map(|(_, _, transform)| {
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
                    })
            })
            .unwrap_or(VerificationClick::Move(action)),
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

fn capture_companies(captured: On<ScreenshotCaptured>, mut verification: ResMut<Verification>) {
    verification.capture(&captured.image, "companies");
    verification.state = VerificationState::LeaveCompanies;
}

fn capture_verified_frame(
    captured: On<ScreenshotCaptured>,
    dashboard: Res<Dashboard>,
    mut verification: ResMut<Verification>,
) {
    let chart = dashboard.chart;
    verification.capture(&captured.image, &chart.file_label());
    match chart {
        Chart::Research => {
            verification.click = VerificationClick::Move(UiAction::Period(Period::Month));
            verification.state = VerificationState::AwaitPeriod(Period::Month);
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
        "viewer interaction and layout verification passed: separate rendering and simulation queues, native mouse clicks, reset/step/comparison feedback, audited days, household and firm inspection, refreshed inspection, company rankings, all five chart menus and 29 charts, four history periods, series focus, retained GPU chart on speed and inspection changes, previous/next navigation, footer controls, compact window"
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

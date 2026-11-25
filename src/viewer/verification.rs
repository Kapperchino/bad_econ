use super::{
    ActorSelection, Chart, Cue, Dashboard, DashboardPage, DisplayState, Launch, Period,
    SeriesSelection, Speed, UiAction, atlas, companies, feedback,
    graph::{HoverDot, PLOT_MARGIN},
    model, presentation,
    theme::{Appearance, AppearanceSettings, Surface, Theme, Tone},
    ui,
};
use crate::runtime::RuntimeState;
use anyhow::{Context, Result};
use bevy::{
    camera::RenderTarget,
    ecs::system::SystemParam,
    mesh::VertexAttributeValues,
    picking::pointer::{Location, PointerAction, PointerButton, PointerId, PointerInput},
    prelude::*,
    render::view::screenshot::{Screenshot, ScreenshotCaptured},
    ui::Checked,
    window::{PrimaryWindow, WindowRef, WindowTheme, WindowThemeChanged},
};
use std::{
    io::Write,
    time::{Duration, Instant},
};

const VERIFICATION_POINTER: PointerId = PointerId::Custom(bevy::asset::uuid::uuid!(
    "8a2262f7-77c2-4084-b0be-97881a1713ad"
));

pub(super) struct VerificationPlugin;
impl Plugin for VerificationPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(Verification {
            state: VerificationState::Start,
            deadline: Instant::now() + Duration::from_secs(120),
            click: VerificationClick::Idle,
            capture_error: None,
        })
        .add_systems(Startup, |mut commands: Commands| {
            commands.spawn(VERIFICATION_POINTER);
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
    Hover(UiAction),
    Press { window: Entity, position: Vec2 },
    Release { window: Entity, position: Vec2 },
}
#[derive(Clone, Copy, Debug)]
enum VerificationState {
    Start,
    AwaitReset,
    AwaitDay,
    FeedbackFrames(u32),
    AwaitFeedbackCapture,
    InspectHousehold,
    AwaitInspection,
    AwaitInspectionRefresh,
    AwaitComparison,
    AwaitFirmInspection,
    AwaitMapInspection(u64),
    AwaitAtlas(atlas::Layer),
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
    AwaitOverviewHover,
    AwaitOverviewLine(Vec2),
    AwaitOverviewPlayback {
        start_day: u64,
        crosshair_left: f32,
    },
    AwaitOverviewPause,
    AwaitOverviewLeave,
    AwaitMenu(Chart),
    AwaitCategoryChart(Chart),
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
    AwaitLightOverride,
    AwaitDarkOverride,
    AwaitSystemSetting,
    AwaitFinalLightOverride,
    AppearanceFrames(u32),
    AwaitAppearanceCapture,
    Complete,
}
#[derive(Clone, Copy, Debug)]
struct RetainedPresentation {
    plot: Entity,
    legend: Entity,
}

#[derive(SystemParam)]
struct VerificationPresentation<'w, 's> {
    plots: Query<'w, 's, (&'static ui::Plot, &'static ComputedNode, &'static Children)>,
    images: Query<'w, 's, &'static ImageNode>,
    image_layouts:
        Query<'w, 's, (&'static ComputedNode, &'static UiGlobalTransform), With<ImageNode>>,
    dots: Query<'w, 's, (&'static Visibility, &'static UiGlobalTransform), With<HoverDot>>,
    crosshairs: Query<
        'w,
        's,
        (
            &'static Visibility,
            &'static Node,
            &'static UiGlobalTransform,
        ),
        With<ui::Crosshair>,
    >,
    tooltip: Query<
        'w,
        's,
        (
            Entity,
            &'static Visibility,
            &'static ComputedNode,
            &'static UiGlobalTransform,
        ),
        With<ui::ChartTooltip>,
    >,
    children: Query<'w, 's, &'static Children>,
    text: Query<'w, 's, &'static Text>,
    legends: Query<'w, 's, &'static Children, With<ui::Legend>>,
    meshes: Query<'w, 's, &'static Mesh2d>,
    mesh_assets: Res<'w, Assets<Mesh>>,
    windows: Query<'w, 's, (Entity, &'static mut Window), With<PrimaryWindow>>,
    theme: Res<'w, Theme>,
    appearance: Res<'w, AppearanceSettings>,
    atlas: Res<'w, atlas::Layer>,
}
impl VerificationPresentation<'_, '_> {
    fn employment_image(&self) -> Entity {
        let (_, _, children) = self
            .plots
            .iter()
            .find(|(plot, _, _)| **plot == ui::Plot::Preview(Chart::Employment))
            .expect("employment plot");
        children
            .iter()
            .find(|child| self.images.contains(*child))
            .expect("employment chart image")
    }

    fn employment_crosshair(&self) -> (&Visibility, &Node, &UiGlobalTransform) {
        self.children
            .relationship_sources::<Children>(self.employment_image())
            .find_map(|child| self.crosshairs.get(child).ok())
            .expect("employment crosshair")
    }

    fn employment_crosshair_left(&self) -> f32 {
        let (visibility, node, _) = self.employment_crosshair();
        assert_eq!(*visibility, Visibility::Inherited);
        match node.left {
            Val::Px(left) => left,
            other => panic!("invalid crosshair position {other:?}"),
        }
    }

    fn hover_employment_line(&mut self, display: &super::Display, period: Period) -> Vec2 {
        let chart = Chart::Employment;
        let (node, transform) = self
            .image_layouts
            .get(self.employment_image())
            .expect("employment chart image");
        let range = model::ChartRange::new(
            chart,
            &display.snapshot,
            &display.history,
            display.comparison.as_ref(),
            period,
            SeriesSelection::All,
        );
        let mut points = chart.samples(&display.history, 1, 2).map(|sample| {
            Vec2::new(
                range.day_fraction(sample.day),
                1.0 - range.fraction(sample.values[3]) as f32,
            )
        });
        let from = points.next().expect("employment sample on day 1");
        let to = points.next().expect("employment sample on day 2");
        let inset = PLOT_MARGIN / node.inverse_scale_factor();
        let extent = node.size() - 2.0 * inset;
        let point = transform.translation - extent * 0.5 + extent * from.lerp(to, 0.6);
        self.windows
            .single_mut()
            .expect("primary window")
            .1
            .set_physical_cursor_position(Some(point.as_dvec2()));
        point
    }

    fn chart_lines_dimmed(&self) -> bool {
        let series = self
            .theme
            .color(Tone::Series(Chart::Employment, 0))
            .to_linear()
            .to_f32_array();
        let mesh = self.meshes.single().expect("chart mesh");
        self.mesh_assets
            .get(&mesh.0)
            .and_then(|mesh| mesh.attribute(Mesh::ATTRIBUTE_COLOR))
            .is_some_and(|colors| match colors {
                VertexAttributeValues::Float32x4(colors) => colors
                    .iter()
                    .any(|color| color[3] == 0.10 && color[..3] == series[..3]),
                _ => false,
            })
    }

    fn tooltip_above_cursor(&self) -> Option<Entity> {
        let (entity, visibility, node, transform) = self.tooltip.single().ok()?;
        let (_, window) = self.windows.single().ok()?;
        let cursor = window.physical_cursor_position()?;
        let bottom = transform.translation.y + node.size().y * 0.5;
        let gap = (cursor.y - bottom) / window.scale_factor();
        (*visibility == Visibility::Inherited && (8.0..=12.0).contains(&gap)).then_some(entity)
    }

    fn retained(&self) -> RetainedPresentation {
        assert_eq!(self.meshes.iter().count(), 1, "charts share one GPU mesh");
        let (_, _, children) = self
            .plots
            .iter()
            .find(|(plot, _, _)| **plot == ui::Plot::Preview(Chart::Employment))
            .expect("employment plot children");
        assert!(
            children.len() <= 8,
            "plot UI must remain bounded independently of history"
        );
        RetainedPresentation {
            plot: children[0],
            legend: self.legends.single().expect("legend children")[0],
        }
    }
    fn grid_plots(&self, menu: model::Menu) -> impl Iterator<Item = &Children> {
        self.plots.iter().filter_map(move |(plot, node, children)| {
            (matches!(plot, ui::Plot::Preview(chart) if chart.menu() == menu)
                && node.size().min_element() > 0.0)
                .then_some(children)
        })
    }

    fn grid_ready(&self, menu: model::Menu) -> bool {
        self.grid_plots(menu).count() == menu.charts().len()
    }

    fn assert_grid(&self, menu: model::Menu) {
        let previews = self.grid_plots(menu);
        let tiles = previews
            .map(|children| {
                children
                    .iter()
                    .find_map(|child| self.images.get(child).ok())
                    .expect("preview image")
            })
            .collect::<Vec<_>>();
        let texture = &tiles[0].image;
        for (index, tile) in tiles.iter().enumerate() {
            assert_eq!(&tile.image, texture, "preview charts share one GPU atlas");
            let bounds = tile.rect.expect("preview region");
            for other in &tiles[..index] {
                let other = other.rect.expect("preview region");
                let overlap = bounds.max.min(other.max) - bounds.min.max(other.min);
                assert!(
                    overlap.min_element() <= 0.0,
                    "preview regions must be disjoint"
                );
            }
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
    mut appearances: MessageWriter<WindowThemeChanged>,
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
                        let (window, settings) =
                            presentation.windows.single().expect("primary window");
                        assert_eq!(presentation.appearance.selection, Appearance::System);
                        assert_eq!(
                            *presentation.theme,
                            settings.window_theme.map(Theme::from).unwrap_or_default(),
                            "startup appearance should follow the primary window"
                        );
                        appearances.write(WindowThemeChanged {
                            window,
                            theme: WindowTheme::Dark,
                        });
                        verification.click = VerificationClick::Move(UiAction::Reset);
                        VerificationState::AwaitReset
                    }
                    VerificationState::AwaitReset if feedback.current() == Some(Cue::Reset) => {
                        verification.click = VerificationClick::Move(UiAction::Step);
                        VerificationState::AwaitDay
                    }
                    VerificationState::AwaitDay if feedback.current() == Some(Cue::Stepped) => {
                        assert_eq!(display.snapshot.day, 1);
                        VerificationState::FeedbackFrames(6)
                    }
                    VerificationState::FeedbackFrames(0) => {
                        commands
                            .spawn(Screenshot::primary_window())
                            .observe(capture_step_feedback);
                        VerificationState::AwaitFeedbackCapture
                    }
                    VerificationState::FeedbackFrames(frames) => {
                        VerificationState::FeedbackFrames(frames - 1)
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
                        verification.click =
                            VerificationClick::Move(UiAction::Atlas(atlas::Layer::Shortage));
                        VerificationState::AwaitAtlas(atlas::Layer::Shortage)
                    }
                    VerificationState::AwaitAtlas(layer)
                        if *presentation.atlas == layer
                            && matches!(verification.click, VerificationClick::Idle) =>
                    {
                        match layer {
                            atlas::Layer::Shortage => {
                                verification.click =
                                    VerificationClick::Move(UiAction::Atlas(atlas::Layer::Sites));
                                VerificationState::AwaitAtlas(atlas::Layer::Sites)
                            }
                            atlas::Layer::Sites => {
                                verification.click =
                                    VerificationClick::Move(UiAction::Speed(Speed::Slow));
                                VerificationState::AwaitSpeed(presentation.retained())
                            }
                        }
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
                        if dashboard.page.company_page().is_some()
                            && matches!(verification.click, VerificationClick::Idle) =>
                    {
                        verification.click =
                            VerificationClick::Move(UiAction::Rank(companies::Ranking::Profit));
                        VerificationState::AwaitCompanyRanking
                    }
                    VerificationState::AwaitCompanyRanking
                        if dashboard
                            .page
                            .company_page()
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
                        if dashboard.page == DashboardPage::Metrics
                            && presentation.grid_ready(model::Menu::Overview)
                            && matches!(verification.click, VerificationClick::Idle) =>
                    {
                        presentation.assert_grid(model::Menu::Overview);
                        verification.click =
                            VerificationClick::Hover(UiAction::Chart(Chart::Employment));
                        VerificationState::AwaitOverviewHover
                    }
                    VerificationState::AwaitOverviewHover
                        if presentation
                            .crosshairs
                            .iter()
                            .any(|(visibility, _, _)| *visibility == Visibility::Inherited) =>
                    {
                        assert_eq!(
                            presentation
                                .crosshairs
                                .iter()
                                .filter(|(visibility, _, _)| **visibility == Visibility::Inherited)
                                .count(),
                            4,
                            "synchronized crosshair on each overview chart"
                        );
                        let point = presentation.hover_employment_line(&display, dashboard.period);
                        VerificationState::AwaitOverviewLine(point)
                    }
                    VerificationState::AwaitOverviewLine(point)
                        if let Some(tooltip) = presentation.tooltip_above_cursor()
                            && let Some((_, dot)) =
                                presentation.dots.iter().find(|(visibility, transform)| {
                                    **visibility == Visibility::Inherited
                                        && transform.translation.distance(point) <= 2.0
                                }) =>
                    {
                        let (_, _, crosshair) = presentation.employment_crosshair();
                        assert!(
                            (crosshair.translation.x - dot.translation.x).abs() <= 1.0,
                            "crosshair and dot align"
                        );
                        let text = presentation
                            .children
                            .iter_descendants::<Children>(tooltip)
                            .filter_map(|entity| presentation.text.get(entity).ok())
                            .map(|text| text.0.as_str())
                            .collect::<Vec<_>>();
                        let history = match text.first().copied() {
                            Some("Day 2 · count") => &display.history,
                            Some("Day 2 · count · Reference") => {
                                &display.comparison.as_ref().expect("comparison run").history
                            }
                            header => panic!("unexpected tooltip header {header:?}"),
                        };
                        let value = Chart::Employment
                            .samples(history, 2, 2)
                            .next()
                            .expect("hovered employment sample")
                            .values[3];
                        assert_eq!(
                            text[1..],
                            ["Farmers without paid work", &format!("{value:.0}")]
                        );
                        assert!(
                            presentation
                                .text
                                .iter()
                                .all(|text| !text.0.starts_with("Cursor  ·"))
                        );
                        assert!(presentation.chart_lines_dimmed(), "other chart lines fade");
                        verification.click = VerificationClick::Move(UiAction::ToggleRun);
                        VerificationState::AwaitOverviewPlayback {
                            start_day: display.snapshot.day,
                            crosshair_left: presentation.employment_crosshair_left(),
                        }
                    }
                    VerificationState::AwaitOverviewPlayback {
                        start_day,
                        crosshair_left,
                    } => {
                        let current = presentation.employment_crosshair_left();
                        assert!(
                            (current - crosshair_left).abs() <= 12.0,
                            "crosshair stays near the cursor during playback"
                        );
                        match display.snapshot.day.saturating_sub(start_day) >= 2 {
                            true => {
                                verification.click = VerificationClick::Move(UiAction::ToggleRun);
                                VerificationState::AwaitOverviewPause
                            }
                            false => verification.state,
                        }
                    }
                    VerificationState::AwaitOverviewPause
                        if matches!(display.status, DisplayState::Active(RuntimeState::Paused))
                            && matches!(verification.click, VerificationClick::Idle) =>
                    {
                        presentation
                            .windows
                            .single_mut()
                            .expect("primary window")
                            .1
                            .set_physical_cursor_position(None);
                        VerificationState::AwaitOverviewLeave
                    }
                    VerificationState::AwaitOverviewLeave
                        if presentation
                            .tooltip
                            .single()
                            .is_ok_and(|(_, visibility, _, _)| {
                                *visibility == Visibility::Hidden
                            }) =>
                    {
                        assert!(
                            !presentation.chart_lines_dimmed(),
                            "chart opacity is restored"
                        );
                        assert!(
                            presentation
                                .dots
                                .iter()
                                .all(|(visibility, _)| *visibility == Visibility::Hidden)
                        );
                        verification.click =
                            VerificationClick::Move(UiAction::Chart(Chart::Employment));
                        VerificationState::AwaitChart(Chart::Employment)
                    }
                    VerificationState::AwaitMenu(chart)
                        if dashboard.page == DashboardPage::Metrics
                            && dashboard.chart.menu() == chart.menu()
                            && presentation.grid_ready(chart.menu())
                            && matches!(verification.click, VerificationClick::Idle) =>
                    {
                        presentation.assert_grid(chart.menu());
                        let first = chart.menu().charts()[0];
                        verification.click = VerificationClick::Move(UiAction::Chart(first));
                        match first == chart {
                            true => VerificationState::AwaitChart(chart),
                            false => VerificationState::AwaitCategoryChart(chart),
                        }
                    }
                    VerificationState::AwaitCategoryChart(chart)
                        if dashboard.page == DashboardPage::Chart
                            && matches!(verification.click, VerificationClick::Idle) =>
                    {
                        verification.click = VerificationClick::Move(UiAction::Chart(chart));
                        VerificationState::AwaitChart(chart)
                    }
                    VerificationState::AwaitChart(chart)
                        if dashboard.page == DashboardPage::Chart
                            && dashboard.chart == chart
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
                                verification.click =
                                    VerificationClick::Move(UiAction::FocusSeries(0));
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
                            .1
                            .resolution
                            .set(1280.0, 800.0);
                        VerificationState::CompactFrames(24)
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
                    VerificationState::AwaitLightOverride
                        if presentation.appearance.selection
                            == Appearance::Override(Theme::Light)
                            && matches!(verification.click, VerificationClick::Idle) =>
                    {
                        assert_eq!(*presentation.theme, Theme::Light);
                        assert_eq!(presentation.appearance.system, Theme::Dark);
                        verification.click = VerificationClick::Move(UiAction::ToggleDarkMode);
                        VerificationState::AwaitDarkOverride
                    }
                    VerificationState::AwaitDarkOverride
                        if presentation.appearance.selection
                            == Appearance::Override(Theme::Dark)
                            && matches!(verification.click, VerificationClick::Idle) =>
                    {
                        assert_eq!(*presentation.theme, Theme::Dark);
                        appearances.write(WindowThemeChanged {
                            window: presentation.windows.single().expect("primary window").0,
                            theme: WindowTheme::Light,
                        });
                        VerificationState::AwaitSystemSetting
                    }
                    VerificationState::AwaitSystemSetting
                        if presentation.appearance.system == Theme::Light =>
                    {
                        assert_eq!(*presentation.theme, Theme::Dark);
                        verification.click = VerificationClick::Move(UiAction::ToggleDarkMode);
                        VerificationState::AwaitFinalLightOverride
                    }
                    VerificationState::AwaitFinalLightOverride
                        if presentation.appearance.selection
                            == Appearance::Override(Theme::Light)
                            && matches!(verification.click, VerificationClick::Idle) =>
                    {
                        assert_eq!(*presentation.theme, Theme::Light);
                        VerificationState::AppearanceFrames(12)
                    }
                    VerificationState::AppearanceFrames(0) => {
                        commands
                            .spawn(Screenshot::primary_window())
                            .observe(capture_light_frame);
                        VerificationState::AwaitAppearanceCapture
                    }
                    VerificationState::AppearanceFrames(frames) => {
                        VerificationState::AppearanceFrames(frames - 1)
                    }
                    state => state,
                };
            }
        }
    }
}

fn verify_clicks(
    mut verification: ResMut<Verification>,
    dashboard: Res<Dashboard>,
    buttons: Query<(
        &UiAction,
        &ComputedNode,
        &UiGlobalTransform,
        Has<ui::ChartTab>,
    )>,
    mut windows: Query<(Entity, &mut Window), With<PrimaryWindow>>,
    mut input: MessageWriter<PointerInput>,
) {
    verification.click = match verification.click {
        VerificationClick::Move(action) | VerificationClick::Hover(action) => windows
            .single_mut()
            .ok()
            .and_then(|(window, mut properties)| {
                buttons
                    .iter()
                    .find(|(candidate, node, _, tab)| {
                        let target = match action {
                            UiAction::Chart(_) => *tab == (dashboard.page == DashboardPage::Chart),
                            _ => true,
                        };
                        target && **candidate == action && node.size().min_element() > 0.0
                    })
                    .map(|(_, _, transform, _)| {
                        let position = transform.translation / properties.scale_factor();
                        assert!(
                            Rect::from_corners(Vec2::ZERO, properties.size()).contains(position),
                            "{action:?} lies outside the window at {position:?}"
                        );
                        input.write(pointer_input(
                            window,
                            position,
                            PointerAction::Move { delta: Vec2::ZERO },
                        ));
                        match verification.click {
                            VerificationClick::Hover(_) => {
                                properties.set_physical_cursor_position(Some(
                                    transform.translation.as_dvec2(),
                                ));
                                VerificationClick::Idle
                            }
                            _ => VerificationClick::Press { window, position },
                        }
                    })
            })
            .unwrap_or(verification.click),
        VerificationClick::Press { window, position } => {
            input.write(pointer_input(
                window,
                position,
                PointerAction::Press(PointerButton::Primary),
            ));
            VerificationClick::Release { window, position }
        }
        VerificationClick::Release { window, position } => {
            input.write(pointer_input(
                window,
                position,
                PointerAction::Release(PointerButton::Primary),
            ));
            VerificationClick::Idle
        }
        VerificationClick::Idle => VerificationClick::Idle,
    };
}

fn pointer_input(window: Entity, position: Vec2, action: PointerAction) -> PointerInput {
    let target = RenderTarget::Window(WindowRef::Entity(window))
        .normalize(Some(window))
        .expect("verification window target");
    PointerInput::new(VERIFICATION_POINTER, Location { target, position }, action)
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
            (verification.click, verification.state) = match next.menu() == chart.menu() {
                true => (
                    VerificationClick::Move(UiAction::Chart(next)),
                    VerificationState::AwaitChart(next),
                ),
                false => (
                    VerificationClick::Move(UiAction::Menu(next.menu())),
                    VerificationState::AwaitMenu(next),
                ),
            };
        }
    }
}

#[derive(SystemParam)]
struct RenderedAppearance<'w, 's> {
    theme: Res<'w, Theme>,
    surfaces: Query<'w, 's, (&'static Surface, &'static BackgroundColor)>,
    switch: Query<'w, 's, Has<Checked>, With<ui::ThemeSwitch>>,
    cameras: Query<'w, 's, (&'static Camera, &'static RenderTarget)>,
}

impl RenderedAppearance<'_, '_> {
    fn verify(&self, expected: Theme) {
        assert_eq!(*self.theme, expected, "viewer appearance");
        assert_eq!(
            self.switch.single().expect("dark mode switch"),
            expected == Theme::Dark
        );
        assert!(
            self.surfaces
                .iter()
                .all(|(surface, color)| color.0 == expected.color(surface.0))
        );
        assert!(self.cameras.iter().any(|(camera, target)| {
            matches!(target, RenderTarget::Image(_))
            && matches!(camera.clear_color, ClearColorConfig::Custom(color) if color == expected.color(Tone::Panel))
        }), "chart render target should follow the viewer appearance");
    }
}

fn capture_compact_frame(
    captured: On<ScreenshotCaptured>,
    buttons: Query<(&UiAction, &ComputedNode, &UiGlobalTransform)>,
    mut verification: ResMut<Verification>,
    appearance: RenderedAppearance,
) {
    appearance.verify(Theme::Dark);
    verification.capture(&captured.image, "compact");
    let viewport = Rect::from_corners(Vec2::ZERO, captured.image.size_f32());
    buttons
        .iter()
        .filter(|(_, node, _)| node.size().min_element() > 0.0)
        .for_each(|(action, node, transform)| {
            assert!(
                viewport.contains(transform.translation - node.size() / 2.0),
                "{action:?} begins outside the compact window"
            );
            let end = transform.translation + node.size() / 2.0;
            assert!(
                viewport.contains(end),
                "{action:?} ends at {end:?} outside {viewport:?}"
            );
        });
    verification.click = VerificationClick::Move(UiAction::ToggleDarkMode);
    verification.state = VerificationState::AwaitLightOverride;
}

fn capture_light_frame(
    captured: On<ScreenshotCaptured>,
    appearance: RenderedAppearance,
    mut verification: ResMut<Verification>,
    mut exit: MessageWriter<AppExit>,
) {
    appearance.verify(Theme::Light);
    verification.capture(&captured.image, "light");
    eprintln!(
        "viewer interaction and layout verification passed: separate rendering and simulation queues, UI pointer clicks, reset/step/comparison feedback, audited days, household and firm inspection, refreshed inspection, company rankings, both atlas layers, five dashboard grids, all 29 metric detail views, synchronized hover and value tooltip, four shared history periods, series focus, retained GPU chart on speed and inspection changes, previous/next navigation, footer controls, compact window, system appearance and manual overrides"
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

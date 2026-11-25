use super::{
    Chart, Dashboard, DashboardPage, Display, atlas,
    graph::{
        self, GraphPanel, GraphRenderer, HoverDot, LineHit, LineObservation, LineRun, PLOT_MARGIN,
    },
    model::{ChartRange, Menu, Period, SeriesSelection, ValueFormat},
    theme::{Theme, Tone},
    ui::{
        self, ActorSelection, ChartTab, ChartTooltip, Crosshair, FirmMarker, Label, Legend,
        MapMarkers, Plot, UiAction,
    },
};
use crate::{
    domain::{Good, ObservationSnapshot, calendar::CalendarDate},
    runtime::RuntimeState,
};
use bevy::{
    ecs::system::SystemParam,
    picking::{Pickable, hover::Hovered},
    prelude::*,
    ui::RelativeCursorPosition,
    ui_widgets::Button,
    window::PrimaryWindow,
};
use std::collections::BTreeMap;

pub(super) struct PresentationPlugin;
impl Plugin for PresentationPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<PlotState>()
            .add_systems(Startup, graph::setup)
            .add_systems(
                Update,
                (
                    draw.run_if(needs_refresh),
                    (draw_plot, draw_hover_dots, draw_cursor).chain(),
                    draw_legend.run_if(chart_changed),
                    draw_markers.run_if(map_changed),
                )
                    .after(super::controls)
                    .in_set(PresentationSystems),
            );
    }
}

#[derive(SystemSet, Debug, Hash, PartialEq, Eq, Clone)]
pub(super) struct PresentationSystems;

const AXIS_WIDTH: f32 = 48.0;
const DATE_HEIGHT: f32 = 20.0;
const TOP_PADDING: f32 = 5.0;
const CROSSHAIR_WIDTH: f32 = 1.0;
const HOVER_DOT_RADIUS: f32 = 4.5;

#[derive(Resource, Default)]
struct PlotState {
    version: Option<PlotVersion>,
    atlas: PlotAtlas,
    hover: Option<PlotHover>,
}

#[derive(Default)]
struct PlotAtlas {
    panels: Vec<GraphPanel>,
    pixels: UVec2,
}

struct PlotHover {
    plot: Entity,
    fraction: f32,
    hit: Option<LineHit>,
}

impl PlotHover {
    fn observation(&self) -> Option<LineObservation> {
        self.hit.as_ref().map(|hit| hit.observation)
    }
}

#[derive(PartialEq)]
struct PlotVersion {
    revision: u64,
    theme: Theme,
    period: Period,
    selection: SeriesSelection,
    plots: Vec<PlotLayout>,
}

#[derive(PartialEq)]
struct PlotLayout {
    entity: Entity,
    chart: Chart,
    scale: f32,
    size: UVec2,
}

impl PlotLayout {
    fn area(&self) -> Vec2 {
        self.size.as_vec2() * self.scale
    }

    fn plot_width(&self) -> f32 {
        (self.area().x - 2.0 * PLOT_MARGIN).max(0.0)
    }

    fn fraction_at(&self, x: f32) -> f32 {
        ((x - PLOT_MARGIN) / self.plot_width().max(f32::EPSILON)).clamp(0.0, 1.0)
    }

    fn x_at(&self, fraction: f32) -> f32 {
        PLOT_MARGIN + fraction * self.plot_width()
    }
}

#[derive(Clone, Copy, PartialEq)]
struct LabelVersion {
    revision: u64,
    chart: Chart,
}

fn needs_refresh(
    dashboard: Option<Res<Dashboard>>,
    mut previous: Local<Option<LabelVersion>>,
) -> bool {
    let current = dashboard.map(|dashboard| LabelVersion {
        revision: dashboard.data.lock().expect("dashboard lock").revision,
        chart: dashboard.chart,
    });
    let changed = current != *previous;
    *previous = current;
    changed
}

fn draw(
    dashboard: Option<Res<Dashboard>>,
    mut labels: Query<(&Label, &mut Text)>,
    mut tabs: Query<(&ChartTab, &mut Node)>,
) {
    if let Some(dashboard) = dashboard {
        let display = dashboard.data.lock().expect("dashboard lock");
        let s = &display.snapshot;
        let chart = dashboard.chart;
        let menu = chart.menu();
        let metrics = metric_values(menu, s);
        let prior = display
            .history
            .iter()
            .rev()
            .find(|point| point.day < s.day)
            .map(|point| metric_values(menu, point));
        labels.iter_mut().for_each(|(kind, mut label)| {
            let value = match kind {
                Label::Seed => format!("Seed {}", display.seed),
                Label::Date => format!(
                    "Day {}   ·   {}   ·   {}°C",
                    s.day, s.weather.date, s.weather.temperature_celsius
                ),
                Label::Status => match &display.status {
                    super::model::DisplayState::Initializing => "Starting",
                    super::model::DisplayState::Active(RuntimeState::Paused) => "Paused",
                    super::model::DisplayState::Active(RuntimeState::Running) => "Running",
                    super::model::DisplayState::Active(RuntimeState::Advancing(_)) => "Advancing",
                    _ => "Fault",
                }
                .into(),
                Label::Title => menu.label().into(),
                Label::Subtitle => menu.description().into(),
                Label::Summary => summary(menu, s),
                Label::ChartTitle => chart.label().into(),
                Label::ChartDescription => chart.description().into(),
                Label::Comparison => comparison_text(&display),
                Label::MetricName(index) => metrics[*index].name.into(),
                Label::MetricValue(index) => match display.history.is_empty() {
                    true => "—".into(),
                    false => number(metrics[*index].value),
                },
                Label::MetricChange(index) => prior.as_ref().map_or_else(
                    || metrics[*index].unit.into(),
                    |prior| {
                        let change = metrics[*index].value as i128 - prior[*index].value as i128;
                        format!(
                            "{}{} vs prior day",
                            match change.cmp(&0) {
                                std::cmp::Ordering::Greater => "+",
                                std::cmp::Ordering::Less => "−",
                                std::cmp::Ordering::Equal => "",
                            },
                            number(change.unsigned_abs())
                        )
                    },
                ),
                Label::Inspection => inspection_text(&display),
                Label::MapSummary => format!(
                    "{} open sites · select a marker to inspect",
                    number(s.firms)
                ),
                Label::Error => display
                    .error
                    .clone()
                    .unwrap_or_else(|| match &display.status {
                        super::model::DisplayState::Unavailable(message) => message.clone(),
                        _ => String::new(),
                    }),
                Label::ToggleRun => match display.status {
                    super::model::DisplayState::Active(RuntimeState::Running) => "Pause".into(),
                    _ => "Resume".into(),
                },
            };
            if label.0 != value {
                label.0 = value;
            }
        });
        tabs.iter_mut().for_each(|(tab, mut node)| {
            let visibility = match tab.0.menu() == chart.menu() {
                true => bevy::ui::Display::Flex,
                false => bevy::ui::Display::None,
            };
            if node.display != visibility {
                node.display = visibility;
            };
        });
    }
}

fn active_chart(plot: Plot, dashboard: &Dashboard) -> Option<Chart> {
    match (dashboard.page, plot) {
        (DashboardPage::Metrics, Plot::Preview(chart))
            if chart.menu() == dashboard.chart.menu() =>
        {
            Some(chart)
        }
        (DashboardPage::Chart, Plot::Detail) => Some(dashboard.chart),
        _ => None,
    }
}

fn chart_changed(dashboard: Option<Res<Dashboard>>, mut previous: Local<Option<Chart>>) -> bool {
    let current = dashboard.map(|dashboard| dashboard.chart);
    let changed = current != *previous;
    *previous = current;
    changed
}

fn map_changed(dashboard: Option<Res<Dashboard>>, mut previous: Local<Option<u64>>) -> bool {
    let current =
        dashboard.map(|dashboard| dashboard.data.lock().expect("dashboard lock").map_revision);
    let changed = current != *previous;
    *previous = current;
    changed
}

fn draw_legend(
    dashboard: Res<Dashboard>,
    legends: Query<Entity, With<Legend>>,
    mut commands: Commands,
) {
    if let Ok(entity) = legends.single() {
        commands
            .entity(entity)
            .despawn_children()
            .with_children(|legend| {
                ui::button(legend, UiAction::AllSeries, "All series");
                for (index, name) in dashboard.chart.series().into_iter().enumerate() {
                    ui::series_button(legend, index, &name, Tone::Series(dashboard.chart, index));
                }
            });
    }
}

fn draw_plot(
    dashboard: Option<Res<Dashboard>>,
    theme: Option<Res<Theme>>,
    plots: Query<(Entity, &Plot, &ComputedNode, &RelativeCursorPosition)>,
    mut renderer: GraphRenderer,
    mut state: ResMut<PlotState>,
    mut commands: Commands,
) {
    if let Some(dashboard) = dashboard {
        let mut layouts = plots
            .iter()
            .filter_map(|(entity, plot, node, _)| {
                active_chart(*plot, &dashboard)
                    .filter(|_| node.size().min_element() > 0.0)
                    .map(|chart| {
                        let scale = node.inverse_scale_factor();
                        let size =
                            node.size() * scale - Vec2::new(AXIS_WIDTH, DATE_HEIGHT + TOP_PADDING);
                        PlotLayout {
                            entity,
                            chart,
                            scale,
                            size: (size / scale).floor().as_uvec2().max(UVec2::ONE),
                        }
                    })
            })
            .collect::<Vec<_>>();
        layouts.sort_unstable_by_key(|plot| plot.entity);
        let display = dashboard.data.lock().expect("dashboard lock");
        let version = PlotVersion {
            theme: theme.as_deref().copied().unwrap_or_default(),
            revision: display.observation_revision,
            period: dashboard.period,
            selection: dashboard.series,
            plots: layouts,
        };
        let changed = state.version.as_ref() != Some(&version);
        if changed {
            state.atlas = plot_atlas(&display, &version);
            paint_plots(
                &display,
                &version,
                &state.atlas.panels,
                renderer.image(),
                &mut commands,
            );
        }
        let hover = version
            .plots
            .iter()
            .zip(&state.atlas.panels)
            .find_map(|(layout, panel)| {
                let (_, _, node, cursor) = plots.get(layout.entity).ok()?;
                let position = cursor_in_plot(cursor, node)?;
                let hit = panel.hit(&display, position, layout.scale);
                let x = hit.as_ref().map_or(position.x, |hit| hit.point.x);
                Some(PlotHover {
                    plot: layout.entity,
                    fraction: layout.fraction_at(x),
                    hit,
                })
            });
        let hovered = hover
            .as_ref()
            .and_then(PlotHover::observation)
            .map(|observation| observation.line);
        let previous = state
            .hover
            .as_ref()
            .and_then(PlotHover::observation)
            .map(|observation| observation.line);
        if let Some(layout) = version
            .plots
            .first()
            .filter(|_| changed || hovered != previous)
        {
            renderer.update(
                &display,
                &state.atlas.panels,
                state.atlas.pixels,
                1.0 / layout.scale,
                version.theme,
                hovered,
            );
        }
        state.hover = hover;
        state.version = Some(version);
    }
}

fn draw_hover_dots(
    state: Res<PlotState>,
    mut dots: Query<(&ChildOf, &mut Node, &mut Visibility, &mut BackgroundColor), With<HoverDot>>,
) {
    if let Some(version) = &state.version {
        for (parent, mut node, mut visibility, mut background) in &mut dots {
            match state
                .hover
                .as_ref()
                .filter(|hover| hover.plot == parent.parent())
                .and_then(|hover| hover.hit.as_ref())
            {
                Some(hit) => {
                    let point = hit.point + Vec2::new(AXIS_WIDTH, TOP_PADDING) - HOVER_DOT_RADIUS;
                    let (left, top) = (px(point.x), px(point.y));
                    if (node.left, node.top) != (left, top) {
                        node.left = left;
                        node.top = top;
                    }
                    let line = hit.observation.line;
                    background.set_if_neq(BackgroundColor(
                        version.theme.color(Tone::Series(line.chart, line.series)),
                    ));
                    visibility.set_if_neq(Visibility::Inherited);
                }
                None => {
                    visibility.set_if_neq(Visibility::Hidden);
                }
            }
        }
    }
}

fn plot_atlas(display: &Display, version: &PlotVersion) -> PlotAtlas {
    let tile = version
        .plots
        .iter()
        .fold(UVec2::ONE, |tile, plot| tile.max(plot.size));
    let columns = version.plots.len().clamp(1, 2);
    let rows = version.plots.len().div_ceil(columns);
    let panels = version
        .plots
        .iter()
        .enumerate()
        .map(|(index, plot)| {
            let origin = tile * UVec2::new((index % columns) as u32, (index / columns) as u32);
            GraphPanel {
                chart: plot.chart,
                range: ChartRange::new(
                    plot.chart,
                    &display.snapshot,
                    &display.history,
                    display.comparison.as_ref(),
                    version.period,
                    version.selection,
                ),
                pixels: URect::from_corners(origin, origin + plot.size),
            }
        })
        .collect::<Vec<_>>();
    PlotAtlas {
        panels,
        pixels: tile * UVec2::new(columns as u32, rows as u32),
    }
}

fn paint_plots(
    display: &Display,
    version: &PlotVersion,
    panels: &[GraphPanel],
    texture: Handle<Image>,
    commands: &mut Commands,
) {
    for (layout, panel) in version.plots.iter().zip(panels) {
        let area = layout.area();
        let rect = Rect::from_corners(panel.pixels.min.as_vec2(), panel.pixels.max.as_vec2());
        commands
            .entity(layout.entity)
            .despawn_children()
            .with_children(|plot| {
                draw_axes(plot, &panel.range, area);
                let mut image = ImageNode::new(texture.clone());
                image.rect = Some(rect);
                plot.spawn((
                    Node {
                        position_type: PositionType::Absolute,
                        left: px(AXIS_WIDTH),
                        top: px(TOP_PADDING),
                        width: px(area.x),
                        height: px(area.y),
                        ..default()
                    },
                    image,
                    Pickable::IGNORE,
                ))
                .with_children(|graph| {
                    graph.spawn((
                        Crosshair,
                        Visibility::Hidden,
                        Node {
                            position_type: PositionType::Absolute,
                            top: px(PLOT_MARGIN),
                            bottom: px(PLOT_MARGIN),
                            width: px(CROSSHAIR_WIDTH),
                            ..default()
                        },
                        Tone::Crosshair.surface(),
                        Pickable::IGNORE,
                    ));
                });
                plot.spawn((
                    HoverDot,
                    Visibility::Hidden,
                    Node {
                        position_type: PositionType::Absolute,
                        width: px(2.0 * HOVER_DOT_RADIUS),
                        height: px(2.0 * HOVER_DOT_RADIUS),
                        border: UiRect::all(px(1)),
                        border_radius: BorderRadius::MAX,
                        ..default()
                    },
                    BackgroundColor::default(),
                    Tone::Panel.outline(),
                    Pickable::IGNORE,
                ));
                if display.history.is_empty() {
                    plot.spawn((
                        Node {
                            position_type: PositionType::Absolute,
                            left: px(AXIS_WIDTH + 12.0),
                            top: px(TOP_PADDING + 12.0),
                            ..default()
                        },
                        ui::text("Waiting for observations…", 12.0, Tone::Muted),
                    ));
                }
            });
    }
}

fn draw_markers(
    dashboard: Option<Res<Dashboard>>,
    containers: Query<Entity, With<MapMarkers>>,
    mut markers: Query<(Entity, &FirmMarker, &mut Node, &mut ImageNode)>,
    mut commands: Commands,
) {
    if let (Some(dashboard), Ok(container)) = (dashboard, containers.single()) {
        let display = dashboard.data.lock().expect("dashboard lock");
        let mut firms = display
            .firms
            .iter()
            .map(|firm| (firm.identity, firm))
            .collect::<BTreeMap<_, _>>();
        let coordinate =
            |value: u32| percent(value as f32 / display.map_size.max(1) as f32 * 100.0);
        markers
            .iter_mut()
            .for_each(|(entity, marker, mut node, mut image)| {
                match firms.remove(&marker.identity) {
                    Some(firm) => {
                        let left = coordinate(firm.position.x);
                        let top = coordinate(firm.position.y);
                        if (node.left, node.top) != (left, top) {
                            node.left = left;
                            node.top = top;
                        }
                        let symbol = atlas::Symbol::for_site(firm.kind).image();
                        if image.rect != symbol.rect {
                            image.rect = symbol.rect;
                        }
                    }
                    None => {
                        commands.entity(entity).despawn();
                    }
                }
            });
        firms.into_values().for_each(|firm| {
            commands.spawn((
                ChildOf(container),
                FirmMarker {
                    identity: firm.identity,
                },
                Button,
                Hovered::default(),
                UiAction::Inspect(ActorSelection::Identity(firm.identity)),
                Node {
                    position_type: PositionType::Absolute,
                    left: coordinate(firm.position.x),
                    top: coordinate(firm.position.y),
                    width: px(17),
                    height: px(17),
                    border: UiRect::all(px(1)),
                    border_radius: BorderRadius::MAX,
                    ..default()
                },
                UiTransform::from_translation(Val2::percent(-50, -50)),
                atlas::Symbol::for_site(firm.kind).image(),
                BackgroundColor::default(),
                BorderColor::default(),
            ));
        });
    }
}

fn number(value: impl Into<u128>) -> String {
    let value = value.into();
    match value {
        0..1_000 => value.to_string(),
        1_000..1_000_000 => format!("{:.1}K", value as f64 / 1_000.0),
        1_000_000..1_000_000_000 => format!("{:.2}M", value as f64 / 1_000_000.0),
        _ => format!("{:.2}B", value as f64 / 1_000_000_000.0),
    }
}

#[derive(Clone, Copy)]
struct MetricValue {
    name: &'static str,
    unit: &'static str,
    value: u128,
}

fn metric_values(menu: Menu, s: &ObservationSnapshot) -> [MetricValue; 4] {
    let metric = |name, unit, value| MetricValue { name, unit, value };
    let people = |name, value| metric(name, "people", u128::from(value));
    let cash = |name, value| metric(name, "cash units", u128::from(value));
    match menu {
        Menu::Overview => [
            people("Population", s.population.living),
            people("Paid workers", s.employed),
            metric("Open sites", "sites", u128::from(s.firms)),
            cash("Total cash", s.cash),
        ],
        Menu::Market => [
            metric(
                "Available stock",
                "goods ticks",
                s.inventories.iter().map(|&value| u128::from(value)).sum(),
            ),
            metric(
                "In transit",
                "goods ticks",
                s.transit.iter().map(|&value| u128::from(value)).sum(),
            ),
            metric("Unmet bread", "bread ticks", u128::from(s.shortages)),
            metric("Distributors", "sites", u128::from(s.distributors)),
        ],
        Menu::Industry => [
            metric("Open sites", "sites", u128::from(s.firms)),
            metric("Site entries", "sites to date", u128::from(s.firm_entries)),
            metric(
                "Site closures",
                "sites to date",
                u128::from(s.firm_closures),
            ),
            metric("Discoveries", "to date", u128::from(s.research.discoveries)),
        ],
        Menu::Population => [
            people("Living population", s.population.living),
            people("Dependents", s.population.dependents),
            people("Births to date", s.population.births),
            metric(
                "Housing shortage",
                "accounts",
                u128::from(s.homeless_households),
            ),
        ],
        Menu::Finance => [
            metric("Active loans", "loans", u128::from(s.active_loans)),
            cash("Loan principal", s.loan_principal),
            cash("Rent arrears", s.rent_arrears),
            cash("Payroll reserves", s.payroll_reserves),
        ],
    }
}

fn summary(menu: Menu, s: &ObservationSnapshot) -> String {
    match menu {
        Menu::Overview => format!(
            "Paid workers    {}\nUnmet bread    {} ticks\nTotal cash    {}\nIn transit    {} goods ticks",
            number(s.employed),
            number(s.shortages),
            s.cash,
            number(
                s.transit
                    .iter()
                    .map(|&value| u128::from(value))
                    .sum::<u128>()
            )
        ),
        Menu::Market => format!(
            "Available stocks    {} goods ticks\nGrain / flour / bread    {} / {} / {}\nHousehold food output    {:?}\nDistributors    {}\nTransport company workers    {}\nCartage receipts    {}\nFreight awaiting pickup    {}",
            number(
                s.inventories
                    .iter()
                    .map(|&value| u128::from(value))
                    .sum::<u128>()
            ),
            number(s.inventories[0]),
            number(s.inventories[1]),
            number(s.inventories[2]),
            &s.household_production[..3],
            number(s.distributors),
            number(s.economy.transport_workers),
            number(s.economy.cartage_receipts),
            number(s.economy.awaiting_pickup)
        ),
        Menu::Industry => format!(
            "Open business sites    {}\nEntries / closures to date    {} / {}\nResearch attempts / discoveries    {} / {}\nStriking    {}",
            number(s.firms),
            number(s.firm_entries),
            number(s.firm_closures),
            number(s.research.attempts),
            number(s.research.discoveries),
            number(s.research.striking)
        ),
        Menu::Population => format!(
            "Population / dependents    {} / {}\nBirths    {}\nDeaths: background / work / famine / cold    {:?}\nCold exposure    {:.1} person-days\nAccounts short of housing    {}",
            number(s.population.living),
            number(s.population.dependents),
            number(s.population.births),
            s.population.deaths,
            s.population.cold_exposure as f64 / 1000.0,
            number(s.homeless_households)
        ),
        Menu::Finance => format!(
            "Bank & support cash    {}\nLoan principal / interest    {} / {}\nIncome tax / employee charges    {} / {}\nPayroll reserves    {}\nRent collected / arrears    {} / {}",
            s.financial_cash,
            s.loan_principal,
            s.loan_interest,
            s.income_tax,
            s.employee_charges,
            s.payroll_reserves,
            s.rent_collected,
            s.rent_arrears
        ),
    }
}

fn comparison_text(display: &Display) -> String {
    match &display.comparison {
        Some(reference) => {
            let current = &display.snapshot;
            let difference = reference
                .at_day(current.day)
                .map_or_else(String::new, |prior| {
                    let workers = i128::from(current.employed) - i128::from(prior.employed);
                    let shortages = i128::from(current.shortages) - i128::from(prior.shortages);
                    format!("  ·  Paid workers {workers:+}, unmet bread {shortages:+}")
                });
            format!(
                "Seed {} (solid)  ·  Seed {} (faint, through day {}){difference}",
                display.seed, reference.seed, reference.horizon
            )
        }
        None => "Current run".into(),
    }
}

fn axis_value(value: f64) -> String {
    match value.abs() {
        1_000.0.. => format!(
            "{}{}",
            if value < 0.0 { "−" } else { "" },
            number(value.abs() as u128)
        ),
        10.0.. => format!("{value:.0}"),
        _ => format!("{value:.1}"),
    }
}

fn draw_axes(parent: &mut ChildSpawnerCommands, range: &ChartRange, area: Vec2) {
    let extent = area.y - 2.0 * PLOT_MARGIN;
    for row in 0..=4 {
        let fraction = row as f32 / 4.0;
        let value = range.maximum - (range.maximum - range.minimum) * f64::from(fraction);
        parent.spawn((
            Node {
                position_type: PositionType::Absolute,
                left: px(0),
                top: px(TOP_PADDING + PLOT_MARGIN + fraction * extent - 7.0),
                ..default()
            },
            ui::text(axis_value(value), 10.0, Tone::Muted),
        ));
    }
    let labels = match range.first_day == range.last_day {
        true => vec![range.last_day],
        false => vec![range.first_day, range.last_day],
    };
    parent
        .spawn(Node {
            position_type: PositionType::Absolute,
            left: px(AXIS_WIDTH + PLOT_MARGIN),
            bottom: px(0),
            width: px(area.x - 2.0 * PLOT_MARGIN),
            justify_content: match labels.len() {
                1 => JustifyContent::End,
                _ => JustifyContent::SpaceBetween,
            },
            ..default()
        })
        .with_children(|axis| {
            for day in labels {
                axis.spawn(ui::text(
                    CalendarDate::on_day(day).to_string(),
                    10.0,
                    Tone::Muted,
                ));
            }
        });
}

fn inspection_text(display: &Display) -> String {
    display
                .inspection
                .as_ref()
                .map(|i| {
                    let detail = match i.actor.kind {
                        0 => format!(
                            "{} | members {} | fatigue {} | labor {} min\nEmployer {}{} | days left in term {}\nGross paid {} | tax {} | charges {} | payroll reserve {}\nHeating fuel: wood {} | coal {}\nHeat supplied {} / {} | deficit {}",
                            i.housing_state.map(|state| format!("{state:?}")).unwrap_or_default(),
                            i.living_members, i.fatigue, i.available_minutes, i.employer,
                            i.contract_employer.filter(|&employer| employer != i.employer)
                                .map(|employer| format!(" | farm contract {employer}")).unwrap_or_default(),
                            i.job_days_remaining,
                            i.gross_income, i.income_tax, i.employee_charges, i.payroll_reserve,
                            i.inventory[Good::Wood as usize - 1], i.inventory[Good::Coal as usize - 1],
                            i.heating_served, i.heating_need, i.heating_need.saturating_sub(i.heating_served)
                        ),
                        _ => format!(
                            "{} {:?} | {}\nCondition {} | safety {}\nRent overdue {} days\nCompany {} ({} sites) | owner {}\nOwner stubbornness {}/1000\nWage {} | price {}\nCapacity {} (+{}) | planned {} | open {}\nMethod {:?} | {} output / {} min",
                            i.enterprise_state.map(|state| format!("{state:?}")).unwrap_or_default(),
                            i.enterprise_model, i.business_scale.map(|scale| format!("{scale:?}")).unwrap_or_default(),
                            i.equipment_condition, i.safety_provision,
                            i.rent_overdue_days,
                            i.company_identity, i.company_sites.len(), i.owner_identity,
                            crate::domain::company::OwnerStubbornness::from_seed(display.seed, i.owner_identity).per_mille(),
                            i.wage,
                            i.price,
                            i.actor.capacity,
                            i.pending_capacity,
                            i.planned_packets,
                            i.open_slots, i.production_method, i.output_ticks, i.labor_minutes
                        ),
                    };
                    let property = match i.property_region {
                        0 if i.subsistence_plots + i.displaced_plots > 0 => format!("\nSubsistence plots {} | displaced {}", i.subsistence_plots, i.displaced_plots),
                        0 if i.actor.kind == Good::Grain as u32 && i.occupied_plots > 0 => format!("\nConsolidated farmland {} plots | scale bonus {}%", i.occupied_plots, i.scale_bonus_per_mille / 10),
                        0 => String::new(),
                        _ => format!("\nRegion {} | plots {} (+{} reserved)\nWorkers/plot {} | scale bonus {}%\nSubsistence {} | displaced {}", i.property_region, i.occupied_plots, i.reserved_plots, i.workers_per_plot, i.scale_bonus_per_mille / 10, i.subsistence_plots, i.displaced_plots),
                    };
                    let construction = match i.project_status {
                        crate::domain::property::ProjectStatus::None => String::new(),
                        status => format!("\nConstruction {status:?} | {} days and {} work remaining", i.construction_days_remaining, i.construction_minutes_remaining),
                    };
                    let accounts = i.business_account.map_or_else(String::new, |account| {
                        format!("\nDaily receipts {} | cash outlays {}\nCash profit {:+} (before owner payouts)", account.revenue, account.outlays(), account.profit())
                    });
                    let railway = i.railway_business.as_ref().map_or_else(String::new, |business| {
                        format!("\nRailway {:?} | {} segments built | {} operating\nRail units available {} / {} daily capacity\nDaily freight and access receipts {}", business.status(), business.built_segments, business.operating_segments, business.available_units, business.daily_capacity, business.daily_receipts)
                    });
                    let rental = i.rental_business.as_ref().map_or_else(String::new, |business| {
                        format!("\nLandlord business | homes {} | rental income {}", business.homes, business.income)
                    });
                    format!(
                        "Actor {} | day {} | {}\n{} | {} | {}°C\nCash {} | transit {}\nRent paid {} | owed {}\nFood {:?} | other stock {}\n{}{}{}{}{}{}",
                        i.actor.identity,
                        i.day,
                        Good::from_index(i.actor.kind).map_or_else(|_| "Household".into(), |good| good.to_string()),
                        i.weather.date,
                        i.weather.season,
                        i.weather.temperature_celsius,
                        i.cash,
                        i.transit,
                        i.rent_collected,
                        i.rent_arrears,
                        &i.inventory[..3],
                        i.inventory[3..].iter().map(|&stock| u128::from(stock)).sum::<u128>(),
                        detail,
                        property,
                        construction,
                        rental,
                        accounts,
                        railway
                    )
                })
                .unwrap_or_else(|| "Select a population account or firm to inspect its accounts, property and production.".into())
}

fn exact_value(value: f64, format: ValueFormat) -> String {
    match (format, value) {
        (_, value) if !value.is_finite() => "unavailable".into(),
        (ValueFormat::Money, _) => format!("{value:.0}"),
        (_, value) if value.fract().abs() < 0.000_001 => format!("{value:.0}"),
        _ => format!("{value:.3}"),
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct ReadoutVersion {
    revision: u64,
    chart: Chart,
    selection: SeriesSelection,
    day: u64,
}

#[derive(SystemParam)]
struct CursorTooltip<'w, 's> {
    node: Single<
        'w,
        's,
        (
            Entity,
            &'static ComputedNode,
            &'static mut Node,
            &'static mut UiTransform,
            &'static mut Visibility,
        ),
        (With<ChartTooltip>, Without<Crosshair>),
    >,
    window: Single<'w, 's, &'static Window, With<PrimaryWindow>>,
    previous: Local<'s, Option<LineObservation>>,
    commands: Commands<'w, 's>,
}

impl CursorTooltip<'_, '_> {
    fn update(&mut self, observation: Option<LineObservation>) {
        let (entity, computed, node, transform, visibility) = &mut *self.node;
        match (observation, self.window.cursor_position()) {
            (Some(observation), Some(cursor)) => {
                if Some(observation) != *self.previous {
                    self.commands
                        .entity(*entity)
                        .despawn_children()
                        .with_children(|tooltip| tooltip_content(tooltip, observation));
                    *self.previous = Some(observation);
                }
                let size = computed.size() * computed.inverse_scale_factor();
                let inset = 8.0;
                let gap = 10.0;
                let minimum_x = inset + size.x * 0.5;
                let maximum_x = (self.window.width() - minimum_x).max(minimum_x);
                let left = px(cursor.x.clamp(minimum_x, maximum_x));
                let above = cursor.y - gap;
                let (top, anchor) = match above - size.y >= inset {
                    true => (above, -100.0),
                    false => (
                        (cursor.y + gap).min(self.window.height() - size.y - inset),
                        0.0,
                    ),
                };
                let top = px(top.max(inset));
                if (node.left, node.top) != (left, top) {
                    node.left = left;
                    node.top = top;
                }
                transform.set_if_neq(UiTransform::from_xy(percent(-50), percent(anchor)));
                visibility.set_if_neq(Visibility::Inherited);
            }
            _ => {
                visibility.set_if_neq(Visibility::Hidden);
            }
        }
    }
}

fn draw_cursor(
    dashboard: Option<Res<Dashboard>>,
    state: Res<PlotState>,
    images: Query<&ChildOf, With<ImageNode>>,
    mut crosshairs: Query<(&ChildOf, &mut Node, &mut Visibility), With<Crosshair>>,
    mut readouts: Query<&mut Text, With<ui::ChartReadout>>,
    mut previous: Local<Option<ReadoutVersion>>,
    mut tooltip: CursorTooltip,
) {
    if let Some(dashboard) = dashboard {
        let display = dashboard.data.lock().expect("dashboard lock");
        let last = display.snapshot.day;
        let first = display
            .history
            .front()
            .map_or(last, |point| point.day)
            .max(dashboard.period.first_day(last));
        let span = last - first;
        let hovered = state.hover.as_ref();
        let observation = hovered.and_then(PlotHover::observation);
        let day = match observation {
            Some(observation) => observation.day.clamp(first, last),
            None => hovered.map_or(last, |hover| {
                first + (f64::from(hover.fraction) * span as f64).round() as u64
            }),
        };
        for (parent, mut node, mut visibility) in &mut crosshairs {
            let layout = images.get(parent.parent()).ok().and_then(|plot| {
                state
                    .version
                    .as_ref()?
                    .plots
                    .iter()
                    .find(|layout| layout.entity == plot.parent())
            });
            match (hovered, layout) {
                (Some(hover), Some(layout)) => {
                    let left = px(layout.x_at(hover.fraction) - CROSSHAIR_WIDTH * 0.5);
                    if node.left != left {
                        node.left = left;
                    }
                    visibility.set_if_neq(Visibility::Inherited);
                }
                _ => {
                    visibility.set_if_neq(Visibility::Hidden);
                }
            }
        }
        let version = (dashboard.page == DashboardPage::Chart).then_some(ReadoutVersion {
            revision: display.observation_revision,
            chart: dashboard.chart,
            selection: dashboard.series,
            day,
        });
        if let Some(version) = version.filter(|version| Some(*version) != *previous) {
            let detail = readout_text(&display, version);
            for mut label in &mut readouts {
                label.0.clone_from(&detail);
            }
        }
        *previous = version;
        tooltip.update(observation);
    }
}

fn cursor_in_plot(cursor: &RelativeCursorPosition, node: &ComputedNode) -> Option<Vec2> {
    cursor
        .normalized
        .filter(|_| cursor.cursor_over())
        .and_then(|position| {
            let size = node.size() * node.inverse_scale_factor();
            let position = (position + 0.5) * size - Vec2::new(AXIS_WIDTH, TOP_PADDING);
            let area = size - Vec2::new(AXIS_WIDTH, TOP_PADDING + DATE_HEIGHT);
            Rect::from_corners(Vec2::ZERO, area)
                .contains(position)
                .then_some(position)
        })
}

fn tooltip_content(tooltip: &mut ChildSpawnerCommands, observation: LineObservation) {
    let line = observation.line;
    let chart = line.chart;
    let name = chart
        .series()
        .into_iter()
        .nth(line.series)
        .unwrap_or_else(|| chart.label().into());
    let run = match line.run {
        LineRun::Current => "",
        LineRun::Reference => " · Reference",
    };
    tooltip.spawn(ui::text(
        format!("Day {} · {}{run}", observation.day, chart.unit()),
        11.0,
        Tone::Muted,
    ));
    tooltip.spawn(ui::row(6.0)).with_children(|row| {
        ui::swatch(row, Tone::Series(chart, line.series));
        row.spawn(ui::text(name, 12.0, Tone::Text));
        row.spawn(ui::text(
            exact_value(observation.value, ValueFormat::Quantity),
            12.0,
            Tone::Text,
        ));
    });
}

fn readout_text(display: &Display, version: ReadoutVersion) -> String {
    match display.history.is_empty() {
        true => "Waiting for observations…".into(),
        false => {
            let chart = version.chart;
            let index = display
                .history
                .partition_point(|point| point.day < version.day)
                .min(display.history.len() - 1);
            let snapshot = &display.history[index];
            let previous = index
                .checked_sub(1)
                .and_then(|index| display.history.get(index));
            let reference = display.comparison.as_ref().and_then(|run| {
                chart
                    .samples(&run.history, snapshot.day, snapshot.day)
                    .next()
                    .map(|sample| sample.values)
            });
            let values = chart.values(snapshot, previous);
            let rows = chart
                .series()
                .into_iter()
                .zip(values)
                .enumerate()
                .filter(|(series, _)| version.selection.includes(*series))
                .map(|(series, (name, value))| {
                    let comparison = reference
                        .as_ref()
                        .and_then(|values| values.get(series))
                        .filter(|prior| prior.is_finite() && value.is_finite())
                        .map_or_else(String::new, |prior| {
                            let change = exact_value(value - prior, chart.value_format());
                            let sign = match value > *prior {
                                true => "+",
                                false => "",
                            };
                            format!(
                                " · reference {} · Δ {sign}{change}",
                                exact_value(*prior, chart.value_format())
                            )
                        });
                    let people = match chart {
                        Chart::Occupations => {
                            format!(" ({} people)", snapshot.population.occupations[series])
                        }
                        _ => String::new(),
                    };
                    format!(
                        "{name}  {}{people}{comparison}",
                        exact_value(value, chart.value_format())
                    )
                })
                .collect::<Vec<_>>()
                .join("\n");
            format!(
                "{} · Day {} · {}\n\n{rows}",
                snapshot.weather.date,
                snapshot.day,
                chart.unit()
            )
        }
    }
}

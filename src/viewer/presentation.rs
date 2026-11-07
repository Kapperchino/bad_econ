use super::{
    Chart, Dashboard, Display,
    model::{ChartRange, Menu},
    ui::{
        self, ActorSelection, ChartTab, FirmMarker, Label, Legend, MapMarkers, Plot, Region,
        UiAction,
    },
};
use crate::{domain::ObservationSnapshot, runtime::RuntimeState};
use bevy::{
    picking::{Pickable, hover::Hovered},
    prelude::*,
    ui_widgets::Button,
};
use std::collections::{BTreeMap, VecDeque};

pub(super) struct PresentationPlugin;
impl Plugin for PresentationPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            (draw, draw_markers)
                .after(super::controls)
                .run_if(needs_refresh),
        );
    }
}

#[derive(Clone, Copy, PartialEq)]
struct PresentationVersion {
    revision: u64,
    chart: Chart,
    plot_size: Vec2,
}

fn needs_refresh(
    dashboard: Option<Res<Dashboard>>,
    plots: Query<&ComputedNode, With<Plot>>,
    mut previous: Local<Option<PresentationVersion>>,
) -> bool {
    let current = dashboard.map(|dashboard| PresentationVersion {
        revision: dashboard.data.lock().expect("dashboard lock").revision,
        chart: dashboard.chart,
        plot_size: plots
            .single()
            .map(|node| node.size() * node.inverse_scale_factor())
            .unwrap_or_default(),
    });
    let changed = current != *previous;
    *previous = current;
    changed
}

fn draw(
    dashboard: Option<Res<Dashboard>>,
    mut labels: Query<(&Label, &mut Text)>,
    mut regions: Query<(&Region, &mut BackgroundColor)>,
    mut tabs: Query<(&ChartTab, &mut Node)>,
    plots: Query<(Entity, &ComputedNode), With<Plot>>,
    legends: Query<Entity, With<Legend>>,
    mut commands: Commands,
) {
    if let Some(dashboard) = dashboard {
        let display = dashboard.data.lock().expect("dashboard lock");
        let s = &display.snapshot;
        let chart = dashboard.chart;
        let range = ChartRange::new(chart, s, &display.history, display.comparison.as_ref());
        labels.iter_mut().for_each(|(kind, mut label)| {
            let value = match kind {
                Label::Resources => format!("Population {}     Paid households {}     Cash {}     Firms {}", number(s.population.living), number(s.employed), number(s.cash), number(s.firms)),
                Label::Date => format!("DAY {:04}   |   {}   |   {}", s.day, display.status.label(), display.speed.label()),
                Label::Title => chart.menu().label().into(),
                Label::Subtitle => chart.menu().description().into(),
                Label::Summary => summary(chart.menu(), s),
                Label::ChartTitle => chart.label().into(),
                Label::ChartScale => range.scale_label(chart),
                Label::Comparison => comparison_text(&display),
                Label::Inspection => inspection_text(&display),
                Label::MapSummary => format!("{} households / {} people\n{} paid / {} unmet bread ticks\n{} firms opened / {} closed\n{} distributors / {} discoveries", number(s.households), number(s.population.living), number(s.employed), number(s.shortages), number(s.firm_entries), number(s.firm_closures), number(s.distributors), number(s.research.discoveries)),
                Label::Error => display.error.clone().unwrap_or_else(|| match &display.status {
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
            node.display = match tab.0.menu() == chart.menu() {
                true => bevy::ui::Display::Flex,
                false => bevy::ui::Display::None,
            };
        });
        let peak = s.regions.iter().copied().max().unwrap_or(1).max(1) as f32;
        regions.iter_mut().for_each(|(region, mut color)| {
            color.0 = ui::region_color(s.regions.get(region.0).copied().unwrap_or(0) as f32 / peak);
        });
        if let Ok(entity) = legends.single() {
            commands
                .entity(entity)
                .despawn_children()
                .with_children(|legend| {
                    chart
                        .series()
                        .into_iter()
                        .enumerate()
                        .for_each(|(series, name)| {
                            legend
                                .spawn(Node {
                                    align_items: AlignItems::Center,
                                    column_gap: px(5),
                                    ..default()
                                })
                                .with_children(|item| {
                                    item.spawn((
                                        Node {
                                            width: px(8),
                                            height: px(8),
                                            ..default()
                                        },
                                        BackgroundColor(chart_color(chart, series)),
                                    ));
                                    item.spawn(ui::text(name, 11.0, ui::MUTED));
                                });
                        });
                });
        }
        if let Ok((entity, computed)) = plots.single() {
            let size = computed.size() * computed.inverse_scale_factor();
            commands
                .entity(entity)
                .despawn_children()
                .with_children(|plot| {
                    draw_chart(plot, &display, chart, &range, size);
                });
        }
    }
}

fn draw_markers(
    dashboard: Option<Res<Dashboard>>,
    containers: Query<Entity, With<MapMarkers>>,
    mut markers: Query<(Entity, &FirmMarker, &mut Node, &mut BackgroundColor)>,
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
            .for_each(|(entity, marker, mut node, mut color)| {
                match firms.remove(&marker.identity) {
                    Some(firm) => {
                        node.left = coordinate(firm.position.x);
                        node.top = coordinate(firm.position.y);
                        color.0 = chart_color(Chart::Prices, firm.kind.saturating_sub(1) as usize);
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
                    width: px(9),
                    height: px(9),
                    border: UiRect::all(px(1)),
                    border_radius: BorderRadius::MAX,
                    ..default()
                },
                UiTransform::from_translation(Val2::percent(-50, -50)),
                BackgroundColor(chart_color(
                    Chart::Prices,
                    firm.kind.saturating_sub(1) as usize,
                )),
                BorderColor::all(ui::INK),
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

fn summary(menu: Menu, s: &ObservationSnapshot) -> String {
    match menu {
        Menu::Overview => format!(
            "Paid households    {}\nUnmet bread    {} ticks\nTotal cash    {}\nIn transit    {} goods ticks",
            number(s.employed),
            number(s.shortages),
            number(s.cash),
            number(
                s.transit
                    .iter()
                    .map(|&value| u128::from(value))
                    .sum::<u128>()
            )
        ),
        Menu::Market => format!(
            "Available stocks    {} goods ticks\nGrain / flour / bread    {} / {} / {}\nHousehold food output    {:?}\nDistributors    {}",
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
            number(s.distributors)
        ),
        Menu::Industry => format!(
            "Active firms    {}\nEntries / closures    {} / {}\nResearch attempts / discoveries    {} / {}\nStriking    {}",
            number(s.firms),
            number(s.firm_entries),
            number(s.firm_closures),
            number(s.research.attempts),
            number(s.research.discoveries),
            number(s.research.striking)
        ),
        Menu::Population => format!(
            "Population / dependents    {} / {}\nBirths    {}\nDeaths: background / work / famine    {:?}\nHomeless households    {}",
            number(s.population.living),
            number(s.population.dependents),
            number(s.population.births),
            s.population.deaths,
            number(s.homeless_households)
        ),
        Menu::Finance => format!(
            "Bank & support cash    {}\nLoan principal / interest    {} / {}\nIncome tax / employee charges    {} / {}\nPayroll reserves    {}\nRent collected / arrears    {} / {}",
            number(s.financial_cash),
            number(s.loan_principal),
            number(s.loan_interest),
            number(s.income_tax),
            number(s.employee_charges),
            number(s.payroll_reserves),
            number(s.rent_collected),
            number(s.rent_arrears)
        ),
    }
}

fn comparison_text(display: &Display) -> String {
    match &display.comparison {
        Some(reference) => match reference.at_day(display.snapshot.day) {
            Some(prior) => format!(
                "Solid: seed {}  /  faint: seed {}\nSame-day change: paid households {:+}, unmet bread {:+}",
                display.seed,
                reference.seed,
                i128::from(display.snapshot.employed) - i128::from(prior.employed),
                i128::from(display.snapshot.shortages) - i128::from(prior.shortages)
            ),
            None => format!(
                "Comparing seed {} with seed {}\nAdvancing to day {}...",
                display.seed, reference.seed, reference.horizon
            ),
        },
        None => format!(
            "SEED {}\nCompare seed saves this history and runs the next seed to the same day. Faint lines show the saved run.",
            display.seed
        ),
    }
}

fn chart_color(chart: Chart, series: usize) -> Color {
    match (chart, series) {
        (Chart::Employment, 0) => Color::srgb(0.48, 0.78, 0.57),
        (Chart::Employment | Chart::Credit, 1) => Color::srgb(0.90, 0.42, 0.32),
        (_, 0) => ui::GOLD,
        (_, 1) => Color::srgb(0.48, 0.70, 0.85),
        (_, 2) => Color::srgb(0.48, 0.78, 0.57),
        _ => Color::hsl((series as f32 * 137.5) % 360.0, 0.55, 0.65),
    }
}

fn draw_chart(
    parent: &mut ChildSpawnerCommands,
    display: &Display,
    chart: Chart,
    range: &ChartRange,
    size: Vec2,
) {
    let area = Vec2::new((size.x - 52.0).max(0.0), (size.y - 30.0).max(0.0));
    (0..=4).for_each(|row| {
        let value = range.maximum * (1.0 - row as f64 / 4.0);
        let caption = match value {
            value if value >= 1000.0 => number(value as u128),
            value if value >= 10.0 => format!("{value:.0}"),
            _ => format!("{value:.1}"),
        };
        parent.spawn((
            Node {
                position_type: PositionType::Absolute,
                left: px(0),
                top: px(row as f32 / 4.0 * area.y),
                ..default()
            },
            ui::text(caption, 10.0, ui::MUTED),
        ));
    });
    let labels = match chart {
        Chart::Cash => vec!["0 cash".into(), "Cash brackets: 1, 2-3, 4-7, ...".into()],
        _ => vec![
            format!("Day {}", range.first_day),
            format!("Day {}", range.last_day),
        ],
    };
    parent
        .spawn(Node {
            position_type: PositionType::Absolute,
            left: px(48),
            bottom: px(0),
            width: px(area.x),
            justify_content: JustifyContent::SpaceBetween,
            ..default()
        })
        .with_children(|axis| {
            labels.into_iter().for_each(|label| {
                axis.spawn(ui::text(label, 10.0, ui::MUTED));
            });
        });
    parent
        .spawn(Node {
            position_type: PositionType::Absolute,
            left: px(48),
            top: px(6),
            width: px(area.x),
            height: px(area.y),
            overflow: Overflow::clip(),
            ..default()
        })
        .with_children(|plot| {
            (0..=4).for_each(|row| {
                let y = row as f32 / 4.0 * (area.y - 1.0).max(0.0);
                line(
                    plot,
                    Vec2::new(0.0, y),
                    Vec2::new(area.x, y),
                    ui::RULE.with_alpha(0.5),
                );
            });
            match chart {
                Chart::Cash => {
                    if let Some(reference) = display
                        .comparison
                        .as_ref()
                        .and_then(|reference| reference.at_day(display.snapshot.day))
                    {
                        draw_cash(plot, reference, range, area, LineStyle::Reference);
                    }
                    draw_cash(plot, &display.snapshot, range, area, LineStyle::Current);
                }
                _ => {
                    if let Some(reference) = &display.comparison {
                        draw_history(
                            plot,
                            &reference.history,
                            chart,
                            range,
                            area,
                            LineStyle::Reference,
                        );
                    }
                    draw_history(
                        plot,
                        &display.history,
                        chart,
                        range,
                        area,
                        LineStyle::Current,
                    );
                }
            }
            if display.history.is_empty() {
                plot.spawn(ui::text(
                    "Waiting for the first observation...",
                    12.0,
                    ui::MUTED,
                ));
            }
        });
}

#[derive(Clone, Copy)]
enum LineStyle {
    Current,
    Reference,
}
impl LineStyle {
    fn color(self, color: Color) -> Color {
        match self {
            Self::Current => color,
            Self::Reference => color.with_alpha(0.30),
        }
    }
}

fn line(parent: &mut ChildSpawnerCommands, from: Vec2, to: Vec2, color: Color) {
    let delta = to - from;
    let middle = (from + to) / 2.0;
    parent.spawn((
        Node {
            position_type: PositionType::Absolute,
            left: px(middle.x - delta.length() / 2.0),
            top: px(middle.y - 1.0),
            width: px(delta.length().max(1.0)),
            height: px(2),
            ..default()
        },
        UiTransform::from_rotation(Rot2::radians(delta.y.atan2(delta.x))),
        BackgroundColor(color),
        Pickable::IGNORE,
    ));
}

fn draw_history(
    parent: &mut ChildSpawnerCommands,
    history: &VecDeque<ObservationSnapshot>,
    chart: Chart,
    range: &ChartRange,
    size: Vec2,
    style: LineStyle,
) {
    let visible = history
        .iter()
        .filter(|snapshot| (range.first_day..=range.last_day).contains(&snapshot.day))
        .collect::<Vec<_>>();
    let span = range.last_day.saturating_sub(range.first_day).max(1) as f32;
    let point = |day: u64, value: f64| {
        Vec2::new(
            3.0 + day.saturating_sub(range.first_day) as f32 / span * (size.x - 6.0).max(0.0),
            3.0 + (1.0 - value as f32 / range.maximum as f32) * (size.y - 6.0).max(0.0),
        )
    };
    visible.windows(2).for_each(|pair| {
        chart
            .values(pair[0])
            .into_iter()
            .zip(chart.values(pair[1]))
            .enumerate()
            .for_each(|(series, (from, to))| {
                line(
                    parent,
                    point(pair[0].day, from),
                    point(pair[1].day, to),
                    style.color(chart_color(chart, series)),
                );
            });
    });
    if let Some(last) = visible.last() {
        chart
            .values(last)
            .into_iter()
            .enumerate()
            .for_each(|(series, value)| {
                let point = point(last.day, value);
                parent.spawn((
                    Node {
                        position_type: PositionType::Absolute,
                        left: px(point.x - 3.0),
                        top: px(point.y - 3.0),
                        width: px(6),
                        height: px(6),
                        border_radius: BorderRadius::MAX,
                        ..default()
                    },
                    BackgroundColor(style.color(chart_color(chart, series))),
                    Pickable::IGNORE,
                ));
            });
    }
}

fn draw_cash(
    parent: &mut ChildSpawnerCommands,
    snapshot: &ObservationSnapshot,
    range: &ChartRange,
    size: Vec2,
    style: LineStyle,
) {
    let width = size.x / (range.last_bin + 1) as f32;
    snapshot
        .wealth_bins
        .iter()
        .take(range.last_bin + 1)
        .enumerate()
        .for_each(|(bin, count)| {
            let height = *count as f32 / range.maximum as f32 * (size.y - 3.0).max(0.0);
            let fraction = match style {
                LineStyle::Current => 0.55,
                LineStyle::Reference => 0.90,
            };
            parent.spawn((
                Node {
                    position_type: PositionType::Absolute,
                    left: px((bin as f32 + (1.0 - fraction) / 2.0) * width),
                    bottom: px(0),
                    width: px(width * fraction),
                    height: px(height),
                    ..default()
                },
                BackgroundColor(style.color(ui::GOLD)),
                Pickable::IGNORE,
            ));
        });
}
fn inspection_text(display: &Display) -> String {
    display
                .inspection
                .as_ref()
                .map(|i| {
                    let detail = match i.actor.kind {
                        0 => format!(
                            "{} | members {} | fatigue {} | labor {} min\nEmployer {} | gross paid {}\nTax {} | charges {} | payroll reserve {}",
                            i.housing_state.map(|state| format!("{state:?}")).unwrap_or_default(),
                            i.living_members, i.fatigue, i.available_minutes, i.employer,
                            i.gross_income, i.income_tax, i.employee_charges, i.payroll_reserve
                        ),
                        _ => format!(
                            "{} {:?} | {}\nBuilding {} {}\nCondition {} | safety {}\nRent overdue {} days\nCompany {} ({} sites) | owner {}\nWage {} | price {}\nCapacity {} (+{}) | planned {} | open {}\nMethod {:?} | {} output / {} min",
                            i.enterprise_state.map(|state| format!("{state:?}")).unwrap_or_default(),
                            i.enterprise_model, i.business_scale.map(|scale| format!("{scale:?}")).unwrap_or_default(), i.construction_minutes_remaining,
                            match i.property_region { 0 => "min", _ => "work" },
                            i.equipment_condition, i.safety_provision,
                            i.rent_overdue_days,
                            i.company_identity, i.company_sites.len(), i.owner_identity,
                            i.wage,
                            i.price,
                            i.actor.capacity,
                            i.pending_capacity,
                            i.planned_packets,
                            i.open_slots, i.production_method, i.output_ticks, i.labor_minutes
                        ),
                    };
                    let property = match i.property_region {
                        0 => String::new(),
                        _ => format!("\nRegion {} | plots {} (+{} reserved)\nWorkers/plot {} | scale bonus {}%\nSubsistence {} | displaced {}\nConstruction {:?} | {} work remaining", i.property_region, i.occupied_plots, i.reserved_plots, i.workers_per_plot, i.scale_bonus_per_mille / 10, i.subsistence_plots, i.displaced_plots, i.project_status, i.construction_minutes_remaining),
                    };
                    let rental = i.rental_business.as_ref().map_or_else(String::new, |business| {
                        format!("\nLandlord business | homes {} | rental income {}\nHousing project {:?} | {} work remaining", business.homes, business.income, i.project_status, i.construction_minutes_remaining)
                    });
                    format!(
                        "Actor {} | day {} | kind {}\nCash {} | transit {}\nRent paid {} | owed {}\nFood {:?} | other stock {}\n{}{}{}",
                        i.actor.identity,
                        i.day,
                        i.actor.kind,
                        i.cash,
                        i.transit,
                        i.rent_collected,
                        i.rent_arrears,
                        &i.inventory[..3],
                        i.inventory[3..].iter().map(|&stock| u128::from(stock)).sum::<u128>(),
                        detail,
                        property,
                        rental
                    )
                })
                .unwrap_or_else(|| "Select a household or firm to inspect its accounts, property and production.".into())
}

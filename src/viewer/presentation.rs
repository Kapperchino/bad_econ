use super::{
    Chart, Dashboard, Display,
    graph::{self, GraphRenderer},
    model::{ChartRange, Menu, Period, SeriesSelection},
    ui::{
        self, ActorSelection, ChartTab, FirmMarker, Label, Legend, MapMarkers, Plot, Region,
        UiAction,
    },
};
use crate::{
    domain::{Good, ObservationSnapshot, calendar::CalendarDate},
    runtime::RuntimeState,
};
use bevy::{
    picking::{Pickable, hover::Hovered},
    prelude::*,
    ui_widgets::Button,
};
use std::collections::BTreeMap;

pub(super) struct PresentationPlugin;
impl Plugin for PresentationPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, graph::setup).add_systems(
            Update,
            (
                draw.run_if(needs_refresh),
                draw_readout,
                draw_plot.run_if(plot_changed),
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

#[derive(Clone, Copy, PartialEq)]
struct PresentationVersion {
    revision: u64,
    chart: Chart,
    period: Period,
    selection: SeriesSelection,
    plot_size: Vec2,
    scale: f32,
}

#[derive(Clone, Copy, PartialEq)]
struct LabelVersion {
    revision: u64,
    chart: Chart,
    selection: SeriesSelection,
}

fn needs_refresh(
    dashboard: Option<Res<Dashboard>>,
    mut previous: Local<Option<LabelVersion>>,
) -> bool {
    let current = dashboard.map(|dashboard| LabelVersion {
        revision: dashboard.data.lock().expect("dashboard lock").revision,
        chart: dashboard.chart,
        selection: dashboard.series,
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
) {
    if let Some(dashboard) = dashboard {
        let display = dashboard.data.lock().expect("dashboard lock");
        let s = &display.snapshot;
        let chart = dashboard.chart;
        labels.iter_mut().for_each(|(kind, mut label)| {
            let value = match kind {
                Label::Resources => format!("Population {}     Paid households {}     Cash {}     Firms {}", number(s.population.living), number(s.employed), number(s.cash), number(s.firms)),
                Label::Date => format!("{} | {} | {}°C\nDAY {:04} | {} | {}", s.weather.date, s.weather.season, s.weather.temperature_celsius, s.day, display.status.label(), display.speed.label()),
                Label::Title => chart.menu().label().into(),
                Label::Subtitle => chart.menu().description().into(),
                Label::Summary => summary(chart.menu(), s),
                Label::ChartTitle => chart.label().into(),
                Label::ChartDescription => chart.description().into(),
                Label::SeriesFocus => match dashboard.series {
                    SeriesSelection::All => "Showing all series".into(),
                    SeriesSelection::Only(index) => format!("Focused: {}", chart.series().get(index).cloned().unwrap_or_default()),
                },
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
            let visibility = match tab.0.menu() == chart.menu() {
                true => bevy::ui::Display::Flex,
                false => bevy::ui::Display::None,
            };
            if node.display != visibility {
                node.display = visibility;
            };
        });
        let peak = s.regions.iter().copied().max().unwrap_or(1).max(1) as f32;
        regions.iter_mut().for_each(|(region, mut color)| {
            color.set_if_neq(BackgroundColor(ui::region_color(
                s.regions.get(region.0).copied().unwrap_or(0) as f32 / peak,
            )));
        });
    }
}

fn plot_changed(
    dashboard: Option<Res<Dashboard>>,
    plots: Query<&ComputedNode, With<Plot>>,
    mut previous: Local<Option<PresentationVersion>>,
) -> bool {
    let current = dashboard.and_then(|dashboard| {
        plots.single().ok().map(|node| PresentationVersion {
            revision: dashboard
                .data
                .lock()
                .expect("dashboard lock")
                .observation_revision,
            chart: dashboard.chart,
            period: dashboard.period,
            selection: dashboard.series,
            plot_size: node.size() * node.inverse_scale_factor(),
            scale: node.inverse_scale_factor(),
        })
    });
    let changed = current != *previous;
    *previous = current;
    changed
}

fn chart_changed(
    dashboard: Option<Res<Dashboard>>,
    mut previous: Local<Option<(Chart, SeriesSelection)>>,
) -> bool {
    let current = dashboard.map(|dashboard| (dashboard.chart, dashboard.series));
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
                dashboard
                    .chart
                    .series()
                    .into_iter()
                    .enumerate()
                    .filter(|(series, _)| dashboard.series.includes(*series))
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
                                    BackgroundColor(chart_color(dashboard.chart, series)),
                                ));
                                item.spawn(ui::text(name, 11.0, ui::MUTED));
                            });
                    });
            });
    }
}

fn draw_plot(
    dashboard: Res<Dashboard>,
    plots: Query<(Entity, &ComputedNode), With<Plot>>,
    mut renderer: GraphRenderer,
    mut scales: Query<&mut Text, With<ui::ChartScale>>,
    mut commands: Commands,
) {
    if let Ok((entity, computed)) = plots.single() {
        let display = dashboard.data.lock().expect("dashboard lock");
        let chart = dashboard.chart;
        let range = ChartRange::new(
            chart,
            &display.snapshot,
            &display.history,
            display.comparison.as_ref(),
            dashboard.period,
            dashboard.series,
        );
        scales.iter_mut().for_each(|mut label| {
            let value = range.scale_label(chart);
            if label.0 != value {
                label.0 = value;
            }
        });
        let size = computed.size() * computed.inverse_scale_factor();
        let area = Vec2::new((size.x - 52.0).max(1.0), (size.y - 30.0).max(1.0));
        renderer.update(
            &display,
            chart,
            &range,
            area,
            computed.inverse_scale_factor(),
        );
        commands
            .entity(entity)
            .despawn_children()
            .with_children(|plot| {
                draw_axes(plot, &range, area);
                plot.spawn((
                    Node {
                        position_type: PositionType::Absolute,
                        left: px(48),
                        top: px(6),
                        width: px(area.x),
                        height: px(area.y),
                        ..default()
                    },
                    ImageNode::new(renderer.image()),
                    Pickable::IGNORE,
                ));
                if display.history.is_empty() {
                    plot.spawn(ui::text(
                        "Waiting for the first observation...",
                        12.0,
                        ui::MUTED,
                    ));
                }
            });
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
                        let left = coordinate(firm.position.x);
                        let top = coordinate(firm.position.y);
                        if (node.left, node.top) != (left, top) {
                            node.left = left;
                            node.top = top;
                        }
                        color.set_if_neq(BackgroundColor(chart_color(
                            Chart::Prices,
                            firm.kind.saturating_sub(1) as usize,
                        )));
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
            "Open business sites    {}\nEntries / closures to date    {} / {}\nResearch attempts / discoveries    {} / {}\nStriking    {}",
            number(s.firms),
            number(s.firm_entries),
            number(s.firm_closures),
            number(s.research.attempts),
            number(s.research.discoveries),
            number(s.research.striking)
        ),
        Menu::Population => format!(
            "Population / dependents    {} / {}\nBirths    {}\nDeaths: background / work / famine / cold    {:?}\nCold exposure    {:.1} person-days\nHomeless households    {}",
            number(s.population.living),
            number(s.population.dependents),
            number(s.population.births),
            s.population.deaths,
            s.population.cold_exposure as f64 / 1000.0,
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
                "Seed {} / reference seed {}\nNo saved observation for day {} (reference ends at day {}).",
                display.seed, reference.seed, display.snapshot.day, reference.horizon
            ),
        },
        None => format!(
            "SEED {}\nCompare seed saves this history and runs the next seed to the same day. Faint lines show the saved run.",
            display.seed
        ),
    }
}

pub(super) fn chart_color(chart: Chart, series: usize) -> Color {
    match (chart, series) {
        (Chart::Deaths, 0) | (Chart::Demography, 1) => ui::MUTED,
        (Chart::Deaths, 1) | (Chart::Demography, 2) => Color::srgb(0.88, 0.66, 0.33),
        (Chart::Deaths, 2) | (Chart::Demography, 3) => Color::srgb(0.90, 0.42, 0.32),
        (Chart::Deaths, 3) | (Chart::Demography, 4) => Color::srgb(0.48, 0.70, 0.85),
        (Chart::Employment, 0) => Color::srgb(0.48, 0.78, 0.57),
        (Chart::Employment | Chart::Credit, 1) => Color::srgb(0.90, 0.42, 0.32),
        (_, 0) => ui::GOLD,
        (_, 1) => Color::srgb(0.48, 0.70, 0.85),
        (_, 2) => Color::srgb(0.48, 0.78, 0.57),
        _ => Color::hsl((series as f32 * 137.5) % 360.0, 0.55, 0.65),
    }
}

fn draw_axes(parent: &mut ChildSpawnerCommands, range: &ChartRange, area: Vec2) {
    (0..=4).for_each(|row| {
        let value = range.maximum - (range.maximum - range.minimum) * row as f64 / 4.0;
        let caption = match value {
            value if value.abs() >= 1000.0 => format!(
                "{}{:.1}k",
                if value < 0.0 { "-" } else { "" },
                value.abs() / 1000.0
            ),
            value if value.abs() >= 10.0 => format!("{value:.0}"),
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
    let labels = [
        CalendarDate::on_day(range.first_day).to_string(),
        CalendarDate::on_day(range.last_day).to_string(),
    ];
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
}

fn inspection_text(display: &Display) -> String {
    display
                .inspection
                .as_ref()
                .map(|i| {
                    let detail = match i.actor.kind {
                        0 => format!(
                            "{} | members {} | fatigue {} | labor {} min\nEmployer {} | gross paid {}\nTax {} | charges {} | payroll reserve {}\nHeating fuel: wood {} | coal {}\nHeat supplied {} / {} | deficit {}",
                            i.housing_state.map(|state| format!("{state:?}")).unwrap_or_default(),
                            i.living_members, i.fatigue, i.available_minutes, i.employer,
                            i.gross_income, i.income_tax, i.employee_charges, i.payroll_reserve,
                            i.inventory[Good::Wood as usize - 1], i.inventory[Good::Coal as usize - 1],
                            i.heating_served, i.heating_need, i.heating_need.saturating_sub(i.heating_served)
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
                        0 if i.subsistence_plots + i.displaced_plots > 0 => format!("\nSubsistence plots {} | displaced {}", i.subsistence_plots, i.displaced_plots),
                        0 if i.actor.kind == Good::Grain as u32 && i.occupied_plots > 0 => format!("\nConsolidated farmland {} plots | scale bonus {}%", i.occupied_plots, i.scale_bonus_per_mille / 10),
                        0 => String::new(),
                        _ => format!("\nRegion {} | plots {} (+{} reserved)\nWorkers/plot {} | scale bonus {}%\nSubsistence {} | displaced {}\nConstruction {:?} | {} work remaining", i.property_region, i.occupied_plots, i.reserved_plots, i.workers_per_plot, i.scale_bonus_per_mille / 10, i.subsistence_plots, i.displaced_plots, i.project_status, i.construction_minutes_remaining),
                    };
                    let accounts = i.business_account.map_or_else(String::new, |account| {
                        format!("\nDaily receipts {} | cash outlays {}\nCash profit {:+} (before owner payouts)", account.revenue, account.outlays(), account.profit())
                    });
                    let railway = i.railway_business.as_ref().map_or_else(String::new, |business| {
                        format!("\nRailway {:?} | {} segments built | {} operating\nRail units available {} / {} daily capacity\nDaily freight and access receipts {}", business.status(), business.built_segments, business.operating_segments, business.available_units, business.daily_capacity, business.daily_receipts)
                    });
                    let rental = i.rental_business.as_ref().map_or_else(String::new, |business| {
                        format!("\nLandlord business | homes {} | rental income {}\nHousing project {:?} | {} work remaining", business.homes, business.income, i.project_status, i.construction_minutes_remaining)
                    });
                    format!(
                        "Actor {} | day {} | kind {}\n{} | {} | {}°C\nCash {} | transit {}\nRent paid {} | owed {}\nFood {:?} | other stock {}\n{}{}{}{}{}",
                        i.actor.identity,
                        i.day,
                        i.actor.kind,
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
                        rental,
                        accounts,
                        railway
                    )
                })
                .unwrap_or_else(|| "Select a household or firm to inspect its accounts, property and production.".into())
}

fn exact_value(value: f64) -> String {
    match value {
        value if !value.is_finite() => "unavailable".into(),
        value if value.fract().abs() < 0.000_001 => format!("{value:.0}"),
        _ => format!("{value:.3}"),
    }
}

fn draw_readout(
    dashboard: Option<Res<Dashboard>>,
    plots: Query<(&bevy::ui::RelativeCursorPosition, &ComputedNode), With<Plot>>,
    mut labels: Query<&mut Text, With<ui::ChartReadout>>,
) {
    if let Some(dashboard) = dashboard {
        let display = dashboard.data.lock().expect("dashboard lock");
        let chart = dashboard.chart;
        let first_day = display
            .history
            .front()
            .map_or(display.snapshot.day, |point| point.day)
            .max(dashboard.period.first_day(display.snapshot.day));
        let last_day = display.snapshot.day;
        let hovered = plots.single().ok().and_then(|(cursor, node)| {
            cursor
                .normalized
                .filter(|_| cursor.cursor_over())
                .map(|position| {
                    let width = node.size().x * node.inverse_scale_factor();
                    (((position.x + 0.5) * width - 48.0) / (width - 52.0).max(1.0)).clamp(0.0, 1.0)
                })
        });
        let day = hovered.map_or(last_day, |fraction| {
            first_day + (f64::from(fraction) * (last_day - first_day) as f64).round() as u64
        });
        let index = display
            .history
            .iter()
            .position(|point| point.day >= day)
            .unwrap_or(display.history.len().saturating_sub(1));
        let snapshot = display.history.get(index).unwrap_or(&display.snapshot);
        let previous = index
            .checked_sub(1)
            .and_then(|index| display.history.get(index));
        let reference = display.comparison.as_ref().and_then(|run| {
            run.history
                .iter()
                .position(|point| point.day == snapshot.day)
                .map(|index| {
                    chart.values(
                        &run.history[index],
                        index
                            .checked_sub(1)
                            .and_then(|index| run.history.get(index)),
                    )
                })
        });
        let detail = {
            let values = chart.values(snapshot, previous);
            let rows = chart
                .series()
                .into_iter()
                .zip(values)
                .enumerate()
                .filter(|(series, _)| dashboard.series.includes(*series))
                .map(|(series, (name, value))| {
                    let comparison = reference
                        .as_ref()
                        .and_then(|values| values.get(series))
                        .filter(|prior| prior.is_finite() && value.is_finite())
                        .map_or_else(String::new, |prior| {
                            format!(
                                " | reference {} | change {:+.3}",
                                exact_value(*prior),
                                value - prior
                            )
                        });
                    let people = match chart {
                        Chart::Occupations => {
                            format!(" ({} people)", snapshot.population.occupations[series])
                        }
                        _ => String::new(),
                    };
                    format!("{name}  {}{people}{comparison}", exact_value(value))
                })
                .collect::<Vec<_>>()
                .join("\n");
            format!(
                "{} | DAY {} | {}\n{}",
                snapshot.weather.date,
                snapshot.day,
                chart.unit(),
                rows
            )
        };
        labels.iter_mut().for_each(|mut text| {
            if text.0 != detail {
                text.0 = detail.clone();
            }
        });
    }
}

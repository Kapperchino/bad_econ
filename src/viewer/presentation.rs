use super::{Chart, ChartRange, Dashboard, Display, PanelLabel, Region};
use crate::domain::{Good, ObservationSnapshot};
use bevy::{
    gizmos::{GizmoAsset, retained::Gizmo},
    prelude::*,
};
use std::collections::VecDeque;

pub(super) struct PresentationPlugin;
impl Plugin for PresentationPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup)
            .add_systems(Update, draw.after(super::controls).run_if(needs_refresh));
    }
}

#[derive(Resource)]
struct Presentation {
    geometry: Handle<GizmoAsset>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct PresentationVersion {
    revision: u64,
    chart: Chart,
}

fn setup(mut commands: Commands, mut assets: ResMut<Assets<GizmoAsset>>) {
    let geometry = assets.add(GizmoAsset::default());
    commands.spawn(Gizmo {
        handle: geometry.clone(),
        ..Default::default()
    });
    commands.insert_resource(Presentation { geometry });
}

fn needs_refresh(
    dashboard: Option<Res<Dashboard>>,
    mut previous: Local<Option<PresentationVersion>>,
) -> bool {
    let current = dashboard.map(|dashboard| {
        let display = dashboard.data.lock().expect("dashboard lock");
        PresentationVersion {
            revision: display.revision,
            chart: dashboard.chart,
        }
    });
    let changed = current != *previous;
    *previous = current;
    changed
}

fn draw(
    dashboard: Option<Res<Dashboard>>,
    mut labels: Query<(&PanelLabel, &mut Text2d)>,
    mut regions: Query<(&Region, &mut Sprite)>,
    geometry: Res<Presentation>,
    mut assets: ResMut<Assets<GizmoAsset>>,
) {
    match dashboard {
        None => (),
        Some(dashboard) => {
            let display = dashboard.data.lock().expect("dashboard lock");
            let mut gizmos = GizmoAsset::default();
            let s = &display.snapshot;
            let unemployment = 100.0 * (s.households.saturating_sub(s.employed)) as f64
                / s.households.max(1) as f64;
            let inspection = display
                .inspection
                .as_ref()
                .map(|i| {
                    let detail = match i.actor.kind {
                        0 => format!(
                            "Health {} | fatigue {} | labor {} min\nEmployer {} | gross paid {}\nTax {} | charges {} | payroll reserve {}",
                            i.health, i.fatigue, i.available_minutes, i.employer,
                            i.gross_income, i.income_tax, i.employee_charges, i.payroll_reserve
                        ),
                        _ => format!(
                            "Owner {} | wage {} | price {}\nCapacity {} | planned {} | open {}\nMethod {:?} | {} output / {} min",
                            i.owner_identity,
                            i.wage,
                            i.price,
                            i.actor.capacity,
                            i.planned_packets,
                            i.open_slots, i.production_method, i.output_ticks, i.labor_minutes
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
                .zip(Good::ALL)
                .map(|(market, name)| {
                    format!(
                        "{name:?}: {:.0} / {:.0}",
                        market.prices.mean(),
                        market.wages.mean(),
                    )
                })
                .collect::<Vec<_>>()
                .chunks(2)
                .map(|pair| pair.join(" | "))
                .collect::<Vec<_>>()
                .join("\n");
            let text = format!(
                "{} | seed {}\nDay {} | target {}\nUnemployment {:.1}% | unmet bread {}\nCash {}\nFood stocks {:?} | other stocks {}\nIn transit {}\nTax {} | charges {} | payroll reserve {}\nBank/support cash {} | loan principal {}\nLoans {} active / {} overdue | interest {}\nResearch {} attempts / {} successes | strikes {}\n\nPrice / wage | means\n{}\n\n{}\n\n{}\n{}",
                display.status.label(),
                display.seed,
                s.day,
                display.speed.label(),
                unemployment,
                s.shortages,
                s.cash,
                &s.inventories[..3],
                s.inventories[3..]
                    .iter()
                    .map(|&value| u128::from(value))
                    .sum::<u128>(),
                s.transit
                    .iter()
                    .map(|&value| u128::from(value))
                    .sum::<u128>(),
                s.income_tax,
                s.employee_charges,
                s.payroll_reserves,
                s.financial_cash,
                s.loan_principal,
                s.active_loans,
                s.overdue_loans,
                s.loan_interest,
                s.research.attempts,
                s.research.discoveries,
                s.research.striking,
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
                None => "Colors follow the goods catalogue".into(),
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
                let color = chart_color(Chart::Prices, firm.kind.saturating_sub(1) as usize);
                let point = Vec2::new(
                    -599.5 + firm.position.x as f32 / display.map_size as f32 * 624.0,
                    324.5 - firm.position.y as f32 / display.map_size as f32 * 624.0,
                );
                gizmos.circle_2d(point, 2.0, color);
            });
            draw_chart(&display, dashboard.chart, &range, &mut gizmos);
            *assets.get_mut(&geometry.geometry).expect("viewer geometry") = gizmos;
        }
    }
}

fn chart_color(chart: Chart, series: usize) -> Color {
    match (chart, series) {
        (Chart::Employment, 0) => Color::srgb(0.45, 0.90, 0.65),
        (Chart::Employment, _) => Color::srgb(0.95, 0.35, 0.30),
        (_, 0) => Color::srgb(0.95, 0.73, 0.25),
        (_, 1) => Color::srgb(0.40, 0.70, 0.95),
        (_, 2) => Color::srgb(0.45, 0.90, 0.65),
        _ => Color::hsl((series as f32 * 137.5) % 360.0, 0.65, 0.65),
    }
}
fn draw_chart(display: &Display, chart: Chart, range: &ChartRange, gizmos: &mut GizmoAsset) {
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
    gizmos: &mut GizmoAsset,
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
    gizmos: &mut GizmoAsset,
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

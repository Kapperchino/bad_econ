use super::{
    Dashboard, cabinet,
    theme::Tone,
    ui::{self, ActorSelection, UiAction},
};
use crate::{domain::company::AcquisitionScope, simulation::reporting::company_report::Business};
use bevy::prelude::*;
use std::cmp::Ordering;

const PAGE_SIZE: usize = 6;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum Ranking {
    #[default]
    Size,
    Profit,
    Loss,
}

impl Ranking {
    pub fn label(self) -> &'static str {
        match self {
            Self::Size => "Worker capacity",
            Self::Profit => "Highest profit",
            Self::Loss => "Lowest profit",
        }
    }

    fn compare(self, a: &Business, b: &Business) -> Ordering {
        match self {
            Self::Size => b
                .size
                .worker_capacity
                .cmp(&a.size.worker_capacity)
                .then(b.size.sites.cmp(&a.size.sites)),
            Self::Profit => b.profit.cmp(&a.profit),
            Self::Loss => a.profit.cmp(&b.profit),
        }
        .then(a.identity.cmp(&b.identity))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Direction {
    Previous,
    Next,
}

impl Direction {
    pub fn label(self) -> &'static str {
        match self {
            Self::Previous => "Previous page",
            Self::Next => "Next page",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct CompanyPage {
    pub ranking: Ranking,
    page: usize,
}

impl CompanyPage {
    pub fn new(ranking: Ranking) -> Self {
        Self { ranking, page: 0 }
    }

    fn current(self, count: usize) -> usize {
        self.page.min(count.saturating_sub(1) / PAGE_SIZE)
    }

    pub fn turn(&mut self, direction: Direction, count: usize) {
        let current = self.current(count);
        self.page = match direction {
            Direction::Previous => current.saturating_sub(1),
            Direction::Next => (current + 1).min(count.saturating_sub(1) / PAGE_SIZE),
        };
    }
}

pub(super) struct CompaniesPlugin;

impl Plugin for CompaniesPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, draw.run_if(changed).after(super::controls));
    }
}

#[derive(Component)]
struct CompanyPanel;

#[derive(Component)]
struct CompanyRows;

pub(super) fn panel(parent: &mut ChildSpawnerCommands) {
    parent.spawn((
        CompanyPanel,
        Node { display: bevy::ui::Display::None, ..ui::report_panel() },
        ScrollPosition::default(), bevy::ui::RelativeCursorPosition::default(),
    )).with_children(|panel| {
        cabinet::masthead(panel, |title| {
            title.spawn(ui::heading("Companies", 36.0, Tone::OnPrimary));
            title.spawn(ui::text("Capacity counts worker places across open sites. Daily cash profit is measured before owner payouts and acquisitions.", 13.0, Tone::OnPrimary));
        });
        panel.spawn(Node { flex_wrap: FlexWrap::Wrap, row_gap: px(4), ..ui::row(4.0) }).with_children(|controls| {
            for order in [Ranking::Size, Ranking::Profit, Ranking::Loss] {
                ui::button(controls, UiAction::Rank(order), order.label());
            }
        });
        panel.spawn((
            CompanyRows,
            ui::card(12.0),
            Tone::Panel.surface(),
            Tone::Rule.outline(),
        ));
        panel.spawn(ui::text("Companies prefer familiar products and business models. With capacity for 250 workers or 25 open sites, a company can also buy into a new line of business.", 12.0, Tone::Muted));
    });
}

fn changed(
    dashboard: Option<Res<Dashboard>>,
    mut previous: Local<Option<(CompanyPage, u64)>>,
) -> bool {
    let current = dashboard.and_then(|dashboard| {
        dashboard.page.company_page().map(|page| {
            (
                page,
                dashboard
                    .data
                    .lock()
                    .expect("dashboard lock")
                    .observation_revision,
            )
        })
    });
    let changed = *previous != current;
    *previous = current;
    changed
}

fn draw(
    dashboard: Res<Dashboard>,
    mut panels: Query<&mut Node, With<CompanyPanel>>,
    rows: Query<Entity, With<CompanyRows>>,
    mut commands: Commands,
) {
    for mut node in &mut panels {
        let visibility = match dashboard.page.company_page().is_some() {
            true => bevy::ui::Display::Flex,
            false => bevy::ui::Display::None,
        };
        if node.display != visibility {
            node.display = visibility;
        }
    }
    if let (Some(page), Ok(container)) = (dashboard.page.company_page(), rows.single()) {
        let display = dashboard.data.lock().expect("dashboard lock");
        let mut businesses = display.companies.iter().collect::<Vec<_>>();
        businesses.sort_unstable_by(|a, b| page.ranking.compare(a, b));
        commands
            .entity(container)
            .despawn_children()
            .with_children(|rows| {
                draw_rows(rows, &businesses, page, display.snapshot.day);
            });
    }
}

fn distribution(
    parent: &mut ChildSpawnerCommands,
    businesses: &[&Business],
    ranking: Ranking,
    day: u64,
) {
    let labels = match ranking {
        Ranking::Size => ["Under 10 workers", "10–49 workers", "50+ workers"],
        _ => ["Profit", "Break-even", "Loss"],
    };
    let colors = match ranking {
        Ranking::Size => [Tone::Muted, Tone::Accent, Tone::Text],
        _ => [Tone::Positive, Tone::Muted, Tone::Negative],
    };
    let counts = businesses.iter().fold([0usize; 3], |mut counts, business| {
        let band = match ranking {
            Ranking::Size => match business.size.worker_capacity {
                0..10 => 0,
                10..50 => 1,
                _ => 2,
            },
            _ => match business.profit.cmp(&0) {
                Ordering::Greater => 0,
                Ordering::Equal => 1,
                Ordering::Less => 2,
            },
        };
        counts[band] += 1;
        counts
    });
    match (ranking, day) {
        (Ranking::Profit | Ranking::Loss, 0) => {
            parent.spawn(ui::text(
                "Daily profits appear after the first day closes.",
                13.0,
                Tone::Muted,
            ));
        }
        _ => {
            parent
                .spawn(Node {
                    width: percent(100),
                    height: px(7),
                    flex_shrink: 0.0,
                    ..default()
                })
                .with_children(|bar| {
                    for (count, color) in counts
                        .into_iter()
                        .zip(colors)
                        .filter(|(count, _)| *count > 0)
                    {
                        bar.spawn((
                            Node {
                                width: percent(100.0 * count as f32 / businesses.len() as f32),
                                height: percent(100),
                                ..default()
                            },
                            color.surface(),
                        ));
                    }
                });
            parent
                .spawn(Node {
                    flex_wrap: FlexWrap::Wrap,
                    ..ui::row(14.0)
                })
                .with_children(|legend| {
                    for ((label, count), color) in labels.into_iter().zip(counts).zip(colors) {
                        legend.spawn(ui::text(format!("{label}  {count}"), 13.0, color));
                    }
                });
        }
    }
}

fn draw_rows(
    parent: &mut ChildSpawnerCommands,
    businesses: &[&Business],
    page: CompanyPage,
    day: u64,
) {
    let count = businesses.len();
    let current = page.current(count);
    match count {
        0 => {
            parent.spawn(ui::text("No open companies on this day.", 15.0, Tone::Text));
        }
        _ => {
            distribution(parent, businesses, page.ranking, day);
            let start = current * PAGE_SIZE;
            let end = (start + PAGE_SIZE).min(count);
            parent
                .spawn(Node {
                    justify_content: JustifyContent::SpaceBetween,
                    flex_wrap: FlexWrap::Wrap,
                    ..ui::row(6.0)
                })
                .with_children(|header| {
                    header.spawn(ui::text(
                        format!("Day {day}  ·  Companies {}–{end} of {count}", start + 1),
                        14.0,
                        Tone::Accent,
                    ));
                    header.spawn(ui::row(6.0)).with_children(|controls| {
                        if current > 0 {
                            ui::button(
                                controls,
                                UiAction::CompanyPage(Direction::Previous),
                                Direction::Previous.label(),
                            );
                        }
                        if end < count {
                            ui::button(
                                controls,
                                UiAction::CompanyPage(Direction::Next),
                                Direction::Next.label(),
                            );
                        }
                    });
                });
            parent.spawn(ui::column(0.0)).with_children(|table| {
                table
                    .spawn((table_row(), Tone::Raised.surface(), Tone::Brass.outline()))
                    .with_children(|heading| {
                        for caption in ["Company", "Sites", "Capacity", "Daily cash profit", ""] {
                            heading.spawn(ui::heading(caption, 12.0, Tone::Text));
                        }
                    });
                for business in &businesses[start..end] {
                    business_row(table, business, day);
                }
            });
        }
    }
}

fn table_row() -> Node {
    Node {
        display: bevy::ui::Display::Grid,
        grid_template_columns: vec![
            GridTrack::flex(1.0),
            GridTrack::px(46.0),
            GridTrack::px(72.0),
            GridTrack::px(119.0),
            GridTrack::px(92.0),
        ],
        column_gap: px(9),
        align_items: AlignItems::Center,
        padding: UiRect::axes(px(10), px(9)),
        border: UiRect::bottom(px(1)),
        ..default()
    }
}

fn business_row(parent: &mut ChildSpawnerCommands, business: &Business, day: u64) {
    let line = business
        .line
        .map_or_else(|| "Multiple business lines".into(), |line| line.to_string());
    let scope = match business.size.acquisition_scope() {
        AcquisitionScope::ExistingLines => "Existing lines only",
        AcquisitionScope::MayDiversify => "May diversify",
    };
    let (result, color) = match (day, business.profit) {
        (0, _) => ("Awaiting first close".into(), Tone::Muted),
        (_, 0) => ("Break-even".into(), Tone::Muted),
        (_, profit @ 1..) => (format!("Profit  +{profit}"), Tone::Positive),
        (_, loss) => (format!("Loss  {loss}"), Tone::Negative),
    };
    parent
        .spawn((table_row(), Tone::Rule.outline()))
        .with_children(|row| {
            row.spawn(ui::column(3.0)).with_children(|company| {
                company.spawn(ui::text(
                    format!("Company #{}  ·  {line}", business.identity),
                    14.0,
                    Tone::Text,
                ));
                company.spawn(ui::text(scope, 11.0, Tone::Muted));
            });
            row.spawn(ui::text(business.size.sites.to_string(), 13.0, Tone::Text));
            row.spawn(ui::text(
                business.size.worker_capacity.to_string(),
                13.0,
                Tone::Text,
            ));
            row.spawn(ui::text(result, 13.0, color));
            ui::button(
                row,
                UiAction::Inspect(ActorSelection::Identity(business.open_site)),
                "Inspect site",
            );
        });
}

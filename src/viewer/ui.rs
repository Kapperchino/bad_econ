use super::{
    Dashboard, Speed, feedback,
    model::{Chart, Menu, Period, SeriesSelection},
};
use bevy::{
    asset::uuid_handle,
    ecs::query::QueryData,
    input::mouse::{MouseScrollUnit, MouseWheel},
    picking::{Pickable, hover::Hovered},
    prelude::*,
    ui::{Pressed, RelativeCursorPosition},
    ui_widgets::{Activate, Button},
};

const BODY_FONT: Handle<Font> = uuid_handle!("a458633a-3eb8-4f44-8b21-1c00e4136524");
const HEADING_FONT: Handle<Font> = uuid_handle!("c23b024e-f065-4383-849d-f62db58bb35b");

pub(super) const INK: Color = Color::srgb(0.055, 0.080, 0.076);
pub(super) const PANEL: Color = Color::srgb(0.092, 0.125, 0.115);
pub(super) const RAISED: Color = Color::srgb(0.145, 0.185, 0.161);
pub(super) const GOLD: Color = Color::srgb(0.76, 0.64, 0.40);
pub(super) const RULE: Color = Color::srgb(0.30, 0.32, 0.25);
pub(super) const PAPER: Color = Color::srgb(0.91, 0.88, 0.77);
pub(super) const MUTED: Color = Color::srgb(0.60, 0.65, 0.57);

pub(super) struct UiPlugin;
impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<UiAction>()
            .add_observer(activate)
            .add_systems(Startup, setup)
            .add_systems(
                Update,
                (
                    style_buttons,
                    style_markers,
                    scroll,
                    reset_report_scroll,
                    reset_inspector_scroll,
                )
                    .after(super::controls),
            );
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ActorSelection {
    Household,
    Firm,
    Next,
    Previous,
    Identity(u64),
}

#[derive(Component, Message, Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum UiAction {
    Menu(Menu),
    Chart(Chart),
    NextChart,
    Period(Period),
    NextSeries,
    AllSeries,
    ToggleRun,
    Step,
    Reset,
    Speed(Speed),
    Inspect(ActorSelection),
    Compare,
    Sound,
    Motion,
}

impl UiAction {
    pub fn hint(self) -> String {
        match self {
            Self::Menu(menu) => menu.description().into(),
            Self::Chart(chart) => chart.label().into(),
            Self::Period(period) => format!("Show {} across charts", period.label()),
            Self::NextSeries => "Focus on the next series and rescale the chart".into(),
            Self::AllSeries => "Show all series together".into(),
            Self::NextChart => "G: cycle through all graphs".into(),
            Self::ToggleRun => "Space: resume or pause the simulation".into(),
            Self::Step => "N: advance one day and pause".into(),
            Self::Reset => "R: discard this run and restore the launch scenario".into(),
            Self::Speed(speed) => format!("Simulation speed: {}", speed.label()),
            Self::Inspect(ActorSelection::Household) => "H: inspect the first household".into(),
            Self::Inspect(ActorSelection::Firm) => "F: inspect the first firm".into(),
            Self::Inspect(ActorSelection::Next) => "Tab: inspect the next actor".into(),
            Self::Inspect(ActorSelection::Previous) => {
                "Shift+Tab: inspect the previous actor".into()
            }
            Self::Inspect(ActorSelection::Identity(identity)) => format!("Inspect firm {identity}"),
            Self::Compare => {
                "C: save this history and compare the next seed at the same day".into()
            }
            Self::Sound => "M: toggle feedback sounds".into(),
            Self::Motion => "V: toggle feedback flashes".into(),
        }
    }
}

#[derive(Component)]
pub(super) enum Label {
    Resources,
    Date,
    Title,
    Subtitle,
    Summary,
    ChartTitle,
    ChartDescription,
    SeriesFocus,
    Comparison,
    Inspection,
    MapSummary,
    Error,
    ToggleRun,
}

#[derive(Component)]
struct Report;
#[derive(Component)]
struct InspectorDetails;
#[derive(Component)]
pub(super) struct Plot;
#[derive(Component)]
pub(super) struct ChartScale;
#[derive(Component)]
pub(super) struct ChartReadout;
#[derive(Component)]
pub(super) struct Legend;
#[derive(Component)]
pub(super) struct MapMarkers;
#[derive(Component)]
pub(super) struct FirmMarker {
    pub identity: u64,
}
#[derive(Component)]
pub(super) struct Region(pub usize);
#[derive(Component)]
pub(super) struct ChartTab(pub Chart);

fn activate(event: On<Activate>, buttons: Query<&UiAction>, mut actions: MessageWriter<UiAction>) {
    if let Ok(action) = buttons.get(event.entity) {
        actions.write(*action);
    }
}

pub(super) fn text(value: impl Into<String>, size: f32, color: Color) -> impl Bundle {
    (
        Text::new(value),
        TextFont {
            font: match size >= 22.0 {
                true => HEADING_FONT.into(),
                false => BODY_FONT.into(),
            },
            font_size: FontSize::Px(size),
            ..default()
        },
        TextColor(color),
        Pickable::IGNORE,
    )
}

fn column(gap: f32) -> Node {
    Node {
        flex_direction: FlexDirection::Column,
        row_gap: px(gap),
        min_width: px(0),
        ..default()
    }
}

fn row(gap: f32) -> Node {
    Node {
        align_items: AlignItems::Center,
        column_gap: px(gap),
        ..default()
    }
}

pub(super) fn button(parent: &mut ChildSpawnerCommands, action: UiAction, caption: &str) -> Entity {
    parent
        .spawn((
            Button,
            Hovered::default(),
            action,
            Node {
                min_height: px(32),
                padding: UiRect::axes(px(10), px(7)),
                border: UiRect::all(px(1)),
                align_items: AlignItems::Center,
                justify_content: JustifyContent::Center,
                ..default()
            },
            BackgroundColor(PANEL),
            BorderColor::all(RULE),
        ))
        .with_children(|parent| {
            let mut label = parent.spawn(text(caption, 13.0, PAPER));
            if let UiAction::ToggleRun = action {
                label.insert(Label::ToggleRun);
            }
        })
        .id()
}

fn label(parent: &mut ChildSpawnerCommands, kind: Label, size: f32, color: Color) {
    parent.spawn((kind, text("", size, color)));
}

fn rule(parent: &mut ChildSpawnerCommands) {
    parent.spawn((
        Node {
            height: px(1),
            width: percent(100),
            flex_shrink: 0.0,
            ..default()
        },
        BackgroundColor(RULE),
    ));
}

fn setup(mut commands: Commands, mut fonts: ResMut<Assets<Font>>) {
    fonts
        .insert(
            BODY_FONT.id(),
            Font::from_bytes(include_bytes!("../../assets/fonts/Lato-Regular.ttf").to_vec()),
        )
        .expect("body font handle");
    fonts
        .insert(
            HEADING_FONT.id(),
            Font::from_bytes(include_bytes!("../../assets/fonts/LibreBaskerville.ttf").to_vec()),
        )
        .expect("heading font handle");
    commands
        .spawn((
            Node {
                display: bevy::ui::Display::Grid,
                width: percent(100),
                height: percent(100),
                grid_template_columns: vec![GridTrack::flex(1.0)],
                grid_template_rows: vec![
                    GridTrack::px(88.0),
                    GridTrack::flex(1.0),
                    GridTrack::px(44.0),
                ],
                ..default()
            },
            BackgroundColor(INK),
        ))
        .with_children(|root| {
            root.spawn((
                Node {
                    height: px(88),
                    flex_shrink: 0.0,
                    padding: UiRect::axes(px(20), px(12)),
                    border: UiRect::bottom(px(2)),
                    justify_content: JustifyContent::SpaceBetween,
                    ..row(20.0)
                },
                BackgroundColor(PANEL),
                BorderColor::all(GOLD),
            ))
            .with_children(|top| {
                top.spawn(Node {
                    width: px(184),
                    flex_shrink: 0.0,
                    ..column(3.0)
                })
                .with_children(|brand| {
                    brand.spawn(text("S P A T I A L", 22.0, PAPER));
                    brand.spawn(text("E C O N O M Y", 13.0, GOLD));
                });
                top.spawn(Node {
                    flex_grow: 1.0,
                    ..column(6.0)
                })
                .with_children(|resources| {
                    resources.spawn(text("NATIONAL ACCOUNTS", 11.0, GOLD));
                    label(resources, Label::Resources, 15.0, PAPER);
                });
                top.spawn(column(6.0)).with_children(|time| {
                    label(time, Label::Date, 14.0, PAPER);
                    time.spawn(row(5.0)).with_children(|controls| {
                        button(controls, UiAction::ToggleRun, "Resume");
                        button(controls, UiAction::Step, "+1 day");
                        button(controls, UiAction::Speed(Speed::Slow), "0.5 d/s");
                        button(controls, UiAction::Speed(Speed::Normal), "2 d/s");
                        button(controls, UiAction::Speed(Speed::Fast), "Max");
                    });
                });
            });
            root.spawn(Node {
                overflow: Overflow::clip(),
                min_height: px(0),
                ..default()
            })
            .with_children(|body| {
                navigation(body);
                body.spawn(Node {
                    flex_grow: 1.0,
                    flex_basis: px(0),
                    padding: UiRect::all(px(14)),
                    column_gap: px(14),
                    min_width: px(0),
                    min_height: px(0),
                    ..default()
                })
                .with_children(|workspace| {
                    report(workspace);
                    map(workspace);
                    inspector(workspace);
                });
            });
            root.spawn((
                feedback::Banner,
                Node {
                    height: px(44),
                    flex_shrink: 0.0,
                    padding: UiRect::axes(px(16), px(5)),
                    border: UiRect::top(px(1)),
                    justify_content: JustifyContent::SpaceBetween,
                    ..row(10.0)
                },
                BackgroundColor(PANEL),
                BorderColor::all(RULE),
            ))
            .with_children(|footer| {
                footer.spawn((
                    feedback::Label,
                    text("Initializing economy...", 12.0, PAPER),
                ));
                footer.spawn(row(6.0)).with_children(|preferences| {
                    button(preferences, UiAction::Sound, "Sound [M]");
                    button(preferences, UiAction::Motion, "Flashes [V]");
                });
            });
        });
}

fn navigation(parent: &mut ChildSpawnerCommands) {
    parent
        .spawn((
            Node {
                width: px(142),
                flex_shrink: 0.0,
                padding: UiRect::axes(px(10), px(24)),
                border: UiRect::right(px(1)),
                ..column(10.0)
            },
            BackgroundColor(PANEL),
            BorderColor::all(RULE),
        ))
        .with_children(|nav| {
            nav.spawn(text("GOVERNMENT", 11.0, GOLD));
            Menu::ALL.into_iter().for_each(|menu| {
                button(nav, UiAction::Menu(menu), menu.label());
            });
            nav.spawn(Node {
                flex_grow: 1.0,
                ..default()
            });
            rule(nav);
            nav.spawn(text("SIMULATION", 11.0, GOLD));
            button(nav, UiAction::Compare, "Compare seed");
            button(nav, UiAction::Reset, "Reset run");
            nav.spawn(text(
                "Space  Pause / run\nN  Advance day\nG  Next graph\nC  Compare seed",
                11.0,
                MUTED,
            ));
        });
}

fn report(parent: &mut ChildSpawnerCommands) {
    parent
        .spawn((
            Report,
            Node {
                width: percent(52),
                min_width: px(400),
                max_width: px(800),
                flex_shrink: 0.0,
                padding: UiRect::all(px(18)),
                border: UiRect::all(px(1)),
                overflow: Overflow::scroll_y(),
                ..column(12.0)
            },
            ScrollPosition::default(),
            RelativeCursorPosition::default(),
            BackgroundColor(PANEL),
            BorderColor::all(RULE),
        ))
        .with_children(|report| {
            report.spawn(text("ECONOMIC LEDGER", 11.0, GOLD));
            label(report, Label::Title, 30.0, PAPER);
            label(report, Label::Subtitle, 13.0, MUTED);
            rule(report);
            report
                .spawn(Node {
                    flex_wrap: FlexWrap::Wrap,
                    row_gap: px(6),
                    ..row(6.0)
                })
                .with_children(|tabs| {
                    Chart::ALL.into_iter().for_each(|chart| {
                        let entity = button(tabs, UiAction::Chart(chart), chart.tab());
                        tabs.commands().entity(entity).insert(ChartTab(chart));
                    });
                });
            report
                .spawn(Node {
                    flex_wrap: FlexWrap::Wrap,
                    row_gap: px(6),
                    ..row(6.0)
                })
                .with_children(|controls| {
                    Period::ALL.into_iter().for_each(|period| {
                        button(controls, UiAction::Period(period), period.label());
                    });
                    button(controls, UiAction::NextSeries, "Focus next");
                    button(controls, UiAction::AllSeries, "Show all");
                });
            label(report, Label::ChartTitle, 15.0, GOLD);
            label(report, Label::SeriesFocus, 11.0, MUTED);
            report
                .spawn((
                    Node {
                        height: px(240),
                        min_height: px(240),
                        flex_shrink: 0.0,
                        padding: UiRect {
                            left: px(8),
                            right: px(8),
                            top: px(12),
                            bottom: px(12),
                        },
                        border: UiRect::all(px(1)),
                        ..default()
                    },
                    BackgroundColor(INK),
                    BorderColor::all(RULE),
                ))
                .with_children(|frame| {
                    frame.spawn((
                        Plot,
                        RelativeCursorPosition::default(),
                        Node {
                            width: percent(100),
                            height: percent(100),
                            overflow: Overflow::clip(),
                            ..default()
                        },
                    ));
                });
            report.spawn((ChartScale, text("", 11.0, MUTED)));
            report.spawn((
                Legend,
                Node {
                    flex_wrap: FlexWrap::Wrap,
                    column_gap: px(12),
                    row_gap: px(6),
                    ..default()
                },
            ));
            rule(report);
            report.spawn((ChartReadout, text("", 13.0, PAPER)));
            label(report, Label::ChartDescription, 12.0, MUTED);
            rule(report);
            label(report, Label::Summary, 13.0, PAPER);
            label(report, Label::Comparison, 12.0, MUTED);
            label(report, Label::Error, 13.0, Color::srgb(0.94, 0.48, 0.39));
        });
}

fn map(parent: &mut ChildSpawnerCommands) {
    parent
        .spawn((
            Node {
                flex_grow: 1.0,
                flex_basis: px(0),
                min_height: px(0),
                overflow: Overflow::scroll_y(),
                ..column(12.0)
            },
            ScrollPosition::default(),
            RelativeCursorPosition::default(),
        ))
        .with_children(|map| {
            map.spawn(text("THE REGIONS", 22.0, PAPER));
            map.spawn(text("Bread shortages & industry", 12.0, MUTED));
            map.spawn((
                Node {
                    width: percent(100),
                    aspect_ratio: Some(1.0),
                    flex_shrink: 0.0,
                    border: UiRect::all(px(2)),
                    padding: UiRect::all(px(4)),
                    ..default()
                },
                BackgroundColor(INK),
                BorderColor::all(GOLD),
            ))
            .with_children(|frame| {
                frame
                    .spawn(Node {
                        width: percent(100),
                        height: percent(100),
                        ..default()
                    })
                    .with_children(|grid| {
                        (0..256).for_each(|index| {
                            grid.spawn((
                                Region(index),
                                Node {
                                    position_type: PositionType::Absolute,
                                    left: percent((index % 16) as f32 * 6.25),
                                    top: percent((index / 16) as f32 * 6.25),
                                    width: percent(6.25),
                                    height: percent(6.25),
                                    border: UiRect::all(px(0.5)),
                                    ..default()
                                },
                                BackgroundColor(RAISED),
                                BorderColor::all(INK),
                                Pickable::IGNORE,
                            ));
                        });
                        grid.spawn((
                            MapMarkers,
                            Node {
                                width: percent(100),
                                height: percent(100),
                                position_type: PositionType::Absolute,
                                ..default()
                            },
                            Pickable::IGNORE,
                        ));
                    });
            });
            map.spawn(Node {
                justify_content: JustifyContent::SpaceBetween,
                ..row(5.0)
            })
            .with_children(|legend| {
                legend.spawn(text("LOW SHORTAGE", 10.0, MUTED));
                (0..8).for_each(|index| {
                    legend.spawn((
                        Node {
                            height: px(8),
                            flex_grow: 1.0,
                            ..default()
                        },
                        BackgroundColor(region_color(index as f32 / 7.0)),
                    ));
                });
                legend.spawn(text("HIGH", 10.0, MUTED));
            });
            map.spawn(text(
                "Colored markers show industries.\nClick a firm to open its accounts.",
                12.0,
                MUTED,
            ));
            rule(map);
            map.spawn(text("STATE OF THE NATION", 11.0, GOLD));
            label(map, Label::MapSummary, 14.0, PAPER);
        });
}

fn inspector(parent: &mut ChildSpawnerCommands) {
    parent
        .spawn((
            Node {
                width: px(248),
                flex_shrink: 0.0,
                padding: UiRect::all(px(14)),
                border: UiRect::all(px(1)),
                ..column(12.0)
            },
            BackgroundColor(PANEL),
            BorderColor::all(RULE),
        ))
        .with_children(|panel| {
            panel.spawn(text("ACTOR INSPECTOR", 12.0, GOLD));
            panel
                .spawn(Node {
                    flex_wrap: FlexWrap::Wrap,
                    row_gap: px(6),
                    ..row(6.0)
                })
                .with_children(|controls| {
                    button(
                        controls,
                        UiAction::Inspect(ActorSelection::Household),
                        "Household",
                    );
                    button(controls, UiAction::Inspect(ActorSelection::Firm), "Firm");
                    controls.spawn(Node { width: percent(100), ..row(6.0) }).with_children(|navigation| {
                        button(navigation, UiAction::Inspect(ActorSelection::Previous), "Previous");
                        button(navigation, UiAction::Inspect(ActorSelection::Next), "Next");
                    });
                });
            rule(panel);
            panel
                .spawn((
                    InspectorDetails,
                    Node {
                        flex_grow: 1.0,
                        min_height: px(0),
                        overflow: Overflow::scroll_y(),
                        ..column(0.0)
                    },
                    ScrollPosition::default(),
                    RelativeCursorPosition::default(),
                ))
                .with_children(|details| label(details, Label::Inspection, 13.0, PAPER));
            panel.spawn(text(
                "H  Household    F  Firm\nTab / Shift+Tab  Next / previous\nScroll to read all details",
                11.0,
                MUTED,
            ));
        });
}

pub(super) fn region_color(heat: f32) -> Color {
    RAISED.mix(&Color::srgb(0.69, 0.29, 0.18), heat)
}

#[derive(QueryData)]
#[query_data(mutable)]
struct ButtonAppearance {
    action: &'static UiAction,
    hovered: &'static Hovered,
    pressed: Has<Pressed>,
    background: &'static mut BackgroundColor,
    border: &'static mut BorderColor,
}

fn style_buttons(
    dashboard: Option<Res<Dashboard>>,
    mut buttons: Query<ButtonAppearance, Without<FirmMarker>>,
) {
    if let Some(dashboard) = dashboard {
        let display = dashboard.data.lock().expect("dashboard lock");
        buttons.iter_mut().for_each(|mut button| {
            let selected = match button.action {
                UiAction::Menu(menu) => *menu == dashboard.chart.menu(),
                UiAction::Chart(chart) => *chart == dashboard.chart,
                UiAction::Period(period) => *period == dashboard.period,
                UiAction::AllSeries => dashboard.series == SeriesSelection::All,
                UiAction::NextSeries => matches!(dashboard.series, SeriesSelection::Only(_)),
                UiAction::Speed(speed) => *speed == display.speed,
                _ => false,
            };
            let (color, outline) = match (button.pressed, selected, button.hovered.get()) {
                (true, _, _) => (GOLD.with_alpha(0.45), GOLD),
                (_, true, _) => (Color::srgb(0.26, 0.29, 0.20), GOLD),
                (_, _, true) => (RAISED, GOLD),
                _ => (PANEL, RULE),
            };
            button.background.set_if_neq(BackgroundColor(color));
            button.border.set_if_neq(BorderColor::all(outline));
        });
    }
}

fn scroll(
    mut wheel: MessageReader<MouseWheel>,
    mut panels: Query<(&RelativeCursorPosition, &ComputedNode, &mut ScrollPosition)>,
) {
    let delta = wheel
        .read()
        .map(|event| {
            -event.y
                * match event.unit {
                    MouseScrollUnit::Line => 28.0,
                    MouseScrollUnit::Pixel => 1.0,
                }
        })
        .sum::<f32>();
    if delta != 0.0 {
        panels
            .iter_mut()
            .filter(|(cursor, _, _)| cursor.cursor_over())
            .for_each(|(_, computed, mut position)| {
                let maximum = ((computed.content_size().y - computed.size().y)
                    * computed.inverse_scale_factor())
                .max(0.0);
                position.0.y = (position.0.y + delta).clamp(0.0, maximum);
            });
    }
}

fn reset_report_scroll(
    dashboard: Option<Res<Dashboard>>,
    mut panels: Query<&mut ScrollPosition, With<Report>>,
    mut previous: Local<Option<Menu>>,
) {
    if let Some(dashboard) = dashboard {
        let menu = dashboard.chart.menu();
        if *previous != Some(menu) {
            panels
                .iter_mut()
                .for_each(|mut position| position.0 = Vec2::ZERO);
            *previous = Some(menu);
        }
    }
}

fn style_markers(
    dashboard: Option<Res<Dashboard>>,
    mut markers: Query<(&FirmMarker, &Hovered, &mut BorderColor, &mut Node)>,
) {
    if let Some(dashboard) = dashboard {
        markers
            .iter_mut()
            .for_each(|(marker, hovered, mut border, mut node)| {
                let (color, width) = match (marker.identity == dashboard.selected, hovered.get()) {
                    (true, _) | (_, true) => (GOLD, 2.0),
                    _ => (INK, 1.0),
                };
                border.set_if_neq(BorderColor::all(color));
                let thickness = UiRect::all(px(width));
                if node.border != thickness {
                    node.border = thickness;
                }
            });
    }
}

fn reset_inspector_scroll(
    dashboard: Option<Res<Dashboard>>,
    mut panels: Query<&mut ScrollPosition, With<InspectorDetails>>,
    mut previous: Local<Option<u64>>,
) {
    let current = dashboard.and_then(|dashboard| {
        dashboard
            .data
            .lock()
            .expect("dashboard lock")
            .inspection
            .as_ref()
            .map(|inspection| inspection.actor.identity)
    });
    if current != *previous {
        panels
            .iter_mut()
            .for_each(|mut position| position.0 = Vec2::ZERO);
        *previous = current;
    }
}

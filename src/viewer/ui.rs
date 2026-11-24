use super::{
    Dashboard, DashboardPage, Speed,
    companies::{self, Direction, Ranking},
    feedback,
    model::{Chart, Menu, Period, SeriesSelection},
    theme::{Appearance, AppearanceSettings, Theme, Tone},
};
use bevy::{
    a11y::AccessibilityNode,
    asset::uuid_handle,
    ecs::query::QueryData,
    input::mouse::{MouseScrollUnit, MouseWheel},
    picking::{Pickable, hover::Hovered},
    prelude::*,
    ui::{Checked, ComputedStackIndex, Pressed, RelativeCursorPosition},
    ui_widgets::{Activate, Button, Checkbox, ValueChange},
};

const BODY_FONT: Handle<Font> = uuid_handle!("a458633a-3eb8-4f44-8b21-1c00e4136524");
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
                    style_theme_switch,
                    style_markers,
                    style_workspace,
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
    Companies,
    Rank(Ranking),
    CompanyPage(Direction),
    Chart(Chart),
    NextChart,
    Period(Period),
    Metrics,
    FocusSeries(usize),
    AllSeries,
    ToggleRun,
    Step,
    Reset,
    Speed(Speed),
    Inspect(ActorSelection),
    Compare,
    ToggleDarkMode,
    Sound,
    Motion,
}

impl UiAction {
    pub fn hint(self) -> String {
        match self {
            Self::Companies => "Compare company sizes and daily cash profits".into(),
            Self::Rank(order) => order.label().into(),
            Self::CompanyPage(direction) => direction.label().into(),
            Self::Menu(menu) => menu.description().into(),
            Self::Chart(chart) => chart.label().into(),
            Self::Period(period) => format!("Show {} across charts", period.label()),
            Self::Metrics => "Esc: return to the dashboard".into(),
            Self::FocusSeries(_) => "Click a series to isolate it or restore all series".into(),
            Self::AllSeries => "Show all series together".into(),
            Self::NextChart => "G: cycle through all graphs".into(),
            Self::ToggleRun => "Space: resume or pause the simulation".into(),
            Self::Step => "N: advance one day and pause".into(),
            Self::Reset => "R: discard this run and restore the launch scenario".into(),
            Self::Speed(speed) => format!("Simulation speed: {}", speed.label()),
            Self::Inspect(ActorSelection::Household) => {
                "Inspect the first population account (H)".into()
            }
            Self::Inspect(ActorSelection::Firm) => "F: inspect the first firm".into(),
            Self::Inspect(ActorSelection::Next) => "Tab: inspect the next actor".into(),
            Self::Inspect(ActorSelection::Previous) => {
                "Shift+Tab: inspect the previous actor".into()
            }
            Self::Inspect(ActorSelection::Identity(identity)) => format!("Inspect firm {identity}"),
            Self::Compare => {
                "C: save this history and compare the next seed at the same day".into()
            }
            Self::ToggleDarkMode => "Toggle dark mode for this session".into(),
            Self::Sound => "M: toggle feedback sounds".into(),
            Self::Motion => "V: toggle feedback flashes".into(),
        }
    }
}

#[derive(Component)]
pub(super) enum Label {
    Seed,
    Date,
    Status,
    Title,
    Subtitle,
    Summary,
    ChartTitle,
    ChartDescription,
    Comparison,
    MetricName(usize),
    MetricValue(usize),
    MetricChange(usize),
    Inspection,
    MapSummary,
    Error,
    ToggleRun,
}

#[derive(Component)]
pub(super) struct Report;
#[derive(Component)]
struct InspectorDetails;
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Plot {
    Preview(Chart),
    Detail,
}

#[derive(Component)]
pub(super) struct Crosshair;

#[derive(Component)]
pub(super) struct CursorDate;

#[derive(Component)]
enum WorkspaceSection {
    Report,
    Grid,
    Detail,
    Preview(Chart),
}

#[derive(Component)]
struct MetricCard;

#[derive(Component)]
struct NavButton;

#[derive(Component)]
struct PrimaryButton;

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

pub(super) fn text(value: impl Into<String>, size: f32, tone: Tone) -> impl Bundle {
    (
        Text::new(value),
        TextFont {
            font: BODY_FONT.into(),
            font_size: FontSize::Px(size),
            ..default()
        },
        tone.foreground(),
        Pickable::IGNORE,
    )
}

pub(super) fn column(gap: f32) -> Node {
    Node {
        flex_direction: FlexDirection::Column,
        row_gap: px(gap),
        min_width: px(0),
        ..default()
    }
}

pub(super) fn row(gap: f32) -> Node {
    Node {
        align_items: AlignItems::Center,
        column_gap: px(gap),
        ..default()
    }
}

pub(super) fn button(parent: &mut ChildSpawnerCommands, action: UiAction, caption: &str) -> Entity {
    button_with(parent, action, caption, Tone::Text, |_| {})
}

fn button_with(
    parent: &mut ChildSpawnerCommands,
    action: UiAction,
    caption: &str,
    tone: Tone,
    content: impl FnOnce(&mut ChildSpawnerCommands),
) -> Entity {
    let mut button = parent.spawn((
        Button,
        Hovered::default(),
        action,
        Node {
            min_height: px(29),
            column_gap: px(6),
            padding: UiRect::axes(px(9), px(5)),
            border: UiRect::all(px(1)),
            border_radius: BorderRadius::all(px(5)),
            align_items: AlignItems::Center,
            justify_content: JustifyContent::Center,
            flex_shrink: 0.0,
            ..default()
        },
        BackgroundColor(Theme::Light.color(Tone::Panel)),
        BorderColor::all(Theme::Light.color(Tone::Rule)),
    ));
    let entity = button.id();
    button.with_children(|parent| {
        content(parent);
        let mut label = parent.spawn(text(caption, 12.0, tone));
        if let UiAction::ToggleRun = action {
            label.insert(Label::ToggleRun);
        }
    });
    entity
}

pub(super) fn series_button(
    parent: &mut ChildSpawnerCommands,
    index: usize,
    name: &str,
    tone: Tone,
) {
    button_with(
        parent,
        UiAction::FocusSeries(index),
        name,
        Tone::Text,
        |parent| swatch(parent, tone),
    );
}

pub(super) fn swatch(parent: &mut ChildSpawnerCommands, tone: Tone) {
    parent.spawn((
        Node {
            width: px(8),
            height: px(8),
            flex_shrink: 0.0,
            border_radius: BorderRadius::MAX,
            ..default()
        },
        tone.surface(),
        Pickable::IGNORE,
    ));
}

fn nav_button(parent: &mut ChildSpawnerCommands, action: UiAction, caption: &str) {
    let entity = button_with(parent, action, caption, Tone::OnPrimary, |_| {});
    parent.commands().entity(entity).insert((
        NavButton,
        Node {
            min_height: px(35),
            padding: UiRect::horizontal(px(12)),
            align_items: AlignItems::Center,
            border: UiRect::left(px(3)),
            border_radius: BorderRadius::all(px(5)),
            ..default()
        },
    ));
}

fn label(parent: &mut ChildSpawnerCommands, kind: Label, size: f32, tone: Tone) {
    parent.spawn((kind, text("", size, tone)));
}

fn card(gap: f32) -> Node {
    Node {
        flex_shrink: 0.0,
        padding: UiRect::all(px(13)),
        border: UiRect::all(px(1)),
        border_radius: BorderRadius::all(px(8)),
        ..column(gap)
    }
}

fn setup(mut commands: Commands, mut fonts: ResMut<Assets<Font>>) {
    fonts
        .insert(
            BODY_FONT.id(),
            Font::from_bytes(include_bytes!("../../assets/fonts/Lato-Regular.ttf").to_vec()),
        )
        .expect("body font handle");
    commands
        .spawn((
            Node {
                width: percent(100),
                height: percent(100),
                ..default()
            },
            Tone::Canvas.surface(),
        ))
        .with_children(|root| {
            navigation(root);
            root.spawn(Node {
                display: bevy::ui::Display::Grid,
                grid_template_columns: vec![GridTrack::flex(1.0)],
                grid_template_rows: vec![
                    GridTrack::px(58.0),
                    GridTrack::flex(1.0),
                    GridTrack::px(40.0),
                ],
                height: percent(100),
                flex_grow: 1.0,
                flex_basis: px(0),
                min_width: px(0),
                min_height: px(0),
                ..default()
            })
            .with_children(|main| {
                run_toolbar(main);
                main.spawn(Node {
                    flex_grow: 1.0,
                    flex_basis: px(0),
                    min_height: px(0),
                    overflow: Overflow::clip(),
                    ..default()
                })
                .with_children(|workspace| {
                    report(workspace);
                    companies::panel(workspace);
                    context(workspace);
                });
                footer(main);
            });
        });
}

fn navigation(parent: &mut ChildSpawnerCommands) {
    parent
        .spawn((
            Node {
                width: px(162),
                flex_shrink: 0.0,
                padding: UiRect::axes(px(11), px(20)),
                ..column(5.0)
            },
            Tone::Nav.surface(),
        ))
        .with_children(|nav| {
            nav.spawn(Node {
                margin: UiRect::bottom(px(24)),
                ..row(8.0)
            })
            .with_children(|brand| {
                brand
                    .spawn((
                        Node {
                            width: px(28),
                            height: px(28),
                            align_items: AlignItems::Center,
                            justify_content: JustifyContent::Center,
                            border_radius: BorderRadius::all(px(7)),
                            ..default()
                        },
                        Tone::Primary.surface(),
                    ))
                    .with_children(|mark| {
                        mark.spawn(text("S", 17.0, Tone::OnPrimary));
                    });
                brand.spawn(text("Spatial Economy", 14.0, Tone::OnPrimary));
            });
            nav.spawn((
                Node {
                    margin: UiRect::axes(px(9), px(5)),
                    ..default()
                },
                text("DASHBOARDS", 11.0, Tone::NavMuted),
            ));
            for menu in Menu::ALL {
                nav_button(nav, UiAction::Menu(menu), menu.label());
            }
            nav_button(nav, UiAction::Companies, "Companies");
            nav.spawn(Node {
                flex_grow: 1.0,
                ..default()
            });
            nav.spawn((
                Node {
                    margin: UiRect::axes(px(9), px(5)),
                    ..default()
                },
                text("SIMULATION", 11.0, Tone::NavMuted),
            ));
            nav_button(nav, UiAction::Compare, "Compare seed");
            nav_button(nav, UiAction::Reset, "Reset run");
        });
}

fn run_toolbar(parent: &mut ChildSpawnerCommands) {
    parent
        .spawn((
            Node {
                height: px(58),
                flex_shrink: 0.0,
                padding: UiRect::axes(px(18), px(10)),
                border: UiRect::bottom(px(1)),
                justify_content: JustifyContent::SpaceBetween,
                ..row(16.0)
            },
            Tone::Panel.surface(),
            Tone::Rule.outline(),
        ))
        .with_children(|top| {
            top.spawn(row(12.0)).with_children(|context| {
                context
                    .spawn((
                        Node {
                            padding: UiRect::axes(px(8), px(4)),
                            border_radius: BorderRadius::all(px(5)),
                            ..row(0.0)
                        },
                        Tone::Raised.surface(),
                    ))
                    .with_children(|status| label(status, Label::Status, 12.0, Tone::Accent));
                label(context, Label::Date, 13.0, Tone::Text);
                label(context, Label::Seed, 12.0, Tone::Muted);
            });
            top.spawn(row(5.0)).with_children(|controls| {
                controls.spawn((
                    Node {
                        margin: UiRect::right(px(4)),
                        ..default()
                    },
                    text("Playback", 12.0, Tone::Muted),
                ));
                let run = button_with(
                    controls,
                    UiAction::ToggleRun,
                    "Resume",
                    Tone::OnPrimary,
                    |_| {},
                );
                controls.commands().entity(run).insert(PrimaryButton);
                button(controls, UiAction::Step, "+1 day");
                button(controls, UiAction::Speed(Speed::Slow), "0.5 d/s");
                button(controls, UiAction::Speed(Speed::Normal), "2 d/s");
                button(controls, UiAction::Speed(Speed::Fast), "Max");
            });
        });
}

fn footer(parent: &mut ChildSpawnerCommands) {
    parent
        .spawn((
            feedback::Banner,
            Node {
                height: px(40),
                flex_shrink: 0.0,
                padding: UiRect::axes(px(16), px(5)),
                border: UiRect::top(px(1)),
                justify_content: JustifyContent::SpaceBetween,
                ..row(10.0)
            },
            BackgroundColor(Theme::Light.color(Tone::Panel)),
            Tone::Rule.outline(),
        ))
        .with_children(|footer| {
            footer.spawn((
                feedback::Label,
                text("Initializing economy…", 12.0, Tone::Text),
            ));
            footer.spawn(row(5.0)).with_children(|preferences| {
                dark_mode_switch(preferences);
                button(preferences, UiAction::Sound, "Sound [M]");
                button(preferences, UiAction::Motion, "Flashes [V]");
            });
        });
}

#[derive(Component)]
pub(super) struct ThemeSwitch;

#[derive(Component)]
struct SwitchTrack;

fn dark_mode_switch(parent: &mut ChildSpawnerCommands) {
    let mut accessible = accesskit::Node::new(accesskit::Role::Switch);
    accessible.set_label("Dark mode");
    parent
        .spawn((
            ThemeSwitch,
            Checkbox,
            AccessibilityNode(accessible),
            UiAction::ToggleDarkMode,
            Hovered::default(),
            Node {
                min_height: px(29),
                margin: UiRect::right(px(10)),
                ..row(7.0)
            },
        ))
        .observe(change_appearance)
        .with_children(|switch| {
            switch.spawn(text("Dark mode", 12.0, Tone::Text));
            switch
                .spawn((
                    SwitchTrack,
                    Node {
                        width: px(34),
                        height: px(19),
                        padding: UiRect::all(px(2)),
                        border: UiRect::all(px(1)),
                        border_radius: BorderRadius::MAX,
                        ..row(0.0)
                    },
                    BackgroundColor::default(),
                    BorderColor::default(),
                    Pickable::IGNORE,
                ))
                .with_children(|track| {
                    track.spawn((
                        Node {
                            width: px(13),
                            height: px(13),
                            border_radius: BorderRadius::MAX,
                            ..default()
                        },
                        Tone::OnPrimary.surface(),
                        Pickable::IGNORE,
                    ));
                });
        });
}

fn change_appearance(event: On<ValueChange<bool>>, mut settings: ResMut<AppearanceSettings>) {
    settings.selection = Appearance::Override(match event.value {
        true => Theme::Dark,
        false => Theme::Light,
    });
}

fn style_theme_switch(
    theme: Res<Theme>,
    switch: Query<(Entity, Has<Checked>, &Hovered), With<ThemeSwitch>>,
    mut track: Query<(&mut Node, &mut BackgroundColor, &mut BorderColor), With<SwitchTrack>>,
    mut commands: Commands,
) {
    if let (Ok((entity, checked, hovered)), Ok((mut node, mut background, mut border))) =
        (switch.single(), track.single_mut())
    {
        match (*theme, checked) {
            (Theme::Dark, false) => {
                commands.entity(entity).insert(Checked);
            }
            (Theme::Light, true) => {
                commands.entity(entity).remove::<Checked>();
            }
            _ => (),
        }
        let (fill, outline, align) = match (*theme, hovered.get()) {
            (Theme::Dark, true) => (Tone::Primary, Tone::Accent, JustifyContent::End),
            (Theme::Dark, false) => (Tone::Primary, Tone::Primary, JustifyContent::End),
            (Theme::Light, true) => (Tone::Muted, Tone::Accent, JustifyContent::Start),
            (Theme::Light, false) => (Tone::Muted, Tone::Muted, JustifyContent::Start),
        };
        if node.justify_content != align {
            node.justify_content = align;
        }
        background.set_if_neq(BackgroundColor(theme.color(fill)));
        border.set_if_neq(BorderColor::all(theme.color(outline)));
    }
}

pub(super) fn report_panel() -> Node {
    Node {
        flex_grow: 1.0,
        flex_basis: px(0),
        min_width: px(0),
        min_height: px(0),
        padding: UiRect::all(px(18)),
        overflow: Overflow::scroll_y(),
        ..column(14.0)
    }
}

fn report(parent: &mut ChildSpawnerCommands) {
    parent
        .spawn((
            Report,
            WorkspaceSection::Report,
            report_panel(),
            ScrollPosition::default(),
            RelativeCursorPosition::default(),
        ))
        .with_children(|report| {
            report
                .spawn(Node {
                    justify_content: JustifyContent::SpaceBetween,
                    align_items: AlignItems::End,
                    flex_wrap: FlexWrap::Wrap,
                    row_gap: px(8),
                    flex_shrink: 0.0,
                    ..row(12.0)
                })
                .with_children(|header| {
                    header.spawn(column(4.0)).with_children(|heading| {
                        label(heading, Label::Title, 30.0, Tone::Text);
                        label(heading, Label::Subtitle, 13.0, Tone::Muted);
                    });
                    header.spawn(row(4.0)).with_children(|periods| {
                        periods.spawn((
                            Node {
                                margin: UiRect::right(px(4)),
                                ..default()
                            },
                            text("Time range", 12.0, Tone::Muted),
                        ));
                        for period in Period::ALL {
                            button(periods, UiAction::Period(period), period.label());
                        }
                    });
                });
            report
                .spawn(Node {
                    justify_content: JustifyContent::SpaceBetween,
                    flex_wrap: FlexWrap::Wrap,
                    flex_shrink: 0.0,
                    ..row(6.0)
                })
                .with_children(|status| {
                    label(status, Label::Comparison, 12.0, Tone::Muted);
                    status.spawn((CursorDate, text("", 12.0, Tone::Accent)));
                });
            label(report, Label::Error, 13.0, Tone::Negative);
            report
                .spawn((WorkspaceSection::Grid, column(14.0)))
                .with_children(|metrics| {
                    metric_values(metrics);
                    metrics
                        .spawn(Node {
                            display: bevy::ui::Display::Grid,
                            grid_template_columns: vec![GridTrack::flex(1.0), GridTrack::flex(1.0)],
                            row_gap: px(12),
                            column_gap: px(12),
                            ..default()
                        })
                        .with_children(|cards| {
                            for chart in Menu::ALL.into_iter().flat_map(Menu::charts).copied() {
                                metric_card(cards, chart);
                            }
                        });
                    metrics
                        .spawn((card(5.0), Tone::Panel.surface(), Tone::Rule.outline()))
                        .with_children(|details| {
                            details.spawn(text("Snapshot details", 15.0, Tone::Text));
                            label(details, Label::Summary, 13.0, Tone::Muted);
                        });
                });
            chart_detail(report);
        });
}

fn metric_values(parent: &mut ChildSpawnerCommands) {
    parent
        .spawn(Node {
            display: bevy::ui::Display::Grid,
            grid_template_columns: vec![GridTrack::flex(1.0); 4],
            column_gap: px(12),
            ..default()
        })
        .with_children(|cards| {
            for index in 0..4 {
                cards
                    .spawn((card(5.0), Tone::Panel.surface(), Tone::Rule.outline()))
                    .with_children(|metric| {
                        label(metric, Label::MetricName(index), 12.0, Tone::Muted);
                        label(metric, Label::MetricValue(index), 25.0, Tone::Text);
                        label(metric, Label::MetricChange(index), 11.0, Tone::Muted);
                    });
            }
        });
}

fn metric_card(parent: &mut ChildSpawnerCommands, chart: Chart) {
    parent
        .spawn((
            MetricCard,
            WorkspaceSection::Preview(chart),
            Button,
            Hovered::default(),
            UiAction::Chart(chart),
            card(5.0),
            BackgroundColor(Theme::Light.color(Tone::Panel)),
            BorderColor::all(Theme::Light.color(Tone::Rule)),
        ))
        .with_children(|card| {
            card.spawn(Node {
                justify_content: JustifyContent::SpaceBetween,
                ..row(8.0)
            })
            .with_children(|heading| {
                heading.spawn(text(chart.label(), 16.0, Tone::Text));
                heading.spawn(text("Open ›", 11.0, Tone::Accent));
            });
            card.spawn(text(chart.unit(), 11.0, Tone::Muted));
            plot(
                card,
                Plot::Preview(chart),
                Node {
                    width: percent(100),
                    height: px(142),
                    ..default()
                },
            );
            card.spawn(Node {
                flex_wrap: FlexWrap::Wrap,
                row_gap: px(4),
                ..row(8.0)
            })
            .with_children(|legend| {
                let series = chart.series();
                for (index, name) in series.iter().take(2).enumerate() {
                    legend.spawn(row(4.0)).with_children(|item| {
                        swatch(item, Tone::Series(chart, index));
                        item.spawn(text(name, 11.0, Tone::Muted));
                    });
                }
                if series.len() > 2 {
                    legend.spawn(text(
                        format!("+{} series", series.len() - 2),
                        11.0,
                        Tone::Muted,
                    ));
                }
            });
        });
}

fn chart_detail(parent: &mut ChildSpawnerCommands) {
    parent
        .spawn((
            WorkspaceSection::Detail,
            Node {
                display: bevy::ui::Display::None,
                ..column(12.0)
            },
        ))
        .with_children(|detail| {
            detail
                .spawn(Node {
                    flex_wrap: FlexWrap::Wrap,
                    row_gap: px(5),
                    ..row(5.0)
                })
                .with_children(|tabs| {
                    button(tabs, UiAction::Metrics, "‹ Dashboard");
                    for chart in Chart::ALL {
                        let entity = button(tabs, UiAction::Chart(chart), chart.tab());
                        tabs.commands().entity(entity).insert(ChartTab(chart));
                    }
                });
            detail
                .spawn((card(10.0), Tone::Panel.surface(), Tone::Rule.outline()))
                .with_children(|card| {
                    label(card, Label::ChartTitle, 21.0, Tone::Text);
                    label(card, Label::ChartDescription, 12.0, Tone::Muted);
                    card.spawn(Node {
                        height: px(294),
                        align_items: AlignItems::Stretch,
                        ..row(12.0)
                    })
                    .with_children(|detail| {
                        plot(
                            detail,
                            Plot::Detail,
                            Node {
                                flex_grow: 1.0,
                                flex_basis: px(0),
                                min_width: px(0),
                                ..default()
                            },
                        );
                        detail
                            .spawn((
                                Node {
                                    width: px(210),
                                    padding: UiRect::left(px(11)),
                                    border: UiRect::left(px(1)),
                                    overflow: Overflow::scroll_y(),
                                    ..column(6.0)
                                },
                                Tone::Rule.outline(),
                                ScrollPosition::default(),
                                RelativeCursorPosition::default(),
                            ))
                            .with_children(|readout| {
                                readout.spawn(text("Observation", 14.0, Tone::Text));
                                readout.spawn((ChartReadout, text("", 12.0, Tone::Muted)));
                            });
                    });
                    card.spawn((
                        Legend,
                        Node {
                            flex_wrap: FlexWrap::Wrap,
                            row_gap: px(5),
                            ..row(5.0)
                        },
                    ));
                });
        });
}

fn plot(parent: &mut ChildSpawnerCommands, kind: Plot, mut layout: Node) {
    layout.flex_shrink = 0.0;
    layout.overflow = Overflow::clip();
    parent.spawn((
        kind,
        RelativeCursorPosition::default(),
        layout,
        Pickable::IGNORE,
    ));
}

fn context(parent: &mut ChildSpawnerCommands) {
    parent
        .spawn((
            Node {
                width: px(298),
                min_height: px(0),
                flex_shrink: 0.0,
                padding: UiRect::all(px(14)),
                border: UiRect::left(px(1)),
                ..column(12.0)
            },
            Tone::Rule.outline(),
            Tone::Canvas.surface(),
        ))
        .with_children(|rail| {
            map(rail);
            inspector(rail);
        });
}

fn map(parent: &mut ChildSpawnerCommands) {
    parent
        .spawn((card(8.0), Tone::Panel.surface(), Tone::Rule.outline()))
        .with_children(|map| {
            map.spawn(text("Regional map", 16.0, Tone::Text));
            map.spawn((
                Node {
                    width: percent(100),
                    aspect_ratio: Some(1.0),
                    flex_shrink: 0.0,
                    border: UiRect::all(px(1)),
                    padding: UiRect::all(px(5)),
                    ..default()
                },
                Tone::Canvas.surface(),
                Tone::Rule.outline(),
            ))
            .with_children(|frame| {
                frame
                    .spawn(Node {
                        width: percent(100),
                        height: percent(100),
                        ..default()
                    })
                    .with_children(|grid| {
                        for index in 0..256 {
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
                                BackgroundColor(Theme::Light.color(Tone::Region(0.0))),
                                Tone::Panel.outline(),
                                Pickable::IGNORE,
                            ));
                        }
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
            map.spawn(row(5.0)).with_children(|legend| {
                legend.spawn(text("Bread shortage", 11.0, Tone::Muted));
                for index in 0..8 {
                    legend.spawn((
                        Node {
                            height: px(7),
                            flex_grow: 1.0,
                            ..default()
                        },
                        Tone::Region(index as f32 / 7.0).surface(),
                    ));
                }
                legend.spawn(text("High", 11.0, Tone::Muted));
            });
            label(map, Label::MapSummary, 11.0, Tone::Muted);
        });
}

fn inspector(parent: &mut ChildSpawnerCommands) {
    parent
        .spawn((
            Node {
                flex_grow: 1.0,
                flex_basis: px(0),
                min_height: px(0),
                ..card(10.0)
            },
            Tone::Panel.surface(),
            Tone::Rule.outline(),
        ))
        .with_children(|panel| {
            panel.spawn(text("Actor inspector", 16.0, Tone::Text));
            panel.spawn(row(4.0)).with_children(|controls| {
                for (selection, caption) in [
                    (ActorSelection::Household, "Account"),
                    (ActorSelection::Firm, "Firm"),
                    (ActorSelection::Previous, "Previous"),
                    (ActorSelection::Next, "Next"),
                ] {
                    button(controls, UiAction::Inspect(selection), caption);
                }
            });
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
                .with_children(|details| label(details, Label::Inspection, 13.0, Tone::Text));
        });
}

fn style_workspace(
    dashboard: Option<Res<Dashboard>>,
    mut nodes: Query<(&WorkspaceSection, &mut Node)>,
) {
    if let Some(dashboard) = dashboard.filter(|dashboard| dashboard.is_changed()) {
        for (section, mut node) in &mut nodes {
            let display = match (dashboard.page, section) {
                (DashboardPage::Companies(_), _) => bevy::ui::Display::None,
                (_, WorkspaceSection::Report)
                | (DashboardPage::Chart, WorkspaceSection::Detail)
                | (DashboardPage::Metrics, WorkspaceSection::Grid) => bevy::ui::Display::Flex,
                (DashboardPage::Metrics, WorkspaceSection::Preview(chart))
                    if chart.menu() == dashboard.chart.menu() =>
                {
                    bevy::ui::Display::Flex
                }
                _ => bevy::ui::Display::None,
            };
            if node.display != display {
                node.display = display;
            }
        }
    }
}

#[derive(QueryData)]
#[query_data(mutable)]
struct ButtonAppearance {
    action: &'static UiAction,
    hovered: &'static Hovered,
    pressed: Has<Pressed>,
    nav: Has<NavButton>,
    primary: Has<PrimaryButton>,
    card: Has<MetricCard>,
    tab: Has<ChartTab>,
    background: &'static mut BackgroundColor,
    border: &'static mut BorderColor,
}

fn style_buttons(
    dashboard: Option<Res<Dashboard>>,
    theme: Res<Theme>,
    mut buttons: Query<ButtonAppearance, Without<FirmMarker>>,
) {
    if let Some(dashboard) = dashboard {
        let display = dashboard.data.lock().expect("dashboard lock");
        buttons.iter_mut().for_each(|mut button| {
            let selected = match button.action {
                UiAction::Menu(menu) => {
                    dashboard.page.company_page().is_none() && *menu == dashboard.chart.menu()
                }
                UiAction::Companies => dashboard.page.company_page().is_some(),
                UiAction::Rank(order) => dashboard
                    .page
                    .company_page()
                    .is_some_and(|page| page.ranking == *order),
                UiAction::Chart(chart) => button.tab && *chart == dashboard.chart,
                UiAction::Period(period) => *period == dashboard.period,
                UiAction::AllSeries => dashboard.series == SeriesSelection::All,
                UiAction::FocusSeries(index) => dashboard.series == SeriesSelection::Only(*index),
                UiAction::Speed(speed) => *speed == display.speed,
                _ => false,
            };
            let (color, outline) = match (
                button.nav,
                button.primary,
                button.card,
                button.pressed,
                selected,
                button.hovered.get(),
            ) {
                (true, _, _, _, true, _) => (Tone::NavSelected, Tone::NavAccent),
                (true, _, _, _, _, true) => (Tone::NavHover, Tone::NavHover),
                (true, _, _, _, _, _) => (Tone::Nav, Tone::Nav),
                (_, true, _, _, _, true) => (Tone::NavSelected, Tone::NavSelected),
                (_, true, ..) => (Tone::Primary, Tone::Primary),
                (_, _, true, _, _, true) => (Tone::Panel, Tone::Accent),
                (_, _, _, true, _, _) => (Tone::Raised, Tone::Accent),
                (_, _, _, _, true, _) => (Tone::Raised, Tone::Accent),
                (_, _, _, _, _, true) => (Tone::Canvas, Tone::Accent),
                _ => (Tone::Panel, Tone::Rule),
            };
            button
                .background
                .set_if_neq(BackgroundColor(theme.color(color)));
            button
                .border
                .set_if_neq(BorderColor::all(theme.color(outline)));
        });
    }
}

fn scroll(
    mut wheel: MessageReader<MouseWheel>,
    mut panels: Query<(
        &ComputedStackIndex,
        &RelativeCursorPosition,
        &ComputedNode,
        &mut ScrollPosition,
    )>,
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
        let target = panels
            .iter_mut()
            .filter_map(|(stack, cursor, computed, position)| {
                let maximum = ((computed.content_size().y - computed.size().y)
                    * computed.inverse_scale_factor())
                .max(0.0);
                let next = (position.0.y + delta).clamp(0.0, maximum);
                (cursor.cursor_over() && next != position.0.y).then_some((stack.0, position, next))
            })
            .max_by_key(|(stack, _, _)| *stack);
        if let Some((_, mut position, next)) = target {
            position.0.y = next;
        }
    }
}

fn reset_report_scroll(
    dashboard: Option<Res<Dashboard>>,
    mut panels: Query<&mut ScrollPosition, With<Report>>,
    mut previous: Local<Option<(Menu, DashboardPage)>>,
) {
    let current = dashboard.map(|dashboard| (dashboard.chart.menu(), dashboard.page));
    if current != *previous {
        panels
            .iter_mut()
            .for_each(|mut position| position.0 = Vec2::ZERO);
        *previous = current;
    }
}

fn style_markers(
    dashboard: Option<Res<Dashboard>>,
    theme: Res<Theme>,
    mut markers: Query<(&FirmMarker, &Hovered, &mut BorderColor, &mut Node)>,
) {
    if let Some(dashboard) = dashboard {
        markers
            .iter_mut()
            .for_each(|(marker, hovered, mut border, mut node)| {
                let (color, width) = match (marker.identity == dashboard.selected, hovered.get()) {
                    (true, _) | (_, true) => (Tone::Text, 2.0),
                    _ => (Tone::Panel, 1.0),
                };
                border.set_if_neq(BorderColor::all(theme.color(color)));
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

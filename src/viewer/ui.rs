use super::{
    Dashboard, DashboardPage, Speed, atlas,
    cabinet::{self, Register},
    companies::{self, Direction, Ranking},
    feedback,
    model::{Chart, Menu, Period, SeriesSelection},
    railway,
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
const DISPLAY_FONT: Handle<Font> = uuid_handle!("26be5744-0ce8-427a-a4f0-e6459a4bc523");
pub(super) struct UiPlugin;
impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<UiAction>()
            .init_resource::<atlas::Layer>()
            .add_systems(Update, atlas::select.before(super::controls))
            .add_observer(activate)
            .add_systems(PostUpdate, style_buttons)
            .add_systems(Startup, setup)
            .add_systems(
                Update,
                (
                    atlas::fit,
                    railway::indicate,
                    style_theme_switch,
                    cabinet::illustrate,
                    atlas::draw,
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
    Atlas(atlas::Layer),
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
            Self::Atlas(atlas::Layer::Sites) => {
                "Open-site concentration, from sparse to dense".into()
            }
            Self::Atlas(atlas::Layer::Shortage) => "Unmet bread demand by survey square".into(),
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
pub(super) struct ChartTooltip;

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
pub(super) struct ChartTab(pub Chart);

fn activate(event: On<Activate>, buttons: Query<&UiAction>, mut actions: MessageWriter<UiAction>) {
    if let Ok(action) = buttons.get(event.entity) {
        actions.write(*action);
    }
}

pub(super) fn text(value: impl Into<String>, size: f32, tone: Tone) -> impl Bundle {
    lettering(value, size, tone, BODY_FONT)
}

pub(super) fn heading(value: impl Into<String>, size: f32, tone: Tone) -> impl Bundle {
    lettering(value, size, tone, DISPLAY_FONT)
}

fn lettering(value: impl Into<String>, size: f32, tone: Tone, font: Handle<Font>) -> impl Bundle {
    (
        Text::new(value),
        TextFont {
            font: font.into(),
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
    button_with(parent, action, text(caption, 12.0, Tone::Text), |_| {})
}

fn button_with(
    parent: &mut ChildSpawnerCommands,
    action: UiAction,
    caption: impl Bundle,
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
            border_radius: BorderRadius::all(px(2)),
            border: UiRect {
                bottom: px(2),
                ..UiRect::all(px(1))
            },
            align_items: AlignItems::Center,
            justify_content: JustifyContent::Center,
            flex_shrink: 0.0,
            ..default()
        },
        cabinet::bevel(),
        BackgroundColor(Theme::Light.color(Tone::Panel)),
        BorderColor::all(Theme::Light.color(Tone::Rule)),
    ));
    let entity = button.id();
    button.with_children(|parent| {
        content(parent);
        parent.spawn(caption);
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
        text(name, 12.0, Tone::Text),
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

fn nav_button(parent: &mut ChildSpawnerCommands, register: Register) {
    let entity = button_with(
        parent,
        register.action(),
        heading(register.label(), 15.0, Tone::OnPrimary),
        |tab| register.seal(tab),
    );
    parent.commands().entity(entity).insert((
        NavButton,
        Node {
            height: percent(100),
            flex_grow: 1.0,
            flex_basis: px(0),
            min_width: px(0),
            column_gap: px(7),
            align_items: AlignItems::Center,
            justify_content: JustifyContent::Center,
            border: UiRect {
                left: px(1),
                bottom: px(3),
                ..default()
            },
            ..default()
        },
    ));
}

fn label(parent: &mut ChildSpawnerCommands, kind: Label, size: f32, tone: Tone) {
    parent.spawn((kind, text("", size, tone)));
}

pub(super) fn card(gap: f32) -> Node {
    Node {
        flex_shrink: 0.0,
        padding: UiRect::all(px(13)),
        border: UiRect::all(px(1)),
        ..column(gap)
    }
}

fn setup(
    mut commands: Commands,
    mut fonts: ResMut<Assets<Font>>,
    mut images: ResMut<Assets<Image>>,
) {
    fonts
        .insert(
            DISPLAY_FONT.id(),
            Font::from_bytes(include_bytes!("../../assets/fonts/LibreBaskerville.ttf").to_vec()),
        )
        .expect("display font handle");
    fonts
        .insert(
            BODY_FONT.id(),
            Font::from_bytes(include_bytes!("../../assets/fonts/Lato-Regular.ttf").to_vec()),
        )
        .expect("body font handle");
    cabinet::load(&mut images);
    commands
        .spawn((
            Node {
                width: percent(100),
                height: percent(100),
                display: bevy::ui::Display::Grid,
                grid_template_columns: vec![GridTrack::flex(1.0)],
                grid_template_rows: vec![
                    GridTrack::px(74.0),
                    GridTrack::px(46.0),
                    GridTrack::flex(1.0),
                    GridTrack::px(36.0),
                ],
                ..default()
            },
            Tone::Canvas.surface(),
        ))
        .with_children(|root| {
            navigation(root);
            run_toolbar(root);
            root.spawn(Node {
                min_width: px(0),
                min_height: px(0),
                overflow: Overflow::clip(),
                ..default()
            })
            .with_children(|workspace| {
                context(workspace);
                report(workspace);
                companies::panel(workspace);
            });
            footer(root);
        });
    commands.spawn((
        ChartTooltip,
        GlobalZIndex(1),
        Visibility::Hidden,
        Pickable::IGNORE,
        Node {
            position_type: PositionType::Absolute,
            padding: UiRect::axes(px(10), px(8)),
            border: UiRect::all(px(1)),
            border_radius: BorderRadius::all(px(4)),
            ..column(5.0)
        },
        Tone::Panel.surface(),
        Tone::Brass.outline(),
    ));
}

fn navigation(parent: &mut ChildSpawnerCommands) {
    parent
        .spawn((
            Node {
                border: UiRect::vertical(px(2)),
                align_items: AlignItems::Stretch,
                ..default()
            },
            Tone::Nav.surface(),
            Tone::Brass.outline(),
        ))
        .with_children(|nav| {
            nav.spawn((
                Node {
                    width: percent(23),
                    max_width: px(355),
                    flex_shrink: 0.0,
                    padding: UiRect::horizontal(px(18)),
                    justify_content: JustifyContent::Center,
                    overflow: Overflow::clip(),
                    ..column(2.0)
                },
                Tone::Nav.surface(),
                cabinet::bevel(),
            ))
            .with_children(|brand| {
                cabinet::imprint(brand);
                brand.spawn(heading("Spatial Economy", 24.0, Tone::OnPrimary));
                label(brand, Label::Date, 12.0, Tone::NavAccent);
            });
            for register in Menu::ALL
                .into_iter()
                .map(Register::Section)
                .chain([Register::Companies])
            {
                nav_button(nav, register);
            }
        });
}

fn run_toolbar(parent: &mut ChildSpawnerCommands) {
    parent
        .spawn((
            Node {
                height: px(46),
                flex_shrink: 0.0,
                padding: UiRect::axes(px(16), px(3)),
                border: UiRect::bottom(px(1)),
                justify_content: JustifyContent::SpaceBetween,
                ..row(16.0)
            },
            Tone::Nav.surface(),
            Tone::Brass.outline(),
        ))
        .with_children(|top| {
            top.spawn(row(10.0)).with_children(|context| {
                railway::locomotive(context);
                context
                    .spawn((
                        Node {
                            padding: UiRect::axes(px(8), px(4)),
                            border: UiRect::all(px(1)),
                            ..row(0.0)
                        },
                        Tone::Raised.surface(),
                        Tone::Brass.outline(),
                    ))
                    .with_children(|status| label(status, Label::Status, 12.0, Tone::Accent));
                label(context, Label::Seed, 12.0, Tone::NavMuted);
                button(context, UiAction::Compare, "Compare seed");
                button(context, UiAction::Reset, "Reset run");
            });
            top.spawn(row(5.0)).with_children(|controls| {
                let run = button_with(
                    controls,
                    UiAction::ToggleRun,
                    (Label::ToggleRun, text("Resume", 12.0, Tone::OnPrimary)),
                    |_| {},
                );
                controls.commands().entity(run).insert(PrimaryButton);
                button(controls, UiAction::Step, "+1 day");
                railway::governor(controls);
                controls.spawn(text("Speed", 12.0, Tone::NavMuted));
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
                height: px(36),
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
                        border_radius: BorderRadius::all(px(2)),
                        ..row(0.0)
                    },
                    BackgroundColor::default(),
                    BorderColor::default(),
                    Pickable::IGNORE,
                ))
                .with_children(|track| {
                    track.spawn((
                        Node {
                            width: px(11),
                            height: px(13),
                            border: UiRect::horizontal(px(2)),
                            ..default()
                        },
                        Tone::NavAccent.surface(),
                        Tone::Brass.outline(),
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
        padding: UiRect::all(px(12)),
        overflow: Overflow::scroll_y(),
        ..column(10.0)
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
            cabinet::masthead(report, |title| {
                title.spawn((Label::Title, heading("", 36.0, Tone::OnPrimary)));
                label(title, Label::Subtitle, 14.0, Tone::OnPrimary);
            });
            report
                .spawn(Node {
                    justify_content: JustifyContent::SpaceBetween,
                    flex_wrap: FlexWrap::Wrap,
                    row_gap: px(5),
                    flex_shrink: 0.0,
                    ..row(8.0)
                })
                .with_children(|header| {
                    header.spawn(column(2.0)).with_children(|status| {
                        label(status, Label::Comparison, 12.0, Tone::Muted);
                        label(status, Label::Error, 13.0, Tone::Negative);
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
                .spawn((WorkspaceSection::Grid, column(14.0)))
                .with_children(|metrics| {
                    metric_values(metrics);
                    metrics
                        .spawn(Node {
                            display: bevy::ui::Display::Grid,
                            grid_template_columns: vec![GridTrack::flex(1.0), GridTrack::flex(1.0)],
                            row_gap: px(0),
                            column_gap: px(0),
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
                            details.spawn(heading("Snapshot details", 15.0, Tone::Text));
                            label(details, Label::Summary, 13.0, Tone::Muted);
                        });
                });
            chart_detail(report);
        });
}

fn metric_values(parent: &mut ChildSpawnerCommands) {
    parent
        .spawn((
            Node {
                display: bevy::ui::Display::Grid,
                grid_template_columns: vec![GridTrack::flex(1.0); 4],
                column_gap: px(1),
                border: UiRect::vertical(px(2)),
                ..default()
            },
            Tone::Rule.surface(),
            Tone::Brass.outline(),
        ))
        .with_children(|cards| {
            for index in 0..4 {
                cards
                    .spawn((
                        Node {
                            border: UiRect::ZERO,
                            ..card(4.0)
                        },
                        Tone::Panel.surface(),
                    ))
                    .with_children(|metric| {
                        label(metric, Label::MetricName(index), 12.0, Tone::Muted);
                        metric.spawn((Label::MetricValue(index), heading("", 25.0, Tone::Text)));
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
            .with_children(|title| {
                title.spawn(heading(chart.label(), 16.0, Tone::Text));
                title.spawn(text("Open ›", 11.0, Tone::Accent));
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
                    card.spawn((Label::ChartTitle, heading("", 21.0, Tone::Text)));
                    label(card, Label::ChartDescription, 12.0, Tone::Muted);
                    card.spawn(Node {
                        height: px(242),
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
                                readout.spawn(heading("Observation", 14.0, Tone::Text));
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
                width: percent(32),
                max_width: px(505),
                min_width: px(360),
                min_height: px(0),
                flex_shrink: 0.0,
                padding: UiRect::all(px(10)),
                border: UiRect::right(px(3)),
                display: bevy::ui::Display::Grid,
                grid_template_columns: vec![GridTrack::flex(1.0)],
                grid_template_rows: vec![GridTrack::flex(1.0), GridTrack::percent(29.0)],
                row_gap: px(10),
                ..default()
            },
            Tone::Rule.outline(),
            Tone::Canvas.surface(),
        ))
        .with_children(|rail| {
            map(rail);
            inspector(rail);
        });
}

fn ruled_heading(parent: &mut ChildSpawnerCommands, caption: &str) {
    parent.spawn(row(8.0)).with_children(|title| {
        title.spawn(heading(caption, 16.0, Tone::Text));
        title.spawn((
            Node {
                height: px(4),
                flex_grow: 1.0,
                border: UiRect::vertical(px(1)),
                ..default()
            },
            Tone::Brass.outline(),
            Pickable::IGNORE,
        ));
    });
}

fn map(parent: &mut ChildSpawnerCommands) {
    parent
        .spawn((
            Node {
                min_height: px(0),
                ..card(7.0)
            },
            Tone::Panel.surface(),
            Tone::Brass.outline(),
        ))
        .with_children(|sheet| {
            sheet.spawn(row(8.0)).with_children(|title| {
                atlas::Symbol::Compass.stamp(title, 50.0);
                title
                    .spawn(Node {
                        flex_grow: 1.0,
                        ..column(4.0)
                    })
                    .with_children(|heading| {
                        ruled_heading(heading, "Regional survey");
                        heading.spawn(row(4.0)).with_children(|layers| {
                            for layer in [atlas::Layer::Sites, atlas::Layer::Shortage] {
                                button(layers, UiAction::Atlas(layer), layer.label());
                            }
                        });
                    });
            });
            atlas::sheet(sheet);
            atlas::key(sheet);
            label(sheet, Label::MapSummary, 12.0, Tone::Muted);
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
            ruled_heading(panel, "Inspection docket");
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
    layer: Res<atlas::Layer>,
    mut buttons: Query<ButtonAppearance, Without<FirmMarker>>,
) {
    if let Some(dashboard) = dashboard {
        let display = dashboard.data.lock().expect("dashboard lock");
        buttons.iter_mut().for_each(|mut button| {
            let selected = match button.action {
                UiAction::Atlas(selection) => *selection == *layer,
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
                (true, _, _, _, _, true) => (Tone::NavHover, Tone::Brass),
                (true, _, _, _, _, _) => (Tone::Nav, Tone::Brass),
                (_, true, _, _, _, true) => (Tone::NavSelected, Tone::NavSelected),
                (_, true, ..) => (Tone::Primary, Tone::Brass),
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
    mut markers: Query<(
        &FirmMarker,
        &Hovered,
        &mut BorderColor,
        &mut BackgroundColor,
    )>,
) {
    if let Some(dashboard) = dashboard {
        markers
            .iter_mut()
            .for_each(|(marker, hovered, mut border, mut background)| {
                let (rim, paper) = match (marker.identity == dashboard.selected, hovered.get()) {
                    (true, _) | (_, true) => {
                        (theme.color(Tone::Primary), theme.color(Tone::NavAccent))
                    }
                    _ => (Color::NONE, Color::NONE),
                };
                border.set_if_neq(BorderColor::all(rim));
                background.set_if_neq(BackgroundColor(paper));
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

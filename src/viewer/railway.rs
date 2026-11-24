use super::{Dashboard, DisplayState, Speed, theme::Tone};
use crate::runtime::RuntimeState;
use bevy::{picking::Pickable, prelude::*};

#[derive(Component)]
pub(super) struct Steam;

#[derive(Component)]
pub(super) struct GovernorHand;

pub(super) fn governor(parent: &mut ChildSpawnerCommands) {
    parent
        .spawn((
            Node {
                width: px(38),
                height: px(38),
                margin: UiRect::left(px(8)),
                border: UiRect::all(px(2)),
                border_radius: BorderRadius::MAX,
                flex_shrink: 0.0,
                ..default()
            },
            Tone::Panel.surface(),
            Tone::Brass.outline(),
            Pickable::IGNORE,
        ))
        .with_children(|dial| {
            for mark in -3..=3 {
                let angle = Rot2::degrees(mark as f32 * 24.0);
                let center = Vec2::splat(17.0) + angle * Vec2::new(0.0, -13.0);
                dial.spawn((
                    plate(center.x - 0.75, center.y - 1.5, 1.5, 3.0),
                    UiTransform::from_rotation(angle),
                    Tone::Muted.surface(),
                    Pickable::IGNORE,
                ));
            }
            dial.spawn((GovernorHand, plate(6.0, 6.0, 22.0, 22.0), Pickable::IGNORE))
                .with_children(|hand| solid(hand, plate(10.0, 0.0, 2.0, 13.0), Tone::Negative));
            solid(dial, circle(14.5, 14.5, 5.0), Tone::Brass);
        });
}

pub(super) fn indicate(
    dashboard: Res<Dashboard>,
    mut steam: Query<&mut Visibility, With<Steam>>,
    mut hands: Query<&mut UiTransform, With<GovernorHand>>,
) {
    let display = dashboard.data.lock().expect("dashboard lock");
    let exhaust = match display.status {
        DisplayState::Active(RuntimeState::Running | RuntimeState::Advancing(_)) => {
            Visibility::Inherited
        }
        _ => Visibility::Hidden,
    };
    for mut plume in &mut steam {
        plume.set_if_neq(exhaust);
    }
    let angle = match display.speed {
        Speed::Slow => -72.0,
        Speed::Normal => 0.0,
        Speed::Fast => 72.0,
    };
    for mut hand in &mut hands {
        hand.set_if_neq(UiTransform::from_rotation(Rot2::degrees(angle)));
    }
}

pub(super) fn locomotive(parent: &mut ChildSpawnerCommands) {
    parent
        .spawn((
            Node {
                width: px(65),
                height: px(30),
                flex_shrink: 0.0,
                ..default()
            },
            Pickable::IGNORE,
        ))
        .with_children(|engine| {
            boiler(engine);
            solid(engine, plate(7.0, 19.0, 50.0, 3.0), Tone::NavAccent);
            wheel(engine, 12.0, 22.0, 7.0);
            wheel(engine, 25.0, 20.0, 10.0);
            wheel(engine, 42.0, 20.0, 10.0);
            solid(engine, plate(30.0, 24.0, 17.0, 1.5), Tone::OnPrimary);
            engine.spawn((
                plate(54.0, 22.0, 8.0, 1.5),
                UiTransform::from_rotation(Rot2::radians(0.65)),
                Tone::NavAccent.surface(),
                Pickable::IGNORE,
            ));
            engine
                .spawn((
                    Steam,
                    Visibility::Hidden,
                    plate(47.0, 0.0, 18.0, 7.0),
                    Pickable::IGNORE,
                ))
                .with_children(|steam| {
                    solid(steam, circle(0.0, 2.0, 4.0), Tone::OnPrimary);
                    solid(steam, circle(5.0, 0.0, 5.0), Tone::OnPrimary);
                    solid(steam, circle(11.0, 0.0, 4.0), Tone::NavMuted);
                });
        });
}

fn boiler(parent: &mut ChildSpawnerCommands) {
    solid(parent, plate(8.0, 6.0, 20.0, 2.0), Tone::OnPrimary);
    solid(parent, plate(11.0, 8.0, 15.0, 12.0), Tone::OnPrimary);
    solid(parent, plate(14.0, 10.0, 8.0, 6.0), Tone::Nav);
    solid(
        parent,
        Node {
            border_radius: BorderRadius::all(px(3)),
            ..plate(24.0, 12.0, 32.0, 8.0)
        },
        Tone::OnPrimary,
    );
    solid(parent, plate(30.0, 12.0, 1.5, 8.0), Tone::NavAccent);
    solid(parent, plate(50.0, 12.0, 1.5, 8.0), Tone::NavAccent);
    solid(parent, plate(35.0, 9.0, 5.0, 3.0), Tone::OnPrimary);
    solid(parent, plate(46.0, 7.0, 4.0, 6.0), Tone::OnPrimary);
    solid(parent, plate(44.0, 5.0, 8.0, 2.0), Tone::OnPrimary);
}

fn wheel(parent: &mut ChildSpawnerCommands, left: f32, top: f32, size: f32) {
    parent
        .spawn((
            Node {
                border: UiRect::all(px(1.5)),
                align_items: AlignItems::Center,
                justify_content: JustifyContent::Center,
                ..circle(left, top, size)
            },
            Tone::Nav.surface(),
            Tone::NavAccent.outline(),
            Pickable::IGNORE,
        ))
        .with_children(|wheel| {
            wheel.spawn((
                Node {
                    width: px(2),
                    height: px(2),
                    ..default()
                },
                Tone::OnPrimary.surface(),
                Pickable::IGNORE,
            ));
        });
}

fn circle(left: f32, top: f32, size: f32) -> Node {
    Node {
        border_radius: BorderRadius::MAX,
        ..plate(left, top, size, size)
    }
}

fn plate(left: f32, top: f32, width: f32, height: f32) -> Node {
    Node {
        position_type: PositionType::Absolute,
        left: px(left),
        top: px(top),
        width: px(width),
        height: px(height),
        ..default()
    }
}

fn solid(parent: &mut ChildSpawnerCommands, node: Node, tone: Tone) {
    parent.spawn((node, tone.surface(), Pickable::IGNORE));
}

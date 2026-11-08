use bevy::{
    prelude::*,
    winit::{UpdateMode, WinitSettings},
};
use std::time::Duration;

pub(super) struct PacingPlugin;
impl Plugin for PacingPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(WinitSettings {
            focused_mode: UpdateMode::reactive_low_power(Duration::from_millis(250)),
            unfocused_mode: UpdateMode::reactive_low_power(Duration::from_millis(500)),
        })
        .init_resource::<Cadence>()
        .add_systems(Update, update.in_set(PacingSystems));
    }
}

#[derive(SystemSet, Debug, Hash, PartialEq, Eq, Clone)]
pub(super) struct PacingSystems;

#[derive(Resource, Default, PartialEq, Eq)]
pub(super) enum Cadence {
    #[default]
    Idle,
    Animating,
}

fn update(
    cadence: Res<Cadence>,
    verification: Option<Res<super::Verification>>,
    mut settings: ResMut<WinitSettings>,
) {
    let interval = match *cadence {
        Cadence::Idle => Duration::from_millis(250),
        Cadence::Animating => Duration::from_secs_f64(1.0 / 60.0),
    };
    *settings = match verification {
        Some(_) => WinitSettings {
            focused_mode: UpdateMode::Continuous,
            unfocused_mode: UpdateMode::Continuous,
        },
        None => WinitSettings {
            focused_mode: UpdateMode::reactive_low_power(interval),
            unfocused_mode: UpdateMode::reactive_low_power(Duration::from_millis(500)),
        },
    };
}

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
        });
    }
}

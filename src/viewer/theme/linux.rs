use bevy::{
    prelude::*,
    window::{PrimaryWindow, WindowTheme, WindowThemeChanged},
    winit::{EventLoopProxyWrapper, WinitUserEvent},
};
use crossbeam_channel::{Receiver, unbounded};
use dark_light::Mode;

#[derive(Resource)]
pub(super) struct PortalAppearance(Receiver<WindowTheme>);

pub(super) fn watch(mut commands: Commands, event_loop: Res<EventLoopProxyWrapper>) {
    let (sender, receiver) = unbounded();
    commands.insert_resource(PortalAppearance(receiver));
    let wake = (**event_loop).clone();
    std::thread::spawn(move || {
        let publish = |mode| {
            let theme = match mode {
                Mode::Dark => WindowTheme::Dark,
                Mode::Light | Mode::Unspecified => WindowTheme::Light,
            };
            sender.send(theme).inspect(|_| {
                let _ = wake.send_event(WinitUserEvent::WakeUp);
            })
        };
        let watcher = dark_light::subscribe();
        if let Ok(mode) = dark_light::detect() {
            let _ = publish(mode);
        }
        if let Ok(watcher) = watcher {
            let _ = watcher.iter().try_for_each(publish);
        }
    });
}

pub(super) fn relay(
    portal: Option<Res<PortalAppearance>>,
    windows: Query<Entity, With<PrimaryWindow>>,
    mut changes: MessageWriter<WindowThemeChanged>,
) {
    let latest = portal.and_then(|portal| portal.0.try_iter().last());
    if let (Some(theme), Ok(window)) = (latest, windows.single()) {
        changes.write(WindowThemeChanged { window, theme });
    }
}

use super::Dashboard;
use bevy::{
    audio::Volume,
    prelude::*,
    sprite::Anchor,
    winit::{UpdateMode, WinitSettings},
};
use std::time::Duration;

pub(super) struct FeedbackPlugin;
#[derive(SystemSet, Debug, Hash, PartialEq, Eq, Clone)]
pub(super) struct FeedbackSystems;

impl Plugin for FeedbackPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Feedback>()
            .add_systems(Startup, setup)
            .add_systems(
                Update,
                (receive, animate)
                    .chain()
                    .in_set(FeedbackSystems)
                    .after(super::controls),
            );
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Cue {
    Ready,
    Paused,
    Running,
    Stepped,
    Reset,
    Compared,
    Error,
    Preferences,
}
impl Cue {
    fn label(self) -> &'static str {
        match self {
            Self::Ready => "Ready to simulate",
            Self::Paused => "Simulation paused",
            Self::Running => "Simulation running",
            Self::Stepped => "Day advanced and audited",
            Self::Reset => "Simulation reset",
            Self::Compared => "Seed comparison ready",
            Self::Error => "Action failed: see the error above",
            Self::Preferences => "Feedback preferences updated",
        }
    }

    fn color(self) -> Color {
        match self {
            Self::Error => Color::srgb(0.95, 0.35, 0.30),
            Self::Paused | Self::Preferences => Color::srgb(0.95, 0.73, 0.25),
            _ => Color::srgb(0.45, 0.90, 0.65),
        }
    }
}

#[derive(Default)]
enum SoundMode {
    #[default]
    On,
    Muted,
}
impl SoundMode {
    fn toggle(&mut self) {
        *self = match self {
            Self::On => Self::Muted,
            Self::Muted => Self::On,
        };
    }

    fn label(&self) -> &'static str {
        match self {
            Self::On => "on",
            Self::Muted => "muted",
        }
    }
}

#[derive(Default)]
enum FlashMode {
    #[default]
    Subtle,
    Reduced,
}
impl FlashMode {
    fn toggle(&mut self) {
        *self = match self {
            Self::Subtle => Self::Reduced,
            Self::Reduced => Self::Subtle,
        };
    }

    fn label(&self) -> &'static str {
        match self {
            Self::Subtle => "subtle",
            Self::Reduced => "reduced",
        }
    }
}

#[derive(Default)]
enum Notice {
    #[default]
    Quiet,
    Showing {
        cue: Cue,
        elapsed: Duration,
    },
}

#[derive(Resource, Default)]
pub(super) struct Feedback {
    sound: SoundMode,
    flashes: FlashMode,
    notice: Notice,
}
impl Feedback {
    pub(super) fn current(&self) -> Option<Cue> {
        match self.notice {
            Notice::Quiet => None,
            Notice::Showing { cue, .. } => Some(cue),
        }
    }

    fn tick(&mut self, delta: Duration) {
        self.notice = match self.notice {
            Notice::Showing { cue, elapsed } => match (cue, elapsed + delta) {
                (Cue::Error, elapsed) => Notice::Showing {
                    cue,
                    elapsed: elapsed.min(Duration::from_secs(3)),
                },
                (_, elapsed) if elapsed < Duration::from_secs(3) => {
                    Notice::Showing { cue, elapsed }
                }
                _ => Notice::Quiet,
            },
            Notice::Quiet => Notice::Quiet,
        };
    }

    fn flash(&self) -> f32 {
        match (&self.flashes, &self.notice) {
            (FlashMode::Subtle, Notice::Showing { elapsed, .. }) => {
                (1.0 - elapsed.as_secs_f32() / 0.24).clamp(0.0, 1.0)
            }
            _ => 0.0,
        }
    }
}

#[derive(Resource)]
struct Sounds {
    confirm: Handle<AudioSource>,
    pause: Handle<AudioSource>,
    error: Handle<AudioSource>,
}

#[derive(Component)]
struct Banner;

#[derive(Component)]
struct Label;

fn setup(mut commands: Commands, mut audio: ResMut<Assets<AudioSource>>) {
    let source = |bytes: &'static [u8]| AudioSource {
        bytes: bytes.into(),
    };
    commands.insert_resource(Sounds {
        confirm: audio.add(source(include_bytes!("../../assets/cues/confirm.wav"))),
        pause: audio.add(source(include_bytes!("../../assets/cues/pause.wav"))),
        error: audio.add(source(include_bytes!("../../assets/cues/error.wav"))),
    });
    commands.spawn((
        Banner,
        Sprite::from_color(Color::srgb(0.055, 0.08, 0.09), Vec2::new(1190.0, 28.0)),
        Transform::from_xyz(-5.0, -397.0, 3.0),
    ));
    commands.spawn((
        Label,
        Text2d::default(),
        TextFont {
            font_size: FontSize::Px(14.0),
            ..Default::default()
        },
        TextColor(Color::srgb(0.85, 0.92, 0.95)),
        Anchor::TOP_LEFT,
        Transform::from_xyz(-588.0, -386.0, 4.0),
    ));
}

fn receive(
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time<Real>>,
    dashboard: Option<Res<Dashboard>>,
    sounds: Res<Sounds>,
    mut feedback: ResMut<Feedback>,
    mut commands: Commands,
) {
    feedback.tick(time.delta());
    let preference = match keys.just_pressed(KeyCode::KeyM) {
        true => {
            feedback.sound.toggle();
            Some(Cue::Preferences)
        }
        false => None,
    };
    let preference = match keys.just_pressed(KeyCode::KeyV) {
        true => {
            feedback.flashes.toggle();
            Some(Cue::Preferences)
        }
        false => preference,
    };
    let queued = dashboard.and_then(|dashboard| {
        dashboard
            .data
            .lock()
            .expect("dashboard lock")
            .cues
            .drain(..)
            .next_back()
    });
    let sound = queued.or(preference).and_then(|cue| {
        feedback.notice = Notice::Showing {
            cue,
            elapsed: Duration::ZERO,
        };
        match (&feedback.sound, cue) {
            (SoundMode::Muted, _) | (_, Cue::Ready) => None,
            (_, Cue::Error) => Some(&sounds.error),
            (_, Cue::Paused | Cue::Reset) => Some(&sounds.pause),
            _ => Some(&sounds.confirm),
        }
    });
    if let Some(sound) = sound {
        commands.spawn((
            AudioPlayer::new(sound.clone()),
            PlaybackSettings::DESPAWN.with_volume(Volume::Linear(0.18)),
        ));
    }
}

fn animate(
    feedback: Res<Feedback>,
    mut pacing: ResMut<WinitSettings>,
    mut labels: Query<&mut Text2d, With<Label>>,
    mut banners: Query<&mut Sprite, With<Banner>>,
) {
    let cue = feedback.current().unwrap_or(Cue::Ready);
    let text = format!(
        "{}  |  M sound: {}  |  V flashes: {}",
        cue.label(),
        feedback.sound.label(),
        feedback.flashes.label(),
    );
    labels
        .iter_mut()
        .for_each(|mut label| match label.0 == text {
            true => (),
            false => label.0.clone_from(&text),
        });
    let flash = feedback.flash();
    let color = Color::srgb(0.055, 0.08, 0.09).mix(&cue.color(), 0.08 + flash * 0.20);
    banners
        .iter_mut()
        .for_each(|mut banner| match banner.color == color {
            true => (),
            false => banner.color = color,
        });
    let interval = match flash > 0.0 {
        true => Duration::from_secs_f64(1.0 / 60.0),
        false => Duration::from_millis(250),
    };
    pacing.focused_mode = UpdateMode::reactive_low_power(interval);
    pacing.unfocused_mode = UpdateMode::reactive_low_power(Duration::from_millis(500));
}

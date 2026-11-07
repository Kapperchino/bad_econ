use super::Dashboard;
use super::ui::UiAction;
use bevy::{audio::Volume, prelude::*};
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
                    .after(super::controls)
                    .before(super::pacing::PacingSystems),
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
pub(super) struct Banner;

#[derive(Component)]
pub(super) struct Label;

fn setup(mut commands: Commands, mut audio: ResMut<Assets<AudioSource>>) {
    let source = |bytes: &'static [u8]| AudioSource {
        bytes: bytes.into(),
    };
    commands.insert_resource(Sounds {
        confirm: audio.add(source(include_bytes!("../../assets/cues/confirm.wav"))),
        pause: audio.add(source(include_bytes!("../../assets/cues/pause.wav"))),
        error: audio.add(source(include_bytes!("../../assets/cues/error.wav"))),
    });
}

fn receive(
    keys: Res<ButtonInput<KeyCode>>,
    mut actions: MessageReader<UiAction>,
    time: Res<Time<Real>>,
    dashboard: Option<Res<Dashboard>>,
    sounds: Res<Sounds>,
    mut feedback: ResMut<Feedback>,
    mut commands: Commands,
) {
    feedback.tick(time.delta());
    let pending = actions.read().copied().collect::<Vec<_>>();
    let preference = match keys.just_pressed(KeyCode::KeyM) || pending.contains(&UiAction::Sound) {
        true => {
            feedback.sound.toggle();
            Some(Cue::Preferences)
        }
        false => None,
    };
    let preference = match keys.just_pressed(KeyCode::KeyV) || pending.contains(&UiAction::Motion) {
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
    mut cadence: ResMut<super::pacing::Cadence>,
    actions: Query<(&UiAction, &bevy::picking::hover::Hovered)>,
    mut labels: Query<&mut Text, With<Label>>,
    mut banners: Query<&mut BackgroundColor, With<Banner>>,
) {
    let cue = feedback.current().unwrap_or(Cue::Ready);
    let status = format!(
        "{}  |  M sound: {}  |  V flashes: {}",
        cue.label(),
        feedback.sound.label(),
        feedback.flashes.label(),
    );
    let text = match &feedback.notice {
        Notice::Showing {
            cue: Cue::Error, ..
        } => status,
        Notice::Showing { elapsed, .. } if *elapsed < Duration::from_secs(1) => status,
        _ => actions
            .iter()
            .find(|(_, hovered)| hovered.get())
            .map(|(action, _)| action.hint())
            .unwrap_or(status),
    };
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
        .for_each(|mut banner| match banner.0 == color {
            true => (),
            false => banner.0 = color,
        });
    cadence.set_if_neq(match flash > 0.0 {
        true => super::pacing::Cadence::Animating,
        false => super::pacing::Cadence::Idle,
    });
}

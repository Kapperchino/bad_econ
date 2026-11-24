use super::Chart;
use bevy::{
    prelude::*,
    window::{PrimaryWindow, WindowTheme, WindowThemeChanged},
};

#[cfg(target_os = "linux")]
mod linux;

pub(super) struct ThemePlugin;

impl Plugin for ThemePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Theme>()
            .init_resource::<AppearanceSettings>()
            .add_systems(PreUpdate, follow_system)
            .add_systems(Update, resolve_appearance.before(super::controls))
            .add_systems(PostUpdate, apply_colors);
        #[cfg(target_os = "linux")]
        app.add_systems(
            Startup,
            linux::watch.run_if(not(resource_exists::<super::verification::Verification>)),
        )
        .add_systems(PreUpdate, linux::relay.before(follow_system));
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum Appearance {
    #[default]
    System,
    Override(Theme),
}

#[derive(Resource, Default)]
pub(super) struct AppearanceSettings {
    pub selection: Appearance,
    pub system: Theme,
}

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum Theme {
    #[default]
    Light,
    Dark,
}

impl From<WindowTheme> for Theme {
    fn from(theme: WindowTheme) -> Self {
        match theme {
            WindowTheme::Light => Self::Light,
            WindowTheme::Dark => Self::Dark,
        }
    }
}

#[derive(Clone, Copy)]
pub(super) enum Tone {
    Canvas,
    Panel,
    Raised,
    Accent,
    Primary,
    Rule,
    Brass,
    Text,
    Muted,
    Positive,
    Negative,
    Nav,
    NavHover,
    NavSelected,
    NavAccent,
    NavMuted,
    OnPrimary,
    Crosshair,
    Series(Chart, usize),
    Region(f32),
}

impl Theme {
    pub fn color(self, tone: Tone) -> Color {
        use Tone::*;
        let rgb = |light: [u8; 3], dark| {
            let [red, green, blue] = match self {
                Self::Light => light,
                Self::Dark => dark,
            };
            Color::srgb_u8(red, green, blue)
        };
        match tone {
            Canvas => rgb([214, 201, 173], [25, 26, 31]),
            Panel => rgb([248, 239, 216], [39, 44, 49]),
            Raised => rgb([222, 202, 162], [80, 66, 52]),
            Accent => rgb([119, 44, 44], [244, 186, 123]),
            Primary => rgb([112, 41, 45], [115, 46, 48]),
            Rule => rgb([174, 150, 109], [99, 87, 70]),
            Brass => rgb([144, 103, 52], [171, 137, 83]),
            Text => rgb([46, 41, 31], [243, 232, 204]),
            Muted => rgb([96, 85, 65], [194, 185, 159]),
            Positive => rgb([39, 99, 76], [137, 198, 162]),
            Negative => rgb([153, 44, 51], [242, 143, 131]),
            Nav => rgb([34, 48, 60], [27, 37, 48]),
            NavHover => rgb([63, 78, 87], [50, 64, 77]),
            NavSelected => rgb([128, 47, 51], [110, 43, 49]),
            NavAccent => Color::srgb_u8(224, 191, 128),
            NavMuted => Color::srgb_u8(185, 185, 155),
            OnPrimary => Color::srgb_u8(255, 245, 218),
            Crosshair => self.color(Accent).with_alpha(0.55),
            Series(chart, series) => self.series(chart, series),
            Region(heat) => {
                rgb([216, 209, 176], [61, 72, 59]).mix(&rgb([155, 57, 36], [224, 134, 93]), heat)
            }
        }
    }

    fn series(self, chart: Chart, series: usize) -> Color {
        let (blue, ochre, plum, other_lightness) = match self {
            Self::Light => ([47, 92, 126], [143, 100, 19], [113, 69, 104], 0.32),
            Self::Dark => ([136, 185, 214], [220, 192, 104], [202, 160, 190], 0.70),
        };
        let rgb = |[red, green, blue]: [u8; 3]| Color::srgb_u8(red, green, blue);
        match (chart, series) {
            (Chart::Deaths, 0) | (Chart::Demography, 1) => self.color(Tone::Muted),
            (Chart::Deaths, 1) | (Chart::Demography, 2) => rgb(ochre),
            (Chart::Deaths, 2)
            | (Chart::Demography, 3)
            | (Chart::Employment | Chart::Credit, 1) => self.color(Tone::Negative),
            (Chart::Deaths, 3) | (Chart::Demography, 4) | (Chart::Employment, 2) | (_, 1) => {
                rgb(blue)
            }
            (Chart::Employment, 0) | (_, 2) => self.color(Tone::Positive),
            (_, 0) => self.color(Tone::Accent),
            (_, 3) => rgb(ochre),
            (_, 4) => rgb(plum),
            _ => Color::hsl((series as f32 * 137.5) % 360.0, 0.42, other_lightness),
        }
    }
}

#[derive(Component)]
pub(super) struct Surface(pub Tone);

#[derive(Component)]
pub(super) struct Outline(Tone);

#[derive(Component)]
pub(super) struct Foreground(Tone);

impl Tone {
    pub fn surface(self) -> (Surface, BackgroundColor) {
        (Surface(self), BackgroundColor(Theme::Light.color(self)))
    }

    pub fn outline(self) -> (Outline, BorderColor) {
        (Outline(self), BorderColor::all(Theme::Light.color(self)))
    }

    pub fn foreground(self) -> (Foreground, TextColor) {
        (Foreground(self), TextColor(Theme::Light.color(self)))
    }
}

fn follow_system(
    primary: Query<(Entity, &Window), With<PrimaryWindow>>,
    mut changes: MessageReader<WindowThemeChanged>,
    mut initialized: Local<Option<Entity>>,
    mut settings: ResMut<AppearanceSettings>,
) {
    let changed = changes
        .read()
        .filter(|event| primary.contains(event.window))
        .last();
    let appearance = primary.single().ok().and_then(|(entity, window)| {
        let initial = window.window_theme.filter(|_| Some(entity) != *initialized);
        changed
            .map(|event| event.theme)
            .or(initial)
            .map(|current| (entity, Theme::from(current)))
    });
    if let Some((entity, current)) = appearance {
        *initialized = Some(entity);
        settings.system = current;
    }
}

fn resolve_appearance(
    settings: Res<AppearanceSettings>,
    mut theme: ResMut<Theme>,
    mut clear: ResMut<ClearColor>,
) {
    let current = match settings.selection {
        Appearance::System => settings.system,
        Appearance::Override(theme) => theme,
    };
    if theme.set_if_neq(current) {
        clear.0 = theme.color(Tone::Canvas);
    }
}

fn apply_colors(
    theme: Res<Theme>,
    mut surfaces: Query<(Ref<Surface>, &mut BackgroundColor)>,
    mut outlines: Query<(Ref<Outline>, &mut BorderColor)>,
    mut foregrounds: Query<(Ref<Foreground>, &mut TextColor)>,
) {
    for (tone, mut color) in &mut surfaces {
        if theme.is_changed() || tone.is_changed() {
            color.set_if_neq(BackgroundColor(theme.color(tone.0)));
        }
    }
    for (tone, mut color) in &mut outlines {
        if theme.is_changed() || tone.is_changed() {
            color.set_if_neq(BorderColor::all(theme.color(tone.0)));
        }
    }
    for (tone, mut color) in &mut foregrounds {
        if theme.is_changed() || tone.is_changed() {
            color.set_if_neq(TextColor(theme.color(tone.0)));
        }
    }
}

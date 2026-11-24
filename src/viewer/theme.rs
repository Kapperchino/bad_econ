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
            Canvas => rgb([243, 244, 249], [17, 18, 26]),
            Panel => rgb([255, 255, 255], [28, 29, 40]),
            Raised => rgb([232, 227, 249], [55, 44, 79]),
            Accent => rgb([90, 57, 169], [200, 174, 255]),
            Primary => rgb([90, 57, 169], [112, 75, 187]),
            Rule => rgb([216, 216, 227], [62, 63, 83]),
            Text => rgb([29, 27, 46], [241, 239, 250]),
            Muted => rgb([90, 89, 112], [177, 178, 199]),
            Positive => rgb([19, 114, 93], [108, 214, 178]),
            Negative => rgb([173, 51, 68], [255, 137, 155]),
            Nav => rgb([33, 27, 52], [23, 19, 35]),
            NavHover => rgb([57, 46, 86], [46, 36, 68]),
            NavSelected => rgb([81, 63, 121], [65, 48, 98]),
            NavAccent => Color::srgb_u8(179, 158, 247),
            NavMuted => Color::srgb_u8(188, 179, 208),
            OnPrimary => Color::WHITE,
            Crosshair => self.color(Accent).with_alpha(0.55),
            Series(chart, series) => self.series(chart, series),
            Region(heat) => {
                rgb([228, 229, 241], [48, 49, 67]).mix(&rgb([189, 66, 82], [208, 79, 104]), heat)
            }
        }
    }

    fn series(self, chart: Chart, series: usize) -> Color {
        let (blue, amber, pink, other_lightness) = match self {
            Self::Light => ([35, 113, 181], [186, 99, 36], [177, 59, 122], 0.41),
            Self::Dark => ([103, 176, 255], [240, 174, 100], [246, 148, 211], 0.67),
        };
        let rgb = |[red, green, blue]: [u8; 3]| Color::srgb_u8(red, green, blue);
        match (chart, series) {
            (Chart::Deaths, 0) | (Chart::Demography, 1) => self.color(Tone::Muted),
            (Chart::Deaths, 1) | (Chart::Demography, 2) => rgb(amber),
            (Chart::Deaths, 2)
            | (Chart::Demography, 3)
            | (Chart::Employment | Chart::Credit, 1) => self.color(Tone::Negative),
            (Chart::Deaths, 3) | (Chart::Demography, 4) | (Chart::Employment, 2) | (_, 1) => {
                rgb(blue)
            }
            (Chart::Employment, 0) | (_, 2) => self.color(Tone::Positive),
            (_, 0) => self.color(Tone::Accent),
            (_, 3) => rgb(amber),
            (_, 4) => rgb(pink),
            _ => Color::hsl((series as f32 * 137.5) % 360.0, 0.64, other_lightness),
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

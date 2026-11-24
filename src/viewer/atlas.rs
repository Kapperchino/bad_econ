use super::{
    Dashboard, Display, cabinet,
    theme::{Theme, Tone},
    ui::{self, MapMarkers, UiAction},
};
use crate::domain::Good;
use bevy::{
    asset::uuid_handle,
    picking::Pickable,
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
};

const GRID: usize = 16;
const PIXELS: usize = 448;
const MAP: Handle<Image> = uuid_handle!("8772a5f0-3544-4115-8583-72bea0da7b79");

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Symbol {
    Land,
    Mine,
    Works,
    Property,
    Rail,
    Compass,
}

impl Symbol {
    pub fn for_site(kind: u32) -> Self {
        match Good::from_index(kind) {
            Ok(Good::Grain | Good::Flax | Good::Wood | Good::Cotton | Good::Dye) => Self::Land,
            Ok(Good::Iron | Good::Coal) => Self::Mine,
            Ok(Good::Construction | Good::Housing) => Self::Property,
            Ok(Good::Railway) => Self::Rail,
            _ => Self::Works,
        }
    }

    pub fn image(self) -> ImageNode {
        let index = self as usize;
        let min = Vec2::new((index % 3) as f32, (index / 3) as f32) * 160.0;
        ImageNode::new(cabinet::SYMBOLS)
            .with_rect(Rect::from_corners(min, min + Vec2::splat(160.0)))
            .with_mode(bevy::ui::widget::NodeImageMode::Stretch)
    }

    pub fn stamp(self, parent: &mut ChildSpawnerCommands, size: f32) {
        parent.spawn((
            self.image(),
            Node {
                width: px(size),
                height: px(size),
                flex_shrink: 0.0,
                ..default()
            },
            Pickable::IGNORE,
        ));
    }
}
#[derive(Resource, Default, Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Layer {
    #[default]
    Sites,
    Shortage,
}

impl Layer {
    pub fn label(self) -> &'static str {
        match self {
            Self::Sites => "Site density",
            Self::Shortage => "Bread shortage",
        }
    }

    fn pigment(self, theme: Theme, band: usize) -> [u8; 4] {
        match self {
            Self::Sites => {
                let daylight = [
                    [239, 226, 191],
                    [226, 218, 173],
                    [198, 202, 154],
                    [164, 185, 148],
                    [124, 161, 143],
                    [87, 136, 132],
                    [58, 108, 116],
                    [41, 78, 96],
                ];
                let [r, g, b] = daylight[band];
                let light = match theme {
                    Theme::Light => 1.0,
                    Theme::Dark => 0.85,
                };
                [
                    (r as f32 * light) as u8,
                    (g as f32 * light) as u8,
                    (b as f32 * light) as u8,
                    255,
                ]
            }
            Self::Shortage => theme
                .color(Tone::Region(band as f32 / 7.0))
                .to_srgba()
                .to_u8_array(),
        }
    }
}

#[derive(Component)]
pub(super) struct Swatch(usize);

#[derive(Clone, Copy, PartialEq)]
pub(super) struct Edition {
    layer: Layer,
    theme: Theme,
    revision: u64,
}

pub(super) fn select(mut actions: MessageReader<UiAction>, mut layer: ResMut<Layer>) {
    for action in actions.read() {
        if let UiAction::Atlas(selection) = action {
            layer.set_if_neq(*selection);
        }
    }
}

#[derive(Component)]
pub(super) struct MapFrame;

pub(super) fn fit(
    mut mount: Single<(&ChildOf, &mut Node), With<MapFrame>>,
    sizes: Query<&ComputedNode>,
) {
    let (parent, frame) = &mut *mount;
    let slot = sizes.get(parent.parent()).expect("map frame slot");
    let side = px((slot.size().min_element() * slot.inverse_scale_factor()).floor());
    if frame.width != side {
        frame.width = side;
        frame.height = side;
    }
}

pub(super) fn sheet(parent: &mut ChildSpawnerCommands) {
    parent
        .spawn(Node {
            flex_basis: px(0),
            flex_grow: 1.0,
            min_height: px(0),
            justify_content: JustifyContent::Center,
            align_items: AlignItems::Center,
            ..default()
        })
        .with_children(|slot| {
            slot.spawn((
                MapFrame,
                Node {
                    width: px(1),
                    height: px(1),
                    flex_shrink: 0.0,
                    padding: UiRect::all(px(5)),
                    border: UiRect::all(px(2)),
                    ..default()
                },
                Tone::Brass.outline(),
                Tone::Raised.surface(),
            ))
            .with_children(|mount| {
                mount
                    .spawn((
                        Node {
                            width: percent(100),
                            height: percent(100),
                            border: UiRect::all(px(1)),
                            ..default()
                        },
                        Tone::Muted.outline(),
                        ImageNode::new(MAP).with_mode(bevy::ui::widget::NodeImageMode::Stretch),
                    ))
                    .with_children(|plot| {
                        plot.spawn((
                            MapMarkers,
                            Node {
                                position_type: PositionType::Absolute,
                                width: percent(100),
                                height: percent(100),
                                ..default()
                            },
                            Pickable::IGNORE,
                        ));
                    });
            });
        });
}

pub(super) fn key(parent: &mut ChildSpawnerCommands) {
    parent
        .spawn(Node {
            flex_wrap: FlexWrap::Wrap,
            row_gap: px(2),
            ..ui::row(8.0)
        })
        .with_children(|key| {
            for (symbol, name) in [
                (Symbol::Land, "Land"),
                (Symbol::Mine, "Mines"),
                (Symbol::Works, "Works"),
                (Symbol::Property, "Property"),
                (Symbol::Rail, "Rail"),
            ] {
                key.spawn(ui::row(3.0)).with_children(|entry| {
                    symbol.stamp(entry, 16.0);
                    entry.spawn(ui::text(name, 11.0, Tone::Text));
                });
            }
        });
    parent.spawn(ui::row(6.0)).with_children(|key| {
        key.spawn(ui::text("Low", 11.0, Tone::Muted));
        key.spawn(Node {
            flex_grow: 1.0,
            ..ui::row(0.0)
        })
        .with_children(|scale| {
            for index in 0..8 {
                scale.spawn((
                    Swatch(index),
                    Node {
                        height: px(6),
                        flex_grow: 1.0,
                        ..default()
                    },
                    BackgroundColor::default(),
                ));
            }
        });
        key.spawn(ui::text("High", 11.0, Tone::Muted));
    });
}

struct Field {
    values: [f32; GRID * GRID],
    layer: Layer,
}

impl Field {
    fn from_display(display: &Display, layer: Layer) -> Self {
        let mut values = [0.0; GRID * GRID];
        match layer {
            Layer::Sites => {
                let coordinate = |p| {
                    ((u64::from(p) * GRID as u64 / u64::from(display.map_size.max(1))) as usize)
                        .min(GRID - 1)
                };
                for site in &display.firms {
                    values[coordinate(site.position.y) * GRID + coordinate(site.position.x)] += 1.0;
                }
                let bins = values;
                for (index, value) in values.iter_mut().enumerate() {
                    let x = index % GRID;
                    let y = index / GRID;
                    let mut weight_sum = 0.0;
                    let mut sum = 0.0;
                    for ny in y.saturating_sub(1)..=(y + 1).min(GRID - 1) {
                        for nx in x.saturating_sub(1)..=(x + 1).min(GRID - 1) {
                            let weight = ((2 - x.abs_diff(nx)) * (2 - y.abs_diff(ny))) as f32;
                            sum += bins[ny * GRID + nx] * weight;
                            weight_sum += weight;
                        }
                    }
                    *value = sum / weight_sum;
                }
            }
            Layer::Shortage => {
                for (value, count) in values.iter_mut().zip(&display.snapshot.regions) {
                    *value = *count as f32;
                }
            }
        }
        let peak = values.iter().copied().fold(f32::MIN_POSITIVE, f32::max);
        Self {
            values: values.map(|v| v / peak),
            layer,
        }
    }

    fn band(&self, x: usize, y: usize) -> usize {
        let axis = |pixel| {
            (((pixel as f32 + 0.5) * GRID as f32 / PIXELS as f32) - 0.5)
                .clamp(0.0, (GRID - 1) as f32)
        };
        let value = match self.layer {
            Layer::Shortage => self.values[(y * GRID / PIXELS) * GRID + x * GRID / PIXELS],
            Layer::Sites => {
                let (x, y) = (axis(x), axis(y));
                let (left, top) = (x as usize, y as usize);
                let (right, bottom) = ((left + 1).min(GRID - 1), (top + 1).min(GRID - 1));
                let row = |j: usize| {
                    self.values[j * GRID + left].lerp(self.values[j * GRID + right], x.fract())
                };
                row(top).lerp(row(bottom), y.fract()).sqrt()
            }
        };
        ((value * 8.0) as usize).min(7)
    }

    fn print(&self, inks: &[[u8; 4]; 8]) -> Image {
        let mut paper = Vec::with_capacity(PIXELS * PIXELS * 4);
        let mut previous = [0; PIXELS];
        for y in 0..PIXELS {
            let mut left = 0;
            for (x, above) in previous.iter_mut().enumerate() {
                let band = self.band(x, y);
                let contour = (x > 0 && left != band) || (y > 0 && *above != band);
                let graticule = x % (PIXELS / 4) == 0 || y % (PIXELS / 4) == 0;
                let grain = ((x * 173 + y * 67 + x * y * 13) % 7) as i16 - 3;
                let ink = match (contour, graticule) {
                    (true, _) => -33,
                    (_, true) => -21,
                    _ => grain,
                };
                let [r, g, b, _] = inks[band];
                let shade = |c| (i16::from(c) + ink).clamp(0, 255) as u8;
                let edge = x.min(y).min(PIXELS - 1 - x).min(PIXELS - 1 - y);
                let color = match edge {
                    0..=2 if (x / (PIXELS / GRID) + y / (PIXELS / GRID)) % 2 == 0 => {
                        [84, 72, 48, 255]
                    }
                    0..=2 => [238, 225, 190, 255],
                    _ => [shade(r), shade(g), shade(b), 255],
                };
                paper.extend(color);
                *above = band;
                left = band;
            }
        }
        Image::new(
            Extent3d {
                width: PIXELS as u32,
                height: PIXELS as u32,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            paper,
            TextureFormat::Rgba8UnormSrgb,
            default(),
        )
    }
}

pub(super) fn draw(
    dashboard: Res<Dashboard>,
    layer: Res<Layer>,
    theme: Res<Theme>,
    mut previous: Local<Option<Edition>>,
    mut images: ResMut<Assets<Image>>,
    mut swatches: Query<(&Swatch, &mut BackgroundColor)>,
) {
    let field = {
        let display = dashboard.data.lock().expect("dashboard lock");
        let revision = match *layer {
            Layer::Sites => display.map_revision,
            Layer::Shortage => display.observation_revision,
        };
        let edition = Edition {
            layer: *layer,
            theme: *theme,
            revision,
        };
        let changed = Some(edition) != *previous;
        *previous = Some(edition);
        changed.then(|| Field::from_display(&display, *layer))
    };
    if let Some(field) = field {
        let inks = std::array::from_fn(|band| layer.pigment(*theme, band));
        images
            .insert(MAP.id(), field.print(&inks))
            .expect("survey sheet handle");
        for (swatch, mut color) in &mut swatches {
            let [r, g, b, _] = inks[swatch.0];
            color.set_if_neq(BackgroundColor(Color::srgb_u8(r, g, b)));
        }
    }
}

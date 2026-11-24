use super::{
    Chart, Display,
    model::ChartRange,
    theme::{Theme, Tone},
};
use bevy::{
    asset::RenderAssetUsages,
    camera::{RenderTarget, ScalingMode, visibility::RenderLayers},
    ecs::system::SystemParam,
    mesh::PrimitiveTopology,
    prelude::*,
    render::render_resource::TextureFormat,
    sprite_render::AlphaMode2d,
};

#[derive(Resource)]
pub(super) struct Graph {
    pub image: Handle<Image>,
    mesh: Handle<Mesh>,
    camera: Entity,
}

pub(super) fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    let image = images.add(Image::new_target_texture(
        1,
        1,
        TextureFormat::Bgra8UnormSrgb,
        None,
    ));
    let mesh = meshes.add(Geometry::default().mesh(Vec2::ONE));
    let camera = commands
        .spawn((
            Camera2d,
            Camera {
                order: -1,
                clear_color: ClearColorConfig::Custom(Theme::Light.color(Tone::Panel)),
                ..default()
            },
            RenderTarget::Image(image.clone().into()),
            RenderLayers::layer(1),
        ))
        .id();
    commands.spawn((
        Mesh2d(mesh.clone()),
        MeshMaterial2d(materials.add(ColorMaterial {
            color: Color::WHITE,
            alpha_mode: AlphaMode2d::Blend,
            ..default()
        })),
        RenderLayers::layer(1),
        bevy::camera::visibility::NoFrustumCulling,
    ));
    commands.insert_resource(Graph {
        image,
        mesh,
        camera,
    });
}

#[derive(SystemParam)]
pub(super) struct GraphRenderer<'w, 's> {
    graph: Res<'w, Graph>,
    images: ResMut<'w, Assets<Image>>,
    meshes: ResMut<'w, Assets<Mesh>>,
    cameras: Query<'w, 's, (&'static mut Projection, &'static mut Camera)>,
}

impl GraphRenderer<'_, '_> {
    pub fn update(
        &mut self,
        display: &Display,
        panels: &[GraphPanel],
        pixels: UVec2,
        scale: f32,
        theme: Theme,
    ) {
        let mut image = self.images.get_mut(&self.graph.image).expect("graph image");
        if image.size() != pixels {
            image.resize(bevy::render::render_resource::Extent3d {
                width: pixels.x,
                height: pixels.y,
                depth_or_array_layers: 1,
            });
        }
        let size = pixels.as_vec2();
        if let Ok((mut projection, mut camera)) = self.cameras.get_mut(self.graph.camera) {
            camera.clear_color = ClearColorConfig::Custom(theme.color(Tone::Panel));
            *projection = Projection::Orthographic(OrthographicProjection {
                scaling_mode: ScalingMode::Fixed {
                    width: size.x,
                    height: size.y,
                },
                ..OrthographicProjection::default_2d()
            });
        }
        let mut geometry = Geometry { theme, ..default() };
        for panel in panels {
            geometry.panel(display, panel, scale);
        }
        *self.meshes.get_mut(&self.graph.mesh).expect("graph mesh") = geometry.mesh(size);
    }

    pub fn image(&self) -> Handle<Image> {
        self.graph.image.clone()
    }
}

pub(super) const PLOT_MARGIN: f32 = 4.0;

pub(super) struct GraphPanel {
    pub chart: Chart,
    pub range: ChartRange,
    pub pixels: URect,
}

#[derive(Clone, Copy)]
enum LineStyle {
    Current,
    Reference,
}
impl LineStyle {
    fn color(self, color: Color) -> Color {
        match self {
            Self::Current => color,
            Self::Reference => color.with_alpha(0.30),
        }
    }
}

#[derive(Default)]
struct Geometry {
    theme: Theme,
    positions: Vec<[f32; 3]>,
    colors: Vec<[f32; 4]>,
}
impl Geometry {
    fn quad(&mut self, points: [Vec2; 4], color: Color) {
        let color = color.to_linear().to_f32_array();
        self.positions
            .extend([0, 2, 1, 0, 3, 2].map(|index| points[index].extend(0.0).to_array()));
        self.colors.extend([color; 6]);
    }

    fn line(&mut self, from: Vec2, to: Vec2, width: f32, color: Color) {
        let normal = (to - from).normalize_or_zero().perp() * (width * 0.5);
        self.quad(
            [from - normal, to - normal, to + normal, from + normal],
            color,
        );
    }

    fn panel(&mut self, display: &Display, panel: &GraphPanel, scale: f32) {
        let inset = Vec2::splat(PLOT_MARGIN * scale);
        let origin = panel.pixels.min.as_vec2() + inset;
        let extent = (panel.pixels.size().as_vec2() - 2.0 * inset).max(Vec2::ZERO);
        let bounds = Rect::from_corners(origin, origin + extent);
        let range = &panel.range;
        for tick in 0..=4 {
            let offset = extent * (tick as f32 / 4.0);
            self.line(
                origin + Vec2::new(0.0, offset.y),
                origin + Vec2::new(extent.x, offset.y),
                scale,
                self.theme.color(Tone::Rule).with_alpha(0.7),
            );
            self.line(
                origin + Vec2::new(offset.x, 0.0),
                origin + Vec2::new(offset.x, extent.y),
                scale,
                self.theme.color(Tone::Rule).with_alpha(0.45),
            );
        }
        for tick in 0..=16 {
            let fraction = tick as f32 / 16.0;
            let foot = Vec2::new(bounds.min.x + extent.x * fraction, bounds.max.y);
            let length = match tick % 4 {
                0 => 4.0,
                _ => 2.0,
            } * scale;
            self.line(
                foot - Vec2::Y * length,
                foot,
                scale,
                self.theme.color(Tone::Brass),
            );
        }
        if range.minimum < 0.0 {
            let y = bounds.max.y - range.fraction(0.0) as f32 * extent.y;
            self.line(
                Vec2::new(bounds.min.x, y),
                Vec2::new(bounds.max.x, y),
                scale,
                self.theme.color(Tone::Muted).with_alpha(0.55),
            );
        }
        if let Some(reference) = &display.comparison {
            self.history(
                &reference.history,
                panel,
                bounds,
                scale,
                LineStyle::Reference,
            );
        }
        self.history(&display.history, panel, bounds, scale, LineStyle::Current);
    }

    fn history(
        &mut self,
        history: &std::collections::VecDeque<crate::domain::ObservationSnapshot>,
        panel: &GraphPanel,
        bounds: Rect,
        scale: f32,
        style: LineStyle,
    ) {
        let range = &panel.range;
        let chart = panel.chart;
        let extent = bounds.size();
        let point = |day: u64, value: f64| {
            Vec2::new(
                bounds.min.x + range.day_fraction(day) * extent.x,
                bounds.max.y - range.fraction(value) as f32 * extent.y,
            )
        };
        let mut previous = Vec::<Vec2>::new();
        let mut current = Vec::new();
        for sample in chart.samples(history, range.first_day, range.last_day) {
            current.extend(
                sample
                    .values
                    .into_iter()
                    .map(|value| point(sample.day, value)),
            );
            for (series, (&from, &to)) in previous.iter().zip(&current).enumerate() {
                if range.selection.includes(series) && from.is_finite() && to.is_finite() {
                    self.line(
                        from,
                        to,
                        1.8 * scale,
                        style.color(self.theme.color(Tone::Series(chart, series))),
                    );
                }
            }
            std::mem::swap(&mut previous, &mut current);
            current.clear();
        }
        for (series, point) in previous.into_iter().enumerate() {
            if range.selection.includes(series) && point.is_finite() {
                let radius = 2.5 * scale;
                self.quad(
                    [
                        point + Vec2::new(-radius, -radius),
                        point + Vec2::new(radius, -radius),
                        point + Vec2::new(radius, radius),
                        point + Vec2::new(-radius, radius),
                    ],
                    style.color(self.theme.color(Tone::Series(chart, series))),
                );
            }
        }
    }

    fn mesh(mut self, size: Vec2) -> Mesh {
        self.positions.iter_mut().for_each(|point| {
            point[0] -= size.x / 2.0;
            point[1] = size.y / 2.0 - point[1];
        });
        Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        )
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, self.positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, self.colors)
    }
}

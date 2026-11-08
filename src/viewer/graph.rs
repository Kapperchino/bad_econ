use super::{Chart, Display, model::ChartRange, presentation::chart_color, ui};
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
                clear_color: ClearColorConfig::Custom(ui::INK),
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
    cameras: Query<'w, 's, &'static mut Projection>,
}

impl GraphRenderer<'_, '_> {
    pub fn update(
        &mut self,
        display: &Display,
        chart: Chart,
        range: &ChartRange,
        size: Vec2,
        scale: f32,
    ) {
        let size = size.max(Vec2::ONE);
        let pixels = (size / scale).ceil().as_uvec2().max(UVec2::ONE);
        let mut image = self.images.get_mut(&self.graph.image).expect("graph image");
        if image.size() != pixels {
            image.resize(bevy::render::render_resource::Extent3d {
                width: pixels.x,
                height: pixels.y,
                depth_or_array_layers: 1,
            });
        }
        if let Ok(mut projection) = self.cameras.get_mut(self.graph.camera) {
            *projection = Projection::Orthographic(OrthographicProjection {
                scaling_mode: ScalingMode::Fixed {
                    width: size.x,
                    height: size.y,
                },
                ..OrthographicProjection::default_2d()
            });
        }
        let mut geometry = Geometry::default();
        (0..=4).for_each(|row| {
            let y = row as f32 / 4.0 * (size.y - 2.0) + 1.0;
            geometry.line(
                Vec2::new(0.0, y),
                Vec2::new(size.x, y),
                ui::RULE.with_alpha(0.5),
            );
        });
        if range.minimum < 0.0 {
            let y = 3.0 + (1.0 - range.fraction(0.0) as f32) * (size.y - 6.0).max(0.0);
            geometry.line(Vec2::new(0.0, y), Vec2::new(size.x, y), ui::MUTED);
        }
        if let Some(reference) = &display.comparison {
            geometry.history(&reference.history, chart, range, size, LineStyle::Reference);
        }
        geometry.history(&display.history, chart, range, size, LineStyle::Current);
        *self.meshes.get_mut(&self.graph.mesh).expect("graph mesh") = geometry.mesh(size);
    }

    pub fn image(&self) -> Handle<Image> {
        self.graph.image.clone()
    }
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
    positions: Vec<[f32; 3]>,
    colors: Vec<[f32; 4]>,
}
impl Geometry {
    fn quad(&mut self, points: [Vec2; 4], color: Color) {
        [0, 2, 1, 0, 3, 2].into_iter().for_each(|index| {
            self.positions.push(points[index].extend(0.0).to_array());
            self.colors.push(color.to_linear().to_f32_array());
        });
    }

    fn line(&mut self, from: Vec2, to: Vec2, color: Color) {
        let normal = (to - from).normalize_or_zero().perp();
        self.quad(
            [from - normal, to - normal, to + normal, from + normal],
            color,
        );
    }

    fn history(
        &mut self,
        history: &std::collections::VecDeque<crate::domain::ObservationSnapshot>,
        chart: Chart,
        range: &ChartRange,
        size: Vec2,
        style: LineStyle,
    ) {
        let span = range.last_day.saturating_sub(range.first_day).max(1) as f32;
        let point = |day: u64, value: f64| {
            Vec2::new(
                3.0 + day.saturating_sub(range.first_day) as f32 / span * (size.x - 6.0).max(0.0),
                3.0 + (1.0 - range.fraction(value) as f32) * (size.y - 6.0).max(0.0),
            )
        };
        let points = history
            .iter()
            .enumerate()
            .filter(|(_, snapshot)| (range.first_day..=range.last_day).contains(&snapshot.day))
            .map(|(index, snapshot)| {
                chart
                    .values(
                        snapshot,
                        index.checked_sub(1).and_then(|index| history.get(index)),
                    )
                    .into_iter()
                    .map(|value| point(snapshot.day, value))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        points.windows(2).for_each(|pair| {
            pair[0]
                .iter()
                .zip(&pair[1])
                .enumerate()
                .filter(|(series, (from, to))| {
                    range.selection.includes(*series) && from.is_finite() && to.is_finite()
                })
                .for_each(|(series, (&from, &to))| {
                    self.line(from, to, style.color(chart_color(chart, series)));
                });
        });
        if let Some(last) = points.last() {
            last.iter()
                .enumerate()
                .filter(|(series, point)| range.selection.includes(*series) && point.is_finite())
                .for_each(|(series, &point)| {
                    self.quad(
                        [
                            point + Vec2::new(-3.0, -3.0),
                            point + Vec2::new(3.0, -3.0),
                            point + Vec2::new(3.0, 3.0),
                            point + Vec2::new(-3.0, 3.0),
                        ],
                        style.color(chart_color(chart, series)),
                    );
                });
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

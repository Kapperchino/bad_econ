use super::{
    Chart, Display,
    model::{ChartRange, ChartSample},
    theme::{Theme, Tone},
};
use crate::domain::ObservationSnapshot;
use bevy::{
    asset::RenderAssetUsages,
    camera::{RenderTarget, ScalingMode, visibility::RenderLayers},
    ecs::system::SystemParam,
    mesh::PrimitiveTopology,
    prelude::*,
    render::render_resource::TextureFormat,
    sprite_render::AlphaMode2d,
};
use std::collections::VecDeque;

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
        hovered: Option<ChartLine>,
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
            geometry.panel(display, panel, scale, hovered);
        }
        *self.meshes.get_mut(&self.graph.mesh).expect("graph mesh") = geometry.mesh(size);
    }

    pub fn image(&self) -> Handle<Image> {
        self.graph.image.clone()
    }
}

pub(super) const PLOT_MARGIN: f32 = 4.0;
const LINE_HIT_RADIUS: f32 = 6.0;

#[derive(Component)]
pub(super) struct HoverDot;

pub(super) struct GraphPanel {
    pub chart: Chart,
    pub range: ChartRange,
    pub pixels: URect,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct ChartLine {
    pub chart: Chart,
    pub series: usize,
    pub run: LineRun,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum LineRun {
    Current,
    Reference,
}

impl ChartLine {
    fn color(self, theme: Theme, hovered: Option<Self>) -> Color {
        let alpha = match (self.run, hovered.filter(|line| line.chart == self.chart)) {
            (_, Some(line)) if line == self => 1.0,
            (LineRun::Current, Some(_)) => 0.10,
            (LineRun::Reference, Some(_)) => 0.05,
            (LineRun::Current, None) => 1.0,
            (LineRun::Reference, None) => 0.30,
        };
        theme
            .color(Tone::Series(self.chart, self.series))
            .with_alpha(alpha)
    }
}

pub(super) struct LineHit {
    pub observation: LineObservation,
    pub point: Vec2,
    distance: f32,
}

#[derive(Clone, Copy, PartialEq)]
pub(super) struct LineObservation {
    pub line: ChartLine,
    pub day: u64,
    pub value: f64,
}

#[derive(Clone, Copy)]
struct LinePoint {
    day: u64,
    value: f64,
    position: Vec2,
}

impl LineHit {
    fn near(line: ChartLine, cursor: Vec2, from: LinePoint, to: LinePoint) -> Option<Self> {
        let segment = to.position - from.position;
        let fraction = ((cursor - from.position).dot(segment)
            / segment.length_squared().max(f32::EPSILON))
        .clamp(0.0, 1.0);
        let point = from.position + fraction * segment;
        let distance = point.distance_squared(cursor);
        let observation = match fraction < 0.5 {
            true => from,
            false => to,
        };
        (distance <= LINE_HIT_RADIUS.powi(2)).then_some(Self {
            observation: LineObservation {
                line,
                day: observation.day,
                value: observation.value,
            },
            point,
            distance,
        })
    }
}

fn plot_bounds(size: Vec2, margin: f32) -> Rect {
    let inset = Vec2::splat(margin);
    Rect::from_corners(inset, inset + (size - 2.0 * inset).max(Vec2::ZERO))
}

impl GraphPanel {
    fn point(&self, bounds: Rect, day: u64, value: f64) -> Vec2 {
        let extent = bounds.size();
        Vec2::new(
            bounds.min.x + self.range.day_fraction(day) * extent.x,
            bounds.max.y - self.range.fraction(value) as f32 * extent.y,
        )
    }

    fn line(&self, run: LineRun, series: usize) -> ChartLine {
        ChartLine {
            chart: self.chart,
            series,
            run,
        }
    }

    pub fn hit(&self, display: &Display, cursor: Vec2, inverse_scale: f32) -> Option<LineHit> {
        let bounds = plot_bounds(self.pixels.size().as_vec2() * inverse_scale, PLOT_MARGIN);
        std::iter::once((&display.history, LineRun::Current))
            .chain(
                display
                    .comparison
                    .as_ref()
                    .map(|run| (&run.history, LineRun::Reference)),
            )
            .filter_map(|(history, run)| self.hit_history(history, bounds, cursor, run))
            .min_by(|first, second| first.distance.total_cmp(&second.distance))
    }

    fn hit_history(
        &self,
        history: &VecDeque<ObservationSnapshot>,
        bounds: Rect,
        cursor: Vec2,
        run: LineRun,
    ) -> Option<LineHit> {
        let range = &self.range;
        let span = (range.last_day - range.first_day) as f64;
        let offset = |x: f32| {
            let fraction = ((x - bounds.min.x) / bounds.width().max(f32::EPSILON)).clamp(0.0, 1.0);
            f64::from(fraction) * span
        };
        let start = range.first_day + offset(cursor.x - LINE_HIT_RADIUS).floor() as u64;
        let end = range.first_day + offset(cursor.x + LINE_HIT_RADIUS).ceil() as u64;
        let last_index = history
            .partition_point(|sample| sample.day <= range.last_day)
            .checked_sub(1)?;
        let first_index = history
            .partition_point(|sample| sample.day < start)
            .saturating_sub(1);
        let end_index = history
            .partition_point(|sample| sample.day <= end)
            .min(last_index);
        let first_day = history.get(first_index)?.day.max(range.first_day);
        let last_day = history.get(end_index)?.day;
        let endpoint = history[last_index].day;
        let mut previous: Option<ChartSample> = None;
        let mut best = None;
        let point = |day, value| LinePoint {
            day,
            value,
            position: self.point(bounds, day, value),
        };
        for sample in self.chart.samples(history, first_day, last_day) {
            let closest = sample
                .values
                .iter()
                .enumerate()
                .filter(|(series, _)| range.selection.includes(*series))
                .flat_map(|(series, value)| {
                    let to = point(sample.day, *value);
                    let from = previous.as_ref().and_then(|prior| {
                        prior
                            .values
                            .get(series)
                            .map(|value| point(prior.day, *value))
                    });
                    from.into_iter()
                        .chain((sample.day == endpoint).then_some(to))
                        .filter_map(move |from| {
                            LineHit::near(self.line(run, series), cursor, from, to)
                        })
                });
            best = best
                .into_iter()
                .chain(closest)
                .min_by(|first: &LineHit, second| first.distance.total_cmp(&second.distance));
            previous = Some(sample);
        }
        best
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

    fn panel(
        &mut self,
        display: &Display,
        panel: &GraphPanel,
        scale: f32,
        hovered: Option<ChartLine>,
    ) {
        let bounds = plot_bounds(panel.pixels.size().as_vec2(), PLOT_MARGIN * scale);
        let bounds = Rect::from_corners(
            bounds.min + panel.pixels.min.as_vec2(),
            bounds.max + panel.pixels.min.as_vec2(),
        );
        let origin = bounds.min;
        let extent = bounds.size();
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
                LineRun::Reference,
                hovered,
            );
        }
        self.history(
            &display.history,
            panel,
            bounds,
            scale,
            LineRun::Current,
            hovered,
        );
    }

    fn history(
        &mut self,
        history: &VecDeque<ObservationSnapshot>,
        panel: &GraphPanel,
        bounds: Rect,
        scale: f32,
        run: LineRun,
        hovered: Option<ChartLine>,
    ) {
        let range = &panel.range;
        let theme = self.theme;
        let color = |series| panel.line(run, series).color(theme, hovered);
        let mut previous = Vec::<Vec2>::new();
        let mut current = Vec::new();
        for sample in panel
            .chart
            .samples(history, range.first_day, range.last_day)
        {
            current.extend(
                sample
                    .values
                    .into_iter()
                    .map(|value| panel.point(bounds, sample.day, value)),
            );
            for (series, (&from, &to)) in previous.iter().zip(&current).enumerate() {
                if range.selection.includes(series) && from.is_finite() && to.is_finite() {
                    self.line(from, to, 1.8 * scale, color(series));
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
                    color(series),
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

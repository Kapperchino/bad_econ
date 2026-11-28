mod history;

use super::{
    Chart, Display,
    model::{ChartRange, Period, SeriesSelection},
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
pub(super) use history::GraphCache;
use history::{Point, Trace};

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
            geometry.panel(panel, scale, hovered);
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
    pub range: ChartRange,
    pub pixels: URect,
    lines: Vec<GraphLine>,
}

struct GraphLine {
    id: ChartLine,
    trace: Trace,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct ChartLine {
    pub chart: Chart,
    pub series: usize,
    pub run: LineRun,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
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
}

#[derive(Clone, Copy, PartialEq)]
pub(super) struct LineObservation {
    pub line: ChartLine,
    pub day: u64,
    pub value: f64,
}

struct SegmentHit {
    line: ChartLine,
    from_day: u64,
    day_offset: f64,
    point: Vec2,
    distance: f32,
}

impl SegmentHit {
    fn near(
        panel: &GraphPanel,
        line: ChartLine,
        bounds: Rect,
        cursor: Vec2,
        from: Point,
        to: Point,
    ) -> Option<Self> {
        let start = panel.point(bounds, from.day, from.value);
        let segment = panel.point(bounds, to.day, to.value) - start;
        let fraction = ((cursor - start).dot(segment) / segment.length_squared().max(f32::EPSILON))
            .clamp(0.0, 1.0);
        let point = start + fraction * segment;
        let distance = point.distance_squared(cursor);
        (distance <= LINE_HIT_RADIUS.powi(2)).then(|| Self {
            line,
            from_day: from.day,
            day_offset: f64::from(fraction) * (to.day - from.day) as f64,
            point,
            distance,
        })
    }

    fn observation(self, display: &Display) -> Option<LineHit> {
        let history = match self.line.run {
            LineRun::Current => &display.history,
            LineRun::Reference => &display.comparison.as_ref()?.history,
        };
        let offset = |day: u64| day.saturating_sub(self.from_day) as f64;
        let next = history.partition_point(|point| {
            point.day < self.from_day || offset(point.day) < self.day_offset
        });
        let next_distance = offset(history.get(next)?.day) - self.day_offset;
        let index = next
            .checked_sub(1)
            .filter(|&prior| self.day_offset - offset(history[prior].day) < next_distance)
            .unwrap_or(next);
        let sample = self.line.chart.sample(history, index);
        Some(LineHit {
            observation: LineObservation {
                line: self.line,
                day: sample.day,
                value: sample.values[self.line.series],
            },
            point: self.point,
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

    pub fn new(
        display: &Display,
        chart: Chart,
        pixels: URect,
        width: f32,
        period: Period,
        selection: SeriesSelection,
        cache: &mut GraphCache,
    ) -> Self {
        let days = period.days(display.snapshot.day, &display.history);
        let span = days.end() - days.start();
        let columns = (width as u64).max(1);
        let bucket_days = span
            .div_ceil(columns)
            .max(1)
            .checked_next_power_of_two()
            .unwrap_or(1 << 63);
        let lines: Vec<_> = display
            .comparison
            .as_ref()
            .map(|run| (&run.history, LineRun::Reference))
            .into_iter()
            .chain([(&display.history, LineRun::Current)])
            .flat_map(|(history, run)| {
                cache.lines(chart, run, history, &days, bucket_days, selection)
            })
            .collect();
        let values = lines
            .iter()
            .flat_map(|line| line.trace.points.iter().map(|point| point.value));
        Self {
            range: ChartRange::from_values(chart, days, values),
            pixels,
            lines,
        }
    }

    pub fn hit(&self, display: &Display, cursor: Vec2, inverse_scale: f32) -> Option<LineHit> {
        let bounds = plot_bounds(self.pixels.size().as_vec2() * inverse_scale, PLOT_MARGIN);
        let span = (self.range.last_day - self.range.first_day) as f64;
        let offset = |x: f32| {
            let fraction = ((x - bounds.min.x) / bounds.width().max(f32::EPSILON)).clamp(0.0, 1.0);
            f64::from(fraction) * span
        };
        let start = self.range.first_day + offset(cursor.x - LINE_HIT_RADIUS).floor() as u64;
        let end = self.range.first_day + offset(cursor.x + LINE_HIT_RADIUS).ceil() as u64;
        self.lines
            .iter()
            .flat_map(|line| {
                let points = &line.trace.points;
                let first = points
                    .partition_point(|point| point.day < start)
                    .saturating_sub(1);
                let last = (points.partition_point(|point| point.day <= end) + 1).min(points.len());
                points[first..last]
                    .windows(2)
                    .map(|pair| (pair[0], pair[1]))
                    .chain(points.last().map(|&point| (point, point)))
                    .filter_map(move |(from, to)| {
                        SegmentHit::near(self, line.id, bounds, cursor, from, to)
                    })
            })
            .min_by(|a, b| {
                a.distance
                    .total_cmp(&b.distance)
                    .then_with(|| a.line.run.cmp(&b.line.run))
            })?
            .observation(display)
    }
}

#[derive(Default)]
struct Geometry {
    theme: Theme,
    positions: Vec<[f32; 3]>,
    colors: Vec<[f32; 4]>,
}
impl Geometry {
    fn quad(&mut self, points: [Vec2; 4], color: LinearRgba) {
        self.positions
            .extend([0, 2, 1, 0, 3, 2].map(|index| points[index].extend(0.0).to_array()));
        self.colors.extend([color.to_f32_array(); 6]);
    }

    fn line(&mut self, from: Vec2, to: Vec2, width: f32, color: LinearRgba) {
        let normal = (to - from).normalize_or_zero().perp() * (width * 0.5);
        self.quad(
            [from - normal, to - normal, to + normal, from + normal],
            color,
        );
    }

    fn panel(&mut self, panel: &GraphPanel, scale: f32, hovered: Option<ChartLine>) {
        let bounds = plot_bounds(panel.pixels.size().as_vec2(), PLOT_MARGIN * scale);
        let bounds = Rect::from_corners(
            bounds.min + panel.pixels.min.as_vec2(),
            bounds.max + panel.pixels.min.as_vec2(),
        );
        let origin = bounds.min;
        let extent = bounds.size();
        let range = &panel.range;
        let rule = self.theme.color(Tone::Rule);
        let horizontal = rule.with_alpha(0.7).to_linear();
        let vertical = rule.with_alpha(0.45).to_linear();
        let brass = self.theme.color(Tone::Brass).to_linear();
        for tick in 0..=4 {
            let offset = extent * (tick as f32 / 4.0);
            self.line(
                origin + Vec2::new(0.0, offset.y),
                origin + Vec2::new(extent.x, offset.y),
                scale,
                horizontal,
            );
            self.line(
                origin + Vec2::new(offset.x, 0.0),
                origin + Vec2::new(offset.x, extent.y),
                scale,
                vertical,
            );
        }
        for tick in 0..=16 {
            let fraction = tick as f32 / 16.0;
            let foot = Vec2::new(bounds.min.x + extent.x * fraction, bounds.max.y);
            let length = match tick % 4 {
                0 => 4.0,
                _ => 2.0,
            } * scale;
            self.line(foot - Vec2::Y * length, foot, scale, brass);
        }
        if range.minimum < 0.0 {
            let y = bounds.max.y - range.fraction(0.0) as f32 * extent.y;
            self.line(
                Vec2::new(bounds.min.x, y),
                Vec2::new(bounds.max.x, y),
                scale,
                self.theme.color(Tone::Muted).with_alpha(0.55).to_linear(),
            );
        }
        for line in &panel.lines {
            let color = line.id.color(self.theme, hovered).to_linear();
            let mut previous = None;
            for sample in &line.trace.points {
                let current = sample
                    .value
                    .is_finite()
                    .then(|| panel.point(bounds, sample.day, sample.value));
                if let Some((from, to)) = previous.zip(current) {
                    self.line(from, to, 1.8 * scale, color);
                }
                previous = current;
            }
            if let Some(point) = previous {
                let radius = 2.5 * scale;
                self.quad(
                    [
                        point + Vec2::new(-radius, -radius),
                        point + Vec2::new(radius, -radius),
                        point + Vec2::new(radius, radius),
                        point + Vec2::new(-radius, radius),
                    ],
                    color,
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

use super::{Dashboard, DashboardPage, model::Menu, theme::Tone, ui};
use bevy::{
    asset::uuid_handle,
    image::{CompressedImageFormats, ImageSampler, ImageType},
    picking::Pickable,
    prelude::*,
    ui::widget::NodeImageMode,
};

const SEALS: Handle<Image> = uuid_handle!("eb5eb65c-80a1-4a72-a6d3-b3d5ba3a2e00");
const SCENES: Handle<Image> = uuid_handle!("a8dda5e4-a378-4195-91b6-fc43ee82517e");
pub(super) const SYMBOLS: Handle<Image> = uuid_handle!("e566097b-2b14-423d-8644-39e15bbe4a7e");
const MILL: Handle<Image> = uuid_handle!("95a596e4-7254-48c4-8778-7dc8ceefdf45");

pub(super) fn load(images: &mut Assets<Image>) {
    let prints: [(Handle<Image>, &[u8]); 4] = [
        (
            SEALS,
            include_bytes!("../../assets/artwork/register-seals.png"),
        ),
        (
            SCENES,
            include_bytes!("../../assets/artwork/register-scenes.png"),
        ),
        (
            SYMBOLS,
            include_bytes!("../../assets/artwork/survey-symbols.png"),
        ),
        (
            MILL,
            include_bytes!("../../assets/artwork/mill-engraving.png"),
        ),
    ];
    for (handle, bytes) in prints {
        let image = Image::from_buffer(
            bytes,
            ImageType::Extension("png"),
            CompressedImageFormats::NONE,
            true,
            ImageSampler::linear(),
            default(),
        )
        .expect("cabinet artwork");
        images
            .insert(handle.id(), image)
            .expect("cabinet image handle");
    }
}

#[derive(Clone, Copy)]
pub(super) enum Register {
    Section(Menu),
    Companies,
}

impl Register {
    fn number(self) -> usize {
        match self {
            Self::Section(Menu::Overview) => 0,
            Self::Section(Menu::Market) => 1,
            Self::Section(Menu::Industry) => 2,
            Self::Section(Menu::Population) => 3,
            Self::Section(Menu::Finance) => 4,
            Self::Companies => 5,
        }
    }

    pub fn action(self) -> ui::UiAction {
        match self {
            Self::Section(menu) => ui::UiAction::Menu(menu),
            Self::Companies => ui::UiAction::Companies,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Section(menu) => menu.label(),
            Self::Companies => "Companies",
        }
    }

    pub fn seal(self, parent: &mut ChildSpawnerCommands) {
        let n = self.number();
        let min = Vec2::new((n % 3) as f32, (n / 3) as f32) * 160.0;
        parent.spawn((
            ImageNode::new(SEALS).with_rect(Rect::from_corners(min, min + Vec2::splat(160.0))),
            Node {
                width: px(44),
                height: px(44),
                flex_shrink: 0.0,
                ..default()
            },
            Pickable::IGNORE,
        ));
    }
}

pub(super) fn bevel() -> BackgroundGradient {
    let shine = Color::srgba(1.0, 0.85, 0.62, 0.14);
    let shade = Color::srgba(0.02, 0.01, 0.01, 0.24);
    LinearGradient::to_bottom(vec![
        ColorStop::new(shine, percent(0)),
        ColorStop::new(Color::NONE, percent(52)),
        ColorStop::new(shade, percent(100)),
    ])
    .into()
}

pub(super) fn imprint(parent: &mut ChildSpawnerCommands) {
    parent.spawn((
        ImageNode::new(MILL).with_color(Color::srgba(0.85, 0.69, 0.43, 0.40)),
        Node {
            position_type: PositionType::Absolute,
            width: px(150),
            height: px(50),
            right: px(2),
            bottom: px(0),
            ..default()
        },
        Pickable::IGNORE,
    ));
}

#[derive(Component)]
pub(super) struct Frontispiece;

pub(super) fn masthead(
    parent: &mut ChildSpawnerCommands,
    title: impl FnOnce(&mut ChildSpawnerCommands),
) {
    parent
        .spawn((
            Node {
                height: px(122),
                flex_shrink: 0.0,
                overflow: Overflow::clip(),
                border: UiRect::all(px(2)),
                align_items: AlignItems::Center,
                ..default()
            },
            Tone::Nav.surface(),
            Tone::Brass.outline(),
        ))
        .with_children(|plate| {
            let full = Node {
                position_type: PositionType::Absolute,
                width: percent(100),
                height: percent(100),
                ..default()
            };
            plate.spawn((
                Frontispiece,
                ImageNode::new(SCENES)
                    .with_rect(Rect::new(0., 0., 768., 192.))
                    .with_mode(NodeImageMode::Stretch),
                full.clone(),
                Pickable::IGNORE,
            ));
            let ink = Color::srgb_u8(24, 31, 40);
            plate.spawn((
                full,
                BackgroundGradient::from(LinearGradient::to_right(vec![
                    ColorStop::new(ink.with_alpha(0.99), percent(0)),
                    ColorStop::new(ink.with_alpha(0.94), percent(25)),
                    ColorStop::new(ink.with_alpha(0.64), percent(46)),
                    ColorStop::new(ink.with_alpha(0.0), percent(68)),
                ])),
                Pickable::IGNORE,
            ));
            plate
                .spawn(Node {
                    width: percent(51),
                    padding: UiRect::horizontal(px(20)),
                    ..ui::column(4.0)
                })
                .with_children(title);
        });
}

pub(super) fn illustrate(
    dashboard: Res<Dashboard>,
    mut scenes: Query<(&ComputedNode, &mut ImageNode), With<Frontispiece>>,
) {
    let register = match dashboard.page {
        DashboardPage::Companies(_) => Register::Companies,
        _ => Register::Section(dashboard.chart.menu()),
    };
    let mid = Vec2::new(384.0, (register.number() as f32 + 0.5) * 192.0);
    for (node, mut scene) in &mut scenes {
        let size = node.size();
        if size.min_element() > 0.0 {
            let ratio = size.x / size.y;
            let half = Vec2::new(383.0_f32.min(95.0 * ratio), 95.0_f32.min(383.0 / ratio));
            let rect = Some(Rect::from_corners(mid - half, mid + half));
            if scene.rect != rect {
                scene.rect = rect;
            }
        }
    }
}

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum GoodKind {
    Natural,
    Manufactured,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Good {
    Grain = 1,
    Flour = 2,
    Bread = 3,
    Iron = 4,
    Coal = 5,
    Steel = 6,
    Flax = 7,
    Linen = 8,
    Cloth = 9,
    #[serde(alias = "Coats")]
    Clothes = 10,
    Wood = 11,
    Tools = 12,
    Furniture = 13,
    Cotton = 14,
    Dye = 15,
    Glass = 16,
    Construction = 17,
    Housing = 18,
    Locomotives = 19,
    Railway = 20,
    LuxuryClothes = 21,
    LuxuryFurniture = 22,
}
impl Good {
    pub const COUNT: usize = 22;
    pub const LUXURY_RESERVE_DAYS: u32 = 30;
    pub const ALL: [Self; Self::COUNT] = [
        Self::Grain,
        Self::Flour,
        Self::Bread,
        Self::Iron,
        Self::Coal,
        Self::Steel,
        Self::Flax,
        Self::Linen,
        Self::Cloth,
        Self::Clothes,
        Self::Wood,
        Self::Tools,
        Self::Furniture,
        Self::Cotton,
        Self::Dye,
        Self::Glass,
        Self::Construction,
        Self::Housing,
        Self::Locomotives,
        Self::Railway,
        Self::LuxuryClothes,
        Self::LuxuryFurniture,
    ];
    pub fn kind(self) -> GoodKind {
        match self {
            Self::Grain
            | Self::Iron
            | Self::Coal
            | Self::Flax
            | Self::Wood
            | Self::Cotton
            | Self::Dye => GoodKind::Natural,
            Self::Flour
            | Self::Bread
            | Self::Steel
            | Self::Linen
            | Self::Cloth
            | Self::Clothes
            | Self::Tools
            | Self::Furniture
            | Self::Glass
            | Self::Construction
            | Self::Housing
            | Self::Locomotives
            | Self::Railway
            | Self::LuxuryClothes
            | Self::LuxuryFurniture => GoodKind::Manufactured,
        }
    }
    pub fn inputs(self) -> [Option<Self>; 2] {
        match self {
            Self::Flour => [Some(Self::Grain), None],
            Self::Bread => [Some(Self::Flour), None],
            Self::Steel => [Some(Self::Iron), Some(Self::Coal)],
            Self::Linen => [Some(Self::Flax), None],
            Self::Cloth => [Some(Self::Cotton), Some(Self::Dye)],
            Self::Clothes => [Some(Self::Cloth), Some(Self::Linen)],
            Self::Tools => [Some(Self::Iron), Some(Self::Wood)],
            Self::Furniture | Self::Construction => [Some(Self::Wood), Some(Self::Tools)],
            Self::Glass => [Some(Self::Coal), None],
            Self::Housing => [Some(Self::Construction), None],
            Self::Locomotives => [Some(Self::Steel), Some(Self::Tools)],
            Self::LuxuryClothes => [Some(Self::Clothes), Some(Self::Dye)],
            Self::LuxuryFurniture => [Some(Self::Furniture), Some(Self::Glass)],
            _ => [None, None],
        }
    }
    pub fn consumption_interval(self) -> u32 {
        match self {
            Self::Bread => 1,
            Self::Linen | Self::Cloth => 20,
            Self::Clothes | Self::LuxuryClothes => 40,
            Self::Furniture | Self::Glass | Self::LuxuryFurniture => 60,
            _ => 0,
        }
    }
    pub fn luxury_basic(self) -> Option<Self> {
        match self {
            Self::LuxuryClothes => Some(Self::Clothes),
            Self::LuxuryFurniture => Some(Self::Furniture),
            _ => None,
        }
    }
    pub fn initial_price(self) -> u32 {
        [
            150, 300, 450, 160, 100, 350, 100, 220, 260, 600, 100, 320, 500, 100, 120, 250, 120,
            10, 800, 1, 1200, 1500,
        ][self as usize - 1]
    }
    pub fn food<T: Default + Copy>(values: [T; 3]) -> [T; Self::COUNT] {
        let mut goods = [T::default(); Self::COUNT];
        goods[..3].copy_from_slice(&values);
        goods
    }
    pub fn from_index(index: u32) -> Result<Self> {
        index
            .checked_sub(1)
            .and_then(|index| Self::ALL.get(index as usize))
            .copied()
            .context("invalid good")
    }
}

impl std::fmt::Display for Good {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LuxuryClothes => formatter.write_str("Luxury clothes"),
            Self::LuxuryFurniture => formatter.write_str("Luxury furniture"),
            _ => write!(formatter, "{self:?}"),
        }
    }
}

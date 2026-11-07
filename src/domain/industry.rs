use super::{Good, Technique};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, Hash, Serialize, Deserialize, PartialEq, Eq)]
pub enum Economy {
    #[default]
    Agrarian,
    Industrial,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum Technology {
    Mechanization = 1,
    SteamPower = 2,
    Steelmaking = 4,
    PowerLooms = 8,
    SewingMachines = 16,
}

impl Technology {
    pub const ALL: [Self; 5] = [
        Self::Mechanization,
        Self::SteamPower,
        Self::Steelmaking,
        Self::PowerLooms,
        Self::SewingMachines,
    ];

    pub fn prerequisites(self) -> u32 {
        match self {
            Self::Mechanization => 0,
            Self::SteamPower | Self::PowerLooms => Self::Mechanization as u32,
            Self::Steelmaking => Self::SteamPower as u32,
            Self::SewingMachines => Self::PowerLooms as u32 | Self::Steelmaking as u32,
        }
    }
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ProductionMethod {
    MechanicalHarvesting = 1,
    SteamMilling,
    MechanicalBaking,
    SteamMining,
    BessemerProcess,
    PowerLooms,
    SewingMachines,
    Sawmills,
    MachineTools,
    FurnitureWorkshops,
    Glassworks,
    MechanizedConstruction,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Recipe {
    pub technique: Technique,
    pub inputs: [Option<Good>; 2],
}

pub struct ProductionScale {
    pub bonus_per_mille: u32,
}

impl ProductionScale {
    pub const BONUS_PER_DOUBLING: u32 = 50;
    pub const MAX_DOUBLINGS: u32 = 3;

    pub fn for_plots(plots: u64) -> Self {
        Self {
            bonus_per_mille: plots.max(1).ilog2().min(Self::MAX_DOUBLINGS)
                * Self::BONUS_PER_DOUBLING,
        }
    }
}

impl ProductionMethod {
    pub const ALL: [Self; 12] = [
        Self::MechanicalHarvesting,
        Self::SteamMilling,
        Self::MechanicalBaking,
        Self::SteamMining,
        Self::BessemerProcess,
        Self::PowerLooms,
        Self::SewingMachines,
        Self::Sawmills,
        Self::MachineTools,
        Self::FurnitureWorkshops,
        Self::Glassworks,
        Self::MechanizedConstruction,
    ];

    pub fn from_index(index: u64) -> anyhow::Result<Option<Self>> {
        match index {
            0 => Ok(None),
            _ => Self::ALL
                .into_iter()
                .find(|method| *method as u64 == index)
                .map(Some)
                .ok_or_else(|| anyhow::anyhow!("unknown production method")),
        }
    }

    pub fn technology(self) -> Technology {
        match self {
            Self::MechanicalHarvesting
            | Self::MechanicalBaking
            | Self::Sawmills
            | Self::FurnitureWorkshops
            | Self::MechanizedConstruction => Technology::Mechanization,
            Self::SteamMilling | Self::SteamMining | Self::Glassworks => Technology::SteamPower,
            Self::BessemerProcess | Self::MachineTools => Technology::Steelmaking,
            Self::PowerLooms => Technology::PowerLooms,
            Self::SewingMachines => Technology::SewingMachines,
        }
    }

    pub fn supports(self, good: Good) -> bool {
        Self::for_good(good).contains(&self)
    }

    pub fn for_good(good: Good) -> Vec<Self> {
        match good {
            Good::Grain | Good::Flax | Good::Cotton | Good::Dye => vec![Self::MechanicalHarvesting],
            Good::Flour => vec![Self::SteamMilling],
            Good::Bread => vec![Self::MechanicalBaking],
            Good::Iron | Good::Coal => vec![Self::SteamMining],
            Good::Steel => vec![Self::BessemerProcess],
            Good::Linen | Good::Cloth => vec![Self::PowerLooms],
            Good::Coats => vec![Self::SewingMachines],
            Good::Wood => vec![Self::Sawmills],
            Good::Tools => vec![Self::MachineTools],
            Good::Furniture => vec![Self::FurnitureWorkshops],
            Good::Glass => vec![Self::Glassworks],
            Good::Construction => vec![Self::MechanizedConstruction],
        }
    }

    pub fn recipe(self, good: Good, base: Technique) -> Recipe {
        let inputs = match self {
            Self::MechanicalHarvesting | Self::Sawmills => [Some(Good::Tools), None],
            Self::SteamMining => [Some(Good::Tools), Some(Good::Coal)],
            Self::SteamMilling => [Some(Good::Grain), Some(Good::Coal)],
            Self::MachineTools => [Some(Good::Steel), Some(Good::Wood)],
            _ => good.inputs(),
        };
        Recipe {
            technique: Technique {
                labor_minutes: (base.labor_minutes * 2 / 3).max(1),
                input_ticks: base.input_ticks.max(1),
                output_ticks: (base.output_ticks * 2).min(100),
            },
            inputs,
        }
    }
}

pub fn goods_array<'de, D, T>(deserializer: D) -> Result<[T; Good::COUNT], D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de> + Default + Copy,
{
    let values = Vec::<T>::deserialize(deserializer)?;
    match values.len() {
        3 | 16 | Good::COUNT => {
            let mut result = [T::default(); Good::COUNT];
            result[..values.len()].copy_from_slice(&values);
            Ok(result)
        }
        _ => Err(serde::de::Error::custom(
            "expected three legacy goods or the full goods catalogue",
        )),
    }
}

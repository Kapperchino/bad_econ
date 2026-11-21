use super::{
    Good, ProductionMethod, ScenarioConfig, Technique, enterprise::LandAccess, industry::Recipe,
};
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, Hash, Serialize, Deserialize, PartialEq, Eq)]
pub enum FoodIncome {
    #[default]
    Nominal,
    PurchasingPower,
}

impl FoodIncome {
    fn is_nominal(&self) -> bool {
        *self == Self::Nominal
    }
}

#[derive(Clone, Copy, Debug, Default, Hash, Serialize, Deserialize, PartialEq, Eq)]
pub enum WheatHusbandry {
    #[default]
    AnnualWheat,
    NorfolkFourCourse,
}

impl WheatHusbandry {
    pub(crate) const ROTATION_FIELDS: u32 = 4;
    pub(crate) const GROSS_PARTS: u32 = 8;
    pub(crate) const MARKETABLE_PARTS: u32 = 7;
    pub(crate) const HARVEST_WORK_PER_PACKET: u32 = 8;
    const HAND_WORK_MINUTES_PER_UNIT: u32 = 12;

    fn is_annual(&self) -> bool {
        *self == Self::AnnualWheat
    }

    pub(crate) fn paid_days(self, growing_days: u32) -> u32 {
        match self {
            Self::AnnualWheat => growing_days,
            Self::NorfolkFourCourse => {
                growing_days + growing_days.div_ceil(Self::HARVEST_WORK_PER_PACKET)
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct FoodPolicy {
    pub medium_daily_salary: u32,
    pub rich_daily_salary: u32,
    pub wheat_kg_per_hectare: u32,
    #[serde(default, skip_serializing_if = "WheatHusbandry::is_annual")]
    pub commercial_wheat: WheatHusbandry,
    #[serde(default, skip_serializing_if = "FoodIncome::is_nominal")]
    pub income: FoodIncome,
}

impl Default for FoodPolicy {
    fn default() -> Self {
        Self {
            medium_daily_salary: 2000,
            rich_daily_salary: 10_000,
            wheat_kg_per_hectare: 1601,
            commercial_wheat: WheatHusbandry::AnnualWheat,
            income: FoodIncome::Nominal,
        }
    }
}

impl FoodPolicy {
    pub const HISTORICAL_PENNY_UNITS: u32 = 100;
    pub const BREAD_GRAMS_PER_UNIT: u32 = 500;
    pub const PRICES: [u32; 3] = [697, 1087, 468];

    pub fn industrial() -> Self {
        Self {
            income: FoodIncome::PurchasingPower,
            commercial_wheat: WheatHusbandry::NorfolkFourCourse,
            ..Self::default()
        }
    }

    pub(super) fn reserve_portions(self, config: &ScenarioConfig) -> u64 {
        match self.income {
            FoodIncome::Nominal => 1,
            FoodIncome::PurchasingPower => (u64::from(config.wage)
                * u64::from(config.packet_minutes)
                / (2 * u64::from(config.demography.initial_household_members)
                    * u64::from(config.household_need)
                    * u64::from(config.prices[2])))
            .clamp(1, 10),
        }
    }

    pub fn new(policy: Self) -> Result<Self> {
        match policy.medium_daily_salary > 0
            && policy.medium_daily_salary < policy.rich_daily_salary
            && (1..=10_000).contains(&policy.wheat_kg_per_hectare)
        {
            true => Ok(policy),
            false => Err(anyhow!(
                "food policy requires ordered salary thresholds and a positive crop yield"
            )),
        }
    }

    pub fn inputs(self, good: Good, output: u64) -> u64 {
        match good {
            Good::Flour => (output * 5).div_ceil(4),
            Good::Bread => (output * 25).div_ceil(64),
            _ => output,
        }
    }

    pub fn bread(self, grain: u64) -> u64 {
        grain * 256 / 125
    }

    pub fn hectares(self, workers: u32, technique: Technique) -> u64 {
        let annual_output = u64::from(workers) * u64::from(technique.output_ticks) * 365;
        let yield_kg = u64::from(self.wheat_kg_per_hectare);
        match self.commercial_wheat {
            WheatHusbandry::AnnualWheat => annual_output.div_ceil(yield_kg),
            WheatHusbandry::NorfolkFourCourse => (annual_output
                * u64::from(WheatHusbandry::ROTATION_FIELDS * WheatHusbandry::GROSS_PARTS))
            .div_ceil(yield_kg * u64::from(WheatHusbandry::MARKETABLE_PARTS)),
        }
    }
}

pub(super) enum FarmLand {
    Generated,
    Limited { remaining: u64 },
}

impl FarmLand {
    pub fn new(land: Option<&LandAccess>) -> Self {
        match land {
            None => Self::Generated,
            Some(land) => Self::Limited {
                remaining: u64::from(land.total_plots)
                    - land
                        .allocations
                        .iter()
                        .map(|grant| u64::from(grant.plots))
                        .sum::<u64>(),
            },
        }
    }

    pub fn allocate(&mut self, requested: u64) -> u64 {
        match self {
            Self::Generated => requested,
            Self::Limited { remaining } => {
                let allocated = requested.min(*remaining);
                *remaining -= allocated;
                allocated
            }
        }
    }
}

pub(super) fn technique(config: &ScenarioConfig, good: Good) -> Technique {
    let output_ticks = match good.consumption_interval() {
        2.. => config.durable_packet_output.unwrap_or(config.packet_output),
        _ => config.packet_output,
    };
    let base = Technique {
        labor_minutes: config.packet_minutes,
        input_ticks: match good.inputs()[0] {
            None => 0,
            Some(_) => output_ticks,
        },
        output_ticks,
    };
    match (config.food, good) {
        (Some(food), Good::Grain) if food.commercial_wheat == WheatHusbandry::NorfolkFourCourse => {
            let minutes = WheatHusbandry::HAND_WORK_MINUTES_PER_UNIT.min(config.work_cap);
            let output_ticks = base.output_ticks.min(config.work_cap / minutes);
            Technique {
                labor_minutes: output_ticks * minutes,
                output_ticks,
                ..base
            }
        }
        (Some(_), Good::Flour) => Technique {
            input_ticks: 25,
            output_ticks: 20,
            ..base
        },
        (Some(_), Good::Bread) => Technique {
            input_ticks: 25,
            output_ticks: 64,
            ..base
        },
        _ => base,
    }
}

pub fn upgraded_recipe(
    config: &ScenarioConfig,
    method: ProductionMethod,
    good: Good,
    base: Technique,
) -> Recipe {
    let mut recipe = method.recipe(good, base);
    if let (Some(_), Good::Flour | Good::Bread) = (config.food, good) {
        let batches = (100 / base.output_ticks)
            .min(100 / base.input_ticks.max(1))
            .min(2);
        recipe.technique.input_ticks = base.input_ticks * batches;
        recipe.technique.output_ticks = base.output_ticks * batches;
    }
    recipe
}

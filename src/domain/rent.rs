use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};

pub const FIRM_RENT_GRACE_DAYS: u32 = 30;

#[repr(u32)]
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub enum HousingState {
    #[default]
    Housed,
    Homeless,
}

impl HousingState {
    pub fn from_index(index: u64) -> Result<Self> {
        match index {
            0 => Ok(Self::Housed),
            1 => Ok(Self::Homeless),
            _ => Err(anyhow!("unknown housing state {index}")),
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RentPolicy {
    pub household_daily: u32,
    pub firm_daily: u32,
}

impl Default for RentPolicy {
    fn default() -> Self {
        Self {
            household_daily: 1,
            firm_daily: 10,
        }
    }
}

impl RentPolicy {
    pub fn new(policy: Self) -> Result<Self> {
        match [policy.household_daily, policy.firm_daily]
            .into_iter()
            .all(|amount| amount <= 1_000_000)
        {
            true => Ok(policy),
            false => Err(anyhow!("daily rent exceeds one million cash ticks")),
        }
    }
}

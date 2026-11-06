use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RentPolicy {
    pub household_daily: u32,
    pub firm_daily: u32,
}

impl Default for RentPolicy {
    fn default() -> Self {
        Self {
            household_daily: 5,
            firm_daily: 5,
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

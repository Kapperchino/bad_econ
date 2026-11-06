use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub enum HouseholdLiquidity {
    #[default]
    Uniform,
    PaycheckToPaycheck,
}

impl HouseholdLiquidity {
    pub(super) fn is_uniform(&self) -> bool {
        *self == Self::Uniform
    }

    pub(super) fn cash(self, reference: u64, seed: u64, household: u64) -> Result<u64> {
        let weight = match self {
            Self::Uniform => 1000,
            Self::PaycheckToPaycheck => {
                let draw = crate::arithmetic::DrawKey {
                    seed,
                    day: 0,
                    phase: 0,
                    actor: household,
                    purpose: 1,
                    index: 0,
                }
                .draw();
                let variation = draw >> 32;
                match draw % 100 {
                    0..=59 => 250 + variation % 751,
                    60..=89 => 1001 + variation % 3000,
                    _ => 4001 + variation % 12_000,
                }
            }
        };
        crate::arithmetic::floor_ratio(reference, weight, 1000)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct HouseholdPolicy {
    pub fatigue_response: u32,
    pub rest_minutes: u32,
}

impl HouseholdPolicy {
    pub fn new(policy: Self, reserved_minutes: u32) -> Result<Self> {
        let coefficients = [policy.fatigue_response];
        match coefficients
            .into_iter()
            .all(|coefficient| coefficient <= 1000)
            && policy.rest_minutes <= reserved_minutes
        {
            true => Ok(policy),
            false => Err(anyhow!(
                "household coefficients must lie in 0..=1000 and rest must fit reserved time"
            )),
        }
    }
}

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct HouseholdPolicy {
    pub recovery: u32,
    pub deficit_damage: u32,
    pub fatigue_damage: u32,
    pub fatigue_response: u32,
    pub rest_minutes: u32,
}

impl HouseholdPolicy {
    pub fn new(policy: Self, reserved_minutes: u32) -> Result<Self> {
        let coefficients = [
            policy.recovery,
            policy.deficit_damage,
            policy.fatigue_damage,
            policy.fatigue_response,
        ];
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

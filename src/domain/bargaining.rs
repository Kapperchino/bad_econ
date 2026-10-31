use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct BargainingPolicy {
    pub legal_minimum: u32,
    pub quantum: u32,
    pub daily_cash_cost: u32,
    pub funded_other: u32,
    pub participating_adults: u32,
    pub normal_day_minutes: u32,
    pub work_cost_per_minute: u32,
    pub desired_wage: u32,
    pub mandate: u32,
    pub withdrawal: u32,
    pub alternatives: u32,
    pub concentration: u32,
    pub replacements: u32,
}

impl BargainingPolicy {
    pub fn new(policy: Self) -> Result<Self> {
        let fractions = [
            policy.mandate,
            policy.withdrawal,
            policy.alternatives,
            policy.concentration,
            policy.replacements,
        ];
        let denominator = policy
            .participating_adults
            .checked_mul(policy.normal_day_minutes)
            .and_then(|minutes| minutes.checked_mul(35_000));
        let admitted = (1..=10_000).contains(&policy.legal_minimum)
            && (1..=10_000).contains(&policy.quantum)
            && (1..=1440).contains(&policy.normal_day_minutes)
            && policy.participating_adults > 0
            && denominator.is_some()
            && policy.work_cost_per_minute <= 10_000
            && policy.desired_wage <= 10_000
            && fractions.into_iter().all(|fraction| fraction <= 1000);
        match admitted {
            true => Ok(policy),
            false => Err(anyhow!(
                "bargaining policy exceeds wage, fraction or household denominator bounds"
            )),
        }
    }
}

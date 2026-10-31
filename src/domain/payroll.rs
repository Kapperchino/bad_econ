use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};

pub const PAYROLL_SCALE: u32 = 1000;
pub const MAX_TAX_BRACKETS: usize = 8;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TaxBracket {
    pub threshold: u64,
    pub marginal_rate: u32,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PayrollPolicy {
    pub base_rate: u32,
    pub employee_charge: u32,
    #[serde(default)]
    pub brackets: Vec<TaxBracket>,
}

impl PayrollPolicy {
    pub fn new(policy: Self) -> Result<Self> {
        let rates = std::iter::once(policy.base_rate)
            .chain(policy.brackets.iter().map(|bracket| bracket.marginal_rate));
        let ordered = policy
            .brackets
            .windows(2)
            .all(|pair| pair[0].threshold < pair[1].threshold);
        let admitted = policy.employee_charge < PAYROLL_SCALE
            && rates
                .into_iter()
                .all(|rate| rate < PAYROLL_SCALE - policy.employee_charge)
            && policy.brackets.len() <= MAX_TAX_BRACKETS
            && ordered;
        match admitted {
            true => Ok(policy),
            false => Err(anyhow!(
                "payroll requires ordered thresholds, at most {MAX_TAX_BRACKETS} brackets, and positive marginal net income"
            )),
        }
    }
}

use super::{Economy, Field, Good, Scenario, State, enterprise::EnterpriseState};
use crate::arithmetic::add;
use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct BusinessAccount {
    pub opening_cash: u64,
    pub closing_cash: u64,
    pub revenue: u64,
}
impl BusinessAccount {
    pub fn profit(self) -> i128 {
        i128::from(self.closing_cash) - i128::from(self.opening_cash)
    }
    pub fn outlays(self) -> i128 {
        i128::from(self.revenue) - self.profit()
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct EconomyObservation {
    pub workforce: u64,
    pub subsistence: u64,
    pub inactive: u64,
    #[serde(default)]
    pub rural_employed: u64,
    #[serde(default)]
    pub transport_workers: u64,
    #[serde(default)]
    pub cartage_minutes: u64,
    #[serde(default)]
    pub cartage_receipts: u64,
    #[serde(default)]
    pub awaiting_pickup: u64,
    pub household_cash: u64,
    pub active_by_good: [u64; Good::COUNT],
    pub business_types: [u64; 3],
    pub business_states: [u64; 4],
    pub firm_cash: [u64; Good::COUNT],
    pub accounts: [BusinessAccount; Good::COUNT],
}
impl EconomyObservation {
    pub fn from_state(scenario: &Scenario, state: &State) -> Result<Self> {
        state
            .actors
            .iter()
            .enumerate()
            .try_fold(Self::default(), |mut result, (index, actor)| {
                result.cartage_minutes = add(
                    result.cartage_minutes,
                    state.get(Field::CartageMinutes, index),
                )?;
                result.cartage_receipts = add(
                    result.cartage_receipts,
                    state.get(Field::CartageIncome, index),
                )?;
                result.awaiting_pickup =
                    add(result.awaiting_pickup, state.get(Field::RoadFreight, index))?;
                let employer = state.get(Field::Employer, index);
                if employer
                    .checked_sub(1)
                    .and_then(|row| state.actors.get(row as usize))
                    .is_some_and(|firm| firm.kind == Good::Railway as u32)
                {
                    result.transport_workers =
                        add(result.transport_workers, state.get(Field::Employed, index))?;
                }
                match actor.kind {
                    0 => {
                        result.household_cash =
                            add(result.household_cash, state.get(Field::Cash, index))?;
                        let working = state.get(Field::Worker, index) != u64::from(u32::MAX)
                            && state.get(Field::LivingMembers, index) > 0;
                        let subsistence = scenario.config().economy == Economy::Industrial
                            && state.get(Field::LandPlots, index) > 0;
                        result.rural_employed += match subsistence {
                            true => state.get(Field::Employed, index),
                            false => 0,
                        };
                        let participants = match scenario.config().people {
                            Some(_) => state
                                .get(Field::EligibleWorkers, index)
                                .min(state.get(Field::LivingMembers, index)),
                            None => u64::from(working),
                        };
                        let population = match scenario.config().people {
                            Some(_) => state.get(Field::LivingMembers, index),
                            None => 1,
                        };
                        result.inactive += population - participants;
                        match (working, subsistence) {
                            (true, false) => result.workforce += participants,
                            (true, true) => result.subsistence += participants,
                            (false, _) => (),
                        }
                    }
                    kind => {
                        let good = kind as usize - 1;
                        let status = state.get(Field::EnterpriseStatus, index);
                        let business = match (kind, state.get(Field::EnterpriseModel, index)) {
                            (kind, _) if kind == Good::Housing as u32 => 2,
                            (_, 1) => 1,
                            _ => 0,
                        };
                        if status < EnterpriseState::Closed as u64 {
                            result.active_by_good[good] += 1;
                            result.business_types[business] += 1;
                            result.business_states[status as usize] += 1;
                        }
                        result.firm_cash[good] =
                            add(result.firm_cash[good], state.get(Field::Cash, index))?;
                        let account = &mut result.accounts[good];
                        account.opening_cash = add(
                            account.opening_cash,
                            state.get(Field::BusinessOpeningCash, index),
                        )?;
                        account.closing_cash = add(
                            account.closing_cash,
                            state.get(Field::BusinessClosingCash, index),
                        )?;
                        account.revenue =
                            add(account.revenue, state.get(Field::BusinessRevenue, index))?;
                    }
                }
                Ok(result)
            })
    }

    pub fn wage_employed(&self, employed: u64) -> u64 {
        employed.saturating_sub(self.rural_employed)
    }
}

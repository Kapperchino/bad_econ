use super::{
    AuditedState, BASE_FIELDS, CONSUMPTION_FIELDS, Field, Good, HousingState, PRODUCTION_FIELDS,
    Scenario, State, TRANSIT_FIELDS, USED_FIELDS, calendar, demography, diagnostics, enterprise,
    finance, railway, research,
};
use crate::arithmetic::add;
use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct QuoteDistribution {
    pub count: u64,
    pub minimum: u64,
    pub maximum: u64,
    pub total: u64,
    pub squares: u64,
}
impl QuoteDistribution {
    pub fn include(&mut self, quote: u64) -> Result<()> {
        self.minimum = match self.count {
            0 => quote,
            _ => self.minimum.min(quote),
        };
        self.maximum = self.maximum.max(quote);
        self.count = add(self.count, 1)?;
        self.total = add(self.total, quote)?;
        self.squares = add(
            self.squares,
            quote.checked_mul(quote).context("quote square overflow")?,
        )?;
        Ok(())
    }
    pub fn mean(&self) -> f64 {
        self.total as f64 / self.count.max(1) as f64
    }
    pub fn deviation(&self) -> f64 {
        let numerator = u128::from(self.count) * u128::from(self.squares)
            - u128::from(self.total) * u128::from(self.total);
        (numerator as f64).sqrt() / self.count.max(1) as f64
    }
}
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct MarketObservation {
    pub wages: QuoteDistribution,
    pub prices: QuoteDistribution,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ObservationSnapshot {
    #[serde(default)]
    pub economy: diagnostics::EconomyObservation,
    pub day: u64,
    #[serde(default)]
    pub weather: calendar::Weather,
    pub cash: u64,
    pub payroll_reserves: u64,
    pub income_tax: u64,
    pub employee_charges: u64,
    pub rent_collected: u64,
    pub rent_arrears: u64,
    pub financial_cash: u64,
    pub loan_principal: u64,
    pub loan_interest: u64,
    pub active_loans: u64,
    pub overdue_loans: u64,
    pub research: research::ResearchSummary,
    pub employed: u64,
    pub households: u64,
    pub homeless_households: u64,
    pub firms: u64,
    pub firm_entries: u64,
    pub firm_closures: u64,
    pub distributors: u64,
    pub household_production: [u64; Good::COUNT],
    pub shortages: u64,
    pub population: demography::PopulationObservation,
    pub inventories: [u64; Good::COUNT],
    pub transit: [u64; Good::COUNT],
    pub produced: [u64; Good::COUNT],
    pub used: [u64; Good::COUNT],
    pub consumed: u64,
    pub consumption: [u64; Good::COUNT],
    pub minimum_wage: u64,
    pub maximum_wage: u64,
    pub minimum_price: u64,
    pub maximum_price: u64,
    pub markets: [MarketObservation; Good::COUNT],
    pub wealth_bins: Vec<u64>,
    pub regions: Vec<u64>,
}
impl ObservationSnapshot {
    pub fn from_state(scenario: &Scenario, state: &State) -> Result<Self> {
        let state = AuditedState::new(scenario, state)?.state;
        let mut snapshot = Self {
            day: state.day,
            weather: scenario.weather(state.day),
            economy: diagnostics::EconomyObservation::from_state(scenario, state)?,
            research: state
                .research
                .as_ref()
                .map(|research| research.summary.clone())
                .unwrap_or_default(),
            households: state.actors.iter().filter(|actor| actor.kind == 0).count() as u64,
            population: demography::PopulationObservation::for_scenario(state, scenario),
            minimum_wage: u64::MAX,
            minimum_price: u64::MAX,
            wealth_bins: vec![0; 64],
            regions: vec![0; 256],
            ..Self::default()
        };
        state
            .actors
            .iter()
            .enumerate()
            .try_for_each(|(i, actor)| -> Result<()> {
                snapshot.cash = add(
                    snapshot.cash,
                    add(state.get(Field::Cash, i), state.get(Field::Escrow, i))?,
                )?;
                let rent = state.get(Field::RentTreasury, i);
                snapshot.rent_collected = add(snapshot.rent_collected, rent)?;
                snapshot.rent_arrears =
                    add(snapshot.rent_arrears, state.get(Field::RentArrears, i))?;
                snapshot.cash = add(snapshot.cash, state.get(Field::RentPending, i))?;
                let reserve = state.get(Field::PayrollReserve, i);
                let tax = state.get(Field::TaxTreasury, i);
                let charge = state.get(Field::ChargeTreasury, i);
                snapshot.payroll_reserves = add(snapshot.payroll_reserves, reserve)?;
                snapshot.income_tax = add(snapshot.income_tax, tax)?;
                snapshot.employee_charges = add(snapshot.employee_charges, charge)?;
                snapshot.cash = add(snapshot.cash, add(reserve, add(tax, charge)?)?)?;
                (0..Good::COUNT).try_for_each(|good| -> Result<()> {
                    snapshot.inventories[good] = add(
                        snapshot.inventories[good],
                        state.values[(good + 1) * state.actors.len() + i],
                    )?;
                    Ok(())
                })?;
                let mut freight_due = 0;
                for good in 0..Good::COUNT {
                    let deliveries = (0..scenario.ring()).try_fold(0, |total, slot| {
                        let field = BASE_FIELDS + good * scenario.ring() + slot;
                        let in_transit = state.values[field * state.actors.len() + i];
                        let awaiting_carriage = state.values
                            [(field + Good::COUNT * scenario.ring()) * state.actors.len() + i];
                        let rate = slot as u64
                            * u64::from(
                                railway::CART_BILLABLE_MINUTES_PER_UNIT_DAY
                                    * scenario.config().wage,
                            );
                        freight_due = add(
                            freight_due,
                            crate::arithmetic::mul(awaiting_carriage, rate)?,
                        )?;
                        add(total, add(in_transit, awaiting_carriage)?)
                    })?;
                    match deliveries
                        == state.values[(TRANSIT_FIELDS + good) * state.actors.len() + i]
                    {
                        true => (),
                        false => Err(anyhow!(
                            "shipment ownership mismatch for actor {} and {:?}",
                            actor.identity,
                            Good::ALL[good]
                        ))?,
                    }
                    snapshot.transit[good] = add(
                        snapshot.transit[good],
                        state.values[(TRANSIT_FIELDS + good) * state.actors.len() + i],
                    )?;
                    let household_output =
                        state.values[(PRODUCTION_FIELDS + good) * state.actors.len() + i];
                    snapshot.household_production[good] =
                        add(snapshot.household_production[good], household_output)?;
                    snapshot.produced[good] = add(snapshot.produced[good], household_output)?;
                    snapshot.used[good] = add(
                        snapshot.used[good],
                        state.values[(USED_FIELDS + good) * state.actors.len() + i],
                    )?;
                    snapshot.consumption[good] = add(
                        snapshot.consumption[good],
                        state.values[(CONSUMPTION_FIELDS + good) * state.actors.len() + i],
                    )?;
                }
                match actor.kind {
                    0 => {
                        snapshot.employed += state.get(Field::Employed, i);
                        snapshot.homeless_households += u64::from(
                            state.get(Field::LivingMembers, i) > 0
                                && state.get(Field::HousingStatus, i)
                                    == HousingState::Homeless as u64,
                        );
                        snapshot.shortages =
                            add(snapshot.shortages, state.get(Field::Shortage, i))?;
                        snapshot.consumed = add(snapshot.consumed, state.get(Field::Consumed, i))?;
                        let members = match scenario.config().people {
                            Some(_) => state.get(Field::LivingMembers, i),
                            None => 1,
                        };
                        let personal_cash = state.get(Field::Cash, i) / members.max(1);
                        snapshot.wealth_bins
                            [(64 - personal_cash.leading_zeros() as usize).min(63)] += members;
                        let x = actor.position.x * 16 / scenario.config().map_size;
                        let y = actor.position.y * 16 / scenario.config().map_size;
                        snapshot.regions[(y * 16 + x) as usize] += state.get(Field::Shortage, i);
                    }
                    kind => {
                        snapshot.firms += 1;
                        snapshot.firm_entries += u64::from(i >= scenario.actors());
                        snapshot.firm_closures += u64::from(
                            state.get(Field::EnterpriseStatus, i)
                                == enterprise::EnterpriseState::Closed as u64,
                        );
                        snapshot.distributors += u64::from(
                            state.get(Field::EnterpriseModel, i)
                                == enterprise::EnterpriseModel::Distributor as u64
                                && state.get(Field::EnterpriseStatus, i)
                                    != enterprise::EnterpriseState::Closed as u64,
                        );
                        let j = kind as usize - 1;
                        let trading = matches!(
                            enterprise::EnterpriseState::from_index(
                                state.get(Field::EnterpriseStatus, i),
                            )?,
                            enterprise::EnterpriseState::Operating
                                | enterprise::EnterpriseState::Dormant
                        );
                        if trading {
                            snapshot.markets[j]
                                .wages
                                .include(state.get(Field::Wage, i))?;
                            snapshot.minimum_wage =
                                snapshot.minimum_wage.min(state.get(Field::Wage, i));
                            snapshot.maximum_wage =
                                snapshot.maximum_wage.max(state.get(Field::Wage, i));
                            let stocked = kind == Good::Housing as u32
                                || kind == Good::Railway as u32
                                || state.get(Field::inventory(Good::from_index(kind)?), i) > 0;
                            if stocked {
                                snapshot.markets[j]
                                    .prices
                                    .include(state.get(Field::Price, i))?;
                                snapshot.minimum_price =
                                    snapshot.minimum_price.min(state.get(Field::Price, i));
                                snapshot.maximum_price =
                                    snapshot.maximum_price.max(state.get(Field::Price, i));
                            }
                        }
                        snapshot.produced[j] =
                            add(snapshot.produced[j], state.get(Field::Produced, i))?;
                        snapshot.inventories[j] =
                            add(snapshot.inventories[j], state.get(Field::Pipeline, i))?;
                    }
                }
                let in_ring = (0..scenario.ring() * Good::COUNT).try_fold(0, |total, slot| {
                    add(
                        total,
                        state.values[(BASE_FIELDS + slot) * state.actors.len() + i],
                    )
                })?;
                match add(in_ring, state.get(Field::RoadFreight, i))?
                    == state.get(Field::Transit, i)
                    && freight_due == state.get(Field::CartagePayable, i)
                {
                    true => Ok(()),
                    false => Err(anyhow!(
                        "transit ownership audit failed for {}",
                        actor.identity
                    )),
                }
            })?;
        snapshot.minimum_wage = match snapshot.minimum_wage {
            u64::MAX => 0,
            value => value,
        };
        snapshot.minimum_price = match snapshot.minimum_price {
            u64::MAX => 0,
            value => value,
        };
        if let Some(finance) = &scenario.config().finance {
            let view = finance::FinanceStateView::new(
                finance,
                finance.layout(state.actors.len() * scenario.fields())?,
                &state.values,
                state.day,
            )?;
            snapshot.financial_cash = view.cash;
            snapshot.loan_principal = view.principal;
            snapshot.loan_interest = view.interest_due;
            snapshot.active_loans = view.active_loans;
            snapshot.overdue_loans = view.overdue_loans;
            snapshot.cash = add(snapshot.cash, view.cash)?;
        }
        snapshot.audit(state.initial_cash, state.initial_goods)?;
        Ok(snapshot)
    }
    pub fn audit(&self, initial_cash: u64, initial_goods: [u64; Good::COUNT]) -> Result<()> {
        match self.cash == initial_cash {
            true => Ok(()),
            false => Err(anyhow!("cash conservation failed")),
        }?;
        (0..Good::COUNT).try_for_each(|i| {
            let sources = u128::from(initial_goods[i]) + u128::from(self.produced[i]);
            let uses = u128::from(self.inventories[i])
                + u128::from(self.transit[i])
                + u128::from(self.used[i])
                + u128::from(self.research.consumed[i])
                + u128::from(self.consumption[i]);
            match sources == uses {
                true => Ok(()),
                false => Err(anyhow!(
                    "goods conservation failed for good {}: {sources} != {uses}",
                    i + 1
                )),
            }
        })
    }
}

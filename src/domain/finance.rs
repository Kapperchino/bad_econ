use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const FINANCE_SCALE: u32 = 1000;
pub const BANK_FIELDS: usize = 3;
pub const LOAN_FIELDS: usize = 7;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum Consent {
    Granted,
    Withheld,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct BankSeed {
    pub identity: u64,
    pub reserves: u64,
    pub minimum_capital: u32,
    pub risk_weight: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct LoanContract {
    pub identity: u64,
    pub bank: u64,
    pub borrower: u64,
    pub requested: u64,
    pub target_cash: u64,
    pub first_day: u32,
    pub term: u32,
    pub daily_rate: u32,
    pub surplus: u64,
    pub debt_service: u64,
    pub service_fraction: u32,
    pub advance: u32,
    pub collateral: u64,
    pub secured: u64,
    pub consent: Consent,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SupportRecipient {
    pub household: u64,
    pub target_cash: u64,
    pub consent: Consent,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SupportPolicy {
    pub treasury: u64,
    pub recipients: Vec<SupportRecipient>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct FinanceConfig {
    pub banks: Vec<BankSeed>,
    pub loans: Vec<LoanContract>,
    pub support: Option<SupportPolicy>,
}

impl FinanceConfig {
    pub fn new(mut config: Self, actors: &[u64], households: &[u64]) -> Result<Self> {
        config.banks.sort_by_key(|bank| bank.identity);
        config.loans.sort_by_key(|loan| (loan.bank, loan.identity));
        if let Some(support) = &mut config.support {
            support
                .recipients
                .sort_by_key(|recipient| recipient.household);
        }
        let bank_ids = config
            .banks
            .iter()
            .map(|bank| bank.identity)
            .collect::<BTreeSet<_>>();
        let loan_ids = config
            .loans
            .iter()
            .map(|loan| loan.identity)
            .collect::<BTreeSet<_>>();
        let borrowers = config
            .loans
            .iter()
            .map(|loan| loan.borrower)
            .collect::<BTreeSet<_>>();
        let actors = actors.iter().copied().collect::<BTreeSet<_>>();
        let households = households.iter().copied().collect::<BTreeSet<_>>();
        let banks = bank_ids.len() == config.banks.len()
            && !bank_ids.contains(&0)
            && config.banks.iter().all(|bank| {
                (1..=FINANCE_SCALE).contains(&bank.minimum_capital)
                    && (1..=10_000).contains(&bank.risk_weight)
            });
        let loans = loan_ids.len() == config.loans.len()
            && !loan_ids.contains(&0)
            && borrowers.len() == config.loans.len()
            && config.loans.iter().all(|loan| {
                bank_ids.contains(&loan.bank)
                    && actors.contains(&loan.borrower)
                    && (1..=36_500).contains(&loan.term)
                    && loan.daily_rate <= FINANCE_SCALE
                    && loan.service_fraction <= FINANCE_SCALE
                    && loan.advance <= FINANCE_SCALE
                    && loan.first_day.checked_add(loan.term).is_some()
            });
        let support = config.support.as_ref().is_none_or(|support| {
            let identities = support
                .recipients
                .iter()
                .map(|recipient| recipient.household)
                .collect::<BTreeSet<_>>();
            identities.len() == support.recipients.len()
                && identities
                    .iter()
                    .all(|identity| households.contains(identity))
        });
        let dimensions = config.banks.len() <= 65_535
            && config.loans.len() <= 1_000_000
            && config
                .support
                .as_ref()
                .is_none_or(|support| support.recipients.len() <= 1_000_000);
        match banks && loans && support && dimensions {
            true => {
                config.initial_cash()?;
                Ok(config)
            }
            false => Err(anyhow!(
                "invalid finance identities, coefficients, recipients, or dimensions"
            )),
        }
    }

    pub fn initial_cash(&self) -> Result<u64> {
        self.banks.iter().try_fold(
            self.support.as_ref().map_or(0, |support| support.treasury),
            |total, bank| {
                total
                    .checked_add(bank.reserves)
                    .context("financial opening cash overflow")
            },
        )
    }

    pub fn state_len(&self) -> usize {
        self.banks.len() * BANK_FIELDS
            + self.loans.len() * LOAN_FIELDS
            + usize::from(self.support.is_some())
    }

    pub fn memory_estimate(&self) -> u64 {
        let banks = self.banks.len() as u64;
        let loans = self.loans.len() as u64;
        let recipients = self
            .support
            .as_ref()
            .map_or(0, |support| support.recipients.len()) as u64;
        let scratch_ticks = recipients.max(banks * 5).max(1);
        banks * 20
            + loans * 80
            + recipients * 16
            + 12
            + scratch_ticks * 24
            + 32
            + 16
            + loans.max(1) * 4
    }

    pub fn initial_values(&self) -> Vec<u64> {
        [BankField::Reserve, BankField::Equity, BankField::RiskAssets]
            .into_iter()
            .flat_map(|field| {
                self.banks.iter().map(move |bank| match field {
                    BankField::Reserve | BankField::Equity => bank.reserves,
                    BankField::RiskAssets => 0,
                })
            })
            .chain(std::iter::repeat_n(0, self.loans.len() * LOAN_FIELDS))
            .chain(self.support.as_ref().map(|support| support.treasury))
            .collect()
    }

    pub fn layout(&self, offset: usize) -> Result<FinanceLayout> {
        let end = offset
            .checked_add(self.state_len())
            .context("financial state length overflow")?;
        let _ = u32::try_from(end).context("financial coordinates exceed u32")?;
        Ok(FinanceLayout {
            banks: offset as u32,
            loans: (offset + self.banks.len() * BANK_FIELDS) as u32,
            treasury: (offset + self.banks.len() * BANK_FIELDS + self.loans.len() * LOAN_FIELDS)
                as u32,
            end: end as u32,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FinanceLayout {
    pub banks: u32,
    pub loans: u32,
    pub treasury: u32,
    pub end: u32,
}

impl FinanceLayout {
    pub fn bank_count(self) -> usize {
        (self.loans - self.banks) as usize / BANK_FIELDS
    }
    pub fn loan_count(self) -> usize {
        (self.treasury - self.loans) as usize / LOAN_FIELDS
    }
    pub fn bank(self, field: BankField, index: usize) -> usize {
        self.banks as usize + field as usize * self.bank_count() + index
    }
    pub fn loan(self, field: LoanField, index: usize) -> usize {
        self.loans as usize + field as usize * self.loan_count() + index
    }
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BankField {
    Reserve,
    Equity,
    RiskAssets,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoanField {
    Principal,
    Interest,
    InterestRemainder,
    Due,
    Status,
    RiskAssets,
    LastAccrual,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoanStatus {
    Dormant,
    Active,
    Overdue,
    Repaid,
}

pub struct FinanceStateView {
    pub cash: u64,
    pub principal: u64,
    pub interest_due: u64,
    pub active_loans: u64,
    pub overdue_loans: u64,
}

impl FinanceStateView {
    pub fn new(
        config: &FinanceConfig,
        layout: FinanceLayout,
        values: &[u64],
        day: u64,
    ) -> Result<Self> {
        match values.len() >= layout.end as usize {
            true => Ok(()),
            false => Err(anyhow!("incomplete financial state")),
        }?;
        let mut view = Self {
            cash: 0,
            principal: 0,
            interest_due: 0,
            active_loans: 0,
            overdue_loans: 0,
        };
        for (bank_index, bank) in config.banks.iter().enumerate() {
            let reserve = values[layout.bank(BankField::Reserve, bank_index)];
            let equity = values[layout.bank(BankField::Equity, bank_index)];
            let risk_assets = values[layout.bank(BankField::RiskAssets, bank_index)];
            let mut principal_total = 0u128;
            let mut risk_total = 0u128;
            let first = config
                .loans
                .partition_point(|loan| loan.bank < bank.identity);
            let end = config
                .loans
                .partition_point(|loan| loan.bank <= bank.identity);
            for (relative, loan) in config.loans[first..end].iter().enumerate() {
                let loan_index = first + relative;
                let principal = values[layout.loan(LoanField::Principal, loan_index)];
                let interest = values[layout.loan(LoanField::Interest, loan_index)];
                let remainder = values[layout.loan(LoanField::InterestRemainder, loan_index)];
                let due = values[layout.loan(LoanField::Due, loan_index)];
                let status = values[layout.loan(LoanField::Status, loan_index)];
                let risk = values[layout.loan(LoanField::RiskAssets, loan_index)];
                let last = values[layout.loan(LoanField::LastAccrual, loan_index)];
                let lifecycle = match status {
                    0 => [principal, interest, remainder, due, risk, last]
                        .into_iter()
                        .all(|value| value == 0),
                    1 => principal > 0 && due > last && last <= day && risk > 0,
                    2 => principal > 0 && due > 0 && last >= due && last <= day && risk > 0,
                    3 => {
                        principal == 0
                            && interest == 0
                            && risk == 0
                            && due > 0
                            && last >= due
                            && last <= day
                    }
                    _ => false,
                };
                match lifecycle
                    && principal <= loan.requested
                    && principal <= loan.target_cash
                    && (loan.consent == Consent::Granted || status == LoanStatus::Dormant as u64)
                    && (status == LoanStatus::Dormant as u64
                        || due >= u64::from(loan.first_day) + u64::from(loan.term))
                    && remainder < u64::from(FINANCE_SCALE)
                    && u128::from(risk)
                        == (u128::from(principal) * u128::from(bank.risk_weight))
                            .div_ceil(u128::from(FINANCE_SCALE))
                    && due <= u64::from(u32::MAX)
                    && last <= u64::from(u32::MAX)
                {
                    true => (),
                    false => Err(anyhow!(
                        "invalid financial loan state for {}",
                        loan.identity
                    ))?,
                }
                principal_total += u128::from(principal);
                risk_total += u128::from(risk);
                view.principal = view
                    .principal
                    .checked_add(principal)
                    .context("loan principal total overflow")?;
                view.interest_due = view
                    .interest_due
                    .checked_add(interest)
                    .context("loan interest total overflow")?;
                view.active_loans += u64::from(status == 1);
                view.overdue_loans += u64::from(status == 2);
            }
            match u128::from(reserve) + principal_total == u128::from(equity)
                && risk_total == u128::from(risk_assets)
                && equity >= bank.reserves
            {
                true => (),
                false => Err(anyhow!("bank balance sheet failed for {}", bank.identity))?,
            }
            view.cash = view
                .cash
                .checked_add(reserve)
                .context("bank reserve total overflow")?;
        }
        if config.support.is_some() {
            view.cash = view
                .cash
                .checked_add(values[layout.treasury as usize])
                .context("support treasury total overflow")?;
        }
        Ok(view)
    }
}

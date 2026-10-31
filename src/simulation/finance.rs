mod kernels;
mod payments;
mod policies;

use super::{
    Device,
    flags::{CONTROL_WORDS, FaultFlags},
    layout::{TICK_WORDS, WORD_BYTES},
    words::*,
};
use crate::{domain::finance::*, ledger::Coordinate};
use anyhow::{Context, Result, anyhow};
use burn_wgpu::CubeTensor;
use bytemuck::{Pod, Zeroable};
use cubecl::prelude::*;
use std::{collections::BTreeMap, mem::offset_of};

pub use payments::{
    BankAccount, BankPayment, BankPaymentResult, bank_payments, post_bank_payments,
};
pub use policies::{
    BudgetPolicy, BudgetResult, DiscountFlow, DiscountPlan, DiscountedValue, SignedAmount,
};
pub use policies::{
    CashFlow, DivisiblePool, FinancePolicies, FundingPlan, FundingResult, LendingInput,
    LendingResult, PoolResult,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinancialActorKind {
    Household,
    Firm,
}

#[derive(Clone, Copy, Debug)]
pub struct FinancialActor {
    pub identity: u64,
    pub cash: Coordinate,
    pub kind: FinancialActorKind,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BankMeta {
    state: u32,
    first_loan: u32,
    loan_count: u32,
    minimum_capital: u32,
    risk_weight: u32,
}
impl BankMeta {
    const STATE: usize = offset_of!(Self, state) / WORD_BYTES;
    const FIRST_LOAN: usize = offset_of!(Self, first_loan) / WORD_BYTES;
    const LOAN_COUNT: usize = offset_of!(Self, loan_count) / WORD_BYTES;
    const MINIMUM_CAPITAL: usize = offset_of!(Self, minimum_capital) / WORD_BYTES;
    const RISK_WEIGHT: usize = offset_of!(Self, risk_weight) / WORD_BYTES;
    const WORDS: usize = size_of::<Self>() / WORD_BYTES;
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LoanMeta {
    state: u32,
    cash: u32,
    requested: crate::ledger::TickLayout,
    target_cash: crate::ledger::TickLayout,
    first_day: u32,
    term: u32,
    daily_rate: u32,
    surplus: crate::ledger::TickLayout,
    debt_service: crate::ledger::TickLayout,
    service_fraction: u32,
    advance: u32,
    collateral: crate::ledger::TickLayout,
    secured: crate::ledger::TickLayout,
    consent: u32,
}
impl LoanMeta {
    const STATE: usize = offset_of!(Self, state) / WORD_BYTES;
    const CASH: usize = offset_of!(Self, cash) / WORD_BYTES;
    const REQUESTED: usize = offset_of!(Self, requested) / WORD_BYTES;
    const TARGET_CASH: usize = offset_of!(Self, target_cash) / WORD_BYTES;
    const FIRST_DAY: usize = offset_of!(Self, first_day) / WORD_BYTES;
    const TERM: usize = offset_of!(Self, term) / WORD_BYTES;
    const DAILY_RATE: usize = offset_of!(Self, daily_rate) / WORD_BYTES;
    const SURPLUS: usize = offset_of!(Self, surplus) / WORD_BYTES;
    const DEBT_SERVICE: usize = offset_of!(Self, debt_service) / WORD_BYTES;
    const SERVICE_FRACTION: usize = offset_of!(Self, service_fraction) / WORD_BYTES;
    const ADVANCE: usize = offset_of!(Self, advance) / WORD_BYTES;
    const COLLATERAL: usize = offset_of!(Self, collateral) / WORD_BYTES;
    const SECURED: usize = offset_of!(Self, secured) / WORD_BYTES;
    const CONSENT: usize = offset_of!(Self, consent) / WORD_BYTES;
    const WORDS: usize = size_of::<Self>() / WORD_BYTES;
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SupportMeta {
    cash: u32,
    target: crate::ledger::TickLayout,
    consent: u32,
}
impl SupportMeta {
    const CASH: usize = offset_of!(Self, cash) / WORD_BYTES;
    const TARGET: usize = offset_of!(Self, target) / WORD_BYTES;
    const CONSENT: usize = offset_of!(Self, consent) / WORD_BYTES;
    const WORDS: usize = size_of::<Self>() / WORD_BYTES;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FinanceStage {
    Opening,
    Lending,
    Audit,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct FinanceObservation {
    pub bank_reserves: u64,
    pub support_treasury: u64,
    pub principal: u64,
    pub interest_due: u64,
    pub active_loans: u64,
    pub overdue_loans: u64,
}
impl FinanceObservation {
    pub fn cash(&self) -> Result<u64> {
        self.bank_reserves
            .checked_add(self.support_treasury)
            .context("financial cash total overflow")
    }
}

pub struct FinanceRuntime {
    banks: CubeTensor,
    loans: CubeTensor,
    support: CubeTensor,
    scan_a: CubeTensor,
    scan_b: CubeTensor,
    output: CubeTensor,
    control: CubeTensor,
    decisions: CubeTensor,
    last_event_count: std::cell::Cell<usize>,
    last_support_count: std::cell::Cell<usize>,
    bank_count: usize,
    support_count: usize,
    loan_count: usize,
    layout: FinanceLayout,
    support_mode: SupportMode,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SupportMode {
    Absent,
    Present,
}

impl FinanceRuntime {
    pub fn new(
        device: &Device,
        config: &FinanceConfig,
        layout: FinanceLayout,
        actors: &[FinancialActor],
    ) -> Result<Self> {
        let ids = actors
            .iter()
            .map(|actor| actor.identity)
            .collect::<Vec<_>>();
        let households = actors
            .iter()
            .filter(|actor| actor.kind == FinancialActorKind::Household)
            .map(|actor| actor.identity)
            .collect::<Vec<_>>();
        let admitted = FinanceConfig::new(config.clone(), &ids, &households)?;
        let expected = admitted.layout(layout.banks as usize)?;
        let coordinates = actors
            .iter()
            .map(|actor| actor.cash.0)
            .collect::<std::collections::BTreeSet<_>>();
        match expected == layout
            && coordinates.len() == actors.len()
            && coordinates
                .iter()
                .all(|&coordinate| coordinate < layout.banks)
        {
            true => Ok(()),
            false => Err(anyhow!(
                "financial coordinate layout overlaps actor resources"
            )),
        }?;
        let actor_map = actors
            .iter()
            .map(|actor| (actor.identity, actor.cash.0))
            .collect::<BTreeMap<_, _>>();
        let banks = admitted
            .banks
            .iter()
            .enumerate()
            .map(|(index, bank)| {
                let first = admitted
                    .loans
                    .partition_point(|loan| loan.bank < bank.identity);
                let end = admitted
                    .loans
                    .partition_point(|loan| loan.bank <= bank.identity);
                BankMeta {
                    state: layout.banks + (index * BANK_FIELDS) as u32,
                    first_loan: first as u32,
                    loan_count: (end - first) as u32,
                    minimum_capital: bank.minimum_capital,
                    risk_weight: bank.risk_weight,
                }
            })
            .collect::<Vec<_>>();
        let loans = admitted
            .loans
            .iter()
            .enumerate()
            .map(|(index, loan)| LoanMeta {
                state: layout.loans + (index * LOAN_FIELDS) as u32,
                cash: actor_map[&loan.borrower],
                requested: loan.requested.into(),
                target_cash: loan.target_cash.into(),
                first_day: loan.first_day,
                term: loan.term,
                daily_rate: loan.daily_rate,
                surplus: loan.surplus.into(),
                debt_service: loan.debt_service.into(),
                service_fraction: loan.service_fraction,
                advance: loan.advance,
                collateral: loan.collateral.into(),
                secured: loan.secured.into(),
                consent: u32::from(loan.consent == Consent::Granted),
            })
            .collect::<Vec<_>>();
        let recipients = admitted
            .support
            .as_ref()
            .map(|support| {
                support
                    .recipients
                    .iter()
                    .map(|recipient| SupportMeta {
                        cash: actor_map[&recipient.household],
                        target: recipient.target_cash.into(),
                        consent: u32::from(recipient.consent == Consent::Granted),
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let scratch = recipients.len().max(banks.len() * 5).max(1) * TICK_WORDS;
        Ok(Self {
            banks: device.tensor(
                bytemuck::cast_slice(&banks)
                    .iter()
                    .copied()
                    .chain(std::iter::once(0))
                    .collect::<Vec<_>>()
                    .as_slice(),
            ),
            loans: device.tensor(
                bytemuck::cast_slice(&loans)
                    .iter()
                    .copied()
                    .chain(std::iter::once(0))
                    .collect::<Vec<_>>()
                    .as_slice(),
            ),
            support: device.tensor(
                bytemuck::cast_slice(&recipients)
                    .iter()
                    .copied()
                    .chain(std::iter::once(0))
                    .collect::<Vec<_>>()
                    .as_slice(),
            ),
            scan_a: device.empty(scratch),
            scan_b: device.empty(scratch),
            output: device.empty(scratch + 4 * TICK_WORDS),
            control: device.tensor(&[0; CONTROL_WORDS]),
            decisions: device.empty(loans.len().max(1)),
            last_event_count: std::cell::Cell::new(0),
            last_support_count: std::cell::Cell::new(0),
            bank_count: banks.len(),
            support_count: recipients.len(),
            loan_count: loans.len(),
            layout,
            support_mode: match admitted.support {
                Some(_) => SupportMode::Present,
                None => SupportMode::Absent,
            },
        })
    }

    fn check_state(&self, state: &CubeTensor) -> Result<()> {
        match state.meta.shape()[0] >= self.layout.end as usize * TICK_WORDS {
            true => Ok(()),
            false => Err(anyhow!("financial state buffer is incomplete")),
        }
    }

    fn clear(&self, device: &Device) {
        device.write(&self.control, &[0; CONTROL_WORDS]);
    }

    fn faults(&self, device: &Device) -> Result<()> {
        FaultFlags::read(&device.read(&self.control)?)?.into_result()
    }

    pub fn phase(
        &self,
        device: &Device,
        state: &CubeTensor,
        day: u32,
        stage: FinanceStage,
    ) -> Result<()> {
        self.check_state(state)?;
        self.clear(device);
        kernels::banks::launch(
            &device.client,
            count(device, self.bank_count),
            CubeDim::new_1d(device.workgroup),
            argument(state),
            argument(&self.banks),
            argument(&self.loans),
            argument(&self.output),
            argument(&self.decisions),
            argument(&self.control),
            day,
            self.bank_count as u32,
            stage,
        );
        let control = device.read(&self.control)?;
        self.last_event_count
            .set(super::flags::ControlLayout::new(&control)?.phase_data as usize);
        FaultFlags::read(&control)?.into_result()
    }

    pub fn support(&self, device: &Device, state: &CubeTensor) -> Result<()> {
        self.check_state(state)?;
        self.clear(device);
        self.last_event_count.set(0);
        self.last_support_count.set(0);
        match self.support_mode {
            SupportMode::Absent => (),
            SupportMode::Present => {
                kernels::support_needs::launch(
                    &device.client,
                    count(device, self.support_count),
                    CubeDim::new_1d(device.workgroup),
                    argument(state),
                    argument(&self.support),
                    argument(&self.scan_a),
                    self.support_count as u32,
                );
                let scanned = self.scan(device, self.support_count, 1)?;
                kernels::support_plan::launch(
                    &device.client,
                    count(device, self.support_count + 1),
                    CubeDim::new_1d(device.workgroup),
                    argument(state),
                    argument(&self.support),
                    argument(scanned),
                    argument(&self.output),
                    argument(&self.control),
                    self.support_count as u32,
                    self.layout.treasury,
                );
                let control = device.read(&self.control)?;
                FaultFlags::read(&control)?.into_result()?;
                self.last_event_count
                    .set(super::flags::ControlLayout::new(&control)?.phase_data as usize);
                self.last_support_count.set(self.last_event_count.get());
                kernels::support_post::launch(
                    &device.client,
                    count(device, self.support_count + 1),
                    CubeDim::new_1d(device.workgroup),
                    argument(state),
                    argument(&self.support),
                    argument(&self.output),
                    self.support_count as u32,
                    self.layout.treasury,
                );
            }
        }
        Ok(())
    }

    fn scan<'a>(
        &'a self,
        device: &Device,
        length: usize,
        channels: usize,
    ) -> Result<&'a CubeTensor> {
        let mut offset = 1;
        let mut current = &self.scan_a;
        let mut next = &self.scan_b;
        while offset < length {
            kernels::scan::launch(
                &device.client,
                count(device, length * channels),
                CubeDim::new_1d(device.workgroup),
                argument(current),
                argument(next),
                argument(&self.control),
                length as u32,
                channels as u32,
                offset as u32,
            );
            std::mem::swap(&mut current, &mut next);
            offset *= 2;
        }
        self.faults(device)?;
        Ok(current)
    }

    pub fn observe(
        &self,
        device: &Device,
        state: &CubeTensor,
        day: u32,
    ) -> Result<FinanceObservation> {
        self.phase(device, state, day, FinanceStage::Audit)?;
        kernels::audit_columns::launch(
            &device.client,
            count(device, self.bank_count * 5),
            CubeDim::new_1d(device.workgroup),
            argument(&self.output),
            argument(&self.scan_a),
            self.bank_count as u32,
        );
        let totals = self.scan(device, self.bank_count, 5)?;
        let values = match self.bank_count {
            0 => vec![0; 5],
            count => (0..5)
                .map(|channel| {
                    let words = device.read_range(
                        totals,
                        (channel * count + count - 1) * TICK_WORDS,
                        TICK_WORDS,
                    )?;
                    Ok(u64::from(words[0]) | (u64::from(words[1]) << 32))
                })
                .collect::<Result<Vec<_>>>()?,
        };
        let support_treasury = match self.support_mode {
            SupportMode::Absent => 0,
            SupportMode::Present => {
                let words = device.read_range(
                    state,
                    self.layout.treasury as usize * TICK_WORDS,
                    TICK_WORDS,
                )?;
                u64::from(words[0]) | (u64::from(words[1]) << 32)
            }
        };
        Ok(FinanceObservation {
            bank_reserves: values[0],
            support_treasury,
            principal: values[1],
            interest_due: values[2],
            active_loans: values[3],
            overdue_loans: values[4],
        })
    }

    pub fn readback_bytes_per_observation(&self) -> u64 {
        (CONTROL_WORDS * WORD_BYTES * 2
            + (5 * usize::from(self.bank_count > 0)
                + usize::from(self.support_mode == SupportMode::Present))
                * TICK_WORDS
                * WORD_BYTES) as u64
    }

    pub fn loan_count(&self) -> usize {
        self.loan_count
    }

    pub fn last_event_count(&self) -> usize {
        self.last_event_count.get()
    }

    pub fn support_decisions(&self) -> Vec<crate::ledger::Decision> {
        vec![crate::ledger::Decision::Accepted; self.last_support_count.get()]
    }

    pub fn decisions(&self, device: &Device) -> Result<Vec<crate::ledger::Decision>> {
        match self.loan_count {
            0 => Ok(Vec::new()),
            count => device
                .read_range(&self.decisions, 0, count)?
                .into_iter()
                .map(|code| match code {
                    0 => Ok(None),
                    1 => Ok(Some(crate::ledger::Decision::Accepted)),
                    2 => Ok(Some(crate::ledger::Decision::Rejected)),
                    _ => Err(anyhow!("unknown financial event decision")),
                })
                .collect::<Result<Vec<_>>>()
                .map(|decisions| decisions.into_iter().flatten().collect()),
        }
    }

    pub fn phase_readback_bytes(&self) -> u64 {
        (CONTROL_WORDS * WORD_BYTES) as u64
    }

    pub fn support_readback_bytes(&self) -> u64 {
        match self.support_mode {
            SupportMode::Absent => 0,
            SupportMode::Present => (2 * CONTROL_WORDS * WORD_BYTES) as u64,
        }
    }

    pub fn decision_readback_bytes(&self) -> u64 {
        (self.loan_count * WORD_BYTES) as u64
    }
}

fn argument(tensor: &CubeTensor) -> BufferArg {
    unsafe { BufferArg::from_raw_parts(tensor.handle.clone(), tensor.meta.shape()[0]) }
}

fn count(device: &Device, length: usize) -> CubeCount {
    let groups = (length as u32).div_ceil(device.workgroup).max(1);
    let x = groups.min(u16::MAX as u32);
    CubeCount::Static(x, groups.div_ceil(x), 1)
}

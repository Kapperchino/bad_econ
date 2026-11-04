mod kernels;
mod payments;
mod policies;
mod scan;
mod systems;

use super::rows::{RowIndex, row_layout};
use super::{
    Device,
    flags::{CONTROL_WORDS, FaultFlags},
    layout::{TICK_WORDS, WORD_BYTES},
    pass::GpuPass,
    system::{System, gpu_passes},
    words::*,
};
use crate::{domain::finance::*, ledger::Coordinate};
use anyhow::{Context, Result, anyhow};
use burn_wgpu::CubeTensor;
use bytemuck::{Pod, Zeroable};
use cubecl::prelude::*;
use std::{collections::BTreeMap, mem::offset_of};
use systems::*;

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

row_layout!(BankMetaRows, BankMeta {
    state: u32 => STATE,
    first_loan: u32 => FIRST_LOAN,
    loan_count: u32 => LOAN_COUNT,
    minimum_capital: u32 => MINIMUM_CAPITAL,
    risk_weight: u32 => RISK_WEIGHT,
});

row_layout!(LoanMetaRows, LoanMeta {
    state: u32 => STATE,
    cash: u32 => CASH,
    requested: crate::ledger::TickLayout => REQUESTED,
    target_cash: crate::ledger::TickLayout => TARGET_CASH,
    first_day: u32 => FIRST_DAY,
    term: u32 => TERM,
    daily_rate: u32 => DAILY_RATE,
    surplus: crate::ledger::TickLayout => SURPLUS,
    debt_service: crate::ledger::TickLayout => DEBT_SERVICE,
    service_fraction: u32 => SERVICE_FRACTION,
    advance: u32 => ADVANCE,
    collateral: crate::ledger::TickLayout => COLLATERAL,
    secured: crate::ledger::TickLayout => SECURED,
    consent: u32 => CONSENT,
});

row_layout!(SupportMetaRows, SupportMeta {
    cash: u32 => CASH,
    target: crate::ledger::TickLayout => TARGET,
    consent: u32 => CONSENT,
});

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
    scanned: CubeTensor,
    bank_scan: scan::Scan,
    support_scan: scan::Scan,
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
                    state: layout.banks + index as u32,
                    first_loan: first as u32,
                    loan_count: (end - first) as u32,
                    minimum_capital: bank.minimum_capital,
                    risk_weight: bank.risk_weight,
                }
            })
            .collect::<BankMetaRows>();
        let loans = admitted
            .loans
            .iter()
            .enumerate()
            .map(|(index, loan)| LoanMeta {
                state: layout.loans + index as u32,
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
            .collect::<LoanMetaRows>();
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
                    .collect::<SupportMetaRows>()
            })
            .unwrap_or_default();
        let scratch = recipients.len().max(banks.len() * 5).max(1) * TICK_WORDS;
        let bank_scan = scan::Scan::new(device, banks.len(), 5);
        let support_scan = scan::Scan::new(device, recipients.len(), 1);
        Ok(Self {
            banks: device.tensor(
                banks
                    .words()
                    .iter()
                    .copied()
                    .chain(std::iter::once(0))
                    .collect::<Vec<_>>()
                    .as_slice(),
            ),
            loans: device.tensor(
                loans
                    .words()
                    .iter()
                    .copied()
                    .chain(std::iter::once(0))
                    .collect::<Vec<_>>()
                    .as_slice(),
            ),
            support: device.tensor(
                recipients
                    .words()
                    .iter()
                    .copied()
                    .chain(std::iter::once(0))
                    .collect::<Vec<_>>()
                    .as_slice(),
            ),
            scanned: device.empty(bank_scan.words().max(support_scan.words())),
            bank_scan,
            support_scan,
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
        BankAccounting::launch(
            device,
            count(device, self.bank_count),
            CubeDim::new_1d(device.workgroup),
            BankInputLaunch {
                state: argument(state),
                banks: argument(&self.banks),
                loans: argument(&self.loans),
                output: argument(&self.scanned),
                decisions: argument(&self.decisions),
                control: argument(&self.control),
                day,
                count: self.bank_count as u32,
                stage,
            },
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
                SupportNeeds::launch(
                    device,
                    count(device, self.support_count),
                    CubeDim::new_1d(device.workgroup),
                    SupportNeedsInputLaunch {
                        state: argument(state),
                        recipients: argument(&self.support),
                        output: argument(&self.scanned),
                        count: self.support_count as u32,
                    },
                );
                let scanned = self.scan(device, &self.support_scan)?;
                SupportPlanning::launch(
                    device,
                    count(device, self.support_count + 1),
                    CubeDim::new_1d(device.workgroup),
                    SupportPlanInputLaunch {
                        state: argument(state),
                        recipients: argument(&self.support),
                        scanned: argument(scanned),
                        output: argument(&self.output),
                        control: argument(&self.control),
                        count: self.support_count as u32,
                        treasury: self.layout.treasury,
                    },
                );
                let control = device.read(&self.control)?;
                FaultFlags::read(&control)?.into_result()?;
                self.last_event_count
                    .set(super::flags::ControlLayout::new(&control)?.phase_data as usize);
                self.last_support_count.set(self.last_event_count.get());
                SupportPosting::launch(
                    device,
                    count(device, self.support_count + 1),
                    CubeDim::new_1d(device.workgroup),
                    SupportPostInputLaunch {
                        state: argument(state),
                        recipients: argument(&self.support),
                        output: argument(&self.output),
                        count: self.support_count as u32,
                        treasury: self.layout.treasury,
                    },
                );
            }
        }
        Ok(())
    }

    fn scan<'a>(&'a self, device: &Device, plan: &scan::Scan) -> Result<&'a CubeTensor> {
        plan.run(device, &self.scanned, &self.control);
        self.faults(device)?;
        Ok(&self.scanned)
    }

    pub fn observe(
        &self,
        device: &Device,
        state: &CubeTensor,
        day: u32,
    ) -> Result<FinanceObservation> {
        self.phase(device, state, day, FinanceStage::Audit)?;
        let totals = self.scan(device, &self.bank_scan)?;
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
    tensor.clone().into_buffer_arg()
}

fn count(device: &Device, length: usize) -> CubeCount {
    device.count(length)
}

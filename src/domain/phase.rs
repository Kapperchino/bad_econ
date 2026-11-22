use super::Good;
use serde::{Deserialize, Serialize};

#[enumflags2::bitflags]
#[repr(u32)]
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum Phase {
    Opening,
    Observation,
    Proposal,
    Commitment,
    WorkPayment,
    ProvisionPurchase,
    PhysicalClose,
    AccountsDecisions,
    Audit,
}
impl Phase {
    pub fn next(self) -> Self {
        match self {
            Self::Opening => Self::Observation,
            Self::Observation => Self::Proposal,
            Self::Proposal => Self::Commitment,
            Self::Commitment => Self::WorkPayment,
            Self::WorkPayment => Self::ProvisionPurchase,
            Self::ProvisionPurchase => Self::PhysicalClose,
            Self::PhysicalClose => Self::AccountsDecisions,
            Self::AccountsDecisions => Self::Audit,
            Self::Audit => Self::Opening,
        }
    }
    pub const ALL: [Self; 9] = [
        Self::Opening,
        Self::Observation,
        Self::Proposal,
        Self::Commitment,
        Self::WorkPayment,
        Self::ProvisionPurchase,
        Self::PhysicalClose,
        Self::AccountsDecisions,
        Self::Audit,
    ];
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Market {
    Labor,
    Input(Good),
    Household,
    Consumer(Good),
    Payroll,
    Distribution,
}
impl Market {
    pub fn code(self) -> u32 {
        match self {
            Self::Labor => 0,
            Self::Input(good) => good as u32,
            Self::Consumer(good) => Good::COUNT as u32 + good as u32,
            Self::Household => 2 * Good::COUNT as u32 + 1,
            Self::Payroll => 2 * Good::COUNT as u32 + 2,
            Self::Distribution => 2 * Good::COUNT as u32 + 3,
        }
    }
}
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PhaseEvidence {
    pub decisions: Vec<crate::ledger::Decision>,
    pub frontier_depth: usize,
    pub events: usize,
}

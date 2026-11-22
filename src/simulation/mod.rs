pub mod analytics;
pub mod bargaining;
mod buffers;
mod checkpoint;
#[cfg(feature = "viewer")]
pub(crate) mod company_report;
mod configuration;
mod demography;
mod dispatch;
mod economy;
mod enterprise;
pub mod finance;
mod flags;
pub mod households;
mod initialization;
mod inspection;
mod layout;
mod ledger_execution;
mod lifecycle;
mod market;
mod market_graphs;
mod memory;
pub mod negotiation;
mod observation;
mod ops;
mod parallel;
mod parameters;
pub mod pass;
mod payroll;
mod policy;
mod profiling;
mod provision;
mod random;
mod registry;
pub mod research;
mod rows;
mod state;
pub mod system;
mod tensor;
#[cfg(test)]
mod tests;
mod words;

#[cfg(test)]
use crate::domain::Market;
use crate::domain::{ActorRows, ObservationSnapshot, Phase, Scenario, State};
use crate::runtime::{PhaseTiming, RuntimeState};
use buffers::DeviceEconomy;
use burn_wgpu::CubeTensor;
use checkpoint::CopyDirection;
use economy::{markets, railway::settlement, scheduler};
use enterprise::{management, ownership};
use layout::{decode, encode};
#[cfg(test)]
use ledger_execution::Execution;
use ledger_execution::execute;
pub use ledger_execution::{LedgerResult, ledger};
#[cfg(test)]
use lifecycle::ClaimCapacityDemand;
use market::{FreightActivity, FreightExecution, MarketExecution, MarketReplay, Provisioning};
pub use memory::MemoryBudget;
#[cfg(test)]
use observation::AuditExecution;
pub use ops::Device;
use parallel::reduction;
use profiling::profiling_day;
use registry::RegistryAudit;
#[cfg(test)]
use registry::RegistryExecution;

#[derive(Clone, Copy, Debug)]
pub enum Trace {
    Record,
    Discard,
}

pub struct Simulation {
    pub scenario: Scenario,
    pub device: Device,
    pub status: RuntimeState,
    pub snapshot: ObservationSnapshot,
    pub timings: Vec<PhaseTiming>,
    pub readback_bytes: u64,
    metadata: State,
    day: u64,
    audited_day: u64,
    next_phase: Phase,
    state: CubeTensor,
    audited: tensor::AuditedBuffer,
    snapshot_copy: tensor::SnapshotCopy,
    actors: CubeTensor,
    config: configuration::Configuration,
    spatial: CubeTensor,
    work: buffers::PhaseBuffers,
    searches: markets::SearchBuffers,
    market_sort: economy::sorting::MarketSort,
    market_graphs: market_graphs::MarketGraphs,
    market_replay: MarketReplay,
    freight_activity: FreightActivity,
    freight_execution: FreightExecution,
    market_execution: MarketExecution,
    control: CubeTensor,
    summary: CubeTensor,
    audit: economy::audit::AuditBuffers,
    #[cfg(test)]
    audit_execution: AuditExecution,
    finance: Option<finance::FinanceRuntime>,
    research: Option<research::ResearchRuntime>,
    research_readback: u64,
    demography: demography::DemographyRuntime,
    enterprise_requests: CubeTensor,
    ownership: Option<ownership::OwnershipRuntime>,
    management: management::EnterpriseManagement,
    migrations: registry::MigrationRegistry,
    #[cfg(test)]
    registry_execution: RegistryExecution,
    matching_round: usize,
    provisioning: Provisioning,
}
impl Simulation {
    pub fn actors(&self) -> &ActorRows {
        &self.metadata.actors
    }
    pub fn memory_usage(&self) -> cubecl::MemoryUsage {
        self.device.memory_usage()
    }
}

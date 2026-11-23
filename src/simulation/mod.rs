pub mod analytics;
pub mod bargaining;
mod demography;
mod economy;
mod enterprise;
pub mod finance;
mod gpu;
pub mod households;
mod market;
mod parallel;
mod registry;
pub(crate) mod reporting;
pub mod research;
mod runtime;
mod storage;
#[cfg(test)]
mod tests;

#[cfg(test)]
use crate::domain::Market;
use crate::domain::{ActorRows, ObservationSnapshot, Phase, Scenario, State};
use crate::runtime::{PhaseTiming, RuntimeState};
pub use bargaining::negotiation;
use burn_wgpu::CubeTensor;
#[cfg(test)]
use economy::settlement::ledger::Execution;
use economy::settlement::ledger::execute;
pub use economy::settlement::ledger::{LedgerResult, ledger};
use economy::{market::orders, settlement::scheduler, transport::railway::settlement};
use enterprise::{management, ownership};
use gpu::tensor;
pub use gpu::{device::Device, memory::MemoryBudget, pass, system};
use market::graphs::MarketGraphs;
use market::{FreightActivity, FreightExecution, MarketExecution, MarketReplay, Provisioning};
use parallel::reduction;
use registry::RegistryAudit;
#[cfg(test)]
use registry::RegistryExecution;
#[cfg(test)]
use reporting::observation::AuditExecution;
use runtime::checkpoint::CopyDirection;
#[cfg(test)]
use runtime::lifecycle::ClaimCapacityDemand;
use runtime::profiling::profiling_day;
use storage::buffers::{self, DeviceEconomy};
use storage::layout::{decode, encode};

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
    config: economy::configuration::Configuration,
    spatial: CubeTensor,
    work: buffers::PhaseBuffers,
    searches: orders::SearchBuffers,
    market_sort: economy::market::sorting::MarketSort,
    market_graphs: MarketGraphs,
    market_replay: MarketReplay,
    freight_activity: FreightActivity,
    freight_execution: FreightExecution,
    market_execution: MarketExecution,
    control: CubeTensor,
    summary: CubeTensor,
    audit: economy::accounting::audit::AuditBuffers,
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

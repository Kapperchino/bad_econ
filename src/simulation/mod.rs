pub mod analytics;
pub mod bargaining;
mod configuration;
mod demography;
mod economy;
mod enterprise;
pub mod finance;
mod flags;
pub mod households;
mod layout;
mod market_graphs;
mod memory;
pub mod negotiation;
mod ops;
mod parallel;
mod parameters;
pub mod pass;
mod payroll;
mod policy;
mod profiling;
mod random;
mod registry;
pub mod research;
mod rows;
pub mod system;
mod tensor;
mod words;
use crate::{
    arithmetic::add,
    domain::*,
    ledger::{Bundle, Decision, MAX_LEGS, OrderedEvents, TickLayout},
    runtime::{PhaseTiming, RuntimeState, SimulationCommand},
};
use anyhow::{Context, Result, anyhow};
use burn_wgpu::{CubeBackend, CubeTensor};
use bytemuck::Zeroable;
use economy::{EconomyStage, markets, railway::settlement, scheduler};
use enterprise::{management, ownership};
use flags::{
    CONTROL_WORDS, ControlLayout, DependencyShape, EventStatus, FaultFlags, SchedulerControl,
};
use layout::*;
pub use memory::MemoryBudget;
pub use ops::Device;
use ops::{EconomyBuffers, EconomyOps, SchedulerBuffers};
use parallel::reduction;
use parameters::{Discovery, EconomyParameters, SchedulerParameters};
use policy::MATCHING_ROUNDS;
use registry::records::PendingRecord;
use scheduler::SchedulerStage;
use std::time::Instant;
use system::{AsBufferArg, System};

const MILLISECONDS_PER_SECOND: f64 = 1000.0;

fn profiling_day() -> u64 {
    static DAY: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *DAY.get_or_init(|| match std::env::var("BAD_ECON_PROFILE_DAY") {
        Ok(day) => day
            .parse::<std::num::NonZeroU64>()
            .expect("BAD_ECON_PROFILE_DAY must be a positive day number")
            .get(),
        Err(std::env::VarError::NotPresent) => 30,
        Err(error) => panic!("BAD_ECON_PROFILE_DAY: {error}"),
    })
}

#[derive(Clone, Copy)]
enum MarketExecution {
    Grouped,
    #[cfg(test)]
    Original,
    #[cfg(test)]
    Reference,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum MarketReplay {
    Reusable,
    RegistryGrowth,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum FreightExecution {
    Synchronous,
    Buffered,
}

impl FreightExecution {
    fn initial(self, trace: Trace) -> settlement::Execution {
        match (trace, self) {
            (Trace::Discard, Self::Buffered) => {
                settlement::Execution::Buffered(settlement::Wave::Initial)
            }
            _ => settlement::Execution::Immediate(settlement::Wave::Initial),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum AdvanceAttempt {
    Pending,
    Complete,
}

#[derive(Debug)]
struct ClaimCapacityDemand {
    records: std::num::NonZeroU64,
}

impl std::fmt::Display for ClaimCapacityDemand {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "freight claim storage requires {} records",
            self.records
        )
    }
}

impl std::error::Error for ClaimCapacityDemand {}

impl MarketReplay {
    fn clock(self, program: market_graphs::MarketProgram) -> parameters::MarketClock {
        match (self, program) {
            (Self::Reusable, _)
            | (
                Self::RegistryGrowth,
                market_graphs::MarketProgram::Consumer(_) | market_graphs::MarketProgram::Input,
            ) => parameters::MarketClock::Device,
            _ => parameters::MarketClock::Scalars,
        }
    }
}

enum RegistryChange {
    Unchanged,
    Births,
    Transitions(registry::RegistryPlan),
    Entries(registry::entries::EntryGrowth),
    Rebuild,
}

#[derive(Clone, Copy)]
enum RegistryAudit {
    #[cfg(test)]
    Full,
    Validated,
}

#[cfg(test)]
#[derive(Clone, Copy)]
enum AuditExecution {
    Tiled,
    Original,
}

#[cfg(test)]
#[derive(Clone, Copy)]
enum RegistryExecution {
    Gpu,
    Reference,
}

#[cfg(test)]
mod tests;

enum MatchingState {
    Searching,
    Exhausted,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Provisioning {
    Standard,
    Minimum,
    Choice,
}

impl Provisioning {
    fn rounds(self) -> usize {
        match self {
            Self::Standard => MATCHING_ROUNDS,
            Self::Minimum => MATCHING_ROUNDS - 1,
            Self::Choice => 1,
        }
    }
}

struct ProvisionMarket {
    market: Market,
    provisioning: Provisioning,
}

impl ProvisionMarket {
    fn sequence(scenario: &Scenario) -> Vec<Self> {
        let markets = scenario.goods_markets();
        match scenario.config().food.map(|food| food.income) {
            Some(crate::domain::food::FoodIncome::PurchasingPower) => {
                let necessities = [Provisioning::Minimum, Provisioning::Choice]
                    .into_iter()
                    .flat_map(|provisioning| {
                        [
                            Market::Household,
                            Market::Consumer(Good::Coal),
                            Market::Consumer(Good::Wood),
                        ]
                        .into_iter()
                        .map(move |market| Self {
                            market,
                            provisioning,
                        })
                    });
                markets.iter().copied().filter(|market| matches!(market, Market::Input(_)))
                    .map(|market| Self { market, provisioning: Provisioning::Standard })
                    .chain(necessities)
                    .chain(markets.iter().copied().filter(|market| matches!(market, Market::Consumer(good) if !matches!(good, Good::Coal | Good::Wood)))
                        .map(|market| Self { market, provisioning: Provisioning::Standard }))
                    .collect()
            }
            _ => markets
                .into_iter()
                .map(|market| Self {
                    market,
                    provisioning: Provisioning::Standard,
                })
                .collect(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FreightActivity {
    Active,
    Dormant,
    #[cfg(test)]
    Unconditional,
}

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
    audited: CubeTensor,
    snapshot_copy: tensor::SnapshotCopy,
    actors: CubeTensor,
    config: configuration::Configuration,
    spatial: CubeTensor,
    proposals: CubeTensor,
    events: CubeTensor,
    scratch: CubeTensor,
    searches: markets::SearchBuffers,
    market_sort: economy::sorting::MarketSort,
    market_graphs: market_graphs::MarketGraphs,
    market_replay: MarketReplay,
    freight_activity: FreightActivity,
    freight_execution: FreightExecution,
    #[cfg(test)]
    links: CubeTensor,
    market_execution: MarketExecution,
    control: CubeTensor,
    summary: CubeTensor,
    audit: economy::audit::AuditBuffers,
    #[cfg(test)]
    audit_execution: AuditExecution,
    feedback: CubeTensor,
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
    #[cfg(feature = "viewer")]
    pub(crate) fn open_firms(&mut self) -> Result<Vec<Actor>> {
        let actors = &self.metadata.actors;
        let start = actors.firm_indices().next().unwrap_or(actors.len());
        let statuses = decode(&self.device.read_range(
            &self.state,
            (Field::EnterpriseStatus as usize * self.config.state_stride as usize + start)
                * TICK_WORDS,
            (actors.len() - start) * TICK_WORDS,
        )?);
        self.readback_bytes += (statuses.len() * TICK_BYTES) as u64;
        Ok(actors
            .firm_indices()
            .filter(|&index| {
                statuses[index - start] != crate::domain::enterprise::EnterpriseState::Closed as u64
            })
            .map(|index| actors.at(index))
            .collect())
    }
    pub fn memory_usage(&self) -> cubecl::MemoryUsage {
        self.device.memory_usage()
    }
    pub fn new(scenario: Scenario, device: Device) -> Result<Self> {
        match scenario.memory_estimate() <= device.memory_budget.bytes() {
            true => Ok(()),
            false => Err(anyhow!(
                "scenario needs an estimated {} bytes, exceeding the device memory budget of {} bytes",
                scenario.memory_estimate(),
                device.memory_budget.bytes()
            )),
        }?;
        let initial = scenario.initial()?;
        Self::from_state(scenario, initial, device)
    }
    pub fn from_checkpoint(checkpoint: Checkpoint, device: Device) -> Result<Self> {
        let restored = checkpoint.restore()?;
        Self::from_state(restored.scenario, restored.state, device)
    }
    fn from_state(scenario: Scenario, mut initial: State, mut device: Device) -> Result<Self> {
        unsafe {
            device.client.set_stream(burn::tensor::StreamId::current());
        }
        let required = scenario.memory_estimate().max(
            initial.values.len() as u64 * 24
                + initial.people.len() as u64 * 144
                + initial.actors.len() as u64 * 160,
        );
        match required <= device.memory_budget.bytes() {
            true => Ok(()),
            false => Err(anyhow!(
                "scenario needs an estimated {} bytes, exceeding the device memory budget of {} bytes",
                required,
                device.memory_budget.bytes()
            )),
        }?;
        device.reclaim_memory(required)?;
        let _ = AuditedState::new(&scenario, &initial)?;
        let snapshot = ObservationSnapshot::default();
        let demography = demography::DemographyRuntime::new(&device, &scenario, &mut initial)?;

        let mut config = configuration::Configuration::new(&device, &scenario, &initial);
        let count = initial.actors.len();
        let stride = match (count, scenario.config().finance.as_ref()) {
            (4096.., None) => registry::entries::state_capacity(count),
            _ => count,
        };
        config.state_stride = stride as u32;
        let actor_layouts = initial
            .actors
            .iter()
            .map(|actor| ActorLayout::new(&actor, &scenario))
            .collect::<ActorMetadataRows>();
        let actors = device.tensor(&actor_layouts.words());
        let spatial = device.tensor(&SpatialIndex::new(&scenario, &initial).words);
        let state = {
            let values = tensor::state_values(&initial.values, count, stride);
            #[cfg(target_endian = "little")]
            let state = device.tensor(bytemuck::cast_slice(&values));
            #[cfg(target_endian = "big")]
            let state = device.tensor(bytemuck::cast_slice(&encode(&values)));
            state
        };
        let state_words = state.meta.num_elements();
        let audited = device.state_buffer(state_words);
        let snapshot_copy = tensor::SnapshotCopy::new(&device, state_words);
        let research = scenario
            .config()
            .research
            .as_ref()
            .map(|config| match &initial.research {
                Some(snapshot) => research::ResearchRuntime::restore(
                    &device, config, &scenario, &initial, snapshot,
                ),
                None => research::ResearchRuntime::new(&device, config, &scenario, &initial),
            })
            .transpose()?
            .map(|research| research.with_actors(&initial.actors, stride as u32))
            .transpose()?;
        let finance = scenario
            .config()
            .finance
            .as_ref()
            .map(|config| {
                let actors = initial
                    .actors
                    .iter()
                    .enumerate()
                    .map(|(index, actor)| finance::FinancialActor {
                        identity: actor.identity,
                        cash: initial.coordinate(Field::Cash, index),
                        kind: match actor.kind {
                            0 => finance::FinancialActorKind::Household,
                            _ => finance::FinancialActorKind::Firm,
                        },
                    })
                    .collect::<Vec<_>>();
                finance::FinanceRuntime::new(
                    &device,
                    config,
                    config.layout(initial.actors.len() * scenario.fields())?,
                    &actors,
                )
            })
            .transpose()?;
        let migrations = registry::MigrationRegistry::new(&initial);
        let management = management::EnterpriseManagement::new(&device, &initial);
        initial.values.clear();
        initial.values.shrink_to_fit();
        initial.research = None;
        initial.people.clear();
        initial.people.shrink_to_fit();
        let event_words =
            initial.actors.len().div_ceil(economy::audit::TILE_ACTORS) * SummaryLayout::WORDS;
        let scratch_words = markets::scratch_words(initial.actors.len());
        #[cfg(test)]
        let batch = (scenario.config().batch_size as usize).min(initial.actors.len());
        #[cfg(test)]
        let event_words = event_words.max(batch * EVENT_WORDS).max(
            initial
                .actors
                .len()
                .div_ceil(SummaryLayout::ACTORS_PER_GROUP as usize)
                * SummaryLayout::WORDS,
        );
        #[cfg(test)]
        let scratch_words = scratch_words.max(batch * ScratchLayout::WORDS_PER_EVENT);
        let events = device.empty(event_words);
        let scratch = device.empty(scratch_words);
        #[cfg(test)]
        let links = device.empty(batch * LinkLayout::WORDS_PER_EVENT);
        let control = device.tensor(bytemuck::cast_slice(&[ObservationControlLayout::zeroed()]));
        let proposals = device.empty(initial.actors.len() * ProposalLayout::WORDS);
        let summary = device.empty((scenario.fields() * TICK_WORDS).max(SummaryLayout::WORDS));
        let feedback = device.tensor(&vec![0; CONTROL_WORDS + initial.actors.len() * TICK_WORDS]);
        let ownership = Some(ownership::OwnershipRuntime::new(&device, &initial)?);
        let enterprise_requests = device.tensor(&vec![0; initial.actors.len()]);
        let day = initial.day;
        let audit = economy::audit::AuditBuffers::new(&device, &initial.actors);
        let searches = markets::SearchBuffers::new(&device, &initial.actors);
        let mut simulation = Self {
            scenario,
            device,
            status: RuntimeState::Paused,
            snapshot,
            timings: Vec::new(),
            readback_bytes: 0,
            metadata: initial,
            day,
            audited_day: day,
            next_phase: Phase::Opening,
            state,
            audited,
            snapshot_copy,
            actors,
            config,
            spatial,
            proposals,
            events,
            scratch,
            searches,
            market_sort: economy::sorting::MarketSort::default(),
            market_graphs: market_graphs::MarketGraphs::default(),
            market_replay: MarketReplay::Reusable,
            freight_activity: FreightActivity::Active,
            freight_execution: FreightExecution::Synchronous,
            #[cfg(test)]
            links,
            market_execution: MarketExecution::Grouped,
            control,
            summary,
            audit,
            #[cfg(test)]
            audit_execution: AuditExecution::Tiled,
            feedback,
            finance,
            research,
            research_readback: 0,
            demography,
            enterprise_requests,
            ownership,
            management,
            migrations,
            #[cfg(test)]
            registry_execution: RegistryExecution::Gpu,
            matching_round: 0,
            provisioning: Provisioning::Standard,
        };
        simulation.snapshot = simulation.observe()?;
        simulation.copy_audited(CopyDirection::Save)?;
        simulation.device.client.memory_cleanup()?;
        match simulation.device.reclaim_memory(0)?.bytes_reserved
            <= simulation.device.memory_budget.bytes()
        {
            true => Ok(simulation),
            false => Err(anyhow!(
                "simulation memory exceeded the device memory budget of {} bytes",
                simulation.device.memory_budget.bytes()
            )),
        }
    }
    fn parameters(
        &self,
        phase: Phase,
        market: Market,
        offset: usize,
        count: usize,
    ) -> Result<EconomyParameters> {
        let day = u32::try_from(self.day).context("day identity capacity exhausted")?;
        match phase {
            Phase::Audit => (),
            _ => {
                let _ = day
                    .checked_add(self.scenario.ring() as u32 + 1)
                    .context("delivery date capacity exhausted")?;
            }
        }
        Ok(EconomyParameters {
            clock: crate::simulation::parameters::MarketClock::Scalars,
            scan: parallel::ScanMode::new(&self.device),
            empty_orders: parameters::EmptyOrderMode::Settle,
            day,
            matching_round: match self.provisioning {
                Provisioning::Standard => self.matching_round as u32,
                Provisioning::Minimum => 0,
                Provisioning::Choice => MATCHING_ROUNDS as u32 - 1,
            },
            settlement_rotation: (crate::arithmetic::DrawKey {
                seed: self.scenario.config().seed,
                day: self.day,
                phase: phase as u32,
                actor: 0,
                purpose: 0x5e771e,
                index: u64::from(market.code()) * MATCHING_ROUNDS as u64
                    + self.matching_round as u64,
            }
            .draw()
                % self.metadata.actors.len().max(1) as u64) as u32,
            phase,
            market: market.into(),
            good: match market {
                Market::Input(good) | Market::Consumer(good) => good as u32,
                Market::Household => Good::Bread as u32,
                _ => 0,
            },
            discovery: match self.market_execution {
                MarketExecution::Grouped => Discovery::Indexed,
                #[cfg(test)]
                MarketExecution::Original => Discovery::Original,
                #[cfg(test)]
                MarketExecution::Reference => Discovery::Linear,
            },
            offset: offset as u32,
            count: count as u32,
            status_start: 0,
            sort_source: 0,
            sort_width: 0,
            sort_count: 0,
        })
    }
    fn economy(
        &self,
        entry: EconomyStage,
        count: usize,
        parameters: &EconomyParameters,
        output: &CubeTensor,
    ) {
        CubeBackend::economy(
            &self.device,
            entry,
            count,
            EconomyBuffers {
                state: &self.state,
                actors: &self.actors,
                config: &self.config,
                spatial: &self.spatial,
                proposals: output,
                events: &self.events,
                control: &self.control,
            },
            parameters,
        );
    }
    fn fault_check(&mut self) -> Result<()> {
        let control = self.device.read_range(&self.control, 0, CONTROL_WORDS)?;
        self.readback_bytes += (CONTROL_WORDS * WORD_BYTES) as u64;
        self.fault_result(control[flags::FAULT_WORD])
    }
    fn fault_result(&mut self, bits: u32) -> Result<()> {
        let mut faults = FaultFlags::new(bits)?;
        let capacity = faults.contains(FaultFlags::CLAIM_CAPACITY);
        faults.remove(FaultFlags::CLAIM_CAPACITY);
        faults.into_result()?;
        match capacity {
            true => {
                let words =
                    self.device
                        .read_range(&self.control, flags::CLAIM_CAPACITY_LOW_WORD, 2)?;
                self.readback_bytes += (2 * WORD_BYTES) as u64;
                let records =
                    std::num::NonZeroU64::new(u64::from(words[0]) | (u64::from(words[1]) << 32))
                        .context("freight capacity demand is empty")?;
                Err(ClaimCapacityDemand { records }.into())
            }
            false => Ok(()),
        }
    }
    fn measure_freight_activity(&mut self) -> Result<FreightActivity> {
        match self.scenario.config().railway.companies.len() {
            0 => Ok(FreightActivity::Dormant),
            _ => {
                let count = self.metadata.actors.len();
                let first = self.metadata.actors.firm_indices().next().unwrap_or(count);
                let words = self.device.read_range(
                    &self.state,
                    (Field::RailOperating as usize * self.config.state_stride as usize + first)
                        * TICK_WORDS,
                    (count - first) * TICK_WORDS,
                )?;
                self.readback_bytes += (words.len() * WORD_BYTES) as u64;
                Ok(match words.into_iter().any(|word| word != 0) {
                    true => FreightActivity::Active,
                    false => FreightActivity::Dormant,
                })
            }
        }
    }
    pub fn phase(&mut self, phase: Phase, trace: Trace) -> Result<PhaseEvidence> {
        match (&self.status, self.next_phase == phase) {
            (RuntimeState::Fault { .. }, _) => {
                Err(anyhow!("reset or restore the faulted simulation"))
            }
            (_, false) => Err(anyhow!(
                "expected {:?}, received {phase:?}",
                self.next_phase
            )),
            (_, true) => Ok(()),
        }?;
        let previous = self.snapshot.clone();
        self.status = RuntimeState::Advancing(phase);
        let result = self
            .device
            .reclaim_memory(0)
            .and_then(|_| self.execute_phase(phase, trace))
            .and_then(|evidence| {
                let memory = self.device.reclaim_memory(0)?;
                match memory.bytes_reserved <= self.device.memory_budget.bytes() {
                    true => Ok(evidence),
                    false => Err(anyhow!(
                        "simulation memory exceeded the device memory budget of {} bytes during {phase:?}: {} active, {} reserved",
                        self.device.memory_budget.bytes(),
                        memory.bytes_in_use,
                        memory.bytes_reserved
                    )),
                }
            });
        match result {
            Ok(evidence) => {
                if let Phase::Audit = phase {
                    self.copy_audited(CopyDirection::Save)?;
                    self.audited_day = self.day;
                    self.status = RuntimeState::Running;
                }
                self.next_phase = phase.next();
                Ok(evidence)
            }
            Err(error) => {
                self.day = self.audited_day;
                self.snapshot = previous;
                self.next_phase = Phase::Opening;
                self.status = RuntimeState::Fault {
                    last_audited_day: self.audited_day,
                    message: error.to_string(),
                };
                self.copy_audited(CopyDirection::Restore)?;
                Err(error)
            }
        }
    }
    fn execute_phase(&mut self, phase: Phase, trace: Trace) -> Result<PhaseEvidence> {
        self.config.heating_per_person =
            crate::domain::calendar::Weather::on_day(self.day).heating_units_per_person;
        self.demography.before(
            &self.device,
            &self.state,
            &self.actors,
            &self.config,
            phase,
            self.day,
        )?;
        if let Phase::Observation = phase {
            self.management
                .assign(&self.device, &self.state, &self.config, self.day as u32);
        }
        let mut evidence = PhaseEvidence::default();
        self.device.write(
            &self.control,
            bytemuck::cast_slice(&[ObservationControlLayout::zeroed()]),
        );
        if let Some(research) = &self.research {
            research.before_phase(&self.device, &self.state, phase, self.day)?;
        }
        if let Some(finance) = &self.finance {
            match phase {
                Phase::Opening | Phase::Proposal => {
                    let stage = match phase {
                        Phase::Opening => finance::FinanceStage::Opening,
                        _ => finance::FinanceStage::Lending,
                    };
                    finance.phase(&self.device, &self.state, self.day as u32, stage)?;
                    self.readback_bytes += finance.phase_readback_bytes();
                    evidence.events += finance.last_event_count();
                    if let Trace::Record = trace {
                        let decisions = finance.decisions(&self.device)?;
                        evidence.decisions.extend(decisions);
                        self.readback_bytes += finance.decision_readback_bytes();
                    }
                }
                Phase::ProvisionPurchase => {
                    finance.support(&self.device, &self.state)?;
                    self.readback_bytes += finance.support_readback_bytes();
                    evidence.events += finance.last_event_count();
                    if let Trace::Record = trace {
                        evidence.decisions.extend(finance.support_decisions());
                    }
                }
                _ => (),
            }
        }
        match phase {
            Phase::Commitment => {
                std::iter::once(policy::RETAINED_LABOR_ROUND)
                    .chain(0..MATCHING_ROUNDS)
                    .try_for_each(|round| {
                        self.matching_round = round;
                        self.market(Market::Labor, phase, trace, &mut evidence)
                            .map(|_| ())
                    })?;
                self.market_result(&mut evidence)
            }
            Phase::WorkPayment => {
                self.market(Market::Payroll, phase, trace, &mut evidence)?;
                self.market_result(&mut evidence)?;
                self.local(phase)
            }
            Phase::ProvisionPurchase => {
                self.railways(economy::railway::RailStage::Access);
                self.railways(economy::railway::RailStage::Routes);
                self.railways(economy::railway::RailStage::Quotes);
                self.freight_activity = match self.freight_activity {
                    #[cfg(test)]
                    FreightActivity::Unconditional => FreightActivity::Unconditional,
                    _ => self.measure_freight_activity()?,
                };
                if std::env::var_os("BAD_ECON_PROFILE_GRAPHS").is_some() {
                    eprintln!(
                        "FREIGHT_ACTIVITY {} {:?}",
                        self.day + 1,
                        self.freight_activity
                    );
                }
                let heating = self.config.heating_per_person;
                let availability = match self.freight_activity {
                    FreightActivity::Dormant => economy::railway::RouteAvailability::Dormant,
                    _ => economy::railway::RouteAvailability::Available,
                };
                let routing = self.config.railway.replace_route_availability(availability);
                let result = ProvisionMarket::sequence(&self.scenario)
                    .into_iter()
                    .filter(|request| match request.market {
                        Market::Consumer(Good::Coal | Good::Wood) => heating > 0,
                        _ => true,
                    })
                    .try_for_each(|request| {
                        self.provisioning = request.provisioning;
                        let market = request.market;
                        let rounds = request.provisioning.rounds();
                        (0..rounds).try_fold(MatchingState::Searching, |state, round| {
                            match (state, request.provisioning, market, round + 1 == rounds) {
                                (MatchingState::Searching, _, _, _)
                                | (
                                    MatchingState::Exhausted,
                                    Provisioning::Standard,
                                    Market::Household | Market::Consumer(Good::Coal | Good::Wood),
                                    true,
                                ) => {
                                    self.matching_round = round;
                                    self.market(market, phase, trace, &mut evidence)
                                }
                                _ => Ok(MatchingState::Exhausted),
                            }
                        })?;
                        self.matching_round = MATCHING_ROUNDS - 1;
                        if let (
                            Market::Household,
                            Some(_),
                            Provisioning::Standard | Provisioning::Choice,
                        ) = (market, self.scenario.config().rent, self.provisioning)
                        {
                            let parameters =
                                self.parameters(phase, market, 0, self.metadata.actors.len())?;
                            self.economy(
                                EconomyStage::HouseholdRent,
                                self.metadata.actors.len(),
                                &parameters,
                                &self.proposals,
                            );
                            self.fault_check()?;
                        }
                        Ok::<(), anyhow::Error>(())
                    });
                self.provisioning = Provisioning::Standard;
                self.config.railway.replace_route_availability(routing);
                result?;
                self.market_result(&mut evidence)?;
                self.apply_feedback()?;
                self.market_replay = MarketReplay::Reusable;
                Ok(())
            }
            Phase::AccountsDecisions => {
                self.local(phase)?;
                match &self.ownership {
                    Some(ownership) => {
                        ownership.distribute(&self.device, &self.state, &self.actors, &self.config)
                    }
                    None => {
                        self.market(Market::Distribution, phase, trace, &mut evidence)?;
                        self.market_result(&mut evidence)
                    }
                }
            }
            Phase::Audit => {
                let started = Instant::now();
                self.railways(economy::railway::RailStage::Close);
                self.fault_check()?;
                if let Some(research) = &self.research {
                    research.phase(&self.device, &self.state, phase, self.day)?;
                }
                self.snapshot = self.observe()?;
                if self.day + 1 == profiling_day()
                    && std::env::var_os("BAD_ECON_PROFILE_STAGES").is_some()
                {
                    eprintln!(
                        "AUDIT {} InitialObservation {:.3}",
                        self.day + 1,
                        started.elapsed().as_secs_f64() * 1000.0
                    );
                }
                self.day = add(self.day, 1)?;
                self.snapshot.day = self.day;
                self.snapshot.weather = self.scenario.weather(self.day);
                self.market_graphs.report(self.day);
                self.commit_enterprises(RegistryAudit::Validated)?;
                Ok(())
            }
            _ => self.local(phase),
        }?;
        self.demography.after(
            &self.device,
            &self.state,
            &self.actors,
            &self.config,
            phase,
            self.day,
        )?;
        match (&self.research, phase) {
            (_, Phase::Audit) | (None, _) => (),
            (Some(research), _) => research.phase(&self.device, &self.state, phase, self.day)?,
        }
        self.account_research_readbacks();
        Ok(evidence)
    }
    fn grow_births(&self, births: &[PendingRecord]) -> Result<demography::DemographyRuntime> {
        #[cfg(test)]
        let grown = match self.registry_execution {
            RegistryExecution::Reference => {
                let mut dense = vec![0; self.metadata.actors.len()];
                for record in births {
                    dense[record.actor as usize] = record.birth;
                }
                self.demography
                    .reference_births(&self.device, &dense, self.metadata.next_identity)
            }
            RegistryExecution::Gpu => self.demography.grow_births(
                &self.device,
                births,
                self.metadata.next_identity,
                &self.scratch,
            ),
        };
        #[cfg(not(test))]
        let grown = self.demography.grow_births(
            &self.device,
            births,
            self.metadata.next_identity,
            &self.scratch,
        );
        grown
    }
    fn commit_enterprises(&mut self, audit: RegistryAudit) -> Result<()> {
        let mut checkpoint = profiling::audit_checkpoints(&self.device, self.day);
        let records = registry::records::RecordBuffers {
            requests: &self.enterprise_requests,
            births: self.demography.birth_buffer(),
            scratch: &self.scratch,
        };
        enterprise::propose(
            &self.device,
            &self.state,
            &self.actors,
            &self.config,
            &self.spatial,
            records,
            self.day as u32,
        );
        let pending = registry::records::PendingRecords::read(
            &self.device,
            records,
            &self.searches.requests,
        )?;
        self.readback_bytes += pending.readback_bytes;
        #[cfg(test)]
        let pending = match self.registry_execution {
            RegistryExecution::Gpu => pending,
            RegistryExecution::Reference => registry::records::PendingRecords::from_dense(
                &self.device.read(&self.enterprise_requests)?,
                &self.device.read(self.demography.birth_buffer())?,
            )?,
        };
        checkpoint("PendingRecords");
        let records = &pending.records;
        if std::env::var_os("BAD_ECON_PROFILE_REGISTRY").is_some() {
            let requests = records.iter().filter(|record| record.request > 0).fold(
                std::collections::BTreeMap::<u32, usize>::new(),
                |mut counts, record| {
                    *counts.entry(record.request >> 24).or_default() += 1;
                    counts
                },
            );
            eprintln!("REGISTRY {} Requests {requests:?}", self.day);
        }
        let count = records.iter().filter(|record| record.request > 0).count();
        let newborns = records.iter().filter(|record| record.birth > 0).count();
        let change = match (count, newborns) {
            (0, 0) => RegistryChange::Unchanged,
            (0, _) => RegistryChange::Births,
            _ => {
                let plan = registry::RegistryPlan::new(
                    &self.scenario,
                    &self.metadata,
                    &self.migrations,
                    records,
                )?;
                checkpoint("RegistryPlan");
                match plan {
                    Some(plan) => RegistryChange::Transitions(plan),
                    None => {
                        let growth = registry::entries::EntryGrowth::new(
                            &self.scenario,
                            &self.metadata,
                            &self.migrations,
                            records,
                        )?;
                        checkpoint("EntryGrowth");
                        growth.map_or(RegistryChange::Rebuild, RegistryChange::Entries)
                    }
                }
            }
        };
        #[cfg(test)]
        let change = match (self.registry_execution, change) {
            (
                RegistryExecution::Reference,
                RegistryChange::Transitions(_) | RegistryChange::Entries(_),
            ) => RegistryChange::Rebuild,
            (_, change) => change,
        };
        match change {
            RegistryChange::Unchanged => Ok(()),
            RegistryChange::Births => self.commit_births(records, newborns),
            RegistryChange::Transitions(plan) => self.commit_registry_audited(plan, records, audit),
            RegistryChange::Entries(growth) => {
                self.commit_entries(&growth.requests, growth.plan, records, audit)
            }
            RegistryChange::Rebuild => self.rebuild_enterprises(records, count, newborns),
        }
    }

    fn commit_births(&mut self, records: &[PendingRecord], newborns: usize) -> Result<()> {
        let next_identity = self
            .metadata
            .next_identity
            .checked_add(newborns as u64)
            .context("identity capacity exhausted")?;
        let grown = self.grow_births(records)?;
        grown.remap_workers_from(
            &self.device,
            &self.state,
            self.config.state_stride,
            &self.demography,
        );
        let population = grown.observation(&self.device, &self.state, &self.config)?;
        match self.device.reclaim_memory(0)?.bytes_reserved <= self.device.memory_budget.bytes() {
            true => (),
            false => Err(anyhow!(
                "person registry capacity exhausted while observing growth"
            ))?,
        }
        self.demography = grown;
        self.metadata.next_identity = next_identity;
        self.snapshot.population = population;
        Ok(())
    }

    fn rebuild_enterprises(
        &mut self,
        records: &[PendingRecord],
        count: usize,
        newborns: usize,
    ) -> Result<()> {
        let mut state = self.download()?;
        let estimated = (state.actors.len() + count) as u64 * self.scenario.fields() as u64 * 24
            + (state.people.len() + newborns) as u64 * 144;
        match estimated <= self.device.memory_budget.bytes() {
            true => (),
            false => Err(anyhow!(
                "registry capacity exhausted: {estimated} bytes exceed device budget"
            ))?,
        }
        for record in records.iter().filter(|record| record.birth > 0) {
            let identity = state.next_identity;
            state.next_identity = identity
                .checked_add(1)
                .context("identity capacity exhausted")?;
            state.people.push(crate::domain::demography::Person {
                identity,
                household: state.actors.at(record.actor as usize).identity,
                age_days: 0,
                sex: match record.birth {
                    1 => crate::domain::demography::Sex::Female,
                    _ => crate::domain::demography::Sex::Male,
                },
                state: crate::domain::demography::LivingState::Living,
                employer: None,
                work_minutes: 0,
                fatigue: 0,
                famine_exposure: 0,
                cold_exposure: 0,
                birth_cooldown: 0,
            });
        }
        let mut requests = vec![0; state.actors.len()];
        for record in records {
            requests[record.actor as usize] = record.request;
        }
        enterprise::enter(&mut state, &self.scenario, &requests)?;
        enterprise::transitions(&mut state, &self.scenario, &requests)?;
        let timings = self.timings.clone();
        let readback = self.readback_bytes;
        self.replace(self.scenario.clone(), state)?;
        self.timings = timings;
        self.readback_bytes += readback;
        Ok(())
    }
    fn commit_entries(
        &mut self,
        entries: &registry::entries::EntryRequests,
        mut plan: registry::RegistryPlan,
        births: &[PendingRecord],
        registry_audit: RegistryAudit,
    ) -> Result<()> {
        let mut checkpoint = profiling::audit_checkpoints(&self.device, self.day);
        self.market_graphs = market_graphs::MarketGraphs::default();
        checkpoint("ReleaseGraphs");
        let newborns = births.iter().filter(|record| record.birth > 0).count();
        let next_identity = self
            .metadata
            .next_identity
            .checked_add(newborns as u64)
            .context("identity capacity exhausted")?;
        let batch = registry::entries::EntryBatch::new(
            &self.device,
            &self.scenario,
            &self.metadata,
            &self.state,
            entries,
            &mut plan,
            next_identity,
        )?;
        checkpoint("EntryBatch");
        let ring_indices = batch.indices().clone();
        let count = batch.metadata.actors.len();
        let stride = batch.stride(self.config.state_stride as usize);
        let state_bytes = match stride == self.config.state_stride as usize {
            true => 0,
            false => stride as u64 * self.scenario.fields() as u64 * 16,
        };
        let additional = state_bytes + count as u64 * 192;
        let reserved = self.device.reclaim_memory(additional)?.bytes_reserved;
        let required = reserved.saturating_add(additional);
        let budget = self.device.memory_budget.bytes();
        match required <= budget {
            true => (),
            false => Err(anyhow!(
                "registry growth exceeds the device memory budget of {budget} bytes: {reserved} reserved and {additional} estimated for growth"
            ))?,
        }
        let state = batch.grow(&self.device, &self.state, self.scenario.fields());
        checkpoint("GrowState");
        let settlement = registry::settlement::SettlementBatch::new(
            &self.device,
            &self.scenario,
            &batch.metadata,
            &state,
            &entries.transitions,
            self.day,
            &plan.migrations,
        )?;
        checkpoint("SettlementBatch");
        let mut metadata = batch.metadata;
        if let Some(mut actors) = plan.actors {
            actors.extend(metadata.actors.iter_from(actors.len()));
            metadata.actors = actors;
        }
        if let Some(batch) = &settlement {
            batch.update_owners(&mut metadata.actors);
        }
        if let Some(ownership) = settlement
            .as_ref()
            .and_then(|batch| batch.ownership.as_ref())
        {
            metadata.ownership = ownership.clone();
        }
        let rental_changes = plan
            .changes
            .iter()
            .filter(|record| record.request >> 24 == 3)
            .map(|record| record.actor as usize)
            .chain(
                settlement
                    .iter()
                    .flat_map(|batch| batch.owners.iter().map(|change| change.firm)),
            )
            .collect::<Vec<_>>();
        let started = Instant::now();
        let mut config = self.config.updated_for_registry(
            &self.device,
            &self.scenario,
            &self.metadata.actors,
            &metadata,
            &rental_changes,
        );
        config.state_stride = stride as u32;
        if std::env::var_os("BAD_ECON_PROFILE_STAGES").is_some() {
            eprintln!(
                "CONFIG {} {:.3}",
                self.day,
                started.elapsed().as_secs_f64() * 1000.0
            );
        }
        checkpoint("Configuration");
        let additions = metadata
            .actors
            .iter_from(self.metadata.actors.len())
            .collect::<ActorRows>();
        let rows = additions
            .iter()
            .map(|actor| ActorLayout::new(&actor, &self.scenario))
            .collect::<ActorMetadataRows>();
        let previous_actors = registry::extend_actors(&self.device, &self.actors, &rows);
        let enterprise_requests =
            registry::records::upload_requests(&self.device, &entries.transitions, count);
        let actors = match plan.transition {
            registry::RegistryTransition::Unchanged => previous_actors.clone(),
            registry::RegistryTransition::Capacity(registry::CapacityUpdate::Patch) => {
                registry::patch_capacity(&self.device, &previous_actors, &enterprise_requests)
            }
            registry::RegistryTransition::Capacity(registry::CapacityUpdate::Rebuild)
            | registry::RegistryTransition::Migration => registry::metadata::patch(
                &self.device,
                &self.scenario,
                &previous_actors,
                &metadata.actors,
                &plan.changes,
            ),
        };
        let actors = match settlement.as_ref().filter(|batch| !batch.owners.is_empty()) {
            Some(batch) => registry::metadata::patch_actors(
                &self.device,
                &self.scenario,
                &actors,
                &metadata.actors,
                batch.owners.iter().map(|change| change.firm),
            ),
            None => actors,
        };
        checkpoint("ActorMetadata");
        let spatial = match plan.transition {
            registry::RegistryTransition::Migration => registry::spatial::migrate(
                &self.device,
                &self.scenario,
                registry::spatial::SpatialMigration {
                    spatial: &self.spatial,
                    actors: &self.actors,
                    previous: &self.metadata.actors,
                    current: &metadata.actors,
                    requests: &plan.changes,
                },
            ),
            _ => registry::spatial::extend(
                &self.device,
                &self.scenario,
                &self.spatial,
                &actors,
                &additions,
            ),
        };
        checkpoint("Spatial");
        let mut demography = match newborns {
            0 => self.demography.clone(),
            _ => self.grow_births(births)?,
        };
        match batch.formations.is_empty() {
            true => demography.extend_firms(&self.device, count),
            false => {
                demography = demography.form_households(&self.device, &batch.formations, count)?;
            }
        }
        checkpoint("Demography");
        let research = self
            .research
            .as_ref()
            .map(|research| research.with_actors(&metadata.actors, stride as u32))
            .transpose()?;
        let scratch_words = markets::scratch_words(count);
        let event_words = count.div_ceil(economy::audit::TILE_ACTORS) * SummaryLayout::WORDS;
        #[cfg(test)]
        let batch_size = (self.scenario.config().batch_size as usize).min(count);
        #[cfg(test)]
        let scratch_words = scratch_words.max(batch_size * ScratchLayout::WORDS_PER_EVENT);
        #[cfg(test)]
        let event_words = event_words
            .max(batch_size * EVENT_WORDS)
            .max(count.div_ceil(SummaryLayout::ACTORS_PER_GROUP as usize) * SummaryLayout::WORDS);
        let ownership = Some(ownership::OwnershipRuntime::new(&self.device, &metadata)?);
        let management = management::EnterpriseManagement::new(&self.device, &metadata);
        let audit = economy::audit::AuditBuffers::new(&self.device, &metadata.actors);
        checkpoint("Miscellaneous");
        let searches = markets::SearchBuffers::new(&self.device, &metadata.actors);
        let mut next = Self {
            scenario: self.scenario.clone(),
            device: self.device.clone(),
            status: self.status.clone(),
            snapshot: self.snapshot.clone(),
            timings: self.timings.clone(),
            readback_bytes: self.readback_bytes
                + batch.readback_bytes
                + settlement.as_ref().map_or(0, |batch| batch.readback_bytes),
            metadata,
            day: self.day,
            audited_day: self.audited_day,
            next_phase: self.next_phase,
            state,
            audited: match stride == self.config.state_stride as usize {
                true => self.audited.clone(),
                false => self
                    .device
                    .state_buffer(stride * self.scenario.fields() * TICK_WORDS),
            },
            snapshot_copy: match stride == self.config.state_stride as usize {
                true => self.snapshot_copy.clone(),
                false => tensor::SnapshotCopy::new(
                    &self.device,
                    stride * self.scenario.fields() * TICK_WORDS,
                ),
            },
            actors,
            config,
            spatial,
            proposals: self.device.empty(count * ProposalLayout::WORDS),
            events: self.device.empty(event_words),
            scratch: self.device.empty(scratch_words),
            searches,
            market_sort: economy::sorting::MarketSort::default(),
            market_graphs: market_graphs::MarketGraphs::default(),
            market_replay: match self.device.cube {
                cubecl::Device::Metal(_) => MarketReplay::Reusable,
                _ => MarketReplay::RegistryGrowth,
            },
            freight_activity: self.freight_activity,
            freight_execution: self.freight_execution,
            #[cfg(test)]
            links: self.device.empty(batch_size * LinkLayout::WORDS_PER_EVENT),
            market_execution: self.market_execution,
            control: self.control.clone(),
            summary: self.summary.clone(),
            audit,
            #[cfg(test)]
            audit_execution: self.audit_execution,
            feedback: self
                .device
                .tensor(&vec![0; CONTROL_WORDS + count * TICK_WORDS]),
            finance: None,
            research,
            research_readback: self.research_readback,
            demography,
            enterprise_requests,
            ownership,
            management,
            migrations: plan.migrations,
            #[cfg(test)]
            registry_execution: self.registry_execution,
            matching_round: self.matching_round,
            provisioning: self.provisioning,
        };
        checkpoint("Buffers");
        match &settlement {
            Some(batch) => batch.apply(&next.device, &next.state, stride),
            None => registry::apply(
                &next.device,
                &next.state,
                &previous_actors,
                &next.config,
                &next.enterprise_requests,
                &next.scratch,
                next.day as u32,
            ),
        }
        checkpoint("Apply");
        let rings = match (registry_audit, &settlement) {
            #[cfg(test)]
            (RegistryAudit::Full, _) => economy::audit::RingValidation::All,
            (RegistryAudit::Validated, None) => {
                economy::audit::RingValidation::Selected(&ring_indices)
            }
            (RegistryAudit::Validated, Some(settlement)) => {
                economy::audit::RingValidation::Changed {
                    entries: &ring_indices,
                    settlements: settlement.indices(),
                }
            }
        };
        next.demography.remap_workers_from(
            &next.device,
            &next.state,
            next.config.state_stride,
            &self.demography,
        );
        next.snapshot = next.observe_rings(rings)?;
        checkpoint("Observe");
        let reserved = next.device.reclaim_memory(0)?.bytes_reserved;
        match reserved <= budget {
            true => *self = next,
            false => Err(anyhow!(
                "registry allocation exceeds the device memory budget of {budget} bytes: {reserved} reserved"
            ))?,
        }
        checkpoint("RegistryCommit");
        Ok(())
    }
    #[cfg(test)]
    fn commit_registry(
        &mut self,
        plan: registry::RegistryPlan,
        births: &[PendingRecord],
    ) -> Result<()> {
        self.commit_registry_audited(plan, births, RegistryAudit::Full)
    }
    fn commit_registry_audited(
        &mut self,
        mut plan: registry::RegistryPlan,
        births: &[PendingRecord],
        audit: RegistryAudit,
    ) -> Result<()> {
        let mut checkpoint = profiling::audit_checkpoints(&self.device, self.day);
        let settlement = registry::settlement::SettlementBatch::new(
            &self.device,
            &self.scenario,
            &self.metadata,
            &self.state,
            births,
            self.day,
            &plan.migrations,
        )?;
        checkpoint("RegistrySettlement");
        let management = settlement
            .as_ref()
            .filter(|batch| !batch.owners.is_empty())
            .map(|batch| {
                let actors = plan
                    .actors
                    .get_or_insert_with(|| self.metadata.actors.clone());
                batch.update_owners(actors);
                let mut metadata = self.metadata.clone();
                metadata.actors = actors.clone();
                management::EnterpriseManagement::new(&self.device, &metadata)
            });
        let ownership = settlement
            .as_ref()
            .and_then(|batch| batch.ownership.as_ref())
            .map(|ownership| {
                let mut metadata = self.metadata.clone();
                metadata.ownership = ownership.clone();
                ownership::OwnershipRuntime::new(&self.device, &metadata)
            })
            .transpose()?;
        checkpoint("RegistryOwnership");
        let newborns = births.iter().filter(|record| record.birth > 0).count();
        let next_identity = self
            .metadata
            .next_identity
            .checked_add(newborns as u64)
            .context("identity capacity exhausted")?;
        let grown = match newborns {
            0 => None,
            _ => Some(self.grow_births(births)?),
        };
        checkpoint("RegistryBirths");
        let previous_actor_buffer = self.actors.clone();
        let previous_spatial = self.spatial.clone();
        let next_buffers = match plan.transition {
            registry::RegistryTransition::Unchanged => None,
            registry::RegistryTransition::Capacity(registry::CapacityUpdate::Patch) => {
                Some(registry::RegistryBuffers {
                    actors: registry::patch_capacity(
                        &self.device,
                        &self.actors,
                        &self.enterprise_requests,
                    ),
                    spatial: self.spatial.clone(),
                })
            }
            registry::RegistryTransition::Capacity(registry::CapacityUpdate::Rebuild)
            | registry::RegistryTransition::Migration => {
                let actors = plan.actors.as_ref().context("missing changed actor rows")?;
                let spatial = match plan.transition {
                    registry::RegistryTransition::Migration => registry::spatial::migrate(
                        &self.device,
                        &self.scenario,
                        registry::spatial::SpatialMigration {
                            spatial: &self.spatial,
                            actors: &self.actors,
                            previous: &self.metadata.actors,
                            current: actors,
                            requests: &plan.changes,
                        },
                    ),
                    _ => self.spatial.clone(),
                };
                Some(registry::RegistryBuffers {
                    actors: registry::metadata::patch(
                        &self.device,
                        &self.scenario,
                        &self.actors,
                        actors,
                        &plan.changes,
                    ),
                    spatial,
                })
            }
        };
        let next_buffers = match settlement.as_ref().filter(|batch| !batch.owners.is_empty()) {
            Some(batch) => Some(registry::RegistryBuffers {
                actors: registry::metadata::patch_actors(
                    &self.device,
                    &self.scenario,
                    next_buffers
                        .as_ref()
                        .map_or(&self.actors, |buffers| &buffers.actors),
                    plan.actors
                        .as_ref()
                        .context("missing inherited actor rows")?,
                    batch.owners.iter().map(|change| change.firm),
                ),
                spatial: next_buffers
                    .map_or_else(|| self.spatial.clone(), |buffers| buffers.spatial),
            }),
            None => next_buffers,
        };
        checkpoint("RegistryBuffers");
        match self.device.reclaim_memory(0)?.bytes_reserved <= self.device.memory_budget.bytes() {
            true => (),
            false => Err(anyhow!(
                "registry capacity exhausted while preparing GPU transitions"
            ))?,
        }
        match &settlement {
            Some(batch) => {
                batch.apply(&self.device, &self.state, self.config.state_stride as usize);
                self.readback_bytes += batch.readback_bytes;
            }
            None => registry::apply(
                &self.device,
                &self.state,
                &self.actors,
                &self.config,
                &self.enterprise_requests,
                &self.scratch,
                self.day as u32,
            ),
        }
        checkpoint("RegistryApply");
        let population = grown
            .as_ref()
            .map(|grown| {
                grown.remap_workers_from(
                    &self.device,
                    &self.state,
                    self.config.state_stride,
                    &self.demography,
                );
                grown.observation(&self.device, &self.state, &self.config)
            })
            .transpose();
        checkpoint("RegistryPopulation");
        let previous_actors = plan
            .actors
            .map(|actors| std::mem::replace(&mut self.metadata.actors, actors));
        if let Some(buffers) = next_buffers {
            self.actors = buffers.actors;
            self.spatial = buffers.spatial;
        }
        let rings = match (audit, &settlement) {
            #[cfg(test)]
            (RegistryAudit::Full, _) => economy::audit::RingValidation::All,
            (RegistryAudit::Validated, Some(batch)) => {
                economy::audit::RingValidation::Selected(batch.indices())
            }
            (RegistryAudit::Validated, None) => economy::audit::RingValidation::Unchanged,
        };
        let registry_observation =
            economy::audit::RegistryObservation::new(audit, births, settlement.as_ref());
        if std::env::var_os("BAD_ECON_PROFILE_REGISTRY").is_some() {
            eprintln!("REGISTRY {} Observation {registry_observation:?}", self.day);
        }
        let observation = population
            .and_then(|population| {
                match registry_observation {
                    economy::audit::RegistryObservation::Existing => {
                        self.fault_check().map(|_| self.snapshot.clone())
                    }
                    economy::audit::RegistryObservation::Updated => self.observe_rings(rings),
                }
                .map(|snapshot| (snapshot, population))
            })
            .and_then(|observation| {
                match self.device.reclaim_memory(0)?.bytes_reserved
                    <= self.device.memory_budget.bytes()
                {
                    true => Ok(observation),
                    false => Err(anyhow!(
                        "registry capacity exhausted while auditing GPU transitions"
                    )),
                }
            });
        checkpoint("RegistryObservation");
        match observation {
            Ok((mut snapshot, population)) => {
                match (
                    &management,
                    plan.transition,
                    self.config.property_mode,
                    self.config.rental_businesses,
                ) {
                    (Some(_), _, _, _)
                    | (
                        _,
                        registry::RegistryTransition::Migration,
                        configuration::Feature::Enabled,
                        _,
                    )
                    | (_, registry::RegistryTransition::Migration, _, 1..) => {
                        let started = Instant::now();
                        self.config.update_rentals(
                            &self.device,
                            &self.scenario,
                            &self.metadata,
                            plan.changes
                                .iter()
                                .filter(|record| record.request >> 24 == 3)
                                .map(|record| record.actor as usize)
                                .chain(settlement.iter().flat_map(|batch| {
                                    batch.owners.iter().map(|change| change.firm)
                                })),
                        );
                        if std::env::var_os("BAD_ECON_PROFILE_STAGES").is_some() {
                            eprintln!(
                                "CONFIG {} {:.3}",
                                self.day,
                                started.elapsed().as_secs_f64() * 1000.0
                            );
                        }
                    }
                    _ => (),
                }
                if let Some(ownership) = ownership {
                    self.ownership = Some(ownership);
                }
                if let Some(management) = management {
                    self.management = management;
                }
                if let Some(ownership) = settlement.and_then(|batch| batch.ownership) {
                    self.metadata.ownership = ownership;
                }
                if let Some(grown) = grown {
                    self.demography = grown;
                }
                if let Some(population) = population {
                    snapshot.population = population;
                }
                self.metadata.next_identity = next_identity;
                self.migrations = plan.migrations;
                self.snapshot = snapshot;
                if !market_graphs::same_buffer(&self.actors, &previous_actor_buffer) {
                    tensor::copy_contiguous(&self.device, &self.actors, &previous_actor_buffer);
                    self.actors = previous_actor_buffer;
                }
                if !market_graphs::same_buffer(&self.spatial, &previous_spatial) {
                    tensor::copy_contiguous(&self.device, &self.spatial, &previous_spatial);
                    self.spatial = previous_spatial;
                }
                Ok(())
            }
            Err(error) => {
                if let Some(actors) = previous_actors {
                    self.metadata.actors = actors;
                }
                self.actors = previous_actor_buffer;
                self.spatial = previous_spatial;
                Err(error)
            }
        }
    }
    fn account_research_readbacks(&mut self) {
        let total = self
            .research
            .as_ref()
            .map_or(0, research::ResearchRuntime::readback_bytes);
        self.readback_bytes += total - self.research_readback;
        self.research_readback = total;
    }
    fn feedback_operation(
        &self,
        stage: EconomyStage,
        count: usize,
        parameters: &EconomyParameters,
    ) {
        CubeBackend::economy(
            &self.device,
            stage,
            count,
            EconomyBuffers {
                state: &self.state,
                actors: &self.actors,
                config: &self.config,
                spatial: match (stage, parameters.empty_orders) {
                    (EconomyStage::RecordFeedback, parameters::EmptyOrderMode::FeedbackOnly) => {
                        &self.searches.requests
                    }
                    _ => &self.spatial,
                },
                proposals: &self.proposals,
                events: &self.scratch,
                control: &self.feedback,
            },
            parameters,
        );
    }
    fn apply_feedback(&mut self) -> Result<()> {
        let parameters = self.parameters(Phase::ProvisionPurchase, Market::Household, 0, 0)?;
        self.feedback_operation(
            EconomyStage::ApplyFeedback,
            self.metadata.actors.len(),
            &parameters,
        );
        let control = self.device.read_range(&self.feedback, 0, CONTROL_WORDS)?;
        self.readback_bytes += (CONTROL_WORDS * WORD_BYTES) as u64;
        FaultFlags::read(&control)?.into_result()
    }
    fn local(&mut self, phase: Phase) -> Result<()> {
        self.device.write(
            &self.control,
            bytemuck::cast_slice(&[ObservationControlLayout::zeroed()]),
        );
        let parameters = self.parameters(phase, Market::Labor, 0, self.metadata.actors.len())?;
        if matches!(phase, Phase::Proposal | Phase::AccountsDecisions) {
            self.economy(
                EconomyStage::InputQuotes,
                self.metadata.actors.len(),
                &parameters,
                &self.proposals,
            );
        }
        if let Phase::Opening = phase {
            self.feedback_operation(
                EconomyStage::ClearFeedback,
                CONTROL_WORDS + self.metadata.actors.len() * TICK_WORDS,
                &parameters,
            );
        }
        self.economy(
            EconomyStage::Local,
            self.metadata.actors.len(),
            &parameters,
            &self.proposals,
        );
        if let (Phase::WorkPayment, Some(_)) = (phase, self.scenario.config().rent) {
            self.economy(
                EconomyStage::FirmRent,
                self.metadata.actors.len(),
                &parameters,
                &self.proposals,
            );
        }
        if phase == Phase::Opening {
            self.railways(economy::railway::RailStage::Infrastructure);
        }
        economy::property::launch(
            &self.device,
            &self.state,
            &self.actors,
            &self.spatial,
            &self.config,
            &self.control,
            phase,
        );
        economy::landlords::launch(
            &self.device,
            &self.state,
            &self.actors,
            &self.config,
            &self.control,
            phase,
        );
        self.fault_check()
    }
    fn railways(&self, stage: economy::railway::RailStage) {
        let started = Instant::now();
        economy::railway::launch(
            &self.device,
            &self.state,
            &self.actors,
            &self.config,
            &self.control,
            stage,
        );
        if self.day + 1 == profiling_day() && std::env::var_os("BAD_ECON_PROFILE_STAGES").is_some()
        {
            self.device
                .sync()
                .expect("railway profiling synchronization");
            eprintln!(
                "RAILWAY {stage:?} {:.3}",
                started.elapsed().as_secs_f64() * 1000.0
            );
        }
    }
    fn market_result(&mut self, evidence: &mut PhaseEvidence) -> Result<()> {
        let execution = match (
            self.next_phase,
            self.scenario.config().railway.companies.is_empty(),
        ) {
            (Phase::ProvisionPurchase, false) => MarketExecution::Grouped,
            _ => self.market_execution,
        };
        match execution {
            MarketExecution::Grouped => {
                let words = self.device.read_range(&self.control, 0, CONTROL_WORDS)?;
                self.readback_bytes += (CONTROL_WORDS * WORD_BYTES) as u64;
                let counters = ControlLayout::new(&words)?;
                self.fault_result(counters.faults)?;
                evidence.frontier_depth = evidence.frontier_depth.max(counters.progress as usize);
                evidence.events += counters.completed as usize;
            }
            #[cfg(test)]
            _ => (),
        }
        Ok(())
    }
    fn freight_stage(
        &self,
        parameters: &EconomyParameters,
        stage: settlement::Stage,
        execution: settlement::Execution,
        profile: &profiling::MarketProfile,
    ) {
        profile.freight(&self.device, stage, execution.wave(), || {
            settlement::launch_buffered(
                &self.device,
                &self.state,
                &self.actors,
                &self.spatial,
                &self.config,
                &self.proposals,
                &self.scratch,
                &self.control,
                parameters,
                stage,
                execution,
            )
        });
    }
    fn market(
        &mut self,
        market: Market,
        phase: Phase,
        trace: Trace,
        evidence: &mut PhaseEvidence,
    ) -> Result<MatchingState> {
        let regional_transport = !self.scenario.config().railway.companies.is_empty()
            && matches!(
                market,
                Market::Input(_) | Market::Consumer(_) | Market::Household
            );
        match self.market_execution {
            _ if regional_transport => self.device_market(market, phase, trace, evidence),
            MarketExecution::Grouped => self.device_market(market, phase, trace, evidence),
            #[cfg(test)]
            _ => self
                .original_market(market, phase, trace, evidence)
                .map(|_| MatchingState::Searching),
        }
    }
    fn market_graph_buffers(&self) -> impl Iterator<Item = &CubeTensor> + Clone {
        [
            &self.state,
            &self.actors,
            &self.spatial,
            &self.proposals,
            &self.scratch,
            &self.control,
            &self.searches.requests,
            &self.searches.successes,
            &self.searches.dispatches,
        ]
        .into_iter()
        .chain(self.config.graph_buffers())
    }

    fn initialize_market(&self, market: Market, trace: Trace, parameters: &EconomyParameters) {
        let count = self.metadata.actors.len();
        let initialization = match (market, self.matching_round, trace) {
            (Market::Input(_) | Market::Consumer(_), 0, Trace::Discard) => {
                EconomyStage::InputMarketPrepare
            }
            (Market::Input(_) | Market::Consumer(_), 0, Trace::Record) => {
                EconomyStage::InputMarketStart
            }
            (Market::Input(_) | Market::Consumer(_), _, Trace::Record) => {
                EconomyStage::InputMarketRepeat
            }
            _ => EconomyStage::MarketStart,
        };
        let initialization_count = match initialization {
            EconomyStage::InputMarketStart | EconomyStage::InputMarketRepeat => count,
            _ => 1,
        };
        economy::EconomySystem::launch_elements(
            &self.device,
            initialization_count,
            economy::EconomyInputLaunch {
                state: self.state.buffer_arg(),
                actors: self.actors.buffer_arg(),
                config: self.config.launch(),
                spatial: self.spatial.buffer_arg(),
                proposals: self.proposals.buffer_arg(),
                bundles: self.scratch.buffer_arg(),
                control: self.control.buffer_arg(),
                parameters: parameters.launch(),
                stage: initialization,
            },
        );
    }

    fn device_market_prefix(
        &mut self,
        market: Market,
        trace: Trace,
        mut parameters: EconomyParameters,
        device_profile: &profiling::MarketProfile,
    ) -> Result<u32> {
        let count = self.metadata.actors.len();
        let members = match market {
            Market::Household | Market::Input(Good::Bread) | Market::Consumer(Good::Bread) => count,
            _ => self.metadata.actors.firms(),
        };
        let regional_transport = !self.scenario.config().railway.companies.is_empty()
            && matches!(
                market,
                Market::Input(_) | Market::Consumer(_) | Market::Household
            );
        let profile = self.day + 1 == profiling_day()
            && std::env::var_os("BAD_ECON_PROFILE_STAGES").is_some();
        let launch = |stage, invocations, parameters: &EconomyParameters| {
            let started = Instant::now();
            let window = device_profile.start(&self.device);
            let dispatch = match stage {
                EconomyStage::GroupExecute => markets::dispatch(
                    &self.searches.dispatches,
                    markets::RANGE_DISPATCH,
                    count.div_ceil(self.device.workgroup as usize),
                ),
                EconomyStage::GroupPrepare
                | EconomyStage::GroupFinalize
                | EconomyStage::GroupRadixLocal
                | EconomyStage::GroupRadixScatter
                | EconomyStage::GroupRemainder
                | EconomyStage::GroupRanges => markets::dispatch(
                    &self.searches.dispatches,
                    markets::SORT_DISPATCH,
                    count.div_ceil(self.device.workgroup as usize),
                ),
                _ => self.device.count(invocations).into(),
            };
            match stage {
                EconomyStage::MarketBoundary => {
                    markets::prepare_dispatch_for(
                        &self.device,
                        &self.control,
                        &self.searches.dispatches,
                        count,
                        markets::PreparationWorkgroup::Native,
                    );
                }
                _ => economy::EconomySystem::launch(
                    &self.device,
                    dispatch,
                    cubecl::CubeDim::new_1d(self.device.workgroup),
                    economy::EconomyInputLaunch {
                        state: self.state.buffer_arg(),
                        actors: self.actors.buffer_arg(),
                        config: self.config.launch(),
                        spatial: self.spatial.buffer_arg(),
                        proposals: self.proposals.buffer_arg(),
                        bundles: self.scratch.buffer_arg(),
                        control: self.control.buffer_arg(),
                        parameters: parameters.launch(),
                        stage,
                    },
                ),
            }
            device_profile.end(&self.device, window, "MARKET", stage);
            if profile {
                self.device.sync().expect("stage profiling synchronization");
                eprintln!(
                    "MARKET {market:?} {} {stage:?} {:.3}",
                    self.matching_round,
                    started.elapsed().as_secs_f64() * 1000.0
                );
            }
        };
        let preparation = market_graphs::Preparation::new(market, self.matching_round);
        let refresh_index = match (market, preparation) {
            (_, market_graphs::Preparation::RetainedLabor) => false,
            (Market::Labor | Market::Household, _) => true,
            (Market::Input(Good::Bread) | Market::Consumer(Good::Bread), _) => true,
            (Market::Input(_) | Market::Consumer(_), market_graphs::Preparation::First) => true,
            _ => false,
        };
        if refresh_index {
            let side = self
                .scenario
                .config()
                .map_size
                .div_ceil(self.scenario.config().search_radius);
            launch(EconomyStage::IndexCount, members, &parameters);
            parallel::scan(
                &self.device,
                &self.scratch,
                count,
                members.div_ceil(self.device.workgroup as usize),
            );
            economy::indexing::MarketIndex::launch_elements(
                &self.device,
                members.max((side * side + 1) as usize),
                economy::indexing::IndexInputLaunch {
                    config: self.config.launch(),
                    spatial: self.spatial.buffer_arg(),
                    scratch: self.scratch.buffer_arg(),
                    control: self.control.buffer_arg(),
                    parameters: parameters.launch(),
                },
            );
        }
        let search = |stage| {
            let started = Instant::now();
            let window = device_profile.start(&self.device);
            let workgroup = match stage {
                economy::search::SearchStage::Search => 32,
                economy::search::SearchStage::SearchIndexed => 256,
                economy::search::SearchStage::EmptySort => 256,
                economy::search::SearchStage::Prepare
                | economy::search::SearchStage::PrepareCandidates => 256,
                _ => self.device.workgroup,
            };
            let dispatch = match stage {
                economy::search::SearchStage::SearchSpatialFuel => self.device.count(count).into(),
                economy::search::SearchStage::EmptySort => {
                    cubecl::CubeCount::Static(1, 1, 1).into()
                }
                economy::search::SearchStage::EmptyBounds => self
                    .device
                    .count(economy::search::empty_bounds_invocations(
                        members,
                        parameters.scan,
                    ))
                    .into(),
                economy::search::SearchStage::FilterEmpty => {
                    self.device.count(count.div_ceil(32)).into()
                }
                economy::search::SearchStage::FinishFiltered => self.device.count(members).into(),
                economy::search::SearchStage::Index => self
                    .device
                    .count(
                        members.div_ceil(DISCOVERY_BLOCK as usize) * self.device.workgroup as usize,
                    )
                    .into(),
                economy::search::SearchStage::Prepare
                | economy::search::SearchStage::PrepareCandidates => {
                    cubecl::calculate_cube_count_elemwise(
                        &self.device.client,
                        count.max(1),
                        cubecl::CubeDim::new_1d(workgroup),
                    )
                    .into()
                }
                economy::search::SearchStage::PrepareSelected => markets::dispatch(
                    &self.searches.dispatches,
                    markets::PREPARATION_DISPATCH,
                    count.div_ceil(workgroup as usize),
                ),
                economy::search::SearchStage::Search
                | economy::search::SearchStage::SearchIndexed => markets::job_dispatch(
                    &self.searches.dispatches,
                    markets::SEARCH_DISPATCH,
                    count,
                ),
                economy::search::SearchStage::SearchFuel
                | economy::search::SearchStage::SearchHousehold
                | economy::search::SearchStage::Compact => markets::dispatch(
                    &self.searches.dispatches,
                    markets::SEARCH_COMPACT_DISPATCH,
                    count.div_ceil(workgroup as usize),
                ),
            };
            economy::search::MarketSearch::launch(
                &self.device,
                dispatch,
                cubecl::CubeDim::new_1d(workgroup),
                economy::search::SearchInputLaunch {
                    state: self.state.buffer_arg(),
                    actors: self.actors.buffer_arg(),
                    config: self.config.launch(),
                    spatial: self.spatial.buffer_arg(),
                    proposals: self.proposals.buffer_arg(),
                    searches: self.searches.requests.buffer_arg(),
                    successes: self.searches.successes.buffer_arg(),
                    scratch: self.scratch.buffer_arg(),
                    control: self.control.buffer_arg(),
                    parameters: parameters.launch(),
                    program: stage.program(&self.config),
                    status_reset: match (market, trace) {
                        (Market::Input(_) | Market::Consumer(Good::Coal | Good::Wood), _) => {
                            economy::search::StatusReset::Active
                        }
                        (_, Trace::Record) => economy::search::StatusReset::Full,
                        (_, Trace::Discard) => economy::search::StatusReset::Active,
                    },
                },
            );
            device_profile.end(&self.device, window, "SEARCH", stage);
            if profile {
                self.device
                    .sync()
                    .expect("search profiling synchronization");
                eprintln!(
                    "SEARCH {market:?} {} {stage:?} {:.3}",
                    self.matching_round,
                    started.elapsed().as_secs_f64() * 1000.0
                );
            }
        };
        if matches!(
            market,
            Market::Input(Good::Bread) | Market::Consumer(_) | Market::Household
        ) {
            search(economy::search::SearchStage::Index);
        }
        if let Market::Consumer(_) = market {
            search(economy::search::SearchStage::EmptySort);
            search(economy::search::SearchStage::EmptyBounds);
            if self.day + 1 == profiling_day()
                && std::env::var_os("BAD_ECON_PROFILE_SUPPLY").is_some()
            {
                let cells = self.config.spatial_side * self.config.spatial_side;
                let suppliers = self.device.read_range(&self.spatial, cells as usize, 1)?[0];
                let records = self
                    .device
                    .read_range(&self.scratch, 0, suppliers as usize * 4)?;
                self.readback_bytes += (records.len() + 1) as u64 * WORD_BYTES as u64;
                let stocked = records
                    .chunks_exact(4)
                    .filter(|record| record[3] & 1 != 0)
                    .count();
                let empty = records
                    .chunks_exact(4)
                    .filter(|record| record[3] & 3 == 2)
                    .count();
                eprintln!(
                    "SUPPLY {market:?} {} suppliers={suppliers} stocked={stocked} empty={empty}",
                    self.matching_round,
                );
            }
        }
        match market {
            Market::Labor | Market::Payroll | Market::Distribution => {
                let stage = match preparation {
                    market_graphs::Preparation::RetainedLabor => {
                        EconomyStage::ProposeRetainedCompact
                    }
                    _ => EconomyStage::ProposeCompact,
                };
                launch(stage, count, &parameters);
                parallel::scan(
                    &self.device,
                    &self.scratch,
                    7 * count,
                    count.div_ceil(self.device.workgroup as usize),
                );
                launch(EconomyStage::GroupCompactFill, count, &parameters);
            }
            _ => {
                let stage = match (market, self.matching_round) {
                    (Market::Input(_) | Market::Consumer(_), 0) => {
                        economy::search::SearchStage::PrepareCandidates
                    }
                    (Market::Input(_) | Market::Consumer(_), _) => {
                        economy::search::SearchStage::PrepareSelected
                    }
                    _ => economy::search::SearchStage::Prepare,
                };
                search(stage);
                launch(EconomyStage::MarketBoundary, 1, &parameters);
                if self.day + 1 == profiling_day()
                    && std::env::var_os("BAD_ECON_PROFILE_SUPPLY").is_some()
                    && market == Market::Input(Good::Bread)
                {
                    let words = self.device.read_range(
                        &self.control,
                        0,
                        economy::search::INPUT_MINIMUM_QUOTE_WORD + 1,
                    )?;
                    let requests = self
                        .device
                        .read_range(&self.searches.requests, 0, 5 * count)?;
                    self.readback_bytes +=
                        (words.len() + requests.len()) as u64 * WORD_BYTES as u64;
                    let minimum_quote = u64::from(words[economy::search::INPUT_MINIMUM_QUOTE_WORD]);
                    let unaffordable = requests[..words[markets::SEARCHES_WORD] as usize]
                        .iter()
                        .filter(|&&buyer| {
                            let buyer = buyer as usize;
                            let cash = u64::from(requests[3 * count + buyer])
                                | (u64::from(requests[4 * count + buyer]) << 32);
                            cash < minimum_quote
                        })
                        .count();
                    eprintln!(
                        "INPUT_QUOTE {market:?} {} minimum={minimum_quote} searches={} unaffordable={unaffordable}",
                        self.matching_round,
                        words[markets::SEARCHES_WORD],
                    );
                }
                search(match market {
                    Market::Consumer(_) => economy::search::SearchStage::SearchSpatialFuel,
                    Market::Household => economy::search::SearchStage::SearchHousehold,
                    Market::Input(Good::Bread) => economy::search::SearchStage::SearchIndexed,
                    _ => economy::search::SearchStage::Search,
                });
                if parameters.empty_orders == parameters::EmptyOrderMode::FeedbackOnly {
                    search(economy::search::SearchStage::FilterEmpty);
                }
                parallel::scan(
                    &self.device,
                    &self.searches.successes,
                    markets::search_counts_offset(count),
                    markets::search_blocks(count),
                );
                search(economy::search::SearchStage::Compact);
                if parameters.empty_orders == parameters::EmptyOrderMode::FeedbackOnly {
                    search(economy::search::SearchStage::FinishFiltered);
                }
                if self.day + 1 == profiling_day()
                    && std::env::var_os("BAD_ECON_PROFILE_SUPPLY").is_some()
                {
                    let words =
                        self.device
                            .read_range(&self.control, 0, markets::WIDE_ORDERS_WORD + 1)?;
                    self.readback_bytes += words.len() as u64 * WORD_BYTES as u64;
                    eprintln!(
                        "ORDER_STATS {market:?} {} mode={:?} fallback={} searches={} constructed={} completed={} frontier={}",
                        self.matching_round,
                        parameters.empty_orders,
                        words[markets::WIDE_ORDERS_WORD],
                        words[markets::SEARCHES_WORD],
                        words[flags::CONSTRUCTED_EVENTS_WORD],
                        words[flags::COMPLETED_WORD],
                        words[flags::PROGRESS_WORD],
                    );
                }
            }
        }
        launch(EconomyStage::MarketBoundary, 1, &parameters);
        let sorting_window = device_profile.start(&self.device);
        let sort_keys =
            economy::sorting::SortKeys::for_market(market, self.scenario.population_rows() as u32);
        parameters.sort_source = match (parameters.clock, device_profile) {
            (parameters::MarketClock::Device, _) | (_, profiling::MarketProfile::Enabled(_)) => {
                economy::sorting::MarketSort::encode_keys(
                    &self.device,
                    &self.scratch,
                    &self.control,
                    &self.searches.dispatches,
                    &parameters,
                    sort_keys,
                )
            }
            (parameters::MarketClock::Scalars, _) => self.market_sort.run_keys(
                &self.device,
                &self.scratch,
                &self.control,
                &self.searches.dispatches,
                &parameters,
                sort_keys,
            )?,
        };
        device_profile.end(
            &self.device,
            sorting_window,
            "SORT",
            EconomyStage::GroupSort,
        );
        let freight = |stage| {
            let execution = self.freight_execution.initial(trace);
            self.freight_stage(&parameters, stage, execution, device_profile)
        };
        if regional_transport {
            freight(settlement::Stage::Snapshot);
        }
        launch(EconomyStage::GroupRanges, count, &parameters);
        launch(EconomyStage::MarketBoundary, 1, &parameters);
        if matches!(market, Market::Labor | Market::Input(_)) {
            launch(EconomyStage::GroupPrepare, count, &parameters);
        }
        let batch_window = match market {
            Market::Payroll | Market::Consumer(_) => device_profile.start(&self.device),
            Market::Labor if self.scenario.config().bargaining.is_none() => {
                device_profile.start(&self.device)
            }
            _ => None,
        };
        match market {
            Market::Payroll => economy::batch::payroll::PayrollSettlement::launch(
                &self.device,
                markets::job_dispatch(&self.searches.dispatches, markets::QUEUE_DISPATCH, count),
                cubecl::CubeDim::new_1d(self.device.workgroup),
                economy::batch::payroll::PayrollInputLaunch {
                    state: self.state.buffer_arg(),
                    actors: self.actors.buffer_arg(),
                    config: self.config.launch(),
                    proposals: self.proposals.buffer_arg(),
                    scratch: self.scratch.buffer_arg(),
                    control: self.control.buffer_arg(),
                    parameters: parameters.launch(),
                },
            ),
            Market::Labor if self.scenario.config().bargaining.is_none() => {
                economy::batch::labor::LaborSettlement::launch(
                    &self.device,
                    markets::job_dispatch(
                        &self.searches.dispatches,
                        markets::QUEUE_DISPATCH,
                        count,
                    ),
                    cubecl::CubeDim::new_1d(self.device.workgroup),
                    economy::batch::labor::LaborInputLaunch {
                        proposals: self.proposals.buffer_arg(),
                        state: self.state.buffer_arg(),
                        config: self.config.launch(),
                        scratch: self.scratch.buffer_arg(),
                        control: self.control.buffer_arg(),
                        parameters: parameters.launch(),
                    },
                )
            }
            Market::Consumer(good)
                if self.scenario.config().people.is_some()
                    || matches!(
                        good,
                        Good::Wood | Good::Coal | Good::LuxuryClothes | Good::LuxuryFurniture
                    ) =>
            {
                economy::batch::quantity::QuantitySettlement::launch(
                    &self.device,
                    markets::job_dispatch(
                        &self.searches.dispatches,
                        markets::QUEUE_DISPATCH,
                        count,
                    ),
                    cubecl::CubeDim::new_1d(self.device.workgroup),
                    economy::batch::quantity::QuantityInputLaunch {
                        state: self.state.buffer_arg(),
                        actors: self.actors.buffer_arg(),
                        config: self.config.launch(),
                        proposals: self.proposals.buffer_arg(),
                        scratch: self.scratch.buffer_arg(),
                        control: self.control.buffer_arg(),
                        parameters: parameters.launch(),
                    },
                )
            }
            Market::Consumer(_) => economy::batch::settle_consumers(
                &self.device,
                &self.state,
                &self.actors,
                &self.config,
                &self.scratch,
                &self.control,
                &self.searches.dispatches,
                &parameters,
            ),
            _ => launch(EconomyStage::GroupExecute, count, &parameters),
        }
        device_profile.end(
            &self.device,
            batch_window,
            "BATCH",
            EconomyStage::GroupExecute,
        );
        if market == Market::Household {
            launch(EconomyStage::GroupRemainder, count, &parameters);
        }
        launch(EconomyStage::MarketBoundary, 1, &parameters);
        if matches!(
            market,
            Market::Labor | Market::Input(_) | Market::Consumer(_) | Market::Household
        ) {
            launch(EconomyStage::GroupFinalize, count, &parameters);
        }
        launch(EconomyStage::MarketBoundary, 1, &parameters);
        if regional_transport {
            for stage in [
                settlement::Stage::Initialize,
                settlement::Stage::Queue,
                settlement::Stage::Propose,
                settlement::Stage::CountClaims,
            ] {
                freight(stage);
            }
            if let (Trace::Discard, FreightExecution::Buffered) = (trace, self.freight_execution) {
                self.finish_freight(&parameters, trace, device_profile)?;
            }
        }
        Ok(parameters.sort_source)
    }

    fn finish_freight(
        &mut self,
        parameters: &EconomyParameters,
        trace: Trace,
        device_profile: &profiling::MarketProfile,
    ) -> Result<MatchingState> {
        use settlement::{Execution, Stage, Wave};
        let freight = |simulation: &Self, stage, execution| {
            simulation.freight_stage(parameters, stage, execution, device_profile)
        };
        let prepare_retry = |simulation: &Self, execution: Execution| {
            if execution.wave() == Wave::Retry {
                for stage in [
                    Stage::Queue,
                    Stage::Reset,
                    Stage::Propose,
                    Stage::CountClaims,
                ] {
                    freight(simulation, stage, execution);
                }
            }
        };
        let mut matching = MatchingState::Searching;
        match (trace, self.freight_activity, self.freight_execution) {
            (Trace::Discard, FreightActivity::Dormant, _) => {
                let execution = Execution::Immediate(Wave::Initial);
                freight(self, Stage::Commit, execution);
                freight(self, Stage::Release, execution);
            }
            (Trace::Discard, _, FreightExecution::Buffered) => {
                let storage = settlement::ClaimStorage::new(&self.config);
                for &execution in storage.executions() {
                    prepare_retry(self, execution);
                    for &stage in storage.stages() {
                        freight(self, stage, execution);
                    }
                    freight(self, Stage::Commit, execution);
                    freight(self, Stage::Release, execution);
                }
            }
            _ => {
                let mut next_wave = Some(Wave::Initial);
                while let Some(wave) = next_wave {
                    let execution = Execution::Immediate(wave);
                    prepare_retry(self, execution);
                    let preparation = settlement::prepare_market_claims(
                        &self.device,
                        &mut self.config,
                        &self.control,
                    )?;
                    self.readback_bytes += (4 * WORD_BYTES) as u64;
                    matching = match preparation.searches {
                        0 => MatchingState::Exhausted,
                        _ => MatchingState::Searching,
                    };
                    let claims = preparation.claims;
                    if let settlement::ClaimWork::Ready = claims {
                        freight(self, Stage::EmitClaims, execution);
                        freight(self, Stage::SortClaims, execution);
                        freight(self, Stage::Resolve, execution);
                    }
                    freight(self, Stage::Commit, execution);
                    freight(self, Stage::Release, execution);
                    next_wave = wave.next(claims);
                }
            }
        }
        freight(self, Stage::Accounts, self.freight_execution.initial(trace));
        Ok(matching)
    }

    fn device_market(
        &mut self,
        market: Market,
        phase: Phase,
        trace: Trace,
        evidence: &mut PhaseEvidence,
    ) -> Result<MatchingState> {
        let submission_started = Instant::now();
        let mut matching = MatchingState::Searching;
        let regional_transport = !self.scenario.config().railway.companies.is_empty()
            && matches!(
                market,
                Market::Input(_) | Market::Consumer(_) | Market::Household
            );
        let count = self.metadata.actors.len();
        let mut parameters = self.parameters(phase, market, 0, count)?;
        parameters.discovery = Discovery::Indexed;
        parameters.empty_orders = match (market, trace) {
            (Market::Consumer(good), Trace::Discard) if good != Good::Bread => {
                parameters::EmptyOrderMode::FeedbackOnly
            }
            _ => parameters::EmptyOrderMode::Settle,
        };
        let profile = self.day + 1 == profiling_day()
            && std::env::var_os("BAD_ECON_PROFILE_STAGES").is_some();
        let device_profile = profiling::MarketProfile::new(&self.device, self.day + 1);
        self.initialize_market(market, trace, &parameters);
        parameters.clock = match (
            trace,
            count,
            profile,
            &device_profile,
            &self.market_sort,
            &self.device.cube,
        ) {
            (
                Trace::Discard,
                4096..,
                false,
                profiling::MarketProfile::Disabled,
                economy::sorting::MarketSort::Profiled,
                _,
            ) => parameters::MarketClock::Scalars,
            (
                Trace::Discard,
                4096..,
                false,
                profiling::MarketProfile::Disabled,
                _,
                cubecl::Device::Wgpu(_) | cubecl::Device::Metal(_),
            ) if self.market_replay.clock(market.into()) == parameters::MarketClock::Device => {
                parameters::MarketClock::Device
            }
            _ => parameters::MarketClock::Scalars,
        };
        if std::env::var_os("BAD_ECON_PROFILE_SUPPLY").is_some() {
            parameters.clock = parameters::MarketClock::Scalars;
        }
        parameters.sort_source = match parameters.clock {
            parameters::MarketClock::Scalars => {
                self.device_market_prefix(market, trace, parameters, &device_profile)?
            }
            parameters::MarketClock::Device => {
                let key = market_graphs::MarketKey {
                    market: market.into(),
                    phase,
                    routing: self.config.railway.route_availability(),
                    freight: self.freight_execution,
                    preparation: market_graphs::Preparation::new(market, self.matching_round),
                };
                let mut graphs = std::mem::take(&mut self.market_graphs);
                let result = graphs.run(self, key, |simulation| {
                    simulation.device_market_prefix(market, trace, parameters, &device_profile)
                });
                let result = result.and_then(|source| match (regional_transport, key.routing) {
                    (true, economy::railway::RouteAvailability::Dormant) => {
                        let available = market_graphs::MarketKey {
                            routing: economy::railway::RouteAvailability::Available,
                            ..key
                        };
                        graphs.prepare(available, source);
                        Ok(source)
                    }
                    _ => Ok(source),
                });
                self.market_graphs = graphs;
                result?
            }
        };
        if let (true, Trace::Record, _) | (true, _, FreightExecution::Synchronous) =
            (regional_transport, trace, self.freight_execution)
        {
            matching = self.finish_freight(&parameters, trace, &device_profile)?;
        }
        if let (
            Trace::Discard,
            FreightActivity::Dormant,
            0,
            Market::Input(Good::Housing | Good::Railway),
        ) = (trace, self.freight_activity, self.matching_round, market)
        {
            let searches = self
                .device
                .read_range(&self.control, markets::SEARCHES_WORD, 1)?[0];
            self.readback_bytes += WORD_BYTES as u64;
            matching = match searches {
                0 => MatchingState::Exhausted,
                _ => MatchingState::Searching,
            };
        }
        parameters.status_start = 4 * count as u32;
        if let (Market::Input(_) | Market::Consumer(_) | Market::Household, true) =
            (market, self.matching_round + 1 == MATCHING_ROUNDS)
        {
            markets::feedback_boundary(
                &self.device,
                &self.control,
                &self.feedback,
                match parameters.empty_orders {
                    parameters::EmptyOrderMode::Settle => markets::FeedbackSource::Constructed,
                    parameters::EmptyOrderMode::FeedbackOnly => markets::FeedbackSource::Searches,
                },
            );
            self.feedback_operation(EconomyStage::RecordFeedback, count, &parameters);
        }
        if let Trace::Record = trace {
            self.fault_check()?;
            let statuses = self.device.read_range(&self.scratch, 4 * count, count)?;
            self.readback_bytes += (count * WORD_BYTES) as u64;
            evidence.decisions.extend(
                statuses
                    .into_iter()
                    .map(|code| EventStatus::try_from(code)?.decision())
                    .collect::<Result<Vec<_>>>()?
                    .into_iter()
                    .flatten(),
            );
        }
        if self.day + 1 == profiling_day()
            && std::env::var_os("BAD_ECON_PROFILE_SUBMISSION").is_some()
        {
            eprintln!(
                "SUBMISSION {market:?} {} {:.3}",
                self.matching_round,
                submission_started.elapsed().as_secs_f64() * 1000.0
            );
        }
        self.readback_bytes +=
            device_profile.resolve(market, self.matching_round, &self.config, &self.control)?;
        Ok(matching)
    }
    #[cfg(test)]
    fn original_market(
        &mut self,
        market: Market,
        phase: Phase,
        trace: Trace,
        evidence: &mut PhaseEvidence,
    ) -> Result<()> {
        self.device.write(
            &self.control,
            bytemuck::cast_slice(&[ObservationControlLayout::zeroed()]),
        );
        let parameters = self.parameters(phase, market, 0, self.metadata.actors.len())?;
        match market {
            Market::Labor | Market::Input(_) | Market::Consumer(_) | Market::Household => {
                let side = self
                    .scenario
                    .config()
                    .map_size
                    .div_ceil(self.scenario.config().search_radius);
                let actors = self.metadata.actors.len();
                let index = |stage, count| {
                    CubeBackend::economy(
                        &self.device,
                        stage,
                        count,
                        EconomyBuffers {
                            state: &self.state,
                            actors: &self.actors,
                            config: &self.config,
                            spatial: &self.spatial,
                            proposals: &self.proposals,
                            events: &self.scratch,
                            control: &self.control,
                        },
                        &parameters,
                    );
                };
                index(EconomyStage::IndexCount, actors);
                parallel::scan(
                    &self.device,
                    &self.scratch,
                    actors,
                    actors.div_ceil(self.device.workgroup as usize),
                );
                index(EconomyStage::IndexOffsets, (side * side + 1) as usize);
                index(EconomyStage::IndexFill, actors);
                if matches!(
                    market,
                    Market::Household | Market::Input(Good::Bread) | Market::Consumer(Good::Bread)
                ) {
                    index(
                        EconomyStage::IndexPrices,
                        actors.div_ceil(DISCOVERY_BLOCK as usize) * self.device.workgroup as usize,
                    );
                }
            }
            _ => (),
        }
        self.economy(
            EconomyStage::Propose,
            self.metadata.actors.len(),
            &parameters,
            &self.proposals,
        );
        self.economy(
            EconomyStage::MarketActivity,
            self.metadata.actors.len(),
            &parameters,
            &self.proposals,
        );
        let words = self.device.read_range(&self.control, 0, CONTROL_WORDS)?;
        self.readback_bytes += (CONTROL_WORDS * WORD_BYTES) as u64;
        let activity = ControlLayout::new(&words)?;
        FaultFlags::new(activity.faults)?.into_result()?;
        let range = 0..self.metadata.actors.len();
        match (activity.phase_data, self.market_execution) {
            (0, _) => Ok(()),
            (_, MarketExecution::Original) => {
                self.clear_market(market, phase, range, activity.phase_data, trace, evidence)
            }
            (_, MarketExecution::Reference) => {
                self.reference_market(market, phase, range, trace, evidence)
            }
            (_, MarketExecution::Grouped) => unreachable!(),
        }
    }
    #[cfg(test)]
    fn clear_market(
        &mut self,
        market: Market,
        phase: Phase,
        range: std::ops::Range<usize>,
        active: u32,
        trace: Trace,
        evidence: &mut PhaseEvidence,
    ) -> Result<()> {
        let count = range.len();
        let mut parameters = self.parameters(phase, market, range.start, count)?;
        parameters.sort_count = active;
        self.device.write(
            &self.control,
            bytemuck::cast_slice(&[ObservationControlLayout::zeroed()]),
        );
        let launch = |stage, parameters: &EconomyParameters| {
            let invocations = match stage {
                EconomyStage::GroupSort | EconomyStage::GroupMerge | EconomyStage::GroupExecute => {
                    active as usize
                }
                _ => count,
            };
            CubeBackend::economy(
                &self.device,
                stage,
                invocations,
                EconomyBuffers {
                    state: &self.state,
                    actors: &self.actors,
                    config: &self.config,
                    spatial: &self.spatial,
                    proposals: &self.proposals,
                    events: &self.scratch,
                    control: &self.control,
                },
                parameters,
            );
        };
        launch(EconomyStage::GroupInitialize, &parameters);
        if let Market::Labor | Market::Input(_) | Market::Consumer(_) | Market::Household = market {
            launch(EconomyStage::GroupPrepare, &parameters);
        }
        launch(EconomyStage::GroupSort, &parameters);
        let mut width = self.device.workgroup as usize;
        while width < active as usize {
            parameters.sort_width = width as u32;
            launch(EconomyStage::GroupMerge, &parameters);
            parameters.sort_source = 2 * count as u32 - parameters.sort_source;
            width *= 2;
        }
        launch(EconomyStage::GroupExecute, &parameters);
        if let Market::Labor | Market::Input(_) | Market::Consumer(_) | Market::Household = market {
            launch(EconomyStage::GroupFinalize, &parameters);
        }
        let words = self.device.read_range(&self.control, 0, CONTROL_WORDS)?;
        self.readback_bytes += (CONTROL_WORDS * WORD_BYTES) as u64;
        let counters = ControlLayout::new(&words)?;
        FaultFlags::new(counters.faults)?.into_result()?;
        evidence.frontier_depth = evidence.frontier_depth.max(counters.progress as usize);
        evidence.events += counters.completed as usize;
        parameters.status_start = 4 * count as u32;
        if let (Market::Input(_) | Market::Consumer(_) | Market::Household, true) =
            (market, self.matching_round + 1 == MATCHING_ROUNDS)
        {
            self.feedback_operation(EconomyStage::RecordFeedback, count, &parameters);
        }
        if let Trace::Record = trace {
            let statuses = self.device.read_range(&self.scratch, 4 * count, count)?;
            self.readback_bytes += (count * WORD_BYTES) as u64;
            evidence.decisions.extend(
                statuses
                    .into_iter()
                    .map(|code| EventStatus::try_from(code)?.decision())
                    .collect::<Result<Vec<_>>>()?
                    .into_iter()
                    .flatten(),
            );
        }
        Ok(())
    }
    #[cfg(test)]
    fn reference_market(
        &mut self,
        market: Market,
        phase: Phase,
        range: std::ops::Range<usize>,
        trace: Trace,
        evidence: &mut PhaseEvidence,
    ) -> Result<()> {
        let rotation = self
            .parameters(phase, market, range.start, range.len())?
            .settlement_rotation as usize;
        let batch_size = self.scenario.config().batch_size as usize;
        let batches = [rotation..range.end, range.start..rotation]
            .into_iter()
            .flat_map(|segment| {
                let end = segment.end;
                segment
                    .step_by(batch_size)
                    .map(move |offset| (offset, (end - offset).min(batch_size)))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut decisions = vec![None; range.len()];
        batches
            .into_iter()
            .try_for_each(|(offset, count)| -> Result<()> {
                let parameters = self.parameters(phase, market, offset, count)?;
                self.device.write(
                    &self.control,
                    bytemuck::cast_slice(&[ObservationControlLayout::zeroed()]),
                );
                self.economy(EconomyStage::Construct, count, &parameters, &self.proposals);
                self.fault_check()?;
                let counters = self.device.read_range(&self.control, 0, CONTROL_WORDS)?;
                self.readback_bytes += (CONTROL_WORDS * WORD_BYTES) as u64;
                let result = match ControlLayout::new(&counters)?.phase_data {
                    0 => Execution {
                        decisions: Vec::new(),
                        depth: 0,
                        events: 0,
                        readback_bytes: 0,
                        status_start: 0,
                    },
                    _ => execute(
                        &self.device,
                        SchedulerBuffers {
                            state: &self.state,
                            events: &self.events,
                            scratch: &self.scratch,
                            control: &self.control,
                            links: &self.links,
                        },
                        count,
                        trace,
                    )?,
                };
                if let (Market::Input(_) | Market::Consumer(_) | Market::Household, 1.., true) = (
                    market,
                    result.events,
                    self.matching_round + 1 == MATCHING_ROUNDS,
                ) {
                    let parameters = EconomyParameters {
                        status_start: result.status_start as u32,
                        ..parameters
                    };
                    self.feedback_operation(EconomyStage::RecordFeedback, count, &parameters);
                }
                evidence.frontier_depth = evidence.frontier_depth.max(result.depth);
                evidence.events += result.events;
                if let (Trace::Record, 1..) = (trace, result.events) {
                    let codes =
                        self.device
                            .read_range(&self.scratch, result.status_start, count)?;
                    for (index, code) in codes.into_iter().enumerate() {
                        decisions[offset + index] = EventStatus::try_from(code)?.decision()?;
                    }
                }
                self.readback_bytes += result.readback_bytes;
                Ok(())
            })?;
        evidence.decisions.extend(decisions.into_iter().flatten());
        Ok(())
    }
    pub fn observe(&mut self) -> Result<ObservationSnapshot> {
        self.observe_rings(economy::audit::RingValidation::All)
    }
    fn observe_rings(
        &mut self,
        rings: economy::audit::RingValidation<'_>,
    ) -> Result<ObservationSnapshot> {
        self.device.write(
            &self.control,
            bytemuck::cast_slice(&[ObservationControlLayout::zeroed()]),
        );
        let input = economy::audit::AuditInput {
            state: &self.state,
            actors: &self.actors,
            config: &self.config,
            spatial: &self.spatial,
            partials: &self.events,
            summary: &self.summary,
            control: &self.control,
            ring_mask: self.snapshot_copy.ring_mask(),
            rings,
        };
        #[cfg(test)]
        match self.audit_execution {
            AuditExecution::Original => {
                if let Some(mask) = self.snapshot_copy.ring_mask() {
                    tensor::reset_ring_mask(&self.device, mask, tensor::RingMaskState::Unknown);
                }
                let count = self.metadata.actors.len();
                let parameters = self.parameters(Phase::Audit, Market::Labor, 0, count)?;
                let groups = count.div_ceil(SummaryLayout::ACTORS_PER_GROUP as usize);
                self.economy(EconomyStage::Summarize, groups, &parameters, &self.summary);
                let groups = self
                    .device
                    .tensor(&vec![groups as u32; SummaryLayout::METRICS]);
                reduction::summary(
                    &self.device,
                    &self.events,
                    &self.summary,
                    &self.control,
                    &groups,
                )
            }
            AuditExecution::Tiled => self.audit.summarize(&self.device, input),
        }?;
        #[cfg(not(test))]
        self.audit.summarize(&self.device, input)?;
        let totals = self
            .device
            .read_range(&self.summary, 0, SummaryLayout::WORDS)?;
        let totals = SummaryLayout::new(&totals)?;
        let control = self.device.read(&self.control)?;
        let control = ObservationControlLayout::new(&control)?;
        self.readback_bytes +=
            ((SummaryLayout::WORDS + SummaryLayout::CONTROL_WORDS) * WORD_BYTES) as u64;
        FaultFlags::new(control.scheduler.faults)?.into_result()?;
        let financial = self
            .finance
            .as_ref()
            .map(|finance| finance.observe(&self.device, &self.state, self.day as u32))
            .transpose()?
            .unwrap_or_default();
        self.readback_bytes += self
            .finance
            .as_ref()
            .map_or(0, finance::FinanceRuntime::readback_bytes_per_observation);
        let snapshot = ObservationSnapshot {
            economy: diagnostics::EconomyObservation {
                workforce: totals.workforce.into(),
                subsistence: totals.subsistence.into(),
                inactive: totals.inactive.into(),
                rural_employed: totals.rural_employed.into(),
                household_cash: totals.household_cash.into(),
                active_by_good: totals.active_by_good.map(u64::from),
                business_types: totals.business_types.map(u64::from),
                business_states: totals.business_states.map(u64::from),
                firm_cash: totals.firm_cash.map(u64::from),
                accounts: std::array::from_fn(|good| diagnostics::BusinessAccount {
                    opening_cash: totals.business_opening[good].into(),
                    closing_cash: totals.business_closing[good].into(),
                    revenue: totals.business_revenue[good].into(),
                }),
            },
            day: self.day,
            weather: self.scenario.weather(self.day),
            cash: add(totals.cash.into(), financial.cash()?)?,
            payroll_reserves: totals.payroll_reserves.into(),
            income_tax: totals.income_tax.into(),
            employee_charges: totals.employee_charges.into(),
            rent_collected: totals.rent_collected.into(),
            rent_arrears: totals.rent_arrears.into(),
            financial_cash: financial.cash()?,
            loan_principal: financial.principal,
            loan_interest: financial.interest_due,
            active_loans: financial.active_loans,
            overdue_loans: financial.overdue_loans,
            research: self
                .research
                .as_ref()
                .map(|research| research.summary(&self.device))
                .transpose()?
                .unwrap_or_default(),
            employed: totals.employed.into(),
            homeless_households: totals.homeless_households.into(),
            households: (self.metadata.actors.len() - self.metadata.actors.firms()) as u64,
            shortages: totals.shortages.into(),
            firms: self.metadata.actors.firms() as u64,
            firm_entries: self
                .metadata
                .actors
                .iter_from(self.scenario.actors())
                .filter(|actor| actor.kind > 0)
                .count() as u64,
            firm_closures: totals.closures.into(),
            distributors: totals.distributors.into(),
            household_production: totals.household_production.map(u64::from),
            population: self
                .demography
                .observation(&self.device, &self.state, &self.config)?,
            inventories: totals.inventories.map(u64::from),
            transit: totals.transit.map(u64::from),
            produced: totals.produced.map(u64::from),
            used: totals.used.map(u64::from),
            consumed: totals.consumed.into(),
            consumption: totals.consumption.map(u64::from),
            minimum_wage: totals.minimum_wage.into(),
            maximum_wage: totals.maximum_wage.into(),
            minimum_price: totals.minimum_price.into(),
            maximum_price: totals.maximum_price.into(),
            markets: std::array::from_fn(|good| {
                let count = u64::from(totals.firms_by_good[good]);
                let wages = match count {
                    0 => QuoteDistribution::default(),
                    _ => QuoteDistribution {
                        count,
                        minimum: totals.wage_minima[good].into(),
                        maximum: totals.wage_maxima[good].into(),
                        total: totals.wage_totals[good].into(),
                        squares: totals.wage_squares[good].into(),
                    },
                };
                let count = u64::from(totals.price_counts[good]);
                let prices = match count {
                    0 => QuoteDistribution::default(),
                    _ => QuoteDistribution {
                        count,
                        minimum: totals.price_minima[good].into(),
                        maximum: totals.price_maxima[good].into(),
                        total: totals.price_totals[good].into(),
                        squares: totals.price_squares[good].into(),
                    },
                };
                MarketObservation { wages, prices }
            }),
            wealth_bins: control.wealth_bins.iter().map(|&n| u64::from(n)).collect(),
            regions: control.regions.iter().map(|&n| u64::from(n)).collect(),
        };
        snapshot.audit(self.metadata.initial_cash, self.metadata.initial_goods)?;
        self.account_research_readbacks();
        Ok(snapshot)
    }
    pub fn property_snapshot(
        &mut self,
    ) -> Result<Option<crate::domain::property::PropertySnapshot>> {
        match self.scenario.config().property.is_some() {
            false => Ok(None),
            true => {
                let state = self.download()?;
                crate::domain::property::PropertySnapshot::from_state(&self.scenario, &state)
            }
        }
    }
    fn read_state(&mut self, direction: CopyDirection) -> Result<Vec<u64>> {
        let source = match direction {
            CopyDirection::Save => &self.state,
            CopyDirection::Restore => &self.audited,
        };
        let mut values = decode(&self.device.read(source)?);
        self.readback_bytes += (values.len() * TICK_BYTES) as u64;
        let count = self.metadata.actors.len();
        let stride = self.config.state_stride as usize;
        if stride != count {
            for field in 1..self.scenario.fields() {
                values.copy_within(field * stride..field * stride + count, field * count);
            }
            values.truncate(count * self.scenario.fields());
        }
        Ok(values)
    }
    fn state_snapshot(&mut self, direction: CopyDirection) -> Result<State> {
        let values = self.read_state(direction)?;
        let research = self
            .research
            .as_ref()
            .map(|research| match direction {
                CopyDirection::Save => research.snapshot(&self.device),
                CopyDirection::Restore => research.audited_snapshot(&self.device),
            })
            .transpose()?;
        self.account_research_readbacks();
        Ok(State {
            people: self
                .demography
                .download(&self.device, &self.metadata, direction)?,
            day: match direction {
                CopyDirection::Save => self.day,
                CopyDirection::Restore => self.audited_day,
            },
            values,
            research,
            ..self.metadata.clone()
        })
    }
    pub fn download(&mut self) -> Result<State> {
        self.state_snapshot(CopyDirection::Save)
    }
    fn copy_audited(&mut self, direction: CopyDirection) -> Result<()> {
        let started = Instant::now();
        let profile =
            self.day == profiling_day() && std::env::var_os("BAD_ECON_PROFILE_STAGES").is_some();
        if profile {
            eprintln!(
                "COPYSHAPE words={} modulo={}",
                self.state.meta.num_elements(),
                self.state.meta.num_elements() % 4
            );
            self.device.sync()?;
            eprintln!("COPYWAIT {:.3}", started.elapsed().as_secs_f64() * 1000.0);
        }
        match direction {
            CopyDirection::Save => self.snapshot_copy.save_state(
                &self.device,
                &self.state,
                &self.audited,
                self.config.state_stride as usize,
                self.scenario.ring(),
            ),
            CopyDirection::Restore => {
                tensor::copy_contiguous(&self.device, &self.audited, &self.state);
                if let Some(mask) = self.snapshot_copy.ring_mask() {
                    tensor::reset_ring_mask(&self.device, mask, tensor::RingMaskState::Unknown);
                }
            }
        }
        if profile {
            self.device.sync()?;
            eprintln!("COPYSTATE {:.3}", started.elapsed().as_secs_f64() * 1000.0);
        }
        let research_started = Instant::now();
        if let Some(research) = &self.research {
            match direction {
                CopyDirection::Save => research.save_audited(&self.device)?,
                CopyDirection::Restore => research.restore_audited(&self.device)?,
            }
        }
        self.account_research_readbacks();
        if profile {
            self.device.sync()?;
            eprintln!(
                "COPYRESEARCH {:.3}",
                research_started.elapsed().as_secs_f64() * 1000.0
            );
        }
        let demography_started = Instant::now();
        self.demography.copy(&self.device, direction);
        self.device.sync()?;
        if profile {
            eprintln!(
                "COPYDEMOGRAPHY {:.3}",
                demography_started.elapsed().as_secs_f64() * 1000.0
            );
            eprintln!(
                "AUDIT {} Copy {:.3}",
                self.day,
                started.elapsed().as_secs_f64() * 1000.0
            );
        }
        Ok(())
    }
    pub fn advance(&mut self) -> Result<()> {
        match (&self.status, self.next_phase) {
            (RuntimeState::Fault { .. }, _) => {
                Err(anyhow!("reset or restore the faulted simulation"))
            }
            (_, Phase::Opening) => Ok(()),
            _ => Err(anyhow!(
                "finish the current diagnostic day before advancing"
            )),
        }?;
        let previous_execution = self.freight_execution;
        let previous_status = self.status.clone();
        self.freight_execution = match self.device.cube {
            cubecl::Device::Wgpu(_) | cubecl::Device::Metal(_) => FreightExecution::Buffered,
            _ => FreightExecution::Synchronous,
        };
        let mut attempt = AdvanceAttempt::Pending;
        let mut result = Ok(());
        while attempt == AdvanceAttempt::Pending {
            self.timings.clear();
            result = Phase::ALL.into_iter().try_for_each(|phase| -> Result<()> {
                let start = Instant::now();
                let evidence = self.phase(phase, Trace::Discard)?;
                self.timings.push(PhaseTiming {
                    phase,
                    milliseconds: start.elapsed().as_secs_f64() * MILLISECONDS_PER_SECOND,
                    events: evidence.events,
                    frontier_depth: evidence.frontier_depth,
                });
                Ok(())
            });
            let required = result
                .as_ref()
                .err()
                .and_then(|error| error.downcast_ref::<ClaimCapacityDemand>())
                .map(|demand| demand.records.get());
            attempt = match required {
                Some(records) => {
                    match settlement::reserve_claims_for_replay(
                        &self.device,
                        &mut self.config,
                        records,
                    ) {
                        Ok(()) => {
                            self.status = previous_status.clone();
                            self.market_graphs = market_graphs::MarketGraphs::default();
                            AdvanceAttempt::Pending
                        }
                        Err(error) => {
                            self.status = RuntimeState::Fault {
                                last_audited_day: self.audited_day,
                                message: error.to_string(),
                            };
                            result = Err(error);
                            AdvanceAttempt::Complete
                        }
                    }
                }
                None => AdvanceAttempt::Complete,
            };
        }
        self.freight_execution = previous_execution;
        result
    }
    pub fn command(&mut self, command: SimulationCommand) -> Result<()> {
        match command {
            SimulationCommand::Pause => {
                match self.status {
                    RuntimeState::Fault { .. } => (),
                    _ => self.status = RuntimeState::Paused,
                };
                Ok(())
            }
            SimulationCommand::Resume => match self.status {
                RuntimeState::Fault { .. } => {
                    Err(anyhow!("reset or restore the faulted simulation"))
                }
                _ => {
                    self.status = RuntimeState::Running;
                    Ok(())
                }
            },
            SimulationCommand::AdvanceDay => {
                self.advance()?;
                self.status = RuntimeState::Paused;
                Ok(())
            }
            SimulationCommand::Reset(scenario) => {
                let initial = scenario.initial()?;
                self.replace(*scenario, initial)
            }
            SimulationCommand::Restore(checkpoint) => {
                let restored = checkpoint.restore()?;
                self.replace(restored.scenario, restored.state)
            }
        }
    }
    fn replace(&mut self, scenario: Scenario, state: State) -> Result<()> {
        let device = self.device.clone();
        #[cfg(test)]
        let registry_execution = self.registry_execution;
        *self = Self::from_state(scenario, state, device)?;
        #[cfg(test)]
        {
            self.registry_execution = registry_execution;
        }
        Ok(())
    }
    pub fn checkpoint(&mut self) -> Result<Checkpoint> {
        let state = self.state_snapshot(CopyDirection::Restore)?;
        Checkpoint::new(&self.scenario, &state)
    }
    pub fn inspect(&mut self, request: InspectionRequest) -> Result<Inspection> {
        let actor = self
            .metadata
            .actors
            .iter()
            .position(|a| a.identity == request.identity())
            .context("unknown actor identity")?;
        let parameters = self.parameters(Phase::Audit, Market::Labor, actor, 1)?;
        self.economy(
            EconomyStage::Inspect,
            self.scenario.fields(),
            &parameters,
            &self.summary,
        );
        let values = decode(&self.device.read_range(
            &self.summary,
            0,
            self.scenario.fields() * TICK_WORDS,
        )?);
        self.readback_bytes += (values.len() * TICK_BYTES) as u64;
        let contract_employer = self
            .metadata
            .actors
            .assigned_identity(values[Field::Job as usize]);
        Ok(Inspection {
            railway_business: (self.metadata.actors.at(actor).kind == Good::Railway as u32).then(
                || railway::RailwayBusiness {
                    built_segments: values[Field::RailBuilt as usize],
                    operating_segments: values[Field::RailOperating as usize],
                    daily_receipts: values[Field::RailRevenue as usize],
                    daily_capacity: values[Field::RailCapacity as usize],
                    available_units: values[Field::Railway as usize],
                },
            ),
            business_account: (self.metadata.actors.at(actor).kind > 0).then(|| {
                diagnostics::BusinessAccount {
                    opening_cash: values[Field::BusinessOpeningCash as usize],
                    closing_cash: values[Field::BusinessClosingCash as usize],
                    revenue: values[Field::BusinessRevenue as usize],
                }
            }),
            production_method: ProductionMethod::from_index(values[Field::ActiveMethod as usize])?,
            labor_minutes: values[Field::LaborMinutes as usize],
            output_ticks: values[Field::OutputTicks as usize],
            actor: self.metadata.actors.at(actor),
            day: self.day,
            weather: self.scenario.weather(self.day),
            cash: values[Field::Cash as usize],
            gross_income: values[Field::GrossIncome as usize],
            payroll_reserve: values[Field::PayrollReserve as usize],
            income_tax: values[Field::TaxTreasury as usize],
            employee_charges: values[Field::ChargeTreasury as usize],
            rent_collected: values[Field::RentTreasury as usize],
            rent_arrears: values[Field::RentArrears as usize],
            landlord_identity: values[Field::Landlord as usize]
                .checked_sub(1)
                .and_then(|landlord| self.metadata.actors.get(landlord as usize))
                .map(|actor| actor.identity),
            rental_business: (self.metadata.actors.at(actor).kind == Good::Housing as u32).then(
                || crate::domain::landlords::RentalBusiness {
                    homes: values[Field::OccupiedPlots as usize]
                        * values[Field::OutputTicks as usize],
                    income: values[Field::RentalIncome as usize],
                },
            ),
            housing_state: match self.metadata.actors.at(actor).kind {
                0 => Some(crate::domain::HousingState::from_index(
                    values[Field::HousingStatus as usize],
                )?),
                _ => None,
            },
            rent_overdue_days: values[Field::RentOverdueDays as usize],
            inventory: values[Field::Grain as usize..Field::Escrow as usize].try_into()?,
            transit: values[Field::Transit as usize],
            fatigue: values[Field::Fatigue as usize],
            job_days_remaining: values[Field::JobDaysRemaining as usize],
            contract_employer,
            employer: self
                .metadata
                .actors
                .assigned_identity(values[Field::Employer as usize])
                .or(contract_employer)
                .unwrap_or(0),
            owner_identity: self
                .metadata
                .actors
                .at(self.metadata.actors.at(actor).owner as usize)
                .identity,
            available_minutes: values[Field::Time as usize],
            living_members: values[Field::LivingMembers as usize],
            heating_need: values[Field::HeatingNeed as usize],
            heating_served: values[Field::HeatingServed as usize],
            equipment_condition: values[Field::EquipmentCondition as usize],
            safety_provision: values[Field::SafetyProvision as usize],
            business_scale: (self.metadata.actors.at(actor).kind > 0).then(|| {
                crate::domain::enterprise::BusinessScale::for_business(
                    &self.metadata.actors.at(actor),
                    values[Field::OccupiedPlots as usize] * values[Field::OutputTicks as usize],
                )
            }),
            enterprise_model: match values[Field::EnterpriseModel as usize] {
                1 => crate::domain::enterprise::EnterpriseModel::Distributor,
                _ => crate::domain::enterprise::EnterpriseModel::Producer,
            },
            enterprise_state: match self.metadata.actors.at(actor).kind {
                0 => None,
                _ => Some(crate::domain::enterprise::EnterpriseState::from_index(
                    values[Field::EnterpriseStatus as usize],
                )?),
            },
            company_identity: match self.metadata.actors.at(actor).kind {
                0 => 0,
                _ => self.metadata.company_identity(request.identity()),
            },
            company_sites: match self.metadata.actors.at(actor).kind {
                0 => Vec::new(),
                _ => self.metadata.company_sites(request.identity()),
            },
            workers_per_plot: values[Field::WorkersPerPlot as usize],
            scale_bonus_per_mille: match self.metadata.actors.at(actor).kind {
                0 => 0,
                _ => {
                    crate::domain::industry::ProductionScale::for_plots(
                        values[Field::OccupiedPlots as usize],
                    )
                    .bonus_per_mille
                }
            },
            construction_minutes_remaining: values[Field::ConstructionRemaining as usize],
            property_region: values[Field::PropertyRegion as usize],
            occupied_plots: values[Field::OccupiedPlots as usize],
            reserved_plots: values[Field::ReservedPlots as usize],
            subsistence_plots: values[Field::LandPlots as usize],
            displaced_plots: values[Field::DisplacedPlots as usize],
            project_status: crate::domain::property::ProjectStatus::from_index(
                values[Field::ProjectStatus as usize],
            )?,
            pending_capacity: values[Field::PendingCapacity as usize],
            wage: values[Field::Wage as usize],
            price: values[Field::Price as usize],
            planned_packets: values[Field::Plan as usize],
            open_slots: values[Field::Slots as usize],
        })
    }
}
#[derive(Clone, Copy)]
enum CopyDirection {
    Save,
    Restore,
}

struct Execution {
    decisions: Vec<Decision>,
    depth: usize,
    events: usize,
    readback_bytes: u64,
    status_start: usize,
}
fn execute(
    device: &Device,
    buffers: SchedulerBuffers<'_>,
    count: usize,
    trace: Trace,
) -> Result<Execution> {
    let SchedulerBuffers {
        scratch, control, ..
    } = buffers;
    let mut result = Execution {
        decisions: Vec::new(),
        depth: 0,
        events: 0,
        readback_bytes: 0,
        status_start: 0,
    };
    match count {
        0 => (),
        _ => {
            let total = count * MAX_LEGS;
            let parameters = SchedulerParameters {
                count: count as u32,
                ..Default::default()
            };
            let launch = |entry: SchedulerStage, size, parameters: SchedulerParameters| {
                CubeBackend::schedule(device, entry, size, buffers, parameters);
            };
            launch(
                SchedulerStage::Initialize,
                total.max(CONTROL_WORDS),
                parameters,
            );
            let mut source = 0;
            let mut width = 1;
            while width < total {
                launch(
                    SchedulerStage::Merge,
                    total,
                    SchedulerParameters {
                        width: width as u32,
                        source: source as u32,
                        ..parameters
                    },
                );
                source = total - source;
                width *= 2;
            }
            launch(
                SchedulerStage::Dependencies,
                total,
                SchedulerParameters {
                    source: source as u32,
                    ..parameters
                },
            );
            launch(SchedulerStage::CheckChains, count, parameters);
            let counters = SchedulerControl::new(&device.read_range(control, 0, CONTROL_WORDS)?)?;
            result.readback_bytes += (CONTROL_WORDS * WORD_BYTES) as u64;
            let mut status_offset = 0;
            counters.faults.into_result()?;
            match counters.shape() {
                DependencyShape::Chains => {
                    launch(SchedulerStage::Chains, count, parameters);
                    let counters =
                        SchedulerControl::new(&device.read_range(control, 0, CONTROL_WORDS)?)?;
                    result.readback_bytes += (CONTROL_WORDS * WORD_BYTES) as u64;
                    result.depth = counters.progress;
                    result.events = counters.completed;
                    counters.faults.into_result()?;
                }
                DependencyShape::Frontiers => {
                    let mut pending = count;
                    while pending > 0 {
                        launch(SchedulerStage::Clear, 1, parameters);
                        launch(
                            SchedulerStage::Frontier,
                            count,
                            SchedulerParameters {
                                status_offset: status_offset as u32,
                                ..parameters
                            },
                        );
                        status_offset = count - status_offset;
                        result.depth += 1;
                        let counters = SchedulerControl::new(&device.read_range(
                            control,
                            0,
                            CONTROL_WORDS,
                        )?)?;
                        result.readback_bytes += (CONTROL_WORDS * WORD_BYTES) as u64;
                        pending = counters.progress;
                        result.events = counters.completed;
                        counters.faults.into_result()?;
                        match result.depth <= count {
                            true => Ok(()),
                            false => Err(anyhow!("stalled dependency graph")),
                        }?;
                    }
                }
            }
            result.status_start = total * ScratchLayout::STATUSES + status_offset;
            match trace {
                Trace::Record => {
                    let statuses = device.read_range(
                        scratch,
                        total * ScratchLayout::STATUSES + status_offset,
                        count,
                    )?;
                    result.readback_bytes += (count * WORD_BYTES) as u64;
                    result.decisions = statuses
                        .into_iter()
                        .map(|code| EventStatus::try_from(code)?.decision())
                        .collect::<Result<Vec<_>>>()?
                        .into_iter()
                        .flatten()
                        .collect();
                    result.events = result.decisions.len();
                }
                Trace::Discard => (),
            }
        }
    }
    Ok(result)
}

pub struct LedgerResult {
    pub state: Vec<u64>,
    pub decisions: Vec<Decision>,
    pub frontier_depth: usize,
}
pub fn ledger(device: &Device, balances: &[u64], bundles: &[Bundle]) -> Result<LedgerResult> {
    OrderedEvents::new(bundles)?;
    match bundles
        .iter()
        .flat_map(Bundle::legs)
        .all(|leg| (leg.coordinate.0 as usize) < balances.len())
    {
        true => Ok(()),
        false => Err(anyhow!("invalid ledger coordinate")),
    }?;
    let state = device.tensor(bytemuck::cast_slice(&encode(balances)));
    let events = device.tensor(bytemuck::cast_slice(
        &bundles.iter().map(Bundle::encoded_legs).collect::<Vec<_>>(),
    ));
    let scratch =
        device.empty((bundles.len() * ScratchLayout::WORDS_PER_EVENT).max(MIN_BUFFER_WORDS));
    let links = device.empty((bundles.len() * LinkLayout::WORDS_PER_EVENT).max(MIN_BUFFER_WORDS));
    let control = device.tensor(bytemuck::cast_slice(&[ControlLayout::zeroed()]));
    let result = execute(
        device,
        SchedulerBuffers {
            state: &state,
            events: &events,
            scratch: &scratch,
            control: &control,
            links: &links,
        },
        bundles.len(),
        Trace::Record,
    )?;
    let mut recorded = result.decisions.into_iter();
    let decisions = bundles
        .iter()
        .map(|bundle| match bundle.legs().is_empty() {
            true => Ok(Decision::Accepted),
            false => recorded.next().context("missing ledger decision"),
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(LedgerResult {
        state: decode(&device.read(&state)?),
        decisions,
        frontier_depth: result.depth,
    })
}
fn encode(values: &[u64]) -> Vec<TickLayout> {
    values.iter().copied().map(TickLayout::from).collect()
}
fn decode(words: &[u32]) -> Vec<u64> {
    bytemuck::cast_slice::<u32, TickLayout>(words)
        .iter()
        .copied()
        .map(u64::from)
        .collect()
}

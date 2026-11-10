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
    ledger::{Bundle, Decision, MAX_LEGS, Schedule, TickLayout},
    runtime::{PhaseTiming, RuntimeState, SimulationCommand},
};
use anyhow::{Context, Result, anyhow};
use burn_wgpu::{CubeBackend, CubeTensor};
use bytemuck::Zeroable;
use economy::{EconomyStage, markets, scheduler};
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
use system::System;

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
            (Field::EnterpriseStatus as usize * actors.len() + start) * TICK_WORDS,
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
        self.device
            .client
            .memory_report(cubecl::MemoryScope::Device)
            .usage()
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
    fn from_state(scenario: Scenario, mut initial: State, device: Device) -> Result<Self> {
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
        let demography = demography::DemographyRuntime::new(&device, &scenario, &initial)?;

        let config = configuration::Configuration::new(&device, &scenario, &initial);
        let actor_layouts = initial
            .actors
            .iter()
            .map(|actor| ActorLayout::new(&actor, &scenario))
            .collect::<ActorMetadataRows>();
        let actors = device.tensor(&actor_layouts.words());
        let spatial = device.tensor(&SpatialIndex::new(&scenario, &initial).words);
        let state = device.tensor(bytemuck::cast_slice(&encode(&initial.values)));
        let audited = device.empty(initial.values.len() * TICK_WORDS);
        let snapshot_copy = tensor::SnapshotCopy::new(&device, initial.values.len() * TICK_WORDS);
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
        let searches = markets::SearchBuffers::new(&device, initial.actors.len());
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
        };
        simulation.snapshot = simulation.observe()?;
        simulation.copy_audited(CopyDirection::Save)?;
        simulation.device.reclaim_memory(0)?;
        match simulation.memory_usage().bytes_reserved <= simulation.device.memory_budget.bytes() {
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
            scan: parallel::ScanMode::new(&self.device),
            day,
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
        FaultFlags::read(&control)?.into_result()
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
                self.device.reclaim_memory(0)?;
                let memory = self.memory_usage();
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
                (0..MATCHING_ROUNDS).try_for_each(|round| {
                    self.matching_round = round;
                    self.market(Market::Labor, phase, trace, &mut evidence)
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
                self.scenario
                    .goods_markets()
                    .into_iter()
                    .try_for_each(|market| {
                        (0..MATCHING_ROUNDS).try_for_each(|round| {
                            self.matching_round = round;
                            self.market(market, phase, trace, &mut evidence)
                        })?;
                        if let (Market::Household, Some(_)) = (market, self.scenario.config().rent)
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
                    })?;
                self.market_result(&mut evidence)?;
                self.apply_feedback()
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
        let profile_device = self.device.clone();
        let profile_day = self.day;
        let profile =
            profile_day == profiling_day() && std::env::var_os("BAD_ECON_PROFILE_STAGES").is_some();
        let profile_time = std::cell::Cell::new(Instant::now());
        let checkpoint = |label: &str| {
            if profile {
                profile_device
                    .sync()
                    .expect("audit profile synchronization");
                let now = Instant::now();
                eprintln!(
                    "AUDIT {profile_day} {label} {:.3}",
                    now.duration_since(profile_time.replace(now)).as_secs_f64() * 1000.0
                );
            }
        };
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
        let count = records.iter().filter(|record| record.request > 0).count();
        let newborns = records.iter().filter(|record| record.birth > 0).count();
        if count == 0 && newborns > 0 {
            let next_identity = self
                .metadata
                .next_identity
                .checked_add(newborns as u64)
                .context("identity capacity exhausted")?;
            let grown = self.grow_births(records)?;
            let population = grown.observation(&self.device, &self.state, &self.config)?;
            self.device.reclaim_memory(0)?;
            match self.memory_usage().bytes_reserved <= self.device.memory_budget.bytes() {
                true => (),
                false => Err(anyhow!(
                    "person registry capacity exhausted while observing growth"
                ))?,
            }
            self.demography = grown;
            self.metadata.next_identity = next_identity;
            self.snapshot.population = population;
        }
        let plan =
            registry::RegistryPlan::new(&self.scenario, &self.metadata, &self.migrations, records)?;
        checkpoint("RegistryPlan");
        #[cfg(test)]
        let plan = match self.registry_execution {
            RegistryExecution::Gpu => plan,
            RegistryExecution::Reference => None,
        };
        let entries = registry::entries::EntryGrowth::new(
            &self.scenario,
            &self.metadata,
            &self.migrations,
            records,
        )?;
        checkpoint("EntryGrowth");
        #[cfg(test)]
        let entries = match self.registry_execution {
            RegistryExecution::Gpu => entries,
            RegistryExecution::Reference => None,
        };
        match (count, plan, entries) {
            (0, _, _) => (),
            (_, Some(plan), _) => self.commit_registry_audited(plan, records, audit)?,
            (_, None, Some(growth)) => {
                self.commit_entries(&growth.requests, growth.plan, records)?
            }
            (_, None, None) => {
                let mut state = self.download()?;
                let estimated =
                    (state.actors.len() + count) as u64 * self.scenario.fields() as u64 * 24
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
            }
        }
        Ok(())
    }
    fn commit_entries(
        &mut self,
        entries: &registry::entries::EntryRequests,
        mut plan: registry::RegistryPlan,
        births: &[PendingRecord],
    ) -> Result<()> {
        let profile_device = self.device.clone();
        let profile_day = self.day;
        let profile =
            profile_day == profiling_day() && std::env::var_os("BAD_ECON_PROFILE_STAGES").is_some();
        let profile_time = std::cell::Cell::new(Instant::now());
        let checkpoint = |label: &str| {
            if profile {
                profile_device
                    .sync()
                    .expect("audit profile synchronization");
                let now = Instant::now();
                eprintln!(
                    "AUDIT {profile_day} {label} {:.3}",
                    now.duration_since(profile_time.replace(now)).as_secs_f64() * 1000.0
                );
            }
        };
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
            next_identity,
        )?;
        checkpoint("EntryBatch");
        let count = batch.metadata.actors.len();
        let additional = count as u64 * self.scenario.fields() as u64 * 16 + count as u64 * 192;
        self.device.reclaim_memory(additional)?;
        match self
            .memory_usage()
            .bytes_reserved
            .saturating_add(additional)
            <= self.device.memory_budget.bytes()
        {
            true => (),
            false => Err(anyhow!("registry growth exceeds the device memory budget"))?,
        }
        let state = batch.grow(
            &self.device,
            &self.state,
            self.metadata.actors.len(),
            self.scenario.fields(),
        );
        checkpoint("GrowState");
        plan.migrations
            .extend_households(&batch.metadata.actors, &batch.formations);
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
        let started = Instant::now();
        let config = self.config.updated(
            &self.device,
            &self.scenario,
            &self.metadata.actors,
            &metadata,
        );
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
            .map(|research| research.with_actors(&metadata.actors))
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
            audited: self
                .device
                .empty(count * self.scenario.fields() * TICK_WORDS),
            snapshot_copy: tensor::SnapshotCopy::new(
                &self.device,
                count * self.scenario.fields() * TICK_WORDS,
            ),
            actors,
            config,
            spatial,
            proposals: self.device.empty(count * ProposalLayout::WORDS),
            events: self.device.empty(event_words),
            scratch: self.device.empty(scratch_words),
            searches: markets::SearchBuffers::new(&self.device, count),
            market_sort: economy::sorting::MarketSort::default(),
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
        };
        checkpoint("Buffers");
        match settlement {
            Some(batch) => batch.apply(&next.device, &next.state, count),
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
        next.snapshot = next.observe()?;
        checkpoint("Observe");
        next.device.reclaim_memory(0)?;
        match next.memory_usage().bytes_reserved <= next.device.memory_budget.bytes() {
            true => *self = next,
            false => Err(anyhow!(
                "registry allocation exceeds the device memory budget"
            ))?,
        }
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
        let profile_device = self.device.clone();
        let profile_day = self.day;
        let profile =
            profile_day == profiling_day() && std::env::var_os("BAD_ECON_PROFILE_STAGES").is_some();
        let profile_time = std::cell::Cell::new(Instant::now());
        let checkpoint = |label: &str| {
            if profile {
                profile_device
                    .sync()
                    .expect("audit profile synchronization");
                let now = Instant::now();
                eprintln!(
                    "AUDIT {profile_day} {label} {:.3}",
                    now.duration_since(profile_time.replace(now)).as_secs_f64() * 1000.0
                );
            }
        };
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
        let population = grown
            .as_ref()
            .map(|grown| grown.observation(&self.device, &self.state, &self.config))
            .transpose()?;
        checkpoint("RegistryPopulation");
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
        self.device.reclaim_memory(0)?;
        match self.memory_usage().bytes_reserved <= self.device.memory_budget.bytes() {
            true => (),
            false => Err(anyhow!(
                "registry capacity exhausted while preparing GPU transitions"
            ))?,
        }
        match &settlement {
            Some(batch) => {
                batch.apply(&self.device, &self.state, self.metadata.actors.len());
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
        let observation = self.observe_rings(rings).and_then(|snapshot| {
            self.device.reclaim_memory(0)?;
            match self.memory_usage().bytes_reserved <= self.device.memory_budget.bytes() {
                true => Ok(snapshot),
                false => Err(anyhow!(
                    "registry capacity exhausted while auditing GPU transitions"
                )),
            }
        });
        checkpoint("RegistryObservation");
        match observation {
            Ok(mut snapshot) => {
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
                spatial: &self.spatial,
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
        economy::railway::launch(
            &self.device,
            &self.state,
            &self.actors,
            &self.config,
            &self.control,
            stage,
        );
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
                FaultFlags::new(counters.faults)?.into_result()?;
                evidence.frontier_depth = evidence.frontier_depth.max(counters.progress as usize);
                evidence.events += counters.completed as usize;
            }
            #[cfg(test)]
            _ => (),
        }
        Ok(())
    }
    fn market(
        &mut self,
        market: Market,
        phase: Phase,
        trace: Trace,
        evidence: &mut PhaseEvidence,
    ) -> Result<()> {
        let regional_transport = !self.scenario.config().railway.companies.is_empty()
            && matches!(
                market,
                Market::Input(_) | Market::Consumer(_) | Market::Household
            );
        match self.market_execution {
            _ if regional_transport => self.device_market(market, phase, trace, evidence),
            MarketExecution::Grouped => self.device_market(market, phase, trace, evidence),
            #[cfg(test)]
            _ => self.original_market(market, phase, trace, evidence),
        }
    }
    fn device_market(
        &mut self,
        market: Market,
        phase: Phase,
        trace: Trace,
        evidence: &mut PhaseEvidence,
    ) -> Result<()> {
        let submission_started = Instant::now();
        let regional_transport = !self.scenario.config().railway.companies.is_empty()
            && matches!(
                market,
                Market::Input(_) | Market::Consumer(_) | Market::Household
            );
        let count = self.metadata.actors.len();
        let mut parameters = self.parameters(phase, market, 0, count)?;
        parameters.discovery = Discovery::Indexed;
        let profile = self.day + 1 == profiling_day()
            && std::env::var_os("BAD_ECON_PROFILE_STAGES").is_some();
        let device_profile = profiling::MarketProfile::new(self.day + 1);
        let launch = |stage, invocations, parameters: &EconomyParameters| {
            let started = Instant::now();
            let window = device_profile.start(&self.device);
            let dispatch = match stage {
                EconomyStage::GroupExecute => {
                    markets::dispatch(&self.searches.dispatches, markets::RANGE_DISPATCH)
                }
                EconomyStage::GroupPrepare
                | EconomyStage::GroupFinalize
                | EconomyStage::GroupRadixLocal
                | EconomyStage::GroupRadixScatter
                | EconomyStage::GroupRemainder
                | EconomyStage::GroupRanges => {
                    markets::dispatch(&self.searches.dispatches, markets::SORT_DISPATCH)
                }
                _ => self.device.count(invocations),
            };
            match stage {
                EconomyStage::MarketBoundary => {
                    markets::prepare_dispatch(
                        &self.device,
                        &self.control,
                        &self.searches.dispatches,
                        count,
                    );
                }
                _ => economy::EconomySystem::launch(
                    &self.device,
                    dispatch,
                    cubecl::CubeDim::new_1d(self.device.workgroup),
                    economy::EconomyInputLaunch {
                        state: self.state.clone().into_buffer_arg(),
                        actors: self.actors.clone().into_buffer_arg(),
                        config: self.config.launch(),
                        spatial: self.spatial.clone().into_buffer_arg(),
                        proposals: self.proposals.clone().into_buffer_arg(),
                        bundles: self.scratch.clone().into_buffer_arg(),
                        control: self.control.clone().into_buffer_arg(),
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
        let initialization = match (market, self.matching_round) {
            (Market::Input(_), 0) => EconomyStage::InputMarketStart,
            (Market::Input(_), _) => EconomyStage::InputMarketRepeat,
            _ => EconomyStage::MarketStart,
        };
        let initialization_count = match market {
            Market::Input(_) => count,
            _ => 1,
        };
        launch(initialization, initialization_count, &parameters);
        if matches!(
            market,
            Market::Labor | Market::Input(_) | Market::Consumer(_) | Market::Household
        ) {
            let side = self
                .scenario
                .config()
                .map_size
                .div_ceil(self.scenario.config().search_radius);
            let members = match market {
                Market::Household | Market::Input(Good::Bread) | Market::Consumer(Good::Bread) => {
                    count
                }
                _ => self.metadata.actors.firms(),
            };
            launch(EconomyStage::IndexCount, members, &parameters);
            parallel::scan(
                &self.device,
                &self.scratch,
                count,
                members.div_ceil(self.device.workgroup as usize),
            );
            economy::indexing::MarketIndex::launch(
                &self.device,
                self.device.count(members.max((side * side + 1) as usize)),
                cubecl::CubeDim::new_1d(self.device.workgroup),
                economy::indexing::IndexInputLaunch {
                    config: self.config.launch(),
                    spatial: self.spatial.clone().into_buffer_arg(),
                    scratch: self.scratch.clone().into_buffer_arg(),
                    control: self.control.clone().into_buffer_arg(),
                    parameters: parameters.launch(),
                },
            );
            if matches!(
                market,
                Market::Household | Market::Input(Good::Bread) | Market::Consumer(Good::Bread)
            ) {
                launch(
                    EconomyStage::IndexPrices,
                    count.div_ceil(DISCOVERY_BLOCK as usize) * self.device.workgroup as usize,
                    &parameters,
                );
            }
        }
        let search = |stage| {
            let started = Instant::now();
            let window = device_profile.start(&self.device);
            let workgroup = match stage {
                economy::search::SearchStage::Search => 32,
                _ => self.device.workgroup,
            };
            let dispatch = match stage {
                economy::search::SearchStage::Index => self.device.count(
                    count.div_ceil(DISCOVERY_BLOCK as usize) * self.device.workgroup as usize,
                ),
                economy::search::SearchStage::Prepare
                | economy::search::SearchStage::PrepareRepeat
                | economy::search::SearchStage::PrepareCandidates => self.device.count(count),
                economy::search::SearchStage::PrepareSelected => {
                    markets::dispatch(&self.searches.dispatches, markets::PREPARATION_DISPATCH)
                }
                economy::search::SearchStage::Search => {
                    markets::dispatch(&self.searches.dispatches, markets::SEARCH_DISPATCH)
                }
                economy::search::SearchStage::Compact => {
                    markets::dispatch(&self.searches.dispatches, markets::SEARCH_COMPACT_DISPATCH)
                }
            };
            economy::search::MarketSearch::launch(
                &self.device,
                dispatch,
                cubecl::CubeDim::new_1d(workgroup),
                economy::search::SearchInputLaunch {
                    state: self.state.clone().into_buffer_arg(),
                    actors: self.actors.clone().into_buffer_arg(),
                    config: self.config.launch(),
                    spatial: self.spatial.clone().into_buffer_arg(),
                    proposals: self.proposals.clone().into_buffer_arg(),
                    searches: self.searches.requests.clone().into_buffer_arg(),
                    successes: self.searches.successes.clone().into_buffer_arg(),
                    scratch: self.scratch.clone().into_buffer_arg(),
                    control: self.control.clone().into_buffer_arg(),
                    parameters: parameters.launch(),
                    status_reset: match (market, trace) {
                        (Market::Input(_), _) => economy::search::StatusReset::Active,
                        (_, Trace::Record) => economy::search::StatusReset::Full,
                        (_, Trace::Discard) => economy::search::StatusReset::Active,
                    },
                    stage,
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
        if matches!(market, Market::Consumer(_) | Market::Household) {
            search(economy::search::SearchStage::Index);
        }
        match market {
            Market::Labor | Market::Payroll | Market::Distribution => {
                launch(EconomyStage::ProposeCompact, count, &parameters);
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
                    (Market::Input(_), 0) => economy::search::SearchStage::PrepareCandidates,
                    (Market::Input(_), _) => economy::search::SearchStage::PrepareSelected,
                    _ => economy::search::SearchStage::Prepare,
                };
                search(stage);
                launch(EconomyStage::MarketBoundary, 1, &parameters);
                search(economy::search::SearchStage::Search);
                parallel::scan(
                    &self.device,
                    &self.searches.successes,
                    markets::search_counts_offset(count),
                    markets::search_blocks(count),
                );
                search(economy::search::SearchStage::Compact);
            }
        }
        launch(EconomyStage::MarketBoundary, 1, &parameters);
        let freight = |stage| {
            economy::railway::settlement::launch(
                &self.device,
                &self.state,
                &self.actors,
                &self.config,
                &self.proposals,
                &self.scratch,
                &self.control,
                &parameters,
                stage,
            );
        };
        if regional_transport {
            freight(economy::railway::settlement::Stage::Snapshot);
        }
        parameters.sort_source = self.market_sort.run(
            &self.device,
            &self.scratch,
            &self.control,
            &self.searches.dispatches,
            &parameters,
        )?;
        launch(EconomyStage::GroupRanges, count, &parameters);
        launch(EconomyStage::MarketBoundary, 1, &parameters);
        if matches!(
            market,
            Market::Labor | Market::Input(_) | Market::Consumer(_) | Market::Household
        ) {
            launch(EconomyStage::GroupPrepare, count, &parameters);
        }
        match market {
            Market::Payroll => economy::batch::payroll::PayrollSettlement::launch(
                &self.device,
                markets::dispatch(&self.searches.dispatches, markets::QUEUE_DISPATCH),
                cubecl::CubeDim::new_1d(self.device.workgroup),
                economy::batch::payroll::PayrollInputLaunch {
                    state: self.state.clone().into_buffer_arg(),
                    actors: self.actors.clone().into_buffer_arg(),
                    config: self.config.launch(),
                    proposals: self.proposals.clone().into_buffer_arg(),
                    scratch: self.scratch.clone().into_buffer_arg(),
                    control: self.control.clone().into_buffer_arg(),
                    parameters: parameters.launch(),
                },
            ),
            Market::Labor if self.scenario.config().bargaining.is_none() => {
                economy::batch::labor::LaborSettlement::launch(
                    &self.device,
                    markets::dispatch(&self.searches.dispatches, markets::QUEUE_DISPATCH),
                    cubecl::CubeDim::new_1d(self.device.workgroup),
                    economy::batch::labor::LaborInputLaunch {
                        state: self.state.clone().into_buffer_arg(),
                        config: self.config.launch(),
                        scratch: self.scratch.clone().into_buffer_arg(),
                        control: self.control.clone().into_buffer_arg(),
                        parameters: parameters.launch(),
                    },
                )
            }
            Market::Consumer(_) => economy::batch::ConsumerSettlement::launch(
                &self.device,
                markets::dispatch(&self.searches.dispatches, markets::QUEUE_DISPATCH),
                cubecl::CubeDim::new_1d(self.device.workgroup),
                economy::batch::SettlementInputLaunch {
                    state: self.state.clone().into_buffer_arg(),
                    actors: self.actors.clone().into_buffer_arg(),
                    config: self.config.launch(),
                    scratch: self.scratch.clone().into_buffer_arg(),
                    control: self.control.clone().into_buffer_arg(),
                    parameters: parameters.launch(),
                },
            ),
            _ => launch(EconomyStage::GroupExecute, count, &parameters),
        }
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
            use economy::railway::settlement::{self, Stage};
            let freight = |simulation: &Self, stage| {
                settlement::launch(
                    &simulation.device,
                    &simulation.state,
                    &simulation.actors,
                    &simulation.config,
                    &simulation.proposals,
                    &simulation.scratch,
                    &simulation.control,
                    &parameters,
                    stage,
                );
            };
            freight(self, Stage::Initialize);
            let mut pending = 1;
            while pending > 0 {
                freight(self, Stage::Reset);
                freight(self, Stage::Propose);
                freight(self, Stage::Commit);
                freight(self, Stage::Release);
                pending = settlement::pending(&self.device, &self.config)?;
                self.readback_bytes += WORD_BYTES as u64;
                self.fault_check()?;
            }
            freight(self, Stage::Accounts);
        }
        parameters.status_start = 4 * count as u32;
        if let (Market::Input(_) | Market::Consumer(_) | Market::Household, true) =
            (market, self.matching_round + 1 == MATCHING_ROUNDS)
        {
            markets::feedback_boundary(&self.device, &self.control, &self.feedback);
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
        device_profile.resolve(market, self.matching_round);
        Ok(())
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
            rings,
        };
        #[cfg(test)]
        match self.audit_execution {
            AuditExecution::Original => {
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
                match count {
                    0 => MarketObservation::default(),
                    _ => MarketObservation {
                        wages: QuoteDistribution {
                            count,
                            minimum: totals.wage_minima[good].into(),
                            maximum: totals.wage_maxima[good].into(),
                            total: totals.wage_totals[good].into(),
                            squares: totals.wage_squares[good].into(),
                        },
                        prices: QuoteDistribution {
                            count,
                            minimum: totals.price_minima[good].into(),
                            maximum: totals.price_maxima[good].into(),
                            total: totals.price_totals[good].into(),
                            squares: totals.price_squares[good].into(),
                        },
                    },
                }
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
    pub fn download(&mut self) -> Result<State> {
        let values = decode(&self.device.read(&self.state)?);
        self.readback_bytes += (values.len() * TICK_BYTES) as u64;
        let research = self
            .research
            .as_ref()
            .map(|research| research.snapshot(&self.device))
            .transpose()?;
        self.account_research_readbacks();
        Ok(State {
            people: self
                .demography
                .download(&self.device, &self.metadata, CopyDirection::Save)?,
            day: self.day,
            values,
            research,
            ..self.metadata.clone()
        })
    }
    fn copy_audited(&mut self, direction: CopyDirection) -> Result<()> {
        let started = Instant::now();
        if self.day == profiling_day() && std::env::var_os("BAD_ECON_PROFILE_STAGES").is_some() {
            eprintln!(
                "COPYSHAPE words={} modulo={}",
                self.state.meta.num_elements(),
                self.state.meta.num_elements() % 4
            );
            self.device.sync()?;
            eprintln!("COPYWAIT {:.3}", started.elapsed().as_secs_f64() * 1000.0);
        }
        match direction {
            CopyDirection::Save => {
                self.snapshot_copy
                    .save(&self.device, &self.state, &self.audited)
            }
            CopyDirection::Restore => {
                tensor::copy_contiguous(&self.device, &self.audited, &self.state)
            }
        }
        if let Some(research) = &self.research {
            match direction {
                CopyDirection::Save => research.save_audited(&self.device)?,
                CopyDirection::Restore => research.restore_audited(&self.device)?,
            }
        }
        self.account_research_readbacks();
        self.demography.copy(&self.device, direction);
        self.device.sync()?;
        if self.day == profiling_day() && std::env::var_os("BAD_ECON_PROFILE_STAGES").is_some() {
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
        self.timings.clear();
        Phase::ALL.into_iter().try_for_each(|phase| -> Result<()> {
            let start = Instant::now();
            let evidence = self.phase(phase, Trace::Discard)?;
            self.timings.push(PhaseTiming {
                phase,
                milliseconds: start.elapsed().as_secs_f64() * MILLISECONDS_PER_SECOND,
                events: evidence.events,
                frontier_depth: evidence.frontier_depth,
            });
            Ok(())
        })
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
        let values = decode(&self.device.read(&self.audited)?);
        self.readback_bytes += (values.len() * TICK_BYTES) as u64;
        let research = self
            .research
            .as_ref()
            .map(|research| research.audited_snapshot(&self.device))
            .transpose()?;
        self.account_research_readbacks();
        let state = State {
            people: self.demography.download(
                &self.device,
                &self.metadata,
                CopyDirection::Restore,
            )?,
            day: self.audited_day,
            values,
            research,
            ..self.metadata.clone()
        };
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
            employer: match values[Field::Employer as usize] {
                0 => values[Field::Job as usize],
                employer => employer,
            }
            .checked_sub(1)
            .and_then(|index| {
                self.metadata
                    .actors
                    .get(index as usize)
                    .map(|actor| actor.identity)
            })
            .unwrap_or(0),
            owner_identity: self
                .metadata
                .actors
                .at(self.metadata.actors.at(actor).owner as usize)
                .identity,
            available_minutes: values[Field::Time as usize],
            living_members: values[Field::LivingMembers as usize],
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
    let _ = Schedule::new(bundles)?;
    match bundles
        .iter()
        .flat_map(Bundle::legs)
        .all(|leg| (leg.coordinate.0 as usize) < balances.len())
    {
        true => Ok(()),
        false => Err(anyhow!("invalid ledger coordinate")),
    }?;
    let state = device.tensor(bytemuck::cast_slice(&encode(balances)));
    let events = device.tensor(&bundles.iter().flat_map(Bundle::encode).collect::<Vec<_>>());
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

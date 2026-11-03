pub mod analytics;
pub mod bargaining;
mod configuration;
mod demography;
mod economy;
mod enterprise;
mod enterprise_opportunities;
pub mod finance;
mod flags;
pub mod households;
mod layout;
mod management;
mod memory;
pub mod negotiation;
mod ops;
mod ownership;
mod parallel;
mod parameters;
mod payroll;
mod policy;
mod random;
mod reduction;
mod registry;
pub mod research;
mod rows;
mod scheduler;
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
use economy::EconomyStage;
use flags::{
    CONTROL_WORDS, ControlLayout, DependencyShape, EventStatus, FaultFlags, SchedulerControl,
};
use layout::*;
pub use memory::MemoryBudget;
pub use ops::Device;
use ops::{EconomyBuffers, EconomyOps, SchedulerBuffers};
use parameters::{Discovery, EconomyParameters, SchedulerParameters};
use policy::MATCHING_ROUNDS;
use scheduler::SchedulerStage;
use std::time::Instant;

const MILLISECONDS_PER_SECOND: f64 = 1000.0;

#[derive(Clone, Copy)]
enum MarketExecution {
    Grouped,
    #[cfg(test)]
    Reference,
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
    actors: CubeTensor,
    config: configuration::Configuration,
    spatial: CubeTensor,
    proposals: CubeTensor,
    events: CubeTensor,
    scratch: CubeTensor,
    #[cfg(test)]
    links: CubeTensor,
    market_execution: MarketExecution,
    control: CubeTensor,
    summary: CubeTensor,
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
                + initial.actors.len() as u64 * 96,
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
        let event_words = initial
            .actors
            .len()
            .div_ceil(SummaryLayout::ACTORS_PER_GROUP as usize)
            * SummaryLayout::WORDS;
        let scratch_words = initial.actors.len() * 5;
        #[cfg(test)]
        let batch = (scenario.config().batch_size as usize).min(initial.actors.len());
        #[cfg(test)]
        let event_words = event_words.max(batch * EVENT_WORDS);
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
        let ownership = match initial
            .ownership
            .iter()
            .any(|ownership| ownership.shares.len() > 1)
        {
            true => Some(ownership::OwnershipRuntime::new(&device, &initial)?),
            false => None,
        };
        let enterprise_requests = device.tensor(&vec![0; initial.actors.len()]);
        let day = initial.day;
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
            actors,
            config,
            spatial,
            proposals,
            events,
            scratch,
            #[cfg(test)]
            links,
            market_execution: MarketExecution::Grouped,
            control,
            summary,
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
            Phase::Commitment => (0..MATCHING_ROUNDS).try_for_each(|round| {
                self.matching_round = round;
                self.market(Market::Labor, phase, trace, &mut evidence)
            }),
            Phase::WorkPayment => {
                self.market(Market::Payroll, phase, trace, &mut evidence)?;
                self.local(phase)
            }
            Phase::ProvisionPurchase => {
                self.scenario
                    .goods_markets()
                    .into_iter()
                    .try_for_each(|market| {
                        (0..MATCHING_ROUNDS).try_for_each(|round| {
                            self.matching_round = round;
                            self.market(market, phase, trace, &mut evidence)
                        })
                    })?;
                self.apply_feedback()
            }
            Phase::AccountsDecisions => {
                self.local(phase)?;
                match &self.ownership {
                    Some(ownership) => {
                        ownership.distribute(&self.device, &self.state, &self.config)
                    }
                    None => self.market(Market::Distribution, phase, trace, &mut evidence),
                }
            }
            Phase::Audit => {
                if let Some(research) = &self.research {
                    research.phase(&self.device, &self.state, phase, self.day)?;
                }
                self.snapshot = self.observe()?;
                self.day = add(self.day, 1)?;
                self.snapshot.day = self.day;
                self.commit_enterprises()?;
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
    fn commit_enterprises(&mut self) -> Result<()> {
        enterprise::propose(
            &self.device,
            &self.state,
            &self.actors,
            &self.config,
            &self.spatial,
            &self.enterprise_requests,
            self.day as u32,
        );
        let requests = self.device.read(&self.enterprise_requests)?;
        let births = self.demography.pending_births(&self.device)?;
        self.readback_bytes += ((requests.len() + births.len()) * WORD_BYTES) as u64;
        let count = requests.iter().filter(|&&action| action > 0).count();
        let newborns = births.iter().filter(|&&birth| birth > 0).count();
        if count == 0 && newborns > 0 {
            let next_identity = self
                .metadata
                .next_identity
                .checked_add(newborns as u64)
                .context("identity capacity exhausted")?;
            let grown =
                self.demography
                    .grow_births(&self.device, &births, self.metadata.next_identity)?;
            let population = grown.observation(&self.device)?;
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
        let plan = registry::RegistryPlan::new(
            &self.scenario,
            &self.metadata,
            &self.migrations,
            &requests,
        )?;
        #[cfg(test)]
        let plan = match self.registry_execution {
            RegistryExecution::Gpu => plan,
            RegistryExecution::Reference => None,
        };
        let entries = registry::entries::EntryGrowth::new(
            &self.scenario,
            &self.metadata,
            &self.migrations,
            &requests,
        )?;
        #[cfg(test)]
        let entries = match self.registry_execution {
            RegistryExecution::Gpu => entries,
            RegistryExecution::Reference => None,
        };
        match (count, plan, entries) {
            (0, _, _) => (),
            (_, Some(plan), _) => self.commit_registry(plan, &births)?,
            (_, None, Some(growth)) => {
                self.commit_entries(&requests, &growth.requests, growth.plan, &births)?
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
                for (household, birth) in births
                    .into_iter()
                    .enumerate()
                    .filter(|(_, birth)| *birth > 0)
                {
                    let identity = state.next_identity;
                    state.next_identity = identity
                        .checked_add(1)
                        .context("identity capacity exhausted")?;
                    state.people.push(crate::domain::demography::Person {
                        identity,
                        household: state.actors.at(household).identity,
                        age_days: 0,
                        sex: match birth {
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
        requests: &[u32],
        entries: &registry::entries::EntryRequests,
        plan: registry::RegistryPlan,
        births: &[u32],
    ) -> Result<()> {
        let newborns = births.iter().filter(|&&birth| birth > 0).count();
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
            requests,
            entries,
            next_identity,
        )?;
        let count = batch.metadata.actors.len();
        let additional = count as u64 * self.scenario.fields() as u64 * 16 + count as u64 * 128;
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
        let mut metadata = batch.metadata;
        if let Some(actors) = plan.actors {
            metadata.actors = actors
                .iter()
                .chain(metadata.actors.iter().skip(actors.len()))
                .collect();
        }
        let config = configuration::Configuration::new(&self.device, &self.scenario, &metadata);
        let rows = metadata
            .actors
            .iter()
            .map(|actor| ActorLayout::new(&actor, &self.scenario))
            .collect::<ActorMetadataRows>();
        let actors = self.device.tensor(&rows.words());
        let spatial = self
            .device
            .tensor(&SpatialIndex::new(&self.scenario, &metadata).words);
        let mut demography = match newborns {
            0 => self.demography.clone(),
            _ => self
                .demography
                .grow_births(&self.device, births, self.metadata.next_identity)?,
        };
        demography.extend_firms(&self.device, count);
        let research = self
            .research
            .as_ref()
            .map(|research| research.with_actors(&self.device, &metadata.actors))
            .transpose()?;
        let mut transitions = entries.transitions.clone();
        transitions.resize(count, 0);
        let enterprise_requests = self.device.tensor(&transitions);
        let scratch_words = count * 5;
        let event_words =
            count.div_ceil(SummaryLayout::ACTORS_PER_GROUP as usize) * SummaryLayout::WORDS;
        #[cfg(test)]
        let batch_size = (self.scenario.config().batch_size as usize).min(count);
        #[cfg(test)]
        let scratch_words = scratch_words.max(batch_size * ScratchLayout::WORDS_PER_EVENT);
        #[cfg(test)]
        let event_words = event_words.max(batch_size * EVENT_WORDS);
        let ownership = self
            .ownership
            .as_ref()
            .map(|_| ownership::OwnershipRuntime::new(&self.device, &metadata))
            .transpose()?;
        let management = management::EnterpriseManagement::new(&self.device, &metadata);
        let mut next = Self {
            scenario: self.scenario.clone(),
            device: self.device.clone(),
            status: self.status.clone(),
            snapshot: self.snapshot.clone(),
            timings: self.timings.clone(),
            readback_bytes: self.readback_bytes + batch.readback_bytes,
            metadata,
            day: self.day,
            audited_day: self.audited_day,
            next_phase: self.next_phase,
            state,
            audited: self
                .device
                .empty(count * self.scenario.fields() * TICK_WORDS),
            actors,
            config,
            spatial,
            proposals: self.device.empty(count * ProposalLayout::WORDS),
            events: self.device.empty(event_words),
            scratch: self.device.empty(scratch_words),
            #[cfg(test)]
            links: self.device.empty(batch_size * LinkLayout::WORDS_PER_EVENT),
            market_execution: self.market_execution,
            control: self.control.clone(),
            summary: self.summary.clone(),
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
        let previous_actors = self
            .metadata
            .actors
            .iter()
            .chain(next.metadata.actors.iter().skip(self.metadata.actors.len()))
            .map(|actor| ActorLayout::new(&actor, &self.scenario))
            .collect::<ActorMetadataRows>();
        let previous_actors = self.device.tensor(&previous_actors.words());
        registry::apply(
            &next.device,
            &next.state,
            &previous_actors,
            &next.config,
            &next.enterprise_requests,
            &next.scratch,
            next.day as u32,
        );
        next.snapshot = next.observe()?;
        next.device.reclaim_memory(0)?;
        match next.memory_usage().bytes_reserved <= next.device.memory_budget.bytes() {
            true => *self = next,
            false => Err(anyhow!(
                "registry allocation exceeds the device memory budget"
            ))?,
        }
        Ok(())
    }
    fn commit_registry(&mut self, plan: registry::RegistryPlan, births: &[u32]) -> Result<()> {
        let newborns = births.iter().filter(|&&birth| birth > 0).count();
        let next_identity = self
            .metadata
            .next_identity
            .checked_add(newborns as u64)
            .context("identity capacity exhausted")?;
        let grown = match newborns {
            0 => None,
            _ => Some(self.demography.grow_births(
                &self.device,
                births,
                self.metadata.next_identity,
            )?),
        };
        let population = grown
            .as_ref()
            .map(|grown| grown.observation(&self.device))
            .transpose()?;
        let previous_actors = self.metadata.actors.clone();
        let previous_actor_buffer = self.actors.clone();
        let previous_spatial = self.spatial.clone();
        let next_buffers = plan.actors.as_ref().map(|actors| {
            let rows = actors
                .iter()
                .map(|actor| ActorLayout::new(&actor, &self.scenario))
                .collect::<ActorMetadataRows>();
            let metadata = State {
                actors: actors.clone(),
                ..self.metadata.clone()
            };
            (
                self.device.tensor(&rows.words()),
                self.device
                    .tensor(&SpatialIndex::new(&self.scenario, &metadata).words),
            )
        });
        self.device.reclaim_memory(0)?;
        match self.memory_usage().bytes_reserved <= self.device.memory_budget.bytes() {
            true => (),
            false => Err(anyhow!(
                "registry capacity exhausted while preparing GPU transitions"
            ))?,
        }
        registry::apply(
            &self.device,
            &self.state,
            &self.actors,
            &self.config,
            &self.enterprise_requests,
            &self.scratch,
            self.day as u32,
        );
        if let Some(actors) = plan.actors {
            self.metadata.actors = actors;
        }
        if let Some((actors, spatial)) = next_buffers {
            self.actors = actors;
            self.spatial = spatial;
        }
        let observation = self.observe().and_then(|snapshot| {
            self.device.reclaim_memory(0)?;
            match self.memory_usage().bytes_reserved <= self.device.memory_budget.bytes() {
                true => Ok(snapshot),
                false => Err(anyhow!(
                    "registry capacity exhausted while auditing GPU transitions"
                )),
            }
        });
        match observation {
            Ok(mut snapshot) => {
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
                self.metadata.actors = previous_actors;
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
        self.fault_check()
    }
    fn market(
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
            (_, MarketExecution::Grouped) => {
                self.clear_market(market, phase, range, activity.phase_data, trace, evidence)
            }
            #[cfg(test)]
            (_, MarketExecution::Reference) => {
                self.reference_market(market, phase, range, trace, evidence)
            }
        }
    }
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
        self.device.write(
            &self.control,
            bytemuck::cast_slice(&[ObservationControlLayout::zeroed()]),
        );
        let parameters =
            self.parameters(Phase::Audit, Market::Labor, 0, self.metadata.actors.len())?;
        self.economy(
            EconomyStage::Summarize,
            self.metadata
                .actors
                .len()
                .div_ceil(SummaryLayout::ACTORS_PER_GROUP as usize),
            &parameters,
            &self.summary,
        );
        reduction::summary(
            &self.device,
            &self.events,
            &self.summary,
            &self.control,
            self.metadata
                .actors
                .len()
                .div_ceil(SummaryLayout::ACTORS_PER_GROUP as usize),
        )?;
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
            day: self.day,
            cash: add(totals.cash.into(), financial.cash()?)?,
            payroll_reserves: totals.payroll_reserves.into(),
            income_tax: totals.income_tax.into(),
            employee_charges: totals.employee_charges.into(),
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
            households: self
                .metadata
                .actors
                .iter()
                .filter(|actor| actor.kind == 0)
                .count() as u64,
            shortages: totals.shortages.into(),
            firms: self
                .metadata
                .actors
                .iter()
                .filter(|actor| actor.kind > 0)
                .count() as u64,
            firm_entries: self
                .metadata
                .actors
                .iter()
                .skip(self.scenario.actors())
                .filter(|actor| actor.kind > 0)
                .count() as u64,
            firm_closures: totals.closures.into(),
            distributors: totals.distributors.into(),
            household_production: totals.household_production.map(u64::from),
            population: self.demography.observation(&self.device)?,
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
        let (source, target) = match direction {
            CopyDirection::Save => (&self.state, &self.audited),
            CopyDirection::Restore => (&self.audited, &self.state),
        };
        tensor::copy(&self.device, source, target);
        if let Some(research) = &self.research {
            match direction {
                CopyDirection::Save => research.save_audited(&self.device)?,
                CopyDirection::Restore => research.restore_audited(&self.device)?,
            }
        }
        self.account_research_readbacks();
        self.demography.copy(&self.device, direction);
        self.device.sync()
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
        let device = Device {
            client: self.device.client.clone(),
            cube: self.device.cube.clone(),
            name: self.device.name.clone(),
            workgroup: self.device.workgroup,
            memory_budget: self.device.memory_budget,
        };
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
            inventory: values[Field::Grain as usize..Field::Escrow as usize].try_into()?,
            transit: values[Field::Transit as usize],
            fatigue: values[Field::Fatigue as usize],
            employer: values[Field::Employer as usize]
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
            construction_minutes_remaining: values[Field::ConstructionRemaining as usize],
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

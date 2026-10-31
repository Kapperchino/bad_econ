mod economy;
mod flags;
mod layout;
mod ops;
mod parameters;
mod policy;
mod scheduler;
mod words;
use crate::{
    arithmetic::{PRICE_TABLE, add},
    domain::*,
    ledger::{Bundle, Decision, MAX_LEGS, Schedule},
    runtime::{PhaseTiming, RuntimeState, SimulationCommand},
};
use anyhow::{Context, Result, anyhow};
use burn_wgpu::{CubeBackend, CubeTensor};
use economy::EconomyStage;
use flags::{
    ActorKind, CONSTRUCTED_EVENTS_WORD, CONTROL_WORDS, DependencyShape, EventStatus, FaultFlags,
    MarketFlag, SchedulerControl,
};
use layout::*;
pub use ops::Device;
use ops::{EconomyBuffers, EconomyOps, SchedulerBuffers};
use parameters::{EconomyParameters, SchedulerParameters};
use policy::MATCHING_ROUNDS;
use scheduler::SchedulerStage;
use std::time::Instant;

const MILLISECONDS_PER_SECOND: f64 = 1000.0;

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
    config: CubeTensor,
    spatial: CubeTensor,
    proposals: CubeTensor,
    events: CubeTensor,
    scratch: CubeTensor,
    links: CubeTensor,
    control: CubeTensor,
    summary: CubeTensor,
    feedback: CubeTensor,
}
impl Simulation {
    pub fn actors(&self) -> &[Actor] {
        &self.metadata.actors
    }
    pub fn memory_usage(&self) -> cubecl::MemoryUsage {
        self.device
            .client
            .memory_report(cubecl::MemoryScope::Device)
            .usage()
    }
    pub fn new(scenario: Scenario, device: Device) -> Result<Self> {
        let initial = scenario.initial()?;
        Self::from_state(scenario, initial, device)
    }
    fn from_state(scenario: Scenario, mut initial: State, device: Device) -> Result<Self> {
        let snapshot = ObservationSnapshot::from_state(&scenario, &initial)?;
        let side = scenario
            .config()
            .map_size
            .div_ceil(scenario.config().search_radius);
        let c = scenario.config();
        let mut configuration = vec![0; ConfigLayout::PRICE_TABLE];
        configuration[ConfigLayout::ACTORS] = scenario.actors() as u32;
        configuration[ConfigLayout::HOUSEHOLDS] = c.households;
        configuration[ConfigLayout::FIRMS] = c.firms;
        configuration[ConfigLayout::FIELDS] = scenario.fields() as u32;
        configuration[ConfigLayout::RING] = scenario.ring() as u32;
        configuration[ConfigLayout::SEARCH_RADIUS] = c.search_radius;
        configuration[ConfigLayout::MAP_SIZE] = c.map_size;
        configuration[ConfigLayout::COMMUTE_SPEED] = c.commute_speed;
        configuration[ConfigLayout::SHIPMENT_SPEED] = c.shipment_speed;
        configuration[ConfigLayout::HOUSEHOLD_NEED] = c.household_need;
        configuration[ConfigLayout::CAPACITY] = c.capacity;
        configuration[ConfigLayout::PACKET_OUTPUT] = c.packet_output;
        configuration[ConfigLayout::PACKET_MINUTES] = c.packet_minutes;
        configuration[ConfigLayout::WORK_CAP] = c.work_cap;
        configuration[ConfigLayout::RESERVED_MINUTES] = c.reserved_minutes;
        configuration[ConfigLayout::FORECAST_WEIGHT] = c.forecast_weight;
        configuration[ConfigLayout::COVERAGE] = c.coverage;
        configuration[ConfigLayout::DISTRIBUTION_RATE] = c.distribution_rate;
        configuration[ConfigLayout::MINIMUM_PRICE] = c.minimum_price;
        configuration[ConfigLayout::MAXIMUM_PRICE] = c.maximum_price;
        configuration[ConfigLayout::DELIVERY_PENALTY] = c.delivery_penalty;
        configuration[ConfigLayout::RESERVE_LOW] = c.reserve as u32;
        configuration[ConfigLayout::RESERVE_HIGH] = (c.reserve >> WORD_BITS) as u32;
        configuration[ConfigLayout::SPATIAL_SIDE] = side;
        configuration.extend(PRICE_TABLE);
        let config = device.tensor(&configuration);
        let actor_words = initial.actors.iter().try_fold(
            Vec::with_capacity(initial.actors.len() * ActorLayout::WORDS),
            |mut words, actor| -> Result<_> {
                let offset = words.len();
                words.resize(offset + ActorLayout::WORDS, 0);
                words[offset + ActorLayout::X] = actor.position.x;
                words[offset + ActorLayout::Y] = actor.position.y;
                words[offset + ActorLayout::KIND] =
                    enumflags2::BitFlags::from_flag(ActorKind::new(actor.kind)?).bits();
                words[offset + ActorLayout::OWNER] = actor.owner;
                words[offset + ActorLayout::CAPACITY] = actor.capacity;
                Ok(words)
            },
        )?;
        let actors = device.tensor(&actor_words);
        let spatial = device.empty((side * side + 1 + c.firms) as usize);
        let state = device.tensor(&encode(&initial.values));
        let audited = device.empty(initial.values.len() * TICK_WORDS);
        initial.values.clear();
        initial.values.shrink_to_fit();
        let batch = (c.batch_size as usize).min(scenario.actors());
        let events = device.empty(
            (batch * EVENT_WORDS).max(
                scenario
                    .actors()
                    .div_ceil(SummaryLayout::ACTORS_PER_GROUP as usize)
                    * SummaryLayout::WORDS,
            ),
        );
        let scratch = device.empty(batch * ScratchLayout::WORDS_PER_EVENT);
        let links = device.empty(batch * LinkLayout::WORDS_PER_EVENT);
        let control = device.tensor(&[0; SummaryLayout::CONTROL_WORDS]);
        let proposals = device.empty(scenario.actors() * ProposalLayout::WORDS);
        let summary = device.empty((scenario.fields() * TICK_WORDS).max(SummaryLayout::WORDS));
        let feedback = device.tensor(&vec![0; CONTROL_WORDS + c.firms as usize * TICK_WORDS]);
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
            links,
            control,
            summary,
            feedback,
        };
        simulation.copy_audited(CopyDirection::Save)?;
        Ok(simulation)
    }
    fn parameters(
        &self,
        phase: Phase,
        market: Market,
        offset: usize,
        count: usize,
    ) -> Result<EconomyParameters> {
        let day = u32::try_from(self.day).context("day identity capacity exhausted")?;
        let _ = day
            .checked_add(self.scenario.ring() as u32 + 1)
            .context("delivery date capacity exhausted")?;
        Ok(EconomyParameters {
            day,
            phase,
            market: market.into(),
            offset: offset as u32,
            count: count as u32,
            status_start: 0,
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
        let result = self.execute_phase(phase, trace).and_then(|evidence| {
            match self.memory_usage().bytes_reserved <= MEMORY_BUDGET {
                true => Ok(evidence),
                false => Err(anyhow!("simulation memory exceeded 4 GiB")),
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
        let mut evidence = PhaseEvidence::default();
        match phase {
            Phase::Commitment => (0..MATCHING_ROUNDS)
                .try_for_each(|_| self.market(Market::Labor, phase, trace, &mut evidence)),
            Phase::WorkPayment => self.market(Market::Payroll, phase, trace, &mut evidence),
            Phase::ProvisionPurchase => {
                [
                    Market::Input(Good::Grain),
                    Market::Input(Good::Flour),
                    Market::Household,
                ]
                .into_iter()
                .try_for_each(|market| {
                    (0..MATCHING_ROUNDS)
                        .try_for_each(|_| self.market(market, phase, trace, &mut evidence))
                })?;
                self.apply_feedback()
            }
            Phase::AccountsDecisions => {
                self.local(phase)?;
                self.market(Market::Distribution, phase, trace, &mut evidence)
            }
            Phase::Audit => {
                self.snapshot = self.observe()?;
                self.day = add(self.day, 1)?;
                self.snapshot.day = self.day;
                Ok(())
            }
            _ => self.local(phase),
        }?;
        Ok(evidence)
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
            self.scenario.config().firms as usize,
            &parameters,
        );
        let control = self.device.read_range(&self.feedback, 0, CONTROL_WORDS)?;
        self.readback_bytes += (CONTROL_WORDS * WORD_BYTES) as u64;
        FaultFlags::read(&control)?.into_result()
    }
    fn local(&mut self, phase: Phase) -> Result<()> {
        self.device
            .write(&self.control, &[0; SummaryLayout::CONTROL_WORDS]);
        let parameters = self.parameters(phase, Market::Labor, 0, self.scenario.actors())?;
        if let Phase::Opening = phase {
            self.feedback_operation(
                EconomyStage::ClearFeedback,
                CONTROL_WORDS + self.scenario.config().firms as usize * TICK_WORDS,
                &parameters,
            );
        }
        self.economy(
            EconomyStage::Local,
            self.scenario.actors(),
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
        self.device
            .write(&self.control, &[0; SummaryLayout::CONTROL_WORDS]);
        let parameters = self.parameters(phase, market, 0, self.scenario.actors())?;
        match market {
            Market::Labor | Market::Input(_) | Market::Household => {
                let side = self
                    .scenario
                    .config()
                    .map_size
                    .div_ceil(self.scenario.config().search_radius);
                self.economy(
                    EconomyStage::IndexCount,
                    (side * side) as usize,
                    &parameters,
                    &self.proposals,
                );
                self.economy(EconomyStage::IndexPrefix, 1, &parameters, &self.proposals);
                self.economy(
                    EconomyStage::IndexFill,
                    (side * side) as usize,
                    &parameters,
                    &self.proposals,
                );
            }
            _ => (),
        }
        self.economy(
            EconomyStage::Propose,
            self.scenario.actors(),
            &parameters,
            &self.proposals,
        );
        self.fault_check()?;
        let range = match market {
            Market::Input(_) | Market::Distribution => {
                self.scenario.config().households as usize..self.scenario.actors()
            }
            _ => 0..self.scenario.config().households as usize,
        };
        let end = range.end;
        range
            .step_by(self.scenario.config().batch_size as usize)
            .try_for_each(|offset| -> Result<()> {
                let count = (end - offset).min(self.scenario.config().batch_size as usize);
                let parameters = self.parameters(phase, market, offset, count)?;
                self.device
                    .write(&self.control, &[0; SummaryLayout::CONTROL_WORDS]);
                self.economy(EconomyStage::Construct, count, &parameters, &self.proposals);
                self.fault_check()?;
                let counters = self.device.read_range(&self.control, 0, CONTROL_WORDS)?;
                self.readback_bytes += (CONTROL_WORDS * WORD_BYTES) as u64;
                let result = match counters[CONSTRUCTED_EVENTS_WORD] {
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
                if let (Market::Input(_) | Market::Household, 1..) = (market, result.events) {
                    let parameters = EconomyParameters {
                        status_start: result.status_start as u32,
                        ..parameters
                    };
                    self.feedback_operation(EconomyStage::RecordFeedback, count, &parameters);
                }
                evidence.frontier_depth = evidence.frontier_depth.max(result.depth);
                evidence.events += result.events;
                evidence.decisions.extend(result.decisions);
                self.readback_bytes += result.readback_bytes;
                Ok(())
            })
    }
    pub fn observe(&mut self) -> Result<ObservationSnapshot> {
        self.device
            .write(&self.control, &[0; SummaryLayout::CONTROL_WORDS]);
        let parameters = self.parameters(Phase::Audit, Market::Labor, 0, self.scenario.actors())?;
        self.economy(
            EconomyStage::Summarize,
            self.scenario
                .actors()
                .div_ceil(SummaryLayout::ACTORS_PER_GROUP as usize),
            &parameters,
            &self.summary,
        );
        self.economy(
            EconomyStage::Reduce,
            SummaryLayout::METRICS,
            &parameters,
            &self.summary,
        );
        let totals = decode(
            &self
                .device
                .read_range(&self.summary, 0, SummaryLayout::WORDS)?,
        );
        let control = self.device.read(&self.control)?;
        self.readback_bytes +=
            ((SummaryLayout::WORDS + SummaryLayout::CONTROL_WORDS) * WORD_BYTES) as u64;
        FaultFlags::read(&control)?.into_result()?;
        let snapshot = ObservationSnapshot {
            day: self.day,
            cash: totals[SummaryLayout::CASH as usize],
            employed: totals[SummaryLayout::EMPLOYED as usize],
            households: u64::from(self.scenario.config().households),
            shortages: totals[SummaryLayout::SHORTAGES as usize],
            inventories: totals[SummaryLayout::INVENTORIES as usize
                ..SummaryLayout::INVENTORIES as usize + Good::COUNT]
                .try_into()?,
            transit: totals
                [SummaryLayout::TRANSIT as usize..SummaryLayout::TRANSIT as usize + Good::COUNT]
                .try_into()?,
            produced: totals
                [SummaryLayout::PRODUCED as usize..SummaryLayout::PRODUCED as usize + Good::COUNT]
                .try_into()?,
            used: totals[SummaryLayout::USED as usize..SummaryLayout::USED as usize + Good::COUNT]
                .try_into()?,
            consumed: totals[SummaryLayout::CONSUMED as usize],
            minimum_wage: totals[SummaryLayout::MINIMUM_WAGE as usize],
            maximum_wage: totals[SummaryLayout::MAXIMUM_WAGE as usize],
            minimum_price: totals[SummaryLayout::MINIMUM_PRICE as usize],
            maximum_price: totals[SummaryLayout::MAXIMUM_PRICE as usize],
            wealth_bins: control[SummaryLayout::WEALTH_START..SummaryLayout::REGIONS_START]
                .iter()
                .map(|&n| u64::from(n))
                .collect(),
            regions: control[SummaryLayout::REGIONS_START..SummaryLayout::CONTROL_WORDS]
                .iter()
                .map(|&n| u64::from(n))
                .collect(),
        };
        snapshot.audit(self.metadata.initial_cash, self.metadata.initial_goods)?;
        Ok(snapshot)
    }
    pub fn download(&mut self) -> Result<State> {
        let values = decode(&self.device.read(&self.state)?);
        self.readback_bytes += (values.len() * TICK_BYTES) as u64;
        Ok(State {
            day: self.day,
            values,
            ..self.metadata.clone()
        })
    }
    fn copy_audited(&mut self, direction: CopyDirection) -> Result<()> {
        let parameters = EconomyParameters {
            day: 0,
            phase: Phase::Opening,
            market: MarketFlag::Labor,
            offset: 0,
            count: 0,
            status_start: 0,
        };
        let (source, target) = match direction {
            CopyDirection::Save => (&self.state, &self.audited),
            CopyDirection::Restore => (&self.audited, &self.state),
        };
        CubeBackend::economy(
            &self.device,
            EconomyStage::CopyState,
            self.scenario.actors() * self.scenario.fields(),
            EconomyBuffers {
                state: source,
                actors: &self.actors,
                config: &self.config,
                spatial: &self.spatial,
                proposals: target,
                events: &self.events,
                control: &self.control,
            },
            &parameters,
        );
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
                self.replace(scenario, initial)
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
        };
        *self = Self::from_state(scenario, state, device)?;
        Ok(())
    }
    pub fn checkpoint(&mut self) -> Result<Checkpoint> {
        let values = decode(&self.device.read(&self.audited)?);
        self.readback_bytes += (values.len() * TICK_BYTES) as u64;
        let state = State {
            day: self.audited_day,
            values,
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
            actor: self.metadata.actors[actor].clone(),
            day: self.day,
            cash: values[Field::Cash as usize],
            inventory: values[Field::Grain as usize..Field::Escrow as usize].try_into()?,
            transit: values[Field::Transit as usize],
            health: values[Field::Health as usize],
            fatigue: values[Field::Fatigue as usize],
            employer: values[Field::Employer as usize],
            available_minutes: values[Field::Time as usize],
            wage: values[Field::Wage as usize],
            price: values[Field::Price as usize],
            planned_packets: values[Field::Plan as usize],
            open_slots: values[Field::Slots as usize],
        })
    }
}
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
    let state = device.tensor(&encode(balances));
    let events = device.tensor(&bundles.iter().flat_map(Bundle::encode).collect::<Vec<_>>());
    let scratch =
        device.empty((bundles.len() * ScratchLayout::WORDS_PER_EVENT).max(MIN_BUFFER_WORDS));
    let links = device.empty((bundles.len() * LinkLayout::WORDS_PER_EVENT).max(MIN_BUFFER_WORDS));
    let control = device.tensor(&[0; CONTROL_WORDS]);
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
fn encode(values: &[u64]) -> Vec<u32> {
    values
        .iter()
        .flat_map(|&value| [value as u32, (value >> WORD_BITS) as u32])
        .collect()
}
fn decode(words: &[u32]) -> Vec<u64> {
    words
        .chunks_exact(TICK_WORDS)
        .map(|pair| u64::from(pair[LOW_WORD]) | (u64::from(pair[HIGH_WORD]) << WORD_BITS))
        .collect()
}

mod economy;
mod ops;
mod scheduler;
mod words;
use crate::{
    arithmetic::{PRICE_TABLE, add},
    domain::*,
    ledger::{Bundle, Decision, Schedule},
    runtime::{PhaseTiming, RuntimeState, SimulationCommand},
};
use anyhow::{Context, Result, anyhow};
use burn_wgpu::{CubeBackend, CubeTensor};
use economy::EconomyStage;
pub use ops::Device;
use ops::EconomyOps;
use scheduler::SchedulerStage;
use std::time::Instant;

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
        let mut configuration = vec![
            scenario.actors() as u32,
            c.households,
            c.firms,
            scenario.fields() as u32,
            scenario.ring() as u32,
            c.search_radius,
            c.map_size,
            c.commute_speed,
            c.shipment_speed,
            c.household_need,
            c.capacity,
            c.packet_output,
            c.packet_minutes,
            c.work_cap,
            c.reserved_minutes,
            c.forecast_weight,
            c.coverage,
            c.distribution_rate,
            c.minimum_price,
            c.maximum_price,
            c.delivery_penalty,
            c.reserve as u32,
            (c.reserve >> 32) as u32,
            side,
        ];
        configuration.resize(32, 0);
        configuration.extend(PRICE_TABLE);
        let config = device.tensor(&configuration);
        let actors = device.tensor(
            &initial
                .actors
                .iter()
                .flat_map(|a| [a.position.x, a.position.y, a.kind, a.owner, a.capacity])
                .collect::<Vec<_>>(),
        );
        let spatial = device.empty((side * side + 1 + c.firms) as usize);
        let state = device.tensor(&encode(&initial.values));
        let audited = device.empty(initial.values.len() * 2);
        initial.values.clear();
        initial.values.shrink_to_fit();
        let batch = (c.batch_size as usize).min(scenario.actors());
        let events = device.empty((batch * 96).max(scenario.actors().div_ceil(256) * 48));
        let scratch = device.empty(batch * 38);
        let links = device.empty(batch * 3);
        let control = device.tensor(&[0; 324]);
        let proposals = device.empty(scenario.actors() * 4);
        let summary = device.empty((scenario.fields() * 2).max(48));
        let feedback = device.tensor(&vec![0; 4 + c.firms as usize * 2]);
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
    ) -> Result<CubeTensor> {
        let day = u32::try_from(self.day).context("day identity capacity exhausted")?;
        let _ = day
            .checked_add(self.scenario.ring() as u32 + 1)
            .context("delivery date capacity exhausted")?;
        Ok(self.device.tensor(&[
            day,
            phase as u32,
            market.code(),
            offset as u32,
            count as u32,
        ]))
    }
    fn economy(
        &self,
        entry: EconomyStage,
        count: usize,
        parameters: &CubeTensor,
        output: &CubeTensor,
    ) {
        CubeBackend::economy(
            &self.device,
            entry,
            count,
            &[
                &self.state,
                &self.actors,
                &self.config,
                &self.spatial,
                output,
                &self.events,
                &self.control,
                parameters,
            ],
        );
    }
    fn fault_check(&mut self) -> Result<()> {
        let flags = self.device.read_range(&self.control, 0, 4)?;
        self.readback_bytes += 16;
        match flags[1] {
            0 => Ok(()),
            code => Err(anyhow!("GPU arithmetic/capacity fault {code}")),
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
            Phase::Commitment => {
                (0..4).try_for_each(|_| self.market(Market::Labor, phase, trace, &mut evidence))
            }
            Phase::WorkPayment => self.market(Market::Payroll, phase, trace, &mut evidence),
            Phase::ProvisionPurchase => {
                [
                    Market::Input(Good::Grain),
                    Market::Input(Good::Flour),
                    Market::Household,
                ]
                .into_iter()
                .try_for_each(|market| {
                    (0..4).try_for_each(|_| self.market(market, phase, trace, &mut evidence))
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
    fn feedback_operation(&self, stage: EconomyStage, count: usize, parameters: &CubeTensor) {
        CubeBackend::economy(
            &self.device,
            stage,
            count,
            &[
                &self.state,
                &self.actors,
                &self.config,
                &self.spatial,
                &self.proposals,
                &self.scratch,
                &self.feedback,
                parameters,
            ],
        );
    }
    fn apply_feedback(&mut self) -> Result<()> {
        let parameters = self.parameters(Phase::ProvisionPurchase, Market::Household, 0, 0)?;
        self.feedback_operation(
            EconomyStage::ApplyFeedback,
            self.scenario.config().firms as usize,
            &parameters,
        );
        let flags = self.device.read_range(&self.feedback, 0, 4)?;
        self.readback_bytes += 16;
        match flags[1] {
            0 => Ok(()),
            _ => Err(anyhow!("demand observation overflow")),
        }
    }
    fn local(&mut self, phase: Phase) -> Result<()> {
        self.device.write(&self.control, &[0; 324]);
        let parameters = self.parameters(phase, Market::Labor, 0, self.scenario.actors())?;
        if let Phase::Opening = phase {
            self.feedback_operation(
                EconomyStage::ClearFeedback,
                4 + self.scenario.config().firms as usize * 2,
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
        self.device.write(&self.control, &[0; 324]);
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
                self.device.write(&self.control, &[0; 324]);
                self.economy(EconomyStage::Construct, count, &parameters, &self.proposals);
                self.fault_check()?;
                let counters = self.device.read_range(&self.control, 0, 4)?;
                self.readback_bytes += 16;
                let result = match counters[3] {
                    0 => Execution {
                        decisions: Vec::new(),
                        depth: 0,
                        events: 0,
                        readback_bytes: 0,
                        status_start: 0,
                    },
                    _ => execute(
                        &self.device,
                        LedgerBuffers {
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
                    let parameters = self.device.tensor(&[
                        self.day as u32,
                        phase as u32,
                        market.code(),
                        offset as u32,
                        count as u32,
                        result.status_start as u32,
                    ]);
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
        self.device.write(&self.control, &[0; 324]);
        let parameters = self.parameters(Phase::Audit, Market::Labor, 0, self.scenario.actors())?;
        self.economy(
            EconomyStage::Summarize,
            self.scenario.actors().div_ceil(256),
            &parameters,
            &self.summary,
        );
        self.economy(EconomyStage::Reduce, 24, &parameters, &self.summary);
        let totals = decode(&self.device.read_range(&self.summary, 0, 48)?);
        let flags = self.device.read(&self.control)?;
        self.readback_bytes += (48 + 324) * 4;
        match flags[1] {
            0 => Ok(()),
            code => Err(anyhow!("GPU summary/transit fault {code}")),
        }?;
        let snapshot = ObservationSnapshot {
            day: self.day,
            cash: totals[0],
            employed: totals[1],
            households: u64::from(self.scenario.config().households),
            shortages: totals[2],
            inventories: totals[3..6].try_into()?,
            transit: totals[6..9].try_into()?,
            produced: totals[9..12].try_into()?,
            used: totals[12..15].try_into()?,
            consumed: totals[15],
            minimum_wage: totals[16],
            maximum_wage: totals[17],
            minimum_price: totals[18],
            maximum_price: totals[19],
            wealth_bins: flags[4..68].iter().map(|&n| u64::from(n)).collect(),
            regions: flags[68..324].iter().map(|&n| u64::from(n)).collect(),
        };
        snapshot.audit(self.metadata.initial_cash, self.metadata.initial_goods)?;
        Ok(snapshot)
    }
    pub fn download(&mut self) -> Result<State> {
        let values = decode(&self.device.read(&self.state)?);
        self.readback_bytes += values.len() as u64 * 8;
        Ok(State {
            day: self.day,
            values,
            ..self.metadata.clone()
        })
    }
    fn copy_audited(&mut self, direction: CopyDirection) -> Result<()> {
        let parameters = self.device.tensor(&[0; 5]);
        let (source, target) = match direction {
            CopyDirection::Save => (&self.state, &self.audited),
            CopyDirection::Restore => (&self.audited, &self.state),
        };
        CubeBackend::economy(
            &self.device,
            EconomyStage::CopyState,
            self.scenario.actors() * self.scenario.fields(),
            &[
                source,
                &self.actors,
                &self.config,
                &self.spatial,
                target,
                &self.events,
                &self.control,
                &parameters,
            ],
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
                milliseconds: start.elapsed().as_secs_f64() * 1000.0,
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
        self.readback_bytes += values.len() as u64 * 8;
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
            self.scenario.fields() * 2,
        )?);
        self.readback_bytes += values.len() as u64 * 8;
        Ok(Inspection {
            actor: self.metadata.actors[actor].clone(),
            day: self.day,
            cash: values[0],
            inventory: values[1..4].try_into()?,
            transit: values[16],
            health: values[11],
            fatigue: values[12],
            employer: values[13],
            available_minutes: values[10],
            wage: values[6],
            price: values[7],
            planned_packets: values[23],
            open_slots: values[5],
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
struct LedgerBuffers<'a> {
    state: &'a CubeTensor,
    events: &'a CubeTensor,
    scratch: &'a CubeTensor,
    control: &'a CubeTensor,
    links: &'a CubeTensor,
}
fn execute(
    device: &Device,
    buffers: LedgerBuffers<'_>,
    count: usize,
    trace: Trace,
) -> Result<Execution> {
    let LedgerBuffers {
        state,
        events,
        scratch,
        control,
        links,
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
            let total = count * 12;
            let launch = |entry: SchedulerStage, size, params: &[u32]| {
                let parameters = device.tensor(params);
                CubeBackend::schedule(
                    device,
                    entry,
                    size,
                    &[state, events, scratch, control, links, &parameters],
                );
            };
            launch(
                SchedulerStage::Initialize,
                total.max(4),
                &[count as u32, 0, 0, 0],
            );
            let mut source = 0;
            let mut width = 1;
            while width < total {
                launch(
                    SchedulerStage::Merge,
                    total,
                    &[count as u32, width as u32, source as u32, 0],
                );
                source = total - source;
                width *= 2;
            }
            launch(
                SchedulerStage::Dependencies,
                total,
                &[count as u32, 0, source as u32, 0],
            );
            launch(SchedulerStage::CheckChains, count, &[count as u32, 0, 0, 0]);
            let counters = device.read_range(control, 0, 4)?;
            result.readback_bytes += 16;
            let mut status = 0;
            match counters[3] {
                0 => {
                    launch(SchedulerStage::Chains, count, &[count as u32, 0, 0, 0]);
                    let counters = device.read_range(control, 0, 4)?;
                    result.readback_bytes += 16;
                    result.depth = counters[0] as usize;
                    result.events = counters[2] as usize;
                    match counters[1] {
                        0 => Ok(()),
                        code => Err(anyhow!("GPU ledger fault {code}")),
                    }?;
                }
                _ => {
                    let mut pending = count;
                    while pending > 0 {
                        launch(SchedulerStage::Clear, 1, &[count as u32, 0, 0, 0]);
                        launch(
                            SchedulerStage::Frontier,
                            count,
                            &[count as u32, 0, 0, status as u32],
                        );
                        status = count - status;
                        result.depth += 1;
                        let counters = device.read_range(control, 0, 4)?;
                        result.readback_bytes += 16;
                        pending = counters[0] as usize;
                        result.events = counters[2] as usize;
                        match (counters[1], result.depth <= count) {
                            (0, true) => Ok(()),
                            _ => Err(anyhow!(
                                "GPU ledger fault {} or stalled dependency graph",
                                counters[1]
                            )),
                        }?;
                    }
                }
            }
            result.status_start = total * 3 + status;
            match trace {
                Trace::Record => {
                    let statuses = device.read_range(scratch, total * 3 + status, count)?;
                    result.readback_bytes += count as u64 * 4;
                    result.decisions = statuses
                        .into_iter()
                        .filter_map(|code| match code {
                            2 => Some(Decision::Accepted),
                            3 => Some(Decision::Rejected),
                            _ => None,
                        })
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
    let scratch = device.empty((bundles.len() * 38).max(4));
    let links = device.empty((bundles.len() * 3).max(4));
    let control = device.tensor(&[0; 4]);
    let result = execute(
        device,
        LedgerBuffers {
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
        .flat_map(|&value| [value as u32, (value >> 32) as u32])
        .collect()
}
fn decode(words: &[u32]) -> Vec<u64> {
    words
        .chunks_exact(2)
        .map(|pair| u64::from(pair[0]) | (u64::from(pair[1]) << 32))
        .collect()
}

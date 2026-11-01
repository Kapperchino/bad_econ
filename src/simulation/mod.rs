pub mod analytics;
pub mod bargaining;
mod economy;
pub mod finance;
mod flags;
pub mod households;
mod layout;
pub mod negotiation;
mod ops;
mod parameters;
mod payroll;
mod policy;
pub mod research;
mod scheduler;
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
    CONTROL_WORDS, ControlLayout, DependencyShape, EventStatus, FaultFlags, MarketFlag,
    SchedulerControl,
};
use layout::*;
pub use ops::Device;
use ops::{EconomyBuffers, EconomyOps, SchedulerBuffers};
use parameters::{EconomyParameters, SchedulerParameters};
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
    config: CubeTensor,
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
        let c = scenario.config();
        let configuration = ConfigLayout::new(&scenario);
        let config = device.tensor(bytemuck::cast_slice(std::slice::from_ref(&configuration)));
        let actor_layouts = initial
            .actors
            .iter()
            .enumerate()
            .map(|(index, actor)| ActorLayout::new(actor, scenario.technique(index)))
            .collect::<Result<Vec<_>>>()?;
        let actors = device.tensor(bytemuck::cast_slice(&actor_layouts));
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
                    config.layout(scenario.actors() * scenario.fields())?,
                    &actors,
                )
            })
            .transpose()?;
        initial.values.clear();
        initial.values.shrink_to_fit();
        initial.research = None;
        let event_words = scenario
            .actors()
            .div_ceil(SummaryLayout::ACTORS_PER_GROUP as usize)
            * SummaryLayout::WORDS;
        let scratch_words = scenario.actors() * 5;
        #[cfg(test)]
        let batch = (c.batch_size as usize).min(scenario.actors());
        #[cfg(test)]
        let event_words = event_words.max(batch * EVENT_WORDS);
        #[cfg(test)]
        let scratch_words = scratch_words.max(batch * ScratchLayout::WORDS_PER_EVENT);
        let events = device.empty(event_words);
        let scratch = device.empty(scratch_words);
        #[cfg(test)]
        let links = device.empty(batch * LinkLayout::WORDS_PER_EVENT);
        let control = device.tensor(bytemuck::cast_slice(&[ObservationControlLayout::zeroed()]));
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
            #[cfg(test)]
            links,
            market_execution: MarketExecution::Grouped,
            control,
            summary,
            feedback,
            finance,
            research,
            research_readback: 0,
        };
        simulation.snapshot = simulation.observe()?;
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
            phase,
            market: market.into(),
            offset: offset as u32,
            count: count as u32,
            status_start: 0,
            sort_source: 0,
            sort_width: 0,
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
                if let Some(research) = &self.research {
                    research.phase(&self.device, &self.state, phase, self.day)?;
                }
                self.snapshot = self.observe()?;
                self.day = add(self.day, 1)?;
                self.snapshot.day = self.day;
                Ok(())
            }
            _ => self.local(phase),
        }?;
        match (&self.research, phase) {
            (_, Phase::Audit) | (None, _) => (),
            (Some(research), _) => research.phase(&self.device, &self.state, phase, self.day)?,
        }
        self.account_research_readbacks();
        Ok(evidence)
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
            self.scenario.config().firms as usize,
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
        self.device.write(
            &self.control,
            bytemuck::cast_slice(&[ObservationControlLayout::zeroed()]),
        );
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
        self.economy(
            EconomyStage::MarketActivity,
            self.scenario.actors().div_ceil(256),
            &parameters,
            &self.proposals,
        );
        let words = self.device.read_range(&self.control, 0, CONTROL_WORDS)?;
        self.readback_bytes += (CONTROL_WORDS * WORD_BYTES) as u64;
        let activity = ControlLayout::new(&words)?;
        FaultFlags::new(activity.faults)?.into_result()?;
        let range = match market {
            Market::Input(_) | Market::Distribution => {
                self.scenario.config().households as usize..self.scenario.actors()
            }
            _ => 0..self.scenario.config().households as usize,
        };
        match (activity.phase_data, self.market_execution) {
            (0, _) => Ok(()),
            (_, MarketExecution::Grouped) => {
                self.clear_market(market, phase, range, trace, evidence)
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
        trace: Trace,
        evidence: &mut PhaseEvidence,
    ) -> Result<()> {
        let count = range.len();
        let mut parameters = self.parameters(phase, market, range.start, count)?;
        let launch = |stage, parameters: &EconomyParameters| {
            let invocations = match (stage, market) {
                (EconomyStage::GroupExecute, Market::Distribution) => {
                    self.scenario.config().households as usize
                }
                (EconomyStage::GroupExecute, _) => self.scenario.config().firms as usize,
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
        if let Market::Labor | Market::Input(_) | Market::Household = market {
            launch(EconomyStage::GroupPrepare, &parameters);
        }
        let mut width = 1;
        while width < count {
            parameters.sort_width = width as u32;
            launch(EconomyStage::GroupMerge, &parameters);
            parameters.sort_source = 2 * count as u32 - parameters.sort_source;
            width *= 2;
        }
        launch(EconomyStage::GroupExecute, &parameters);
        if let Market::Labor | Market::Input(_) | Market::Household = market {
            launch(EconomyStage::GroupFinalize, &parameters);
        }
        let words = self.device.read_range(&self.control, 0, CONTROL_WORDS)?;
        self.readback_bytes += (CONTROL_WORDS * WORD_BYTES) as u64;
        let counters = ControlLayout::new(&words)?;
        FaultFlags::new(counters.faults)?.into_result()?;
        evidence.frontier_depth = evidence.frontier_depth.max(counters.progress as usize);
        evidence.events += counters.completed as usize;
        parameters.status_start = 4 * count as u32;
        if let Market::Input(_) | Market::Household = market {
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
        let end = range.end;
        range
            .step_by(self.scenario.config().batch_size as usize)
            .try_for_each(|offset| -> Result<()> {
                let count = (end - offset).min(self.scenario.config().batch_size as usize);
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
        self.device.write(
            &self.control,
            bytemuck::cast_slice(&[ObservationControlLayout::zeroed()]),
        );
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
            households: u64::from(self.scenario.config().households),
            shortages: totals.shortages.into(),
            inventories: totals.inventories.map(u64::from),
            transit: totals.transit.map(u64::from),
            produced: totals.produced.map(u64::from),
            used: totals.used.map(u64::from),
            consumed: totals.consumed.into(),
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
            day: self.day,
            values,
            research,
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
            sort_source: 0,
            sort_width: 0,
        };
        let (source, target) = match direction {
            CopyDirection::Save => (&self.state, &self.audited),
            CopyDirection::Restore => (&self.audited, &self.state),
        };
        CubeBackend::economy(
            &self.device,
            EconomyStage::CopyState,
            self.scenario.state_len(),
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
        if let Some(research) = &self.research {
            match direction {
                CopyDirection::Save => research.save_audited(&self.device)?,
                CopyDirection::Restore => research.restore_audited(&self.device)?,
            }
        }
        self.account_research_readbacks();
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
        };
        *self = Self::from_state(scenario, state, device)?;
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
            actor: self.metadata.actors[actor].clone(),
            day: self.day,
            cash: values[Field::Cash as usize],
            gross_income: values[Field::GrossIncome as usize],
            payroll_reserve: values[Field::PayrollReserve as usize],
            income_tax: values[Field::TaxTreasury as usize],
            employee_charges: values[Field::ChargeTreasury as usize],
            inventory: values[Field::Grain as usize..Field::Escrow as usize].try_into()?,
            transit: values[Field::Transit as usize],
            health: values[Field::Health as usize],
            fatigue: values[Field::Fatigue as usize],
            employer: values[Field::Employer as usize]
                .checked_sub(1)
                .and_then(|index| self.scenario.identity(index as usize))
                .unwrap_or(0),
            owner_identity: self.metadata.actors[self.metadata.actors[actor].owner as usize]
                .identity,
            available_minutes: values[Field::Time as usize],
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

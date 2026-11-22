use super::{
    CopyDirection, Simulation, Trace,
    economy::{self, EconomyStage},
    finance,
    flags::{self, CONTROL_WORDS, FaultFlags},
    layout::*,
    market::FreightExecution,
    market_graphs,
    policy::{self, MATCHING_ROUNDS},
    profiling_day,
    registry::RegistryAudit,
    settlement,
};
use crate::{
    arithmetic::add,
    domain::*,
    runtime::{PhaseTiming, RuntimeState, SimulationCommand},
};
use anyhow::{Context, Result, anyhow};
use bytemuck::Zeroable;
use std::time::Instant;

const MILLISECONDS_PER_SECOND: f64 = 1000.0;

#[derive(Clone, Copy, PartialEq, Eq)]
enum AdvanceAttempt {
    Pending,
    Complete,
}

#[derive(Debug)]
pub(super) struct ClaimCapacityDemand {
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

impl Simulation {
    pub(super) fn fault_check(&mut self) -> Result<()> {
        let control = self.device.read_range(&self.control, 0, CONTROL_WORDS)?;
        self.readback_bytes += (CONTROL_WORDS * WORD_BYTES) as u64;
        self.fault_result(control[flags::FAULT_WORD])
    }
    pub(super) fn fault_result(&mut self, bits: u32) -> Result<()> {
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
        self.demography.before(self.gpu(), phase, self.day)?;
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
        self.finance_phase(phase, trace, &mut evidence)?;
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
            Phase::ProvisionPurchase => self.provision(trace, &mut evidence),
            Phase::AccountsDecisions => {
                self.local(phase)?;
                match &self.ownership {
                    Some(ownership) => ownership.distribute(self.gpu()),
                    None => {
                        self.market(Market::Distribution, phase, trace, &mut evidence)?;
                        self.market_result(&mut evidence)
                    }
                }
            }
            Phase::Audit => self.audit_day(),
            _ => self.local(phase),
        }?;
        self.demography.after(self.gpu(), phase, self.day)?;
        match (&self.research, phase) {
            (_, Phase::Audit) | (None, _) => (),
            (Some(research), _) => research.phase(&self.device, &self.state, phase, self.day)?,
        }
        self.account_research_readbacks();
        Ok(evidence)
    }

    fn finance_phase(
        &mut self,
        phase: Phase,
        trace: Trace,
        evidence: &mut PhaseEvidence,
    ) -> Result<()> {
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
        Ok(())
    }

    fn audit_day(&mut self) -> Result<()> {
        let started = Instant::now();
        self.railways(economy::railway::RailStage::Close);
        self.fault_check()?;
        if let Some(research) = &self.research {
            research.phase(&self.device, &self.state, Phase::Audit, self.day)?;
        }
        self.snapshot = self.observe()?;
        if self.day + 1 == profiling_day() && std::env::var_os("BAD_ECON_PROFILE_STAGES").is_some()
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
                &self.work.proposals,
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
            &self.work.proposals,
        );
        if let (Phase::WorkPayment, Some(_)) = (phase, self.scenario.config().rent) {
            self.economy(
                EconomyStage::FirmRent,
                self.metadata.actors.len(),
                &parameters,
                &self.work.proposals,
            );
        }
        if phase == Phase::WorkPayment {
            self.railways(economy::railway::RailStage::Infrastructure);
        }
        economy::property::launch(self.gpu(), &self.spatial, &self.control, phase);
        economy::landlords::launch(self.gpu(), &self.control, phase);
        self.fault_check()
    }
    pub(super) fn railways(&self, stage: economy::railway::RailStage) {
        let started = Instant::now();
        economy::railway::launch(self.gpu(), &self.control, stage);
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
}

mod freight;
mod preparation;
#[cfg(test)]
mod reference;
mod settlement;
mod submission;

use crate::{
    domain::{Good, Market, Phase, PhaseEvidence, Scenario, food::FoodIncome},
    simulation::{
        Simulation, Trace,
        economy::{EconomyStage, EconomySystem, markets, railway::RouteAvailability},
        flags::{self, CONTROL_WORDS, ControlLayout, EventStatus},
        layout::WORD_BYTES,
        market_graphs::MarketProgram,
        parameters::{Discovery, EconomyParameters, EmptyOrderMode, MarketClock},
        policy::MATCHING_ROUNDS,
        profiling::MarketProfile,
        profiling_day,
        system::System,
    },
};
use anyhow::Result;
use burn_wgpu::CubeTensor;
use std::time::Instant;

pub(super) use freight::{FreightActivity, FreightExecution};

#[derive(Clone, Copy)]
pub(super) enum MarketExecution {
    Grouped,
    #[cfg(test)]
    Original,
    #[cfg(test)]
    Reference,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum MarketReplay {
    Reusable,
    RegistryGrowth,
}

impl MarketReplay {
    fn clock(self, program: MarketProgram) -> MarketClock {
        match (self, program) {
            (Self::Reusable, _)
            | (Self::RegistryGrowth, MarketProgram::Consumer(_) | MarketProgram::Input) => {
                MarketClock::Device
            }
            _ => MarketClock::Scalars,
        }
    }
}

pub(super) enum MatchingState {
    Searching,
    Exhausted,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Provisioning {
    Standard,
    Minimum,
    Choice,
}

impl Provisioning {
    pub(super) fn rounds(self) -> usize {
        match self {
            Self::Standard => MATCHING_ROUNDS,
            Self::Minimum => MATCHING_ROUNDS - 1,
            Self::Choice => 1,
        }
    }
}

pub(super) struct ProvisionMarket {
    pub(super) market: Market,
    pub(super) provisioning: Provisioning,
}

impl ProvisionMarket {
    pub(super) fn sequence(scenario: &Scenario) -> Vec<Self> {
        let markets = scenario.goods_markets();
        let standard = |market| Self {
            market,
            provisioning: Provisioning::Standard,
        };
        match scenario.config().food.map(|food| food.income) {
            Some(FoodIncome::PurchasingPower) => {
                let inputs = markets
                    .iter()
                    .copied()
                    .filter(|market| matches!(market, Market::Input(_)))
                    .map(standard);
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
                let consumers = markets.iter().copied()
                    .filter(|market| matches!(market, Market::Consumer(good) if !matches!(good, Good::Coal | Good::Wood)))
                    .map(standard);
                inputs.chain(necessities).chain(consumers).collect()
            }
            _ => markets.into_iter().map(standard).collect(),
        }
    }
}

#[derive(Clone, Copy)]
struct MarketRound<'a> {
    market: Market,
    trace: Trace,
    count: usize,
    index: usize,
    timing: StageTiming,
    parameters: EconomyParameters,
    profile: &'a MarketProfile,
}

#[derive(Clone, Copy)]
enum StageTiming {
    Disabled,
    Synchronized,
}

struct InputOrders {
    count: usize,
    routing: RouteAvailability,
}

impl Simulation {
    pub(super) fn market_result(&mut self, evidence: &mut PhaseEvidence) -> Result<()> {
        let execution = match (
            self.next_phase,
            self.scenario.config().railway.companies.is_empty(),
        ) {
            (Phase::ProvisionPurchase, false) => MarketExecution::Grouped,
            _ => self.market_execution,
        };
        match execution {
            MarketExecution::Grouped => {
                let counters = self.read_market_control()?;
                self.fault_result(counters.faults)?;
                evidence.frontier_depth = evidence.frontier_depth.max(counters.progress as usize);
                evidence.events += counters.completed as usize;
            }
            #[cfg(test)]
            _ => (),
        }
        Ok(())
    }

    pub(super) fn market(
        &mut self,
        market: Market,
        phase: Phase,
        trace: Trace,
        evidence: &mut PhaseEvidence,
    ) -> Result<MatchingState> {
        match self.market_execution {
            _ if self.market_has_freight(market) => {
                self.device_market(market, phase, trace, evidence)
            }
            MarketExecution::Grouped => self.device_market(market, phase, trace, evidence),
            #[cfg(test)]
            _ => self
                .original_market(market, phase, trace, evidence)
                .map(|_| MatchingState::Searching),
        }
    }

    pub(super) fn market_graph_buffers(&self) -> impl Iterator<Item = &CubeTensor> + Clone {
        [
            &self.state,
            &self.actors,
            &self.spatial,
            &self.work.proposals,
            &self.work.scratch,
            &self.control,
            &self.searches.requests,
            &self.searches.successes,
            &self.searches.dispatches,
        ]
        .into_iter()
        .chain(self.config.graph_buffers())
    }

    pub(super) fn device_market(
        &mut self,
        market: Market,
        phase: Phase,
        trace: Trace,
        evidence: &mut PhaseEvidence,
    ) -> Result<MatchingState> {
        let started = Instant::now();
        let profile = MarketProfile::new(&self.device, self.day + 1);
        let mut round = MarketRound::new(self, market, phase, trace, &profile)?;
        round.initialize(self);
        round.parameters.clock = round.clock(self);
        let submission = round.submit(self)?;
        round.parameters.sort_source = submission.sort_source;
        let matching = round.complete_matching(self, submission.constructed)?;
        round.record_feedback(self);
        if let Trace::Record = trace {
            self.fault_check()?;
            self.record_market_decisions(round.count, evidence)?;
        }
        if self.market_profiling("BAD_ECON_PROFILE_SUBMISSION") {
            eprintln!(
                "SUBMISSION {market:?} {} {:.3}",
                round.index,
                started.elapsed().as_secs_f64() * 1000.0
            );
        }
        self.readback_bytes +=
            profile.resolve(market, self.matching_round, &self.config, &self.control)?;
        Ok(matching)
    }

    fn market_has_freight(&self, market: Market) -> bool {
        !self.scenario.config().railway.companies.is_empty()
            && matches!(
                market,
                Market::Input(_) | Market::Consumer(_) | Market::Household
            )
    }

    fn market_profiling(&self, setting: &str) -> bool {
        self.day + 1 == profiling_day() && std::env::var_os(setting).is_some()
    }

    fn read_market_control(&mut self) -> Result<ControlLayout> {
        let words = self.device.read_range(&self.control, 0, CONTROL_WORDS)?;
        self.readback_bytes += (CONTROL_WORDS * WORD_BYTES) as u64;
        ControlLayout::new(&words).copied()
    }

    fn record_market_decisions(
        &mut self,
        count: usize,
        evidence: &mut PhaseEvidence,
    ) -> Result<()> {
        let statuses = self
            .device
            .read_range(&self.work.scratch, 4 * count, count)?;
        self.readback_bytes += (count * WORD_BYTES) as u64;
        evidence.decisions.extend(
            statuses
                .into_iter()
                .map(|code| EventStatus::try_from(code)?.decision())
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .flatten(),
        );
        Ok(())
    }
}

impl<'a> MarketRound<'a> {
    fn new(
        simulation: &Simulation,
        market: Market,
        phase: Phase,
        trace: Trace,
        profile: &'a MarketProfile,
    ) -> Result<Self> {
        let count = simulation.metadata.actors.len();
        let mut parameters = simulation.parameters(phase, market, 0, count)?;
        parameters.discovery = Discovery::Indexed;
        parameters.empty_orders = match (market, trace) {
            (Market::Consumer(good), Trace::Discard) if good != Good::Bread => {
                EmptyOrderMode::FeedbackOnly
            }
            _ => EmptyOrderMode::Settle,
        };
        Ok(Self {
            market,
            trace,
            count,
            index: simulation.matching_round,
            timing: match simulation.market_profiling("BAD_ECON_PROFILE_STAGES") {
                true => StageTiming::Synchronized,
                false => StageTiming::Disabled,
            },
            parameters,
            profile,
        })
    }

    fn initialize(&self, simulation: &Simulation) {
        use EconomyStage::*;
        let stage = match (self.market, self.index, self.trace) {
            (Market::Input(_) | Market::Consumer(_), 0, Trace::Discard) => InputMarketPrepare,
            (Market::Input(_) | Market::Consumer(_), 0, Trace::Record) => InputMarketStart,
            (Market::Input(_) | Market::Consumer(_), _, Trace::Record) => InputMarketRepeat,
            _ => MarketStart,
        };
        let count = match stage {
            InputMarketStart | InputMarketRepeat => self.count,
            _ => 1,
        };
        EconomySystem::launch_elements(
            &simulation.device,
            count,
            simulation
                .market_buffers()
                .launch_input(stage, &self.parameters),
        );
    }

    fn launch_stage(&self, simulation: &Simulation, stage: EconomyStage, invocations: usize) {
        use EconomyStage::*;
        let device = &simulation.device;
        let started = Instant::now();
        let window = self.profile.start(device);
        let groups = invocations.div_ceil(device.workgroup as usize);
        let dispatch = match stage {
            GroupExecute => markets::dispatch(
                &simulation.searches.dispatches,
                markets::RANGE_DISPATCH,
                groups,
            ),
            GroupPrepare | GroupFinalize | FinalizeFixedWages | GroupRadixLocal
            | GroupRadixScatter | GroupRemainder | GroupRanges => markets::dispatch(
                &simulation.searches.dispatches,
                markets::SORT_DISPATCH,
                groups,
            ),
            _ => device.count(invocations).into(),
        };
        match stage {
            MarketBoundary => markets::prepare_dispatch_for(
                device,
                &simulation.control,
                &simulation.searches.dispatches,
                self.count,
                markets::PreparationWorkgroup::Native,
            ),
            _ => EconomySystem::launch(
                device,
                dispatch,
                cubecl::CubeDim::new_1d(device.workgroup),
                simulation
                    .market_buffers()
                    .launch_input(stage, &self.parameters),
            ),
        }
        self.profile.end(device, window, "MARKET", stage);
        self.report_stage(simulation, "MARKET", stage, started);
    }

    fn report_stage(
        &self,
        simulation: &Simulation,
        family: &str,
        stage: impl std::fmt::Debug,
        started: Instant,
    ) {
        if let StageTiming::Synchronized = self.timing {
            simulation
                .device
                .sync()
                .expect("stage profiling synchronization");
            eprintln!(
                "{family} {:?} {} {stage:?} {:.3}",
                self.market,
                self.index,
                started.elapsed().as_secs_f64() * 1000.0
            );
        }
    }

    fn complete_matching(
        self,
        simulation: &mut Simulation,
        constructed: Option<usize>,
    ) -> Result<MatchingState> {
        let matching = match (
            simulation.market_has_freight(self.market),
            self.trace,
            simulation.freight_execution,
        ) {
            (true, Trace::Record, _) | (true, _, FreightExecution::Synchronous) => {
                self.finish_freight(simulation)?
            }
            _ => MatchingState::Searching,
        };
        match (self.trace, self.market, self.index + 1 < MATCHING_ROUNDS) {
            (Trace::Discard, Market::Input(_), true) => {
                let count = match constructed {
                    Some(count) => count,
                    None => {
                        simulation.readback_bytes += WORD_BYTES as u64;
                        simulation.device.read_range(
                            &simulation.control,
                            flags::CONSTRUCTED_EVENTS_WORD,
                            1,
                        )?[0] as usize
                    }
                };
                Ok(match count {
                    0 => MatchingState::Exhausted,
                    _ => matching,
                })
            }
            _ => Ok(matching),
        }
    }

    fn record_feedback(mut self, simulation: &Simulation) {
        if let (Market::Input(_) | Market::Consumer(_) | Market::Household, true) =
            (self.market, self.index + 1 == MATCHING_ROUNDS)
        {
            self.parameters.status_start = 4 * self.count as u32;
            markets::feedback_boundary(
                &simulation.device,
                &simulation.control,
                &simulation.work.feedback,
                match self.parameters.empty_orders {
                    EmptyOrderMode::Settle => markets::FeedbackSource::Constructed,
                    EmptyOrderMode::FeedbackOnly => markets::FeedbackSource::Searches,
                },
            );
            simulation.feedback_operation(
                EconomyStage::RecordFeedback,
                self.count,
                &self.parameters,
            );
        }
    }
}

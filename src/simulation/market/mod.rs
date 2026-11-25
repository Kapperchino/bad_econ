mod freight;
pub(super) mod graphs;
mod preparation;
mod provision;
#[cfg(test)]
mod reference;
mod settlement;
mod submission;

use crate::simulation::economy::market::farm_offers::BidRound;
use crate::{
    domain::{Good, Market, Phase, PhaseEvidence, Scenario, food::FoodIncome},
    simulation::{
        Simulation, Trace,
        economy::{
            EconomyStage, EconomySystem,
            market::{orders, search::ConsumerReserves},
            parameters::{Discovery, EconomyParameters, EmptyOrderMode, MarketClock},
            policy::MATCHING_ROUNDS,
            transport::railway::RouteAvailability,
        },
        profiling_day,
        runtime::profiling::MarketProfile,
        storage::{
            flags::{self, CONTROL_WORDS, ControlLayout, EventStatus},
            layout::WORD_BYTES,
        },
        system::System,
    },
};
use anyhow::Result;
use burn_wgpu::CubeTensor;
use graphs::MarketProgram;
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
            | (
                Self::RegistryGrowth,
                MarketProgram::Consumer(_) | MarketProgram::HouseholdGoods | MarketProgram::Input,
            ) => MarketClock::Device,
            _ => MarketClock::Scalars,
        }
    }
}

pub(super) enum MatchingState {
    Searching,
    DeviceInput { capacity: usize },
    Exhausted,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Provisioning {
    Standard,
    Minimum,
    Choice,
    Nonfuel { reserves: ConsumerReserves },
}

impl Provisioning {
    pub(super) fn rounds(self, standard: usize) -> usize {
        match self {
            Self::Standard | Self::Nonfuel { .. } => standard,
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
                    .enumerate()
                    .map(|(index, market)| Self {
                        market,
                        provisioning: Provisioning::Nonfuel {
                            reserves: match index {
                                0 => ConsumerReserves::Capture,
                                _ => ConsumerReserves::Cached,
                            },
                        },
                    });
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
    resources: crate::simulation::settlement::ResourceSelection,
    timing: StageTiming,
    parameters: EconomyParameters,
    profile: &'a MarketProfile,
}

#[derive(Clone, Copy)]
enum StageTiming {
    Disabled,
    Synchronized,
}

#[derive(Clone, Copy)]
struct InputOrders {
    count: usize,
    routing: RouteAvailability,
}

impl Simulation {
    pub(super) fn market_result(&mut self, evidence: &mut PhaseEvidence) -> Result<()> {
        let counters = self.read_market_control()?;
        self.record_market_result(counters, evidence)
    }

    fn record_market_result(
        &mut self,
        counters: ControlLayout,
        evidence: &mut PhaseEvidence,
    ) -> Result<()> {
        self.fault_result(counters.faults)?;
        let execution = match (
            self.next_phase,
            self.scenario.config().railway.companies.is_empty(),
        ) {
            (Phase::ProvisionPurchase, false) => MarketExecution::Grouped,
            _ => self.market_execution,
        };
        match execution {
            MarketExecution::Grouped => {
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
        if crate::simulation::economy::market::farm_offers::FarmMarket::takes_offers(
            market,
            self.config.home_mode,
        ) {
            let parameters = self.parameters(phase, market, 0, self.metadata.actors.len())?;
            self.config.farm_market.collect(
                self.gpu(),
                &self.spatial,
                &self.control,
                &parameters,
                BidRound::Open,
            );
        }
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
        round.parameters.clock = round.clock(self, self.market_replay);
        let submission = round.submit(self)?;
        round.parameters.sort_source = submission.sort_source;
        let matching = round.complete_matching(self, submission.activity)?;
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

    fn continue_input(&mut self, market: Market, capacity: usize) -> Result<MatchingState> {
        use crate::simulation::economy::market::MarketStart;
        let profile = MarketProfile::Disabled;
        let mut round = MarketRound::new(
            self,
            market,
            Phase::ProvisionPurchase,
            Trace::Discard,
            &profile,
        )?;
        MarketStart::ContinueInput.launch(self.market_buffers(), &round.parameters);
        round.parameters.clock = MarketClock::Device;
        round.resources = crate::simulation::settlement::ResourceSelection::RoutedInputs;
        round.parameters.sort_source = round.submit_input_continuation(self, capacity)?;
        round.record_feedback(self);
        Ok(MatchingState::DeviceInput { capacity })
    }

    fn matching_rounds(&self, market: Market) -> usize {
        match market {
            Market::Input(Good::Grain) => self.config.grain_rounds as usize,
            Market::Input(good) | Market::Consumer(good)
                if self.config.food_income == FoodIncome::PurchasingPower
                    && !matches!(
                        good,
                        Good::Grain
                            | Good::Flour
                            | Good::Bread
                            | Good::Coal
                            | Good::Wood
                            | Good::Construction
                    ) =>
            {
                match market {
                    Market::Input(_) => 1,
                    _ => 2,
                }
            }
            _ => MATCHING_ROUNDS,
        }
    }

    fn market_has_freight(&self, market: Market) -> bool {
        !self.scenario.config().railway.companies.is_empty()
            && market != Market::Input(Good::Construction)
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
    fn consumer_reserves(&self, simulation: &Simulation) -> ConsumerReserves {
        match (
            simulation.market_execution,
            simulation.provisioning,
            self.trace,
            self.index,
        ) {
            (MarketExecution::Grouped, Provisioning::Nonfuel { reserves }, Trace::Discard, 0) => {
                reserves
            }
            (MarketExecution::Grouped, Provisioning::Nonfuel { .. }, Trace::Discard, _) => {
                ConsumerReserves::Cached
            }
            _ => ConsumerReserves::Ledger,
        }
    }

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
            resources: crate::simulation::settlement::ResourceSelection::MarketSuppliers,
            timing: match simulation.market_profiling("BAD_ECON_PROFILE_STAGES") {
                true => StageTiming::Synchronized,
                false => StageTiming::Disabled,
            },
            parameters,
            profile,
        })
    }

    fn initialize(&self, simulation: &Simulation) {
        use crate::simulation::economy::market::MarketStart;
        let start = match (self.market, self.index, self.trace) {
            (Market::Input(_) | Market::Consumer(_), 0, Trace::Discard) => {
                MarketStart::PrepareCandidates
            }
            (Market::Input(_) | Market::Consumer(_), 0, Trace::Record) => {
                MarketStart::ResetCandidatesAndOrders
            }
            (Market::Input(_) | Market::Consumer(_), _, Trace::Record) => MarketStart::ResetOrders,
            _ => MarketStart::Standard,
        };
        start.launch(simulation.market_buffers(), &self.parameters);
    }

    fn launch_stage(&self, simulation: &Simulation, stage: EconomyStage, invocations: usize) {
        use EconomyStage::*;
        let device = &simulation.device;
        let started = Instant::now();
        let window = self.profile.start(device);
        let groups = invocations.div_ceil(device.workgroup as usize);
        let dispatch = match stage {
            GroupExecute => orders::dispatch(
                &simulation.searches.dispatches,
                orders::RANGE_DISPATCH,
                groups,
            ),
            GroupPrepare | GroupFinalize | FinalizeFixedWages | GroupRadixLocal
            | GroupRadixScatter | GroupRemainder | GroupRanges => orders::dispatch(
                &simulation.searches.dispatches,
                orders::SORT_DISPATCH,
                groups,
            ),
            _ => device.count(invocations).into(),
        };
        match stage {
            MarketBoundary => orders::prepare_dispatch(
                device,
                &simulation.control,
                &simulation.searches.dispatches,
                self.count,
            ),
            GroupExecute | GroupPrepare | GroupFinalize | FinalizeFixedWages | GroupRemainder
            | GroupRanges
                if matches!(device.cube, cubecl::Device::Metal(_)) && groups > 256 =>
            {
                let offset = match stage {
                    GroupExecute => orders::RANGE_DISPATCH,
                    _ => orders::SORT_DISPATCH,
                };
                crate::simulation::economy::settle_blocks::launch(
                    &device.client,
                    cubecl::CubeCount::Static(256, 1, 1),
                    cubecl::CubeDim::new_1d(device.workgroup),
                    simulation
                        .market_buffers()
                        .launch_input(stage, &self.parameters),
                    crate::simulation::system::AsBufferArg::buffer_arg(
                        &simulation.searches.dispatches,
                    ),
                    offset as u32,
                )
            }
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
        activity: Option<submission::InputActivity>,
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
        match (
            self.trace,
            self.market,
            self.index + 1 < simulation.matching_rounds(self.market),
        ) {
            (Trace::Discard, Market::Input(_), true) => {
                let count = match &activity {
                    Some(activity) => activity.orders.count,
                    None => {
                        simulation.readback_bytes += WORD_BYTES as u64;
                        simulation.device.read_range(
                            &simulation.control,
                            flags::CONSTRUCTED_EVENTS_WORD,
                            1,
                        )?[0] as usize
                    }
                };
                Ok(match (count, activity) {
                    (0, _) => MatchingState::Exhausted,
                    (_, Some(activity)) if self.parameters.clock == MarketClock::Device => {
                        MatchingState::DeviceInput {
                            capacity: activity.candidates.next_power_of_two().min(self.count),
                        }
                    }
                    _ => matching,
                })
            }
            _ => Ok(matching),
        }
    }

    fn record_feedback(mut self, simulation: &Simulation) {
        if let (Market::Input(_) | Market::Consumer(_) | Market::Household, true) = (
            self.market,
            self.index + 1 == simulation.matching_rounds(self.market),
        ) {
            self.parameters.status_start = 4 * self.count as u32;
            orders::feedback_boundary(
                &simulation.device,
                &simulation.control,
                &simulation.work.feedback,
                match self.parameters.empty_orders {
                    EmptyOrderMode::Settle => orders::FeedbackSource::Constructed,
                    EmptyOrderMode::FeedbackOnly => orders::FeedbackSource::Searches,
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

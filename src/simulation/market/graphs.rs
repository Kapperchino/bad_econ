use super::FreightExecution;
use crate::domain::Good;
use crate::simulation::{
    Device, Simulation,
    economy::{
        market::search::ConsumerReserves, policy::RETAINED_LABOR_ROUND,
        transport::railway::RouteAvailability,
    },
    gpu::tensor::same_buffer,
};
use anyhow::Result;
use burn::tensor::StreamId;
use burn_wgpu::CubeTensor;
use cubecl::client::Graph;

#[derive(Default)]
pub(in crate::simulation) struct MarketGraphs {
    bindings: Option<Bindings>,
    entries: Vec<RecordedMarket>,
    replays: usize,
    records: usize,
    invalidations: usize,
}

struct Bindings {
    buffers: Vec<CubeTensor>,
    workgroup: u32,
    thread: std::thread::ThreadId,
    stream: StreamId,
}

struct RecordedMarket {
    key: MarketKey,
    source: u32,
    recording: Recording,
}

enum Recording {
    Unseen,
    Prepared,
    Recorded { graph: Graph },
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum GraphScope {
    PrepareOrders,
    SingleInput,
    InputContinuation {
        capacity: usize,
    },
    SettleInput {
        routing: RouteAvailability,
        capacity: usize,
    },
    FullMarket,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct MarketKey {
    pub scope: GraphScope,
    pub market: MarketProgram,
    pub phase: crate::domain::Phase,
    pub preparation: Preparation,
    pub consumer_reserves: ConsumerReserves,
    pub routing: RouteAvailability,
    pub freight: FreightExecution,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum MarketProgram {
    Labor,
    GrainInput,
    FlourInput,
    BreadInput,
    ConstructionInput,
    Input,
    BreadConsumer,
    Fuel,
    Consumer(Good),
    HouseholdGoods,
    LuxuryConsumer,
    Household,
    Payroll,
    Distribution,
}

impl From<crate::domain::Market> for MarketProgram {
    fn from(market: crate::domain::Market) -> Self {
        use crate::domain::{Good, Market};
        match market {
            Market::Labor => Self::Labor,
            Market::Input(Good::Grain) => Self::GrainInput,
            Market::Input(Good::Flour) => Self::FlourInput,
            Market::Input(Good::Bread) => Self::BreadInput,
            Market::Input(Good::Construction) => Self::ConstructionInput,
            Market::Input(_) => Self::Input,
            Market::Consumer(Good::Bread) => Self::BreadConsumer,
            Market::Consumer(Good::Coal | Good::Wood) => Self::Fuel,
            Market::Consumer(Good::LuxuryClothes | Good::LuxuryFurniture) => Self::LuxuryConsumer,
            Market::Consumer(
                Good::Linen | Good::Cloth | Good::Clothes | Good::Furniture | Good::Glass,
            ) => Self::HouseholdGoods,
            Market::Consumer(good) => Self::Consumer(good),
            Market::Household => Self::Household,
            Market::Payroll => Self::Payroll,
            Market::Distribution => Self::Distribution,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Preparation {
    First,
    Repeat,
    RetainedLabor,
}

impl Preparation {
    pub(super) fn new(market: crate::domain::Market, round: usize) -> Self {
        match (market, round) {
            (crate::domain::Market::Labor, RETAINED_LABOR_ROUND) => Self::RetainedLabor,
            (_, 0) => Self::First,
            _ => Self::Repeat,
        }
    }
}

impl Bindings {
    fn compatible<'a>(
        &self,
        device: &Device,
        buffers: impl Iterator<Item = &'a CubeTensor> + Clone,
    ) -> bool {
        [
            self.workgroup == device.workgroup,
            self.thread == std::thread::current().id(),
            self.stream == StreamId::current(),
            self.buffers.len() == buffers.clone().count(),
            self.buffers
                .first()
                .is_some_and(|buffer| buffer.handle.service == device.client.service_id()),
            self.buffers
                .iter()
                .zip(buffers)
                .all(|(recorded, current)| same_buffer(recorded, current)),
        ]
        .into_iter()
        .all(std::convert::identity)
    }
}

impl MarketGraphs {
    pub(in crate::simulation) fn report(&self, day: u64) {
        if std::env::var_os("BAD_ECON_PROFILE_GRAPHS").is_some() {
            eprintln!(
                "GRAPHS {day} records={} replays={} invalidations={} entries={}",
                self.records,
                self.replays,
                self.invalidations,
                self.entries.len(),
            );
        }
    }

    fn bind<'a>(
        &mut self,
        device: &Device,
        market: MarketProgram,
        buffers: impl Iterator<Item = &'a CubeTensor> + Clone,
    ) {
        let compatible = self
            .bindings
            .as_ref()
            .is_some_and(|bindings| bindings.compatible(device, buffers.clone()));
        if !compatible {
            if let Some(bindings) = self
                .bindings
                .as_ref()
                .filter(|_| std::env::var_os("BAD_ECON_PROFILE_GRAPHS").is_some())
            {
                let changed = bindings
                    .buffers
                    .iter()
                    .zip(buffers.clone())
                    .enumerate()
                    .filter_map(|(index, (recorded, current))| {
                        (!same_buffer(recorded, current)).then_some(index)
                    })
                    .collect::<Vec<_>>();
                eprintln!("GRAPH_INVALID {market:?} {changed:?}");
            }
            self.invalidations += 1;
            self.entries.clear();
            self.bindings = Some(Bindings {
                buffers: buffers.cloned().collect(),
                workgroup: device.workgroup,
                thread: std::thread::current().id(),
                stream: StreamId::current(),
            });
        }
    }

    pub(super) fn run(
        &mut self,
        simulation: &mut Simulation,
        key: MarketKey,
        mut encode: impl FnMut(&mut Simulation) -> Result<u32>,
    ) -> Result<u32> {
        let device = simulation.device.clone();
        self.bind(&device, key.market, simulation.market_graph_buffers());
        self.execute(&device, key, || encode(simulation))
    }

    fn execute(
        &mut self,
        device: &Device,
        key: MarketKey,
        mut encode: impl FnMut() -> Result<u32>,
    ) -> Result<u32> {
        match self.entries.iter().position(|entry| entry.key == key) {
            Some(index) => match &self.entries[index].recording {
                Recording::Unseen => {
                    let source = encode()?;
                    match source == self.entries[index].source {
                        true => {
                            self.entries[index].recording = Recording::Prepared;
                            Ok(source)
                        }
                        false => Err(anyhow::anyhow!("market graph changed its sort output")),
                    }
                }
                Recording::Recorded { graph } => {
                    unsafe { graph.replay() }?;
                    self.replays += 1;
                    Ok(self.entries[index].source)
                }
                Recording::Prepared => {
                    device.client.graph_prepare()?;
                    let source = encode()?;
                    device.client.start_capture()?;
                    let recorded = encode();
                    let graph = device.client.stop_capture()?;
                    let recorded = recorded?;
                    match source == recorded && source == self.entries[index].source {
                        true => {
                            self.records += 1;
                            self.entries[index].recording = Recording::Recorded { graph };
                            Ok(source)
                        }
                        false => Err(anyhow::anyhow!("market graph changed its sort output")),
                    }
                }
            },
            None => {
                let source = encode()?;
                self.entries.push(RecordedMarket {
                    key,
                    source,
                    recording: Recording::Prepared,
                });
                Ok(source)
            }
        }
    }

    pub(super) fn prepare(&mut self, key: MarketKey, source: u32) {
        match self.entries.iter().any(|entry| entry.key == key) {
            true => (),
            false => {
                self.entries.push(RecordedMarket {
                    key,
                    source,
                    recording: Recording::Unseen,
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{Experiment, Field, Phase, Scenario};
    use crate::simulation::system::AsBufferArg;
    use crate::simulation::{
        FreightActivity, Simulation, Trace,
        economy::{
            market::sorting::MarketSort,
            parameters::{EconomyParameters, MarketClock},
        },
        encode,
        gpu::tensor,
    };
    use cubecl::prelude::*;

    #[cube(launch)]
    fn increment(counter: &mut [u32]) {
        if ABSOLUTE_POS == 0 {
            counter[0usize] += 1;
        }
    }

    #[cube(launch)]
    fn record_consumer_good(
        counter: &mut [u32],
        mut config: crate::simulation::economy::configuration::EconomyConfig,
        mut parameters: EconomyParameters,
    ) {
        parameters.refresh(&mut config);
        if ABSOLUTE_POS == 0 {
            let good = parameters.traded_good();
            counter[good as usize - 1] += good;
        }
    }

    #[test]
    fn consumer_graphs_preserve_exact_goods_with_shared_buffers_on_gpu() -> Result<()> {
        let mut device = crate::simulation::tests::device(64)?;
        unsafe {
            device.client.set_stream(StreamId::current());
        }
        let simulation = Simulation::new(
            Scenario::baseline(1, 42, Experiment::Concentrated)?,
            device.clone(),
        )?;
        let counter = device.tensor(&[0; Good::COUNT]);
        let mut expected = vec![0; Good::COUNT];
        let mut graphs = MarketGraphs::default();
        for _ in 0..3 {
            for good in [
                Good::Coal,
                Good::Wood,
                Good::Linen,
                Good::Cloth,
                Good::LuxuryClothes,
                Good::LuxuryFurniture,
            ] {
                let market = crate::domain::Market::Consumer(good);
                let mut parameters =
                    simulation.parameters(Phase::ProvisionPurchase, market, 0, 1)?;
                crate::simulation::economy::market::MarketStart::Standard
                    .launch(simulation.market_buffers(), &parameters);
                parameters.clock = MarketClock::Device;
                let key = MarketKey {
                    scope: GraphScope::FullMarket,
                    market: market.into(),
                    phase: Phase::ProvisionPurchase,
                    preparation: Preparation::First,
                    consumer_reserves: ConsumerReserves::Ledger,
                    routing: RouteAvailability::Dormant,
                    freight: FreightExecution::Synchronous,
                };
                graphs.bind(&device, key.market, std::iter::once(&counter));
                assert_eq!(
                    graphs.execute(&device, key, || {
                        record_consumer_good::launch(
                            &device.client,
                            CubeCount::Static(1, 1, 1),
                            CubeDim::new_1d(32),
                            counter.buffer_arg(),
                            simulation.config.launch(),
                            parameters.launch(),
                        );
                        Ok(7)
                    })?,
                    7,
                );
                expected[good as usize - 1] += good as u32;
                assert_eq!(device.read(&counter)?, expected, "{good:?}");
                assert!(!device.client.is_capturing());
            }
        }
        assert_eq!(graphs.entries.len(), 3);
        assert_eq!(graphs.records, 3);
        assert_eq!(graphs.replays, 12);
        assert_eq!(graphs.invalidations, 1);
        Ok(())
    }

    #[test]
    fn lazy_graph_capture_executes_each_call_once_and_leaves_the_stream_ready() -> Result<()> {
        let mut device = crate::simulation::tests::device(64)?;
        unsafe {
            device.client.set_stream(StreamId::current());
        }
        let counter = device.tensor(&[0]);
        let mut graphs = MarketGraphs::default();
        let key = MarketKey {
            scope: GraphScope::FullMarket,
            market: MarketProgram::Household,
            phase: Phase::ProvisionPurchase,
            preparation: Preparation::First,
            consumer_reserves: ConsumerReserves::Ledger,
            routing: RouteAvailability::Dormant,
            freight: FreightExecution::Synchronous,
        };
        let fallback = MarketKey {
            routing: RouteAvailability::Available,
            ..key
        };
        let run = |graphs: &mut MarketGraphs, key: MarketKey, counter: &CubeTensor| {
            graphs.bind(&device, key.market, std::iter::once(counter));
            graphs.execute(&device, key, || {
                increment::launch(
                    &device.client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new_1d(32),
                    counter.buffer_arg(),
                );
                Ok(7)
            })
        };
        for expected in 1..=3 {
            assert_eq!(run(&mut graphs, key, &counter)?, 7);
            graphs.prepare(fallback, 7);
            assert!(!device.client.is_capturing());
            assert_eq!(device.read(&counter)?, [expected]);
            assert_eq!(graphs.records, usize::from(expected >= 2));
            assert_eq!(graphs.replays, usize::from(expected == 3));
        }
        assert_eq!(run(&mut graphs, fallback, &counter)?, 7);
        assert_eq!(device.read(&counter)?, [4]);
        assert_eq!(graphs.records, 1);
        assert!(!device.client.is_capturing());
        let replacement = device.tensor(&[10]);
        assert_eq!(run(&mut graphs, key, &replacement)?, 7);
        assert_eq!(device.read(&replacement)?, [11]);
        assert_eq!(device.read(&counter)?, [4]);
        assert_eq!(graphs.invalidations, 2);
        assert_eq!(graphs.records, 1);
        assert!(!device.client.is_capturing());
        Ok(())
    }

    #[test]
    fn exhausted_grain_graphs_keep_candidates_and_final_feedback_empty_on_gpu() -> Result<()> {
        use crate::simulation::economy::market::{orders, search::INPUT_RAIL_WORD};
        use crate::simulation::storage::flags::{CONSTRUCTED_EVENTS_WORD, CONTROL_WORDS};

        let device = crate::simulation::tests::device(64)?;
        let scenario = Scenario::industrial(4097, 42, Experiment::Baseline)?;
        let mut simulation = Simulation::new(scenario, device.clone())?;
        simulation
            .config
            .railway
            .replace_route_availability(RouteAvailability::Dormant);
        simulation.freight_execution = FreightExecution::Buffered;
        let before = simulation.download()?;
        let capacity = before.actors.len();
        let mut control = vec![0; simulation.control.meta.num_elements()];
        control[orders::PREPARATION_ACTORS_WORD] = capacity as u32;
        control[INPUT_RAIL_WORD] = 1;
        device.write(&simulation.control, &control);
        let feedback = vec![0; simulation.work.feedback.meta.num_elements()];
        device.write(&simulation.work.feedback, &feedback);
        for round in 1..simulation.config.grain_rounds {
            simulation.matching_round = round as usize;
            simulation.continue_input(crate::domain::Market::Input(Good::Grain), capacity)?;
        }
        let control = device.read(&simulation.control)?;
        assert_eq!(control[orders::PREPARATION_ACTORS_WORD], 0);
        assert_eq!(control[CONSTRUCTED_EVENTS_WORD], 0);
        assert_eq!(control[INPUT_RAIL_WORD], 0);
        let actual = device.read(&simulation.work.feedback)?;
        assert_eq!(actual[orders::FEEDBACK_EVENTS_WORD], 0);
        assert_eq!(&actual[CONTROL_WORDS..], &feedback[CONTROL_WORDS..]);
        assert_eq!(simulation.download()?, before);
        Ok(())
    }

    #[test]
    fn dormant_rail_markets_match_full_freight_searches_and_accounts_on_gpu() -> Result<()> {
        let device = crate::simulation::tests::device(64)?;
        let scenario = Scenario::industrial(4097, 42, Experiment::Baseline)?;
        let mut actual = Simulation::new(scenario.clone(), device.clone())?;
        let mut expected = Simulation::new(scenario, device)?;
        expected.freight_activity = FreightActivity::Unconditional;
        for _ in 0..2 {
            for phase in Phase::ALL {
                if let Phase::ProvisionPurchase = phase {
                    for simulation in [&mut actual, &mut expected] {
                        let mut state = simulation.download()?;
                        let firms = state.actors.firm_indices().collect::<Vec<_>>();
                        for firm in firms {
                            state.set(Field::RailOperating, firm, 0);
                        }
                        simulation.device.write(
                            &simulation.state,
                            bytemuck::cast_slice(&encode(&tensor::state_values(
                                &state.values,
                                state.actors.len(),
                                simulation.config.state_stride as usize,
                            ))),
                        );
                    }
                }
                actual.phase(phase, Trace::Discard)?;
                expected.phase(phase, Trace::Discard)?;
                if let Phase::ProvisionPurchase = phase {
                    for routing in [RouteAvailability::Available, RouteAvailability::Dormant] {
                        assert!(actual.market_graphs.entries.iter().any(|entry| {
                            entry.key.routing == routing
                                && matches!(entry.key.market, MarketProgram::Household)
                        }));
                    }
                }
                assert_eq!(
                    actual.config.railway.route_availability(),
                    RouteAvailability::Available,
                );
                assert_eq!(actual.download()?, expected.download()?, "{phase:?}");
            }
            assert_eq!(actual.freight_activity, FreightActivity::Dormant);
            assert_eq!(actual.snapshot, expected.snapshot);
        }
        Ok(())
    }

    #[test]
    fn market_graphs_refresh_dates_heat_and_rotations_and_replace_bindings_on_gpu() -> Result<()> {
        let device = crate::simulation::tests::device(64)?;
        let mut config = Scenario::industrial(4097, 42, Experiment::Baseline)?
            .config()
            .clone();
        config.entrepreneurship.interested_per_mille = 0;
        config.demography.annual_births_per_million_fertile_women = 0;
        let scenario = Scenario::new(config)?;
        let initial = scenario.initial()?;
        let mut actual = Simulation::new(scenario.clone(), device.clone())?;
        assert!(actual.config.state_stride as usize > initial.actors.len());
        let initialized = actual.download()?;
        assert_eq!(initialized.values, initial.values);
        assert_eq!(actual.checkpoint()?.restore()?.state, initialized);
        let mut expected = Simulation::from_state(scenario, initial, device)?;
        expected.market_sort = MarketSort::Profiled;
        for day in [0, 57, 120] {
            actual.day = day;
            expected.day = day;
            expected.market_sort = MarketSort::Profiled;
            actual.advance()?;
            expected.advance()?;
            assert_eq!(actual.snapshot, expected.snapshot, "day {day}");
            assert_eq!(actual.download()?, expected.download()?, "day {day}");
            assert!(actual.market_graphs.replays > 0);
        }
        actual.actors = actual.device.tensor(&actual.device.read(&actual.actors)?);
        actual.device.workgroup = 32;
        expected.device.workgroup = 32;
        expected.market_sort = MarketSort::Profiled;
        actual.advance()?;
        expected.advance()?;
        assert_eq!(actual.download()?, expected.download()?);
        actual = std::thread::spawn(move || -> Result<Simulation> {
            actual.advance()?;
            Ok(actual)
        })
        .join()
        .map_err(|_| anyhow::anyhow!("market graph replay thread panicked"))??;
        expected.market_sort = MarketSort::Profiled;
        expected.advance()?;
        assert_eq!(actual.snapshot, expected.snapshot);
        assert_eq!(actual.download()?, expected.download()?);
        Ok(())
    }
}

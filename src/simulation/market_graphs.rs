use super::Device;
use crate::domain::Good;
use anyhow::Result;
use burn::tensor::StreamId;
use burn_wgpu::CubeTensor;
use cubecl::client::Graph;

#[derive(Default)]
pub(super) struct MarketGraphs {
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
pub(super) struct MarketKey {
    pub market: MarketProgram,
    pub phase: crate::domain::Phase,
    pub preparation: Preparation,
    pub routing: super::economy::railway::RouteAvailability,
    pub freight: super::FreightExecution,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum MarketProgram {
    Labor,
    GrainInput,
    BreadInput,
    Input,
    BreadConsumer,
    Fuel(Good),
    Consumer(Good),
    LuxuryConsumer(Good),
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
            Market::Input(Good::Bread) => Self::BreadInput,
            Market::Input(_) => Self::Input,
            Market::Consumer(Good::Bread) => Self::BreadConsumer,
            Market::Consumer(good @ (Good::Coal | Good::Wood)) => Self::Fuel(good),
            Market::Consumer(good @ (Good::LuxuryClothes | Good::LuxuryFurniture)) => {
                Self::LuxuryConsumer(good)
            }
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
}

impl Bindings {
    fn compatible(&self, device: &Device, buffers: &[CubeTensor]) -> bool {
        [
            self.workgroup == device.workgroup,
            self.thread == std::thread::current().id(),
            self.stream == StreamId::current(),
            self.buffers.len() == buffers.len(),
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
    pub(super) fn report(&self, day: u64) {
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

    pub(super) fn run(
        &mut self,
        device: &Device,
        key: MarketKey,
        buffers: Vec<CubeTensor>,
        mut encode: impl FnMut() -> Result<u32>,
    ) -> Result<u32> {
        let compatible = self
            .bindings
            .as_ref()
            .is_some_and(|bindings| bindings.compatible(device, &buffers));
        if !compatible {
            if let Some(bindings) = self
                .bindings
                .as_ref()
                .filter(|_| std::env::var_os("BAD_ECON_PROFILE_GRAPHS").is_some())
            {
                let changed = bindings
                    .buffers
                    .iter()
                    .zip(&buffers)
                    .enumerate()
                    .filter_map(|(index, (recorded, current))| {
                        (!same_buffer(recorded, current)).then_some(index)
                    })
                    .collect::<Vec<_>>();
                eprintln!("GRAPH_INVALID {:?} {changed:?}", key.market);
            }
            self.invalidations += 1;
            self.entries.clear();
            self.bindings = Some(Bindings {
                buffers,
                workgroup: device.workgroup,
                thread: std::thread::current().id(),
                stream: StreamId::current(),
            });
        }
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
    use crate::simulation::{
        FreightActivity, Simulation, Trace,
        economy::sorting::MarketSort,
        encode,
        parameters::{EconomyParameters, MarketClock},
        tensor,
    };
    use cubecl::prelude::*;

    #[cube(launch)]
    fn increment(counter: &mut [u32]) {
        if ABSOLUTE_POS == 0 {
            counter[0usize] += 1;
        }
    }

    #[cube(launch)]
    fn record_consumer_good(counter: &mut [u32], parameters: EconomyParameters) {
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
                parameters.clock = MarketClock::Scalars;
                parameters.good = Good::Bread as u32;
                let key = MarketKey {
                    market: market.into(),
                    phase: Phase::ProvisionPurchase,
                    preparation: Preparation::First,
                    routing: crate::simulation::economy::railway::RouteAvailability::Dormant,
                    freight: crate::simulation::FreightExecution::Synchronous,
                };
                assert_eq!(
                    graphs.run(&device, key, vec![counter.clone()], || {
                        record_consumer_good::launch(
                            &device.client,
                            CubeCount::Static(1, 1, 1),
                            CubeDim::new_1d(32),
                            counter.clone().into_buffer_arg(),
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
        assert_eq!(graphs.entries.len(), 6);
        assert_eq!(graphs.records, 6);
        assert_eq!(graphs.replays, 6);
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
            market: MarketProgram::Household,
            phase: Phase::ProvisionPurchase,
            preparation: Preparation::First,
            routing: crate::simulation::economy::railway::RouteAvailability::Dormant,
            freight: crate::simulation::FreightExecution::Synchronous,
        };
        let fallback = MarketKey {
            routing: crate::simulation::economy::railway::RouteAvailability::Available,
            ..key
        };
        let run = |graphs: &mut MarketGraphs, key, counter: &CubeTensor| {
            graphs.run(&device, key, vec![counter.clone()], || {
                increment::launch(
                    &device.client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new_1d(32),
                    counter.clone().into_buffer_arg(),
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
                    for routing in [
                        crate::simulation::economy::railway::RouteAvailability::Available,
                        crate::simulation::economy::railway::RouteAvailability::Dormant,
                    ] {
                        assert!(actual.market_graphs.entries.iter().any(|entry| {
                            entry.key.routing == routing
                                && matches!(entry.key.market, MarketProgram::Household)
                        }));
                    }
                }
                assert_eq!(
                    actual.config.railway.route_availability(),
                    crate::simulation::economy::railway::RouteAvailability::Available,
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
        let mut actual = Simulation::new(scenario.clone(), device.clone())?;
        let mut expected = Simulation::new(scenario, device)?;
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

pub(super) fn same_buffer(recorded: &CubeTensor, current: &CubeTensor) -> bool {
    (
        recorded.handle.memory.descriptor(),
        recorded.handle.service,
        recorded.handle.offset_start,
        recorded.handle.offset_end,
        recorded.handle.size(),
        &recorded.meta,
        recorded.dtype,
    ) == (
        current.handle.memory.descriptor(),
        current.handle.service,
        current.handle.offset_start,
        current.handle.offset_end,
        current.handle.size(),
        &current.meta,
        current.dtype,
    )
}

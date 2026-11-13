use super::Device;
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
    graph: Graph,
    source: u32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct MarketKey {
    pub market: crate::domain::Market,
    pub phase: crate::domain::Phase,
    pub preparation: Preparation,
    pub routing: super::economy::railway::RouteAvailability,
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
            if std::env::var_os("BAD_ECON_PROFILE_GRAPHS").is_some() {
                if let Some(bindings) = &self.bindings {
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
        match self.entries.iter().find(|entry| entry.key == key) {
            Some(recorded) => {
                unsafe { recorded.graph.replay() }?;
                self.replays += 1;
                Ok(recorded.source)
            }
            None => {
                device.client.graph_prepare()?;
                let source = encode()?;
                device.client.start_capture()?;
                let recorded = encode();
                let graph = device.client.stop_capture()?;
                let recorded = recorded?;
                match source == recorded {
                    true => {
                        self.records += 1;
                        self.entries.push(RecordedMarket { key, graph, source });
                        Ok(source)
                    }
                    false => Err(anyhow::anyhow!("market graph changed its sort output")),
                }
            }
        }
    }

    pub(super) fn prerecord(
        &mut self,
        device: &Device,
        key: MarketKey,
        expected: u32,
        mut encode: impl FnMut() -> Result<u32>,
    ) -> Result<()> {
        match self.entries.iter().any(|entry| entry.key == key) {
            true => Ok(()),
            false => {
                device.client.graph_prepare()?;
                device.client.start_capture()?;
                let source = encode();
                let graph = device.client.stop_capture()?;
                let source = source?;
                match source == expected {
                    true => {
                        self.records += 1;
                        self.entries.push(RecordedMarket { key, graph, source });
                        Ok(())
                    }
                    false => Err(anyhow::anyhow!("market graph changed its sort output")),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{Experiment, Field, Phase, Scenario};
    use crate::simulation::{
        FreightActivity, Simulation, Trace, economy::sorting::MarketSort, encode,
    };

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
                            bytemuck::cast_slice(&encode(&state.values)),
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
                                && matches!(entry.key.market, crate::domain::Market::Household)
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
        let scenario = Scenario::industrial(4097, 42, Experiment::Baseline)?;
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

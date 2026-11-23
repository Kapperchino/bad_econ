use super::graphs::{GraphScope, MarketKey, Preparation};
use super::{FreightExecution, InputOrders, MarketRound, StageTiming};
use crate::{
    domain::Market,
    simulation::{
        Simulation, Trace,
        economy::{
            EconomyStage,
            market::{
                search::{INPUT_RAIL_WORD, OrderCollection},
                sorting::{INPUT_TILE, MarketSort},
            },
            parameters::MarketClock,
            transport::railway::RouteAvailability,
        },
        runtime::profiling::MarketProfile,
        storage::{flags, layout::WORD_BYTES},
    },
};
use anyhow::Result;

const DIRECT_INPUT_TILE: usize = 256;

pub(super) struct MarketSubmission {
    pub(super) sort_source: u32,
    pub(super) constructed: Option<usize>,
}

impl MarketRound<'_> {
    pub(super) fn clock(&self, simulation: &Simulation) -> MarketClock {
        match (
            self.trace,
            self.count,
            self.profile,
            self.timing,
            &simulation.market_sort,
            &simulation.device.cube,
        ) {
            (
                Trace::Discard,
                4096..,
                MarketProfile::Disabled,
                StageTiming::Disabled,
                MarketSort::Profiled,
                _,
            ) => MarketClock::Scalars,
            (
                Trace::Discard,
                4096..,
                MarketProfile::Disabled,
                StageTiming::Disabled,
                _,
                cubecl::Device::Wgpu(_) | cubecl::Device::Metal(_),
            ) if !self.has_unallocated_freight(simulation)
                && std::env::var_os("BAD_ECON_PROFILE_SUPPLY").is_none() =>
            {
                simulation.market_replay.clock(self.market.into())
            }
            _ => MarketClock::Scalars,
        }
    }

    fn graph_key(&self, simulation: &Simulation) -> MarketKey {
        let scope = match (
            self.market,
            self.trace,
            simulation.freight_execution,
            &simulation.device.cube,
            self.count,
        ) {
            (
                Market::Input(_),
                Trace::Discard,
                FreightExecution::Buffered,
                cubecl::Device::Metal(_),
                4096..,
            ) => GraphScope::PrepareOrders,
            _ => GraphScope::FullMarket,
        };
        MarketKey {
            scope,
            market: self.market.into(),
            phase: self.parameters.phase,
            routing: simulation.config.railway.route_availability(),
            freight: simulation.freight_execution,
            preparation: Preparation::new(self.market, self.index),
        }
    }

    pub(super) fn submit(self, simulation: &mut Simulation) -> Result<MarketSubmission> {
        let key = self.graph_key(simulation);
        let submission = match key.scope {
            GraphScope::PrepareOrders => self.submit_sparse_input(simulation, key)?,
            GraphScope::FullMarket => {
                let sort_source = match self.parameters.clock {
                    MarketClock::Scalars => self.prepare_and_settle(simulation)?,
                    MarketClock::Device => self.run_graph(simulation, key, |simulation| {
                        self.prepare_and_settle(simulation)
                    })?,
                };
                MarketSubmission {
                    sort_source,
                    constructed: None,
                }
            }
        };
        Ok(submission)
    }

    fn run_graph(
        self,
        simulation: &mut Simulation,
        key: MarketKey,
        encode: impl FnMut(&mut Simulation) -> Result<u32>,
    ) -> Result<u32> {
        let mut graphs = std::mem::take(&mut simulation.market_graphs);
        let result = graphs.run(simulation, key, encode);
        if let (Ok(source), GraphScope::FullMarket, RouteAvailability::Dormant, true) = (
            &result,
            key.scope,
            key.routing,
            simulation.market_has_freight(self.market),
        ) {
            graphs.prepare(
                MarketKey {
                    routing: RouteAvailability::Available,
                    ..key
                },
                *source,
            );
        }
        simulation.market_graphs = graphs;
        result
    }

    fn prepare_and_settle(self, simulation: &mut Simulation) -> Result<u32> {
        self.prepare_orders(simulation, self.order_collection(simulation))?;
        self.settle_orders(simulation, None)
    }

    fn submit_sparse_input(
        mut self,
        simulation: &mut Simulation,
        key: MarketKey,
    ) -> Result<MarketSubmission> {
        match self.parameters.clock {
            MarketClock::Scalars => self.prepare_orders(simulation, OrderCollection::InputTile)?,
            MarketClock::Device => {
                self.run_graph(simulation, key, |simulation| {
                    self.prepare_orders(simulation, OrderCollection::InputTile)?;
                    Ok(0)
                })?;
            }
        }
        let orders = simulation.observe_input_orders()?;
        let count = orders.count;
        if count > INPUT_TILE {
            self.compact_sparse_input(simulation);
        }
        let sort_source = match (count, orders.routing) {
            (0, _) => 0,
            (1..=DIRECT_INPUT_TILE, RouteAvailability::Dormant) => {
                self.sort_input_tile(simulation, count);
                self.parameters.sort_source = 0;
                self.launch_stage(
                    simulation,
                    EconomyStage::SettleInputTile,
                    simulation.device.workgroup as usize,
                );
                0
            }
            _ => self.settle_orders(simulation, Some(orders))?,
        };
        Ok(MarketSubmission {
            sort_source,
            constructed: Some(count),
        })
    }
}

impl Simulation {
    fn observe_input_orders(&mut self) -> Result<InputOrders> {
        let start = flags::CONSTRUCTED_EVENTS_WORD;
        let rail = INPUT_RAIL_WORD - start;
        let words = self.device.read_range(&self.control, start, rail + 1)?;
        self.readback_bytes += (words.len() * WORD_BYTES) as u64;
        Ok(InputOrders {
            count: words[0] as usize,
            routing: match words[rail] {
                0 => RouteAvailability::Dormant,
                _ => RouteAvailability::Available,
            },
        })
    }
}

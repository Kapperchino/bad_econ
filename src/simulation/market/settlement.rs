use super::{InputOrders, MarketRound};
use crate::{
    domain::{Good, Market},
    simulation::{
        Simulation,
        economy::{
            EconomyStage,
            configuration::Feature,
            market::{
                orders,
                sorting::{self, INPUT_TILE, MarketSort, SortKeys},
            },
            parameters::{EconomyParameters, MarketClock},
            settlement::batch::{
                self,
                labor::{LaborExecution, LaborInputLaunch, LaborSettlement},
                payroll::{PayrollInputLaunch, PayrollSettlement},
                quantity::{QuantityInputLaunch, QuantitySettlement},
            },
            transport::railway::settlement::Stage as FreightStage,
        },
        runtime::profiling::MarketProfile,
        system::{AsBufferArg, System},
    },
};
use anyhow::Result;
use cubecl::CubeDim;

#[derive(Clone, Copy)]
enum MarketBatch {
    Payroll,
    Labor(LaborExecution),
    Quantity,
    Consumers,
    Grouped,
}

impl MarketRound<'_> {
    pub(super) fn settle_orders(
        mut self,
        simulation: &mut Simulation,
        observed: Option<InputOrders>,
    ) -> Result<u32> {
        use EconomyStage::*;
        let freight = self.freight_settlement(simulation, observed.as_ref());
        let observed = observed.map(|orders| orders.count);
        let limit = observed.unwrap_or(self.count);
        self.parameters.sort_source = self.sort_orders(simulation, observed)?;
        if let Some(execution) = freight {
            self.launch_freight(simulation, FreightStage::Snapshot, execution);
        }
        self.prepare_ranges(simulation, observed, limit);
        let batch = self.settlement_batch(simulation);
        self.execute_orders(simulation, batch, limit);
        if let Market::Household = self.market {
            self.launch_stage(simulation, GroupRemainder, self.count);
        }
        self.finalize_orders(simulation, batch, limit);
        self.launch_stage(simulation, MarketBoundary, 1);
        if let Some(execution) = freight {
            self.book_freight(simulation, execution)?;
        }
        Ok(self.parameters.sort_source)
    }

    fn sort_orders(&self, simulation: &mut Simulation, observed: Option<usize>) -> Result<u32> {
        let window = self.profile.start(&simulation.device);
        let sort_keys = SortKeys::for_market(
            self.market,
            simulation.scenario.population_rows() as u32,
            simulation.config.home_mode,
        );
        let source = match (observed, self.parameters.clock, self.profile) {
            (Some(1..=INPUT_TILE), _, _) => {
                self.sort_input_tile(simulation);
                0
            }
            (_, MarketClock::Device, _) | (_, _, MarketProfile::Enabled(_)) => {
                MarketSort::encode_keys(
                    &simulation.device,
                    &simulation.work.scratch,
                    &simulation.control,
                    &simulation.searches.dispatches,
                    &EconomyParameters {
                        sort_count: observed.unwrap_or(self.count) as u32,
                        ..self.parameters
                    },
                    sort_keys,
                )
            }
            (_, MarketClock::Scalars, _) => simulation.market_sort.run_keys(
                &simulation.device,
                &simulation.work.scratch,
                &simulation.control,
                &simulation.searches.dispatches,
                &self.parameters,
                sort_keys,
            )?,
        };
        self.profile
            .end(&simulation.device, window, "SORT", EconomyStage::GroupSort);
        Ok(source)
    }

    pub(super) fn sort_input_tile(&self, simulation: &Simulation) {
        sorting::sort_input_tile(
            &simulation.device,
            &simulation.searches.requests,
            &simulation.work.proposals,
            &simulation.work.scratch,
            &simulation.control,
            &simulation.config.market_table,
            self.parameters.count,
        );
    }

    fn prepare_ranges(&self, simulation: &Simulation, observed: Option<usize>, limit: usize) {
        use EconomyStage::*;
        match (self.market, observed) {
            (_, Some(1..=INPUT_TILE)) => (),
            (Market::Consumer(good), _) if !matches!(good, Good::Bread | Good::Grain) => {
                self.launch_stage(
                    simulation,
                    SupplierRanges,
                    simulation.metadata.actors.firms(),
                );
            }
            _ => self.launch_stage(simulation, GroupRanges, limit),
        }
        self.launch_stage(simulation, MarketBoundary, 1);
        if let Market::Labor | Market::Input(_) = self.market {
            self.launch_stage(simulation, GroupPrepare, limit);
        }
    }

    fn settlement_batch(&self, simulation: &Simulation) -> MarketBatch {
        let config = simulation.scenario.config();
        match (
            self.market,
            config.bargaining.as_ref(),
            config.people.as_ref(),
        ) {
            (Market::Payroll, _, _) => MarketBatch::Payroll,
            (Market::Labor, None, _) => {
                MarketBatch::Labor(match (simulation.config.cohorts, self.count) {
                    (Feature::Enabled, 4096..) => LaborExecution::Deferred,
                    _ => LaborExecution::Complete,
                })
            }
            (Market::Consumer(_), _, Some(_))
            | (
                Market::Consumer(
                    Good::Wood | Good::Coal | Good::LuxuryClothes | Good::LuxuryFurniture,
                ),
                _,
                _,
            ) => MarketBatch::Quantity,
            (Market::Consumer(_), _, _) => MarketBatch::Consumers,
            _ => MarketBatch::Grouped,
        }
    }

    fn execute_orders(&self, simulation: &Simulation, batch: MarketBatch, limit: usize) {
        let device = &simulation.device;
        let window = match batch {
            MarketBatch::Grouped => None,
            _ => self.profile.start(device),
        };
        let groups = || {
            orders::job_dispatch(
                &simulation.searches.dispatches,
                orders::QUEUE_DISPATCH,
                self.count,
            )
        };
        let workgroup = CubeDim::new_1d(device.workgroup);
        match batch {
            MarketBatch::Payroll => PayrollSettlement::launch(
                device,
                groups(),
                workgroup,
                PayrollInputLaunch {
                    state: simulation.state.buffer_arg(),
                    actors: simulation.actors.buffer_arg(),
                    config: simulation.config.launch(),
                    proposals: simulation.work.proposals.buffer_arg(),
                    scratch: simulation.work.scratch.buffer_arg(),
                    control: simulation.control.buffer_arg(),
                    parameters: self.parameters.launch(),
                },
            ),
            MarketBatch::Labor(execution) => LaborSettlement::launch(
                device,
                groups(),
                workgroup,
                LaborInputLaunch {
                    proposals: simulation.work.proposals.buffer_arg(),
                    state: simulation.state.buffer_arg(),
                    config: simulation.config.launch(),
                    scratch: simulation.work.scratch.buffer_arg(),
                    control: simulation.control.buffer_arg(),
                    parameters: self.parameters.launch(),
                    execution,
                },
            ),
            MarketBatch::Quantity => {
                QuantitySettlement::launch(
                    device,
                    groups(),
                    workgroup,
                    QuantityInputLaunch {
                        state: simulation.state.buffer_arg(),
                        actors: simulation.actors.buffer_arg(),
                        config: simulation.config.launch(),
                        proposals: simulation.work.proposals.buffer_arg(),
                        scratch: simulation.work.scratch.buffer_arg(),
                        control: simulation.control.buffer_arg(),
                        parameters: self.parameters.launch(),
                    },
                );
                self.launch_stage(simulation, EconomyStage::GroupRemainder, self.count);
            }
            MarketBatch::Consumers => batch::settle_consumers(
                simulation.gpu(),
                &simulation.work.scratch,
                &simulation.control,
                &simulation.searches.dispatches,
                &self.parameters,
            ),
            MarketBatch::Grouped => {
                self.launch_stage(simulation, EconomyStage::GroupExecute, limit)
            }
        }
        self.profile
            .end(device, window, "BATCH", EconomyStage::GroupExecute);
    }

    fn finalize_orders(&self, simulation: &Simulation, batch: MarketBatch, limit: usize) {
        if let Market::Labor | Market::Input(_) | Market::Consumer(_) | Market::Household =
            self.market
        {
            let stage = match batch {
                MarketBatch::Labor(LaborExecution::Deferred) => EconomyStage::FinalizeFixedWages,
                _ => EconomyStage::GroupFinalize,
            };
            self.launch_stage(simulation, EconomyStage::MarketBoundary, 1);
            self.launch_stage(simulation, stage, limit);
        }
    }
}

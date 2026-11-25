use super::MarketExecution;
use crate::{
    domain::{Good, Market, Phase, PhaseEvidence},
    simulation::{
        Execution, Simulation, Trace,
        economy::EconomyStage,
        execute, parallel,
        storage::{
            buffers::SchedulerBuffers,
            flags::{EventStatus, FaultFlags},
            layout::DISCOVERY_BLOCK,
        },
    },
};
use anyhow::Result;

struct ReferenceRound {
    market: Market,
    phase: Phase,
    trace: Trace,
    count: usize,
}

struct ReferenceBatch {
    offset: usize,
    count: usize,
}

impl Simulation {
    pub(super) fn original_market(
        &mut self,
        market: Market,
        phase: Phase,
        trace: Trace,
        evidence: &mut PhaseEvidence,
    ) -> Result<()> {
        let round = ReferenceRound {
            market,
            phase,
            trace,
            count: self.metadata.actors.len(),
        };
        let mixed = phase == Phase::ProvisionPurchase
            && !self.scenario.config().railway.companies.is_empty();
        if mixed {
            self.market_result(evidence)?;
        }
        let active = round.prepare_orders(self)?;
        match (active, self.market_execution) {
            (0, _) => Ok(()),
            (_, MarketExecution::Original) => round.clear_grouped(self, active, evidence),
            (_, MarketExecution::Reference) => round.clear_batched(self, evidence),
            (_, MarketExecution::Grouped) => unreachable!(),
        }?;
        if mixed {
            self.clear_control();
        }
        Ok(())
    }
}

impl ReferenceRound {
    fn prepare_orders(&self, simulation: &mut Simulation) -> Result<u32> {
        use EconomyStage::*;
        simulation.clear_control();
        let parameters = simulation.parameters(self.phase, self.market, 0, self.count)?;
        if let Market::Labor | Market::Input(_) | Market::Consumer(_) | Market::Household =
            self.market
        {
            let config = simulation.scenario.config();
            let side = config.map_size.div_ceil(config.search_radius);
            let workgroup = simulation.device.workgroup as usize;
            let buffers = simulation.market_buffers();
            buffers.run(IndexCount, self.count, &parameters);
            parallel::scan(
                &simulation.device,
                &simulation.work.scratch,
                self.count,
                self.count.div_ceil(workgroup),
            );
            buffers.run(IndexOffsets, (side * side + 1) as usize, &parameters);
            buffers.run(IndexFill, self.count, &parameters);
            if let Market::Household | Market::Input(Good::Bread) | Market::Consumer(Good::Bread) =
                self.market
            {
                buffers.run(
                    IndexPrices,
                    self.count.div_ceil(DISCOVERY_BLOCK as usize) * workgroup,
                    &parameters,
                );
            }
        }
        simulation.economy(Propose, self.count, &parameters, &simulation.work.proposals);
        simulation.economy(
            MarketActivity,
            self.count,
            &parameters,
            &simulation.work.proposals,
        );
        let activity = simulation.read_market_control()?;
        FaultFlags::new(activity.faults)?.into_result()?;
        Ok(activity.phase_data)
    }

    fn clear_grouped(
        &self,
        simulation: &mut Simulation,
        active: u32,
        evidence: &mut PhaseEvidence,
    ) -> Result<()> {
        use EconomyStage::*;
        let mut parameters = simulation.parameters(self.phase, self.market, 0, self.count)?;
        parameters.sort_count = active;
        simulation.clear_control();
        let buffers = simulation.market_buffers();
        buffers.run(GroupInitialize, self.count, &parameters);
        if let Market::Labor | Market::Input(_) | Market::Consumer(_) | Market::Household =
            self.market
        {
            buffers.run(GroupPrepare, self.count, &parameters);
        }
        buffers.run(GroupSort, active as usize, &parameters);
        let mut width = simulation.device.workgroup as usize;
        while width < active as usize {
            parameters.sort_width = width as u32;
            buffers.run(GroupMerge, active as usize, &parameters);
            parameters.sort_source = 2 * self.count as u32 - parameters.sort_source;
            width *= 2;
        }
        buffers.run(GroupExecute, active as usize, &parameters);
        if let Market::Labor | Market::Input(_) | Market::Consumer(_) | Market::Household =
            self.market
        {
            buffers.run(GroupFinalize, self.count, &parameters);
        }
        let counters = simulation.read_market_control()?;
        FaultFlags::new(counters.faults)?.into_result()?;
        evidence.frontier_depth = evidence.frontier_depth.max(counters.progress as usize);
        evidence.events += counters.completed as usize;
        parameters.status_start = 4 * self.count as u32;
        if self.records_feedback(simulation) {
            simulation.feedback_operation(RecordFeedback, self.count, &parameters);
        }
        if let Trace::Record = self.trace {
            simulation.record_market_decisions(self.count, evidence)?;
        }
        Ok(())
    }

    fn clear_batched(
        &self,
        simulation: &mut Simulation,
        evidence: &mut PhaseEvidence,
    ) -> Result<()> {
        let rotation = simulation
            .parameters(self.phase, self.market, 0, self.count)?
            .settlement_rotation as usize;
        let batch_size = simulation.scenario.config().batch_size as usize;
        let batches = [rotation..self.count, 0..rotation]
            .into_iter()
            .flat_map(|range| {
                let end = range.end;
                range.step_by(batch_size).map(move |offset| ReferenceBatch {
                    offset,
                    count: (end - offset).min(batch_size),
                })
            });
        let mut decisions = vec![None; self.count];
        for batch in batches {
            let result = self.clear_batch(simulation, &batch)?;
            evidence.frontier_depth = evidence.frontier_depth.max(result.depth);
            evidence.events += result.events;
            if let (Trace::Record, 1..) = (self.trace, result.events) {
                let codes = simulation.device.read_range(
                    &simulation.work.scratch,
                    result.status_start,
                    batch.count,
                )?;
                for (index, code) in codes.into_iter().enumerate() {
                    decisions[batch.offset + index] = EventStatus::try_from(code)?.decision()?;
                }
            }
            simulation.readback_bytes += result.readback_bytes;
        }
        evidence.decisions.extend(decisions.into_iter().flatten());
        Ok(())
    }

    fn clear_batch(
        &self,
        simulation: &mut Simulation,
        batch: &ReferenceBatch,
    ) -> Result<Execution> {
        use EconomyStage::*;
        let mut parameters =
            simulation.parameters(self.phase, self.market, batch.offset, batch.count)?;
        simulation.clear_control();
        simulation.economy(
            Construct,
            batch.count,
            &parameters,
            &simulation.work.proposals,
        );
        simulation.fault_check()?;
        let result = match simulation.read_market_control()?.phase_data {
            0 => Execution {
                decisions: Vec::new(),
                depth: 0,
                events: 0,
                readback_bytes: 0,
                status_start: 0,
            },
            _ => execute(
                &simulation.device,
                SchedulerBuffers {
                    state: &simulation.state,
                    events: &simulation.work.events,
                    scratch: &simulation.work.scratch,
                    control: &simulation.control,
                    links: &simulation.work.links,
                },
                batch.count,
                self.trace,
            )?,
        };
        if let (1.., true) = (result.events, self.records_feedback(simulation)) {
            parameters.status_start = result.status_start as u32;
            simulation.feedback_operation(RecordFeedback, batch.count, &parameters);
        }
        Ok(result)
    }

    fn records_feedback(&self, simulation: &Simulation) -> bool {
        matches!(
            (
                self.market,
                simulation.matching_round + 1 == simulation.matching_rounds(self.market)
            ),
            (
                Market::Input(_) | Market::Consumer(_) | Market::Household,
                true
            )
        )
    }
}

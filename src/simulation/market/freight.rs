use super::{InputOrders, MarketRound, MatchingState};
use crate::simulation::{
    Simulation, Trace,
    economy::railway::{
        RouteAvailability,
        settlement::{
            self, ClaimPreparation, ClaimStorage, ClaimWork, Execution, ResourceSelection, Stage,
            Wave,
        },
    },
    layout::WORD_BYTES,
};
use anyhow::Result;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::simulation) enum FreightActivity {
    Active,
    Dormant,
    #[cfg(test)]
    Unconditional,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(in crate::simulation) enum FreightExecution {
    Synchronous,
    Buffered,
}

impl FreightExecution {
    fn initial(self, trace: Trace) -> settlement::Execution {
        match (trace, self) {
            (Trace::Discard, Self::Buffered) => {
                settlement::Execution::Buffered(settlement::Wave::Initial)
            }
            _ => settlement::Execution::Immediate(settlement::Wave::Initial),
        }
    }
}
impl MarketRound<'_> {
    pub(super) fn has_unallocated_freight(&self, simulation: &Simulation) -> bool {
        match (
            simulation.market_has_freight(self.market),
            simulation.freight_activity,
            simulation.freight_execution,
            ClaimStorage::new(&simulation.config),
        ) {
            (true, FreightActivity::Dormant, _, _) => false,
            (true, _, FreightExecution::Buffered, ClaimStorage::Unallocated) => true,
            _ => false,
        }
    }

    pub(super) fn freight_settlement(
        &self,
        simulation: &Simulation,
        observed: Option<&InputOrders>,
    ) -> Option<Execution> {
        match (
            simulation.market_has_freight(self.market),
            observed.map(|orders| orders.routing),
            simulation.freight_activity,
            simulation.freight_execution,
            self.trace,
        ) {
            (false, _, _, _, _)
            | (_, Some(RouteAvailability::Dormant), _, _, _)
            | (_, _, FreightActivity::Dormant, FreightExecution::Buffered, Trace::Discard) => None,
            (_, _, _, execution, trace) => Some(execution.initial(trace)),
        }
    }

    pub(super) fn launch_freight(
        &self,
        simulation: &Simulation,
        stage: Stage,
        execution: Execution,
    ) {
        self.profile
            .freight(&simulation.device, stage, execution.wave(), || {
                settlement::launch_buffered(
                    simulation.gpu(),
                    &simulation.spatial,
                    &simulation.work.proposals,
                    &simulation.work.scratch,
                    &simulation.control,
                    &self.parameters,
                    stage,
                    execution,
                    match execution {
                        Execution::Buffered(_) => ResourceSelection::MarketSuppliers,
                        Execution::Immediate(_) => ResourceSelection::AllFirms,
                    },
                )
            });
    }

    pub(super) fn book_freight(
        &self,
        simulation: &mut Simulation,
        execution: Execution,
    ) -> Result<()> {
        for stage in [
            Stage::Initialize,
            Stage::Queue,
            Stage::Propose,
            Stage::CountClaims,
        ] {
            self.launch_freight(simulation, stage, execution);
        }
        if let Execution::Buffered(_) = execution {
            self.finish_freight(simulation)?;
        }
        Ok(())
    }

    pub(super) fn finish_freight(&self, simulation: &mut Simulation) -> Result<MatchingState> {
        let matching = match (
            self.trace,
            simulation.freight_activity,
            simulation.freight_execution,
        ) {
            (Trace::Discard, FreightActivity::Dormant, _) => {
                self.commit_freight(simulation, Execution::Immediate(Wave::Initial));
                MatchingState::Searching
            }
            (Trace::Discard, _, FreightExecution::Buffered) => {
                self.finish_buffered_freight(simulation)?;
                MatchingState::Searching
            }
            _ => self.finish_immediate_freight(simulation)?,
        };
        self.launch_freight(
            simulation,
            Stage::Accounts,
            simulation.freight_execution.initial(self.trace),
        );
        Ok(matching)
    }

    fn finish_buffered_freight(&self, simulation: &mut Simulation) -> Result<()> {
        if let ClaimStorage::Unallocated = ClaimStorage::new(&simulation.config) {
            settlement::reserve_market_claims(
                &simulation.device,
                &mut simulation.config,
                &simulation.control,
            )?;
            simulation.readback_bytes += (3 * WORD_BYTES) as u64;
        }
        let storage = ClaimStorage::new(&simulation.config);
        for &execution in storage.executions() {
            self.prepare_freight_retry(simulation, execution);
            for &stage in storage.stages() {
                self.launch_freight(simulation, stage, execution);
            }
            self.commit_freight(simulation, execution);
        }
        Ok(())
    }

    fn finish_immediate_freight(&self, simulation: &mut Simulation) -> Result<MatchingState> {
        let mut next_wave = Some(Wave::Initial);
        let mut matching = MatchingState::Searching;
        while let Some(wave) = next_wave {
            let execution = Execution::Immediate(wave);
            self.prepare_freight_retry(simulation, execution);
            let ClaimPreparation { claims, searches } = settlement::prepare_market_claims(
                &simulation.device,
                &mut simulation.config,
                &simulation.control,
            )?;
            simulation.readback_bytes += (4 * WORD_BYTES) as u64;
            matching = match searches {
                0 => MatchingState::Exhausted,
                _ => MatchingState::Searching,
            };
            if let ClaimWork::Ready = claims {
                for stage in [Stage::EmitClaims, Stage::SortClaims, Stage::Resolve] {
                    self.launch_freight(simulation, stage, execution);
                }
            }
            self.commit_freight(simulation, execution);
            next_wave = wave.next(claims);
        }
        Ok(matching)
    }

    fn prepare_freight_retry(&self, simulation: &Simulation, execution: Execution) {
        if let Wave::Retry = execution.wave() {
            for stage in [
                Stage::Queue,
                Stage::Reset,
                Stage::Propose,
                Stage::CountClaims,
            ] {
                self.launch_freight(simulation, stage, execution);
            }
        }
    }

    fn commit_freight(&self, simulation: &Simulation, execution: Execution) {
        self.launch_freight(simulation, Stage::Commit, execution);
        self.launch_freight(simulation, Stage::Release, execution);
    }
}

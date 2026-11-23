use super::{
    FreightActivity, FreightExecution, MarketReplay, MatchingState, ProvisionMarket, Provisioning,
};
use crate::domain::{Good, Market, Phase, PhaseEvidence};
use crate::simulation::{
    Simulation, Trace,
    economy::{
        self, EconomyStage,
        policy::MATCHING_ROUNDS,
        transport::railway::{self, RailStage},
    },
    settlement,
    storage::layout::WORD_BYTES,
};
use anyhow::Result;

impl ProvisionMarket {
    fn clear(
        self,
        simulation: &mut Simulation,
        trace: Trace,
        evidence: &mut PhaseEvidence,
    ) -> Result<()> {
        simulation.provisioning = self.provisioning;
        let market = self.market;
        let rounds = self.provisioning.rounds(simulation.matching_rounds(market));
        (0..rounds).try_fold(MatchingState::Searching, |state, round| {
            match (state, self.provisioning, market, round + 1 == rounds) {
                (MatchingState::Searching, _, _, _)
                | (
                    MatchingState::Exhausted,
                    Provisioning::Standard,
                    Market::Household | Market::Consumer(Good::Coal | Good::Wood),
                    true,
                ) => {
                    simulation.matching_round = round;
                    simulation.market(market, Phase::ProvisionPurchase, trace, evidence)
                }
                _ => Ok(MatchingState::Exhausted),
            }
        })?;
        let parameters = simulation.parameters(
            Phase::ProvisionPurchase,
            market,
            0,
            simulation.metadata.actors.len(),
        )?;
        simulation.feedback_operation(
            EconomyStage::RetailOrders,
            simulation.metadata.actors.len() - simulation.config.first_firm as usize,
            &parameters,
        );
        simulation.matching_round = MATCHING_ROUNDS - 1;
        if let (Market::Household, Some(_), Provisioning::Standard | Provisioning::Choice) =
            (market, simulation.scenario.config().rent, self.provisioning)
        {
            let parameters = simulation.parameters(
                Phase::ProvisionPurchase,
                market,
                0,
                simulation.metadata.actors.len(),
            )?;
            simulation.economy(
                EconomyStage::HouseholdRent,
                simulation.metadata.actors.len(),
                &parameters,
                &simulation.work.proposals,
            );
            simulation.fault_check()?;
        }
        Ok(())
    }
}
impl Simulation {
    pub(in crate::simulation) fn provision(
        &mut self,
        trace: Trace,
        evidence: &mut PhaseEvidence,
    ) -> Result<()> {
        self.refresh_freight_quotes();
        self.freight_activity = match self.freight_activity {
            #[cfg(test)]
            FreightActivity::Unconditional => FreightActivity::Unconditional,
            _ => self.measure_freight_activity()?,
        };
        if std::env::var_os("BAD_ECON_PROFILE_GRAPHS").is_some() {
            eprintln!(
                "FREIGHT_ACTIVITY {} {:?}",
                self.day + 1,
                self.freight_activity
            );
        }
        if let (FreightActivity::Active, FreightExecution::Buffered, Trace::Discard) =
            (self.freight_activity, self.freight_execution, trace)
        {
            settlement::reserve_active_market(&self.device, &mut self.config)?;
        }
        let heating = self.config.heating_per_person;
        let availability = match self.freight_activity {
            FreightActivity::Dormant => railway::RouteAvailability::Dormant,
            _ => railway::RouteAvailability::Available,
        };
        let routing = self.config.railway.replace_route_availability(availability);
        let result = ProvisionMarket::sequence(&self.scenario)
            .into_iter()
            .filter(|request| match (request.market, request.provisioning) {
                (Market::Consumer(Good::Coal | Good::Wood), Provisioning::Minimum) => heating > 2,
                (Market::Consumer(Good::Coal | Good::Wood), _) => heating > 0,
                _ => true,
            })
            .try_for_each(|request| request.clear(self, trace, evidence));
        self.provisioning = Provisioning::Standard;
        if let (FreightExecution::Buffered, Trace::Discard) = (self.freight_execution, trace) {
            self.railways(RailStage::DormantDemand);
        }
        self.config.railway.replace_route_availability(routing);
        result?;
        self.market_result(evidence)?;
        self.apply_feedback()?;
        economy::transport::cartage::dispatch(
            self.gpu(),
            &self.control,
            self.day as u32,
            self.scenario.config().seed,
            &self.searches.requests,
            &self.work.proposals,
        );
        self.fault_check()?;
        self.market_replay = MarketReplay::Reusable;
        Ok(())
    }
    fn measure_freight_activity(&mut self) -> Result<FreightActivity> {
        match self.scenario.config().railway.companies.len() {
            0 => Ok(FreightActivity::Dormant),
            _ => {
                let words = self.config.railway.operating_counts(&self.device)?;
                self.readback_bytes += (words.len() * WORD_BYTES) as u64;
                Ok(match words.into_iter().any(|word| word != 0) {
                    true => FreightActivity::Active,
                    false => FreightActivity::Dormant,
                })
            }
        }
    }
}

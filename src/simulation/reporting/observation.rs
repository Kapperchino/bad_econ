use crate::simulation::{
    Simulation, economy, finance, research,
    storage::{flags::FaultFlags, layout::*},
};
#[cfg(test)]
use crate::simulation::{economy::EconomyStage, gpu::tensor, reduction};
use crate::{arithmetic::add, domain::*};
use anyhow::Result;

#[cfg(test)]
#[derive(Clone, Copy)]
pub(in crate::simulation) enum AuditExecution {
    Tiled,
    Original,
}

impl Simulation {
    pub(in crate::simulation) fn account_research_readbacks(&mut self) {
        let total = self
            .research
            .as_ref()
            .map_or(0, research::ResearchRuntime::readback_bytes);
        self.readback_bytes += total - self.research_readback;
        self.research_readback = total;
    }

    pub fn observe(&mut self) -> Result<ObservationSnapshot> {
        self.observe_rings(economy::accounting::audit::RingValidation::All)
    }
    pub(in crate::simulation) fn observe_rings(
        &mut self,
        rings: economy::accounting::audit::RingValidation<'_>,
    ) -> Result<ObservationSnapshot> {
        self.clear_control();
        let input = economy::accounting::audit::AuditInput {
            economy: self.gpu(),
            spatial: &self.spatial,
            partials: &self.work.events,
            summary: &self.summary,
            control: &self.control,
            ring_mask: self.snapshot_copy.ring_mask(),
            rings,
        };
        #[cfg(test)]
        match self.audit_execution {
            AuditExecution::Original => {
                if let Some(mask) = self.snapshot_copy.ring_mask() {
                    tensor::reset_ring_mask(&self.device, mask, tensor::RingMaskState::Unknown);
                }
                let count = self.metadata.actors.len();
                let parameters = self.parameters(Phase::Audit, Market::Labor, 0, count)?;
                let groups = count.div_ceil(SummaryLayout::ACTORS_PER_GROUP as usize);
                self.economy(EconomyStage::Summarize, groups, &parameters, &self.summary);
                let groups = self
                    .device
                    .tensor(&vec![groups as u32; SummaryLayout::METRICS]);
                reduction::summary(
                    &self.device,
                    &self.work.events,
                    &self.summary,
                    &self.control,
                    &groups,
                )
            }
            AuditExecution::Tiled => self.audit.summarize(input),
        }?;
        #[cfg(not(test))]
        self.audit.summarize(input)?;
        let totals = self
            .device
            .read_range(&self.summary, 0, SummaryLayout::WORDS)?;
        let totals = SummaryLayout::new(&totals)?;
        let control = self.device.read(&self.control)?;
        let control = ObservationControlLayout::new(&control)?;
        self.readback_bytes +=
            ((SummaryLayout::WORDS + SummaryLayout::CONTROL_WORDS) * WORD_BYTES) as u64;
        FaultFlags::new(control.scheduler.faults)?.into_result()?;
        let financial = self
            .finance
            .as_ref()
            .map(|finance| finance.observe(&self.device, &self.state, self.day as u32))
            .transpose()?
            .unwrap_or_default();
        self.readback_bytes += self
            .finance
            .as_ref()
            .map_or(0, finance::FinanceRuntime::readback_bytes_per_observation);
        let snapshot = ObservationSnapshot {
            economy: diagnostics::EconomyObservation {
                workforce: totals.workforce.into(),
                subsistence: totals.subsistence.into(),
                inactive: totals.inactive.into(),
                rural_employed: totals.rural_employed.into(),
                transport_workers: totals.transport_workers.into(),
                cartage_minutes: totals.cartage_minutes.into(),
                cartage_receipts: totals.cartage_receipts.into(),
                awaiting_pickup: totals.awaiting_pickup.into(),
                household_cash: totals.household_cash.into(),
                active_by_good: totals.active_by_good.map(u64::from),
                business_types: totals.business_types.map(u64::from),
                business_states: totals.business_states.map(u64::from),
                firm_cash: totals.firm_cash.map(u64::from),
                accounts: std::array::from_fn(|good| diagnostics::BusinessAccount {
                    opening_cash: totals.business_opening[good].into(),
                    closing_cash: totals.business_closing[good].into(),
                    revenue: totals.business_revenue[good].into(),
                }),
            },
            day: self.day,
            weather: self.scenario.weather(self.day),
            cash: add(totals.cash.into(), financial.cash()?)?,
            payroll_reserves: totals.payroll_reserves.into(),
            income_tax: totals.income_tax.into(),
            employee_charges: totals.employee_charges.into(),
            rent_collected: totals.rent_collected.into(),
            rent_arrears: totals.rent_arrears.into(),
            financial_cash: financial.cash()?,
            loan_principal: financial.principal,
            loan_interest: financial.interest_due,
            active_loans: financial.active_loans,
            overdue_loans: financial.overdue_loans,
            research: self
                .research
                .as_ref()
                .map(|research| research.summary(&self.device))
                .transpose()?
                .unwrap_or_default(),
            employed: totals.employed.into(),
            homeless_households: totals.homeless_households.into(),
            households: (self.metadata.actors.len() - self.metadata.actors.firms()) as u64,
            shortages: totals.shortages.into(),
            firms: self.metadata.actors.firms() as u64,
            firm_entries: self
                .metadata
                .actors
                .iter_from(self.scenario.actors())
                .filter(|actor| actor.kind > 0)
                .count() as u64,
            firm_closures: totals.closures.into(),
            distributors: totals.distributors.into(),
            household_production: totals.household_production.map(u64::from),
            population: self
                .demography
                .observation(&self.device, &self.state, &self.config)?,
            inventories: totals.inventories.map(u64::from),
            transit: totals.transit.map(u64::from),
            produced: totals.produced.map(u64::from),
            used: totals.used.map(u64::from),
            consumed: totals.consumed.into(),
            consumption: totals.consumption.map(u64::from),
            minimum_wage: totals.minimum_wage.into(),
            maximum_wage: totals.maximum_wage.into(),
            minimum_price: totals.minimum_price.into(),
            maximum_price: totals.maximum_price.into(),
            markets: std::array::from_fn(|good| {
                let count = u64::from(totals.firms_by_good[good]);
                let wages = match count {
                    0 => QuoteDistribution::default(),
                    _ => QuoteDistribution {
                        count,
                        minimum: totals.wage_minima[good].into(),
                        maximum: totals.wage_maxima[good].into(),
                        total: totals.wage_totals[good].into(),
                        squares: totals.wage_squares[good].into(),
                    },
                };
                let count = u64::from(totals.price_counts[good]);
                let prices = match count {
                    0 => QuoteDistribution::default(),
                    _ => QuoteDistribution {
                        count,
                        minimum: totals.price_minima[good].into(),
                        maximum: totals.price_maxima[good].into(),
                        total: totals.price_totals[good].into(),
                        squares: totals.price_squares[good].into(),
                    },
                };
                MarketObservation { wages, prices }
            }),
            wealth_bins: control.wealth_bins.iter().map(|&n| u64::from(n)).collect(),
            regions: control.regions.iter().map(|&n| u64::from(n)).collect(),
        };
        snapshot.audit(self.metadata.initial_cash, self.metadata.initial_goods)?;
        self.account_research_readbacks();
        Ok(snapshot)
    }
}

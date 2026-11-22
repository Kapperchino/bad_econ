use super::{
    DeviceEconomy, Simulation,
    buffers::EconomyBuffers,
    economy::EconomyStage,
    flags::{CONTROL_WORDS, FaultFlags},
    layout::*,
    market::{MarketExecution, Provisioning},
    parallel,
    parameters::{self, Discovery, EconomyParameters},
    policy::MATCHING_ROUNDS,
};
use crate::domain::{Good, Market, Phase};
use anyhow::{Context, Result};
use burn_wgpu::CubeTensor;

impl Simulation {
    pub(super) fn gpu(&self) -> DeviceEconomy<'_> {
        DeviceEconomy {
            device: &self.device,
            state: &self.state,
            actors: &self.actors,
            config: &self.config,
        }
    }

    pub(super) fn parameters(
        &self,
        phase: Phase,
        market: Market,
        offset: usize,
        count: usize,
    ) -> Result<EconomyParameters> {
        let day = u32::try_from(self.day).context("day identity capacity exhausted")?;
        match phase {
            Phase::Audit => (),
            _ => {
                let _ = day
                    .checked_add(self.scenario.ring() as u32 + 1)
                    .context("delivery date capacity exhausted")?;
            }
        }
        Ok(EconomyParameters {
            clock: crate::simulation::parameters::MarketClock::Scalars,
            scan: parallel::ScanMode::new(&self.device),
            empty_orders: parameters::EmptyOrderMode::Settle,
            day,
            matching_round: match self.provisioning {
                Provisioning::Standard => self.matching_round as u32,
                Provisioning::Minimum => 0,
                Provisioning::Choice => MATCHING_ROUNDS as u32 - 1,
            },
            settlement_rotation: (crate::arithmetic::DrawKey {
                seed: self.scenario.config().seed,
                day: self.day,
                phase: phase as u32,
                actor: 0,
                purpose: 0x5e771e,
                index: u64::from(market.code()) * MATCHING_ROUNDS as u64
                    + self.matching_round as u64,
            }
            .draw()
                % self.metadata.actors.len().max(1) as u64) as u32,
            phase,
            market: market.into(),
            good: match market {
                Market::Input(good) | Market::Consumer(good) => good as u32,
                Market::Household => Good::Bread as u32,
                _ => 0,
            },
            discovery: match self.market_execution {
                MarketExecution::Grouped => Discovery::Indexed,
                #[cfg(test)]
                MarketExecution::Original => Discovery::Original,
                #[cfg(test)]
                MarketExecution::Reference => Discovery::Linear,
            },
            offset: offset as u32,
            count: count as u32,
            status_start: 0,
            sort_source: 0,
            sort_width: 0,
            sort_count: 0,
        })
    }
    pub(super) fn market_buffers(&self) -> EconomyBuffers<'_> {
        EconomyBuffers {
            economy: self.gpu(),
            spatial: &self.spatial,
            proposals: &self.work.proposals,
            events: &self.work.scratch,
            control: &self.control,
        }
    }

    pub(super) fn economy(
        &self,
        entry: EconomyStage,
        count: usize,
        parameters: &EconomyParameters,
        output: &CubeTensor,
    ) {
        EconomyBuffers {
            proposals: output,
            events: &self.work.events,
            ..self.market_buffers()
        }
        .run(entry, count, parameters);
    }

    pub(super) fn feedback_operation(
        &self,
        stage: EconomyStage,
        count: usize,
        parameters: &EconomyParameters,
    ) {
        let spatial = match (stage, parameters.empty_orders) {
            (EconomyStage::RecordFeedback, parameters::EmptyOrderMode::FeedbackOnly) => {
                &self.searches.requests
            }
            _ => &self.spatial,
        };
        EconomyBuffers {
            spatial,
            control: &self.work.feedback,
            ..self.market_buffers()
        }
        .run(stage, count, parameters);
    }

    pub(super) fn apply_feedback(&mut self) -> Result<()> {
        let parameters = self.parameters(Phase::ProvisionPurchase, Market::Household, 0, 0)?;
        self.feedback_operation(
            EconomyStage::ApplyFeedback,
            self.metadata.actors.len(),
            &parameters,
        );
        let control = self
            .device
            .read_range(&self.work.feedback, 0, CONTROL_WORDS)?;
        self.readback_bytes += (CONTROL_WORDS * WORD_BYTES) as u64;
        FaultFlags::read(&control)?.into_result()
    }
}

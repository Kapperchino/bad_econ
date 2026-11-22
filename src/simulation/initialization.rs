#[cfg(target_endian = "big")]
use super::encode;
#[cfg(test)]
use super::{AuditExecution, RegistryExecution};
use super::{
    CopyDirection, Device, Simulation,
    buffers::PhaseBuffers,
    configuration, demography, economy, finance,
    layout::*,
    management,
    market::{FreightActivity, FreightExecution, MarketExecution, MarketReplay, Provisioning},
    market_graphs, orders, ownership, registry, research, tensor,
};
use crate::{domain::*, runtime::RuntimeState};
use anyhow::{Result, anyhow};
use bytemuck::Zeroable;

impl Simulation {
    pub fn new(scenario: Scenario, device: Device) -> Result<Self> {
        match scenario.memory_estimate() <= device.memory_budget.bytes() {
            true => Ok(()),
            false => Err(anyhow!(
                "scenario needs an estimated {} bytes, exceeding the device memory budget of {} bytes",
                scenario.memory_estimate(),
                device.memory_budget.bytes()
            )),
        }?;
        let initial = scenario.initial()?;
        Self::from_state(scenario, initial, device)
    }
    pub fn from_checkpoint(checkpoint: Checkpoint, device: Device) -> Result<Self> {
        let restored = checkpoint.restore()?;
        Self::from_state(restored.scenario, restored.state, device)
    }
    pub(super) fn from_state(
        scenario: Scenario,
        mut initial: State,
        mut device: Device,
    ) -> Result<Self> {
        unsafe {
            device.client.set_stream(burn::tensor::StreamId::current());
        }
        let required = scenario.memory_estimate().max(
            initial.values.len() as u64 * 24
                + initial.people.len() as u64 * 144
                + initial.actors.len() as u64 * 160,
        );
        match required <= device.memory_budget.bytes() {
            true => Ok(()),
            false => Err(anyhow!(
                "scenario needs an estimated {} bytes, exceeding the device memory budget of {} bytes",
                required,
                device.memory_budget.bytes()
            )),
        }?;
        device.reclaim_memory(required)?;
        let _ = AuditedState::new(&scenario, &initial)?;
        let snapshot = ObservationSnapshot::default();
        let demography = demography::DemographyRuntime::new(&device, &scenario, &mut initial)?;

        let mut config = configuration::Configuration::new(&device, &scenario, &initial);
        let count = initial.actors.len();
        let stride = match (count, scenario.config().finance.as_ref()) {
            (4096.., None) => registry::entries::registry_stride(
                &scenario,
                &initial.actors,
                device.memory_budget.bytes().saturating_sub(required),
            ),
            _ => count,
        };
        config.state_stride = stride as u32;
        let actor_layouts = initial
            .actors
            .iter()
            .map(|actor| ActorLayout::new(&actor, &scenario))
            .collect::<ActorMetadataRows>();
        let actors = device.tensor(&actor_layouts.words());
        let spatial = device.tensor(&SpatialIndex::new(&scenario, &initial).words);
        let state = {
            let values = tensor::state_values(&initial.values, count, stride);
            #[cfg(target_endian = "little")]
            let state = device.tensor(bytemuck::cast_slice(&values));
            #[cfg(target_endian = "big")]
            let state = device.tensor(bytemuck::cast_slice(&encode(&values)));
            state
        };
        let state_words = state.meta.num_elements();
        let audited = tensor::AuditedBuffer::default().extended(&device, state_words);
        let snapshot_copy = tensor::SnapshotCopy::new(&device, state_words);
        let research = scenario
            .config()
            .research
            .as_ref()
            .map(|config| match &initial.research {
                Some(snapshot) => research::ResearchRuntime::restore(
                    &device, config, &scenario, &initial, snapshot,
                ),
                None => research::ResearchRuntime::new(&device, config, &scenario, &initial),
            })
            .transpose()?
            .map(|research| research.with_actors(&initial.actors, stride as u32))
            .transpose()?;
        let finance = scenario
            .config()
            .finance
            .as_ref()
            .map(|config| {
                let actors = initial
                    .actors
                    .iter()
                    .enumerate()
                    .map(|(index, actor)| finance::FinancialActor {
                        identity: actor.identity,
                        cash: initial.coordinate(Field::Cash, index),
                        kind: match actor.kind {
                            0 => finance::FinancialActorKind::Household,
                            _ => finance::FinancialActorKind::Firm,
                        },
                    })
                    .collect::<Vec<_>>();
                finance::FinanceRuntime::new(
                    &device,
                    config,
                    config.layout(initial.actors.len() * scenario.fields())?,
                    &actors,
                )
            })
            .transpose()?;
        let migrations = registry::MigrationRegistry::new(&initial);
        let management = management::EnterpriseManagement::new(&device, &initial);
        initial.values.clear();
        initial.values.shrink_to_fit();
        initial.research = None;
        initial.people.clear();
        initial.people.shrink_to_fit();
        let work = PhaseBuffers::new(&device, count, scenario.config().batch_size);
        let control = device.tensor(bytemuck::cast_slice(&[ObservationControlLayout::zeroed()]));
        let summary = device.empty((scenario.fields() * TICK_WORDS).max(SummaryLayout::WORDS));
        let ownership = Some(ownership::OwnershipRuntime::new(&device, &initial)?);
        let enterprise_requests = device.tensor(&vec![0; initial.actors.len()]);
        let day = initial.day;
        let audit = economy::accounting::audit::AuditBuffers::new(&device, &initial.actors);
        let searches = orders::SearchBuffers::new(&device, &initial.actors);
        let mut simulation = Self {
            scenario,
            device,
            status: RuntimeState::Paused,
            snapshot,
            timings: Vec::new(),
            readback_bytes: 0,
            metadata: initial,
            day,
            audited_day: day,
            next_phase: Phase::Opening,
            state,
            audited,
            snapshot_copy,
            actors,
            config,
            spatial,
            work,
            searches,
            market_sort: economy::market::sorting::MarketSort::default(),
            market_graphs: market_graphs::MarketGraphs::default(),
            market_replay: MarketReplay::Reusable,
            freight_activity: FreightActivity::Active,
            freight_execution: FreightExecution::Synchronous,
            market_execution: MarketExecution::Grouped,
            control,
            summary,
            audit,
            #[cfg(test)]
            audit_execution: AuditExecution::Tiled,
            finance,
            research,
            research_readback: 0,
            demography,
            enterprise_requests,
            ownership,
            management,
            migrations,
            #[cfg(test)]
            registry_execution: RegistryExecution::Gpu,
            matching_round: 0,
            provisioning: Provisioning::Standard,
        };
        simulation.snapshot = simulation.observe()?;
        simulation.copy_audited(CopyDirection::Save)?;
        simulation.device.client.memory_cleanup()?;
        match simulation.device.reclaim_memory(0)?.bytes_reserved
            <= simulation.device.memory_budget.bytes()
        {
            true => Ok(simulation),
            false => Err(anyhow!(
                "simulation memory exceeded the device memory budget of {} bytes",
                simulation.device.memory_budget.bytes()
            )),
        }
    }

    pub(super) fn replace(&mut self, scenario: Scenario, state: State) -> Result<()> {
        let device = self.device.clone();
        #[cfg(test)]
        let registry_execution = self.registry_execution;
        *self = Self::from_state(scenario, state, device)?;
        #[cfg(test)]
        {
            self.registry_execution = registry_execution;
        }
        Ok(())
    }
}

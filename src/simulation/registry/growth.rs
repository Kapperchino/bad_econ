use super::commit::{RegistryAudit, apply_records, newborn_count};
use super::{self as registry, records::PendingRecord};
use crate::{
    domain::ActorRows,
    simulation::{
        DeviceEconomy, MarketReplay, Simulation, economy,
        enterprise::{management, ownership},
        gpu::tensor,
        market::graphs,
        runtime::profiling,
        storage::layout::{ActorLayout, ActorMetadataRows, TICK_WORDS, WORD_BYTES},
    },
};
use anyhow::{Result, anyhow};
use std::time::Instant;

struct EntryAllocation {
    stride: usize,
    budget: u64,
}

impl EntryAllocation {
    fn new(simulation: &Simulation, actors: &ActorRows) -> Result<Self> {
        let device = &simulation.device;
        let scenario = &simulation.scenario;
        let count = actors.len();
        let previous_stride = simulation.config.state_stride as usize;
        let budget = device.memory_budget.bytes();
        let actor_bytes = count as u64 * 192;
        let state_bytes = |stride| {
            let words = stride * scenario.fields() * TICK_WORDS;
            ((words + simulation.audited.additional_words(words)) * WORD_BYTES) as u64
        };
        let stride = match count > previous_stride {
            true => {
                let minimum = state_bytes(count) + actor_bytes;
                let reserved = device.reclaim_memory(minimum)?.bytes_reserved;
                registry::entries::registry_stride(
                    scenario,
                    actors,
                    budget.saturating_sub(reserved.saturating_add(minimum)),
                )
            }
            false => previous_stride,
        };
        let additional = actor_bytes
            + match stride == previous_stride {
                true => 0,
                false => state_bytes(stride),
            };
        let reserved = device.reclaim_memory(additional)?.bytes_reserved;
        match reserved.saturating_add(additional) <= budget {
            true => Ok(Self { stride, budget }),
            false => Err(anyhow!(
                "registry growth exceeds the device memory budget of {budget} bytes: {reserved} reserved and {additional} estimated for growth"
            )),
        }
    }

    fn check_reserved(&self, simulation: &Simulation) -> Result<()> {
        let reserved = simulation.device.reclaim_memory(0)?.bytes_reserved;
        let budget = self.budget;
        match reserved <= budget {
            true => Ok(()),
            false => Err(anyhow!(
                "registry allocation exceeds the device memory budget of {budget} bytes: {reserved} reserved"
            )),
        }
    }
}

impl Simulation {
    pub(in crate::simulation) fn commit_entries(
        &mut self,
        entries: &registry::entries::EntryRequests,
        mut plan: registry::RegistryPlan,
        births: &[PendingRecord],
        registry_audit: RegistryAudit,
    ) -> Result<()> {
        let mut checkpoint = profiling::audit_checkpoints(&self.device, self.day);
        self.market_graphs = graphs::MarketGraphs::default();
        checkpoint("ReleaseGraphs");
        let newborns = newborn_count(births);
        let next_identity = self.identity_after_births(newborns)?;
        let batch = registry::entries::EntryBatch::new(
            &self.device,
            &self.scenario,
            &self.metadata,
            &self.state,
            entries,
            &mut plan,
            next_identity,
        )?;
        checkpoint("EntryBatch");
        let ring_indices = batch.indices().clone();
        let count = batch.metadata.actors.len();
        let allocation = EntryAllocation::new(self, &batch.metadata.actors)?;
        let stride = allocation.stride;
        let state = batch.grow(&self.device, &self.state, self.scenario.fields(), stride);
        checkpoint("GrowState");
        let settlement = registry::settlement::SettlementBatch::new(
            &self.device,
            &self.scenario,
            &batch.metadata,
            &state,
            &entries.transitions,
            self.day,
            &plan.migrations,
        )?;
        checkpoint("SettlementBatch");
        let mut metadata = batch.metadata;
        if let Some(mut actors) = plan.actors.take() {
            actors.extend(metadata.actors.iter_from(actors.len()));
            metadata.actors = actors;
        }
        if let Some(batch) = &settlement {
            batch.update_owners(&mut metadata.actors);
            if let Some(companies) = &batch.companies {
                metadata.companies.clone_from(companies);
            }
        }
        if let Some(ownership) = settlement
            .as_ref()
            .and_then(|batch| batch.ownership.as_ref())
        {
            metadata.ownership = ownership.clone();
        }
        let rental_changes = plan
            .changed_rentals(settlement.as_ref())
            .collect::<Vec<_>>();
        let started = Instant::now();
        let mut config = self.config.updated_for_registry(
            &self.device,
            &self.scenario,
            &self.metadata.actors,
            &metadata,
            &rental_changes,
        );
        config.state_stride = stride as u32;
        if std::env::var_os("BAD_ECON_PROFILE_STAGES").is_some() {
            eprintln!(
                "CONFIG {} {:.3}",
                self.day,
                started.elapsed().as_secs_f64() * 1000.0
            );
        }
        checkpoint("Configuration");
        let additions = metadata
            .actors
            .iter_from(self.metadata.actors.len())
            .collect::<ActorRows>();
        let rows = additions
            .iter()
            .map(|actor| ActorLayout::new(&actor, &self.scenario))
            .collect::<ActorMetadataRows>();
        let previous_actors = registry::extend_actors(&self.device, &self.actors, &rows);
        let enterprise_requests =
            registry::records::upload_requests(&self.device, &entries.transitions, count);
        let actors = plan.patch_actors(
            self,
            &previous_actors,
            &metadata.actors,
            &enterprise_requests,
            settlement.as_ref(),
        );
        checkpoint("ActorMetadata");
        let spatial = match plan.transition {
            registry::RegistryTransition::Migration => plan.migrate_spatial(self, &metadata.actors),
            _ => registry::spatial::extend(
                &self.device,
                &self.scenario,
                &self.spatial,
                &actors,
                &additions,
            ),
        };
        checkpoint("Spatial");
        let mut demography = match newborns {
            0 => self.demography.clone(),
            _ => self.grow_births(births)?,
        };
        match batch.formations.is_empty() {
            true => demography.extend_firms(&self.device, count),
            false => {
                demography = demography.form_households(&self.device, &batch.formations, count)?;
            }
        }
        checkpoint("Demography");
        let research = self
            .research
            .as_ref()
            .map(|research| research.with_actors(&metadata.actors, stride as u32))
            .transpose()?;
        let ownership = Some(ownership::OwnershipRuntime::new(&self.device, &metadata)?);
        let management = management::EnterpriseManagement::new(&self.device, &metadata);
        let audit = economy::accounting::audit::AuditBuffers::new(&self.device, &metadata.actors);
        checkpoint("Miscellaneous");
        let searches = self.searches.resized(&self.device, &metadata.actors);
        let mut next = Self {
            scenario: self.scenario.clone(),
            device: self.device.clone(),
            status: self.status.clone(),
            snapshot: self.snapshot.clone(),
            timings: self.timings.clone(),
            readback_bytes: self.readback_bytes
                + batch.readback_bytes
                + settlement.as_ref().map_or(0, |batch| batch.readback_bytes),
            metadata,
            day: self.day,
            audited_day: self.audited_day,
            next_phase: self.next_phase,
            state,
            audited: self
                .audited
                .extended(&self.device, stride * self.scenario.fields() * TICK_WORDS),
            snapshot_copy: match stride == self.config.state_stride as usize {
                true => self.snapshot_copy.clone(),
                false => tensor::SnapshotCopy::new(
                    &self.device,
                    stride * self.scenario.fields() * TICK_WORDS,
                ),
            },
            actors,
            config,
            spatial,
            work: self
                .work
                .resized(&self.device, count, self.scenario.config().batch_size),
            searches,
            market_sort: economy::market::sorting::MarketSort::default(),
            market_graphs: graphs::MarketGraphs::default(),
            market_replay: match self.device.cube {
                cubecl::Device::Metal(_) => MarketReplay::Reusable,
                _ => MarketReplay::RegistryGrowth,
            },
            freight_activity: self.freight_activity,
            freight_execution: self.freight_execution,
            market_execution: self.market_execution,
            control: self.control.clone(),
            summary: self.summary.clone(),
            audit,
            #[cfg(test)]
            audit_execution: self.audit_execution,
            finance: None,
            research,
            research_readback: self.research_readback,
            demography,
            enterprise_requests,
            ownership,
            management,
            migrations: plan.migrations,
            #[cfg(test)]
            registry_execution: self.registry_execution,
            matching_round: self.matching_round,
            provisioning: self.provisioning,
        };
        checkpoint("Buffers");
        apply_records(
            DeviceEconomy {
                actors: &previous_actors,
                ..next.gpu()
            },
            settlement.as_ref(),
            &next.enterprise_requests,
            &next.work.scratch,
            next.day as u32,
        );
        checkpoint("Apply");
        let rings = match (registry_audit, &settlement) {
            #[cfg(test)]
            (RegistryAudit::Full, _) => economy::accounting::audit::RingValidation::All,
            (RegistryAudit::Validated, None) => {
                economy::accounting::audit::RingValidation::Selected(&ring_indices)
            }
            (RegistryAudit::Validated, Some(settlement)) => {
                economy::accounting::audit::RingValidation::Changed {
                    entries: &ring_indices,
                    settlements: settlement.indices(),
                }
            }
        };
        next.demography.remap_workers_from(
            &next.device,
            &next.state,
            next.config.state_stride,
            &self.demography,
        );
        next.snapshot = next.observe_rings(rings)?;
        checkpoint("Observe");
        allocation.check_reserved(&next)?;
        *self = next;
        checkpoint("RegistryCommit");
        Ok(())
    }
}

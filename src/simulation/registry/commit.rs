use super::{
    self as registry, RegistryPlan, RegistryTransition, records::PendingRecord,
    settlement::SettlementBatch,
};
use crate::{
    domain::{ActorRows, demography::PopulationObservation},
    simulation::{DeviceEconomy, Simulation, demography, enterprise, profiling},
};
use anyhow::{Context, Result, anyhow};
use burn_wgpu::CubeTensor;

enum RegistryChange {
    Unchanged,
    Births,
    Transitions(registry::RegistryPlan),
    Entries(registry::entries::EntryGrowth),
    Rebuild,
}

#[derive(Clone, Copy)]
pub(in crate::simulation) enum RegistryAudit {
    #[cfg(test)]
    Full,
    Validated,
}

#[cfg(test)]
#[derive(Clone, Copy)]
pub(in crate::simulation) enum RegistryExecution {
    Gpu,
    Reference,
}

pub(super) fn newborn_count(records: &[PendingRecord]) -> usize {
    records.iter().filter(|record| record.birth > 0).count()
}

pub(super) fn apply_records(
    gpu: DeviceEconomy<'_>,
    settlement: Option<&SettlementBatch>,
    requests: &CubeTensor,
    scratch: &CubeTensor,
    day: u32,
) {
    match settlement {
        Some(batch) => batch.apply(gpu.device, gpu.state, gpu.config.state_stride as usize),
        None => registry::apply(gpu, requests, scratch, day),
    }
}

impl RegistryPlan {
    pub(super) fn changed_rentals<'a>(
        &'a self,
        settlement: Option<&'a SettlementBatch>,
    ) -> impl Iterator<Item = usize> + 'a {
        self.changes
            .iter()
            .filter(|record| record.request >> 24 == 3)
            .map(|record| record.actor as usize)
            .chain(
                settlement
                    .into_iter()
                    .flat_map(|batch| batch.owners.iter().map(|change| change.firm)),
            )
    }

    pub(super) fn patch_actors(
        &self,
        simulation: &Simulation,
        buffer: &CubeTensor,
        rows: &ActorRows,
        requests: &CubeTensor,
        settlement: Option<&SettlementBatch>,
    ) -> CubeTensor {
        let device = &simulation.device;
        let scenario = &simulation.scenario;
        let patched = match self.transition {
            RegistryTransition::Unchanged => buffer.clone(),
            RegistryTransition::Capacity(registry::CapacityUpdate::Patch) => {
                registry::patch_capacity(device, buffer, requests)
            }
            RegistryTransition::Capacity(registry::CapacityUpdate::Rebuild)
            | RegistryTransition::Migration => {
                registry::metadata::patch(device, scenario, buffer, rows, &self.changes)
            }
        };
        match settlement.filter(|batch| !batch.owners.is_empty()) {
            Some(batch) => registry::metadata::patch_actors(
                device,
                scenario,
                &patched,
                rows,
                batch.owners.iter().map(|change| change.firm),
            ),
            None => patched,
        }
    }

    pub(super) fn migrate_spatial(
        &self,
        simulation: &Simulation,
        current: &ActorRows,
    ) -> CubeTensor {
        registry::spatial::migrate(
            &simulation.device,
            &simulation.scenario,
            registry::spatial::SpatialMigration {
                spatial: &simulation.spatial,
                actors: &simulation.actors,
                previous: &simulation.metadata.actors,
                current,
                requests: &self.changes,
            },
        )
    }
}

impl Simulation {
    pub(in crate::simulation::registry) fn identity_after_births(
        &self,
        newborns: usize,
    ) -> Result<u64> {
        self.metadata
            .next_identity
            .checked_add(newborns as u64)
            .context("identity capacity exhausted")
    }

    pub(in crate::simulation::registry) fn observe_grown_population(
        &self,
        grown: &demography::DemographyRuntime,
    ) -> Result<PopulationObservation> {
        grown.remap_workers_from(
            &self.device,
            &self.state,
            self.config.state_stride,
            &self.demography,
        );
        grown.observation(&self.device, &self.state, &self.config)
    }

    pub(in crate::simulation::registry) fn grow_births(
        &self,
        births: &[PendingRecord],
    ) -> Result<demography::DemographyRuntime> {
        #[cfg(test)]
        let grown = match self.registry_execution {
            RegistryExecution::Reference => {
                let mut dense = vec![0; self.metadata.actors.len()];
                for record in births {
                    dense[record.actor as usize] = record.birth;
                }
                self.demography
                    .reference_births(&self.device, &dense, self.metadata.next_identity)
            }
            RegistryExecution::Gpu => self.demography.grow_births(
                &self.device,
                births,
                self.metadata.next_identity,
                &self.work.scratch,
            ),
        };
        #[cfg(not(test))]
        let grown = self.demography.grow_births(
            &self.device,
            births,
            self.metadata.next_identity,
            &self.work.scratch,
        );
        grown
    }
    pub(in crate::simulation) fn commit_enterprises(&mut self, audit: RegistryAudit) -> Result<()> {
        let mut checkpoint = profiling::audit_checkpoints(&self.device, self.day);
        let records = registry::records::RecordBuffers {
            requests: &self.enterprise_requests,
            births: self.demography.birth_buffer(),
            scratch: &self.work.scratch,
        };
        enterprise::propose(self.gpu(), &self.spatial, records, self.day as u32);
        let pending = registry::records::PendingRecords::read(
            &self.device,
            records,
            &self.searches.requests,
        )?;
        self.readback_bytes += pending.readback_bytes;
        #[cfg(test)]
        let pending = match self.registry_execution {
            RegistryExecution::Gpu => pending,
            RegistryExecution::Reference => registry::records::PendingRecords::from_dense(
                &self.device.read(&self.enterprise_requests)?,
                &self.device.read(self.demography.birth_buffer())?,
            )?,
        };
        checkpoint("PendingRecords");
        let records = &pending.records;
        if std::env::var_os("BAD_ECON_PROFILE_REGISTRY").is_some() {
            let requests = records.iter().filter(|record| record.request > 0).fold(
                std::collections::BTreeMap::<u32, usize>::new(),
                |mut counts, record| {
                    *counts.entry(record.request >> 24).or_default() += 1;
                    counts
                },
            );
            eprintln!("REGISTRY {} Requests {requests:?}", self.day);
        }
        let count = records.iter().filter(|record| record.request > 0).count();
        let newborns = newborn_count(records);
        let change = match (count, newborns) {
            (0, 0) => RegistryChange::Unchanged,
            (0, _) => RegistryChange::Births,
            _ => {
                let plan = registry::RegistryPlan::new(
                    &self.scenario,
                    &self.metadata,
                    &self.migrations,
                    records,
                )?;
                checkpoint("RegistryPlan");
                match plan {
                    Some(plan) => RegistryChange::Transitions(plan),
                    None => {
                        let growth = registry::entries::EntryGrowth::new(
                            &self.scenario,
                            &self.metadata,
                            &self.migrations,
                            records,
                        )?;
                        checkpoint("EntryGrowth");
                        growth.map_or(RegistryChange::Rebuild, RegistryChange::Entries)
                    }
                }
            }
        };
        #[cfg(test)]
        let change = match (self.registry_execution, change) {
            (
                RegistryExecution::Reference,
                RegistryChange::Transitions(_) | RegistryChange::Entries(_),
            ) => RegistryChange::Rebuild,
            (_, change) => change,
        };
        match change {
            RegistryChange::Unchanged => Ok(()),
            RegistryChange::Births => self.commit_births(records, newborns),
            RegistryChange::Transitions(plan) => self.commit_registry_audited(plan, records, audit),
            RegistryChange::Entries(growth) => {
                self.commit_entries(&growth.requests, growth.plan, records, audit)
            }
            RegistryChange::Rebuild => self.rebuild_enterprises(records, count, newborns),
        }
    }

    fn commit_births(&mut self, records: &[PendingRecord], newborns: usize) -> Result<()> {
        let next_identity = self.identity_after_births(newborns)?;
        let grown = self.grow_births(records)?;
        let population = self.observe_grown_population(&grown)?;
        match self.device.reclaim_memory(0)?.bytes_reserved <= self.device.memory_budget.bytes() {
            true => (),
            false => Err(anyhow!(
                "person registry capacity exhausted while observing growth"
            ))?,
        }
        self.demography = grown;
        self.metadata.next_identity = next_identity;
        self.snapshot.population = population;
        Ok(())
    }

    fn rebuild_enterprises(
        &mut self,
        records: &[PendingRecord],
        count: usize,
        newborns: usize,
    ) -> Result<()> {
        let mut state = self.download()?;
        let estimated = (state.actors.len() + count) as u64 * self.scenario.fields() as u64 * 24
            + (state.people.len() + newborns) as u64 * 144;
        match estimated <= self.device.memory_budget.bytes() {
            true => (),
            false => Err(anyhow!(
                "registry capacity exhausted: {estimated} bytes exceed device budget"
            ))?,
        }
        for record in records.iter().filter(|record| record.birth > 0) {
            let identity = state.next_identity;
            state.next_identity = identity
                .checked_add(1)
                .context("identity capacity exhausted")?;
            state.people.push(crate::domain::demography::Person {
                identity,
                household: state.actors.at(record.actor as usize).identity,
                age_days: 0,
                sex: match record.birth {
                    1 => crate::domain::demography::Sex::Female,
                    _ => crate::domain::demography::Sex::Male,
                },
                state: crate::domain::demography::LivingState::Living,
                employer: None,
                work_minutes: 0,
                fatigue: 0,
                famine_exposure: 0,
                cold_exposure: 0,
                birth_cooldown: 0,
            });
        }
        let mut requests = vec![0; state.actors.len()];
        for record in records {
            requests[record.actor as usize] = record.request;
        }
        enterprise::enter(&mut state, &self.scenario, &requests)?;
        enterprise::transitions(&mut state, &self.scenario, &requests)?;
        let timings = self.timings.clone();
        let readback = self.readback_bytes;
        self.replace(self.scenario.clone(), state)?;
        self.timings = timings;
        self.readback_bytes += readback;
        Ok(())
    }
}

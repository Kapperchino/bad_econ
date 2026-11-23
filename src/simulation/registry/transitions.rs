use super::commit::{RegistryAudit, apply_records, newborn_count};
use super::{self as registry, records::PendingRecord, settlement::SettlementBatch};
use crate::simulation::{
    Simulation,
    economy::{self, configuration, transport::railway::Network},
    enterprise::{management, ownership},
    gpu::tensor,
    market::graphs,
    runtime::profiling,
};
use anyhow::{Context, Result, anyhow};
use std::time::Instant;

struct RegistryBindings {
    rows: Option<crate::domain::ActorRows>,
    buffers: registry::RegistryBuffers,
}

impl RegistryBindings {
    fn install(
        simulation: &mut Simulation,
        rows: Option<crate::domain::ActorRows>,
        buffers: registry::RegistryBuffers,
    ) -> Self {
        Self {
            rows: rows.map(|rows| std::mem::replace(&mut simulation.metadata.actors, rows)),
            buffers: registry::RegistryBuffers {
                actors: std::mem::replace(&mut simulation.actors, buffers.actors),
                spatial: std::mem::replace(&mut simulation.spatial, buffers.spatial),
            },
        }
    }

    fn commit(self, simulation: &mut Simulation) {
        let reused = |current: &mut burn_wgpu::CubeTensor, previous| {
            if !tensor::same_buffer(current, &previous) {
                tensor::copy_contiguous(&simulation.device, current, &previous);
                *current = previous;
            }
        };
        reused(&mut simulation.actors, self.buffers.actors);
        reused(&mut simulation.spatial, self.buffers.spatial);
    }

    fn restore(self, simulation: &mut Simulation) {
        if let Some(rows) = self.rows {
            simulation.metadata.actors = rows;
        }
        simulation.actors = self.buffers.actors;
        simulation.spatial = self.buffers.spatial;
    }
}

#[derive(Default)]
struct SettlementRuntimes {
    management: Option<management::EnterpriseManagement>,
    ownership: Option<ownership::OwnershipRuntime>,
    railway: Option<Network>,
}

impl SettlementRuntimes {
    fn new(
        simulation: &Simulation,
        plan: &mut registry::RegistryPlan,
        batch: Option<&SettlementBatch>,
    ) -> Result<Self> {
        match batch {
            Some(batch) => Self::prepare(simulation, plan, batch),
            None => Ok(Self::default()),
        }
    }

    fn prepare(
        simulation: &Simulation,
        plan: &mut registry::RegistryPlan,
        batch: &SettlementBatch,
    ) -> Result<Self> {
        let mut staged = None;
        let mut metadata = || simulation.metadata.clone();
        let management = match batch.owners.is_empty() {
            false => {
                let actors = plan
                    .actors
                    .get_or_insert_with(|| simulation.metadata.actors.clone());
                batch.update_owners(actors);
                let state = staged.get_or_insert_with(&mut metadata);
                state.actors.clone_from(actors);
                Some(management::EnterpriseManagement::new(
                    &simulation.device,
                    state,
                ))
            }
            true => None,
        };
        let ownership = batch
            .ownership
            .as_ref()
            .map(|ownership| {
                let state = staged.get_or_insert_with(&mut metadata);
                state.ownership.clone_from(ownership);
                ownership::OwnershipRuntime::new(&simulation.device, state)
            })
            .transpose()?;
        let railway = batch
            .companies
            .as_ref()
            .filter(|_| !simulation.scenario.config().railway.companies.is_empty())
            .map(|companies| {
                let state = staged.get_or_insert_with(&mut metadata);
                state.companies.clone_from(companies);
                simulation
                    .config
                    .railway
                    .updated(&simulation.device, &simulation.scenario, state)
            });
        Ok(Self {
            management,
            ownership,
            railway,
        })
    }

    fn commit(self, simulation: &mut Simulation) {
        if let Some(ownership) = self.ownership {
            simulation.ownership = Some(ownership);
        }
        if let Some(management) = self.management {
            simulation.management = management;
        }
        if let Some(railway) = self.railway {
            simulation.config.railway = railway;
            simulation.market_graphs = graphs::MarketGraphs::default();
        }
    }
}

impl Simulation {
    #[cfg(test)]
    pub(in crate::simulation) fn commit_registry(
        &mut self,
        plan: registry::RegistryPlan,
        births: &[PendingRecord],
    ) -> Result<()> {
        self.commit_registry_audited(plan, births, RegistryAudit::Full)
    }
    pub(in crate::simulation) fn commit_registry_audited(
        &mut self,
        mut plan: registry::RegistryPlan,
        births: &[PendingRecord],
        audit: RegistryAudit,
    ) -> Result<()> {
        let mut checkpoint = profiling::audit_checkpoints(&self.device, self.day);
        let mut settlement = registry::settlement::SettlementBatch::new(
            &self.device,
            &self.scenario,
            &self.metadata,
            &self.state,
            births,
            self.day,
            &plan.migrations,
        )?;
        checkpoint("RegistrySettlement");
        let runtimes = SettlementRuntimes::new(self, &mut plan, settlement.as_ref())?;
        checkpoint("RegistryOwnership");
        let newborns = newborn_count(births);
        let next_identity = self.identity_after_births(newborns)?;
        let grown = match newborns {
            0 => None,
            _ => Some(self.grow_births(births)?),
        };
        checkpoint("RegistryBirths");
        let rows = match plan.transition {
            registry::RegistryTransition::Capacity(registry::CapacityUpdate::Rebuild)
            | registry::RegistryTransition::Migration => {
                plan.actors.as_ref().context("missing changed actor rows")?
            }
            _ => plan.actors.as_ref().unwrap_or(&self.metadata.actors),
        };
        let spatial = match plan.transition {
            registry::RegistryTransition::Migration => plan.migrate_spatial(self, rows),
            _ => self.spatial.clone(),
        };
        let buffers = registry::RegistryBuffers {
            actors: plan.patch_actors(
                self,
                &self.actors,
                rows,
                &self.enterprise_requests,
                settlement.as_ref(),
            ),
            spatial,
        };
        checkpoint("RegistryBuffers");
        match self.device.reclaim_memory(0)?.bytes_reserved <= self.device.memory_budget.bytes() {
            true => (),
            false => Err(anyhow!(
                "registry capacity exhausted while preparing GPU transitions"
            ))?,
        }
        apply_records(
            self.gpu(),
            settlement.as_ref(),
            &self.enterprise_requests,
            &self.work.scratch,
            self.day as u32,
        );
        self.readback_bytes += settlement.as_ref().map_or(0, |batch| batch.readback_bytes);
        checkpoint("RegistryApply");
        let population = grown
            .as_ref()
            .map(|grown| self.observe_grown_population(grown))
            .transpose();
        checkpoint("RegistryPopulation");
        let bindings = RegistryBindings::install(self, plan.actors.take(), buffers);
        let rings = match (audit, &settlement) {
            #[cfg(test)]
            (RegistryAudit::Full, _) => economy::accounting::audit::RingValidation::All,
            (RegistryAudit::Validated, Some(batch)) => {
                economy::accounting::audit::RingValidation::Selected(batch.indices())
            }
            (RegistryAudit::Validated, None) => {
                economy::accounting::audit::RingValidation::Unchanged
            }
        };
        let registry_observation = economy::accounting::audit::RegistryObservation::new(
            audit,
            births,
            settlement.as_ref(),
        );
        if std::env::var_os("BAD_ECON_PROFILE_REGISTRY").is_some() {
            eprintln!("REGISTRY {} Observation {registry_observation:?}", self.day);
        }
        let observation = population
            .and_then(|population| {
                match registry_observation {
                    economy::accounting::audit::RegistryObservation::Existing => {
                        self.fault_check().map(|_| self.snapshot.clone())
                    }
                    economy::accounting::audit::RegistryObservation::Updated => {
                        self.observe_rings(rings)
                    }
                }
                .map(|mut snapshot| {
                    if let Some(population) = population {
                        snapshot.population = population;
                    }
                    snapshot
                })
            })
            .and_then(|observation| {
                match self.device.reclaim_memory(0)?.bytes_reserved
                    <= self.device.memory_budget.bytes()
                {
                    true => Ok(observation),
                    false => Err(anyhow!(
                        "registry capacity exhausted while auditing GPU transitions"
                    )),
                }
            });
        checkpoint("RegistryObservation");
        match observation {
            Ok(snapshot) => {
                match (
                    &runtimes.management,
                    plan.transition,
                    self.config.property_mode,
                    self.config.rental_businesses,
                ) {
                    (Some(_), _, _, _)
                    | (
                        _,
                        registry::RegistryTransition::Migration,
                        configuration::Feature::Enabled,
                        _,
                    )
                    | (_, registry::RegistryTransition::Migration, _, 1..) => {
                        let started = Instant::now();
                        self.config.update_rentals(
                            &self.device,
                            &self.scenario,
                            &self.metadata,
                            plan.changed_rentals(settlement.as_ref()),
                        );
                        if std::env::var_os("BAD_ECON_PROFILE_STAGES").is_some() {
                            eprintln!(
                                "CONFIG {} {:.3}",
                                self.day,
                                started.elapsed().as_secs_f64() * 1000.0
                            );
                        }
                    }
                    _ => (),
                }
                runtimes.commit(self);
                if let Some(companies) =
                    settlement.as_mut().and_then(|batch| batch.companies.take())
                {
                    self.metadata.companies = companies;
                }
                if let Some(ownership) = settlement.and_then(|batch| batch.ownership) {
                    self.metadata.ownership = ownership;
                }
                if let Some(grown) = grown {
                    self.demography = grown;
                }
                self.metadata.next_identity = next_identity;
                self.migrations = plan.migrations;
                self.snapshot = snapshot;
                bindings.commit(self);
                Ok(())
            }
            Err(error) => {
                bindings.restore(self);
                Err(error)
            }
        }
    }
}

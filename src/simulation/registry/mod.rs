use crate::simulation::DeviceEconomy;
use crate::simulation::configuration::EconomyConfig;
mod commit;
pub(super) mod entries;
mod growth;
pub(super) mod metadata;
pub(super) mod records;
pub(super) mod settlement;
pub(super) mod spatial;
mod staging;
mod systems;
mod transitions;
use super::{
    Device, enterprise,
    layout::ActorLayout as A,
    pass::GpuPass,
    state::{EconomyView, ReadLedger, ReadLedgerExpand},
    system::{AsBufferArg, System, gpu_passes},
    words::*,
};
use crate::domain::{ActorRows, Field, Position, Scenario, State};
use anyhow::{Context, Result};
use burn_wgpu::CubeTensor;
pub(super) use commit::RegistryAudit;
#[cfg(test)]
pub(super) use commit::RegistryExecution;
use cubecl::{CubeDim, prelude::*};
use records::PendingRecord;
use std::collections::BTreeMap;
use systems::*;

pub(super) const WAGE_DEPENDENCY: u32 = 10 << 24;

#[derive(Clone, Default)]
pub(super) struct MigrationRegistry {
    destinations: BTreeMap<usize, Position>,
    children: BTreeMap<u64, Vec<usize>>,
}
impl MigrationRegistry {
    pub fn new(state: &State) -> Self {
        Self {
            destinations: (0..state.actors.len())
                .filter(|&actor| state.get(Field::MoveDue, actor) > 0)
                .map(|actor| {
                    (
                        actor,
                        Position {
                            x: state.get(Field::MoveX, actor) as u32,
                            y: state.get(Field::MoveY, actor) as u32,
                        },
                    )
                })
                .collect(),
            children: (0..state.actors.len())
                .filter(|&actor| state.get(Field::ParentHousehold, actor) > 0)
                .fold(BTreeMap::new(), |mut children, actor| {
                    children
                        .entry(state.get(Field::ParentHousehold, actor))
                        .or_default()
                        .push(actor);
                    children
                }),
        }
    }

    pub fn extend_households(
        &mut self,
        actors: &ActorRows,
        formations: &[super::enterprise::HouseholdFormation],
    ) {
        for formation in formations {
            self.children
                .entry(actors.at(formation.parent).identity)
                .or_default()
                .push(formation.household);
        }
    }

    pub fn children(&self, household: u64) -> impl Iterator<Item = usize> + '_ {
        self.children.get(&household).into_iter().flatten().copied()
    }
}

#[derive(Clone, Copy)]
pub(super) enum CapacityUpdate {
    Patch,
    Rebuild,
}

#[derive(Clone, Copy)]
pub(super) enum RegistryTransition {
    Unchanged,
    Capacity(CapacityUpdate),
    Migration,
}

pub(super) struct RegistryBuffers {
    pub actors: CubeTensor,
    pub spatial: CubeTensor,
}

pub(super) struct RegistryPlan {
    pub transition: RegistryTransition,
    pub actors: Option<ActorRows>,
    pub changes: Vec<PendingRecord>,
    pub migrations: MigrationRegistry,
}
impl RegistryPlan {
    pub fn new(
        scenario: &Scenario,
        metadata: &State,
        migrations: &MigrationRegistry,
        requests: &[PendingRecord],
    ) -> Result<Option<Self>> {
        Self::with_growth(scenario, metadata, migrations, requests, 0)
    }

    fn with_growth(
        scenario: &Scenario,
        metadata: &State,
        migrations: &MigrationRegistry,
        requests: &[PendingRecord],
        additions: usize,
    ) -> Result<Option<Self>> {
        let supported = requests.iter().all(|record| match record.request >> 24 {
            0 => record.request == 0,
            2 | 3 | 10 => true,
            4 | 7 | enterprise::acquisitions::PURCHASE => scenario.config().finance.is_none(),
            5 => {
                scenario.config().finance.is_none()
                    || !migrations
                        .children
                        .contains_key(&metadata.actors.at(record.actor as usize).identity)
            }
            6 => record.payload() > 0,
            _ => false,
        });
        match supported {
            false => Ok(None),
            true => {
                let mut next = migrations.clone();
                let mut actors = requests
                    .iter()
                    .any(|record| matches!(record.request >> 24, 3 | 6))
                    .then(|| metadata.actors.clone_for_growth(additions));
                let mut capacity_update = CapacityUpdate::Patch;
                for record in requests {
                    let index = record.actor as usize;
                    let action = record.request;
                    match action >> 24 {
                        2 => {
                            let destination = record.target();
                            next.destinations.insert(
                                index,
                                actors
                                    .as_ref()
                                    .unwrap_or(&metadata.actors)
                                    .get(destination)
                                    .context("unknown moving destination")?
                                    .position,
                            );
                        }
                        3 => {
                            let position = migrations
                                .destinations
                                .get(&index)
                                .context("missing migration in progress")?;
                            let rows = actors.as_mut().context("missing arrival registry")?;
                            let mut actor = rows.at(index);
                            actor.position = *position;
                            rows.set(index, actor);
                            next.destinations.remove(&index);
                        }
                        6 => {
                            let rows = actors.as_mut().context("missing capacity registry")?;
                            let mut actor = rows.at(index);
                            if actor.capacity == 0 {
                                capacity_update = CapacityUpdate::Rebuild;
                            }
                            actor.capacity = actor
                                .capacity
                                .checked_add(record.payload())
                                .context("firm capacity exhausted")?;
                            rows.set(index, actor);
                        }
                        _ => (),
                    }
                }
                let transition = match (
                    requests.iter().any(|record| record.request >> 24 == 3),
                    requests.iter().any(|record| record.request >> 24 == 6),
                ) {
                    (true, _) => RegistryTransition::Migration,
                    (_, true) => RegistryTransition::Capacity(capacity_update),
                    _ => RegistryTransition::Unchanged,
                };
                Ok(Some(Self {
                    transition,
                    actors,
                    changes: requests
                        .iter()
                        .copied()
                        .filter(|record| matches!(record.request >> 24, 3 | 6 | 10))
                        .collect(),
                    migrations: next,
                }))
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Stage {
    Clear,
    Count,
    Apply,
}

#[allow(
    clippy::manual_div_ceil,
    reason = "CubeCL lowers explicit integer division"
)]
#[cube]
fn transition(
    economy: &mut EconomyView<'_>,
    requests: &[u32],
    incoming: &[Atomic<u32>],
    day: u32,
    #[comptime] stage: Stage,
) {
    let config = economy.ledger.config;
    let actor = ABSOLUTE_POS as u32;
    let count = config.actors;
    if actor < count {
        let action = requests[actor as usize];
        let kind = action >> 24;
        if comptime!(stage == Stage::Clear) {
            incoming[actor as usize].store(0);
        }
        if comptime!(stage == Stage::Count) && kind == 2 {
            let destination = (action & 0x00ffffff) - 1;
            incoming[destination as usize].fetch_add(1);
        }
        if comptime!(stage == Stage::Apply) {
            let received = times32(incoming[actor as usize].load(), 100);
            let cash = economy.ledger.value(Field::Cash, actor);
            let available = plus(cash, received).value;
            let moving_cost = select(kind == 2, ticks(100, 0), zero());
            economy
                .ledger
                .set(Field::Cash, actor, minus(available, moving_cost));
            if kind == 2 {
                let destination = (action & 0x00ffffff) - 1;
                let x = economy.actors.x(actor);
                let y = economy.actors.y(actor);
                let arrived = requests[destination as usize] >> 24 == 3 && destination < actor;
                let dx = select(
                    arrived,
                    economy.ledger.number(Field::MoveX, destination),
                    economy.actors.x(destination),
                );
                let dy = select(
                    arrived,
                    economy.ledger.number(Field::MoveY, destination),
                    economy.actors.y(destination),
                );
                let distance = max(x, dx) - min(x, dx) + max(y, dy) - min(y, dy);
                let speed = config.shipment_speed;
                let duration = max((distance + speed - 1) / speed, 1);
                economy.ledger.set_number(Field::PreviousX, actor, x);
                economy.ledger.set_number(Field::PreviousY, actor, y);
                economy.ledger.set_number(Field::MoveX, actor, dx);
                economy.ledger.set_number(Field::MoveY, actor, dy);
                economy
                    .ledger
                    .set_number(Field::MoveDue, actor, day + duration);
            }
            if kind == 3 || kind == 10 {
                economy.ledger.set(Field::MoveDue, actor, zero());
                economy.ledger.set(Field::MoveTarget, actor, zero());
            }
            if (kind == 3 || kind == 10)
                && comptime!(
                    config.property_mode == crate::simulation::configuration::Feature::Enabled
                        || config.economy == crate::domain::Economy::Industrial
                )
            {
                economy.ledger.set(Field::LandPlots, actor, zero());
                economy.ledger.set(Field::CropWork, actor, zero());
            }
            if kind == 6 {
                economy.ledger.set(Field::PendingCapacity, actor, zero());
            }
            if kind == 5 {
                economy.ledger.set(
                    Field::EstateSettled,
                    actor,
                    ticks(
                        comptime!(crate::domain::enterprise::EstateStatus::Unclaimed as u32),
                        0,
                    ),
                );
            }
        }
    }
}

pub(super) fn apply(
    DeviceEconomy {
        device,
        state,
        actors,
        config,
    }: DeviceEconomy<'_>,
    requests: &CubeTensor,
    scratch: &CubeTensor,
    day: u32,
) {
    let cube_count = device.count(requests.meta.num_elements());
    for stage in [Stage::Clear, Stage::Count, Stage::Apply] {
        RegistryTransitions::launch(
            device,
            cube_count.clone(),
            CubeDim::new_1d(device.workgroup),
            TransitionInputLaunch {
                state: state.buffer_arg(),
                actors: actors.buffer_arg(),
                config: config.launch(),
                requests: requests.buffer_arg(),
                incoming: scratch.buffer_arg(),
                day,
                stage,
            },
        );
    }
}

pub(super) fn patch_capacity(
    device: &Device,
    actors: &CubeTensor,
    requests: &CubeTensor,
) -> CubeTensor {
    let output = device.empty(actors.meta.num_elements());
    CapacityPatching::launch_elements(
        device,
        actors.meta.num_elements(),
        CapacityInputLaunch {
            actors: actors.buffer_arg(),
            requests: requests.buffer_arg(),
            output: output.buffer_arg(),
        },
    );
    output
}

pub(super) fn extend_actors(
    device: &Device,
    actors: &CubeTensor,
    additions: &super::layout::ActorMetadataRows,
) -> CubeTensor {
    let additions = device.tensor(&additions.words());
    let output = device.empty(actors.meta.num_elements() + additions.meta.num_elements());
    ActorGrowth::launch_elements(
        device,
        output.meta.num_elements(),
        ActorGrowthInputLaunch {
            actors: actors.buffer_arg(),
            additions: additions.into_buffer_arg(),
            output: output.buffer_arg(),
        },
    );
    output
}

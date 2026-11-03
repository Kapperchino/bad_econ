use crate::simulation::configuration::{Configuration, EconomyConfig};
pub(super) mod entries;
use super::{Device, layout::ActorLayout as A, words::*};
use crate::domain::{ActorRows, Field, Position, Scenario, State};
use anyhow::{Context, Result};
use burn_wgpu::CubeTensor;
use cubecl::{CubeDim, prelude::*};
use std::collections::BTreeMap;

#[derive(Clone, Default)]
pub(super) struct MigrationRegistry {
    destinations: BTreeMap<usize, Position>,
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
        }
    }
}

pub(super) struct RegistryPlan {
    pub actors: Option<ActorRows>,
    pub migrations: MigrationRegistry,
}
impl RegistryPlan {
    pub fn new(
        scenario: &Scenario,
        metadata: &State,
        migrations: &MigrationRegistry,
        requests: &[u32],
    ) -> Result<Option<Self>> {
        let heirs_possible = metadata
            .actors
            .iter()
            .filter(|actor| actor.kind == 0)
            .count()
            > scenario.config().households as usize;
        let supported = requests.iter().all(|&action| match action >> 24 {
            0 => action == 0,
            2 | 3 => true,
            5 => !heirs_possible,
            6 => action & 0x00ffffff > 0,
            _ => false,
        });
        match supported {
            false => Ok(None),
            true => {
                let mut next = migrations.clone();
                let mut actors = requests
                    .iter()
                    .any(|action| matches!(action >> 24, 3 | 6))
                    .then(|| metadata.actors.clone());
                for (index, action) in requests.iter().enumerate() {
                    match action >> 24 {
                        2 => {
                            let destination = (action & 0x00ffffff) as usize - 1;
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
                            actor.capacity = actor
                                .capacity
                                .checked_add(action & 0x00ffffff)
                                .context("firm capacity exhausted")?;
                            rows.set(index, actor);
                        }
                        _ => (),
                    }
                }
                Ok(Some(Self {
                    actors,
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
#[cube(launch)]
fn transition(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    requests: &[u32],
    incoming: &[Atomic<u32>],
    day: u32,
    #[comptime] stage: Stage,
) {
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
            let cash = load(state, comptime!(Field::Cash as u32) * count + actor);
            let available = plus(cash, received).value;
            let moving_cost = select(kind == 2, ticks(100, 0), zero());
            store(
                state,
                comptime!(Field::Cash as u32) * count + actor,
                minus(available, moving_cost),
            );
            if kind == 2 {
                let destination = (action & 0x00ffffff) - 1;
                let stride = actors.len() / A::WORDS;
                let x = actors[A::X * stride + actor as usize];
                let y = actors[A::Y * stride + actor as usize];
                let arrived = requests[destination as usize] >> 24 == 3 && destination < actor;
                let dx = select(
                    arrived,
                    load(state, comptime!(Field::MoveX as u32) * count + destination)
                        .extract(0usize),
                    actors[A::X * stride + destination as usize],
                );
                let dy = select(
                    arrived,
                    load(state, comptime!(Field::MoveY as u32) * count + destination)
                        .extract(0usize),
                    actors[A::Y * stride + destination as usize],
                );
                let distance = max(x, dx) - min(x, dx) + max(y, dy) - min(y, dy);
                let speed = config.shipment_speed;
                let duration = max((distance + speed - 1) / speed, 1);
                store(
                    state,
                    comptime!(Field::PreviousX as u32) * count + actor,
                    ticks(x, 0),
                );
                store(
                    state,
                    comptime!(Field::PreviousY as u32) * count + actor,
                    ticks(y, 0),
                );
                store(
                    state,
                    comptime!(Field::MoveX as u32) * count + actor,
                    ticks(dx, 0),
                );
                store(
                    state,
                    comptime!(Field::MoveY as u32) * count + actor,
                    ticks(dy, 0),
                );
                store(
                    state,
                    comptime!(Field::MoveDue as u32) * count + actor,
                    ticks(day + duration, 0),
                );
            }
            if kind == 3 {
                store(
                    state,
                    comptime!(Field::MoveDue as u32) * count + actor,
                    zero(),
                );
                store(
                    state,
                    comptime!(Field::MoveTarget as u32) * count + actor,
                    zero(),
                );
            }
            if kind == 6 {
                store(
                    state,
                    comptime!(Field::PendingCapacity as u32) * count + actor,
                    zero(),
                );
            }
            if kind == 5 {
                store(
                    state,
                    comptime!(Field::EstateSettled as u32) * count + actor,
                    ticks(2, 0),
                );
            }
        }
    }
}

pub(super) fn apply(
    device: &Device,
    state: &CubeTensor,
    actors: &CubeTensor,
    config: &Configuration,
    requests: &CubeTensor,
    scratch: &CubeTensor,
    day: u32,
) {
    let argument = |tensor: &CubeTensor| tensor.clone().into_buffer_arg();
    let cube_count = device.count(requests.meta.num_elements());
    for stage in [Stage::Clear, Stage::Count, Stage::Apply] {
        transition::launch(
            &device.client,
            cube_count.clone(),
            CubeDim::new_1d(device.workgroup),
            argument(state),
            argument(actors),
            config.launch(),
            argument(requests),
            argument(scratch),
            day,
            stage,
        );
    }
}

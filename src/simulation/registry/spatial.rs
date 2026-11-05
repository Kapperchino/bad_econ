use super::*;
use crate::simulation::layout::{DISCOVERY_BLOCK, SpatialIndex};

#[derive(CubeType, CubeLaunch)]
struct SpatialGrowth {
    before: u32,
    added: u32,
    side: u32,
    radius: u32,
    query_side: u32,
    query_tile: u32,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum CellKind {
    Market,
    Query,
}

pub(in crate::simulation) fn extend(
    device: &Device,
    scenario: &Scenario,
    prior: &CubeTensor,
    actors: &CubeTensor,
    additions: &ActorRows,
) -> CubeTensor {
    let config = scenario.config();
    let count = actors.meta.num_elements() / A::WORDS;
    let query_tile = config.map_size.div_ceil(128);
    let growth = SpatialGrowth {
        before: (count - additions.len()) as u32,
        added: additions.len() as u32,
        side: config.map_size.div_ceil(config.search_radius),
        radius: config.search_radius,
        query_side: config.map_size.div_ceil(query_tile),
        query_tile,
    };
    let cells = (growth.side * growth.side) as usize;
    let words = 3 * (cells + 1 + count) + count.div_ceil(DISCOVERY_BLOCK as usize) + 1 + count;
    let added = device.tensor(&SpatialIndex::from_actors(scenario, additions).words);
    let output = device.empty(words);
    SpatialClearing::launch(
        device,
        device.count(words),
        CubeDim::new_1d(device.workgroup),
        SpatialClearInputLaunch {
            output: output.clone().into_buffer_arg(),
        },
    );
    SpatialGrowthSystem::launch(
        device,
        device.count(count.max(cells + 1)),
        CubeDim::new_1d(device.workgroup),
        SpatialGrowthInputLaunch {
            prior: prior.clone().into_buffer_arg(),
            additions: added.into_buffer_arg(),
            actors: actors.clone().into_buffer_arg(),
            output: output.clone().into_buffer_arg(),
            growth: SpatialGrowthLaunch {
                before: growth.before,
                added: growth.added,
                side: growth.side,
                radius: growth.radius,
                query_side: growth.query_side,
                query_tile: growth.query_tile,
            },
        },
    );
    output
}

#[cube]
fn cell(actors: &[u32], growth: &SpatialGrowth, actor: u32, #[comptime] kind: CellKind) -> u32 {
    let count = growth.before + growth.added;
    let x = actors[A::X * count as usize + actor as usize];
    let y = actors[A::Y * count as usize + actor as usize];
    match kind {
        CellKind::Market => y / growth.radius * growth.side + x / growth.radius,
        CellKind::Query => y / growth.query_tile * growth.query_side + x / growth.query_tile,
    }
}

#[cube]
fn query_offset(
    spatial: &[u32],
    actors: &[u32],
    growth: &SpatialGrowth,
    base: u32,
    count: u32,
    actor_offset: u32,
    target: u32,
) -> u32 {
    let mut low = 0u32;
    let mut high = count;
    while low < high {
        let middle = (low + high) / 2;
        let actor = spatial[(base + middle) as usize] + actor_offset;
        let before = cell(actors, growth, actor, comptime!(CellKind::Query)) < target;
        low = select(before, middle + 1, low);
        high = select(before, high, middle);
    }
    low
}

#[derive(CubeType, CubeLaunch)]
struct SpatialClearInput {
    output: Box<[u32]>,
}

#[derive(CubeType, CubeLaunch)]
struct SpatialGrowthInput {
    prior: Box<[u32]>,
    additions: Box<[u32]>,
    actors: Box<[u32]>,
    output: Box<[u32]>,
    growth: SpatialGrowth,
}

struct SpatialClearing;
struct SpatialGrowthSystem;
struct ClearSpatialRows;
struct MergeSpatialRows;

impl System for SpatialClearing {
    type Input = SpatialClearInput;
    type Passes = gpu_passes![ClearSpatialRows];
}

impl System for SpatialGrowthSystem {
    type Input = SpatialGrowthInput;
    type Passes = gpu_passes![MergeSpatialRows];
}

#[cube]
impl GpuPass<SpatialClearInput> for ClearSpatialRows {
    type Output = ();

    fn execute(mut input: SpatialClearInput) {
        clear(&mut input.output);
    }
}

#[cube]
impl GpuPass<SpatialGrowthInput> for MergeSpatialRows {
    type Output = ();

    fn execute(mut input: SpatialGrowthInput) {
        merge(
            &input.prior,
            &input.additions,
            &input.actors,
            &mut input.output,
            &input.growth,
        );
    }
}

#[cube]
fn clear(output: &mut [u32]) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = 0;
    }
}

#[cube]
fn merge(
    prior: &[u32],
    additions: &[u32],
    actors: &[u32],
    output: &mut [u32],
    growth: &SpatialGrowth,
) {
    let id = ABSOLUTE_POS as u32;
    let cells = growth.side * growth.side;
    let before = growth.before;
    let added = growth.added;
    let count = before + added;
    let prior_segment = cells + 1 + before;
    let added_segment = cells + 1 + added;
    let segment = cells + 1 + count;
    #[unroll]
    for part in 1..3u32 {
        let old = part * prior_segment;
        let extra = part * added_segment;
        let destination = part * segment;
        if id <= cells {
            output[(destination + id) as usize] =
                prior[(old + id) as usize] + additions[(extra + id) as usize];
        }
        if id < prior[(old + cells) as usize] {
            let actor = prior[(old + cells + 1 + id) as usize];
            let cell = cell(actors, growth, actor, comptime!(CellKind::Market));
            let slot = id + additions[(extra + cell) as usize];
            output[(destination + cells + 1 + slot) as usize] = actor;
        }
        if id < additions[(extra + cells) as usize] {
            let actor = additions[(extra + cells + 1 + id) as usize] + before;
            let cell = cell(actors, growth, actor, comptime!(CellKind::Market));
            let slot = prior[(old + cell + 1) as usize] + id;
            output[(destination + cells + 1 + slot) as usize] = actor;
        }
    }
    let prior_query = 3 * prior_segment + ceildiv(before, DISCOVERY_BLOCK) + 1;
    let added_query = 3 * added_segment + ceildiv(added, DISCOVERY_BLOCK) + 1;
    let query = 3 * segment + ceildiv(count, DISCOVERY_BLOCK) + 1;
    if id < before {
        let actor = prior[(prior_query + id) as usize];
        let cell = cell(actors, growth, actor, comptime!(CellKind::Query));
        let slot = id + query_offset(additions, actors, growth, added_query, added, before, cell);
        output[(query + slot) as usize] = actor;
    }
    if id < added {
        let actor = additions[(added_query + id) as usize] + before;
        let cell = cell(actors, growth, actor, comptime!(CellKind::Query));
        let slot = id + query_offset(prior, actors, growth, prior_query, before, 0, cell + 1);
        output[(query + slot) as usize] = actor;
    }
}

pub(in crate::simulation) struct SpatialMigration<'a> {
    pub spatial: &'a CubeTensor,
    pub actors: &'a CubeTensor,
    pub previous: &'a ActorRows,
    pub current: &'a ActorRows,
    pub requests: &'a [PendingRecord],
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum SpatialPopulation {
    MarketActors,
    MarketFirms,
    QueryActors,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SpatialRecord {
    cell: u32,
    actor: u32,
}

impl SpatialRecord {
    const WORDS: usize = size_of::<Self>() / size_of::<u32>();
}

#[derive(CubeType, CubeLaunch)]
struct SpatialMigrationParameters {
    source: u32,
    target: u32,
    members: u32,
    border: u32,
    before: u32,
    cells: u32,
    side: u32,
    radius: u32,
    removed_start: u32,
    removed_count: u32,
    inserted_start: u32,
    inserted_count: u32,
}

pub(in crate::simulation) fn migrate(
    device: &Device,
    scenario: &Scenario,
    migration: SpatialMigration<'_>,
) -> CubeTensor {
    let count = migration.current.len();
    let before = migration.previous.len();
    let config = scenario.config();
    let side = config.map_size.div_ceil(config.search_radius);
    let cells = side * side;
    let words =
        3 * (cells as usize + 1 + count) + count.div_ceil(DISCOVERY_BLOCK as usize) + 1 + count;
    let output = device.empty(words);
    SpatialClearing::launch(
        device,
        device.count(words),
        CubeDim::new_1d(device.workgroup),
        SpatialClearInputLaunch {
            output: output.clone().into_buffer_arg(),
        },
    );
    let arrivals = migration
        .requests
        .iter()
        .filter(|record| record.request >> 24 == 3)
        .map(|record| record.actor as usize)
        .collect::<Vec<_>>();
    let records = |actors: &ActorRows, indices: &[usize], population| {
        let (radius, side) = match population {
            SpatialPopulation::MarketActors | SpatialPopulation::MarketFirms => {
                (config.search_radius, side)
            }
            SpatialPopulation::QueryActors => {
                let radius = config.map_size.div_ceil(128);
                (radius, config.map_size.div_ceil(radius))
            }
        };
        let mut records = indices
            .iter()
            .filter_map(|&index| {
                let actor = actors.at(index);
                let included = match population {
                    SpatialPopulation::MarketFirms => actor.kind > 0,
                    _ => true,
                };
                included.then_some(SpatialRecord {
                    cell: actor.position.y / radius * side + actor.position.x / radius,
                    actor: index as u32,
                })
            })
            .collect::<Vec<_>>();
        records.sort_unstable_by_key(|record| (record.cell, record.actor));
        records
    };
    let inserted = arrivals
        .iter()
        .copied()
        .chain(before..count)
        .collect::<Vec<_>>();
    for population in [
        SpatialPopulation::MarketActors,
        SpatialPopulation::MarketFirms,
        SpatialPopulation::QueryActors,
    ] {
        let mut removed = records(migration.previous, &arrivals, population);
        let inserted = records(migration.current, &inserted, population);
        let removed_count = removed.len();
        let inserted_count = inserted.len();
        removed.extend(inserted);
        let changes = device.tensor(bytemuck::cast_slice(&removed));
        let parameters = SpatialMigrationParametersLaunch {
            source: cells + 1 + before as u32,
            target: cells + 1 + count as u32,
            members: before as u32,
            border: cells + 1,
            before: before as u32,
            cells,
            side,
            radius: config.search_radius,
            removed_start: 0,
            removed_count: removed_count as u32,
            inserted_start: removed_count as u32,
            inserted_count: inserted_count as u32,
        };
        let parameters = match population {
            SpatialPopulation::MarketActors => parameters,
            SpatialPopulation::MarketFirms => SpatialMigrationParametersLaunch {
                source: parameters.source * 2,
                target: parameters.target * 2,
                members: migration.previous.firms() as u32,
                ..parameters
            },
            SpatialPopulation::QueryActors => {
                let radius = config.map_size.div_ceil(128);
                SpatialMigrationParametersLaunch {
                    source: parameters.source * 3
                        + before.div_ceil(DISCOVERY_BLOCK as usize) as u32
                        + 1,
                    target: parameters.target * 3
                        + count.div_ceil(DISCOVERY_BLOCK as usize) as u32
                        + 1,
                    border: 0,
                    radius,
                    side: config.map_size.div_ceil(radius),
                    ..parameters
                }
            }
        };
        SpatialMigrationSystem::launch(
            device,
            device.count(
                (parameters.members as usize)
                    .max(inserted_count)
                    .max(cells as usize + 1),
            ),
            CubeDim::new_1d(device.workgroup),
            SpatialMigrationInputLaunch {
                prior: migration.spatial.clone().into_buffer_arg(),
                actors: migration.actors.clone().into_buffer_arg(),
                changes: changes.into_buffer_arg(),
                output: output.clone().into_buffer_arg(),
                parameters,
                population,
            },
        );
    }
    output
}

#[derive(CubeType, CubeLaunch)]
struct SpatialMigrationInput {
    prior: Box<[u32]>,
    actors: Box<[u32]>,
    changes: Box<[u32]>,
    output: Box<[u32]>,
    parameters: SpatialMigrationParameters,
    #[cube(comptime)]
    population: SpatialPopulation,
}

struct SpatialMigrationSystem;
struct MigrateSpatialRows;

impl System for SpatialMigrationSystem {
    type Input = SpatialMigrationInput;
    type Passes = gpu_passes![MigrateSpatialRows];
}

#[cube]
impl GpuPass<SpatialMigrationInput> for MigrateSpatialRows {
    type Output = ();

    fn execute(mut input: SpatialMigrationInput) {
        migrate_rows(
            &input.prior,
            &input.actors,
            &input.changes,
            &mut input.output,
            &input.parameters,
            comptime!(input.population),
        );
    }
}

#[cube]
fn changed_offset(changes: &[u32], start: u32, count: u32, cell: u32, actor: u32) -> u32 {
    let mut low = 0u32;
    let mut high = count;
    while low < high {
        let middle = (low + high) / 2;
        let base = ((start + middle) * SpatialRecord::WORDS as u32) as usize;
        let other_cell = changes[base];
        let other_actor = changes[base + 1];
        let before = other_cell < cell || (other_cell == cell && other_actor < actor);
        low = select(before, middle + 1, low);
        high = select(before, high, middle);
    }
    low
}

#[cube]
fn prior_offset(
    prior: &[u32],
    actors: &[u32],
    parameters: &SpatialMigrationParameters,
    cell: u32,
    actor: u32,
) -> u32 {
    let mut low = 0u32;
    let mut high = parameters.members;
    while low < high {
        let middle = (low + high) / 2;
        let other = prior[(parameters.source + parameters.border + middle) as usize];
        let other_cell = previous_cell(actors, parameters, other);
        let before = other_cell < cell || (other_cell == cell && other < actor);
        low = select(before, middle + 1, low);
        high = select(before, high, middle);
    }
    low
}

#[cube]
fn previous_cell(actors: &[u32], parameters: &SpatialMigrationParameters, actor: u32) -> u32 {
    let x = actors[(A::X as u32 * parameters.before + actor) as usize];
    let y = actors[(A::Y as u32 * parameters.before + actor) as usize];
    y / parameters.radius * parameters.side + x / parameters.radius
}

#[cube]
fn migrate_rows(
    prior: &[u32],
    actors: &[u32],
    changes: &[u32],
    output: &mut [u32],
    parameters: &SpatialMigrationParameters,
    #[comptime] population: SpatialPopulation,
) {
    let id = ABSOLUTE_POS as u32;
    if comptime!(population != SpatialPopulation::QueryActors) && id <= parameters.cells {
        let removed = changed_offset(
            changes,
            parameters.removed_start,
            parameters.removed_count,
            id,
            0,
        );
        let inserted = changed_offset(
            changes,
            parameters.inserted_start,
            parameters.inserted_count,
            id,
            0,
        );
        output[(parameters.target + id) as usize] =
            prior[(parameters.source + id) as usize] - removed + inserted;
    }
    if id < parameters.members {
        let actor = prior[(parameters.source + parameters.border + id) as usize];
        let cell = previous_cell(actors, parameters, actor);
        let removed = changed_offset(
            changes,
            parameters.removed_start,
            parameters.removed_count,
            cell,
            actor,
        );
        let mut retained = true;
        if removed < parameters.removed_count {
            let base =
                ((parameters.removed_start + removed) * SpatialRecord::WORDS as u32) as usize;
            retained = changes[base + 1] != actor;
        }
        if retained {
            let inserted = changed_offset(
                changes,
                parameters.inserted_start,
                parameters.inserted_count,
                cell,
                actor,
            );
            output[(parameters.target + parameters.border + id - removed + inserted) as usize] =
                actor;
        }
    }
    if id < parameters.inserted_count {
        let base = ((parameters.inserted_start + id) * SpatialRecord::WORDS as u32) as usize;
        let cell = changes[base];
        let actor = changes[base + 1];
        let previous = prior_offset(prior, actors, parameters, cell, actor);
        let removed = changed_offset(
            changes,
            parameters.removed_start,
            parameters.removed_count,
            cell,
            actor,
        );
        output[(parameters.target + parameters.border + previous - removed + id) as usize] = actor;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::Experiment;
    use crate::simulation::layout::{ActorLayout, ActorMetadataRows};

    #[test]
    fn migrating_spatial_growth_matches_complete_indexes_on_gpu() -> Result<()> {
        let scenario = Scenario::industrial(257, 42, Experiment::Transport)?;
        let initial = scenario.initial()?;
        for workgroup in [32, 64, 128, 256] {
            let device = Device::headless(workgroup)?;
            let mut previous = initial.actors.clone();
            for generation in 0..4 {
                let mut current = previous.clone();
                let indices = match generation {
                    0 | 1 => vec![0, 31, 128, 256, previous.len() - 1],
                    2 => (0..previous.len()).collect(),
                    _ => Vec::new(),
                };
                for &index in &indices {
                    let mut actor = current.at(index);
                    actor.position = match index % 3 {
                        0 => Position { x: 0, y: 0 },
                        1 => Position { x: 999, y: 999 },
                        _ => Position { x: 500, y: 500 },
                    };
                    current.set(index, actor);
                }
                let additions = match generation {
                    0 | 2 => 0,
                    _ => 3,
                };
                current.extend((0..additions).map(|offset| {
                    let mut actor = previous.at(previous.len() - 1);
                    actor.identity += 10_000 + generation * 10 + offset;
                    actor.position = Position {
                        x: offset as u32 * 300,
                        y: generation as u32 * 200,
                    };
                    actor
                }));
                let requests = indices
                    .iter()
                    .map(|&actor| PendingRecord {
                        actor: actor as u32,
                        request: 3 << 24,
                        birth: 0,
                    })
                    .collect::<Vec<_>>();
                let rows = previous
                    .iter()
                    .map(|actor| ActorLayout::new(&actor, &scenario))
                    .collect::<ActorMetadataRows>();
                let actors = device.tensor(&rows.words());
                let spatial = device.tensor(&SpatialIndex::from_actors(&scenario, &previous).words);
                let actual = migrate(
                    &device,
                    &scenario,
                    SpatialMigration {
                        spatial: &spatial,
                        actors: &actors,
                        previous: &previous,
                        current: &current,
                        requests: &requests,
                    },
                );
                let expected = SpatialIndex::from_actors(&scenario, &current);
                let actual = device.read(&actual)?;
                for (word, (actual, expected)) in actual.iter().zip(&expected.words).enumerate() {
                    assert_eq!(
                        actual, expected,
                        "workgroup {workgroup}, generation {generation}, word {word}"
                    );
                }
                assert_eq!(actual.len(), expected.words.len());
                previous = current;
            }
        }
        Ok(())
    }

    #[test]
    fn incremental_spatial_growth_matches_complete_indexes_on_gpu() -> Result<()> {
        let scenario = Scenario::industrial(257, 42, Experiment::Transport)?;
        let initial = scenario.initial()?;
        let expected = SpatialIndex::new(&scenario, &initial);
        for workgroup in [32, 64, 128, 256] {
            let device = Device::headless(workgroup)?;
            for before in [1, 31, 128, initial.actors.len() - 1] {
                let prior = initial.actors.iter().take(before).collect::<ActorRows>();
                let additions = initial.actors.iter().skip(before).collect::<ActorRows>();
                let prior_rows = prior
                    .iter()
                    .map(|actor| ActorLayout::new(&actor, &scenario))
                    .collect::<ActorMetadataRows>();
                let added_rows = additions
                    .iter()
                    .map(|actor| ActorLayout::new(&actor, &scenario))
                    .collect::<ActorMetadataRows>();
                let prior_actors = device.tensor(&prior_rows.words());
                let actors = super::super::extend_actors(&device, &prior_actors, &added_rows);
                let complete_rows = initial
                    .actors
                    .iter()
                    .map(|actor| ActorLayout::new(&actor, &scenario))
                    .collect::<ActorMetadataRows>();
                assert_eq!(device.read(&actors)?, complete_rows.words());
                let prior_index =
                    device.tensor(&SpatialIndex::from_actors(&scenario, &prior).words);
                let actual = extend(&device, &scenario, &prior_index, &actors, &additions);
                assert_eq!(
                    device.read(&actual)?,
                    expected.words,
                    "workgroup {workgroup}, before {before}"
                );
            }
        }
        Ok(())
    }
}

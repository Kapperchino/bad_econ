use super::*;
use crate::simulation::storage::layout::{DISCOVERY_BLOCK, SpatialIndex};
use crate::simulation::storage::state::ActorTable;

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
    let query_cells = (growth.query_side * growth.query_side) as usize;
    let words = 3 * (cells + 1 + count)
        + count.div_ceil(DISCOVERY_BLOCK as usize)
        + 1
        + count
        + query_cells
        + 1;
    let added = device.tensor(&SpatialIndex::from_actors(scenario, additions).words);
    let output = device.empty(words);
    device.clear(&output);
    SpatialGrowthSystem::launch_elements(
        device,
        count.max(cells.max(query_cells) + 1),
        SpatialGrowthInputLaunch {
            prior: prior.buffer_arg(),
            additions: added.into_buffer_arg(),
            actors: actors.buffer_arg(),
            output: output.buffer_arg(),
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
fn cell(
    actors: &ActorTable<'_>,
    growth: &SpatialGrowth,
    actor: u32,
    #[comptime] kind: CellKind,
) -> u32 {
    let x = actors.x(actor);
    let y = actors.y(actor);
    match kind {
        CellKind::Market => y / growth.radius * growth.side + x / growth.radius,
        CellKind::Query => y / growth.query_tile * growth.query_side + x / growth.query_tile,
    }
}

#[derive(CubeType, CubeLaunch)]
struct SpatialGrowthInput {
    prior: Box<[u32]>,
    additions: Box<[u32]>,
    actors: Box<[u32]>,
    output: Box<[u32]>,
    growth: SpatialGrowth,
}

struct SpatialGrowthSystem;
struct MergeSpatialRows;

impl System for SpatialGrowthSystem {
    type Input = SpatialGrowthInput;
    type Passes = gpu_passes![MergeSpatialRows];
}

#[cube]
impl GpuPass<SpatialGrowthInput> for MergeSpatialRows {
    type Output = ();

    fn execute(mut input: SpatialGrowthInput) {
        merge(
            &input.prior,
            &input.additions,
            &ActorTable::with_stride(
                &input.actors,
                (input.growth.before + input.growth.added) as usize,
            ),
            &mut input.output,
            &input.growth,
        );
    }
}

#[cube]
fn merge(
    prior: &[u32],
    additions: &[u32],
    actors: &ActorTable<'_>,
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
    if id <= growth.query_side * growth.query_side {
        output[(query + count + id) as usize] = prior[(prior_query + before + id) as usize]
            + additions[(added_query + added + id) as usize];
    }
    if id < before {
        let actor = prior[(prior_query + id) as usize];
        let cell = cell(actors, growth, actor, comptime!(CellKind::Query));
        let slot = id + additions[(added_query + added + cell) as usize];
        output[(query + slot) as usize] = actor;
    }
    if id < added {
        let actor = additions[(added_query + id) as usize] + before;
        let cell = cell(actors, growth, actor, comptime!(CellKind::Query));
        let slot = id + prior[(prior_query + before + cell + 1) as usize];
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
    let tile = config.map_size.div_ceil(128);
    let query_side = config.map_size.div_ceil(tile);
    let words = 3 * (cells as usize + 1 + count)
        + count.div_ceil(DISCOVERY_BLOCK as usize)
        + 1
        + count
        + (query_side * query_side + 1) as usize;
    let output = device.empty(words);
    device.clear(&output);
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
                    cells: query_side * query_side,
                    side: query_side,
                    ..parameters
                }
            }
        };
        SpatialMigrationSystem::launch_elements(
            device,
            (parameters.members as usize)
                .max(inserted_count)
                .max(parameters.cells as usize + 1),
            SpatialMigrationInputLaunch {
                prior: migration.spatial.buffer_arg(),
                actors: migration.actors.buffer_arg(),
                changes: changes.into_buffer_arg(),
                output: output.buffer_arg(),
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
            &ActorTable::with_stride(&input.actors, input.parameters.before as usize),
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
    actors: &ActorTable<'_>,
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
fn previous_cell(
    actors: &ActorTable<'_>,
    parameters: &SpatialMigrationParameters,
    actor: u32,
) -> u32 {
    let x = actors.x(actor);
    let y = actors.y(actor);
    y / parameters.radius * parameters.side + x / parameters.radius
}

#[cube]
fn migrate_rows(
    prior: &[u32],
    actors: &ActorTable<'_>,
    changes: &[u32],
    output: &mut [u32],
    parameters: &SpatialMigrationParameters,
    #[comptime] population: SpatialPopulation,
) {
    let id = ABSOLUTE_POS as u32;
    if id <= parameters.cells {
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
        let mut source = parameters.source;
        let mut target = parameters.target;
        if comptime!(population == SpatialPopulation::QueryActors) {
            source += parameters.members;
            target += parameters.members - parameters.removed_count + parameters.inserted_count;
        }
        output[(target + id) as usize] = prior[(source + id) as usize] - removed + inserted;
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
    use crate::simulation::storage::layout::{ActorLayout, ActorMetadataRows};

    #[test]
    fn migrating_spatial_growth_matches_complete_indexes_on_gpu() -> Result<()> {
        let scenario = Scenario::industrial(257, 42, Experiment::Transport)?;
        let initial = scenario.initial()?;
        let workgroup = 64;
        let device = crate::simulation::tests::device(workgroup)?;
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
        Ok(())
    }

    #[test]
    fn incremental_spatial_growth_matches_complete_indexes_on_gpu() -> Result<()> {
        let scenario = Scenario::industrial(257, 42, Experiment::Transport)?;
        let initial = scenario.initial()?;
        let expected = SpatialIndex::new(&scenario, &initial);
        let workgroup = 64;
        let device = crate::simulation::tests::device(workgroup)?;
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
            let prior_index = device.tensor(&SpatialIndex::from_actors(&scenario, &prior).words);
            let actual = extend(&device, &scenario, &prior_index, &actors, &additions);
            assert_eq!(
                device.read(&actual)?,
                expected.words,
                "workgroup {workgroup}, before {before}"
            );
        }
        Ok(())
    }
}

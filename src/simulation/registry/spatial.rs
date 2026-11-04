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
    clear::launch(
        &device.client,
        device.count(words),
        CubeDim::new_1d(device.workgroup),
        output.clone().into_buffer_arg(),
    );
    merge::launch(
        &device.client,
        device.count(count.max(cells + 1)),
        CubeDim::new_1d(device.workgroup),
        prior.clone().into_buffer_arg(),
        added.into_buffer_arg(),
        actors.clone().into_buffer_arg(),
        output.clone().into_buffer_arg(),
        SpatialGrowthLaunch {
            before: growth.before,
            added: growth.added,
            side: growth.side,
            radius: growth.radius,
            query_side: growth.query_side,
            query_tile: growth.query_tile,
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

#[cube(launch)]
fn clear(output: &mut [u32]) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = 0;
    }
}

#[cube(launch)]
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::Experiment;
    use crate::simulation::layout::{ActorLayout, ActorMetadataRows};

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

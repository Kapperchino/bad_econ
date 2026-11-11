use super::*;
use crate::domain::{Experiment, Position, Scenario};
use crate::simulation::encode;
use anyhow::Result;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum Selection {
    Serial,
    Bulk,
}

#[derive(CubeType, CubeLaunch)]
struct ConsolidationInput {
    state: Box<[u32]>,
    actors: Box<[u32]>,
    spatial: Box<[u32]>,
    config: EconomyConfig,
    control: Box<[Atomic<u32>]>,
    farm: u32,
    required: u32,
    #[cube(comptime)]
    selection: Selection,
}

struct ConsolidationCheck;
struct SelectPlots;

impl System for ConsolidationCheck {
    type Input = ConsolidationInput;
    type Passes = gpu_passes![SelectPlots];
}

#[cube]
impl GpuPass<ConsolidationInput> for SelectPlots {
    type Output = ();

    fn execute(mut input: ConsolidationInput) {
        match comptime!(input.selection) {
            Selection::Serial => consolidate_serial(
                &mut input.state,
                &input.actors,
                &input.config,
                &input.control,
                input.farm,
                input.required,
            ),
            Selection::Bulk => consolidate_farm(
                &mut input.state,
                &input.actors,
                &input.spatial,
                &input.config,
                &input.control,
                input.farm,
                input.required,
            ),
        }
    }
}

#[test]
fn bulk_consolidation_preserves_nearest_whole_grants_and_ties_on_gpu() -> Result<()> {
    let mut config = Scenario::industrial(513, 42, Experiment::Baseline)?
        .config()
        .clone();
    config.search_radius = 50;
    let scenario = Scenario::new(config)?;
    let mut initial = scenario.initial()?;
    let farms = initial
        .actors
        .iter()
        .enumerate()
        .filter(|(_, actor)| actor.kind == Good::Grain as u32)
        .map(|(index, _)| index)
        .take(2)
        .collect::<Vec<_>>();
    assert_eq!(farms.len(), 2);
    for household in 0..513 {
        let mut actor = initial.actors.at(household);
        actor.position = match household {
            400.. => Position { x: 900, y: 900 },
            _ => Position {
                x: 96 + household as u32 % 9,
                y: 97 + household as u32 % 7,
            },
        };
        initial.actors.set(household, actor);
        initial.set(Field::LandPlots, household, household as u64 % 5);
        initial.set(Field::CropWork, household, 100);
    }
    for &farm in &farms {
        let mut actor = initial.actors.at(farm);
        actor.position = Position { x: 100, y: 100 };
        initial.actors.set(farm, actor);
    }
    let workgroup = 64;
    let device = crate::simulation::tests::device(workgroup)?;
    let config = Configuration::new(&device, &scenario, &initial);
    let spatial = device.tensor(&SpatialIndex::new(&scenario, &initial).words);
    let actors = device.tensor(
        &initial
            .actors
            .iter()
            .map(|actor| ActorLayout::new(&actor, &scenario))
            .collect::<crate::simulation::layout::ActorMetadataRows>()
            .words(),
    );
    let spatial = parcel_index(&device, &actors, &spatial, &config);
    for required in [0, 1, 3, 127, 255, 257, 1024, u32::MAX] {
        let mut results = Vec::new();
        for selection in [Selection::Serial, Selection::Bulk] {
            let state = device.tensor(bytemuck::cast_slice(&encode(&initial.values)));
            let control = device.tensor(&[0; CONTROL_WORDS]);
            for &farm in &farms {
                ConsolidationCheck::launch(
                    &device,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new_1d(workgroup),
                    ConsolidationInputLaunch {
                        state: state.clone().into_buffer_arg(),
                        actors: actors.clone().into_buffer_arg(),
                        spatial: spatial.clone().into_buffer_arg(),
                        config: config.launch(),
                        control: control.clone().into_buffer_arg(),
                        farm: farm as u32,
                        required,
                        selection,
                    },
                );
            }
            results.push(device.read(&state)?);
            assert_eq!(device.read(&control)?, [0; CONTROL_WORDS]);
        }
        assert_eq!(results[0], results[1], "{workgroup} {required}");
    }
    Ok(())
}
#[cube]
fn consolidate_serial(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
    farm: u32,
    required: u32,
) {
    let mut remaining = required;
    while remaining > 0 {
        let selected = nearest_farmer(state, actors, config, farm);
        if selected < config.actors {
            let plots = value(state, config, comptime!(Field::LandPlots as u32), selected);
            sync_storage();
            if UNIT_POS == 0 {
                increase(
                    state,
                    config,
                    control,
                    comptime!(Field::OccupiedPlots as u32),
                    farm,
                    plots,
                );
                increase(
                    state,
                    config,
                    control,
                    comptime!(Field::DisplacedPlots as u32),
                    selected,
                    plots,
                );
                set32(
                    state,
                    config,
                    comptime!(Field::LandPlots as u32),
                    selected,
                    0,
                );
                set32(
                    state,
                    config,
                    comptime!(Field::CropWork as u32),
                    selected,
                    0,
                );
                discard_grain(state, config, control, selected);
            }
            remaining = max(remaining, plots.extract(LOW_WORD)) - plots.extract(LOW_WORD);
        } else {
            remaining = 0;
        }
        sync_storage();
    }
}

#[cube]
fn nearest_farmer(state: &mut [u32], actors: &[u32], config: &EconomyConfig, farm: u32) -> u32 {
    let mut nearest = u64::new(comptime!(u64::MAX as i64));
    let mut household = UNIT_POS;
    while household < config.actors {
        let plots = number(state, config, comptime!(Field::LandPlots as u32), household);
        let trip = distance(actors, farm, household);
        let eligible = kind(actors, household).is(comptime!(ActorKind::Household))
            && plots > 0
            && trip <= config.search_radius;
        let rank = (u64::cast_from(trip) << 32) | u64::cast_from(household);
        nearest = min(nearest, select(eligible, rank, u64::MAX));
        household += CUBE_DIM;
    }
    let mut ranks = Shared::<[u64]>::new_slice(256usize);
    ranks[UNIT_POS as usize] = nearest;
    sync_cube();
    let mut stride = CUBE_DIM / 2;
    while stride > 0 {
        if UNIT_POS < stride {
            ranks[UNIT_POS as usize] = min(
                ranks[UNIT_POS as usize],
                ranks[(UNIT_POS + stride) as usize],
            );
        }
        sync_cube();
        stride /= 2;
    }
    u32::cast_from(ranks[0])
}

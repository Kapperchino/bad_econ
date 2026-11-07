use super::records::PendingRecord;
use crate::domain::{ActorRows, Scenario};
use crate::simulation::{
    Device,
    layout::ActorLayout,
    pass::GpuPass,
    system::{System, gpu_passes},
    tensor,
};
use burn_wgpu::CubeTensor;
use bytemuck::{Pod, Zeroable};
use cubecl::prelude::*;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ActorPatch {
    index: u32,
    metadata: ActorLayout,
}

impl ActorPatch {
    const WORDS: usize = size_of::<Self>() / size_of::<u32>();
}

pub(in crate::simulation) fn patch(
    device: &Device,
    scenario: &Scenario,
    source: &CubeTensor,
    actors: &ActorRows,
    requests: &[PendingRecord],
) -> CubeTensor {
    patch_actors(
        device,
        scenario,
        source,
        actors,
        requests
            .iter()
            .filter(|record| matches!(record.request >> 24, 3 | 6))
            .map(|record| record.actor as usize),
    )
}

pub(in crate::simulation) fn patch_actors(
    device: &Device,
    scenario: &Scenario,
    source: &CubeTensor,
    actors: &ActorRows,
    indices: impl Iterator<Item = usize>,
) -> CubeTensor {
    let patches = indices
        .map(|index| ActorPatch {
            index: index as u32,
            metadata: ActorLayout::new(&actors.at(index), scenario),
        })
        .collect::<Vec<_>>();
    let output = device.empty(source.meta.num_elements());
    tensor::copy(device, source, &output);
    let records = device.tensor(bytemuck::cast_slice(&patches));
    MetadataPatching::launch(
        device,
        device.count(patches.len()),
        CubeDim::new_1d(device.workgroup),
        PatchInputLaunch {
            patches: records.into_buffer_arg(),
            actors: output.clone().into_buffer_arg(),
        },
    );
    output
}

#[derive(CubeType, CubeLaunch)]
struct PatchInput {
    patches: Box<[u32]>,
    actors: Box<[u32]>,
}

struct MetadataPatching;
struct ApplyPatches;

impl System for MetadataPatching {
    type Input = PatchInput;
    type Passes = gpu_passes![ApplyPatches];
}

#[cube]
impl GpuPass<PatchInput> for ApplyPatches {
    type Output = ();

    fn execute(mut input: PatchInput) {
        let patch = ABSOLUTE_POS;
        if patch < input.patches.len() / ActorPatch::WORDS {
            let base = patch * ActorPatch::WORDS;
            let actor = input.patches[base] as usize;
            let count = input.actors.len() / ActorLayout::WORDS;
            #[unroll]
            for field in 0..ActorLayout::WORDS {
                input.actors[field * count + actor] = input.patches[base + 1 + field];
            }
        }
    }
}

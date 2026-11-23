use super::*;
use crate::simulation::{decode, storage::layout::TICK_WORDS};
use anyhow::anyhow;

pub(super) enum Staging {
    Entry,
    Settlement { fields: usize },
}

pub(super) struct StagedRows {
    pub(super) indices: CubeTensor,
    pub(super) stride: usize,
    pub(super) elements: usize,
}

impl StagedRows {
    pub(super) fn new(
        device: &Device,
        state: &CubeTensor,
        source_fields: usize,
        selected: &[usize],
        staging: Staging,
    ) -> Result<Self> {
        let fields = match staging {
            Staging::Entry => crate::domain::BASE_FIELDS,
            Staging::Settlement { fields } => fields,
        };
        let elements = selected.len() * fields;
        let required = elements as u64 * 16;
        let reserved = device.reclaim_memory(required)?.bytes_reserved;
        match reserved.saturating_add(required) <= device.memory_budget.bytes() {
            true => Ok(Self {
                indices: device.tensor(
                    &selected
                        .iter()
                        .map(|&actor| actor as u32)
                        .collect::<Vec<_>>(),
                ),
                stride: state.meta.num_elements() / (source_fields * TICK_WORDS),
                elements,
            }),
            false => Err(match staging {
                Staging::Entry => anyhow!("entry staging exceeds the device memory budget"),
                Staging::Settlement { .. } => {
                    anyhow!("settlement staging exceeds the device memory budget")
                }
            }),
        }
    }

    pub(super) fn readback_bytes(&self) -> u64 {
        self.elements as u64 * size_of::<u64>() as u64
    }

    pub(super) fn read(&self, device: &Device, state: &CubeTensor) -> Result<Vec<u64>> {
        let values = device.empty(self.elements * TICK_WORDS);
        transfer(
            device,
            state,
            &values,
            &self.indices,
            self.stride,
            Transfer::Gather,
        );
        Ok(decode(&device.read(&values)?))
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(super) enum Transfer {
    Gather,
    Scatter,
}

#[derive(CubeType, CubeLaunch)]
struct TransferInput {
    source: Box<[Ticks]>,
    target: Box<[Ticks]>,
    indices: Box<[u32]>,
    actors: u32,
    #[cube(comptime)]
    stage: Transfer,
}

struct RegistryTransfers;
struct TransferRows;

impl System for RegistryTransfers {
    type Input = TransferInput;
    type Passes = gpu_passes![TransferRows];
}

#[cube]
impl GpuPass<TransferInput> for TransferRows {
    type Output = ();

    fn execute(mut input: TransferInput) {
        transfer_rows(
            &input.source,
            &mut input.target,
            &input.indices,
            input.actors,
            comptime!(input.stage),
        );
    }
}

#[cube]
fn transfer_rows(
    source: &[Ticks],
    target: &mut [Ticks],
    indices: &[u32],
    actors: u32,
    #[comptime] stage: Transfer,
) {
    let id = ABSOLUTE_POS;
    match stage {
        Transfer::Gather => {
            if id < target.len() {
                let field = id / indices.len();
                let actor = indices[id % indices.len()] as usize;
                target[id] = source[field * actors as usize + actor];
            }
        }
        Transfer::Scatter => {
            if id < source.len() {
                let field = id / indices.len();
                let actor = indices[id % indices.len()] as usize;
                target[field * actors as usize + actor] = source[id];
            }
        }
    }
}

pub(super) fn transfer(
    device: &Device,
    source: &CubeTensor,
    target: &CubeTensor,
    indices: &CubeTensor,
    actors: usize,
    stage: Transfer,
) {
    let count = match stage {
        Transfer::Scatter => source.meta.shape()[0],
        _ => target.meta.shape()[0],
    };
    RegistryTransfers::launch_elements(
        device,
        count / TICK_WORDS,
        TransferInputLaunch {
            source: source.buffer_arg(),
            target: target.buffer_arg(),
            indices: indices.buffer_arg(),
            actors: actors as u32,
            stage,
        },
    );
}

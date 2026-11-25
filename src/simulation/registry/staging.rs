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

pub(super) struct EntryValues {
    pub values: Vec<u64>,
    pub readback_bytes: u64,
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

    pub(super) fn read_entry(&self, device: &Device, state: &CubeTensor) -> Result<EntryValues> {
        match (&device.cube, self.elements) {
            (cubecl::Device::Metal(_), 16_384..=4_194_304) => self.read_packed(device, state),
            _ => Ok(EntryValues {
                values: self.read(device, state)?,
                readback_bytes: self.readback_bytes(),
            }),
        }
    }

    fn read_packed(&self, device: &Device, state: &CubeTensor) -> Result<EntryValues> {
        let pairs = self.elements.div_ceil(2);
        let wide_capacity = self.elements.div_ceil(32);
        let packed = device.empty(pairs + wide_capacity * 3);
        let written = device.empty(1);
        device.clear(&written);
        device.clear(&packed);
        gather_entry_pairs::launch(
            &device.client,
            device.count(pairs),
            CubeDim::new_1d(device.workgroup),
            state.buffer_arg(),
            self.indices.buffer_arg(),
            packed.buffer_arg(),
            written.buffer_arg(),
            self.stride as u32,
            self.elements as u32,
        );
        let [count, words] =
            device.read_prefixes([(&written, 1), (&packed, packed.meta.num_elements())])?;
        let bytes = ((1 + words.len()) * size_of::<u32>()) as u64;
        match count[0] as usize {
            count if count <= wide_capacity => {
                let mut values = words[..pairs]
                    .iter()
                    .flat_map(|&word| [u64::from(word & 0xffff), u64::from(word >> 16)])
                    .take(self.elements)
                    .collect::<Vec<_>>();
                for entry in words[pairs..pairs + count * 3].chunks_exact(3) {
                    values[entry[0] as usize] = u64::from(entry[1]) | (u64::from(entry[2]) << 32);
                }
                Ok(EntryValues {
                    values,
                    readback_bytes: bytes,
                })
            }
            _ => Ok(EntryValues {
                values: self.read(device, state)?,
                readback_bytes: bytes + self.readback_bytes(),
            }),
        }
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

#[cube(launch)]
fn gather_entry_pairs(
    state: &[Ticks],
    indices: &[u32],
    packed: &mut [u32],
    written: &[Atomic<u32>],
    stride: u32,
    elements: u32,
) {
    let pair = ABSOLUTE_POS as u32;
    let pairs = ceildiv(elements, 2);
    if pair < pairs {
        let rows = indices.len() as u32;
        let mut narrow = 0u32;
        #[unroll]
        for part in 0..2u32 {
            let id = 2 * pair + part;
            if id < elements {
                let value = state[(id / rows * stride + indices[(id % rows) as usize]) as usize];
                let low = value.extract(0usize);
                let high = value.extract(1usize);
                narrow |= (low & 0xffff) << (part * 16);
                if low > 0xffff || high != 0 {
                    let slot = written[0usize].fetch_add(1);
                    let offset = pairs + slot * 3;
                    if slot < ceildiv(elements, 32) {
                        packed[offset as usize] = id;
                        packed[(offset + 1) as usize] = low;
                        packed[(offset + 2) as usize] = high;
                    }
                }
            }
        }
        packed[pair as usize] = narrow;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_entry_read_preserves_sparse_and_dense_wide_values_on_gpu() -> Result<()> {
        let device = crate::simulation::tests::device(64)?;
        let selected = [5u32, 2, 0];
        let rows = StagedRows {
            indices: device.tensor(&selected),
            stride: 7,
            elements: 33,
        };
        let mut mixed = (0..77).map(|i| 65_535 - i).collect::<Vec<_>>();
        mixed[5] = u64::MAX;
        mixed[70] = 65_536;
        for values in [mixed, vec![1u64 << 48; 77]] {
            let state = device.tensor(bytemuck::cast_slice(&crate::simulation::encode(&values)));
            let expected = values
                .chunks_exact(7)
                .flat_map(|row| selected.map(|id| row[id as usize]))
                .collect::<Vec<_>>();
            let packed = rows.read_packed(&device, &state)?;
            assert_eq!(packed.values, expected);
            let first_read = (1 + 17 + 2 * 3) * size_of::<u32>() as u64;
            let dense = expected.iter().filter(|&&v| v > u16::MAX as u64).count() > 2;
            assert_eq!(
                packed.readback_bytes,
                first_read + u64::from(dense) * rows.readback_bytes()
            );
        }
        Ok(())
    }
}

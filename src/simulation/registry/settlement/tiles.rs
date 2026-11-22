use super::super::{AsBufferArg, Device, Ticks, ticks, zero};
use crate::simulation::{decode, encode};
use anyhow::Result;
use burn_wgpu::CubeTensor;
use cubecl::prelude::*;

const WIDTH: usize = 32;
const RECORD: usize = WIDTH + 1;

pub(super) struct GatheredTiles {
    records: Vec<u64>,
    slots: Vec<u32>,
}

impl GatheredTiles {
    pub fn supported(device: &Device, elements: usize) -> bool {
        let properties = device.client.properties();
        elements >= 131_072
            && super::super::super::parallel::ScanMode::new(device)
                == super::super::super::parallel::ScanMode::Plane
            && properties.hardware.plane_size_min == WIDTH as u32
            && properties.hardware.plane_size_max == WIDTH as u32
    }

    pub fn read(
        device: &Device,
        state: &CubeTensor,
        indices: &CubeTensor,
        stride: usize,
        elements: usize,
    ) -> Result<Self> {
        let tiles = elements.div_ceil(WIDTH);
        let written = device.tensor(&[0]);
        let packed = device.empty(tiles * RECORD * 2);
        let dim = CubeDim::new_1d(WIDTH as u32);
        gather_tiles::launch(
            &device.client,
            cubecl::calculate_cube_count_elemwise(&device.client, elements, dim),
            dim,
            state.buffer_arg(),
            indices.buffer_arg(),
            packed.buffer_arg(),
            written.buffer_arg(),
            stride as u32,
            elements as u32,
        );
        let count = device.read(&written)?[0] as usize;
        let records = decode(&device.read_range(&packed, 0, count * RECORD * 2)?);
        let mut slots = vec![u32::MAX; tiles];
        for (slot, record) in records.chunks_exact(RECORD).enumerate() {
            slots[record[0] as usize] = slot as u32;
        }
        Ok(Self { records, slots })
    }

    pub fn readback_bytes(&self) -> u64 {
        4 + self.records.len() as u64 * 8
    }

    pub fn unpack(&self, elements: usize) -> Vec<u64> {
        let mut values = vec![0; elements];
        for record in self.records.chunks_exact(RECORD) {
            let start = record[0] as usize * WIDTH;
            let len = WIDTH.min(elements - start);
            values[start..start + len].copy_from_slice(&record[1..1 + len]);
        }
        values
    }

    pub fn changes(&self, device: &Device, values: &[u64]) -> TileChanges {
        let mut records = Vec::new();
        for (tile, chunk) in values.chunks(WIDTH).enumerate() {
            let before = match self.slots[tile] {
                u32::MAX => &[0; WIDTH][..chunk.len()],
                slot => {
                    let start = slot as usize * RECORD + 1;
                    &self.records[start..start + chunk.len()]
                }
            };
            if chunk != before {
                records.push(tile as u64);
                records.extend_from_slice(chunk);
                records.resize(records.len() + WIDTH - chunk.len(), 0);
            }
        }
        let tiles = records.len() / RECORD;
        TileChanges {
            records: (tiles > 0).then(|| device.tensor(bytemuck::cast_slice(&encode(&records)))),
            tiles,
            elements: values.len(),
        }
    }
}

pub(super) struct TileChanges {
    records: Option<CubeTensor>,
    tiles: usize,
    elements: usize,
}

impl TileChanges {
    pub fn apply(&self, device: &Device, state: &CubeTensor, indices: &CubeTensor, stride: usize) {
        if let Some(records) = &self.records {
            let dim = CubeDim::new_1d(WIDTH as u32);
            scatter_tiles::launch(
                &device.client,
                cubecl::calculate_cube_count_elemwise(&device.client, self.tiles * WIDTH, dim),
                dim,
                records.buffer_arg(),
                state.buffer_arg(),
                indices.buffer_arg(),
                stride as u32,
                self.elements as u32,
            );
        }
    }
}

#[cube(launch)]
fn gather_tiles(
    state: &[Ticks],
    indices: &[u32],
    packed: &mut [Ticks],
    written: &[Atomic<u32>],
    stride: u32,
    elements: u32,
) {
    let id = ABSOLUTE_POS as u32;
    let mut value = zero();
    if id < elements {
        let rows = indices.len() as u32;
        value = state[(id / rows * stride + indices[(id % rows) as usize]) as usize];
    }
    if plane_any((value.extract(0usize) | value.extract(1usize)) != 0) {
        let mut slot = 0u32;
        if UNIT_POS == 0 {
            slot = written[0usize].fetch_add(1);
            packed[slot as usize * RECORD] = ticks(CUBE_POS as u32, 0);
        }
        let offset = plane_broadcast(slot, 0u32) as usize * RECORD;
        packed[offset + 1 + UNIT_POS as usize] = value;
    }
}

#[cube(launch)]
fn scatter_tiles(
    packed: &[Ticks],
    state: &mut [Ticks],
    indices: &[u32],
    stride: u32,
    elements: u32,
) {
    let offset = CUBE_POS * RECORD;
    if offset < packed.len() {
        let id = packed[offset].extract(0usize) * WIDTH as u32 + UNIT_POS;
        if id < elements {
            let rows = indices.len() as u32;
            state[(id / rows * stride + indices[(id % rows) as usize]) as usize] =
                packed[offset + 1 + UNIT_POS as usize];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_settlement_tiles_preserve_field_boundaries_and_zeroed_rows_on_gpu() -> Result<()> {
        let device = crate::simulation::tests::device(64)?;
        let stride = 80;
        let fields = 3;
        let rows = (1..stride).step_by(2).collect::<Vec<_>>();
        let indices = device.tensor(&rows.iter().map(|&row| row as u32).collect::<Vec<_>>());
        let positions = (0..fields)
            .flat_map(|field| rows.iter().map(move |row| field * stride + row))
            .collect::<Vec<_>>();
        let mut expected = vec![u64::MAX; stride * fields];
        for &position in &positions {
            expected[position] = 0;
        }
        expected[positions[33]] = 1u64 << 40;
        expected[*positions.last().unwrap()] = 9;
        let state = device.tensor(bytemuck::cast_slice(&encode(&expected)));
        let tiles = GatheredTiles::read(&device, &state, &indices, stride, positions.len())?;
        let mut selected = tiles.unpack(positions.len());
        assert_eq!(
            selected,
            positions.iter().map(|&i| expected[i]).collect::<Vec<_>>()
        );
        assert_eq!(tiles.readback_bytes(), 4 + 2 * RECORD as u64 * 8);
        selected[33] = 0;
        selected[5] = u64::MAX;
        let changes = tiles.changes(&device, &selected);
        assert_eq!(changes.tiles, 2);
        changes.apply(&device, &state, &indices, stride);
        for (&position, &value) in positions.iter().zip(&selected) {
            expected[position] = value;
        }
        assert_eq!(decode(&device.read(&state)?), expected);
        Ok(())
    }
}

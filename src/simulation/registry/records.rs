use super::*;
use crate::simulation::parallel;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(in crate::simulation) struct PendingRecord {
    pub actor: u32,
    pub request: u32,
    pub birth: u32,
}

pub(in crate::simulation) struct PendingRecords {
    pub records: Vec<PendingRecord>,
    pub readback_bytes: u64,
}

impl PendingRecords {
    pub fn read(
        device: &Device,
        requests: &CubeTensor,
        births: &CubeTensor,
        scratch: &CubeTensor,
        output: &CubeTensor,
    ) -> Result<Self> {
        let actors = requests.meta.num_elements();
        let argument = |tensor: &CubeTensor| tensor.clone().into_buffer_arg();
        count_records::launch(
            &device.client,
            device.count(actors),
            CubeDim::new_1d(device.workgroup),
            argument(requests),
            argument(births),
            argument(scratch),
        );
        parallel::scan(
            device,
            scratch,
            actors,
            actors.div_ceil(device.workgroup as usize),
        );
        write_records::launch(
            &device.client,
            device.count(actors),
            CubeDim::new_1d(device.workgroup),
            argument(requests),
            argument(births),
            argument(scratch),
            argument(output),
        );
        let count = device.read_range(output, 0, 1)?[0] as usize;
        let words = device.read_range(output, 1, count * 3)?;
        let records = words
            .chunks_exact(3)
            .map(|row| PendingRecord {
                actor: row[0],
                request: row[1],
                birth: row[2],
            })
            .collect();
        Ok(Self {
            records,
            readback_bytes: ((1 + count * 3) * size_of::<u32>()) as u64,
        })
    }
}

#[cube(launch)]
fn count_records(requests: &[u32], births: &[u32], scratch: &mut [u32]) {
    let actor = ABSOLUTE_POS;
    let count = requests.len();
    let mut present = 0u32;
    if actor < count {
        present = u32::cast_from(requests[actor] > 0 || births[actor] > 0);
    }
    let prefix = parallel::prefix_sum(present);
    if actor < count {
        scratch[actor] = prefix.inclusive;
    }
    if UNIT_POS == 0 {
        scratch[count + CUBE_POS] = prefix.total;
    }
}

#[cube(launch)]
fn write_records(requests: &[u32], births: &[u32], scratch: &mut [u32], output: &mut [u32]) {
    let actor = ABSOLUTE_POS;
    let count = requests.len();
    if actor == 0 {
        output[0usize] = scratch[count + ceildiv(count as u32, CUBE_DIM) as usize - 1];
    }
    if actor < count && (requests[actor] > 0 || births[actor] > 0) {
        let mut slot = scratch[actor] - 1;
        if CUBE_POS > 0 {
            slot += scratch[count + CUBE_POS - 1];
        }
        let base = slot as usize * 3 + 1;
        output[base] = actor as u32;
        output[base + 1] = requests[actor];
        output[base + 2] = births[actor];
    }
}

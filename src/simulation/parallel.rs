use super::{Device, words::ceildiv};
use burn_wgpu::CubeTensor;
use cubecl::prelude::*;

#[derive(CubeType)]
pub(super) struct PrefixSum {
    pub inclusive: u32,
    pub total: u32,
}

#[cube]
pub(super) fn prefix_sum(value: u32) -> PrefixSum {
    let mut values = Shared::<[u32]>::new_slice(256usize);
    let lane = UNIT_POS as usize;
    values[lane] = value;
    sync_cube();
    let mut stride = 1u32;
    while stride < CUBE_DIM {
        let mut previous = 0u32;
        if UNIT_POS >= stride {
            previous = values[lane - stride as usize];
        }
        sync_cube();
        values[lane] += previous;
        sync_cube();
        stride *= 2;
    }
    PrefixSum {
        inclusive: values[lane],
        total: values[CUBE_DIM as usize - 1],
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Reduction {
    Sum,
    Minimum,
}

#[cube]
pub(super) fn reduce(value: u32, #[comptime] operation: Reduction) -> u32 {
    let mut values = Shared::<[u32]>::new_slice(256usize);
    let lane = UNIT_POS as usize;
    values[lane] = value;
    sync_cube();
    let mut stride = CUBE_DIM / 2;
    while stride > 0 {
        if UNIT_POS < stride {
            values[lane] = match comptime!(operation) {
                Reduction::Sum => values[lane] + values[lane + stride as usize],
                Reduction::Minimum => min(values[lane], values[lane + stride as usize]),
            };
        }
        sync_cube();
        stride /= 2;
    }
    values[0]
}

struct ScanLevel {
    offset: usize,
    count: usize,
    blocks: usize,
    divisor: u32,
}

pub(super) fn scan(device: &Device, buffer: &CubeTensor, offset: usize, count: usize) {
    let mut levels = Vec::new();
    let mut offset = offset;
    let mut count = count;
    while count > 1 {
        let blocks = count.div_ceil(device.workgroup as usize);
        scan_blocks::launch(
            &device.client,
            device.count(count),
            CubeDim::new_1d(device.workgroup),
            buffer.clone().into_buffer_arg(),
            offset as u32,
            count as u32,
            ScanMode::new(device),
        );
        levels.push(ScanLevel {
            offset,
            count,
            blocks,
            divisor: 1,
        });
        offset += count;
        count = blocks;
    }
    for level in levels.into_iter().rev().filter(|level| level.blocks > 1) {
        add_block_offsets::launch(
            &device.client,
            device.count(level.count),
            CubeDim::new_1d(device.workgroup),
            buffer.clone().into_buffer_arg(),
            level.offset as u32,
            level.count as u32,
        );
    }
}

#[cube(launch)]
fn scan_blocks(buffer: &mut [u32], offset: u32, count: u32, #[comptime] mode: ScanMode) {
    let id = ABSOLUTE_POS as u32;
    let block = CUBE_POS as u32;
    let mut value = 0u32;
    if id < count {
        value = buffer[(offset + id) as usize];
    }
    let prefix = scan_prefix(value, mode);
    if id < count {
        buffer[(offset + id) as usize] = prefix.inclusive;
    }
    if UNIT_POS == 0 && block < ceildiv(count, CUBE_DIM) {
        buffer[(offset + count + block) as usize] = prefix.total;
    }
}

#[cube(launch)]
fn add_block_offsets(buffer: &mut [u32], offset: u32, count: u32) {
    let id = ABSOLUTE_POS as u32;
    let block = CUBE_POS as u32;
    if id < count && block > 0 {
        buffer[(offset + id) as usize] += buffer[(offset + count + block - 1) as usize];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;

    #[test]
    fn hierarchical_prefix_sums_preserve_partial_blocks_on_gpu() -> Result<()> {
        for workgroup in [32, 64, 128, 256] {
            let device = Device::headless(workgroup)?;
            for count in [0, 1, 31, 32, 33, 127, 128, 129, 4097, 65537] {
                let offset = 7;
                let input = (0..count)
                    .map(|index| (index % 7) as u32)
                    .collect::<Vec<_>>();
                let mut expected = 0;
                let expected = input
                    .iter()
                    .map(|value| {
                        expected += value;
                        expected
                    })
                    .collect::<Vec<_>>();
                let mut words = vec![u32::MAX; offset + count * 2 + 256];
                words[offset..offset + count].copy_from_slice(&input);
                let buffer = device.tensor(&words);
                scan(&device, &buffer, offset, count);
                let actual = device.read(&buffer)?;
                assert_eq!(
                    actual[offset..offset + count],
                    expected,
                    "workgroup {workgroup}, count {count}"
                );
                assert_eq!(actual[..offset], words[..offset]);
                assert_eq!(actual[actual.len() - 16..], [u32::MAX; 16]);
            }
        }
        Ok(())
    }
}

pub(super) fn scan_active(
    device: &Device,
    buffer: &CubeTensor,
    offset: usize,
    count: usize,
    control: &CubeTensor,
    dispatches: &CubeTensor,
) {
    let mut levels = Vec::new();
    let mut offset = offset;
    let mut count = count;
    let mut divisor = 1u32;
    let mut level_index = 0usize;
    while count > 1 {
        let blocks = count.div_ceil(device.workgroup as usize);
        scan_active_blocks::launch(
            &device.client,
            crate::simulation::markets::dispatch(
                dispatches,
                crate::simulation::markets::HISTOGRAM_DISPATCH + level_index * 3,
            ),
            CubeDim::new_1d(device.workgroup),
            buffer.clone().into_buffer_arg(),
            control.clone().into_buffer_arg(),
            offset as u32,
            count as u32,
            divisor,
            ScanMode::new(device),
        );
        levels.push(ScanLevel {
            offset,
            count,
            blocks,
            divisor,
        });
        offset += count;
        count = blocks;
        divisor *= device.workgroup;
        level_index += 1;
    }
    for (index, level) in levels
        .into_iter()
        .enumerate()
        .rev()
        .filter(|(_, level)| level.blocks > 1)
    {
        add_active_offsets::launch(
            &device.client,
            crate::simulation::markets::dispatch(
                dispatches,
                crate::simulation::markets::HISTOGRAM_DISPATCH + index * 3,
            ),
            CubeDim::new_1d(device.workgroup),
            buffer.clone().into_buffer_arg(),
            control.clone().into_buffer_arg(),
            level.offset as u32,
            level.count as u32,
            level.divisor,
        );
    }
}

#[cube(launch)]
fn scan_active_blocks(
    buffer: &mut [u32],
    control: &[Atomic<u32>],
    offset: u32,
    bound: u32,
    divisor: u32,
    #[comptime] mode: ScanMode,
) {
    let count = ceildiv(
        16 * ceildiv(
            control[crate::simulation::flags::CONSTRUCTED_EVENTS_WORD].load(),
            CUBE_DIM,
        ),
        divisor,
    );
    let block = CUBE_POS as u32;
    if block * CUBE_DIM < count && control[crate::simulation::markets::LATCH_WORD].load() == 0 {
        let id = ABSOLUTE_POS as u32;
        let mut value = 0u32;
        if id < count {
            value = buffer[(offset + id) as usize];
        }
        let prefix = scan_prefix(value, mode);
        if id < count {
            buffer[(offset + id) as usize] = prefix.inclusive;
        }
        if UNIT_POS == 0 {
            buffer[(offset + bound + block) as usize] = prefix.total;
        }
    }
}

#[cube(launch)]
fn add_active_offsets(
    buffer: &mut [u32],
    control: &[Atomic<u32>],
    offset: u32,
    bound: u32,
    divisor: u32,
) {
    let count = ceildiv(
        16 * ceildiv(
            control[crate::simulation::flags::CONSTRUCTED_EVENTS_WORD].load(),
            CUBE_DIM,
        ),
        divisor,
    );
    let id = ABSOLUTE_POS as u32;
    let block = CUBE_POS as u32;
    if id < count && block > 0 && control[crate::simulation::markets::LATCH_WORD].load() == 0 {
        buffer[(offset + id) as usize] += buffer[(offset + bound + block - 1) as usize];
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum ScanMode {
    Shared,
    Plane,
}

impl ScanMode {
    pub fn new(device: &Device) -> Self {
        match device
            .client
            .properties()
            .features
            .plane
            .contains(cubecl::features::Plane::Ops)
        {
            true => Self::Plane,
            false => Self::Shared,
        }
    }
}

#[cube]
pub(super) fn scan_prefix(value: u32, #[comptime] mode: ScanMode) -> PrefixSum {
    match mode {
        ScanMode::Shared => prefix_sum(value),
        ScanMode::Plane => {
            let inclusive = plane_inclusive_sum(value);
            let mut planes = Shared::<[u32]>::new_slice(256usize);
            let plane = UNIT_POS / PLANE_DIM;
            if UNIT_POS_PLANE == min(PLANE_DIM, CUBE_DIM - plane * PLANE_DIM) - 1 {
                planes[plane as usize] = inclusive;
            }
            sync_cube();
            let mut preceding = 0u32;
            let mut total = 0u32;
            for index in 0..ceildiv(CUBE_DIM, PLANE_DIM) {
                let count = planes[index as usize];
                total += count;
                if index < plane {
                    preceding += count;
                }
            }
            PrefixSum {
                inclusive: inclusive + preceding,
                total,
            }
        }
    }
}

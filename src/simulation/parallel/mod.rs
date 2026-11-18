pub(super) mod reduction;
mod systems;
pub(super) mod workgroup;
use super::{
    Device,
    system::{AsBufferArg, System},
    words::ceildiv,
};
use burn_wgpu::CubeTensor;
use cubecl::prelude::*;
use systems::*;
pub(super) use workgroup::{PrefixSum, Reduction, prefix_sum, reduce, sum_wide};

pub(super) struct ScanLevel {
    pub offset: usize,
    pub count: usize,
    pub blocks: usize,
}

pub(super) struct ScanHierarchy {
    pub levels: Vec<ScanLevel>,
    pub end: usize,
}

impl ScanHierarchy {
    pub fn new(mut offset: usize, mut count: usize, span: u32, channels: usize) -> Self {
        assert!(span > 1, "a scan block must cover at least two values");
        let mut levels = Vec::new();
        while count > 1 {
            let blocks = count.div_ceil(span as usize);
            levels.push(ScanLevel {
                offset,
                count,
                blocks,
            });
            offset += count * channels;
            count = blocks;
        }
        Self {
            levels,
            end: offset + count * channels,
        }
    }

    pub fn offset_levels(&self) -> &[ScanLevel] {
        &self.levels[..self.levels.len().saturating_sub(1)]
    }
}

pub(super) fn scan(device: &Device, buffer: &CubeTensor, offset: usize, count: usize) {
    let mode = ScanMode::new(device);
    let hierarchy = ScanHierarchy::new(offset, count, device.workgroup, 1);
    for level in &hierarchy.levels {
        BlockScan::launch_elements(
            device,
            level.count,
            BlockScanInputLaunch {
                buffer: buffer.buffer_arg(),
                offset: level.offset as u32,
                count: level.count as u32,
                mode,
            },
        );
    }
    for level in hierarchy.offset_levels().iter().rev() {
        BlockOffsets::launch_elements(
            device,
            level.count,
            BlockOffsetInputLaunch {
                buffer: buffer.buffer_arg(),
                offset: level.offset as u32,
                count: level.count as u32,
            },
        );
    }
}

#[cube]
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

#[cube]
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
    fn block_prefix_sums_preserve_inclusive_values_and_totals_on_gpu() -> Result<()> {
        for workgroup in [32, 256] {
            let device = crate::simulation::tests::device(workgroup)?;
            for mode in [ScanMode::Shared, ScanMode::new(&device)] {
                for count in [0usize, 1, 31, 32, 33, 127, 128, 129, 257, 1025] {
                    let offset = 7;
                    let blocks = count.div_ceil(workgroup as usize);
                    let input = (0..count)
                        .map(|index| match index % 5 {
                            0 => u32::MAX,
                            _ => (index % 7) as u32,
                        })
                        .collect::<Vec<_>>();
                    let mut expected = Vec::new();
                    let mut totals = Vec::new();
                    for block in input.chunks(workgroup as usize) {
                        let mut sum = 0u32;
                        for value in block {
                            sum = sum.wrapping_add(*value);
                            expected.push(sum);
                        }
                        totals.push(sum);
                    }
                    let mut words = vec![u32::MAX; offset + count + blocks + 16];
                    words[offset..offset + count].copy_from_slice(&input);
                    let buffer = device.tensor(&words);
                    BlockScan::launch(
                        &device,
                        device.count(count),
                        CubeDim::new_1d(workgroup),
                        BlockScanInputLaunch {
                            buffer: buffer.buffer_arg(),
                            offset: offset as u32,
                            count: count as u32,
                            mode,
                        },
                    );
                    let actual = device.read(&buffer)?;
                    assert_eq!(
                        actual[offset..offset + count],
                        expected,
                        "workgroup {workgroup}, mode {mode:?}, count {count}"
                    );
                    assert_eq!(actual[offset + count..offset + count + blocks], totals);
                    assert_eq!(actual[..offset], words[..offset]);
                    assert_eq!(actual[actual.len() - 16..], [u32::MAX; 16]);
                }
            }
        }
        Ok(())
    }

    #[derive(Clone, Copy, Debug)]
    enum ActiveScanState {
        Ready,
        Faulted,
    }

    #[test]
    fn active_histogram_prefix_sums_preserve_sparse_tiles_wrapping_and_faults_on_gpu() -> Result<()>
    {
        for workgroup in [32, 256] {
            let device = crate::simulation::tests::device(workgroup)?;
            let boundary =
                workgroup as usize * workgroup as usize * ACTIVE_SCAN_ITEMS as usize / 16;
            let scalar_capacity = (workgroup as usize).pow(3) / 16;
            for capacity in [scalar_capacity, scalar_capacity + 1, 2_097_153] {
                let bound = 16 * capacity.div_ceil(workgroup as usize);
                for active in [0, 1, boundary - 1, boundary, boundary + 1, capacity] {
                    let count = 16 * active.div_ceil(workgroup as usize);
                    let input = (0..count)
                        .map(|index| match index % 5 {
                            0 => u32::MAX,
                            _ => (index % 7) as u32,
                        })
                        .collect::<Vec<_>>();
                    for state in [ActiveScanState::Ready, ActiveScanState::Faulted] {
                        let offset = 7;
                        let mut words = vec![u32::MAX; offset + bound * 2 + 256];
                        words[offset..offset + count].copy_from_slice(&input);
                        let mut control = vec![0; crate::simulation::flags::CONTROL_WORDS + 3];
                        control[crate::simulation::flags::CONSTRUCTED_EVENTS_WORD] = active as u32;
                        control[crate::simulation::flags::FAULT_WORD] = match state {
                            ActiveScanState::Ready => 0,
                            ActiveScanState::Faulted => 1,
                        };
                        let buffer = device.tensor(&words);
                        let control = device.tensor(&control);
                        let dispatches =
                            crate::simulation::markets::SearchBuffers::dispatch_buffer(&device);
                        crate::simulation::markets::prepare_dispatch(
                            &device,
                            &control,
                            &dispatches,
                            capacity,
                        );
                        scan_active(
                            &device,
                            &buffer,
                            offset,
                            bound,
                            &control,
                            &dispatches,
                            HistogramScan::Complete,
                        );
                        let actual = device.read(&buffer)?;
                        let expected = match state {
                            ActiveScanState::Ready => {
                                let mut total = 0u32;
                                input
                                    .iter()
                                    .map(|value| {
                                        total = total.wrapping_add(*value);
                                        total
                                    })
                                    .collect::<Vec<_>>()
                            }
                            ActiveScanState::Faulted => input.clone(),
                        };
                        assert_eq!(
                            actual[offset..offset + count],
                            expected,
                            "workgroup {workgroup}, active {active}, {state:?}"
                        );
                        assert_eq!(actual[..offset], words[..offset]);
                        assert_eq!(actual[actual.len() - 16..], [u32::MAX; 16]);
                        match state {
                            ActiveScanState::Ready => (),
                            ActiveScanState::Faulted => assert_eq!(actual, words),
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn hierarchical_prefix_sums_preserve_partial_blocks_on_gpu() -> Result<()> {
        for workgroup in [32, 256] {
            let device = crate::simulation::tests::device(workgroup)?;
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

const ACTIVE_SCAN_ITEMS: u32 = 8;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(super) enum HistogramScan {
    Complete,
    BlockLocal,
}

#[derive(Clone, Copy)]
pub(super) struct HistogramLayout {
    pub bins: u32,
    pub source_workgroup: u32,
}

#[derive(Clone, Copy)]
pub(super) enum ActiveScanTile {
    Scalar,
    Tiled,
}

impl ActiveScanTile {
    pub fn new(device: &Device, bound: usize) -> Self {
        match bound > (device.workgroup * device.workgroup) as usize {
            true => Self::Tiled,
            false => Self::Scalar,
        }
    }

    pub fn items(self) -> u32 {
        match self {
            Self::Scalar => 1,
            Self::Tiled => ACTIVE_SCAN_ITEMS,
        }
    }
}

pub(super) fn scan_active(
    device: &Device,
    buffer: &CubeTensor,
    offset: usize,
    count: usize,
    control: &CubeTensor,
    dispatches: &CubeTensor,
    output: HistogramScan,
) {
    scan_histogram(
        device,
        buffer,
        offset,
        count,
        control,
        dispatches,
        output,
        HistogramLayout {
            bins: 16,
            source_workgroup: device.workgroup,
        },
    );
}

pub(super) fn scan_histogram(
    device: &Device,
    buffer: &CubeTensor,
    offset: usize,
    count: usize,
    control: &CubeTensor,
    dispatches: &CubeTensor,
    output: HistogramScan,
    layout: HistogramLayout,
) {
    let items = ActiveScanTile::new(device, count).items();
    let mode = ScanMode::new(device);
    let span = device.workgroup * items;
    let hierarchy = ScanHierarchy::new(offset, count, span, 1);
    for (index, level) in hierarchy.levels.iter().enumerate() {
        ActiveBlockScan::launch(
            device,
            crate::simulation::markets::dispatch(
                dispatches,
                crate::simulation::markets::HISTOGRAM_DISPATCH + index * 3,
                level.blocks,
            ),
            CubeDim::new_1d(device.workgroup),
            ActiveScanInputLaunch {
                buffer: buffer.buffer_arg(),
                control: control.buffer_arg(),
                offset: level.offset as u32,
                bound: level.count as u32,
                divisor: span.pow(index as u32),
                items,
                mode,
                histogram_bins: layout.bins,
                source_workgroup: layout.source_workgroup,
            },
        );
    }
    for (index, level) in hierarchy
        .offset_levels()
        .iter()
        .enumerate()
        .rev()
        .filter(|(index, _)| match output {
            HistogramScan::Complete => true,
            HistogramScan::BlockLocal => *index > 0,
        })
    {
        ActiveBlockOffsets::launch(
            device,
            crate::simulation::markets::dispatch(
                dispatches,
                crate::simulation::markets::HISTOGRAM_DISPATCH + index * 3,
                level.blocks,
            ),
            CubeDim::new_1d(device.workgroup),
            ActiveOffsetInputLaunch {
                buffer: buffer.buffer_arg(),
                control: control.buffer_arg(),
                offset: level.offset as u32,
                bound: level.count as u32,
                divisor: span.pow(index as u32),
                items,
                histogram_bins: layout.bins,
                source_workgroup: layout.source_workgroup,
            },
        );
    }
}

#[cube]
fn scan_active_blocks(
    buffer: &mut [u32],
    control: &[Atomic<u32>],
    offset: u32,
    bound: u32,
    divisor: u32,
    #[comptime] items: u32,
    #[comptime] mode: ScanMode,
    #[comptime] histogram_bins: u32,
    #[comptime] source_workgroup: u32,
) {
    let count = ceildiv(
        histogram_bins
            * ceildiv(
                control[crate::simulation::flags::CONSTRUCTED_EVENTS_WORD].load(),
                source_workgroup,
            ),
        divisor,
    );
    let block = CUBE_POS as u32;
    let start = block * CUBE_DIM * items + UNIT_POS * items;
    if block * CUBE_DIM * items < count && crate::simulation::markets::faults_clear(control) {
        let mut values = Array::<u32>::new(items as usize);
        let mut total = 0u32;
        #[unroll]
        for item in 0..items {
            let id = start + item;
            if id < count {
                total += buffer[(offset + id) as usize];
            }
            values[item as usize] = total;
        }
        let prefix = scan_prefix(total, mode);
        let preceding = prefix.inclusive - total;
        #[unroll]
        for item in 0..items {
            let id = start + item;
            if id < count {
                buffer[(offset + id) as usize] = preceding + values[item as usize];
            }
        }
        if UNIT_POS == 0 {
            buffer[(offset + bound + block) as usize] = prefix.total;
        }
    }
}

#[cube]
fn add_active_offsets(
    buffer: &mut [u32],
    control: &[Atomic<u32>],
    offset: u32,
    bound: u32,
    divisor: u32,
    #[comptime] items: u32,
    #[comptime] histogram_bins: u32,
    #[comptime] source_workgroup: u32,
) {
    let count = ceildiv(
        histogram_bins
            * ceildiv(
                control[crate::simulation::flags::CONSTRUCTED_EVENTS_WORD].load(),
                source_workgroup,
            ),
        divisor,
    );
    let block = CUBE_POS as u32;
    let start = block * CUBE_DIM * items + UNIT_POS;
    if start < count && block > 0 && crate::simulation::markets::faults_clear(control) {
        let preceding = buffer[(offset + bound + block - 1) as usize];
        #[unroll]
        for item in 0..items {
            let id = start + item * CUBE_DIM;
            if id < count {
                buffer[(offset + id) as usize] += preceding;
            }
        }
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
pub(super) fn scan_prefix(value: u32, #[comptime] mode: ScanMode) -> PrefixSum<u32> {
    match mode {
        ScanMode::Shared => prefix_sum::<u32>(value),
        ScanMode::Plane => {
            let inclusive = plane_inclusive_sum(value);
            let mut preceding = 0u32;
            let mut total = 0u32;
            if CUBE_DIM <= PLANE_DIM {
                total = plane_sum(value);
            } else {
                let mut planes = workgroup::WorkgroupTree::planes::<u32>();
                let plane = UNIT_POS / PLANE_DIM;
                if UNIT_POS_PLANE == min(PLANE_DIM, CUBE_DIM - plane * PLANE_DIM) - 1 {
                    planes[plane as usize] = inclusive;
                }
                sync_cube();
                for index in 0..ceildiv(CUBE_DIM, PLANE_DIM) {
                    let count = planes[index as usize];
                    total += count;
                    if index < plane {
                        preceding += count;
                    }
                }
            }
            PrefixSum {
                inclusive: inclusive + preceding,
                total,
            }
        }
    }
}

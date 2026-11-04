use super::*;
use crate::simulation::flags::KernelFaults;

struct ScanLevel {
    offset: usize,
    count: usize,
    blocks: usize,
}

pub(super) struct Scan {
    levels: Vec<ScanLevel>,
    channels: usize,
    cube_dim: CubeDim,
    words: usize,
}

impl Scan {
    pub(super) fn new(device: &Device, count: usize, channels: usize) -> Self {
        let mut levels = Vec::new();
        let mut offset = 0;
        let mut count = count;
        while count > 1 {
            let blocks = count.div_ceil(device.workgroup as usize);
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
            channels,
            cube_dim: CubeDim::new_1d(device.workgroup),
            words: (offset + count * channels).max(1) * TICK_WORDS,
        }
    }

    pub(super) fn words(&self) -> usize {
        self.words
    }

    pub(super) fn run(&self, device: &Device, buffer: &CubeTensor, control: &CubeTensor) {
        for level in &self.levels {
            scan_blocks::launch(
                &device.client,
                cubecl::calculate_cube_count_elemwise(
                    &device.client,
                    level.blocks * self.channels * self.cube_dim.num_elems() as usize,
                    self.cube_dim,
                ),
                self.cube_dim,
                argument(buffer),
                argument(control),
                level.offset as u32,
                level.count as u32,
                self.channels as u32,
            );
        }
        for level in self.levels.iter().rev().filter(|level| level.blocks > 1) {
            add_block_offsets::launch(
                &device.client,
                cubecl::calculate_cube_count_elemwise(
                    &device.client,
                    level.count * self.channels,
                    self.cube_dim,
                ),
                self.cube_dim,
                argument(buffer),
                argument(control),
                level.offset as u32,
                level.count as u32,
                self.channels as u32,
            );
        }
    }
}

#[cube(launch)]
fn scan_blocks(
    buffer: &mut [u32],
    control: &[Atomic<u32>],
    offset: u32,
    count: u32,
    channels: u32,
) {
    let blocks = ceildiv(count, CUBE_DIM);
    let channel = CUBE_POS as u32 / blocks;
    let block = CUBE_POS as u32 % blocks;
    let id = block * CUBE_DIM + UNIT_POS;
    let mut value = zero();
    if channel < channels && id < count {
        value = load(buffer, offset + channel * count + id);
    }
    let mut values = Shared::<[Ticks]>::new_slice(256usize);
    values[UNIT_POS as usize] = value;
    sync_cube();
    let mut faults = KernelFaults::none();
    let mut stride = 1u32;
    while stride < CUBE_DIM {
        let index = (UNIT_POS + 1) * stride * 2 - 1;
        if index < CUBE_DIM {
            let sum = plus(values[index as usize], values[(index - stride) as usize]);
            values[index as usize] = sum.value;
            faults.insert(sum.faults);
        }
        sync_cube();
        stride *= 2;
    }
    stride = CUBE_DIM / 4;
    while stride > 0 {
        let index = (UNIT_POS + 1) * stride * 2 - 1;
        if index + stride < CUBE_DIM {
            let sum = plus(values[(index + stride) as usize], values[index as usize]);
            values[(index + stride) as usize] = sum.value;
            faults.insert(sum.faults);
        }
        sync_cube();
        stride /= 2;
    }
    faults.record(control);
    if channel < channels && id < count {
        store(
            buffer,
            offset + channel * count + id,
            values[UNIT_POS as usize],
        );
    }
    if channel < channels && UNIT_POS == 0 {
        store(
            buffer,
            offset + count * channels + channel * blocks + block,
            values[CUBE_DIM as usize - 1],
        );
    }
}

#[cube(launch)]
fn add_block_offsets(
    buffer: &mut [u32],
    control: &[Atomic<u32>],
    offset: u32,
    count: u32,
    channels: u32,
) {
    let id = ABSOLUTE_POS as u32;
    let channel = id / count;
    let block = (id % count) / CUBE_DIM;
    if channel < channels && block > 0 {
        let preceding = offset + count * channels + channel * ceildiv(count, CUBE_DIM) + block - 1;
        let sum = plus(load(buffer, offset + id), load(buffer, preceding));
        sum.faults.record(control);
        store(buffer, offset + id, sum.value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_scan(device: &Device, input: &[u64], channels: usize) -> Result<()> {
        let count = input.len() / channels;
        let plan = Scan::new(device, count, channels);
        let mut words = vec![u32::MAX; plan.words() + 16];
        for (value, words) in input.iter().zip(words.chunks_exact_mut(TICK_WORDS)) {
            words.copy_from_slice(&[*value as u32, (value >> 32) as u32]);
        }
        let buffer = device.tensor(&words);
        let control = device.tensor(&[0; CONTROL_WORDS]);
        plan.run(device, &buffer, &control);
        let actual = device.read(&buffer)?;
        let mut expected_faults = FaultFlags::empty();
        for (channel, values) in input.chunks(count.max(1)).enumerate() {
            let mut sum = 0u128;
            for (index, value) in values.iter().enumerate() {
                sum += u128::from(*value);
                let word = (channel * count + index) * TICK_WORDS;
                let prefix = u64::from(actual[word]) | (u64::from(actual[word + 1]) << 32);
                assert_eq!(
                    prefix, sum as u64,
                    "workgroup {}, count {count}, channel {channel}, index {index}",
                    device.workgroup,
                );
            }
            if sum > u128::from(u64::MAX) {
                expected_faults.insert(FaultFlags::ARITHMETIC_OVERFLOW);
            }
        }
        assert_eq!(FaultFlags::read(&device.read(&control)?)?, expected_faults);
        assert_eq!(actual[plan.words()..], [u32::MAX; 16]);
        Ok(())
    }

    #[test]
    fn hierarchical_tick_scans_preserve_channels_and_partial_blocks_on_gpu() -> Result<()> {
        for workgroup in [32, 64, 128, 256] {
            let device = Device::headless(workgroup)?;
            let block = workgroup as usize;
            for channels in [1, 5] {
                for count in [0, 1, block - 1, block, block + 1, block * block + 1, 65537] {
                    let input = (0..channels)
                        .flat_map(|channel| {
                            (0..count).map(move |index| {
                                ((channel as u64 + index as u64 % 13) << 32) + index as u64 % 17
                            })
                        })
                        .collect::<Vec<_>>();
                    check_scan(&device, &input, channels)?;
                }
            }
        }
        Ok(())
    }

    #[test]
    fn hierarchical_tick_scans_report_overflow_at_every_level_on_gpu() -> Result<()> {
        for workgroup in [32, 64, 128, 256] {
            let device = Device::headless(workgroup)?;
            let block = workgroup as usize;
            check_scan(&device, &[u64::MAX], 1)?;
            check_scan(&device, &[1, u64::MAX - 1, 0], 1)?;
            check_scan(&device, &[1, u64::MAX - 1, 1], 1)?;
            for boundary in [1, block, block * block] {
                let count = boundary + 1;
                let mut input = vec![0; count * 5];
                input[2 * count] = u64::MAX;
                check_scan(&device, &input, 5)?;
                input[2 * count + boundary] = 1;
                check_scan(&device, &input, 5)?;
            }
        }
        Ok(())
    }
}

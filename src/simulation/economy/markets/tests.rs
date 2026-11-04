use super::*;
use crate::{
    domain::Phase,
    simulation::{Device, parallel::ScanMode},
};
use anyhow::Result;
use burn_wgpu::CubeTensor;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum SortStage {
    Local,
    Scatter,
}

#[derive(CubeType, CubeLaunch)]
struct SortInput {
    scratch: Box<[u32]>,
    control: Box<[Atomic<u32>]>,
    parameters: EconomyParameters,
    #[cube(comptime)]
    stage: SortStage,
}

struct SortSystem;
struct SortPass;

impl System for SortSystem {
    type Input = SortInput;
    type Passes = gpu_passes![SortPass];
}

#[cube]
impl GpuPass<SortInput> for SortPass {
    type Output = ();

    fn execute(mut input: SortInput) {
        if input.control[LATCH_WORD].load() == 0 {
            match comptime!(input.stage) {
                SortStage::Local => radix_local(
                    &mut input.scratch,
                    &input.control,
                    &input.parameters,
                    ABSOLUTE_POS as u32,
                ),
                SortStage::Scatter => radix_scatter(
                    &mut input.scratch,
                    &input.control,
                    &input.parameters,
                    ABSOLUTE_POS as u32,
                ),
            }
        }
    }
}

struct SortBuffers {
    scratch: CubeTensor,
    control: CubeTensor,
    dispatches: CubeTensor,
}

impl SortBuffers {
    fn new(device: &Device, words: &[u32], active: usize) -> Self {
        let mut control = vec![0; CONTROL_WORDS + 3];
        control[CONSTRUCTED_EVENTS_WORD] = active as u32;
        let buffers = Self {
            scratch: device.tensor(words),
            control: device.tensor(&control),
            dispatches: device
                .client
                .memory_dedicated_allocation(120, |_| device.empty(30)),
        };
        prepare_dispatch(device, &buffers.control, &buffers.dispatches);
        buffers
    }
}

struct SortCase {
    capacity: usize,
    scan: ScanMode,
}

impl SortCase {
    fn encode(&self, device: &Device, buffers: &SortBuffers) -> usize {
        let mut parameters = EconomyParameters {
            scan: self.scan,
            day: 0,
            settlement_rotation: 0,
            phase: Phase::Opening,
            market: MarketFlag::Labor,
            discovery: Discovery::Indexed,
            offset: 0,
            count: self.capacity as u32,
            status_start: 0,
            sort_source: 0,
            sort_width: 0,
            sort_count: 0,
        };
        for shift in (0..self.capacity.max(1).ilog2() + 1).step_by(4) {
            parameters.sort_width = shift;
            self.launch(device, buffers, &parameters, SortStage::Local);
            parallel::scan_active(
                device,
                &buffers.scratch,
                7 * self.capacity,
                16 * self.capacity.div_ceil(device.workgroup as usize),
                &buffers.control,
                &buffers.dispatches,
            );
            self.launch(device, buffers, &parameters, SortStage::Scatter);
            parameters.sort_source = 2 * self.capacity as u32 - parameters.sort_source;
        }
        parameters.sort_source as usize
    }

    fn launch(
        &self,
        device: &Device,
        buffers: &SortBuffers,
        parameters: &EconomyParameters,
        stage: SortStage,
    ) {
        SortSystem::launch(
            device,
            dispatch(&buffers.dispatches, SORT_DISPATCH),
            CubeDim::new_1d(device.workgroup),
            SortInputLaunch {
                scratch: buffers.scratch.clone().into_buffer_arg(),
                control: buffers.control.clone().into_buffer_arg(),
                parameters: parameters.launch(),
                stage,
            },
        );
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SortItem {
    key: u32,
    item: u32,
}

#[derive(Clone, Copy, Debug)]
enum KeyPattern {
    Shuffled,
    Constant(u32),
}

struct SortExtent {
    capacity: usize,
    active: usize,
}

fn input_words(capacity: usize, active: usize, pattern: KeyPattern) -> Vec<u32> {
    let mut words = vec![u32::MAX; scratch_words(capacity)];
    for index in 0..active {
        words[index] = match pattern {
            KeyPattern::Shuffled => {
                let hash = (index as u32)
                    .wrapping_mul(747_796_405)
                    .wrapping_add(2_891_336_453);
                ((hash >> 13) ^ hash) % capacity.max(1) as u32
            }
            KeyPattern::Constant(key) => key,
        };
        words[capacity + index] = index as u32;
    }
    words
}

fn check_sort(
    device: &Device,
    extent: SortExtent,
    scan: ScanMode,
    pattern: KeyPattern,
) -> Result<()> {
    let SortExtent { capacity, active } = extent;
    let words = input_words(capacity, active, pattern);
    let mut expected = (0..active)
        .map(|index| SortItem {
            key: words[index],
            item: words[capacity + index],
        })
        .collect::<Vec<_>>();
    expected.sort_by_key(|item| item.key);
    let buffers = SortBuffers::new(device, &words, active);
    let case = SortCase { capacity, scan };
    let source = case.encode(device, &buffers);
    let actual = device.read(&buffers.scratch)?;
    let actual_items = (0..active)
        .map(|index| SortItem {
            key: actual[source + index],
            item: actual[source + capacity + index],
        })
        .collect::<Vec<_>>();
    assert_eq!(
        actual_items, expected,
        "workgroup {}, {scan:?}, active {active}, {pattern:?}",
        device.workgroup
    );
    Ok(())
}

#[test]
fn radix_sort_preserves_stable_items_and_sparse_partial_blocks_on_gpu() -> Result<()> {
    for workgroup in [32, 64, 128, 256] {
        let device = Device::headless(workgroup)?;
        let block = workgroup as usize;
        let mut counts = vec![1, 31, 32, 33, block - 1, block, block + 1, 65537];
        counts.sort_unstable();
        counts.dedup();
        for scan in [ScanMode::Shared, ScanMode::new(&device)] {
            let extents = counts
                .iter()
                .map(|&active| SortExtent {
                    capacity: active + 17,
                    active,
                })
                .chain([
                    SortExtent {
                        capacity: 65537,
                        active: 0,
                    },
                    SortExtent {
                        capacity: 65537,
                        active: 257,
                    },
                ]);
            for extent in extents {
                check_sort(&device, extent, scan, KeyPattern::Shuffled)?;
            }
            for digit in 0..4 {
                check_sort(
                    &device,
                    SortExtent {
                        capacity: block,
                        active: block,
                    },
                    scan,
                    KeyPattern::Constant(digit),
                )?;
            }
        }
    }
    Ok(())
}

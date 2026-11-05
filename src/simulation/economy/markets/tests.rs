use super::*;
use crate::{
    domain::Phase,
    simulation::{
        Device,
        layout::{NO_ACTOR, ProposalLayout},
        parallel::ScanMode,
    },
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
                .memory_dedicated_allocation(132, |_| device.empty(33)),
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

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum ProposalStage {
    CountDirect,
    FillDirect,
    RecordSearch,
    CompactSearch,
}

#[derive(CubeType, CubeLaunch)]
struct ProposalInput {
    proposals: Box<[u32]>,
    searches: Box<[u32]>,
    successes: Box<[Atomic<u32>]>,
    scratch: Box<[u32]>,
    control: Box<[Atomic<u32>]>,
    parameters: EconomyParameters,
    #[cube(comptime)]
    stage: ProposalStage,
}

struct ProposalSystem;
struct ProposalPass;

impl System for ProposalSystem {
    type Input = ProposalInput;
    type Passes = gpu_passes![ProposalPass];
}

#[cube]
impl GpuPass<ProposalInput> for ProposalPass {
    type Output = ();

    fn execute(mut input: ProposalInput) {
        let id = ABSOLUTE_POS as u32;
        let count = input.parameters.count;
        match comptime!(input.stage) {
            ProposalStage::CountDirect => {
                let mut peer = u32::new(comptime!(i64::from(NO_ACTOR)));
                if id < count {
                    let actor = (id + input.parameters.settlement_rotation) % count;
                    peer = input.proposals[actor as usize * ProposalLayout::WORDS];
                }
                compact_proposal(peer, &mut input.scratch, &input.parameters, id);
            }
            ProposalStage::FillDirect => compact_fill(
                &mut input.proposals,
                &mut input.scratch,
                &input.control,
                &input.parameters,
                id,
            ),
            ProposalStage::RecordSearch => {
                if id < input.control[SEARCHES_WORD].load() {
                    let actor = input.searches[id as usize];
                    let peer = input.proposals[actor as usize * ProposalLayout::WORDS];
                    if peer != NO_ACTOR {
                        super::super::search::record_success(
                            &input.successes,
                            &input.parameters,
                            actor,
                        );
                    }
                }
            }
            ProposalStage::CompactSearch => super::super::search::compact(
                &input.proposals,
                &input.searches,
                &input.successes,
                &mut input.scratch,
                &input.control,
                &input.parameters,
                id,
            ),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum ProposalDensity {
    Empty,
    Sparse,
    Dense,
}

struct ProposalCase {
    count: usize,
    rotation: usize,
    scan: ScanMode,
    density: ProposalDensity,
}

impl ProposalCase {
    fn check(&self, device: &Device) -> Result<()> {
        let count = self.count;
        let queue = (0..count)
            .rev()
            .filter(|actor| match self.density {
                ProposalDensity::Empty => false,
                ProposalDensity::Sparse => actor % 13 == 0,
                ProposalDensity::Dense => true,
            })
            .collect::<Vec<_>>();
        let mut proposals = vec![NO_ACTOR; count * ProposalLayout::WORDS];
        for &actor in &queue {
            let present = match self.density {
                ProposalDensity::Sparse => actor % 3 != 0,
                _ => true,
            };
            if present {
                proposals[actor * ProposalLayout::WORDS] = (actor % 7) as u32;
            }
        }
        let expected = (0..count)
            .map(|position| (position + self.rotation) % count)
            .filter(|&actor| proposals[actor * ProposalLayout::WORDS] != NO_ACTOR)
            .map(|actor| SortItem {
                key: proposals[actor * ProposalLayout::WORDS],
                item: actor as u32,
            })
            .collect::<Vec<_>>();
        let mut searches = vec![0; count * 5];
        for (job, &actor) in queue.iter().enumerate() {
            searches[job] = actor as u32;
        }
        let searches = device.tensor(&searches);
        let successes = device.tensor(&vec![
            0;
            search_counts_offset(count)
                + 2 * search_blocks(count)
                + 512
        ]);
        let proposals = device.tensor(&proposals);
        let mut control = vec![0; CONTROL_WORDS + 3];
        control[SEARCHES_WORD] = queue.len() as u32;
        let buffers = SortBuffers {
            scratch: device.tensor(&vec![NO_ACTOR; scratch_words(count)]),
            control: device.tensor(&control),
            dispatches: device
                .client
                .memory_dedicated_allocation(132, |_| device.empty(33)),
        };
        let parameters = EconomyParameters {
            scan: self.scan,
            day: 0,
            settlement_rotation: self.rotation as u32,
            phase: Phase::Opening,
            market: MarketFlag::Labor,
            discovery: Discovery::Indexed,
            offset: 0,
            count: count as u32,
            status_start: 0,
            sort_source: 0,
            sort_width: 0,
            sort_count: 0,
        };
        let launch = |stage| {
            let dispatch = match stage {
                ProposalStage::RecordSearch | ProposalStage::CompactSearch => {
                    dispatch(&buffers.dispatches, SEARCH_COMPACT_DISPATCH)
                }
                _ => device.count(count),
            };
            ProposalSystem::launch(
                device,
                dispatch,
                CubeDim::new_1d(device.workgroup),
                ProposalInputLaunch {
                    proposals: proposals.clone().into_buffer_arg(),
                    searches: searches.clone().into_buffer_arg(),
                    successes: successes.clone().into_buffer_arg(),
                    scratch: buffers.scratch.clone().into_buffer_arg(),
                    control: buffers.control.clone().into_buffer_arg(),
                    parameters: parameters.launch(),
                    stage,
                },
            );
        };
        prepare_dispatch(device, &buffers.control, &buffers.dispatches);
        launch(ProposalStage::RecordSearch);
        parallel::scan(
            device,
            &successes,
            search_counts_offset(count),
            search_blocks(count),
        );
        launch(ProposalStage::CompactSearch);
        self.check_output(device, &buffers, &expected)?;
        launch(ProposalStage::CountDirect);
        parallel::scan(
            device,
            &buffers.scratch,
            7 * count,
            count.div_ceil(device.workgroup as usize),
        );
        launch(ProposalStage::FillDirect);
        self.check_output(device, &buffers, &expected)?;
        let statuses = device.read_range(&buffers.scratch, 4 * count, count)?;
        assert_eq!(statuses, vec![EventStatus::Inactive as u32; count]);
        Ok(())
    }

    fn check_output(
        &self,
        device: &Device,
        buffers: &SortBuffers,
        expected: &[SortItem],
    ) -> Result<()> {
        let control = device.read(&buffers.control)?;
        assert_eq!(control[CONSTRUCTED_EVENTS_WORD] as usize, expected.len());
        let scratch = device.read(&buffers.scratch)?;
        let actual = (0..expected.len())
            .map(|slot| SortItem {
                key: scratch[slot],
                item: scratch[self.count + slot],
            })
            .collect::<Vec<_>>();
        assert_eq!(
            actual, expected,
            "count {}, rotation {}, {:?}, {:?}, workgroup {}",
            self.count, self.rotation, self.scan, self.density, device.workgroup
        );
        Ok(())
    }
}

#[test]
fn proposal_compaction_preserves_rotated_order_and_sparse_partial_blocks_on_gpu() -> Result<()> {
    for workgroup in [32, 64, 128, 256] {
        let device = Device::headless(workgroup)?;
        for scan in [ScanMode::Shared, ScanMode::new(&device)] {
            for count in [1, 31, 32, 33, 255, 256, 257, 1025, 65537] {
                let mut rotations = vec![0, 1 % count, count / 2, count - 1];
                rotations.sort_unstable();
                rotations.dedup();
                for rotation in rotations {
                    for density in [
                        ProposalDensity::Empty,
                        ProposalDensity::Sparse,
                        ProposalDensity::Dense,
                    ] {
                        ProposalCase {
                            count,
                            rotation,
                            scan,
                            density,
                        }
                        .check(&device)?;
                    }
                }
            }
        }
    }
    Ok(())
}

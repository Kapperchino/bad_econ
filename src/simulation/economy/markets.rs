use crate::simulation::{
    configuration::{EconomyConfig, Feature},
    flags::*,
    parallel,
    parameters::*,
    pass::GpuPass,
    system::{System, gpu_passes},
    words::ceildiv,
};
use cubecl::prelude::*;

#[cfg(test)]
mod tests;

pub(in crate::simulation) const LATCH_WORD: usize = CONTROL_WORDS;
pub(in crate::simulation) const RANGE_SLOTS_WORD: usize = CONTROL_WORDS + 1;
pub(in crate::simulation) const SEARCHES_WORD: usize = CONTROL_WORDS + 2;
pub(in crate::simulation) const PREPARATION_ACTORS_WORD: usize = CONTROL_WORDS + 3;
pub(in crate::simulation) const FEEDBACK_EVENTS_WORD: usize = PROGRESS_WORD;
const MAX_RANGE_SPACING: u32 = 32;
pub(in crate::simulation) const SEARCH_DISPATCH: usize = 0;
pub(in crate::simulation) const SORT_DISPATCH: usize = 3;
pub(in crate::simulation) const RANGE_DISPATCH: usize = 6;
pub(in crate::simulation) const HISTOGRAM_DISPATCH: usize = 9;
pub(in crate::simulation) const QUEUE_DISPATCH: usize = 27;
pub(in crate::simulation) const SEARCH_COMPACT_DISPATCH: usize = 30;
pub(in crate::simulation) const PREPARATION_DISPATCH: usize = 33;
pub(in crate::simulation) const COMPACTION_BLOCK: u32 = 256;

#[cube]
pub(in crate::simulation) fn faults_clear(control: &[Atomic<u32>]) -> bool {
    (control[LATCH_WORD].load() | control[FAULT_WORD].load()) == 0
}

pub(in crate::simulation) struct SearchBuffers {
    pub requests: burn_wgpu::CubeTensor,
    pub successes: burn_wgpu::CubeTensor,
    pub dispatches: burn_wgpu::CubeTensor,
}

impl SearchBuffers {
    pub fn new(device: &crate::simulation::Device, actors: usize) -> Self {
        Self {
            requests: device.empty(actors * 6),
            successes: device.empty(search_counts_offset(actors) + 2 * search_blocks(actors) + 512),
            dispatches: device
                .client
                .memory_dedicated_allocation(144, |_| device.empty(36)),
        }
    }
}

pub(in crate::simulation) fn search_counts_offset(actors: usize) -> usize {
    actors.div_ceil(32)
}

pub(in crate::simulation) fn search_blocks(actors: usize) -> usize {
    actors.div_ceil(COMPACTION_BLOCK as usize)
}

pub(in crate::simulation) fn dispatch(
    control: &burn_wgpu::CubeTensor,
    offset: usize,
) -> cubecl::CubeCount {
    let handle = control.handle.clone();
    let size = handle.size_in_used();
    cubecl::CubeCount::Dynamic(
        handle
            .offset_start((offset * 4) as u64)
            .offset_end(size - ((offset + 3) * 4) as u64)
            .binding(),
    )
}

pub(in crate::simulation) fn prepare_dispatch(
    device: &crate::simulation::Device,
    control: &burn_wgpu::CubeTensor,
    dispatches: &burn_wgpu::CubeTensor,
    capacity: usize,
) {
    DispatchPreparation::launch(
        device,
        cubecl::CubeCount::Static(1, 1, 1),
        cubecl::CubeDim::new_1d(1),
        DispatchInputLaunch {
            control: control.clone().into_buffer_arg(),
            dispatches: dispatches.clone().into_buffer_arg(),
            workgroup: device.workgroup,
            histogram_span: device.workgroup
                * parallel::ActiveScanTile::new(
                    device,
                    16 * capacity.div_ceil(device.workgroup as usize),
                )
                .items(),
            preparation_resets: (search_counts_offset(capacity) + search_blocks(capacity)) as u32,
        },
    );
}

#[derive(CubeType, CubeLaunch)]
struct DispatchInput {
    control: Box<[Atomic<u32>]>,
    dispatches: Box<[u32]>,
    workgroup: u32,
    histogram_span: u32,
    preparation_resets: u32,
}

struct DispatchPreparation;
struct PrepareCounts;

impl System for DispatchPreparation {
    type Input = DispatchInput;
    type Passes = gpu_passes![PrepareCounts];
}

#[cube]
impl GpuPass<DispatchInput> for PrepareCounts {
    type Output = ();

    fn execute(mut input: DispatchInput) {
        prepare_counts(
            &input.control,
            &mut input.dispatches,
            input.workgroup,
            input.histogram_span,
            input.preparation_resets,
        );
    }
}

#[cube]
fn prepare_counts(
    control: &[Atomic<u32>],
    dispatches: &mut [u32],
    workgroup: u32,
    histogram_span: u32,
    preparation_resets: u32,
) {
    control[LATCH_WORD].store(control[crate::simulation::flags::FAULT_WORD].load());
    let mut counts = Array::<u32>::new(12usize);
    counts[0usize] = control[SEARCHES_WORD].load();
    counts[1usize] = ceildiv(control[CONSTRUCTED_EVENTS_WORD].load(), workgroup);
    counts[2usize] = ceildiv(control[RANGE_SLOTS_WORD].load(), workgroup);
    counts[3usize] = ceildiv(16 * counts[1usize], histogram_span);
    for index in 4..9usize {
        counts[index] = ceildiv(counts[index - 1], histogram_span);
    }
    counts[9usize] = control[RANGE_SLOTS_WORD].load();
    counts[10usize] = ceildiv(counts[0usize], workgroup);
    let mut preparation_actors = 0u32;
    if control.len() > PREPARATION_ACTORS_WORD {
        preparation_actors = control[PREPARATION_ACTORS_WORD].load();
    }
    counts[11usize] = ceildiv(max(preparation_actors, preparation_resets), workgroup);
    for index in 0..12usize {
        let count = counts[index];
        dispatches[index * 3] = min(count, 65535);
        dispatches[index * 3 + 1] = ceildiv(max(count, 1), 65535);
        dispatches[index * 3 + 2] = 1;
    }
}

pub(in crate::simulation) fn scratch_words(actors: usize) -> usize {
    actors * 8 + 512
}

#[cube]
pub(in crate::simulation) fn active(
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
) -> u32 {
    match comptime!(parameters.discovery) {
        Discovery::Indexed => control[CONSTRUCTED_EVENTS_WORD].load(),
        #[cfg(test)]
        _ => parameters.sort_count,
    }
}

#[cube]
pub(in crate::simulation) fn ranges(
    scratch: &mut [u32],
    control: &[Atomic<u32>],
    config: &EconomyConfig,
    parameters: &EconomyParameters,
    id: u32,
) {
    let count = parameters.count;
    let active = active(control, parameters);
    let source = parameters.sort_source;
    let mut first = false;
    if id < active {
        first = id == 0;
        if id > 0 {
            first = scratch[(source + id - 1) as usize] != scratch[(source + id) as usize];
        }
    }
    let mut slots = 0u32;
    let mut length = 0u32;
    if first {
        let key = scratch[(source + id) as usize];
        let mut low = id + 1;
        let mut high = active;
        while low < high {
            let middle = (low + high) / 2;
            let same = scratch[(source + middle) as usize] == key;
            low = select(same, middle + 1, low);
            high = select(same, high, middle);
        }
        length = low - id;
        scratch[(7 * count + key) as usize] = low;
        slots = min(length, MAX_RANGE_SPACING);
        if comptime!(
            matches!(parameters.market, MarketFlag::Consumer(_))
                || parameters.market == MarketFlag::Labor
                    && config.bargaining.enabled == Feature::Disabled
        ) {
            slots = 1;
        }
    }
    let prefix = parallel::scan_prefix(slots, comptime!(parameters.scan));
    let mut base = Shared::<u32>::new();
    if UNIT_POS == 0 && prefix.total > 0 {
        *base = control[RANGE_SLOTS_WORD].fetch_add(prefix.total);
    }
    sync_cube();
    if first {
        let slot = *base + prefix.inclusive - slots;
        scratch[(5 * count + slot) as usize] = id;
        scratch[(6 * count + slot) as usize] = length;
        for gap in 1..slots {
            scratch[(5 * count + slot + gap) as usize] = u32::MAX;
        }
    }
}

#[cube]
pub(in crate::simulation) fn compact_proposal(
    peer: u32,
    scratch: &mut [u32],
    parameters: &EconomyParameters,
    id: u32,
) {
    let count = parameters.count;
    let mut present = 0u32;
    if id < count {
        present = u32::cast_from(peer != u32::MAX);
        KernelEventStatus::new(comptime!(EventStatus::Inactive)).store(scratch, 4 * count + id);
        scratch[(5 * count + id) as usize] = u32::MAX;
    }
    let prefix = parallel::scan_prefix(present, comptime!(parameters.scan));
    if id < count {
        scratch[(6 * count + id) as usize] = prefix.inclusive;
    }
    if UNIT_POS == 0 {
        scratch[(7 * count + CUBE_POS as u32) as usize] = prefix.total;
    }
}

#[cube]
pub(in crate::simulation) fn compact_fill(
    proposals: &mut [u32],
    scratch: &mut [u32],
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
    id: u32,
) {
    let count = parameters.count;
    let block = CUBE_POS as u32;
    if id == 0 {
        let blocks = ceildiv(count, CUBE_DIM);
        control[CONSTRUCTED_EVENTS_WORD].store(scratch[(7 * count + blocks - 1) as usize]);
    }
    if id < count {
        let actor = (id + parameters.settlement_rotation) % count;
        let peer = proposals[actor as usize * crate::simulation::layout::ProposalLayout::WORDS];
        if peer != u32::MAX {
            let mut slot = scratch[(6 * count + id) as usize] - 1;
            if block > 0 {
                slot += scratch[(7 * count + block - 1) as usize];
            }
            scratch[slot as usize] = peer;
            scratch[(count + slot) as usize] = actor;
        }
    }
}

#[cube]
pub(in crate::simulation) fn radix_local(
    scratch: &mut [u32],
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
    id: u32,
) {
    let count = parameters.count;
    let active_count = active(control, parameters);
    let block = CUBE_POS as u32;
    let blocks = ceildiv(active_count, CUBE_DIM);
    let lane = UNIT_POS as usize;
    let source = parameters.sort_source;
    let shift = parameters.sort_width;
    if block * CUBE_DIM < active_count {
        let mut keys = Shared::<[u32]>::new_slice(256usize);
        let mut items = Shared::<[u32]>::new_slice(256usize);
        let mut starts = Shared::<[u32]>::new_slice(16usize);
        let mut ends = Shared::<[u32]>::new_slice(16usize);
        if UNIT_POS < 16 {
            starts[lane] = 0;
            ends[lane] = 0;
        }
        let mut key = u32::new(comptime!(i64::from(u32::MAX)));
        let mut item = 0u32;
        if id < active_count {
            key = scratch[(source + id) as usize];
            item = scratch[(source + count + id) as usize];
        }
        #[unroll]
        for pair in 0..2u32 {
            let digit = (key >> (shift + 2 * pair)) & 3;
            let encoded = select(digit < 3, 1u32 << (9 * digit), 0u32);
            let prefix = parallel::scan_prefix(encoded, comptime!(parameters.scan));
            let preceding = prefix.total & ((1u32 << (9 * digit)) - 1);
            let base = (preceding & 511) + ((preceding >> 9) & 511) + (preceding >> 18);
            let counted = (prefix.inclusive & 511)
                + ((prefix.inclusive >> 9) & 511)
                + (prefix.inclusive >> 18);
            let inclusive = select(
                digit < 3,
                (prefix.inclusive >> (9 * digit)) & 511,
                UNIT_POS + 1 - counted,
            );
            let target = base + inclusive - 1;
            keys[target as usize] = key;
            items[target as usize] = item;
            sync_cube();
            key = keys[lane];
            item = items[lane];
            sync_cube();
        }
        let digit = (key >> shift) & 15;
        let length = min(CUBE_DIM, active_count - block * CUBE_DIM);
        let previous = (keys[(max(UNIT_POS, 1) - 1) as usize] >> shift) & 15;
        let next = (keys[min(UNIT_POS + 1, CUBE_DIM - 1) as usize] >> shift) & 15;
        if id < active_count && (UNIT_POS == 0 || previous != digit) {
            starts[digit as usize] = UNIT_POS;
        }
        if id < active_count && (UNIT_POS + 1 == length || next != digit) {
            ends[digit as usize] = UNIT_POS + 1;
        }
        sync_cube();
        if id < active_count {
            scratch[(source + id) as usize] = key;
            scratch[(source + count + id) as usize] = item;
            scratch[(6 * count + id) as usize] = UNIT_POS - starts[digit as usize];
        }
        if UNIT_POS < 16 {
            scratch[(7 * count + UNIT_POS * blocks + block) as usize] = ends[lane] - starts[lane];
        }
    }
}

#[cube]
pub(in crate::simulation) fn radix_scatter(
    scratch: &mut [u32],
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
    id: u32,
) {
    let count = parameters.count;
    if id < active(control, parameters) {
        let blocks = ceildiv(active(control, parameters), CUBE_DIM);
        let source = parameters.sort_source;
        let destination = 2 * count - source;
        let key = scratch[(source + id) as usize];
        let digit = (key >> parameters.sort_width) & 15;
        let histogram = digit * blocks + CUBE_POS as u32;
        let mut target = scratch[(6 * count + id) as usize];
        if histogram > 0 {
            target += scratch[(7 * count + histogram - 1) as usize];
        }
        scratch[(destination + target) as usize] = key;
        scratch[(destination + count + target) as usize] = scratch[(source + count + id) as usize];
    }
}

pub(in crate::simulation) fn feedback_boundary(
    device: &crate::simulation::Device,
    control: &burn_wgpu::CubeTensor,
    feedback: &burn_wgpu::CubeTensor,
) {
    FeedbackBoundary::launch(
        device,
        cubecl::CubeCount::Static(1, 1, 1),
        cubecl::CubeDim::new_1d(1),
        FeedbackInputLaunch {
            control: control.clone().into_buffer_arg(),
            feedback: feedback.clone().into_buffer_arg(),
        },
    );
}

#[derive(CubeType, CubeLaunch)]
struct FeedbackInput {
    control: Box<[Atomic<u32>]>,
    feedback: Box<[Atomic<u32>]>,
}

struct FeedbackBoundary;
struct LatchFeedback;

impl System for FeedbackBoundary {
    type Input = FeedbackInput;
    type Passes = gpu_passes![LatchFeedback];
}

#[cube]
impl GpuPass<FeedbackInput> for LatchFeedback {
    type Output = ();

    fn execute(input: FeedbackInput) {
        latch_feedback(&input.control, &input.feedback);
    }
}

#[cube]
fn latch_feedback(control: &[Atomic<u32>], feedback: &[Atomic<u32>]) {
    feedback[CONSTRUCTED_EVENTS_WORD].store(control[FAULT_WORD].load());
    feedback[FEEDBACK_EVENTS_WORD].store(control[CONSTRUCTED_EVENTS_WORD].load());
}

use super::{flags::*, parallel, parameters::*, words::ceildiv};
use cubecl::prelude::*;

pub(super) const LATCH_WORD: usize = CONTROL_WORDS;
pub(super) const RANGES_WORD: usize = CONTROL_WORDS + 1;
pub(super) const SEARCHES_WORD: usize = CONTROL_WORDS + 2;
pub(super) const SEARCH_DISPATCH: usize = 0;
pub(super) const SORT_DISPATCH: usize = 3;
pub(super) const RANGE_DISPATCH: usize = 6;
pub(super) const HISTOGRAM_DISPATCH: usize = 9;

pub(super) struct SearchBuffers {
    pub requests: burn_wgpu::CubeTensor,
    pub dispatches: burn_wgpu::CubeTensor,
}

impl SearchBuffers {
    pub fn new(device: &super::Device, actors: usize) -> Self {
        Self {
            requests: device.empty(actors * 5),
            dispatches: device
                .client
                .memory_dedicated_allocation(108, |_| device.empty(27)),
        }
    }
}

pub(super) fn dispatch(control: &burn_wgpu::CubeTensor, offset: usize) -> cubecl::CubeCount {
    let handle = control.handle.clone();
    let size = handle.size_in_used();
    cubecl::CubeCount::Dynamic(
        handle
            .offset_start((offset * 4) as u64)
            .offset_end(size - ((offset + 3) * 4) as u64)
            .binding(),
    )
}

pub(super) fn prepare_dispatch(
    device: &super::Device,
    control: &burn_wgpu::CubeTensor,
    dispatches: &burn_wgpu::CubeTensor,
) {
    prepare_counts::launch(
        &device.client,
        cubecl::CubeCount::Static(1, 1, 1),
        cubecl::CubeDim::new_1d(1),
        control.clone().into_buffer_arg(),
        dispatches.clone().into_buffer_arg(),
        device.workgroup,
    );
}

#[cube(launch)]
fn prepare_counts(control: &[Atomic<u32>], dispatches: &mut [u32], workgroup: u32) {
    let mut counts = Array::<u32>::new(9usize);
    counts[0usize] = control[SEARCHES_WORD].load();
    counts[1usize] = ceildiv(control[CONSTRUCTED_EVENTS_WORD].load(), workgroup);
    counts[2usize] = ceildiv(control[RANGES_WORD].load(), workgroup);
    counts[3usize] = ceildiv(16 * counts[1usize], workgroup);
    for index in 4..9usize {
        counts[index] = ceildiv(counts[index - 1], workgroup);
    }
    for index in 0..9usize {
        let count = counts[index];
        dispatches[index * 3] = min(max(count, 1), 65535);
        dispatches[index * 3 + 1] = ceildiv(max(count, 1), 65535);
        dispatches[index * 3 + 2] = 1;
    }
}

pub(super) fn scratch_words(actors: usize) -> usize {
    actors * 8 + 512
}

#[cube]
pub(super) fn active(control: &[Atomic<u32>], parameters: &EconomyParameters) -> u32 {
    match comptime!(parameters.discovery) {
        Discovery::Indexed => control[CONSTRUCTED_EVENTS_WORD].load(),
        #[cfg(test)]
        _ => parameters.sort_count,
    }
}

#[cube]
pub(super) fn ranges(
    scratch: &mut [u32],
    control: &[Atomic<u32>],
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
    let prefix = parallel::scan_prefix(u32::cast_from(first), comptime!(parameters.scan));
    let mut base = Shared::<u32>::new();
    if UNIT_POS == 0 {
        *base = control[RANGES_WORD].fetch_add(prefix.total);
    }
    sync_cube();
    if first {
        scratch[(5 * count + *base + prefix.inclusive - 1) as usize] = id;
    }
}

#[cube]
pub(super) fn compact_count(
    proposals: &mut [u32],
    scratch: &mut [u32],
    parameters: &EconomyParameters,
    id: u32,
) {
    let count = parameters.count;
    let actor = (id + parameters.settlement_rotation) % max(count, 1);
    let mut present = 0u32;
    if id < count {
        present = u32::cast_from(
            proposals[actor as usize * crate::simulation::layout::ProposalLayout::WORDS]
                != u32::MAX,
        );
        KernelEventStatus::new(comptime!(EventStatus::Inactive)).store(scratch, 4 * count + id);
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
pub(super) fn compact_fill(
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
pub(super) fn radix_local(
    scratch: &mut [u32],
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
    id: u32,
) {
    let count = parameters.count;
    let active_count = active(control, parameters);
    let block = CUBE_POS as u32;
    let blocks = ceildiv(active(control, parameters), CUBE_DIM);
    let lane = UNIT_POS as usize;
    let source = parameters.sort_source;
    let shift = parameters.sort_width;
    if UNIT_POS < 16 && block < blocks {
        scratch[(7 * count + UNIT_POS * blocks + block) as usize] = 0;
    }
    if block * CUBE_DIM < active_count {
        let mut keys = Shared::<[u32]>::new_slice(256usize);
        let mut items = Shared::<[u32]>::new_slice(256usize);
        let mut key = u32::new(comptime!(i64::from(u32::MAX)));
        let mut item = 0u32;
        if id < active_count {
            key = scratch[(source + id) as usize];
            item = scratch[(source + count + id) as usize];
        }
        #[unroll]
        for bit in 0..4u32 {
            let zero = u32::cast_from((key >> (shift + bit)) & 1 == 0);
            let prefix = parallel::scan_prefix(zero, comptime!(parameters.scan));
            let target = select(
                zero == 1,
                prefix.inclusive - 1,
                prefix.total + UNIT_POS - prefix.inclusive,
            );
            keys[target as usize] = key;
            items[target as usize] = item;
            sync_cube();
            key = keys[lane];
            item = items[lane];
            sync_cube();
        }
        let digit = (key >> shift) & 15;
        let mut low = 0u32;
        let mut high = UNIT_POS;
        while low < high {
            let mid = (low + high) / 2;
            let before = ((keys[mid as usize] >> shift) & 15) < digit;
            low = select(before, mid + 1, low);
            high = select(before, high, mid);
        }
        if id < active_count {
            scratch[(source + id) as usize] = key;
            scratch[(source + count + id) as usize] = item;
            scratch[(6 * count + id) as usize] = UNIT_POS - low;
        }
        if UNIT_POS < 16 {
            let mut total = 0u32;
            let length = min(CUBE_DIM, active_count - block * CUBE_DIM);
            for index in 0..length {
                total += u32::cast_from(((keys[index as usize] >> shift) & 15) == UNIT_POS);
            }
            scratch[(7 * count + UNIT_POS * blocks + block) as usize] = total;
        }
    }
}

#[cube]
pub(super) fn radix_scatter(
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

pub(super) fn feedback_boundary(
    device: &super::Device,
    control: &burn_wgpu::CubeTensor,
    feedback: &burn_wgpu::CubeTensor,
) {
    latch_feedback::launch(
        &device.client,
        cubecl::CubeCount::Static(1, 1, 1),
        cubecl::CubeDim::new_1d(1),
        control.clone().into_buffer_arg(),
        feedback.clone().into_buffer_arg(),
    );
}

#[cube(launch)]
fn latch_feedback(control: &[Atomic<u32>], feedback: &[Atomic<u32>]) {
    feedback[CONSTRUCTED_EVENTS_WORD].store(control[LATCH_WORD].load());
}

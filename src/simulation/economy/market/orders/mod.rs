use crate::simulation::{
    economy::{
        configuration::{EconomyConfig, Feature},
        parameters::*,
    },
    parallel,
    pass::GpuPass,
    storage::{flags::*, words::ceildiv},
    system::{AsBufferArg, System, gpu_passes},
};
use cubecl::prelude::*;

#[cfg(test)]
mod tests;

pub(in crate::simulation) const LATCH_WORD: usize = CONTROL_WORDS;
pub(in crate::simulation) const RANGE_SLOTS_WORD: usize = CONTROL_WORDS + 1;
pub(in crate::simulation) const SEARCHES_WORD: usize = CONTROL_WORDS + 2;
pub(in crate::simulation) const PREPARATION_ACTORS_WORD: usize = CONTROL_WORDS + 3;
pub(in crate::simulation) const WIDE_ORDERS_WORD: usize = CONTROL_WORDS + 4;
pub(in crate::simulation) const FEEDBACK_EVENTS_WORD: usize = PROGRESS_WORD;
pub(in crate::simulation) const FEEDBACK_SOURCE_WORD: usize = COMPLETED_WORD;
const MAX_RANGE_SPACING: u32 = 32;
pub(in crate::simulation) const SEARCH_DISPATCH: usize = 0;
pub(in crate::simulation) const SORT_DISPATCH: usize = 3;
pub(in crate::simulation) const RANGE_DISPATCH: usize = 6;
pub(in crate::simulation) const HISTOGRAM_DISPATCH: usize = 9;
pub(in crate::simulation) const QUEUE_DISPATCH: usize = 27;
pub(in crate::simulation) const SEARCH_COMPACT_DISPATCH: usize = 30;
pub(in crate::simulation) const PREPARATION_DISPATCH: usize = 33;
pub(in crate::simulation) const COMPACTION_BLOCK: u32 = 256;
pub(in crate::simulation) const SEARCH_REQUEST_FIELDS: usize = 18;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
#[repr(u32)]
pub(in crate::simulation) enum FeedbackSource {
    Constructed,
    Searches,
}

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
    pub fn new(device: &crate::simulation::Device, rows: &crate::domain::ActorRows) -> Self {
        let actors = rows.len();
        let layout = crate::domain::SupplierCacheLayout::new(rows);
        Self {
            requests: device.empty(actors * (SEARCH_REQUEST_FIELDS + layout.extra_words)),
            successes: device.empty(Self::success_words(rows)),
            dispatches: Self::dispatch_buffer(device),
        }
    }

    pub fn resized(
        &self,
        device: &crate::simulation::Device,
        rows: &crate::domain::ActorRows,
    ) -> Self {
        let actors = rows.len();
        let layout = crate::domain::SupplierCacheLayout::new(rows);
        Self {
            requests: device.resize_scratch(
                &self.requests,
                actors * (SEARCH_REQUEST_FIELDS + layout.extra_words),
            ),
            successes: device.resize_scratch(&self.successes, Self::success_words(rows)),
            dispatches: self.dispatches.clone(),
        }
    }

    fn success_words(rows: &crate::domain::ActorRows) -> usize {
        let actors = rows.len();
        let first_firm = rows.firm_indices().next().unwrap_or(actors);
        search_counts_offset(actors)
            + 2 * search_blocks(actors)
            + 512
            + actors
            + 2 * (actors - first_firm)
    }

    pub(in crate::simulation) fn dispatch_buffer(
        device: &crate::simulation::Device,
    ) -> burn_wgpu::CubeTensor {
        device
            .client
            .memory_dedicated_allocation(144, |_| device.empty(36))
    }
}

pub(in crate::simulation) fn search_counts_offset(actors: usize) -> usize {
    actors.div_ceil(32)
}

pub(in crate::simulation) fn search_blocks(actors: usize) -> usize {
    actors.div_ceil(COMPACTION_BLOCK as usize)
}

#[cube]
pub(in crate::simulation) fn filtered_count(actors: u32, peer: u32) -> u32 {
    ceildiv(actors, 32) + 2 * ceildiv(actors, COMPACTION_BLOCK) + 512 + peer
}

pub(in crate::simulation) fn dispatch(
    control: &burn_wgpu::CubeTensor,
    offset: usize,
    max_groups: usize,
) -> crate::simulation::system::Dispatch {
    use crate::simulation::system::Dispatch;

    match control.device {
        cubecl::Device::Metal(_) => Dispatch::Bounded {
            counts: control.buffer_arg(),
            offset: u32::try_from(offset).expect("dispatch offset exceeds GPU indexing capacity"),
            max_groups: u32::try_from(max_groups).expect("dispatch exceeds GPU indexing capacity"),
        },
        _ => {
            let handle = control.handle.clone();
            let size = handle.size_in_used();
            cubecl::CubeCount::Dynamic(
                handle
                    .offset_start((offset * 4) as u64)
                    .offset_end(size - ((offset + 3) * 4) as u64)
                    .binding(),
            )
            .into()
        }
    }
}

pub(in crate::simulation) fn job_dispatch(
    dispatches: &burn_wgpu::CubeTensor,
    offset: usize,
    capacity: usize,
) -> crate::simulation::system::Dispatch {
    match dispatches.device {
        cubecl::Device::Metal(_) => {
            const WORKGROUPS_PER_CORE: usize = 32;
            let cores = dispatches
                .client
                .properties()
                .hardware
                .num_streaming_multiprocessors
                .unwrap_or(16);
            let groups = capacity.min(cores as usize * WORKGROUPS_PER_CORE).max(1);
            cubecl::CubeCount::Static(groups as u32, 1, 1).into()
        }
        _ => dispatch(dispatches, offset, capacity),
    }
}

#[cube]
pub(in crate::simulation) fn job_count(control: &[Atomic<u32>], #[comptime] word: usize) -> u32 {
    select(control[LATCH_WORD].load() == 0, control[word].load(), 0)
}

#[derive(Clone, Copy)]
pub(in crate::simulation) enum PreparationWorkgroup {
    Native,
    #[cfg(test)]
    Wide,
}

impl PreparationWorkgroup {
    pub(in crate::simulation) fn width(self, device: &crate::simulation::Device) -> u32 {
        match self {
            Self::Native => device.workgroup,
            #[cfg(test)]
            Self::Wide => 256,
        }
    }
}

pub(in crate::simulation) fn prepare_dispatch(
    device: &crate::simulation::Device,
    control: &burn_wgpu::CubeTensor,
    dispatches: &burn_wgpu::CubeTensor,
    capacity: usize,
) {
    prepare_dispatch_for(
        device,
        control,
        dispatches,
        capacity,
        PreparationWorkgroup::Native,
    );
}

pub(in crate::simulation) fn prepare_dispatch_for(
    device: &crate::simulation::Device,
    control: &burn_wgpu::CubeTensor,
    dispatches: &burn_wgpu::CubeTensor,
    capacity: usize,
    preparation: PreparationWorkgroup,
) {
    DispatchPreparation::launch_single(
        device,
        DispatchInputLaunch {
            control: control.buffer_arg(),
            dispatches: dispatches.buffer_arg(),
            workgroup: device.workgroup,
            histogram_span: device.workgroup
                * parallel::ActiveScanTile::new(
                    device,
                    16 * capacity.div_ceil(device.workgroup as usize),
                )
                .items(),
            preparation_resets: (search_counts_offset(capacity) + search_blocks(capacity)) as u32,
            preparation_workgroup: preparation.width(device),
        },
    );
}

#[derive(CubeType, CubeLaunch)]
struct DispatchInput {
    control: Box<[Atomic<u32>]>,
    dispatches: Box<[u32]>,
    #[cube(comptime)]
    workgroup: u32,
    #[cube(comptime)]
    histogram_span: u32,
    preparation_resets: u32,
    #[cube(comptime)]
    preparation_workgroup: u32,
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
            comptime!(input.workgroup),
            comptime!(input.histogram_span),
            input.preparation_resets,
            comptime!(input.preparation_workgroup),
        );
    }
}

#[cube]
fn prepare_counts(
    control: &[Atomic<u32>],
    dispatches: &mut [u32],
    #[comptime] workgroup: u32,
    #[comptime] histogram_span: u32,
    preparation_resets: u32,
    #[comptime] preparation_workgroup: u32,
) {
    control[LATCH_WORD].store(control[crate::simulation::storage::flags::FAULT_WORD].load());
    let searches = control[SEARCHES_WORD].load();
    let order_groups = ceildiv(control[CONSTRUCTED_EVENTS_WORD].load(), workgroup);
    let ranges = control[RANGE_SLOTS_WORD].load();
    store_dispatch(dispatches, SEARCH_DISPATCH, searches);
    store_dispatch(dispatches, SORT_DISPATCH, order_groups);
    store_dispatch(dispatches, RANGE_DISPATCH, ceildiv(ranges, workgroup));
    store_histogram_dispatches(dispatches, 16 * order_groups, histogram_span);
    store_dispatch(dispatches, QUEUE_DISPATCH, ranges);
    store_dispatch(
        dispatches,
        SEARCH_COMPACT_DISPATCH,
        ceildiv(searches, workgroup),
    );
    let mut preparation_actors = 0u32;
    if control.len() > PREPARATION_ACTORS_WORD {
        preparation_actors = control[PREPARATION_ACTORS_WORD].load();
    }
    store_dispatch(
        dispatches,
        PREPARATION_DISPATCH,
        ceildiv(
            max(preparation_actors, preparation_resets),
            preparation_workgroup,
        ),
    );
}

#[cube]
pub(super) fn store_histogram_dispatches(
    dispatches: &mut [u32],
    mut count: u32,
    #[comptime] span: u32,
) {
    for level in 0..6usize {
        count = ceildiv(count, span);
        store_dispatch(dispatches, HISTOGRAM_DISPATCH + level * 3, count);
    }
}

#[cube]
pub(in crate::simulation) fn store_dispatch(dispatches: &mut [u32], offset: usize, groups: u32) {
    dispatches[offset] = min(groups, 65535);
    dispatches[offset + 1] = ceildiv(max(groups, 1), 65535);
    dispatches[offset + 2] = 1;
}

pub(in crate::simulation) fn scratch_words(actors: usize) -> usize {
    actors * 9 + 512
}

#[cube]
pub(in crate::simulation) fn active(
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
) -> u32 {
    active_count(
        control,
        comptime!(parameters.discovery),
        parameters.sort_count,
    )
}

#[cube]
pub(in crate::simulation::economy) fn active_count(
    control: &[Atomic<u32>],
    #[comptime] discovery: Discovery,
    _sort_count: u32,
) -> u32 {
    match comptime!(discovery) {
        Discovery::Indexed => control[CONSTRUCTED_EVENTS_WORD].load(),
        #[cfg(test)]
        _ => _sort_count,
    }
}

#[derive(CubeType)]
pub(in crate::simulation::economy) struct OrderRanges<'a> {
    scratch: &'a mut [u32],
    control: &'a [Atomic<u32>],
    parameters: &'a EconomyParameters,
}

#[cube]
impl<'a> OrderRanges<'a> {
    pub(in crate::simulation::economy) fn new(
        scratch: &'a mut [u32],
        control: &'a [Atomic<u32>],
        parameters: &'a EconomyParameters,
    ) -> Self {
        OrderRanges::<'a> {
            scratch,
            control,
            parameters,
        }
    }

    fn peer(&self, order: u32) -> u32 {
        self.scratch[(self.parameters.sort_source + order) as usize]
    }

    fn after_peer(&self, start: u32, active: u32, peer: u32) -> u32 {
        let mut low = start + 1;
        let mut high = active;
        while low < high {
            let middle = low + (high - low) / 2;
            let same = self.peer(middle) == peer;
            low = select(same, middle + 1, low);
            high = select(same, high, middle);
        }
        low
    }

    fn store(&mut self, slot: u32, start: u32, length: u32) {
        let count = self.parameters.count;
        self.scratch[(5 * count + slot) as usize] = start;
        self.scratch[(6 * count + slot) as usize] = length;
    }

    pub(in crate::simulation::economy) fn record_supplier(
        &mut self,
        spatial: &[u32],
        config: &EconomyConfig,
        member: u32,
    ) {
        let cells = config.spatial_side * config.spatial_side;
        let orders = active(self.control, self.parameters);
        if orders > 0 && member < spatial[cells as usize] {
            let peer = spatial[(cells + 1 + member) as usize];
            let mut start = 0u32;
            let mut end = orders;
            while start < end {
                let middle = start + (end - start) / 2;
                let smaller = self.peer(middle) < peer;
                start = select(smaller, middle + 1, start);
                end = select(smaller, end, middle);
            }
            if start < orders && self.peer(start) == peer {
                let after = self.after_peer(start, orders, peer);
                let slot = self.control[RANGE_SLOTS_WORD].fetch_add(1);
                self.store(slot, start, after - start);
                self.scratch[(7 * self.parameters.count + peer) as usize] = after;
            }
        }
    }

    pub(in crate::simulation::economy) fn record_order(&mut self, config: &EconomyConfig, id: u32) {
        let parameters = self.parameters;
        let count = parameters.count;
        let active = active(self.control, parameters);
        let mut first = false;
        if id < active {
            first = id == 0;
            if id > 0 {
                first = self.peer(id - 1) != self.peer(id);
            }
        }
        let mut slots = 0u32;
        let mut length = 0u32;
        if first {
            let peer = self.peer(id);
            let after = self.after_peer(id, active, peer);
            length = after - id;
            self.scratch[(7 * count + peer) as usize] = after;
            slots = min(length, MAX_RANGE_SPACING);
            if comptime!(
                matches!(
                    parameters.market,
                    MarketFlag::Consumer(_) | MarketFlag::Payroll
                ) || parameters.market == MarketFlag::Labor
                    && config.bargaining.enabled == Feature::Disabled
            ) {
                slots = 1;
            }
        }
        let prefix = parallel::scan_prefix(slots, comptime!(parameters.scan));
        let mut base = Shared::<u32>::new();
        if UNIT_POS == 0 && prefix.total > 0 {
            *base = self.control[RANGE_SLOTS_WORD].fetch_add(prefix.total);
        }
        sync_cube();
        if first {
            let slot = *base + prefix.inclusive - slots;
            self.store(slot, id, length);
            for gap in 1..slots {
                self.scratch[(5 * count + slot + gap) as usize] = u32::MAX;
            }
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
        let peer =
            proposals[actor as usize * crate::simulation::storage::layout::ProposalLayout::WORDS];
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
    radix_local_keys(scratch, control, parameters, id, 0);
}

#[cube]
fn radix_key(key: u32, base: u32) -> u32 {
    select(key == u32::MAX, key, key - base)
}

#[derive(CubeType)]
pub(in crate::simulation) struct RadixPassSize {
    active: u32,
    pub blocks: u32,
}

#[cube]
impl RadixPassSize {
    pub fn new(control: &[Atomic<u32>], parameters: &EconomyParameters) -> Self {
        let active = active(control, parameters);
        RadixPassSize {
            active,
            blocks: ceildiv(active, CUBE_DIM),
        }
    }
}

#[cube]
pub(in crate::simulation) fn radix_local_keys(
    scratch: &mut [u32],
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
    id: u32,
    key_base: u32,
) {
    let size = RadixPassSize::new(control, parameters);
    radix_local_digits(scratch, &size, parameters, id, key_base, 4u32, 16u32);
}

#[cube]
pub(in crate::simulation) fn radix_local_digits(
    scratch: &mut [u32],
    size: &RadixPassSize,
    parameters: &EconomyParameters,
    id: u32,
    key_base: u32,
    #[comptime] bits: u32,
    histogram_bins: u32,
) {
    let count = parameters.count;
    let active_count = size.active;
    let block = id / CUBE_DIM;
    let blocks = size.blocks;
    let lane = UNIT_POS as usize;
    let source = parameters.sort_source;
    let shift = parameters.sort_width;
    if block * CUBE_DIM < active_count {
        let mut keys = Shared::<[u32]>::new_slice(256usize);
        let mut items = Shared::<[u32]>::new_slice(256usize);
        let mut starts = Shared::<[u32]>::new_slice((1u32 << bits) as usize);
        let mut ends = Shared::<[u32]>::new_slice((1u32 << bits) as usize);
        if UNIT_POS < (1u32 << bits) {
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
        for pair in 0..(bits + 1) / 2 {
            let mask = select(2 * pair + 1 < bits, 3u32, 1u32);
            let digit = (radix_key(key, key_base) >> (shift + 2 * pair)) & mask;
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
        let digit = (radix_key(key, key_base) >> shift) & ((1u32 << bits) - 1);
        let length = min(CUBE_DIM, active_count - block * CUBE_DIM);
        let previous = (radix_key(keys[(max(UNIT_POS, 1) - 1) as usize], key_base) >> shift)
            & ((1u32 << bits) - 1);
        let next = (radix_key(keys[min(UNIT_POS + 1, CUBE_DIM - 1) as usize], key_base) >> shift)
            & ((1u32 << bits) - 1);
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
        if UNIT_POS < histogram_bins {
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
    radix_scatter_histogram(
        scratch,
        control,
        parameters,
        id,
        comptime!(parallel::HistogramScan::Complete),
        1u32,
    );
}

#[cube]
pub(in crate::simulation) fn radix_scatter_histogram(
    scratch: &mut [u32],
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
    id: u32,
    #[comptime] output: parallel::HistogramScan,
    #[comptime] items: u32,
) {
    radix_scatter_keys(scratch, control, parameters, id, 0, output, items);
}

#[cube]
pub(in crate::simulation) fn radix_scatter_keys(
    scratch: &mut [u32],
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
    id: u32,
    key_base: u32,
    #[comptime] output: parallel::HistogramScan,
    #[comptime] items: u32,
) {
    let size = RadixPassSize::new(control, parameters);
    radix_scatter_digits(
        scratch,
        &size,
        parameters,
        id,
        key_base,
        comptime!(output),
        4u32,
        CUBE_DIM * items,
        16 * ceildiv(parameters.count, CUBE_DIM),
    );
}

#[cube]
pub(in crate::simulation) fn radix_scatter_digits(
    scratch: &mut [u32],
    size: &RadixPassSize,
    parameters: &EconomyParameters,
    id: u32,
    key_base: u32,
    #[comptime] output: parallel::HistogramScan,
    #[comptime] bits: u32,
    histogram_span: u32,
    histogram_bound: u32,
) {
    let count = parameters.count;
    if id < size.active {
        let blocks = size.blocks;
        let source = parameters.sort_source;
        let destination = 2 * count - source;
        let key = scratch[(source + id) as usize];
        let digit = (radix_key(key, key_base) >> parameters.sort_width) & ((1u32 << bits) - 1);
        let histogram = digit * blocks + id / CUBE_DIM;
        let mut target = scratch[(6 * count + id) as usize];
        if histogram > 0 {
            target += scratch[(7 * count + histogram - 1) as usize];
        }
        target += radix_histogram_offset(
            scratch,
            count,
            histogram,
            output,
            histogram_span,
            histogram_bound,
        );
        scratch[(destination + target) as usize] = key;
        scratch[(destination + count + target) as usize] = scratch[(source + count + id) as usize];
    }
}

#[cube]
fn radix_histogram_offset(
    scratch: &mut [u32],
    count: u32,
    histogram: u32,
    #[comptime] output: parallel::HistogramScan,
    histogram_span: u32,
    histogram_bound: u32,
) -> u32 {
    match comptime!(output) {
        parallel::HistogramScan::Complete => 0u32,
        parallel::HistogramScan::BlockLocal => {
            let tile = (max(histogram, 1) - 1) / histogram_span;
            let mut offset = 0u32;
            if tile > 0 {
                offset = scratch[(7 * count + histogram_bound + tile - 1) as usize];
            }
            offset
        }
    }
}

pub(in crate::simulation) fn feedback_boundary(
    device: &crate::simulation::Device,
    control: &burn_wgpu::CubeTensor,
    feedback: &burn_wgpu::CubeTensor,
    source: FeedbackSource,
) {
    FeedbackBoundary::launch_single(
        device,
        FeedbackInputLaunch {
            control: control.buffer_arg(),
            feedback: feedback.buffer_arg(),
            source,
        },
    );
}

#[derive(CubeType, CubeLaunch)]
struct FeedbackInput {
    control: Box<[Atomic<u32>]>,
    feedback: Box<[Atomic<u32>]>,
    #[cube(comptime)]
    source: FeedbackSource,
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
        latch_feedback(&input.control, &input.feedback, comptime!(input.source));
    }
}

#[cube]
fn latch_feedback(
    control: &[Atomic<u32>],
    feedback: &[Atomic<u32>],
    #[comptime] requested: FeedbackSource,
) {
    let source = match comptime!(requested) {
        FeedbackSource::Constructed => u32::new(comptime!(FeedbackSource::Constructed as i64)),
        FeedbackSource::Searches => select(
            control[WIDE_ORDERS_WORD].load() == 0,
            comptime!(FeedbackSource::Searches as u32),
            comptime!(FeedbackSource::Constructed as u32),
        ),
    };
    feedback[CONSTRUCTED_EVENTS_WORD].store(control[FAULT_WORD].load());
    feedback[FEEDBACK_SOURCE_WORD].store(source);
    feedback[FEEDBACK_EVENTS_WORD].store(select(
        source == comptime!(FeedbackSource::Searches as u32),
        control[SEARCHES_WORD].load(),
        control[CONSTRUCTED_EVENTS_WORD].load(),
    ));
}

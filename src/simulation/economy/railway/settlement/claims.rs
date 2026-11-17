use super::*;

#[cfg(test)]
mod tests;

const SORT_DISPATCH: usize = 12;

const SORT_TILE: usize = 2048;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SortMethod {
    Merge,
    #[cfg(test)]
    Block256,
    Block2048,
}

impl SortMethod {
    fn tile(self) -> usize {
        match self {
            Self::Merge => 1,
            #[cfg(test)]
            Self::Block256 => 256,
            Self::Block2048 => SORT_TILE,
        }
    }

    pub(super) fn for_device(self, device: &Device) -> Self {
        let hardware = &device.client.properties().hardware;
        match (
            hardware.max_shared_memory_size >= self.tile() * 12,
            hardware.max_units_per_cube >= 256,
            hardware.max_cube_dim.0 >= 256,
        ) {
            (true, true, true) => self,
            _ => Self::Merge,
        }
    }
}

#[derive(CubeType)]
struct Claim {
    resource: u32,
    priority: u32,
    buyer: u32,
    amount: Ticks,
}

#[cube]
impl Claim {
    fn load(words: &[u32], index: u32) -> Self {
        let base = index as usize * FREIGHT_CLAIM_WORDS;
        Claim {
            resource: words[base],
            priority: words[base + 1],
            buyer: words[base + 2],
            amount: ticks(words[base + 3], words[base + 4]),
        }
    }

    fn store(&self, words: &mut [u32], index: u32) {
        let base = index as usize * FREIGHT_CLAIM_WORDS;
        words[base] = self.resource;
        words[base + 1] = self.priority;
        words[base + 2] = self.buyer;
        words[base + 3] = self.amount.extract(LOW_WORD);
        words[base + 4] = self.amount.extract(HIGH_WORD);
    }

    fn before(&self, other: &Self) -> bool {
        self.resource < other.resource
            || self.resource == other.resource && self.priority < other.priority
    }
}

#[cube]
pub(super) fn add_demand(words: &[Atomic<u32>], index: usize, overflow: usize, amount: Ticks) {
    let low = amount.extract(LOW_WORD);
    let mut carry = 0u32;
    if low > 0 {
        let previous = words[index].fetch_add(low);
        carry = u32::cast_from(previous + low < previous);
    }
    let high = amount.extract(HIGH_WORD);
    let increment = high + carry;
    let mut overflowed = increment < high;
    if increment > 0 {
        let previous = words[index + 1].fetch_add(increment);
        overflowed = overflowed || previous + increment < previous;
    }
    if overflowed {
        words[overflow].fetch_or(1);
    }
}

#[cube]
pub(super) fn demand(words: &[Atomic<u32>], index: usize, overflow: usize) -> Ticks {
    let amount = atomic_ticks(words, index);
    select(
        words[overflow].load() == 0,
        amount,
        ticks(u32::MAX, u32::MAX),
    )
}

#[cube]
pub(super) fn saturated_sum(left: Ticks, right: Ticks) -> Ticks {
    let sum = plus(left, right);
    select(sum.faults.is_empty(), sum.value, ticks(u32::MAX, u32::MAX))
}

#[cube]
pub(super) fn contested(input: &Input, resource: u32) -> bool {
    let base = resource as usize * RESOURCE_WORDS;
    input.resources[base + REQUEST_OVERFLOW].load() != 0
        || lt(
            budget(input, resource),
            atomic_ticks(&input.resources, base + REQUESTED),
        )
}

#[cube]
fn emit_claim(input: &mut Input, claim: Claim) {
    if !equal(claim.amount, zero()) && contested(&*input, claim.resource) {
        let index = input.progress[CLAIM_CURSOR].fetch_add(1);
        claim.store(&mut input.claims, index);
        let capacity = input.claim_offsets
            / (FREIGHT_CLAIM_WORDS as u32
                + crate::domain::railway::FREIGHT_CLAIM_SORT_WORDS as u32);
        input.claims[(capacity * FREIGHT_CLAIM_WORDS as u32 + index) as usize] = index;
    }
}

#[cube]
pub(super) fn emit(input: &mut Input, invocation: u32) {
    if invocation < input.progress[ACTIVE].load() {
        let priority = input.queue[invocation as usize];
        let buyer = input.scratch
            [(input.parameters.sort_source + input.parameters.count + priority) as usize]
            + input.parameters.offset;
        if input.bookings[buyer as usize * FREIGHT_BOOKING_WORDS]
            == comptime!(BookingState::Ready as u32)
        {
            let seller =
                input.proposals[buyer as usize * ProposalLayout::WORDS + ProposalLayout::ACTOR];
            let path = candidate(
                &input.actors,
                &input.config,
                buyer,
                seller,
                input.bookings[buyer as usize * FREIGHT_BOOKING_WORDS + 1],
            );
            let amount = quantity(&*input, buyer);
            let cost = multiply(&input.control, amount, path.fee);
            emit_claim(
                input,
                Claim {
                    resource: seller,
                    priority,
                    buyer,
                    amount: cost,
                },
            );
            let mut cursor = RegionCursor::new(&path);
            let mut record = cursor.next(&input.config);
            while record < input.config.railway.region_count {
                let resource = input.config.actors + record;
                emit_claim(
                    input,
                    Claim {
                        resource,
                        priority,
                        buyer,
                        amount,
                    },
                );
                record = cursor.next(&input.config);
            }
        }
    }
}

#[derive(CubeType, CubeLaunch)]
struct DispatchInput {
    progress: Box<[Atomic<u32>]>,
    control: Box<[Atomic<u32>]>,
    dispatches: Box<[u32]>,
    workgroup: u32,
    #[cube(comptime)]
    execution: Execution,
}

struct ClaimDispatch;
struct PrepareClaims;

impl System for ClaimDispatch {
    type Input = DispatchInput;
    type Passes = gpu_passes![PrepareClaims];
}

#[cube]
impl GpuPass<DispatchInput> for PrepareClaims {
    type Output = ();

    fn execute(mut input: DispatchInput) {
        let mut groups = 0u32;
        if execution_enabled(&input.progress, &input.control, comptime!(input.execution)) {
            groups = ceildiv(input.progress[CLAIM_COUNT].load(), input.workgroup);
        }
        input.dispatches[SORT_DISPATCH] = min(groups, 65535);
        input.dispatches[SORT_DISPATCH + 1] = ceildiv(max(groups, 1), 65535);
        input.dispatches[SORT_DISPATCH + 2] = 1;
    }
}

#[derive(CubeType, CubeLaunch)]
struct SortInput {
    claims: Box<[u32]>,
    progress: Box<[Atomic<u32>]>,
    capacity: u32,
    source: u32,
    width: u32,
}

struct ClaimSort;
struct SortClaims;

impl System for ClaimSort {
    type Input = SortInput;
    type Passes = gpu_passes![SortClaims];
}

pub(super) fn sorted_source(capacity: usize) -> u32 {
    let passes = capacity.max(1).next_power_of_two().ilog2();
    (capacity * (FREIGHT_CLAIM_WORDS + passes as usize % 2)) as u32
}

pub(super) fn settlement_source(buffers: &Buffers) -> u32 {
    match buffers.sorting {
        SortMethod::Merge => sorted_source(buffers.claim_capacity),
        method => block_sorted_source(buffers.claim_capacity, method.tile()),
    }
}

fn block_sorted_source(capacity: usize, tile: usize) -> u32 {
    let passes = capacity
        .max(1)
        .next_power_of_two()
        .ilog2()
        .saturating_sub(tile.ilog2());
    (capacity * (FREIGHT_CLAIM_WORDS + passes as usize % 2)) as u32
}

pub(super) fn offsets_start(capacity: usize) -> u32 {
    (capacity * (FREIGHT_CLAIM_WORDS + crate::domain::railway::FREIGHT_CLAIM_SORT_WORDS)) as u32
}

pub(super) fn sort_buffered(
    device: &Device,
    buffers: &Buffers,
    control: &CubeTensor,
    execution: Execution,
) {
    if buffers.claim_capacity > 0 {
        match buffers.sorting {
            SortMethod::Merge => sort_merges_buffered(device, buffers, control, execution),
            method => sort_blocks_buffered(device, buffers, control, execution, method.tile()),
        }
    }
}

fn sort_merges_buffered(
    device: &Device,
    buffers: &Buffers,
    control: &CubeTensor,
    execution: Execution,
) {
    ClaimDispatch::launch(
        device,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        DispatchInputLaunch {
            progress: buffers.progress.clone().into_buffer_arg(),
            control: control.clone().into_buffer_arg(),
            dispatches: buffers.dispatches.clone().into_buffer_arg(),
            workgroup: device.workgroup,
            execution,
        },
    );
    let mut width = 1;
    let mut source = (buffers.claim_capacity * FREIGHT_CLAIM_WORDS) as u32;
    while width < buffers.claim_capacity {
        ClaimSort::launch(
            device,
            markets::dispatch(&buffers.dispatches, SORT_DISPATCH),
            CubeDim::new_1d(device.workgroup),
            SortInputLaunch {
                claims: buffers.claims.clone().into_buffer_arg(),
                progress: buffers.progress.clone().into_buffer_arg(),
                capacity: buffers.claim_capacity as u32,
                source,
                width: width as u32,
            },
        );
        source = (buffers.claim_capacity * (2 * FREIGHT_CLAIM_WORDS + 1) - source as usize) as u32;
        width *= 2;
    }
}

#[derive(CubeType, CubeLaunch)]
struct BlockInput {
    claims: Box<[u32]>,
    progress: Box<[Atomic<u32>]>,
    capacity: u32,
    #[cube(comptime)]
    tile: usize,
}

struct ClaimBlocks;
struct SortClaimBlocks;

impl System for ClaimBlocks {
    type Input = BlockInput;
    type Passes = gpu_passes![SortClaimBlocks];
}

#[cube]
impl GpuPass<BlockInput> for SortClaimBlocks {
    type Output = ();

    fn execute(mut input: BlockInput) {
        let count = input.progress[CLAIM_COUNT].load();
        let start = CUBE_POS as u32 * comptime!(input.tile as u32);
        if start < count {
            let mut extent = 1u32;
            while extent < min(count - start, comptime!(input.tile as u32)) {
                extent *= 2;
            }
            let source = input.capacity * FREIGHT_CLAIM_WORDS as u32;
            let mut keys = Shared::<[u64]>::new_slice(comptime!(input.tile));
            let mut indices = Shared::<[u32]>::new_slice(comptime!(input.tile));
            let mut slot = UNIT_POS;
            while slot < extent {
                let position = start + slot;
                let mut key = u64::new(-1);
                let mut index = u32::new(comptime!(i64::from(u32::MAX)));
                if position < count {
                    index = input.claims[(source + position) as usize];
                    let claim = Claim::load(&input.claims, index);
                    key = (u64::cast_from(claim.resource) << 32) | u64::cast_from(claim.priority);
                }
                keys[slot as usize] = key;
                indices[slot as usize] = index;
                slot += CUBE_DIM;
            }
            sync_cube();
            let mut width = 2u32;
            while width <= extent {
                let mut stride = width / 2;
                while stride > 0 {
                    let mut slot = UNIT_POS;
                    while slot < extent {
                        let other = slot ^ stride;
                        if other > slot {
                            let key = keys[slot as usize];
                            let index = indices[slot as usize];
                            let other_key = keys[other as usize];
                            let other_index = indices[other as usize];
                            let smaller =
                                other_key < key || other_key == key && other_index < index;
                            let larger = key < other_key || other_key == key && index < other_index;
                            let preferred = select(slot & width == 0, smaller, larger);
                            keys[slot as usize] = select(preferred, other_key, key);
                            indices[slot as usize] = select(preferred, other_index, index);
                            keys[other as usize] = select(preferred, key, other_key);
                            indices[other as usize] = select(preferred, index, other_index);
                        }
                        slot += CUBE_DIM;
                    }
                    sync_cube();
                    stride /= 2;
                }
                width *= 2;
            }
            let mut slot = UNIT_POS;
            while slot < min(count - start, comptime!(input.tile as u32)) {
                input.claims[(source + start + slot) as usize] = indices[slot as usize];
                slot += CUBE_DIM;
            }
        }
    }
}

fn sort_blocks_buffered(
    device: &Device,
    buffers: &Buffers,
    control: &CubeTensor,
    execution: Execution,
    tile: usize,
) {
    ClaimDispatch::launch(
        device,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        DispatchInputLaunch {
            progress: buffers.progress.clone().into_buffer_arg(),
            control: control.clone().into_buffer_arg(),
            dispatches: buffers.dispatches.clone().into_buffer_arg(),
            workgroup: tile as u32,
            execution,
        },
    );
    ClaimBlocks::launch(
        device,
        markets::dispatch(&buffers.dispatches, SORT_DISPATCH),
        CubeDim::new_1d(256),
        BlockInputLaunch {
            claims: buffers.claims.clone().into_buffer_arg(),
            progress: buffers.progress.clone().into_buffer_arg(),
            capacity: buffers.claim_capacity as u32,
            tile,
        },
    );
    if buffers.claim_capacity > tile {
        ClaimDispatch::launch(
            device,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            DispatchInputLaunch {
                progress: buffers.progress.clone().into_buffer_arg(),
                control: control.clone().into_buffer_arg(),
                dispatches: buffers.dispatches.clone().into_buffer_arg(),
                workgroup: device.workgroup,
                execution,
            },
        );
    }
    let mut width = tile;
    let mut source = (buffers.claim_capacity * FREIGHT_CLAIM_WORDS) as u32;
    while width < buffers.claim_capacity {
        ClaimSort::launch(
            device,
            markets::dispatch(&buffers.dispatches, SORT_DISPATCH),
            CubeDim::new_1d(device.workgroup),
            SortInputLaunch {
                claims: buffers.claims.clone().into_buffer_arg(),
                progress: buffers.progress.clone().into_buffer_arg(),
                capacity: buffers.claim_capacity as u32,
                source,
                width: width as u32,
            },
        );
        source = (buffers.claim_capacity * (2 * FREIGHT_CLAIM_WORDS + 1) - source as usize) as u32;
        width *= 2;
    }
}

#[cube]
impl GpuPass<SortInput> for SortClaims {
    type Output = ();

    fn execute(mut input: SortInput) {
        let invocation = ABSOLUTE_POS as u32;
        let count = input.progress[CLAIM_COUNT].load();
        if invocation < count {
            let index = input.claims[(input.source + invocation) as usize];
            let claim = Claim::load(&input.claims, index);
            let start = invocation / (2 * input.width) * (2 * input.width);
            let middle = min(start + input.width, count);
            let end = min(start + 2 * input.width, count);
            let left = invocation < middle;
            let other_start = select(left, middle, start);
            let mut low = other_start;
            let mut high = select(left, end, middle);
            while low < high {
                let probe = low + (high - low) / 2;
                let other =
                    Claim::load(&input.claims, input.claims[(input.source + probe) as usize]);
                let precedes = select(left, other.before(&claim), !claim.before(&other));
                if precedes {
                    low = probe + 1;
                } else {
                    high = probe;
                }
            }
            let rank = invocation - select(left, start, middle);
            let target = start + rank + low - other_start;
            input.claims[(input.capacity * (2 * FREIGHT_CLAIM_WORDS as u32 + 1) - input.source
                + target) as usize] = index;
        }
    }
}

#[cube]
fn lower_bound(input: &Input, resource: u32) -> u32 {
    let mut low = 0u32;
    let mut high = input.progress[CLAIM_COUNT].load();
    while low < high {
        let middle = low + (high - low) / 2;
        let index = input.claims[(input.claim_source + middle) as usize];
        let claim = Claim::load(&input.claims, index);
        if claim.resource < resource {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    low
}

#[cube]
pub(super) fn offset(input: &mut Input, resource: u32) {
    if resource <= input.config.actors + input.config.railway.region_count {
        input.claims[(input.claim_offsets + resource) as usize] = lower_bound(&*input, resource);
    }
}

#[derive(Clone, Copy)]
#[repr(u32)]
enum PrefixState {
    Available,
    Exceeded,
}

#[cube]
pub(super) fn resolve(input: &mut Input, resource: u32) {
    if contested(&*input, resource) {
        let start = lower_bound(&*input, resource);
        let end = lower_bound(&*input, resource + 1);
        let mut remaining = budget(&*input, resource);
        let mut state = u32::new(comptime!(PrefixState::Available as i64));
        for position in start..end {
            let index = input.claims[(input.claim_source + position) as usize];
            let claim = Claim::load(&input.claims, index);
            state = select(
                lt(remaining, claim.amount),
                comptime!(PrefixState::Exceeded as u32),
                state,
            );
            if state == comptime!(PrefixState::Available as u32) {
                remaining = minus(remaining, claim.amount);
            } else {
                input.accounts[claim.buyer as usize * ACCOUNT_WORDS + DENIED].fetch_or(1);
            }
        }
    }
}

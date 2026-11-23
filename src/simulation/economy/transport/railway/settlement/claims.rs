use super::booking::ReadyBooking;
use super::*;

#[cfg(test)]
mod tests;

const SORT_TILE: usize = 2048;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SortMethod {
    Merge,
    Block2048,
}

impl SortMethod {
    pub(super) fn tile(self) -> usize {
        match self {
            Self::Merge => 1,
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
    if AtomicBalance::at(words, index).overflowing_add(amount) {
        words[overflow].fetch_or(1);
    }
}

#[cube]
pub(super) fn demand(words: &[Atomic<u32>], index: usize, overflow: usize) -> Ticks {
    let amount = AtomicBalance::at(words, index).load();
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

#[derive(CubeType, CubeLaunch)]
struct SortInput {
    control: Box<[Atomic<u32>]>,
    #[cube(comptime)]
    execution: Execution,
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

pub(super) fn settlement_source(buffers: &Buffers) -> u32 {
    let capacity = buffers.claim_capacity;
    let tile = buffers.sorting.tile();
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
) -> usize {
    let mut launches = 0;
    if buffers.claim_capacity > 0 {
        if let Execution::Immediate(_) = execution {
            admit_claims(device, buffers, control, execution);
            launches += 1;
        }
        let tile = buffers.sorting.tile();
        if let SortMethod::Block2048 = buffers.sorting {
            ClaimBlocks::launch(
                device,
                orders::dispatch(
                    &buffers.dispatches,
                    BLOCK_SORT_DISPATCH,
                    buffers.claim_capacity.div_ceil(tile),
                ),
                CubeDim::new_1d(256),
                BlockInputLaunch {
                    claims: buffers.claims.buffer_arg(),
                    progress: buffers.progress.buffer_arg(),
                    capacity: buffers.claim_capacity as u32,
                    tile,
                    control: control.buffer_arg(),
                    execution,
                },
            );
            launches += 1;
        }
        launches += merge_passes(device, buffers, control, execution, tile);
    }
    launches
}

fn merge_passes(
    device: &Device,
    buffers: &Buffers,
    control: &CubeTensor,
    execution: Execution,
    mut width: usize,
) -> usize {
    let mut launches = 0;
    let mut source = (buffers.claim_capacity * FREIGHT_CLAIM_WORDS) as u32;
    while width < buffers.claim_capacity {
        ClaimSort::launch(
            device,
            orders::dispatch(
                &buffers.dispatches,
                MERGE_SORT_DISPATCH,
                buffers.claim_capacity.div_ceil(device.workgroup as usize),
            ),
            CubeDim::new_1d(device.workgroup),
            SortInputLaunch {
                claims: buffers.claims.buffer_arg(),
                progress: buffers.progress.buffer_arg(),
                capacity: buffers.claim_capacity as u32,
                source,
                width: width as u32,
                control: control.buffer_arg(),
                execution,
            },
        );
        source = (buffers.claim_capacity * (2 * FREIGHT_CLAIM_WORDS + 1) - source as usize) as u32;
        width *= 2;
        launches += 1;
    }
    launches
}

#[derive(CubeType, CubeLaunch)]
struct BlockInput {
    control: Box<[Atomic<u32>]>,
    #[cube(comptime)]
    execution: Execution,
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
        if start < count
            && execution_enabled(&input.progress, &input.control, comptime!(input.execution))
        {
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

#[cube]
impl GpuPass<SortInput> for SortClaims {
    type Output = ();

    fn execute(mut input: SortInput) {
        let invocation = ABSOLUTE_POS as u32;
        let count = input.progress[CLAIM_COUNT].load();
        if invocation < count
            && execution_enabled(&input.progress, &input.control, comptime!(input.execution))
        {
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

#[derive(Clone, Copy)]
#[repr(u32)]
enum PrefixState {
    Available,
    Exceeded,
}

#[cube]
impl FreightSettlementInput {
    pub(super) fn contested(&self, resource: u32) -> bool {
        let base = resource as usize * RESOURCE_WORDS;
        self.resources[base + REQUEST_OVERFLOW].load() != 0
            || lt(
                self.budget(resource),
                AtomicBalance::at(&self.resources, base + REQUESTED).load(),
            )
    }

    fn emit_claim(&mut self, claim: Claim) {
        if !equal(claim.amount, zero()) && self.contested(claim.resource) {
            let index = self.progress[CLAIM_CURSOR].fetch_add(1);
            claim.store(&mut self.claims, index);
            let capacity = self.claim_offsets
                / (FREIGHT_CLAIM_WORDS as u32
                    + crate::domain::railway::FREIGHT_CLAIM_SORT_WORDS as u32);
            self.claims[(capacity * FREIGHT_CLAIM_WORDS as u32 + index) as usize] = index;
        }
    }

    pub(super) fn emit_claims(&mut self, booking: &ReadyBooking) {
        let cost = times(booking.quantity, booking.path.fee).record(&self.control);
        self.emit_claim(Claim {
            resource: booking.seller,
            priority: booking.priority,
            buyer: booking.buyer,
            amount: cost,
        });
        let mut cursor = RegionCursor::new(&booking.path);
        let mut record = cursor.next(&self.config);
        while record < self.config.railway.region_count {
            self.emit_claim(Claim {
                resource: self.config.actors + record,
                priority: booking.priority,
                buyer: booking.buyer,
                amount: booking.quantity,
            });
            record = cursor.next(&self.config);
        }
    }

    fn first_claim(&self, resource: u32) -> u32 {
        let mut low = 0u32;
        let mut high = self.progress[CLAIM_COUNT].load();
        while low < high {
            let middle = low + (high - low) / 2;
            let index = self.claims[(self.claim_source + middle) as usize];
            let claim = Claim::load(&self.claims, index);
            if claim.resource < resource {
                low = middle + 1;
            } else {
                high = middle;
            }
        }
        low
    }

    pub(super) fn store_claim_offset(&mut self, resource: u32) {
        if resource <= self.config.actors + self.config.railway.region_count {
            self.claims[(self.claim_offsets + resource) as usize] = self.first_claim(resource);
        }
    }

    pub(super) fn resolve_claims(&mut self, resource: u32) {
        if self.contested(resource) {
            let start = self.first_claim(resource);
            let end = self.first_claim(resource + 1);
            let mut remaining = self.budget(resource);
            let mut state = u32::new(comptime!(PrefixState::Available as i64));
            for position in start..end {
                let index = self.claims[(self.claim_source + position) as usize];
                let claim = Claim::load(&self.claims, index);
                state = select(
                    lt(remaining, claim.amount),
                    comptime!(PrefixState::Exceeded as u32),
                    state,
                );
                if state == comptime!(PrefixState::Available as u32) {
                    remaining = minus(remaining, claim.amount);
                } else {
                    self.accounts[claim.buyer as usize * ACCOUNT_WORDS + DENIED].fetch_or(1);
                }
            }
        }
    }
}

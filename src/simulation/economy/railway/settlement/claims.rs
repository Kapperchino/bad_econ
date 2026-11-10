use super::*;

#[cfg(test)]
mod tests;

const SORT_DISPATCH: usize = 12;

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
    let previous = words[index].fetch_add(low);
    let carry = u32::cast_from(previous + low < previous);
    let high = amount.extract(HIGH_WORD);
    let previous_high = words[index + 1].fetch_add(high);
    let before_carry = words[index + 1].fetch_add(carry);
    if previous_high + high < previous_high || before_carry + carry < before_carry {
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
    dispatches: Box<[u32]>,
    workgroup: u32,
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
        let groups = ceildiv(input.progress[CLAIM_COUNT].load(), input.workgroup);
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

pub(super) fn offsets_start(capacity: usize) -> u32 {
    (capacity * (FREIGHT_CLAIM_WORDS + crate::domain::railway::FREIGHT_CLAIM_SORT_WORDS)) as u32
}

pub(super) fn sort(device: &Device, buffers: &Buffers) {
    ClaimDispatch::launch(
        device,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        DispatchInputLaunch {
            progress: buffers.progress.clone().into_buffer_arg(),
            dispatches: buffers.dispatches.clone().into_buffer_arg(),
            workgroup: device.workgroup,
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
pub(super) fn offset(input: &mut Input, resource: u32) {
    if resource <= input.config.actors + input.config.railway.region_count {
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
        input.claims[(input.claim_offsets + resource) as usize] = low;
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
    let start = input.claims[(input.claim_offsets + resource) as usize];
    let end = input.claims[(input.claim_offsets + resource + 1) as usize];
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

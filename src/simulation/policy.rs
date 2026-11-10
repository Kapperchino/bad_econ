pub const MATCHING_ROUNDS: usize = 4;
pub const DISCOVERY_CANDIDATES: u32 = 8;
pub const HOUSEHOLD_SEARCH_SAMPLES: u32 = 128;
pub const COMMUTE_LEGS: u32 = 2;
pub const POLICY_SCALE: u32 = 1000;
pub const PRICE_SCALE: u32 = 1_000_000;
pub const PRICE_INDEX_BITS: u32 = 5;
pub const PRICE_INDEX_LIMIT: u32 = 1 << PRICE_INDEX_BITS;
pub const MINIMUM_WAGE: u32 = 1;
pub const MAXIMUM_WAGE: u32 = 10_000;
use super::words::*;
use cubecl::prelude::*;

#[derive(CubeType, Clone, Copy)]
pub struct KernelStarts {
    pub coverage: u32,
    pub demand: Ticks,
    pub finished: Ticks,
    pub pipeline: Ticks,
}

#[cube]
pub fn proposed_starts(request: KernelStarts, control: &[Atomic<u32>]) -> Ticks {
    let target = times(request.demand, request.coverage);
    let available = plus(request.finished, request.pipeline);
    target.faults.union(available.faults).record(control);
    positive(target.value, available.value)
}

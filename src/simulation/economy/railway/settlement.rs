use super::*;
use crate::simulation::DeviceEconomy;

use crate::domain::railway::{
    FREIGHT_ACCOUNT_WORDS as ACCOUNT_WORDS, FREIGHT_BOOKING_WORDS, FREIGHT_CLAIM_WORDS,
    FREIGHT_DISPATCH_WORDS, FREIGHT_PROGRESS_WORDS, FREIGHT_QUEUE_WORDS,
    FREIGHT_RESOURCE_WORDS as RESOURCE_WORDS,
};
const CLAIM_COUNT: usize = 2;
const CLAIM_CURSOR: usize = 4;
const RESOURCE_CLAIMS: usize = 10;
const REQUESTED: usize = 2;
const CONSUMED: usize = 4;
const PREFERRED: usize = 6;
const REQUEST_OVERFLOW: usize = 8;
const PREFERRED_OVERFLOW: usize = 9;
const DENIED: usize = 8;
const RESOURCE_DISPATCH: usize = 0;
const INITIALIZE_DISPATCH: usize = 3;
const BOOKING_DISPATCH: usize = 6;
const BLOCK_SORT_DISPATCH: usize = 9;
const MERGE_SORT_DISPATCH: usize = 21;
const RETRY_RESOURCE_DISPATCH: usize = 12;
const CLAIM_ACTIVE_DISPATCH: usize = 15;
const CLAIM_RESOURCE_DISPATCH: usize = 18;

mod booking;
mod capacity;
mod claims;
mod dispatch;
mod storage;

#[cfg(test)]
mod tests;

use booking::{
    active_bookings, booking_buyer, booking_priority, commit, initialize_booking, propose,
    quantity, settle_account,
};
use capacity::{
    RegionCursor, atomic_add, atomic_add_overflows, atomic_store, atomic_ticks, budget, claim,
    consume_resource, fits, release_resource, reset_resource,
};
#[cfg(test)]
use capacity::{fits_exhaustive, fits_tracked, fits_unconstrained, visits};
#[cfg(test)]
pub use dispatch::launch;
pub use dispatch::{ResourceSelection, launch_buffered};
use dispatch::{SettleFreight, admit_claims, resource};
#[cfg(test)]
pub use storage::prepare_claims;
pub use storage::{
    Buffers, ClaimPreparation, ClaimStorage, ClaimWork, WaveBookings, prepare_market_claims,
    reserve_active_market, reserve_claims_for_replay, reserve_market_claims,
};
#[cfg(test)]
use storage::{ClaimAllocation, claim_work};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Stage {
    Snapshot,
    Initialize,
    Queue,
    Reset,
    Propose,
    CountClaims,
    AdmitClaims,
    EmitClaims,
    SortClaims,
    #[cfg_attr(not(test), expect(dead_code))]
    ClaimOffsets,
    Resolve,
    Commit,
    Release,
    Accounts,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Execution {
    Immediate(Wave),
    Buffered(Wave),
}

impl Execution {
    pub fn wave(self) -> Wave {
        match self {
            Self::Immediate(wave) | Self::Buffered(wave) => wave,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum Wave {
    Initial,
    Retry,
}

impl Wave {
    pub fn next(self, claims: ClaimWork) -> Option<Self> {
        match (self, claims) {
            (Self::Initial, ClaimWork::Ready) => Some(Self::Retry),
            _ => None,
        }
    }
}

#[cube]
fn execution_enabled(
    progress: &[Atomic<u32>],
    control: &[Atomic<u32>],
    #[comptime] execution: Execution,
) -> bool {
    match comptime!(execution) {
        Execution::Immediate(_) => true,
        Execution::Buffered(Wave::Initial) => {
            control[crate::simulation::flags::FAULT_WORD].load() == 0
        }
        Execution::Buffered(Wave::Retry) => {
            progress[comptime!(Wave::Retry as usize)].load() > 0
                && control[crate::simulation::flags::FAULT_WORD].load() == 0
        }
    }
}

#[derive(Clone, Copy)]
#[repr(u32)]
enum BookingState {
    Pending,
    Ready,
    Complete,
    Retrying,
}

#[derive(CubeType, CubeLaunch)]
struct Input {
    state: Box<[u32]>,
    actors: Box<[u32]>,
    spatial: Box<[u32]>,
    config: EconomyConfig,
    proposals: Box<[u32]>,
    scratch: Box<[u32]>,
    control: Box<[Atomic<u32>]>,
    resources: Box<[Atomic<u32>]>,
    accounts: Box<[Atomic<u32>]>,
    bookings: Box<[u32]>,
    queue: Box<[u32]>,
    claims: Box<[u32]>,
    claim_source: u32,
    claim_offsets: u32,
    progress: Box<[Atomic<u32>]>,
    parameters: EconomyParameters,
    #[cube(comptime)]
    stage: Stage,
    #[cube(comptime)]
    resource_selection: ResourceSelection,
    #[cube(comptime)]
    execution: Execution,
}

#[cube]
impl GpuPass<Input> for SettleFreight {
    type Output = ();

    fn execute(mut input: Input) {
        input.parameters.refresh(&mut input.config);

        if execution_enabled(&input.progress, &input.control, comptime!(input.execution)) {
            settle_freight(&mut input, ABSOLUTE_POS as u32);
        }
    }
}

#[cube]
fn settle_freight(input: &mut Input, invocation: u32) {
    match comptime!(input.stage) {
        Stage::Snapshot => {
            let id = resource(&*input, invocation);
            let actors = input.config.actors;
            let base = id as usize * RESOURCE_WORDS;
            if invocation == 0 {
                input.progress[comptime!(Wave::Initial as usize)].store(0);
                input.progress[comptime!(Wave::Retry as usize)].store(0);
            }
            if id < actors {
                atomic_store(
                    &input.resources,
                    base,
                    Ledger::new(&mut input.state, &input.config).value(Field::Cash, id),
                );
            }
        }
        Stage::Initialize => {
            let id = resource(&*input, invocation);
            let actors = input.config.actors;
            let resources = actors + input.config.railway.region_count;
            let base = id as usize * RESOURCE_WORDS;
            reset_resource(&*input, id);
            if id < actors {
                let cash = Ledger::new(&mut input.state, &input.config).value(Field::Cash, id);
                let available = minimum(cash, budget(&*input, id));
                atomic_store(&input.resources, base, available);
                for field in 0..ACCOUNT_WORDS {
                    input.accounts[id as usize * ACCOUNT_WORDS + field].store(0);
                }
            }
            initialize_booking(input, invocation);
            if id >= actors && id < resources {
                let stock = regional_value(
                    &Ledger::new(&mut input.state, &input.config),
                    id - actors,
                    comptime!(RailCapacityField::Available),
                );
                atomic_store(&input.resources, base, stock);
            }
        }
        Stage::Queue | Stage::SortClaims => (),
        Stage::Reset => reset_resource(&*input, resource(&*input, invocation)),
        Stage::Propose => propose(input, invocation),
        Stage::CountClaims => {
            let id = resource(&*input, invocation);
            let resources = input.config.actors + input.config.railway.region_count;
            let base = id as usize * RESOURCE_WORDS;
            if id < resources {
                let count = select(
                    claims::contested(&*input, id),
                    input.resources[base + RESOURCE_CLAIMS].load(),
                    0,
                );
                atomic_add(
                    &input.progress,
                    CLAIM_COUNT,
                    ticks(count, 0),
                    &input.control,
                );
            }
        }
        Stage::AdmitClaims => (),
        Stage::EmitClaims => claims::emit(input, invocation),
        Stage::ClaimOffsets => claims::offset(input, invocation),
        Stage::Resolve => {
            let id = resource(&*input, invocation);
            let resources = input.config.actors + input.config.railway.region_count;
            if id < resources {
                claims::resolve(input, id);
            }
        }
        Stage::Commit => commit(input, invocation),
        Stage::Release => {
            let id = resource(&*input, invocation);
            release_resource(input, id);
        }
        Stage::Accounts => {
            let id = resource(&*input, invocation);
            settle_account(input, id);
        }
    }
}

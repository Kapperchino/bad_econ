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

use capacity::{AtomicBalance, CapacityRequest, RegionCursor};
#[cfg(test)]
pub use dispatch::launch;
pub use dispatch::{ResourceSelection, launch_buffered};
use dispatch::{SettleFreight, admit_claims};
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
struct FreightSettlementInput {
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
impl GpuPass<FreightSettlementInput> for SettleFreight {
    type Output = ();

    fn execute(mut input: FreightSettlementInput) {
        input.parameters.refresh(&mut input.config);

        if execution_enabled(&input.progress, &input.control, comptime!(input.execution)) {
            input.settle_freight(ABSOLUTE_POS as u32);
        }
    }
}

#[cube]
impl FreightSettlementInput {
    fn settle_freight(&mut self, invocation: u32) {
        match comptime!(self.stage) {
            Stage::Snapshot => {
                let id = self.resource(invocation);
                let actors = self.config.actors;
                let base = id as usize * RESOURCE_WORDS;
                if invocation == 0 {
                    self.progress[comptime!(Wave::Initial as usize)].store(0);
                    self.progress[comptime!(Wave::Retry as usize)].store(0);
                }
                if id < actors {
                    AtomicBalance::at(&self.resources, base)
                        .store(Ledger::new(&mut self.state, &self.config).value(Field::Cash, id));
                }
            }
            Stage::Initialize => {
                let id = self.resource(invocation);
                let actors = self.config.actors;
                let resources = actors + self.config.railway.region_count;
                let base = id as usize * RESOURCE_WORDS;
                self.reset_resource(id);
                if id < actors {
                    let cash = Ledger::new(&mut self.state, &self.config).value(Field::Cash, id);
                    let available = minimum(cash, self.budget(id));
                    AtomicBalance::at(&self.resources, base).store(available);
                    for field in 0..ACCOUNT_WORDS {
                        self.accounts[id as usize * ACCOUNT_WORDS + field].store(0);
                    }
                }
                self.initialize_booking(invocation);
                if id >= actors && id < resources {
                    let capacity = RegionalCapacity::new(&self.config.railway, id - actors);
                    let stock = capacity.value(
                        &Ledger::new(&mut self.state, &self.config),
                        comptime!(RailCapacityField::Available),
                    );
                    AtomicBalance::at(&self.resources, base).store(stock);
                }
            }
            Stage::Queue | Stage::SortClaims => (),
            Stage::Reset => self.reset_resource(self.resource(invocation)),
            Stage::Propose | Stage::EmitClaims | Stage::Commit => {
                self.process_booking(invocation);
            }
            Stage::CountClaims => {
                let id = self.resource(invocation);
                let resources = self.config.actors + self.config.railway.region_count;
                let base = id as usize * RESOURCE_WORDS;
                if id < resources {
                    let count = select(
                        self.contested(id),
                        self.resources[base + RESOURCE_CLAIMS].load(),
                        0,
                    );
                    AtomicBalance::at(&self.progress, CLAIM_COUNT)
                        .add(ticks(count, 0), &self.control);
                }
            }
            Stage::AdmitClaims => (),
            Stage::ClaimOffsets => self.store_claim_offset(invocation),
            Stage::Resolve => {
                let id = self.resource(invocation);
                let resources = self.config.actors + self.config.railway.region_count;
                if id < resources {
                    self.resolve_claims(id);
                }
            }
            Stage::Release => {
                let id = self.resource(invocation);
                self.release_resource(id);
            }
            Stage::Accounts => {
                let id = self.resource(invocation);
                self.settle_account(id);
            }
        }
    }
}

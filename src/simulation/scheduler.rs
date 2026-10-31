use super::{
    flags::{
        COMPLETED_WORD, CONTROL_WORDS, DependencyFlag, EventStatus, FaultFlags,
        KernelAuthorization, KernelEventStatus, KernelFaults, PROGRESS_WORD, record_dependency,
    },
    layout::*,
    parameters::SchedulerParameters,
    words::*,
};
use crate::ledger::{EMPTY, LEG_WORDS, LegLayout, MAX_LEGS};
use cubecl::prelude::*;

#[enumflags2::bitflags]
#[repr(u32)]
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum SchedulerStage {
    Initialize,
    Merge,
    Dependencies,
    CheckChains,
    Chains,
    Clear,
    Frontier,
}

#[cube]
fn coordinate(events: &[u32], incidence: u32) -> u32 {
    events[incidence as usize * LEG_WORDS + LegLayout::COORDINATE]
}
#[cube]
fn before(events: &[u32], a: u32, b: u32) -> bool {
    let ca = coordinate(events, a);
    let cb = coordinate(events, b);
    ca < cb || (ca == cb && a < b)
}

#[allow(
    clippy::collapsible_if,
    reason = "Guard the storage read before indexing a partial workgroup"
)]
#[cube(launch)]
pub fn schedule(
    balances: &mut [u32],
    events: &[u32],
    scratch: &mut [u32],
    control: &[Atomic<u32>],
    links: &[Atomic<u32>],
    params: SchedulerParameters,
    #[comptime] stage: SchedulerStage,
) {
    let id = ABSOLUTE_POS as u32;
    let count = params.count;
    let total = count * MAX_LEGS as u32;
    match stage {
        SchedulerStage::Initialize => {
            if id < total {
                scratch[id as usize] = id;
                scratch[(total * ScratchLayout::PREDECESSORS as u32 + id) as usize] =
                    NO_PREDECESSOR;
            }
            if id < count {
                let mut status = KernelEventStatus::new(comptime!(EventStatus::Inactive));
                if events[id as usize * EVENT_WORDS + LegLayout::COORDINATE] != EMPTY {
                    status.set(comptime!(EventStatus::Pending));
                }
                status.store(scratch, total * ScratchLayout::STATUSES as u32 + id);
                links[LinkLayout::FIRST_PREDECESSOR * count as usize + id as usize]
                    .store(NO_PREDECESSOR);
                links[LinkLayout::LAST_PREDECESSOR * count as usize + id as usize].store(0);
                links[LinkLayout::SUCCESSOR * count as usize + id as usize].store(NO_SUCCESSOR);
            }
            if id < CONTROL_WORDS as u32 {
                control[id as usize].store(0);
            }
        }
        SchedulerStage::Merge => {
            if id < total {
                let width = params.width;
                let source = params.source;
                let destination = total - source;
                let base = id / (width * 2) * width * 2;
                let middle = min(base + width, total);
                let end = min(base + width * 2, total);
                let first = id < middle;
                let start = select(first, middle, base);
                let stop = select(first, end, middle);
                let item = scratch[(source + id) as usize];
                let mut low = start;
                let mut high = stop;
                while low < high {
                    let probe = (low + high) / 2;
                    let precedes = before(events, scratch[(source + probe) as usize], item);
                    low = select(precedes, probe + 1, low);
                    high = select(precedes, high, probe);
                }
                let own = id - select(first, base, middle);
                scratch[(destination + base + own + low - start) as usize] = item;
            }
        }
        SchedulerStage::Dependencies => {
            if id > 0 && id < total {
                let previous = scratch[(params.source + id - 1) as usize];
                let current = scratch[(params.source + id) as usize];
                if coordinate(events, current) != EMPTY
                    && coordinate(events, previous) == coordinate(events, current)
                {
                    scratch[(total * ScratchLayout::PREDECESSORS as u32 + current) as usize] =
                        previous / MAX_LEGS as u32;
                    let prior = previous / MAX_LEGS as u32;
                    let event = current / MAX_LEGS as u32;
                    links[LinkLayout::FIRST_PREDECESSOR * count as usize + event as usize]
                        .fetch_min(prior);
                    links[LinkLayout::LAST_PREDECESSOR * count as usize + event as usize]
                        .fetch_max(prior);
                    let old = links[LinkLayout::SUCCESSOR * count as usize + prior as usize]
                        .exchange(event);
                    if old != NO_SUCCESSOR && old != event {
                        record_dependency(control, comptime!(DependencyFlag::MultipleSuccessors));
                    }
                }
            }
        }
        SchedulerStage::CheckChains => {
            if id < count {
                let low =
                    links[LinkLayout::FIRST_PREDECESSOR * count as usize + id as usize].load();
                let high =
                    links[LinkLayout::LAST_PREDECESSOR * count as usize + id as usize].load();
                if low != NO_PREDECESSOR && low != high {
                    record_dependency(control, comptime!(DependencyFlag::MultiplePredecessors));
                }
            }
        }
        SchedulerStage::Chains => {
            if id < count {
                let status =
                    KernelEventStatus::load(scratch, total * ScratchLayout::STATUSES as u32 + id);
                if links[LinkLayout::FIRST_PREDECESSOR * count as usize + id as usize].load()
                    == NO_PREDECESSOR
                    && status.contains(comptime!(EventStatus::Pending))
                {
                    let mut event = id;
                    let mut depth = 0u32;
                    while event != NO_SUCCESSOR {
                        post(balances, events, control, event)
                            .store(scratch, total * ScratchLayout::STATUSES as u32 + event);
                        event =
                            links[LinkLayout::SUCCESSOR * count as usize + event as usize].load();
                        depth += 1;
                    }
                    control[PROGRESS_WORD].fetch_max(depth);
                }
            }
        }
        SchedulerStage::Clear => {
            if id == 0 {
                control[PROGRESS_WORD].store(0);
            }
        }
        SchedulerStage::Frontier => {
            if id < count {
                let old = total * ScratchLayout::STATUSES as u32 + params.status_offset;
                let next = total * ScratchLayout::STATUSES as u32 + count - params.status_offset;
                let mut status = KernelEventStatus::load(scratch, old + id);
                if status.contains(comptime!(EventStatus::Pending)) {
                    let mut ready = true;
                    for slot in 0..MAX_LEGS as u32 {
                        let predecessor = scratch[(total * ScratchLayout::PREDECESSORS as u32
                            + id * MAX_LEGS as u32
                            + slot) as usize];
                        if predecessor != NO_PREDECESSOR {
                            let prior = KernelEventStatus::load(scratch, old + predecessor);
                            ready = ready && !prior.contains(comptime!(EventStatus::Pending));
                        }
                    }
                    if ready {
                        status.replace(post(balances, events, control, id));
                    }
                }
                status.store(scratch, next + id);
                if status.contains(comptime!(EventStatus::Pending)) {
                    control[PROGRESS_WORD].fetch_add(1);
                }
            }
        }
    }
}

#[cube]
fn post(
    balances: &mut [u32],
    events: &[u32],
    control: &[Atomic<u32>],
    id: u32,
) -> KernelEventStatus {
    control[COMPLETED_WORD].fetch_add(1);
    let mut valid = true;
    let mut faults = KernelFaults::none();
    let mut after = Array::<u32>::new(MAX_LEGS * TICK_WORDS);
    for slot in 0..MAX_LEGS as u32 {
        let base = (id * MAX_LEGS as u32 + slot) * LEG_WORDS as u32;
        let cell = events[base as usize + LegLayout::COORDINATE];
        if cell != EMPTY {
            if cell < balances.len() as u32 / TICK_WORDS as u32 {
                let balance = load(balances, cell);
                let debit = ticks(
                    events[base as usize + LegLayout::DEBIT_LOW],
                    events[base as usize + LegLayout::DEBIT_HIGH],
                );
                let credit = ticks(
                    events[base as usize + LegLayout::CREDIT_LOW],
                    events[base as usize + LegLayout::CREDIT_HIGH],
                );
                let guard = ticks(
                    events[base as usize + LegLayout::GUARD_LOW],
                    events[base as usize + LegLayout::GUARD_HIGH],
                );
                let funded = !lt(balance, debit);
                valid = valid && funded;
                let authorization =
                    KernelAuthorization::load(events[base as usize + LegLayout::AUTHORIZATION]);
                valid = valid && authorization.permits(balance, guard);
                if funded {
                    let proposed = plus(minus(balance, debit), credit);
                    after[(slot * TICK_WORDS as u32) as usize] = proposed.value.extract(LOW_WORD);
                    after[(slot * TICK_WORDS as u32 + 1) as usize] =
                        proposed.value.extract(HIGH_WORD);
                    faults.insert(proposed.faults);
                }
            } else {
                faults.insert(KernelFaults::new(comptime!(FaultFlags::INVALID_COORDINATE)));
            }
        }
    }
    let mut status = KernelEventStatus::new(comptime!(EventStatus::Rejected));
    if valid {
        status.set(comptime!(EventStatus::Accepted));
    }
    if !faults.is_empty() {
        status.set(comptime!(EventStatus::Faulted));
        faults.record(control);
    }
    if status.contains(comptime!(EventStatus::Accepted)) {
        for slot in 0..MAX_LEGS as u32 {
            let cell = events[((id * MAX_LEGS as u32 + slot) * LEG_WORDS as u32) as usize];
            if cell != EMPTY {
                store(
                    balances,
                    cell,
                    ticks(
                        after[(slot * TICK_WORDS as u32) as usize],
                        after[(slot * TICK_WORDS as u32 + 1) as usize],
                    ),
                );
            }
        }
    }
    status
}

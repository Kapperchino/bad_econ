use super::words::*;
use cubecl::prelude::*;

#[derive(CubeType, Clone, Copy, Debug, Hash, PartialEq, Eq)]
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
    events[(incidence * 8) as usize]
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
    params: &[u32],
    #[comptime] stage: SchedulerStage,
) {
    let id = ABSOLUTE_POS as u32;
    let count = params[0];
    let total = count * 12;
    match stage {
        SchedulerStage::Initialize => {
            if id < total {
                scratch[id as usize] = id;
                scratch[(total * 2 + id) as usize] = u32::MAX;
            }
            if id < count {
                scratch[(total * 3 + id) as usize] =
                    u32::cast_from(events[(id * 96) as usize] != u32::MAX);
                links[id as usize].store(u32::MAX);
                links[(count + id) as usize].store(0);
                links[(count * 2 + id) as usize].store(u32::MAX);
            }
            if id < 4 {
                control[id as usize].store(0);
            }
        }
        SchedulerStage::Merge => {
            if id < total {
                let width = params[1];
                let source = params[2];
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
                let previous = scratch[(params[2] + id - 1) as usize];
                let current = scratch[(params[2] + id) as usize];
                if coordinate(events, current) != u32::MAX
                    && coordinate(events, previous) == coordinate(events, current)
                {
                    scratch[(total * 2 + current) as usize] = previous / 12;
                    let prior = previous / 12;
                    let event = current / 12;
                    links[event as usize].fetch_min(prior);
                    links[(count + event) as usize].fetch_max(prior);
                    let old = links[(count * 2 + prior) as usize].exchange(event);
                    if old != u32::MAX && old != event {
                        control[3].store(1);
                    }
                }
            }
        }
        SchedulerStage::CheckChains => {
            if id < count {
                let low = links[id as usize].load();
                let high = links[(count + id) as usize].load();
                if low != u32::MAX && low != high {
                    control[3].store(1);
                }
            }
        }
        SchedulerStage::Chains => {
            if id < count {
                if links[id as usize].load() == u32::MAX && scratch[(total * 3 + id) as usize] == 1
                {
                    let mut event = id;
                    let mut depth = 0u32;
                    while event != u32::MAX {
                        scratch[(total * 3 + event) as usize] =
                            post(balances, events, control, event);
                        event = links[(count * 2 + event) as usize].load();
                        depth += 1;
                    }
                    control[0].fetch_max(depth);
                }
            }
        }
        SchedulerStage::Clear => {
            if id == 0 {
                control[0].store(0);
            }
        }
        SchedulerStage::Frontier => {
            if id < count {
                let old = total * 3 + params[3];
                let next = total * 3 + count - params[3];
                let prior = scratch[(old + id) as usize];
                let mut status = prior;
                if prior == 1 {
                    let mut ready = true;
                    for slot in 0..12u32 {
                        let predecessor = scratch[(total * 2 + id * 12 + slot) as usize];
                        if predecessor != u32::MAX {
                            ready = ready && scratch[(old + predecessor) as usize] != 1;
                        }
                    }
                    if ready {
                        status = post(balances, events, control, id);
                    }
                }
                scratch[(next + id) as usize] = status;
                if status == 1 {
                    control[0].fetch_add(1);
                }
            }
        }
    }
}

#[cube]
fn post(balances: &mut [u32], events: &[u32], control: &[Atomic<u32>], id: u32) -> u32 {
    control[2].fetch_add(1);
    let mut valid = true;
    let mut fault = 0u32;
    let mut after = Array::<u32>::new(24usize);
    for slot in 0..12u32 {
        let base = (id * 12 + slot) * 8;
        let cell = events[base as usize];
        if cell != u32::MAX {
            match u32::cast_from(cell < balances.len() as u32 / 2) {
                1 => {
                    let balance = load(balances, cell);
                    let debit = ticks(events[(base + 1) as usize], events[(base + 2) as usize]);
                    let credit = ticks(events[(base + 3) as usize], events[(base + 4) as usize]);
                    let guard = ticks(events[(base + 6) as usize], events[(base + 7) as usize]);
                    let funded = !lt(balance, debit);
                    valid = valid && funded;
                    match events[(base + 5) as usize] {
                        1 => valid = valid && equal(balance, guard),
                        2 => valid = valid && !lt(balance, guard),
                        _ => (),
                    }
                    if funded {
                        let proposed = plus(minus(balance, debit), credit);
                        after[(slot * 2) as usize] = proposed.value.extract(0usize);
                        after[(slot * 2 + 1) as usize] = proposed.value.extract(1usize);
                        fault |= proposed.overflow;
                    }
                }
                _ => fault = 2,
            }
        }
    }
    let mut status = select(valid, 2u32, 3u32);
    if fault > 0 {
        status = 4;
        control[1].fetch_max(fault);
    }
    if status == 2 {
        for slot in 0..12u32 {
            let cell = events[((id * 12 + slot) * 8) as usize];
            if cell != u32::MAX {
                store(
                    balances,
                    cell,
                    ticks(after[(slot * 2) as usize], after[(slot * 2 + 1) as usize]),
                );
            }
        }
    }
    status
}

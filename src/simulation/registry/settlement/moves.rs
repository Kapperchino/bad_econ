use super::super::{PendingRecord, move_actor};
use crate::domain::Field;
use crate::simulation::{
    Device, DeviceEconomy,
    economy::configuration::EconomyConfig,
    storage::{
        flags::{FaultFlags, KernelFaults},
        state::{EconomyView, ReadLedger, ReadLedgerExpand},
        words::*,
    },
    system::AsBufferArg,
};
use burn_wgpu::CubeTensor;
use cubecl::prelude::*;
use std::collections::{BTreeMap, BTreeSet, HashMap};

#[repr(C)]
#[derive(Default, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MoveAccount {
    actor: u32,
    earlier: u32,
    later: u32,
}

impl MoveAccount {
    const WORDS: usize = size_of::<Self>() / size_of::<u32>();
}

pub(super) struct IndependentMoves {
    accounts: Option<CubeTensor>,
}

impl IndependentMoves {
    pub fn new(
        device: &Device,
        requests: &[PendingRecord],
        settlement: &mut BTreeSet<usize>,
    ) -> Self {
        let mut neighbors = HashMap::<usize, Vec<usize>>::new();
        for movement in requests.iter().filter(|record| record.request >> 24 == 2) {
            let from = movement.actor as usize;
            let to = movement.target();
            neighbors.entry(from).or_default().push(to);
            neighbors.entry(to).or_default().push(from);
        }
        let mut connected = settlement.iter().copied().collect::<Vec<_>>();
        while let Some(actor) = connected.pop() {
            for &other in neighbors.get(&actor).into_iter().flatten() {
                if settlement.insert(other) {
                    connected.push(other);
                }
            }
        }

        let mut accounts = BTreeMap::<u32, MoveAccount>::new();
        for record in requests.iter().filter(|record| {
            matches!(record.request >> 24, 2 | 3) && !settlement.contains(&(record.actor as usize))
        }) {
            accounts.entry(record.actor).or_default().actor = record.actor;
            if record.request >> 24 == 2 {
                let target = record.target() as u32;
                let receiver = accounts.entry(target).or_default();
                receiver.actor = target;
                match record.actor < target {
                    true => receiver.earlier += 1,
                    false => receiver.later += 1,
                }
            }
        }
        let accounts = accounts.into_values().collect::<Vec<_>>();
        Self {
            accounts: (!accounts.is_empty()).then(|| device.record_tensor(&accounts)),
        }
    }

    pub fn apply(
        &self,
        gpu: DeviceEconomy<'_>,
        requests: &CubeTensor,
        control: &CubeTensor,
        day: u32,
    ) {
        if let Some(accounts) = &self.accounts {
            post_moves::launch(
                &gpu.device.client,
                gpu.device
                    .count(accounts.meta.num_elements() / MoveAccount::WORDS),
                CubeDim::new_1d(gpu.device.workgroup),
                gpu.state.buffer_arg(),
                gpu.actors.buffer_arg(),
                gpu.config.launch(),
                requests.buffer_arg(),
                accounts.buffer_arg(),
                control.buffer_arg(),
                day,
            );
        }
    }
}

#[cube(launch)]
fn post_moves(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    requests: &[u32],
    accounts: &[u32],
    control: &[Atomic<u32>],
    day: u32,
) {
    let base = ABSOLUTE_POS * comptime!(MoveAccount::WORDS);
    if base < accounts.len() {
        let actor = accounts[base];
        let earlier = accounts[base + 1];
        let later = accounts[base + 2];
        let mut economy = EconomyView::new(state, actors, config);
        let cash = economy.ledger.value(Field::Cash, actor);
        let available = plus(cash, times32(earlier, 100)).record(control);
        let cost = ticks(
            select(requests[actor as usize] >> 24u32 == 2u32, 100u32, 0u32),
            0,
        );
        if lt(available, cost) {
            KernelFaults::new(comptime!(FaultFlags::ARITHMETIC_OVERFLOW)).record(control);
        }
        let remaining = minus(available, cost);
        let total = plus(remaining, times32(later, 100)).record(control);
        economy.ledger.set(Field::Cash, actor, total);
        move_actor(&mut economy, requests, actor, day);
    }
}

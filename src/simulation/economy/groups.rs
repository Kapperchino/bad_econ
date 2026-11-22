use super::*;

#[cube]
pub(super) fn group_initialize(
    proposals: &mut [u32],
    scratch: &mut [u32],
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
    id: u32,
) {
    let count = parameters.count;
    let mut peer = u32::new(comptime!(i64::from(NO_ACTOR)));
    if id < count {
        peer = proposals
            [(parameters.offset + id) as usize * ProposalLayout::WORDS + ProposalLayout::ACTOR];
        KernelEventStatus::new(comptime!(EventStatus::Inactive)).store(scratch, 4 * count + id);
    }
    let prefix = parallel::prefix_sum::<u32>(u32::cast_from(peer != NO_ACTOR));
    let mut first = Shared::<u32>::new();
    if UNIT_POS == 0 {
        *first = control[CONSTRUCTED_EVENTS_WORD].fetch_add(prefix.total);
    }
    sync_cube();
    if peer != NO_ACTOR {
        let slot = *first + prefix.inclusive - 1;
        scratch[slot as usize] = peer;
        scratch[(count + slot) as usize] = id;
    }
}

#[cube]
pub(super) fn group_sort(
    scratch: &mut [u32],
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
    id: u32,
) {
    let count = parameters.count;
    let active = markets::active(control, parameters);
    let lane = UNIT_POS as usize;
    let mut keys = Shared::<[u32]>::new_slice(256usize);
    let mut items = Shared::<[u32]>::new_slice(256usize);
    let mut key = u32::new(comptime!(i64::from(NO_ACTOR)));
    let mut item = 0u32;
    if id < active {
        key = scratch[id as usize];
        item = scratch[(count + id) as usize];
    }
    keys[lane] = key;
    items[lane] = item;
    sync_cube();
    let rotation = parameters.settlement_rotation % max(count, 1);
    let mut width = 2u32;
    while width <= CUBE_DIM {
        let mut stride = width / 2;
        while stride > 0 {
            let other = (UNIT_POS ^ stride) as usize;
            let other_key = keys[other];
            let other_item = items[other];
            let priority = (item + count - rotation) % max(count, 1);
            let other_priority = (other_item + count - rotation) % max(count, 1);
            let precedes = other_key < key || (other_key == key && other_priority < priority);
            let lower = (UNIT_POS & width == 0) == (UNIT_POS & stride == 0);
            if precedes == lower {
                key = other_key;
                item = other_item;
            }
            sync_cube();
            keys[lane] = key;
            items[lane] = item;
            sync_cube();
            stride /= 2;
        }
        width *= 2;
    }
    if id < active {
        scratch[id as usize] = key;
        scratch[(count + id) as usize] = item;
    }
}

#[cube]
pub(super) fn group_merge(
    scratch: &mut [u32],
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
    id: u32,
) {
    let count = parameters.count;
    let active = markets::active(control, parameters);
    if id < active {
        let width = parameters.sort_width;
        let source = parameters.sort_source;
        let destination = 2 * count - source;
        let base = id / (width * 2) * width * 2;
        let middle = min(base + width, active);
        let end = min(base + width * 2, active);
        let first = id < middle;
        let start = select(first, middle, base);
        let stop = select(first, end, middle);
        let key = scratch[(source + id) as usize];
        let item = scratch[(source + count + id) as usize];
        let mut low = start;
        let mut high = stop;
        while low < high {
            let probe = (low + high) / 2;
            let other_key = scratch[(source + probe) as usize];
            let other_item = scratch[(source + count + probe) as usize];
            let rotation = parameters.settlement_rotation % max(count, 1);
            let priority = (item + count - rotation) % max(count, 1);
            let other_priority = (other_item + count - rotation) % max(count, 1);
            let precedes = other_key < key || (other_key == key && other_priority < priority);
            low = select(precedes, probe + 1, low);
            high = select(precedes, high, probe);
        }
        let target = destination + base + id - select(first, base, middle) + low - start;
        scratch[target as usize] = key;
        scratch[(target + count) as usize] = item;
    }
}

#[cube]
pub(super) fn group_execute(
    economy: &mut EconomyView<'_>,
    proposals: &mut [u32],
    scratch: &mut [u32],
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
    id: u32,
) {
    let range = id;
    let count = parameters.count;
    let active_count = markets::active(control, parameters);
    let mut invocation = id;
    if comptime!(parameters.discovery == Discovery::Indexed) {
        invocation = active_count;
        if id < control[RANGE_SLOTS_WORD].load() {
            invocation = scratch[(5 * count + id) as usize];
        }
    }
    let id = invocation;
    if id < active_count {
        let source = parameters.sort_source;
        let key = scratch[(source + id) as usize];
        let mut first = id == 0;
        if id > 0 {
            first = scratch[(source + id - 1) as usize] != key;
        }
        if first {
            let low = id;
            let mut cursor = low;
            let mut active = true;
            let mut event = Array::<u32>::new(EVENT_WORDS);
            let mut peer = PeerBalances::new(economy, parameters, key);
            let deferred = comptime!(
                parameters.market == MarketFlag::Household
                    && parameters.discovery == Discovery::Indexed
            );
            if deferred {
                active = peer.has_supply();
            }
            while cursor < active_count && active {
                active = scratch[(source + cursor) as usize] == key;
                if active {
                    let actor = scratch[(source + count + cursor) as usize];
                    if comptime!(
                        parameters.market == MarketFlag::Payroll
                            || parameters.market == MarketFlag::Distribution
                    ) {
                        construct(
                            economy,
                            proposals,
                            &mut event[..],
                            control,
                            parameters,
                            actor,
                            comptime!(EventStorage::Local),
                        );
                        apply(economy.ledger.words, &event[..], control, 0)
                            .store(scratch, 4 * count + actor);
                    } else {
                        let prior = KernelEventStatus::load(scratch, 4 * count + actor);
                        peer.offer(
                            &mut economy.ledger,
                            proposals,
                            control,
                            parameters,
                            PeerOffer {
                                actor: actor + parameters.offset,
                                peer: key,
                                status: prior,
                            },
                        )
                        .store(scratch, 4 * count + actor);
                    }
                    cursor += 1;
                    if deferred {
                        active = peer.has_supply();
                    }
                }
            }
            if deferred {
                scratch[(7 * count + key) as usize] = cursor;
                cursor = low + scratch[(6 * count + range) as usize];
            }
            control[PROGRESS_WORD].fetch_max(cursor - low);
            control[COMPLETED_WORD].fetch_add(cursor - low);
            peer.store(economy, parameters, key);
        }
    }
}

#[cube]
pub(super) fn group_remainder(
    economy: &mut EconomyView<'_>,
    proposals: &mut [u32],
    scratch: &mut [u32],
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
    id: u32,
) {
    let count = parameters.count;
    if id < markets::active(control, parameters) {
        let key = scratch[(parameters.sort_source + id) as usize];
        if id >= scratch[(7 * count + key) as usize] {
            let actor = scratch[(parameters.sort_source + count + id) as usize];
            let prior = KernelEventStatus::load(scratch, 4 * count + actor);
            let mut peer = PeerBalances::new(economy, parameters, key);
            peer.offer(
                &mut economy.ledger,
                proposals,
                control,
                parameters,
                PeerOffer {
                    actor,
                    peer: key,
                    status: prior,
                },
            )
            .store(scratch, 4 * count + actor);
        }
    }
}

#[cube]
pub(super) fn settle_input_tile(
    economy: &mut EconomyView<'_>,
    proposals: &mut [u32],
    scratch: &mut [u32],
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
) {
    let orders = markets::active(control, parameters);
    if control[LATCH_WORD].load() == 0 {
        let mut slot = UNIT_POS;
        while slot < orders {
            actor_settlement(
                economy,
                proposals,
                scratch,
                control,
                parameters,
                slot,
                comptime!(Posting::CheckActor),
            );
            slot += CUBE_DIM;
        }
        sync_storage();
        let mut slot = UNIT_POS;
        while slot < orders {
            group_execute(economy, proposals, scratch, control, parameters, slot);
            slot += CUBE_DIM;
        }
        sync_storage();
        if UNIT_POS == 0 {
            control[LATCH_WORD].store(control[crate::simulation::flags::FAULT_WORD].load());
        }
        sync_storage();
        if control[LATCH_WORD].load() == 0 {
            let mut slot = UNIT_POS;
            while slot < orders {
                actor_settlement(
                    economy,
                    proposals,
                    scratch,
                    control,
                    parameters,
                    slot,
                    comptime!(Posting::ApplyActor),
                );
                slot += CUBE_DIM;
            }
        }
    }
}

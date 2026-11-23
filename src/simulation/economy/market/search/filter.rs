use super::*;
use crate::simulation::economy::market::orders::{
    COMPACTION_BLOCK, WIDE_ORDERS_WORD, filtered_count,
};
use crate::simulation::economy::parameters::EmptyOrderMode;

const PREFIX_BUYERS: u32 = 8192;

#[cube]
pub(super) fn prefix_slot(successes: &[Atomic<u32>], count: u32, peer: u32) -> usize {
    successes.len() - 2 * (count - peer) as usize
}

#[cube]
pub(super) fn sample_order(
    economy: &EconomyView<'_>,
    successes: &[Atomic<u32>],
    parameters: &EconomyParameters,
    actor: u32,
    seller: u32,
    quantity: Ticks,
) {
    let ledger = &economy.ledger;
    let config = ledger.config;
    let position = (actor + config.actors - parameters.settlement_rotation) % config.actors;
    let units = quantity.extract(LOW_WORD);
    if position < comptime!(PREFIX_BUYERS)
        && quantity.extract(HIGH_WORD) == 0
        && units > 0
        && units <= u32::MAX / comptime!(PREFIX_BUYERS)
    {
        let terms = household_production::TradeFields::new(
            &economy.actors,
            config,
            seller,
            parameters.traded_good(),
        );
        let cost = times32(units, ledger.number_at(terms.price, seller));
        if !lt(ledger.value(Field::Cash, actor), cost) {
            let slot = prefix_slot(successes, config.actors, seller);
            successes[slot].fetch_add(units);
            successes[slot + 1].fetch_max(units);
        }
    }
}

#[cube]
pub(super) fn stock_limits(
    ledger: &Ledger<'_>,
    spatial: &[u32],
    scratch: &[u32],
    successes: &[Atomic<u32>],
) {
    let config = ledger.config;
    let member = ABSOLUTE_POS as u32;
    let cells = config.spatial_side * config.spatial_side;
    if member < spatial[cells as usize] {
        let peer = spatial[(cells + 1 + member) as usize];
        let slot = prefix_slot(successes, config.actors, peer);
        let demand = successes[slot].load();
        let largest = successes[slot + 1].load();
        let supply = index::Supply::load(scratch, member);
        let stock = supply.available.extract(LOW_WORD);
        let mut limit = u32::new(comptime!(i64::from(u32::MAX)));
        if demand > 0
            && supply.available.extract(HIGH_WORD) == 0
            && supply.quote <= i32::MAX as u32
            && ledger.value(Field::Cash, peer).extract(HIGH_WORD) == 0
            && ledger.value(Field::Sales, peer).extract(HIGH_WORD) == 0
        {
            limit = max(stock - min(demand, stock), largest - 1);
        }
        successes[slot].store(limit);
        successes[slot + 1].store(0);
    }
}

#[cube]
fn empty_order(
    economy: &EconomyView<'_>,
    proposals: &[u32],
    parameters: &EconomyParameters,
    actor: u32,
    peer: u32,
) -> bool {
    let base = actor as usize * ProposalLayout::WORDS;
    let quantity = ticks(
        proposals[base + ProposalLayout::QUANTITY_LOW],
        proposals[base + ProposalLayout::QUANTITY_HIGH],
    );
    peer != NO_ACTOR
        && !equal(quantity, zero())
        && !economy
            .actors
            .kind(peer)
            .is(comptime!(ActorKind::Household))
        && equal(
            economy.ledger.value_at(parameters.traded_good(), peer),
            zero(),
        )
        && clearing::empty_seller_safe(economy, parameters, peer)
}

#[cube]
pub(super) fn buyers(
    economy: &EconomyView<'_>,
    proposals: &[u32],
    searches: &[u32],
    successes: &[Atomic<u32>],
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
) {
    let config = economy.ledger.config;
    let position = ABSOLUTE_POS as u32;
    let count = config.actors;
    let mut kept = false;
    if position < count {
        let rotated = position + parameters.settlement_rotation;
        let actor = select(rotated < count, rotated, rotated - count);
        let searching =
            (searches[(count + actor) as usize] | searches[(2 * count + actor) as usize]) != 0;
        if searching {
            let peer = proposals[actor as usize * ProposalLayout::WORDS + ProposalLayout::ACTOR];
            kept = peer != NO_ACTOR;
            if kept && control[WIDE_ORDERS_WORD].load() == 0 {
                let slot = prefix_slot(successes, count, peer);
                let limit = successes[slot].load();
                let units = proposals
                    [actor as usize * ProposalLayout::WORDS + ProposalLayout::QUANTITY_LOW];
                let exhausted = position >= comptime!(PREFIX_BUYERS) && units > limit;
                kept = !exhausted && !empty_order(economy, proposals, parameters, actor, peer);
                if !kept {
                    successes[filtered_count(count, peer) as usize].fetch_add(1);
                } else if limit != u32::MAX {
                    successes[slot + 1].fetch_add(1);
                }
            }
        }
    }
    let mask = plane_ballot(kept).extract(0usize);
    let masks = ceildiv(count, 32);
    let mut counts = Shared::<[u32]>::new_slice(8usize);
    if UNIT_POS_PLANE == 0 {
        counts[PLANE_POS as usize] = mask.count_ones();
        if position < count {
            successes[(position / 32) as usize].store(mask);
        }
    }
    sync_cube();
    if UNIT_POS == 0 && position < count {
        let mut total = 0u32;
        #[unroll]
        for plane in 0..8usize {
            total += counts[plane];
        }
        successes[(masks + position / COMPACTION_BLOCK) as usize].store(total);
    }
}

#[cube]
pub(super) fn orders(
    economy: &EconomyView<'_>,
    proposals: &[u32],
    successes: &[Atomic<u32>],
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
) {
    let config = economy.ledger.config;
    if comptime!(parameters.empty_orders == EmptyOrderMode::FeedbackOnly)
        && control[WIDE_ORDERS_WORD].load() == 0
        && parameters.traded_good() != comptime!(Good::Bread as u32)
    {
        let masks = ceildiv(parameters.count, 32);
        let word = ABSOLUTE_POS as u32;
        let mut count = 0u32;
        if word < masks {
            let original = successes[word as usize].load();
            let mut remaining = original;
            let mut kept = original;
            while remaining != 0 {
                let bit = remaining.trailing_zeros();
                let position = word * 32 + bit;
                let actor = (position + parameters.settlement_rotation) % parameters.count;
                let base = actor as usize * ProposalLayout::WORDS;
                let peer = proposals[base + ProposalLayout::ACTOR];
                let filtered = empty_order(economy, proposals, parameters, actor, peer);
                if filtered {
                    kept &= !(1u32 << bit);
                    successes[filtered_count(config.actors, peer) as usize].fetch_add(1);
                }
                remaining &= remaining - 1;
            }
            successes[word as usize].store(kept);
            count = kept.count_ones();
        }
        let mut counts = Shared::<[u32]>::new_slice(256usize);
        counts[UNIT_POS as usize] = count;
        sync_cube();
        if UNIT_POS % 8 == 0 && word < masks {
            let mut total = 0u32;
            for offset in 0..8u32 {
                total += counts[(UNIT_POS + offset) as usize];
            }
            successes[(masks + word / 8) as usize].store(total);
        }
    }
}

#[cube]
pub(super) fn finish(
    config: &EconomyConfig,
    spatial: &[u32],
    successes: &[Atomic<u32>],
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
    #[comptime] collection: OrderCollection,
) {
    if comptime!(parameters.empty_orders == EmptyOrderMode::FeedbackOnly)
        && control[WIDE_ORDERS_WORD].load() == 0
        && control[crate::simulation::storage::flags::FAULT_WORD].load() == 0
    {
        let cells = config.spatial_side * config.spatial_side;
        let member = ABSOLUTE_POS as u32;
        let mut count = 0u32;
        let mut kept = 0u32;
        if member < spatial[cells as usize] {
            let peer = spatial[(cells + 1 + member) as usize];
            count = successes[filtered_count(config.actors, peer) as usize].load();
            if comptime!(collection == OrderCollection::ConsumerMask) && count > 0 {
                let slot = prefix_slot(successes, config.actors, peer);
                kept = successes[slot + 1].load();
            }
        }
        let total = parallel::reduce::<u32>(count, comptime!(Reduction::Sum));
        let maximum = parallel::reduce::<u32>(count + kept, comptime!(Reduction::Maximum));
        if UNIT_POS == 0 {
            control[COMPLETED_WORD].fetch_add(total);
            control[PROGRESS_WORD].fetch_max(maximum);
        }
    }
}

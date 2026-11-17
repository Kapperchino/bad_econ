use super::*;
use crate::simulation::parameters::EmptyOrderMode;

#[cube]
pub(super) fn orders(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    proposals: &[u32],
    successes: &[Atomic<u32>],
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
) {
    if comptime!(parameters.empty_orders == EmptyOrderMode::FeedbackOnly)
        && control[markets::WIDE_ORDERS_WORD].load() == 0
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
                let quantity = ticks(
                    proposals[base + ProposalLayout::QUANTITY_LOW],
                    proposals[base + ProposalLayout::QUANTITY_HIGH],
                );
                let mut filtered = false;
                if peer != NO_ACTOR && !equal(quantity, zero()) {
                    filtered = !kind(actors, peer).is(comptime!(ActorKind::Household))
                        && equal(value(state, config, parameters.traded_good(), peer), zero())
                        && super::super::clearing::empty_seller_safe(
                            state, config, parameters, peer,
                        );
                }
                if filtered {
                    kept &= !(1u32 << bit);
                    successes[markets::filtered_count(config.actors, peer) as usize].fetch_add(1);
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
) {
    if comptime!(parameters.empty_orders == EmptyOrderMode::FeedbackOnly)
        && control[markets::WIDE_ORDERS_WORD].load() == 0
        && control[crate::simulation::flags::FAULT_WORD].load() == 0
    {
        let cells = config.spatial_side * config.spatial_side;
        let member = ABSOLUTE_POS as u32;
        let mut count = 0u32;
        if member < spatial[cells as usize] {
            let peer = spatial[(cells + 1 + member) as usize];
            count = successes[markets::filtered_count(config.actors, peer) as usize].load();
        }
        let total = parallel::reduce::<u32>(count, comptime!(Reduction::Sum));
        let maximum = parallel::reduce::<u32>(count, comptime!(Reduction::Maximum));
        if UNIT_POS == 0 {
            control[COMPLETED_WORD].fetch_add(total);
            control[PROGRESS_WORD].fetch_max(maximum);
        }
    }
}

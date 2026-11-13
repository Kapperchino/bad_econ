use super::*;

#[cube]
pub(super) fn search(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    spatial: &[u32],
    proposals: &mut [u32],
    searches: &[u32],
    successes: &[Atomic<u32>],
    scratch: &[u32],
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
) {
    let job = ABSOLUTE_POS as u32;
    if job < control[SEARCHES_WORD].load() && control[LATCH_WORD].load() == 0 {
        let id = searches[job as usize];
        let wanted = ticks(
            searches[(config.actors + id) as usize],
            searches[(2 * config.actors + id) as usize],
        );
        let purchasing_power = ticks(
            searches[(3 * config.actors + id) as usize],
            searches[(4 * config.actors + id) as usize],
        );
        let mut buyer_access = 0u32;
        if comptime!(config.railway.enabled == Feature::Enabled) && config.railway.nodes > 0 {
            buyer_access = config.railway.access[id as usize * 2 + 1];
        }
        let px = actors[ActorLayout::X * (actors.len() / ActorLayout::WORDS) + id as usize];
        let py = actors[ActorLayout::Y * (actors.len() / ActorLayout::WORDS) + id as usize];
        let radius = config.search_radius;
        let cx = px / radius;
        let cy = py / radius;
        let side = config.spatial_side;
        let mut chosen = u32::new(comptime!(i64::from(NO_ACTOR)));
        let mut trip = 0u32;
        let mut best = ticks(u32::MAX, u32::MAX);
        let mut considered = 0u32;
        let mut empty_seller = u32::new(comptime!(i64::from(NO_ACTOR)));
        let mut empty_trip = 0u32;
        let mut empty_score = ticks(u32::MAX, u32::MAX);
        let mut y = max(cy, 1u32) - 1;
        while y <= min(cy + 1, side - 1) && considered < DISCOVERY_CANDIDATES {
            let mut x = max(cx, 1u32) - 1;
            while x <= min(cx + 1, side - 1) && considered < DISCOVERY_CANDIDATES {
                let cell = y * side + x;
                let begin = spatial[cell as usize];
                let end = spatial[(cell + 1) as usize];
                let length = end - begin;
                let mut cursor = (parameters.settlement_rotation + id) % max(length, 1);
                let mut j = begin;
                let nearest = max(px, x * radius) - min(px, (x + 1) * radius - 1)
                    + max(py, y * radius)
                    - min(py, (y + 1) * radius - 1);
                if nearest > radius {
                    j = end;
                }
                while j < end && considered < DISCOVERY_CANDIDATES {
                    let previous = j;
                    let rotated = begin + cursor;
                    let block = rotated / index::BLOCK;
                    let unavailable = index::skip(
                        scratch,
                        config,
                        block,
                        px,
                        py,
                        purchasing_power,
                        chosen,
                        empty_score,
                        1,
                    );
                    let skipped = select(
                        unavailable,
                        min(min((block + 1) * index::BLOCK, end) - rotated, end - j),
                        0,
                    );
                    if skipped == 0 {
                        let width = select(j == begin, 8u32, 32u32);
                        let mut offset = 0u32;
                        while offset < min(width, end - j) && considered < DISCOVERY_CANDIDATES {
                            let position = cursor + offset;
                            let member =
                                begin + select(position < length, position, position - length);
                            let firm = spatial[(side * side + 1 + member) as usize];
                            let supply = index::Supply::load(scratch, config, member);
                            let distance = distance(actors, id, firm);
                            let viable = firm != id
                                && distance <= radius
                                && supply.quote > 0
                                && !lt(purchasing_power, ticks(supply.quote, 0))
                                && !equal(wanted, zero());
                            let mut relevant = viable && supply.offered;
                            if viable && supply.producer && !supply.offered && chosen == NO_ACTOR {
                                let days =
                                    minimum_delivery_days(config, buyer_access, firm, distance);
                                let lower_score = plus(
                                    ticks(supply.quote, 0),
                                    times32(days, config.delivery_penalty),
                                )
                                .value;
                                relevant = lt(lower_score, empty_score);
                            }
                            if relevant {
                                let route = railway::route(actors, config, id, firm);
                                let score =
                                    ticks(supply.quote + route.days * config.delivery_penalty, 0);
                                if !supply.offered && lt(score, empty_score) {
                                    empty_seller = firm;
                                    empty_trip = distance;
                                    empty_score = score;
                                }
                                if supply.offered && (chosen == NO_ACTOR || lt(score, best)) {
                                    chosen = firm;
                                    trip = distance;
                                    best = score;
                                }
                                considered += u32::cast_from(supply.offered);
                            }
                            offset += 1;
                        }
                        j += width;
                    }
                    j += skipped;
                    cursor = (cursor + j - previous) % max(length, 1);
                }
                x += 1;
            }
            y += 1;
        }
        let mut missing_supply = false;
        if chosen == NO_ACTOR && empty_seller != NO_ACTOR {
            chosen = empty_seller;
            trip = empty_trip;
            missing_supply = true;
        }
        let mut quantity = zero();
        if chosen != NO_ACTOR {
            let supply =
                index::Supply::new(state, actors, config, chosen, parameters.traded_good());
            let affordable = minimum(wanted, divide(purchasing_power, max(supply.quote, 1)));
            quantity = select(
                missing_supply,
                affordable,
                minimum(affordable, supply.available),
            );
        }
        proposals[id as usize * ProposalLayout::WORDS + ProposalLayout::ACTOR] = chosen;
        proposals[id as usize * ProposalLayout::WORDS + ProposalLayout::QUANTITY_LOW] =
            quantity.extract(LOW_WORD);
        proposals[id as usize * ProposalLayout::WORDS + ProposalLayout::QUANTITY_HIGH] =
            quantity.extract(HIGH_WORD);
        proposals[id as usize * ProposalLayout::WORDS + ProposalLayout::DISTANCE] = trip;
        if chosen != NO_ACTOR {
            record_success(successes, parameters, id);
        }
    }
}

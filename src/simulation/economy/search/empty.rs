use super::*;

pub(super) const BLOCK: u32 = 8;
pub(super) const LIMIT: u32 = 2048;
const FIELDS: u32 = 8;

#[derive(Clone, Copy)]
enum Bound {
    Price,
    MinX,
    MaxX,
    MinY,
    MaxY,
    Access,
    Stocked,
    MinimumQuote,
}

#[derive(Clone, Copy)]
#[repr(u32)]
enum NodeState {
    Branch,
    Leaf,
    Pruned,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CacheMode {
    Discard,
    Capture,
}

#[derive(CubeType)]
pub(super) struct Offer {
    pub actor: u32,
    pub trip: u32,
}

#[derive(CubeType)]
struct Winner {
    actor: u32,
    trip: u32,
    score: u32,
    cell: u32,
    rank: u32,
}

#[derive(CubeType)]
struct Buyer {
    actor: u32,
    x: u32,
    y: u32,
    access: u32,
    cash: Ticks,
    rotation: u32,
}

#[cube]
pub(super) fn key(config: &EconomyConfig, x: u32, y: u32) -> u32 {
    let tile = ceildiv(config.map_size, 128);
    let x = x / tile;
    let y = y / tile;
    let mut key = 0u32;
    #[unroll]
    for bit in 0..7u32 {
        key |= ((x >> bit) & 1) << (2 * bit);
        key |= ((y >> bit) & 1) << (2 * bit + 1);
    }
    key
}

#[cube]
fn leaves(count: u32) -> u32 {
    1u32 << (32 - (max(ceildiv(count, BLOCK), 1) - 1).leading_zeros())
}

#[cube]
fn tree(config: &EconomyConfig, count: u32) -> bool {
    2 * leaves(count) * FIELDS <= config.actors + 512
}

#[cube]
pub(super) fn compacted(config: &EconomyConfig, count: u32) -> bool {
    config.spatial_side == 1 && count <= LIMIT && tree(config, count)
}

#[cube]
pub(super) fn stock_cursor(
    config: &EconomyConfig,
    searches: &[u32],
    stocked: u32,
    target: u32,
) -> u32 {
    let mut low = 0u32;
    let mut high = stocked;
    while low < high {
        let middle = (low + high) / 2;
        let member = searches[(17 * config.actors + middle) as usize];
        if member < target {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    select(low < stocked, low, 0)
}

#[cube]
pub(super) fn sort(
    config: &EconomyConfig,
    spatial: &[u32],
    scratch: &mut [u32],
    searches: &mut [u32],
) {
    let count = spatial[(config.spatial_side * config.spatial_side) as usize];
    let member = ABSOLUTE_POS as u32;
    if member < count && count <= LIMIT {
        let key = scratch[(5 * config.actors + member) as usize];
        let mut rank = 0u32;
        let mut stocked_rank = 0u32;
        for other in 0..count {
            let other_key = scratch[(5 * config.actors + other) as usize];
            rank += u32::cast_from(other_key < key || other_key == key && other < member);
            stocked_rank += u32::cast_from(
                other < member && scratch[(3 * config.actors + other) as usize] & 1 != 0,
            );
        }
        scratch[(7 * config.actors + rank) as usize] = member;
        if scratch[(3 * config.actors + member) as usize] & 1 != 0 {
            searches[(17 * config.actors + stocked_rank) as usize] = member;
        }
    }
}

#[cube]
fn write_bound(
    actors: &[u32],
    config: &EconomyConfig,
    spatial: &[u32],
    scratch: &mut [u32],
    node: u32,
    start: u32,
    end: u32,
) {
    let cells = config.spatial_side * config.spatial_side;
    let mut price = u32::new(comptime!(i64::from(u32::MAX)));
    let mut min_x = u32::new(comptime!(i64::from(u32::MAX)));
    let mut max_x = 0u32;
    let mut min_y = u32::new(comptime!(i64::from(u32::MAX)));
    let mut max_y = 0u32;
    let mut access = u32::new(comptime!(i64::from(u32::MAX)));
    let mut stocked = 0u32;
    let mut minimum_quote = u32::new(comptime!(i64::from(u32::MAX)));
    for position in start..end {
        let member = scratch[(7 * config.actors + position) as usize];
        let supply = index::Supply::load(&*scratch, config, member);
        stocked += u32::cast_from(supply.offered);
        if (supply.offered || supply.producer) && supply.quote > 0 {
            minimum_quote = min(minimum_quote, supply.quote);
        }
        if supply.producer && !supply.offered && supply.quote > 0 {
            let actor = spatial[(cells + 1 + member) as usize];
            let stride = actors.len() / ActorLayout::WORDS;
            let x = actors[ActorLayout::X * stride + actor as usize];
            let y = actors[ActorLayout::Y * stride + actor as usize];
            price = min(price, supply.quote);
            min_x = min(min_x, x);
            max_x = max(max_x, x);
            min_y = min(min_y, y);
            max_y = max(max_y, y);
            if comptime!(config.railway.routes_enabled == Feature::Enabled)
                && config.railway.nodes > 0
            {
                access = min(access, config.railway.access[actor as usize * 2 + 1]);
            }
        }
    }
    let base = 8 * config.actors + node * FIELDS;
    scratch[(base + comptime!(Bound::Price as u32)) as usize] = price;
    scratch[(base + comptime!(Bound::MinX as u32)) as usize] = min_x;
    scratch[(base + comptime!(Bound::MaxX as u32)) as usize] = max_x;
    scratch[(base + comptime!(Bound::MinY as u32)) as usize] = min_y;
    scratch[(base + comptime!(Bound::MaxY as u32)) as usize] = max_y;
    scratch[(base + comptime!(Bound::Access as u32)) as usize] = access;
    scratch[(base + comptime!(Bound::Stocked as u32)) as usize] = stocked;
    scratch[(base + comptime!(Bound::MinimumQuote as u32)) as usize] = minimum_quote;
}

#[cube]
pub(super) fn affordable(config: &EconomyConfig, scratch: &[u32], count: u32, cash: Ticks) -> bool {
    let mut affordable = true;
    if tree(config, count) {
        let quote =
            scratch[(8 * config.actors + FIELDS + comptime!(Bound::MinimumQuote as u32)) as usize];
        affordable = !lt(cash, ticks(quote, 0));
    }
    affordable
}

#[cube]
pub(super) fn stocked_count(config: &EconomyConfig, scratch: &[u32], count: u32) -> u32 {
    let mut stocked = count;
    if tree(config, count) {
        stocked = scratch[(8 * config.actors + FIELDS + comptime!(Bound::Stocked as u32)) as usize];
    }
    stocked
}

#[cube]
pub(super) fn bounds(actors: &[u32], config: &EconomyConfig, spatial: &[u32], scratch: &mut [u32]) {
    let count = spatial[(config.spatial_side * config.spatial_side) as usize];
    if count <= LIMIT {
        let invocation = ABSOLUTE_POS as u32;
        let leaves = leaves(count);
        if tree(config, count) {
            for node in 4 * invocation + 1..min(4 * invocation + 5, 2 * leaves) {
                let level = 1u32 << (31 - node.leading_zeros());
                let width = leaves / level * BLOCK;
                let start = (node - level) * width;
                let end = min(start + width, count);
                write_bound(actors, config, spatial, scratch, node, min(start, end), end);
            }
        } else if invocation < ceildiv(count, BLOCK) {
            write_bound(
                actors,
                config,
                spatial,
                scratch,
                invocation,
                invocation * BLOCK,
                min((invocation + 1) * BLOCK, count),
            );
        }
    }
}

#[cube]
fn first_block(config: &EconomyConfig, scratch: &[u32], count: u32, target: u32) -> u32 {
    let mut low = 0u32;
    let mut high = count;
    while low < high {
        let middle = (low + high) / 2;
        let member = scratch[(7 * config.actors + middle) as usize];
        let key = scratch[(5 * config.actors + member) as usize];
        if key < target {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    min(low, max(count, 1) - 1) / BLOCK
}

#[cube]
fn possible(
    config: &EconomyConfig,
    scratch: &[u32],
    node: u32,
    buyer: &Buyer,
    winner: &Winner,
) -> bool {
    let base = 8 * config.actors + node * FIELDS;
    let price = scratch[(base + comptime!(Bound::Price as u32)) as usize];
    let mut possible = false;
    if price != u32::MAX && !lt(buyer.cash, ticks(price, 0)) {
        let min_x = scratch[(base + comptime!(Bound::MinX as u32)) as usize];
        let max_x = scratch[(base + comptime!(Bound::MaxX as u32)) as usize];
        let min_y = scratch[(base + comptime!(Bound::MinY as u32)) as usize];
        let max_y = scratch[(base + comptime!(Bound::MaxY as u32)) as usize];
        let distance =
            max(buyer.x, min_x) - min(buyer.x, max_x) + max(buyer.y, min_y) - min(buyer.y, max_y);
        let mut days = max(ceildiv(distance, config.shipment_speed), 1);
        if comptime!(config.railway.routes_enabled == Feature::Enabled) && config.railway.nodes > 0
        {
            let access = scratch[(base + comptime!(Bound::Access as u32)) as usize];
            days = min(
                days,
                ceildiv(buyer.access + access, config.shipment_speed) + 1,
            );
        }
        let bound = price + days * config.delivery_penalty;
        possible = distance <= config.search_radius && bound <= winner.score;
    }
    possible
}

#[cube]
fn visit(
    actors: &[u32],
    config: &EconomyConfig,
    spatial: &[u32],
    scratch: &[u32],
    cache: &mut [u32],
    buyer: &Buyer,
    winner: &mut Winner,
    block: u32,
    #[comptime] cache_mode: CacheMode,
) {
    let cells = config.spatial_side * config.spatial_side;
    let count = spatial[cells as usize];
    let rotation = buyer.rotation % max(count, 1);
    for position in block * BLOCK..min((block + 1) * BLOCK, count) {
        let member = scratch[(7 * config.actors + position) as usize];
        let supply = index::Supply::load(scratch, config, member);
        let seller = spatial[(cells + 1 + member) as usize];
        if supply.producer
            && !supply.offered
            && supply.quote > 0
            && seller != buyer.actor
            && !lt(buyer.cash, ticks(supply.quote, 0))
        {
            let stride = actors.len() / ActorLayout::WORDS;
            let sx = actors[ActorLayout::X * stride + seller as usize];
            let sy = actors[ActorLayout::Y * stride + seller as usize];
            let distance =
                max(buyer.x, sx) - min(buyer.x, sx) + max(buyer.y, sy) - min(buyer.y, sy);
            let mut cell = 0u32;
            let mut relative = member;
            let mut length = count;
            let mut cell_rotation = rotation;
            if cells > 1 {
                cell = sy / config.search_radius * config.spatial_side + sx / config.search_radius;
                let begin = spatial[cell as usize];
                length = spatial[(cell + 1) as usize] - begin;
                relative = member - begin;
                cell_rotation = buyer.rotation % max(length, 1);
            }
            let rank = select(
                relative >= cell_rotation,
                relative - cell_rotation,
                relative + length - cell_rotation,
            );
            let preceding = cell < winner.cell || cell == winner.cell && rank < winner.rank;
            let days = minimum_delivery_days(config, buyer.access, seller, distance);
            let bound = supply.quote + days * config.delivery_penalty;
            let improves = bound < winner.score
                || bound == winner.score
                    && (preceding || comptime!(cache_mode == CacheMode::Capture));
            if distance <= config.search_radius && improves {
                let days = railway::delivery_days(
                    config,
                    railway::DeliveryRequest {
                        buyer: buyer.actor,
                        seller,
                        trip: distance,
                        buyer_access: buyer.access,
                    },
                );
                let score = supply.quote + days * config.delivery_penalty;
                if comptime!(cache_mode == CacheMode::Capture) {
                    let base = 6 * config.actors + 8 * buyer.actor;
                    if score < winner.score {
                        #[unroll]
                        for word in 0..8u32 {
                            cache[(base + word) as usize] = 0;
                        }
                    }
                    if score <= winner.score {
                        let word = base + member / 32;
                        cache[word as usize] |= 1 << (member % 32);
                    }
                }
                if score < winner.score || score == winner.score && preceding {
                    winner.actor = seller;
                    winner.trip = distance;
                    winner.score = score;
                    winner.cell = cell;
                    winner.rank = rank;
                }
            }
        }
    }
}

#[cube]
fn query_with_cache(
    actors: &[u32],
    config: &EconomyConfig,
    spatial: &[u32],
    scratch: &[u32],
    cache: &mut [u32],
    parameters: &EconomyParameters,
    buyer: u32,
    x: u32,
    y: u32,
    buyer_access: u32,
    cash: Ticks,
    #[comptime] cache_mode: CacheMode,
) -> Offer {
    let count = spatial[(config.spatial_side * config.spatial_side) as usize];
    let blocks = ceildiv(count, BLOCK);
    let first = first_block(config, scratch, count, key(config, x, y));
    let buyer = Buyer {
        actor: buyer,
        x,
        y,
        access: buyer_access,
        cash,
        rotation: parameters.settlement_rotation + buyer,
    };
    let mut winner = Winner {
        actor: u32::new(comptime!(i64::from(NO_ACTOR))),
        trip: 0u32,
        score: u32::new(comptime!(i64::from(u32::MAX))),
        cell: u32::new(comptime!(i64::from(u32::MAX))),
        rank: u32::new(comptime!(i64::from(u32::MAX))),
    };
    if tree(config, count) {
        visit(
            actors,
            config,
            spatial,
            scratch,
            cache,
            &buyer,
            &mut winner,
            first,
            cache_mode,
        );
        let leaves = leaves(count);
        let mut node = 1u32;
        while node < 2 * leaves {
            let possible = possible(config, scratch, node, &buyer, &winner);
            let state = select(
                possible,
                select(
                    node < leaves,
                    comptime!(NodeState::Branch as u32),
                    comptime!(NodeState::Leaf as u32),
                ),
                comptime!(NodeState::Pruned as u32),
            );
            match state {
                0 => node *= 2,
                _ => {
                    if state == comptime!(NodeState::Leaf as u32) && node != leaves + first {
                        visit(
                            actors,
                            config,
                            spatial,
                            scratch,
                            cache,
                            &buyer,
                            &mut winner,
                            node - leaves,
                            cache_mode,
                        );
                    }
                    let mut ancestor = node;
                    while ancestor > 1 && ancestor & 1 == 1 {
                        ancestor /= 2;
                    }
                    node = select(ancestor > 1, ancestor + 1, 2 * leaves);
                }
            }
        }
    } else {
        for step in 0..blocks {
            let position = first + step;
            let block = select(position < blocks, position, position - blocks);
            if possible(config, scratch, block, &buyer, &winner) {
                visit(
                    actors,
                    config,
                    spatial,
                    scratch,
                    cache,
                    &buyer,
                    &mut winner,
                    block,
                    cache_mode,
                );
            }
        }
    }
    Offer {
        actor: winner.actor,
        trip: winner.trip,
    }
}

#[cube]
pub(super) fn query(
    actors: &[u32],
    config: &EconomyConfig,
    spatial: &[u32],
    scratch: &[u32],
    parameters: &EconomyParameters,
    buyer: u32,
    x: u32,
    y: u32,
    buyer_access: u32,
    cash: Ticks,
) -> Offer {
    let mut cache = Array::<u32>::new(1usize);
    query_with_cache(
        actors,
        config,
        spatial,
        scratch,
        &mut cache[..],
        parameters,
        buyer,
        x,
        y,
        buyer_access,
        cash,
        comptime!(CacheMode::Discard),
    )
}

#[cube]
fn cached_offer(
    actors: &[u32],
    config: &EconomyConfig,
    spatial: &[u32],
    cache: &[u32],
    parameters: &EconomyParameters,
    buyer: u32,
) -> Offer {
    let count = spatial[1usize];
    let rotation = (parameters.settlement_rotation + buyer) % max(count, 1);
    let base = 6 * config.actors + 8 * buyer;
    let mut chosen = u32::new(comptime!(i64::from(NO_ACTOR)));
    let mut best = u32::new(comptime!(i64::from(u32::MAX)));
    #[unroll]
    for word in 0..8u32 {
        let mask = cache[(base + word) as usize];
        if mask != 0 {
            let mut after = mask;
            if word == rotation / 32 {
                after &= u32::MAX << (rotation % 32);
            }
            let selected = select(after != 0, after, mask);
            let member = word * 32 + selected.trailing_zeros();
            let rank = select(
                member >= rotation,
                member - rotation,
                member + count - rotation,
            );
            if rank < best {
                chosen = member;
                best = rank;
            }
        }
    }
    let mut actor = u32::new(comptime!(i64::from(NO_ACTOR)));
    let mut trip = 0u32;
    if chosen < count {
        actor = spatial[(2 + chosen) as usize];
        trip = distance(actors, buyer, actor);
    }
    Offer { actor, trip }
}

#[cube]
pub(super) fn cached_query(
    actors: &[u32],
    config: &EconomyConfig,
    spatial: &[u32],
    scratch: &[u32],
    cache: &mut [u32],
    parameters: &EconomyParameters,
    buyer: u32,
    x: u32,
    y: u32,
    buyer_access: u32,
    cash: Ticks,
) -> Offer {
    let valid = 14 * config.actors + buyer;
    let cash_low = 15 * config.actors + buyer;
    let cash_high = 16 * config.actors + buyer;
    let cached = cache[valid as usize] == 1
        && cache[cash_low as usize] == cash.extract(LOW_WORD)
        && cache[cash_high as usize] == cash.extract(HIGH_WORD);
    let mut offer = Offer {
        actor: u32::new(comptime!(i64::from(NO_ACTOR))),
        trip: 0u32,
    };
    if cached {
        let selected = cached_offer(actors, config, spatial, &*cache, parameters, buyer);
        offer.actor = selected.actor;
        offer.trip = selected.trip;
    } else {
        let base = 6 * config.actors + 8 * buyer;
        #[unroll]
        for word in 0..8u32 {
            cache[(base + word) as usize] = 0;
        }
        let selected = query_with_cache(
            actors,
            config,
            spatial,
            scratch,
            cache,
            parameters,
            buyer,
            x,
            y,
            buyer_access,
            cash,
            comptime!(CacheMode::Capture),
        );
        offer.actor = selected.actor;
        offer.trip = selected.trip;
        cache[valid as usize] = 1;
        cache[cash_low as usize] = cash.extract(LOW_WORD);
        cache[cash_high as usize] = cash.extract(HIGH_WORD);
    }
    offer
}

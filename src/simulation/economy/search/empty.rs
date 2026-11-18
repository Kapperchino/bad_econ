use super::*;
use crate::domain::SupplierCacheLayout;

pub(super) const BLOCK: u32 = 8;
pub(super) const LIMIT: u32 = SupplierCacheLayout::SUPPLIER_LIMIT as u32;
const INLINE_WORDS: u32 = SupplierCacheLayout::INLINE_WORDS as u32;
const FIELDS: u32 = 8;
const CACHE_STATE_BITS: u32 = 4;
const CACHE_STATE_MASK: u32 = (1 << CACHE_STATE_BITS) - 1;
const CACHE_STOCKED_MASK: u32 = (1 << 12) - 1;
const CACHE_SUMMARY_SHIFT: u32 = 16;
const CACHE_SUMMARY_MASK: u32 = u32::MAX << CACHE_SUMMARY_SHIFT;
const CACHE_ENTRIES_SHIFT: u32 = 30;
const CACHE_WORD_MASK: u32 = LIMIT / 32 - 1;

#[derive(Clone, Copy)]
#[repr(u32)]
enum CacheEntries {
    Empty,
    SingleWord,
    MultipleWords,
}

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
    summary: u32,
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
fn cache_word(config: &EconomyConfig, buyer: u32, word: u32) -> u32 {
    select(
        word < INLINE_WORDS,
        6 * config.actors + INLINE_WORDS * buyer + word,
        (comptime!(markets::SEARCH_REQUEST_FIELDS as u32) + word - INLINE_WORDS) * config.actors
            + buyer,
    )
}

#[cube]
pub(super) fn cache_fits(config: &EconomyConfig, searches: &[u32], suppliers: u32) -> bool {
    let extra = max(ceildiv(suppliers, 32), INLINE_WORDS) - INLINE_WORDS;
    let fields = comptime!(markets::SEARCH_REQUEST_FIELDS) + extra as usize;
    fields * config.actors as usize <= searches.len()
}

#[cube]
fn clear_cache(config: &EconomyConfig, cache: &mut [u32], buyer: u32, count: u32) {
    for word in 0..ceildiv(max(count, 1), 32) {
        cache[cache_word(config, buyer, word) as usize] = 0;
    }
}

#[cube]
fn record_word(summary: u32, word: u32) -> u32 {
    let entries = summary >> CACHE_ENTRIES_SHIFT;
    let first = (summary >> CACHE_SUMMARY_SHIFT) & CACHE_WORD_MASK;
    let updated = match entries {
        0 => u32::new(comptime!(CacheEntries::SingleWord as i64)),
        1 => select(
            first == word,
            comptime!(CacheEntries::SingleWord as u32),
            comptime!(CacheEntries::MultipleWords as u32),
        ),
        _ => u32::new(comptime!(CacheEntries::MultipleWords as i64)),
    };
    let first = select(
        entries == comptime!(CacheEntries::Empty as u32),
        word,
        first,
    );
    (summary & !CACHE_SUMMARY_MASK)
        | (updated << CACHE_ENTRIES_SHIFT)
        | (first << CACHE_SUMMARY_SHIFT)
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
    config.spatial_side == 1 && count <= LIMIT
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
    #[comptime] mode: parallel::ScanMode,
) {
    let count = spatial[(config.spatial_side * config.spatial_side) as usize];
    if CUBE_POS == 0 && count > 0 && count <= LIMIT {
        let chunk = LIMIT / CUBE_DIM;
        let start = UNIT_POS * chunk;
        let end = min(start + chunk, count);
        let mut offered = 0u32;
        for member in start..end {
            offered += u32::cast_from(index::Supply::load(&*scratch, member).offered);
        }
        let prefix = parallel::scan_prefix(offered, comptime!(mode));
        let mut target = prefix.inclusive - offered;
        for member in start..end {
            if index::Supply::load(&*scratch, member).offered {
                searches[(17 * config.actors + target) as usize] = member;
                target += 1;
            }
        }
        if UNIT_POS == 0 {
            scratch[(7 * config.actors + count) as usize] = prefix.total;
        }
        let mut members = Shared::<[u32]>::new_slice(LIMIT as usize);
        let length = 1u32 << (32 - (count - 1).leading_zeros());
        let mut position = UNIT_POS;
        while position < length {
            let mut packed = u32::new(comptime!(i64::from(u32::MAX)));
            if position < count {
                packed = (scratch[(5 * config.actors + position) as usize] << 11) | position;
            }
            members[position as usize] = packed;
            position += CUBE_DIM;
        }
        sync_cube();
        let mut width = 2u32;
        while width <= length {
            let mut stride = width / 2;
            while stride > 0 {
                let mut position = UNIT_POS;
                while position < length {
                    let other = position ^ stride;
                    if other > position {
                        let low = min(members[position as usize], members[other as usize]);
                        let high = max(members[position as usize], members[other as usize]);
                        let ascending = position & width == 0;
                        members[position as usize] = select(ascending, low, high);
                        members[other as usize] = select(ascending, high, low);
                    }
                    position += CUBE_DIM;
                }
                sync_cube();
                stride /= 2;
            }
            width *= 2;
        }
        let mut position = UNIT_POS;
        while position < count {
            scratch[(7 * config.actors + position) as usize] =
                members[position as usize] & (LIMIT - 1);
            position += CUBE_DIM;
        }
    }
}

#[cfg(test)]
#[cube]
pub(super) fn sort_scalar(
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
            stocked_rank +=
                u32::cast_from(other < member && index::Supply::load(&*scratch, other).offered);
        }
        scratch[(7 * config.actors + rank) as usize] = member;
        if index::Supply::load(&*scratch, member).offered {
            searches[(17 * config.actors + stocked_rank) as usize] = member;
        }
        if member == 0 {
            let mut total = 0u32;
            for other in 0..count {
                total += u32::cast_from(index::Supply::load(&*scratch, other).offered);
            }
            scratch[(7 * config.actors + count) as usize] = total;
        }
    }
}

#[cube]
fn write_bound_scalar(
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
        let supply = index::Supply::load(&*scratch, member);
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
fn write_bound_plane(
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
    let mut position = start + UNIT_POS_PLANE;
    while position < end {
        let member = scratch[(7 * config.actors + position) as usize];
        let supply = index::Supply::load(&*scratch, member);
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
        position += 32;
    }
    price = plane_min(price);
    min_x = plane_min(min_x);
    max_x = plane_max(max_x);
    min_y = plane_min(min_y);
    max_y = plane_max(max_y);
    access = plane_min(access);
    stocked = plane_sum(stocked);
    minimum_quote = plane_min(minimum_quote);
    if UNIT_POS_PLANE == 0 {
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
    #[comptime] mode: parallel::ScanMode,
) {
    if comptime!(mode == parallel::ScanMode::Plane) && PLANE_DIM == 32 {
        write_bound_plane(actors, config, spatial, scratch, node, start, end);
    } else {
        write_bound_scalar(actors, config, spatial, scratch, node, start, end);
    }
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
    if count <= LIMIT {
        stocked = select(
            count == 0,
            0u32,
            scratch[(7 * config.actors + count) as usize],
        );
    }
    stocked
}

#[cube]
pub(super) fn bounds(
    actors: &[u32],
    config: &EconomyConfig,
    spatial: &[u32],
    scratch: &mut [u32],
    #[comptime] mode: parallel::ScanMode,
) {
    let count = spatial[(config.spatial_side * config.spatial_side) as usize];
    if count <= LIMIT {
        let mut invocation = ABSOLUTE_POS as u32;
        if comptime!(mode == parallel::ScanMode::Plane) && PLANE_DIM == 32 {
            invocation /= 32;
        }
        let leaves = leaves(count);
        if tree(config, count) {
            for node in 4 * invocation + 1..min(4 * invocation + 5, 2 * leaves) {
                let level = 1u32 << (31 - node.leading_zeros());
                let width = leaves / level * BLOCK;
                let start = (node - level) * width;
                let end = min(start + width, count);
                write_bound(
                    actors,
                    config,
                    spatial,
                    scratch,
                    node,
                    min(start, end),
                    end,
                    comptime!(mode),
                );
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
                comptime!(mode),
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
        let supply = index::Supply::load(scratch, member);
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
                    if score < winner.score {
                        clear_cache(config, cache, buyer.actor, count);
                        winner.summary &= !CACHE_SUMMARY_MASK;
                    }
                    if score <= winner.score {
                        let word = cache_word(config, buyer.actor, member / 32);
                        cache[word as usize] |= 1 << (member % 32);
                        winner.summary = record_word(winner.summary, member / 32);
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
    cache_header: u32,
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
        summary: cache_header,
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
    if comptime!(cache_mode == CacheMode::Capture) {
        cache[(14 * config.actors + buyer.actor) as usize] = winner.summary;
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
        0u32,
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
    captured: u32,
) -> Offer {
    let count = spatial[1usize];
    let entries = captured >> CACHE_ENTRIES_SHIFT;
    let mut chosen = u32::new(comptime!(i64::from(NO_ACTOR)));
    match entries {
        0 => (),
        1 => {
            let rotation = (parameters.settlement_rotation + buyer) % max(count, 1);
            let word = (captured >> CACHE_SUMMARY_SHIFT) & CACHE_WORD_MASK;
            let mask = cache[cache_word(config, buyer, word) as usize];
            let start = select(word == rotation / 32, rotation % 32, 0);
            let after = mask & (u32::MAX << start);
            if mask != 0 {
                chosen =
                    word * 32 + select(after != 0, after.trailing_zeros(), mask.trailing_zeros());
            }
        }
        _ => {
            let rotation = (parameters.settlement_rotation + buyer) % max(count, 1);
            let words = ceildiv(max(count, 1), 32);
            let first = rotation / 32;
            let mask = cache[cache_word(config, buyer, first) as usize];
            let after = mask & (u32::MAX << (rotation % 32));
            if after != 0 {
                chosen = first * 32 + after.trailing_zeros();
            }
            let mut step = 1u32;
            while chosen == NO_ACTOR && step < words {
                let shifted = first + step;
                let word = select(shifted < words, shifted, shifted - words);
                let current = cache[cache_word(config, buyer, word) as usize];
                if current != 0 {
                    chosen = word * 32 + current.trailing_zeros();
                }
                step += 1;
            }
            if chosen == NO_ACTOR && mask != 0 {
                chosen = first * 32 + mask.trailing_zeros();
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
    let count = spatial[1usize];
    let stocked = stocked_count(config, scratch, count);
    let captured = cache[valid as usize];
    let prior_stocked = (captured >> CACHE_STATE_BITS) & CACHE_STOCKED_MASK;
    let unchanged = select(
        parameters.traded_good() == comptime!(Good::Bread as u32),
        stocked == prior_stocked,
        stocked <= prior_stocked,
    );
    let cached = captured & CACHE_STATE_MASK > 0
        && unchanged
        && cache[cash_low as usize] == cash.extract(LOW_WORD)
        && cache[cash_high as usize] == cash.extract(HIGH_WORD);
    let mut offer = Offer {
        actor: u32::new(comptime!(i64::from(NO_ACTOR))),
        trip: 0u32,
    };
    if cached {
        let selected = cached_offer(
            actors, config, spatial, &*cache, parameters, buyer, captured,
        );
        offer.actor = selected.actor;
        offer.trip = selected.trip;
        cache[valid as usize] = (captured & CACHE_SUMMARY_MASK) | 1 | (stocked << CACHE_STATE_BITS);
    } else {
        clear_cache(config, cache, buyer, count);
        cache[valid as usize] = 0;
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
            1 | (stocked << CACHE_STATE_BITS),
            comptime!(CacheMode::Capture),
        );
        offer.actor = selected.actor;
        offer.trip = selected.trip;
        cache[cash_low as usize] = cash.extract(LOW_WORD);
        cache[cash_high as usize] = cash.extract(HIGH_WORD);
    }
    offer
}

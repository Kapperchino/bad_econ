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
pub(super) struct Buyer {
    inbound: bool,
    actor: u32,
    location: BuyerLocation,
    cash: Ticks,
    rotation: u32,
}

#[cube]
impl Buyer {
    pub(super) fn new(
        config: &EconomyConfig,
        actor: u32,
        location: BuyerLocation,
        cash: Ticks,
        settlement_rotation: u32,
    ) -> Self {
        Buyer {
            inbound: config.railway.serves_buyer(actor, config.ring),
            actor,
            location,
            cash,
            rotation: settlement_rotation + actor,
        }
    }
}

#[derive(CubeType)]
pub(super) struct EmptySuppliers<'a> {
    config: &'a EconomyConfig,
    spatial: &'a [u32],
    scratch: &'a [u32],
    count: u32,
}

#[cube]
fn cache_word(config: &EconomyConfig, buyer: u32, word: u32) -> u32 {
    select(
        word < INLINE_WORDS,
        6 * config.actors + INLINE_WORDS * buyer + word,
        (comptime!(orders::SEARCH_REQUEST_FIELDS as u32) + word - INLINE_WORDS) * config.actors
            + buyer,
    )
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
fn write_bound(
    actors: &ActorTable<'_>,
    config: &EconomyConfig,
    spatial: &[u32],
    scratch: &mut [u32],
    node: u32,
    start: u32,
    end: u32,
    #[comptime] mode: parallel::ScanMode,
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
    let mut position = start;
    let mut stride = 1u32;
    let mut writer = true;
    if comptime!(mode == parallel::ScanMode::Plane) && PLANE_DIM == 32 {
        position += UNIT_POS_PLANE;
        stride = 32;
        writer = UNIT_POS_PLANE == 0;
    }
    while position < end {
        let member = scratch[(7 * config.actors + position) as usize];
        let supply = index::Supply::load(&*scratch, member);
        stocked += u32::cast_from(supply.offered);
        if (supply.offered || supply.producer) && supply.quote > 0 {
            minimum_quote = min(minimum_quote, supply.quote);
        }
        if supply.producer && !supply.offered && supply.quote > 0 {
            let actor = spatial[(cells + 1 + member) as usize];
            let x = actors.x(actor);
            let y = actors.y(actor);
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
        position += stride;
    }
    if comptime!(mode == parallel::ScanMode::Plane) && PLANE_DIM == 32 {
        price = plane_min(price);
        min_x = plane_min(min_x);
        max_x = plane_max(max_x);
        min_y = plane_min(min_y);
        max_y = plane_max(max_y);
        access = plane_min(access);
        stocked = plane_sum(stocked);
        minimum_quote = plane_min(minimum_quote);
    }
    if writer {
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
pub(super) fn bounds(
    actors: &ActorTable<'_>,
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
impl<'a> EmptySuppliers<'a> {
    pub(super) fn new(
        config: &'a EconomyConfig,
        spatial: &'a [u32],
        scratch: &'a [u32],
        count: u32,
    ) -> Self {
        EmptySuppliers::<'a> {
            config,
            spatial,
            scratch,
            count,
        }
    }

    pub(super) fn cache_fits(&self, searches: &[u32]) -> bool {
        let config = self.config;
        let extra = max(ceildiv(self.count, 32), INLINE_WORDS) - INLINE_WORDS;
        let fields = comptime!(orders::SEARCH_REQUEST_FIELDS) + extra as usize;
        fields * config.actors as usize <= searches.len()
    }

    pub(super) fn compacted(&self) -> bool {
        self.config.spatial_side == 1 && self.count <= LIMIT
    }

    pub(super) fn stock_cursor(&self, searches: &[u32], stocked: u32, target: u32) -> u32 {
        let mut low = 0u32;
        let mut high = stocked;
        while low < high {
            let middle = (low + high) / 2;
            let member = searches[(17 * self.config.actors + middle) as usize];
            if member < target {
                low = middle + 1;
            } else {
                high = middle;
            }
        }
        select(low < stocked, low, 0)
    }

    pub(super) fn affordable(&self, cash: Ticks) -> bool {
        let mut affordable = true;
        if tree(self.config, self.count) {
            let quote = self.scratch[(8 * self.config.actors
                + FIELDS
                + comptime!(Bound::MinimumQuote as u32))
                as usize];
            affordable = !lt(cash, ticks(quote, 0));
        }
        affordable
    }

    pub(super) fn stocked_count(&self) -> u32 {
        let count = self.count;
        let mut stocked = count;
        if count <= LIMIT {
            stocked = select(
                count == 0,
                0u32,
                self.scratch[(7 * self.config.actors + count) as usize],
            );
        }
        stocked
    }

    fn first_block(&self, target: u32) -> u32 {
        let count = self.count;
        let mut low = 0u32;
        let mut high = count;
        while low < high {
            let middle = (low + high) / 2;
            let member = self.scratch[(7 * self.config.actors + middle) as usize];
            let key = self.scratch[(5 * self.config.actors + member) as usize];
            if key < target {
                low = middle + 1;
            } else {
                high = middle;
            }
        }
        min(low, max(count, 1) - 1) / BLOCK
    }

    fn possible(&self, node: u32, buyer: &Buyer, winner: &Winner) -> bool {
        let config = self.config;
        let scratch = self.scratch;
        let base = 8 * config.actors + node * FIELDS;
        let price = scratch[(base + comptime!(Bound::Price as u32)) as usize];
        let mut possible = false;
        if price != u32::MAX && !lt(buyer.cash, ticks(price, 0)) {
            let min_x = scratch[(base + comptime!(Bound::MinX as u32)) as usize];
            let max_x = scratch[(base + comptime!(Bound::MaxX as u32)) as usize];
            let min_y = scratch[(base + comptime!(Bound::MinY as u32)) as usize];
            let max_y = scratch[(base + comptime!(Bound::MaxY as u32)) as usize];
            let distance = buyer
                .location
                .distance_to_bounds(min_x, max_x, min_y, max_y);
            let mut days = max(ceildiv(distance, config.shipment_speed), 1);
            let mut delivered = price;
            if buyer.inbound {
                let access = scratch[(base + comptime!(Bound::Access as u32)) as usize];
                days = min(
                    days,
                    ceildiv(buyer.location.access + access, config.shipment_speed) + 1,
                );
            }
            if !buyer.inbound {
                delivered = cartage::delivered_price(config, price, distance, days);
            }
            let bound = delivered + days * config.delivery_penalty;
            possible = distance <= config.search_radius
                && bound <= winner.score
                && !lt(buyer.cash, ticks(delivered, 0));
        }
        possible
    }

    fn visit(
        &self,
        actors: &ActorTable<'_>,
        cache: &mut [u32],
        buyer: &Buyer,
        winner: &mut Winner,
        block: u32,
        #[comptime] cache_mode: CacheMode,
    ) {
        let config = self.config;
        let spatial = self.spatial;
        let scratch = self.scratch;
        let cells = config.spatial_side * config.spatial_side;
        let count = self.count;
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
                let sx = actors.x(seller);
                let sy = actors.y(seller);
                let distance = max(buyer.location.x, sx) - min(buyer.location.x, sx)
                    + max(buyer.location.y, sy)
                    - min(buyer.location.y, sy);
                let mut cell = 0u32;
                let mut relative = member;
                let mut length = count;
                let mut cell_rotation = rotation;
                if cells > 1 {
                    cell =
                        sy / config.search_radius * config.spatial_side + sx / config.search_radius;
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
                let mut days = max(ceildiv(distance, config.shipment_speed), 1);
                if buyer.inbound {
                    days = minimum_delivery_days(config, buyer.location.access, seller, distance);
                }
                let minimum = cartage::delivered_price(config, supply.quote, distance, days);
                let bound = minimum + days * config.delivery_penalty;
                let improves = bound < winner.score
                    || bound == winner.score
                        && (preceding || comptime!(cache_mode == CacheMode::Capture));
                if distance <= config.search_radius
                    && improves
                    && !lt(buyer.cash, ticks(minimum, 0))
                {
                    if buyer.inbound {
                        days = railway::DeliveryRequest {
                            buyer: buyer.actor,
                            seller,
                            trip: distance,
                            buyer_access: buyer.location.access,
                        }
                        .days(config);
                    }
                    let delivered = cartage::delivered_price(config, supply.quote, distance, days);
                    let affordable = !lt(buyer.cash, ticks(delivered, 0));
                    let score = delivered + days * config.delivery_penalty;
                    if comptime!(cache_mode == CacheMode::Capture) && affordable {
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
                    if affordable && (score < winner.score || score == winner.score && preceding) {
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

    fn query_with_cache(
        &self,
        actors: &ActorTable<'_>,
        cache: &mut [u32],
        buyer: &Buyer,
        cache_header: u32,
        #[comptime] cache_mode: CacheMode,
    ) -> Offer {
        let config = self.config;
        let count = self.count;
        let blocks = ceildiv(count, BLOCK);
        let first = self.first_block(key(config, buyer.location.x, buyer.location.y));
        let mut winner = Winner {
            actor: u32::new(comptime!(i64::from(NO_ACTOR))),
            trip: 0u32,
            score: u32::new(comptime!(i64::from(u32::MAX))),
            cell: u32::new(comptime!(i64::from(u32::MAX))),
            rank: u32::new(comptime!(i64::from(u32::MAX))),
            summary: cache_header,
        };
        if tree(config, count) {
            self.visit(actors, cache, buyer, &mut winner, first, cache_mode);
            let leaves = leaves(count);
            let mut node = 1u32;
            while node < 2 * leaves {
                let possible = self.possible(node, buyer, &winner);
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
                            self.visit(
                                actors,
                                cache,
                                buyer,
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
                if self.possible(block, buyer, &winner) {
                    self.visit(actors, cache, buyer, &mut winner, block, cache_mode);
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

    pub(super) fn query(&self, actors: &ActorTable<'_>, buyer: &Buyer) -> Offer {
        let mut cache = Array::<u32>::new(1usize);
        self.query_with_cache(
            actors,
            &mut cache[..],
            buyer,
            0u32,
            comptime!(CacheMode::Discard),
        )
    }

    fn cached_offer(
        &self,
        actors: &ActorTable<'_>,
        cache: &[u32],
        buyer: &Buyer,
        captured: u32,
    ) -> Offer {
        let config = self.config;
        let spatial = self.spatial;
        let count = spatial[1usize];
        let entries = captured >> CACHE_ENTRIES_SHIFT;
        let mut chosen = u32::new(comptime!(i64::from(NO_ACTOR)));
        match entries {
            0 => (),
            1 => {
                let rotation = buyer.rotation % max(count, 1);
                let word = (captured >> CACHE_SUMMARY_SHIFT) & CACHE_WORD_MASK;
                let mask = cache[cache_word(config, buyer.actor, word) as usize];
                let start = select(word == rotation / 32, rotation % 32, 0);
                let after = mask & (u32::MAX << start);
                if mask != 0 {
                    chosen = word * 32
                        + select(after != 0, after.trailing_zeros(), mask.trailing_zeros());
                }
            }
            _ => {
                let rotation = buyer.rotation % max(count, 1);
                let words = ceildiv(max(count, 1), 32);
                let first = rotation / 32;
                let mask = cache[cache_word(config, buyer.actor, first) as usize];
                let after = mask & (u32::MAX << (rotation % 32));
                if after != 0 {
                    chosen = first * 32 + after.trailing_zeros();
                }
                let mut step = 1u32;
                while chosen == NO_ACTOR && step < words {
                    let shifted = first + step;
                    let word = select(shifted < words, shifted, shifted - words);
                    let current = cache[cache_word(config, buyer.actor, word) as usize];
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
            trip = actors.distance(buyer.actor, actor);
        }
        Offer { actor, trip }
    }

    pub(super) fn cached_query(
        &self,
        actors: &ActorTable<'_>,
        cache: &mut [u32],
        buyer: &Buyer,
        good: u32,
    ) -> Offer {
        let config = self.config;
        let valid = 14 * config.actors + buyer.actor;
        let cash_low = 15 * config.actors + buyer.actor;
        let cash_high = 16 * config.actors + buyer.actor;
        let count = self.spatial[1usize];
        let stocked = self.stocked_count();
        let captured = cache[valid as usize];
        let prior_stocked = (captured >> CACHE_STATE_BITS) & CACHE_STOCKED_MASK;
        let unchanged = select(
            good == comptime!(Good::Bread as u32),
            stocked == prior_stocked,
            stocked <= prior_stocked,
        );
        let cached = captured & CACHE_STATE_MASK > 0
            && unchanged
            && cache[cash_low as usize] == buyer.cash.extract(LOW_WORD)
            && cache[cash_high as usize] == buyer.cash.extract(HIGH_WORD);
        let mut offer = Offer {
            actor: u32::new(comptime!(i64::from(NO_ACTOR))),
            trip: 0u32,
        };
        if cached {
            let selected = self.cached_offer(actors, &*cache, buyer, captured);
            offer.actor = selected.actor;
            offer.trip = selected.trip;
            cache[valid as usize] =
                (captured & CACHE_SUMMARY_MASK) | 1 | (stocked << CACHE_STATE_BITS);
        } else {
            clear_cache(config, cache, buyer.actor, count);
            cache[valid as usize] = 0;
            let selected = self.query_with_cache(
                actors,
                cache,
                buyer,
                1 | (stocked << CACHE_STATE_BITS),
                comptime!(CacheMode::Capture),
            );
            offer.actor = selected.actor;
            offer.trip = selected.trip;
            cache[cash_low as usize] = buyer.cash.extract(LOW_WORD);
            cache[cash_high as usize] = buyer.cash.extract(HIGH_WORD);
        }
        offer
    }
}

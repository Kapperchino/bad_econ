use super::*;

const MEMBERS: u32 = 6;

#[cube]
pub(super) fn collect_firms(
    economy: &EconomyView<'_>,
    spatial: &[u32],
    searches: &mut [u32],
    cache: &mut [u32],
    control: &[Atomic<u32>],
) {
    let config = economy.ledger.config;
    let member = ABSOLUTE_POS as u32;
    let mut firm = u32::new(comptime!(i64::from(u32::MAX)));
    let mut household_quote = u32::new(comptime!(i64::from(u32::MAX)));
    if member < spatial[1usize] {
        let actor = spatial[(2 + member) as usize];
        index::Supply::new(economy, actor, comptime!(Good::Grain as u32)).store(cache, member);
        if !economy
            .actors
            .kind(actor)
            .is(comptime!(ActorKind::Household))
        {
            firm = actor;
        } else {
            household_quote = economy.ledger.number(Field::GrainQuote, actor);
        }
    }
    let minimum = plane_min(household_quote);
    if UNIT_POS_PLANE == 0 && minimum < control[INPUT_MINIMUM_QUOTE_WORD].load() {
        control[INPUT_MINIMUM_QUOTE_WORD].fetch_min(minimum);
    }
    let active = u32::cast_from(firm != u32::MAX);
    let prefix = plane_exclusive_sum(active);
    let total = plane_sum(active);
    let mut base = 0u32;
    if UNIT_POS_PLANE == 0 && total > 0 {
        base = control[GRAIN_FIRMS_WORD].fetch_add(total);
    }
    let slot = plane_broadcast(base, 0u32) + prefix;
    if active > 0 {
        searches[(comptime!(MEMBERS) * config.actors + slot) as usize] = member;
    }
}

#[cube]
pub(super) fn refresh_supply(economy: &EconomyView<'_>, spatial: &[u32], cache: &mut [u32]) {
    let member = ABSOLUTE_POS as u32;
    if member < spatial[1usize] {
        let seller = spatial[(2 + member) as usize];
        index::Supply::new(economy, seller, comptime!(Good::Grain as u32)).store(cache, member);
    }
}

#[cube]
fn inverse_stride(length: u32, stride: u32) -> u32 {
    let mut divisor = stride;
    let mut remainder = length;
    let mut coefficient = 1i64;
    let mut previous = 0i64;
    while divisor > 0 {
        let quotient = remainder / divisor;
        let next = remainder % divisor;
        remainder = divisor;
        divisor = next;
        let next_coefficient = previous - i64::cast_from(quotient) * coefficient;
        previous = coefficient;
        coefficient = next_coefficient;
    }
    u32::cast_from(select(
        previous < 0,
        previous + i64::cast_from(length),
        previous,
    ))
}

#[derive(CubeType)]
pub(super) struct Trade {
    pub seller: u32,
    pub quantity: Ticks,
    pub trip: u32,
}

#[cube]
pub(super) fn search(
    economy: &EconomyView<'_>,
    spatial: &[u32],
    searches: &[u32],
    cache: &[u32],
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
    buyer: u32,
    wanted: Ticks,
    cash: Ticks,
    access: u32,
) -> Trade {
    let config = economy.ledger.config;
    let length = max(spatial[1usize], 1);
    let stride = supplier_stride(&economy.actors, parameters, buyer, length);
    let inverse = inverse_stride(length, stride);
    let start = (parameters.settlement_rotation + buyer) % length;
    let mut first = Array::<u64>::new(GRAIN_MARKET_QUOTES as usize);
    #[unroll]
    for i in 0..GRAIN_MARKET_QUOTES as usize {
        first[i] = u64::MAX;
    }
    let mut partial = u64::cast_from(-1i64);
    let mut partial_rank = u32::new(comptime!(i64::from(u32::MAX)));
    let mut empty = u64::cast_from(-1i64);
    let count = control[GRAIN_FIRMS_WORD].load();
    let mut slot = UNIT_POS_PLANE;
    while slot < count {
        let member = searches[(comptime!(MEMBERS) * config.actors + slot) as usize];
        let seller = spatial[(2 + member) as usize];
        let supply = index::Supply::load(cache, member);
        if !lt(cash, ticks(max(supply.quote, 1), 0)) {
            let distance = economy.actors.distance(buyer, seller);
            let offer = OfferRequest {
                buyer,
                seller,
                trip: distance,
                buyer_access: access,
                demand: BuyerDemand {
                    wanted,
                    purchasing_power: cash,
                },
            }
            .with_supply(economy, control, parameters, &supply);
            if offer.empty || offer.eligible {
                let offset = (member + length - start) % length;
                let rank = u32::cast_from(
                    u64::cast_from(offset) * u64::cast_from(inverse) % u64::cast_from(length),
                );
                let score = native(offer.score);
                if offer.empty {
                    empty = min(empty, (score << 32) | u64::cast_from(rank));
                } else if offer.score.extract(HIGH_WORD) == 0 {
                    let mut item = (u64::cast_from(rank) << 32) | score;
                    if item < first[comptime!(GRAIN_MARKET_QUOTES as usize - 1)] {
                        #[unroll]
                        for i in 0..GRAIN_MARKET_QUOTES as usize {
                            let prior = first[i];
                            first[i] = min(item, prior);
                            item = max(item, prior);
                        }
                    }
                } else if score < partial || (score == partial && rank < partial_rank) {
                    partial = score;
                    partial_rank = rank;
                }
            }
        }
        slot += PLANE_DIM;
    }
    let mut full = u64::cast_from(-1i64);
    let mut head = 0usize;
    #[unroll]
    for _ in 0..GRAIN_MARKET_QUOTES {
        let item = first[head];
        let rank = plane_min(u32::cast_from(item >> 32));
        if item != u64::MAX && u32::cast_from(item >> 32) == rank {
            full = min(full, (item << 32) | u64::cast_from(rank));
            head = min(head + 1, comptime!(GRAIN_MARKET_QUOTES as usize - 1));
        }
    }
    let full_price = plane_min(u32::cast_from(full >> 32));
    let full_rank = plane_min(select(
        u32::cast_from(full >> 32) == full_price,
        u32::cast_from(full),
        u32::MAX,
    ));
    let missing = plane_min(u32::cast_from(partial >> 32));
    let price = plane_min(select(
        u32::cast_from(partial >> 32) == missing,
        u32::cast_from(partial),
        u32::MAX,
    ));
    let partial_best = plane_min(select(
        partial == ((u64::cast_from(missing) << 32) | u64::cast_from(price)),
        partial_rank,
        u32::MAX,
    ));
    let empty_price = plane_min(u32::cast_from(empty >> 32));
    let empty_rank = plane_min(select(
        u32::cast_from(empty >> 32) == empty_price,
        u32::cast_from(empty),
        u32::MAX,
    ));
    let rank = select(
        full_rank != u32::MAX,
        full_rank,
        select(partial_best != u32::MAX, partial_best, empty_rank),
    );
    let mut trade = Trade {
        seller: u32::MAX,
        quantity: zero(),
        trip: 0,
    };
    if rank != u32::MAX {
        let member = u32::cast_from(
            (u64::cast_from(start) + u64::cast_from(stride) * u64::cast_from(rank))
                % u64::cast_from(length),
        );
        let seller = spatial[(2 + member) as usize];
        let supply = index::Supply::load(cache, member);
        let trip = economy.actors.distance(buyer, seller);
        let offer = OfferRequest {
            buyer,
            seller,
            trip,
            buyer_access: access,
            demand: BuyerDemand {
                wanted,
                purchasing_power: cash,
            },
        }
        .with_supply(economy, control, parameters, &supply);
        trade.seller = seller;
        trade.quantity = offer.quantity;
        trade.trip = trip;
    }
    trade
}

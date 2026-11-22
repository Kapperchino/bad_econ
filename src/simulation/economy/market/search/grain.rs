use super::*;

const MEMBERS: u32 = 6;
const SELLERS: u32 = 7;
const OFFERS: u32 = 8;

#[cube]
pub(super) fn collect_firms(
    economy: &EconomyView<'_>,
    spatial: &[u32],
    searches: &mut [u32],
    control: &[Atomic<u32>],
) {
    let config = economy.ledger.config;
    let member = ABSOLUTE_POS as u32;
    let mut firm = u32::new(comptime!(i64::from(u32::MAX)));
    let mut household_quote = u32::new(comptime!(i64::from(u32::MAX)));
    if member < spatial[1usize] {
        let actor = spatial[(2 + member) as usize];
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
        searches[(comptime!(SELLERS) * config.actors + slot) as usize] = firm;
        let supply = index::Supply::new(economy, firm, comptime!(Good::Grain as u32));
        supply.store(
            &mut searches[(comptime!(OFFERS) * config.actors) as usize..],
            slot,
        );
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
    let mut first = Array::<u64>::new(DISCOVERY_CANDIDATES as usize);
    #[unroll]
    for i in 0..DISCOVERY_CANDIDATES as usize {
        first[i] = u64::MAX;
    }
    let mut partial = u64::cast_from(-1i64);
    let mut partial_rank = u32::new(comptime!(i64::from(u32::MAX)));
    let mut empty = u64::cast_from(-1i64);
    let count = control[GRAIN_FIRMS_WORD].load();
    let mut slot = UNIT_POS_PLANE;
    while slot < count {
        let seller = searches[(comptime!(SELLERS) * config.actors + slot) as usize];
        let supply = index::Supply::load(
            &searches[(comptime!(OFFERS) * config.actors) as usize..],
            slot,
        );
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
                let member = searches[(comptime!(MEMBERS) * config.actors + slot) as usize];
                let offset = (member + length - start) % length;
                let rank = u32::cast_from(
                    u64::cast_from(offset) * u64::cast_from(inverse) % u64::cast_from(length),
                );
                let score = native(offer.score);
                if offer.empty {
                    empty = min(empty, (score << 32) | u64::cast_from(rank));
                } else if offer.score.extract(HIGH_WORD) == 0 {
                    let mut item = (u64::cast_from(rank) << 32) | score;
                    if item < first[comptime!(DISCOVERY_CANDIDATES as usize - 1)] {
                        #[unroll]
                        for i in 0..DISCOVERY_CANDIDATES as usize {
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
    for _ in 0..DISCOVERY_CANDIDATES {
        let item = first[head];
        let rank = plane_min(u32::cast_from(item >> 32));
        if item != u64::MAX && u32::cast_from(item >> 32) == rank {
            full = min(full, (item << 32) | u64::cast_from(rank));
            head = min(head + 1, comptime!(DISCOVERY_CANDIDATES as usize - 1));
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
        let supply = index::Supply::new(economy, seller, comptime!(Good::Grain as u32));
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

pub(in crate::simulation) const DEALERS: u32 = 32;
const TILE: u32 = 256;

#[cube]
pub(super) fn tiled_dealer(
    economy: &EconomyView<'_>,
    spatial: &[u32],
    control: &[Atomic<u32>],
    jobs: u32,
    buyer: u32,
    cash: Ticks,
) -> bool {
    let quote = control[INPUT_MINIMUM_QUOTE_WORD].load();
    jobs <= comptime!(DEALERS)
        && spatial[1usize] >= 8192
        && cash.extract(HIGH_WORD) == 0
        && cash.extract(LOW_WORD) >= quote
        && cash.extract(LOW_WORD) / 8 <= quote
        && household_production::grain_dealer(economy, buyer)
}

#[derive(CubeType)]
struct DealerQuery {
    buyer: u32,
    wanted: Ticks,
    cash: Ticks,
    access: u32,
    first: u32,
    stride: u32,
    length: u32,
}

#[cube]
impl DealerQuery {
    fn new(
        actors: &ActorTable<'_>,
        parameters: &EconomyParameters,
        buyer: u32,
        wanted: Ticks,
        cash: Ticks,
        access: u32,
        length: u32,
    ) -> Self {
        DealerQuery {
            buyer,
            wanted,
            cash,
            access,
            length,
            first: (buyer + parameters.settlement_rotation) % length,
            stride: supplier_stride(actors, parameters, buyer, length),
        }
    }

    fn seller(&self, spatial: &[u32], rank: u32) -> u32 {
        let member = (u64::cast_from(self.first)
            + u64::cast_from(self.stride) * u64::cast_from(rank))
            % u64::cast_from(self.length);
        spatial[2 + member as usize]
    }

    fn offer(
        &self,
        economy: &EconomyView<'_>,
        spatial: &[u32],
        control: &[Atomic<u32>],
        parameters: &EconomyParameters,
        rank: u32,
    ) -> Candidate {
        let seller = self.seller(spatial, rank);
        let supply = index::Supply::new(economy, seller, comptime!(Good::Grain as u32));
        OfferRequest {
            buyer: self.buyer,
            seller,
            trip: economy.actors.distance(self.buyer, seller),
            buyer_access: self.access,
            demand: BuyerDemand {
                wanted: self.wanted,
                purchasing_power: self.cash,
            },
        }
        .with_supply(economy, control, parameters, &supply)
    }
}

#[cube]
fn tile_record(config: &EconomyConfig, job: u32, tile: u32, tiles: u32) -> usize {
    (13 * config.actors + 2 * (job * tiles + tile)) as usize
}

#[cube]
pub(super) fn dealer_tiles(
    economy: &EconomyView<'_>,
    spatial: &[u32],
    searches: &mut [u32],
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
) {
    let config = economy.ledger.config;
    let jobs = orders::job_count(control, SEARCHES_WORD);
    let job = CUBE_POS_Y;
    if jobs <= comptime!(DEALERS) && job < jobs {
        let buyer = searches[job as usize];
        let demand = BuyerDemand::load(&*searches, config.actors, buyer);
        let cash = demand.purchasing_power;
        if tiled_dealer(economy, spatial, control, jobs, buyer, cash) {
            let wanted = demand.wanted;
            let mut access = 0u32;
            if comptime!(config.railway.routes_enabled == Feature::Enabled)
                && config.railway.nodes > 0
            {
                access = config.railway.access[buyer as usize * 2 + 1];
            }
            let query = DealerQuery::new(
                &economy.actors,
                parameters,
                buyer,
                wanted,
                cash,
                access,
                spatial[1usize],
            );
            let tiles = ceildiv(query.length, comptime!(TILE));
            let mut tile = CUBE_POS_X;
            while tile < tiles {
                let rank = tile * comptime!(TILE) + UNIT_POS;
                let mut eligible = false;
                let mut empty = false;
                let mut score = ticks(u32::MAX, u32::MAX);
                if rank < query.length {
                    let offer = query.offer(economy, spatial, control, parameters, rank);
                    eligible = offer.eligible;
                    empty = offer.empty;
                    score = offer.score;
                }
                let full = parallel::reduce::<u32>(
                    u32::cast_from(eligible && score.extract(HIGH_WORD) == 0),
                    comptime!(Reduction::Sum),
                );
                let best = group_winner(
                    CandidateRank {
                        active: u32::cast_from(eligible),
                        score,
                    },
                    comptime!(parallel::ScanMode::Plane),
                );
                let fallback = group_winner(
                    CandidateRank {
                        active: u32::cast_from(empty),
                        score,
                    },
                    comptime!(parallel::ScanMode::Plane),
                );
                if UNIT_POS == 0 {
                    let record = tile_record(config, job, tile, tiles);
                    let lane = select(best == NO_ACTOR, fallback, best);
                    searches[record] = full;
                    searches[record + 1] =
                        select(lane == NO_ACTOR, NO_ACTOR, tile * comptime!(TILE) + lane);
                }
                tile += CUBE_COUNT_X;
            }
        }
    }
}

#[cube]
pub(super) fn dealer_trade(
    economy: &EconomyView<'_>,
    spatial: &[u32],
    searches: &[u32],
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
    job: u32,
    buyer: u32,
    wanted: Ticks,
    cash: Ticks,
    access: u32,
) -> Trade {
    let config = economy.ledger.config;
    let query = DealerQuery::new(
        &economy.actors,
        parameters,
        buyer,
        wanted,
        cash,
        access,
        spatial[1usize],
    );
    let tiles = ceildiv(query.length, comptime!(TILE));
    let mut remaining = u32::new(comptime!(DISCOVERY_CANDIDATES as i64));
    let mut category = 2u32;
    let mut best = u64::new(-1);
    let mut winner = u32::new(comptime!(i64::from(NO_ACTOR)));
    let mut block = 0u32;
    while block < tiles && remaining > 0 {
        let tile = block + UNIT_POS_PLANE;
        let record = tile_record(config, job, tile, tiles);
        let mut full = 0u32;
        let mut rank = u32::new(comptime!(i64::from(NO_ACTOR)));
        if tile < tiles {
            full = searches[record];
            rank = searches[record + 1];
        }
        let preceding = plane_exclusive_sum(full);
        let included = remaining - min(preceding, remaining);
        let mut end = rank;
        let mut seen = 0u32;
        if rank != NO_ACTOR {
            end = rank + 1;
            if full > included {
                rank = tile * comptime!(TILE);
                end = min(rank + comptime!(TILE), query.length);
            }
        }
        while rank < end && seen < included {
            let offer = query.offer(economy, spatial, control, parameters, rank);
            let class = select(offer.eligible, 0u32, select(offer.empty, 1u32, 2u32));
            let score = native(offer.score);
            if class < 2
                && (class < category
                    || (class == category && (score < best || (score == best && rank < winner))))
            {
                best = score;
                category = class;
                winner = rank;
            }
            seen += u32::cast_from(offer.eligible && offer.score.extract(HIGH_WORD) == 0);
            rank += 1;
        }
        remaining -= min(remaining, plane_sum(full));
        block += PLANE_DIM;
    }
    let class = plane_min(category);
    let high = plane_min(select(
        category == class,
        u32::cast_from(best >> 32),
        u32::MAX,
    ));
    let low = plane_min(select(
        category == class && u32::cast_from(best >> 32) == high,
        u32::cast_from(best),
        u32::MAX,
    ));
    let rank = plane_min(select(
        category == class && best == ((u64::cast_from(high) << 32) | u64::cast_from(low)),
        winner,
        NO_ACTOR,
    ));
    let mut trade = Trade {
        seller: NO_ACTOR,
        quantity: zero(),
        trip: 0,
    };
    if rank != NO_ACTOR {
        let offer = query.offer(economy, spatial, control, parameters, rank);
        trade.seller = query.seller(spatial, rank);
        trade.quantity = offer.quantity;
        trade.trip = offer.trip;
    }
    trade
}

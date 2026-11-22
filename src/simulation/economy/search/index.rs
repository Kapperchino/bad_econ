use super::*;

const FIELDS: usize = 10;
pub(super) const BLOCK: u32 = 32;
pub(super) const HOUSEHOLD_STRIDE_FIELD: u32 = 6;
pub(super) const HOUSEHOLD_X_FIELD: u32 = 8;
pub(super) const HOUSEHOLD_Y_FIELD: u32 = 9;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(super) enum SupplySource {
    State,
    Grain,
    Index,
}

#[derive(Clone, Copy)]
enum BlockField {
    StockedPrice,
    EmptyPrice,
    MinX,
    MaxX,
    MinY,
    MaxY,
    Stocked,
    MinAccess,
    MaximumAvailable,
    MaximumQuote,
}

#[derive(Clone, Copy)]
enum SupplyField {
    Quote,
    AvailableLow,
    AvailableHigh,
    Flags,
}

#[derive(CubeType, Clone, Copy)]
pub(super) struct Supply {
    pub quote: u32,
    pub available: Ticks,
    pub offered: bool,
    pub producer: bool,
}

#[cube]
impl Supply {
    pub fn for_search(
        state: &mut [u32],
        actors: &[u32],
        config: &EconomyConfig,
        scratch: &[u32],
        parameters: &EconomyParameters,
        member: u32,
        actor: u32,
        #[comptime] source: SupplySource,
    ) -> Self {
        if comptime!(
            source == SupplySource::Index
                || matches!(
                    parameters.market,
                    MarketFlag::Consumer(_) | MarketFlag::Household
                )
        ) {
            Self::load(scratch, member)
        } else {
            Self::new(state, actors, config, actor, parameters.traded_good())
        }
    }

    pub fn new(
        state: &mut [u32],
        actors: &[u32],
        config: &EconomyConfig,
        actor: u32,
        good: u32,
    ) -> Self {
        let kind = kind(actors, actor);
        let producer = !kind.is(comptime!(ActorKind::Household)) && kind.output() == good;
        let supplier = household_production::supplier(state, actors, config, actor, good);
        let terms = household_production::TradeFields::new(actors, actor, good);
        let available = positive(
            value(state, config, good, actor),
            household_production::reserve(state, actors, config, actor, good),
        );
        Supply {
            quote: number(state, config, terms.price, actor),
            available,
            offered: supplier && !equal(available, zero()),
            producer,
        }
    }

    pub(super) fn store(&self, scratch: &mut [u32], member: u32) {
        let base = member * 4;
        scratch[(base + comptime!(SupplyField::Quote as u32)) as usize] = self.quote;
        scratch[(base + comptime!(SupplyField::AvailableLow as u32)) as usize] =
            self.available.extract(LOW_WORD);
        scratch[(base + comptime!(SupplyField::AvailableHigh as u32)) as usize] =
            self.available.extract(HIGH_WORD);
        scratch[(base + comptime!(SupplyField::Flags as u32)) as usize] =
            u32::cast_from(self.offered) | (u32::cast_from(self.producer) << 1);
    }

    pub fn load(scratch: &[u32], member: u32) -> Self {
        let base = member * 4;
        let flags = scratch[(base + comptime!(SupplyField::Flags as u32)) as usize];
        Supply {
            quote: scratch[(base + comptime!(SupplyField::Quote as u32)) as usize],
            available: ticks(
                scratch[(base + comptime!(SupplyField::AvailableLow as u32)) as usize],
                scratch[(base + comptime!(SupplyField::AvailableHigh as u32)) as usize],
            ),
            offered: flags & 1 != 0,
            producer: flags & 2 != 0,
        }
    }
}

#[cube]
fn reduce_minimum(value: u32, #[comptime] reduction: parallel::ScanMode) -> u32 {
    match comptime!(reduction) {
        parallel::ScanMode::Shared => parallel::reduce::<u32>(value, comptime!(Reduction::Minimum)),
        parallel::ScanMode::Plane => plane_min(value),
    }
}

#[cube]
pub(super) fn build(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    spatial: &mut [u32],
    scratch: &mut [u32],
    searches: &mut [u32],
    successes: &[Atomic<u32>],
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
    #[comptime] collection: OrderCollection,
) {
    let reduction = comptime!(parameters.scan);
    let cells = config.spatial_side * config.spatial_side;
    let count = spatial[cells as usize];
    let start = CUBE_POS as u32 * (DISCOVERY_BLOCK / BLOCK);
    let end = min(start + DISCOVERY_BLOCK / BLOCK, ceildiv(count, BLOCK));
    let width = match comptime!(reduction) {
        parallel::ScanMode::Plane => min(PLANE_DIM, CUBE_DIM),
        parallel::ScanMode::Shared => CUBE_DIM,
    };
    let lane = match comptime!(reduction) {
        parallel::ScanMode::Plane => UNIT_POS_PLANE,
        parallel::ScanMode::Shared => UNIT_POS,
    };
    let step = ceildiv(CUBE_DIM, width);
    let mut block = start
        + match comptime!(reduction) {
            parallel::ScanMode::Plane => PLANE_POS,
            parallel::ScanMode::Shared => 0u32,
        };
    let good = parameters.traded_good();
    let mut minimum_quote = u32::new(comptime!(i64::from(u32::MAX)));
    let mut minimum_eligible = u32::new(comptime!(i64::from(u32::MAX)));
    while block < end {
        let mut stocked_price = u32::new(comptime!(i64::from(u32::MAX)));
        let mut empty_price = u32::new(comptime!(i64::from(u32::MAX)));
        let mut min_x = u32::new(comptime!(i64::from(u32::MAX)));
        let mut max_x = 0u32;
        let mut min_y = u32::new(comptime!(i64::from(u32::MAX)));
        let mut max_y = 0u32;
        let mut stocked = 0u32;
        let mut maximum_available = 0u32;
        let mut maximum_quote = 0u32;
        let mut min_access = u32::new(comptime!(i64::from(u32::MAX)));
        let mut member = block * BLOCK + lane;
        while member < min((block + 1) * BLOCK, count) {
            let actor = spatial[(cells + 1 + member) as usize];
            if comptime!(
                parameters.empty_orders
                    == crate::simulation::parameters::EmptyOrderMode::FeedbackOnly
                    && matches!(parameters.market, MarketFlag::Consumer(_))
            ) {
                successes[markets::filtered_count(config.actors, actor) as usize].store(0);
                if comptime!(collection == OrderCollection::ConsumerMask) {
                    let slot = filter::prefix_slot(successes, config.actors, actor);
                    successes[slot].store(0);
                    successes[slot + 1].store(0);
                }
            }
            let supply = Supply::new(state, actors, config, actor, good);
            supply.store(scratch, member);
            let quote = supply.quote;
            if good == comptime!(Good::Bread as u32) {
                minimum_quote = min(minimum_quote, quote);
            }
            let offered = supply.offered;
            if comptime!(matches!(parameters.market, MarketFlag::Input(_)))
                && good == comptime!(Good::Bread as u32)
                && quote > 0
                && (offered || supply.producer)
            {
                minimum_eligible = min(minimum_eligible, quote);
            }
            maximum_quote = max(maximum_quote, select(offered, quote, 0));
            maximum_available = max(
                maximum_available,
                select(
                    supply.available.extract(HIGH_WORD) > 0,
                    u32::MAX,
                    supply.available.extract(LOW_WORD),
                ),
            );
            let producer = supply.producer;
            let price = select(quote > 0, quote, u32::MAX);
            stocked_price = min(stocked_price, select(offered, price, u32::MAX));
            empty_price = min(empty_price, select(producer && !offered, price, u32::MAX));
            stocked = max(stocked, u32::cast_from(offered));
            let stride = actors.len() / ActorLayout::WORDS;
            let x = actors[ActorLayout::X * stride + actor as usize];
            let y = actors[ActorLayout::Y * stride + actor as usize];
            if comptime!(parameters.market == MarketFlag::Household) {
                searches[(HOUSEHOLD_X_FIELD * config.actors + member) as usize] = x;
                searches[(HOUSEHOLD_Y_FIELD * config.actors + member) as usize] = y;
                let cell =
                    y / config.search_radius * config.spatial_side + x / config.search_radius;
                let begin = spatial[cell as usize];
                let length = spatial[(cell + 1) as usize] - begin;
                let offset = member - begin;
                if length > HOUSEHOLD_SEARCH_SAMPLES && offset < length / 2 {
                    searches[(HOUSEHOLD_STRIDE_FIELD * config.actors + member) as usize] =
                        coprime_stride(length, offset);
                }
            }
            if comptime!(matches!(parameters.market, MarketFlag::Consumer(_))) {
                scratch[(5 * config.actors + member) as usize] = empty::key(config, x, y);
            }
            min_x = min(min_x, x);
            max_x = max(max_x, x);
            min_y = min(min_y, y);
            max_y = max(max_y, y);
            if comptime!(config.railway.routes_enabled == Feature::Enabled)
                && config.railway.nodes > 0
            {
                min_access = min(min_access, config.railway.access[actor as usize * 2 + 1]);
            }
            member += width;
        }
        stocked_price = reduce_minimum(stocked_price, comptime!(reduction));
        empty_price = reduce_minimum(empty_price, comptime!(reduction));
        min_x = reduce_minimum(min_x, comptime!(reduction));
        max_x = !reduce_minimum(!max_x, comptime!(reduction));
        min_y = reduce_minimum(min_y, comptime!(reduction));
        max_y = !reduce_minimum(!max_y, comptime!(reduction));
        stocked = !reduce_minimum(!stocked, comptime!(reduction));
        maximum_available = !reduce_minimum(!maximum_available, comptime!(reduction));
        maximum_quote = !reduce_minimum(!maximum_quote, comptime!(reduction));
        if comptime!(config.railway.routes_enabled == Feature::Enabled) && config.railway.nodes > 0
        {
            min_access = reduce_minimum(min_access, comptime!(reduction));
        }
        if lane == 0 {
            let base = 6 * config.actors as usize + block as usize * FIELDS;
            scratch[base + comptime!(BlockField::StockedPrice as usize)] = stocked_price;
            scratch[base + comptime!(BlockField::EmptyPrice as usize)] = empty_price;
            scratch[base + comptime!(BlockField::MinX as usize)] = min_x;
            scratch[base + comptime!(BlockField::MaxX as usize)] = max_x;
            scratch[base + comptime!(BlockField::MinY as usize)] = min_y;
            scratch[base + comptime!(BlockField::MaxY as usize)] = max_y;
            scratch[base + comptime!(BlockField::Stocked as usize)] = stocked;
            scratch[base + comptime!(BlockField::MinAccess as usize)] = min_access;
            scratch[base + comptime!(BlockField::MaximumAvailable as usize)] = maximum_available;
            scratch[base + comptime!(BlockField::MaximumQuote as usize)] = maximum_quote;
        }
        block += step;
    }
    if comptime!(matches!(parameters.market, MarketFlag::Input(_)))
        && good == comptime!(Good::Bread as u32)
    {
        let minimum = parallel::reduce::<u32>(minimum_eligible, comptime!(Reduction::Minimum));
        if UNIT_POS == 0 && minimum < control[INPUT_MINIMUM_QUOTE_WORD].load() {
            control[INPUT_MINIMUM_QUOTE_WORD].fetch_min(minimum);
        }
    }
    if good == comptime!(Good::Bread as u32) {
        let minimum = parallel::reduce::<u32>(minimum_quote, comptime!(Reduction::Minimum));
        if UNIT_POS == 0 && CUBE_POS as u32 * DISCOVERY_BLOCK < count {
            let base = 3 * (cells + 1 + config.actors);
            spatial[(base + CUBE_POS as u32) as usize] = minimum;
        }
    }
}

#[cube]
pub(super) fn maximum_available(scratch: &[u32], config: &EconomyConfig, block: u32) -> u32 {
    let base = 6 * config.actors as usize + block as usize * FIELDS;
    scratch[base + comptime!(BlockField::MaximumAvailable as usize)]
}

#[cube]
pub(super) fn partial_input_price_floor(
    scratch: &[u32],
    config: &EconomyConfig,
    block: u32,
    x: u32,
    y: u32,
    buyer_access: u32,
    maximum_delivery_cost: Ticks,
) -> u32 {
    let base = 6 * config.actors as usize + block as usize * FIELDS;
    let minimum_quote = scratch[base + comptime!(BlockField::StockedPrice as usize)];
    let maximum_quote = scratch[base + comptime!(BlockField::MaximumQuote as usize)];
    let min_x = scratch[base + comptime!(BlockField::MinX as usize)];
    let max_x = scratch[base + comptime!(BlockField::MaxX as usize)];
    let min_y = scratch[base + comptime!(BlockField::MinY as usize)];
    let max_y = scratch[base + comptime!(BlockField::MaxY as usize)];
    let distance = max(x, min_x) - min(x, max_x) + max(y, min_y) - min(y, max_y);
    let mut days = max(ceildiv(distance, config.shipment_speed), 1);
    if comptime!(config.railway.routes_enabled == Feature::Enabled) && config.railway.nodes > 0 {
        let seller_access = scratch[base + comptime!(BlockField::MinAccess as usize)];
        days = min(
            days,
            ceildiv(buyer_access + seller_access, config.shipment_speed) + 1,
        );
    }
    let lower = plus(
        ticks(minimum_quote, 0),
        times32(days, config.delivery_penalty),
    )
    .value;
    let upper = plus(ticks(maximum_quote, 0), maximum_delivery_cost).value;
    select(upper.extract(HIGH_WORD) == 0, lower.extract(LOW_WORD), 0)
}

#[cube]
pub(super) fn skip(
    scratch: &[u32],
    config: &EconomyConfig,
    block: u32,
    x: u32,
    y: u32,
    buyer_access: u32,
    cash: Ticks,
    chosen: u32,
    empty_score: Ticks,
    visibility: u32,
) -> bool {
    let base = 6 * config.actors as usize + block as usize * FIELDS;
    let stocked_price = scratch[base + comptime!(BlockField::StockedPrice as usize)];
    let empty_price = scratch[base + comptime!(BlockField::EmptyPrice as usize)];
    let min_x = scratch[base + comptime!(BlockField::MinX as usize)];
    let max_x = scratch[base + comptime!(BlockField::MaxX as usize)];
    let min_y = scratch[base + comptime!(BlockField::MinY as usize)];
    let max_y = scratch[base + comptime!(BlockField::MaxY as usize)];
    let stocked = scratch[base + comptime!(BlockField::Stocked as usize)];
    let distance = max(x, min_x) - min(x, max_x) + max(y, min_y) - min(y, max_y);
    let mut days = max(ceildiv(distance, config.shipment_speed), 1);
    if comptime!(config.railway.routes_enabled == Feature::Enabled) && config.railway.nodes > 0 {
        let seller_access = scratch[base + comptime!(BlockField::MinAccess as usize)];
        let rail_days = ceildiv(buyer_access + seller_access, config.shipment_speed) + 1;
        days = min(days, rail_days);
    }
    let delivery = times32(days, config.delivery_penalty);
    let lower_score = plus(ticks(empty_price, 0), delivery).value;
    let stock_possible = stocked_price != u32::MAX && !lt(cash, ticks(stocked_price, 0));
    let empty_improves = chosen == NO_ACTOR
        && empty_price != u32::MAX
        && !lt(cash, ticks(empty_price, 0))
        && lt(lower_score, empty_score);
    let unseen_stock = stocked > 0 && visibility == 0;
    distance > config.search_radius || !(stock_possible || empty_improves || unseen_stock)
}

#[cube]
pub(super) fn skip_stocked(
    scratch: &[u32],
    config: &EconomyConfig,
    block: u32,
    x: u32,
    y: u32,
    cash: Ticks,
) -> bool {
    let base = 6 * config.actors as usize + block as usize * FIELDS;
    let price = scratch[base + comptime!(BlockField::StockedPrice as usize)];
    let min_x = scratch[base + comptime!(BlockField::MinX as usize)];
    let max_x = scratch[base + comptime!(BlockField::MaxX as usize)];
    let min_y = scratch[base + comptime!(BlockField::MinY as usize)];
    let max_y = scratch[base + comptime!(BlockField::MaxY as usize)];
    let distance = max(x, min_x) - min(x, max_x) + max(y, min_y) - min(y, max_y);
    distance > config.search_radius || price == u32::MAX || lt(cash, ticks(price, 0))
}

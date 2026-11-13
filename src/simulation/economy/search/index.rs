use super::*;

const FIELDS: usize = 8;
pub(super) const BLOCK: u32 = 32;
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
    pub fn new(
        state: &mut [u32],
        actors: &[u32],
        config: &EconomyConfig,
        actor: u32,
        good: u32,
    ) -> Self {
        let kind = kind(actors, actor);
        let producer = !kind.is(comptime!(ActorKind::Household)) && kind.output() == good;
        let supplier = kind.output() == good
            || (kind.is(comptime!(ActorKind::Household)) && good == comptime!(Good::Bread as u32));
        let available = positive(
            value(state, config, good, actor),
            household_production::reserve(state, actors, config, actor, good),
        );
        Supply {
            quote: number(state, config, comptime!(Field::Price as u32), actor),
            available,
            offered: supplier && !equal(available, zero()),
            producer,
        }
    }

    fn store(&self, scratch: &mut [u32], config: &EconomyConfig, member: u32) {
        let count = config.actors;
        scratch[(comptime!(SupplyField::Quote as u32) * count + member) as usize] = self.quote;
        scratch[(comptime!(SupplyField::AvailableLow as u32) * count + member) as usize] =
            self.available.extract(LOW_WORD);
        scratch[(comptime!(SupplyField::AvailableHigh as u32) * count + member) as usize] =
            self.available.extract(HIGH_WORD);
        scratch[(comptime!(SupplyField::Flags as u32) * count + member) as usize] =
            u32::cast_from(self.offered) | (u32::cast_from(self.producer) << 1);
    }

    pub fn load(scratch: &[u32], config: &EconomyConfig, member: u32) -> Self {
        let count = config.actors;
        let flags = scratch[(comptime!(SupplyField::Flags as u32) * count + member) as usize];
        Supply {
            quote: scratch[(comptime!(SupplyField::Quote as u32) * count + member) as usize],
            available: ticks(
                scratch[(comptime!(SupplyField::AvailableLow as u32) * count + member) as usize],
                scratch[(comptime!(SupplyField::AvailableHigh as u32) * count + member) as usize],
            ),
            offered: flags & 1 != 0,
            producer: flags & 2 != 0,
        }
    }
}

#[derive(Clone, Copy)]
enum IndexReduction {
    Workgroup,
    Plane,
}

#[cube]
fn reduce_minimum(value: u32, #[comptime] reduction: IndexReduction) -> u32 {
    match comptime!(reduction) {
        IndexReduction::Workgroup => parallel::reduce(value, comptime!(Reduction::Minimum)),
        IndexReduction::Plane => plane_min(value),
    }
}

#[cube]
pub(super) fn build(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    spatial: &mut [u32],
    scratch: &mut [u32],
    parameters: &EconomyParameters,
) {
    let cooperative = comptime!(parameters.scan == crate::simulation::parallel::ScanMode::Plane)
        && PLANE_DIM == 32;
    if cooperative {
        build_blocks(
            state,
            actors,
            config,
            spatial,
            scratch,
            parameters,
            comptime!(IndexReduction::Plane),
        );
    } else {
        build_blocks(
            state,
            actors,
            config,
            spatial,
            scratch,
            parameters,
            comptime!(IndexReduction::Workgroup),
        );
    }
}

#[cube]
fn build_blocks(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    spatial: &mut [u32],
    scratch: &mut [u32],
    parameters: &EconomyParameters,
    #[comptime] reduction: IndexReduction,
) {
    let cells = config.spatial_side * config.spatial_side;
    let count = spatial[cells as usize];
    let start = CUBE_POS as u32 * (DISCOVERY_BLOCK / BLOCK);
    let end = min(start + DISCOVERY_BLOCK / BLOCK, ceildiv(count, BLOCK));
    let width = match comptime!(reduction) {
        IndexReduction::Plane => 32u32,
        IndexReduction::Workgroup => CUBE_DIM,
    };
    let lane = UNIT_POS % width;
    let step = CUBE_DIM / width;
    let mut block = start + UNIT_POS / width;
    let good = parameters.traded_good();
    let mut minimum_quote = u32::new(comptime!(i64::from(u32::MAX)));
    while block < end {
        let mut stocked_price = u32::new(comptime!(i64::from(u32::MAX)));
        let mut empty_price = u32::new(comptime!(i64::from(u32::MAX)));
        let mut min_x = u32::new(comptime!(i64::from(u32::MAX)));
        let mut max_x = 0u32;
        let mut min_y = u32::new(comptime!(i64::from(u32::MAX)));
        let mut max_y = 0u32;
        let mut stocked = 0u32;
        let mut min_access = u32::new(comptime!(i64::from(u32::MAX)));
        let mut member = block * BLOCK + lane;
        while member < min((block + 1) * BLOCK, count) {
            let actor = spatial[(cells + 1 + member) as usize];
            let supply = Supply::new(state, actors, config, actor, good);
            supply.store(scratch, config, member);
            let quote = supply.quote;
            if good == comptime!(Good::Bread as u32) {
                minimum_quote = min(minimum_quote, quote);
            }
            let offered = supply.offered;
            let producer = supply.producer;
            let price = select(quote > 0, quote, u32::MAX);
            stocked_price = min(stocked_price, select(offered, price, u32::MAX));
            empty_price = min(empty_price, select(producer && !offered, price, u32::MAX));
            stocked = max(stocked, u32::cast_from(offered));
            let stride = actors.len() / ActorLayout::WORDS;
            let x = actors[ActorLayout::X * stride + actor as usize];
            let y = actors[ActorLayout::Y * stride + actor as usize];
            if comptime!(parameters.market == MarketFlag::Consumer) {
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
        }
        block += step;
    }
    if good == comptime!(Good::Bread as u32) {
        let minimum = parallel::reduce(minimum_quote, comptime!(Reduction::Minimum));
        if UNIT_POS == 0 && CUBE_POS as u32 * DISCOVERY_BLOCK < count {
            let base = 3 * (cells + 1 + config.actors);
            spatial[(base + CUBE_POS as u32) as usize] = minimum;
        }
    }
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

use super::*;

const FIELDS: usize = 6;

#[derive(Clone, Copy)]
enum BlockField {
    StockedPrice,
    EmptyPrice,
    MinX,
    MaxX,
    MinY,
    MaxY,
}

#[cube]
pub(super) fn build(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    spatial: &[u32],
    scratch: &mut [u32],
    parameters: &EconomyParameters,
) {
    let cells = config.spatial_side * config.spatial_side;
    let count = spatial[cells as usize];
    let block = CUBE_POS as u32;
    if block * DISCOVERY_BLOCK < count {
        let mut stocked_price = u32::new(comptime!(i64::from(u32::MAX)));
        let mut empty_price = u32::new(comptime!(i64::from(u32::MAX)));
        let mut min_x = u32::new(comptime!(i64::from(u32::MAX)));
        let mut max_x = 0u32;
        let mut min_y = u32::new(comptime!(i64::from(u32::MAX)));
        let mut max_y = 0u32;
        let good = comptime!(parameters.market.good().expect("consumer good") as u32);
        let mut member = block * DISCOVERY_BLOCK + UNIT_POS;
        while member < min((block + 1) * DISCOVERY_BLOCK, count) {
            let actor = spatial[(cells + 1 + member) as usize];
            let quote = number(state, config, comptime!(Field::Price as u32), actor);
            let offered = household_production::offers(state, actors, config, actor, good);
            let producer = !kind(actors, actor).is(comptime!(ActorKind::Household))
                && kind(actors, actor).output() == good;
            let price = select(quote > 0, quote, u32::MAX);
            stocked_price = min(stocked_price, select(offered, price, u32::MAX));
            empty_price = min(empty_price, select(producer && !offered, price, u32::MAX));
            let stride = actors.len() / ActorLayout::WORDS;
            let x = actors[ActorLayout::X * stride + actor as usize];
            let y = actors[ActorLayout::Y * stride + actor as usize];
            min_x = min(min_x, x);
            max_x = max(max_x, x);
            min_y = min(min_y, y);
            max_y = max(max_y, y);
            member += CUBE_DIM;
        }
        stocked_price = parallel::reduce(stocked_price, comptime!(Reduction::Minimum));
        empty_price = parallel::reduce(empty_price, comptime!(Reduction::Minimum));
        min_x = parallel::reduce(min_x, comptime!(Reduction::Minimum));
        max_x = !parallel::reduce(!max_x, comptime!(Reduction::Minimum));
        min_y = parallel::reduce(min_y, comptime!(Reduction::Minimum));
        max_y = !parallel::reduce(!max_y, comptime!(Reduction::Minimum));
        if UNIT_POS == 0 {
            let base = 6 * config.actors as usize + block as usize * FIELDS;
            scratch[base + comptime!(BlockField::StockedPrice as usize)] = stocked_price;
            scratch[base + comptime!(BlockField::EmptyPrice as usize)] = empty_price;
            scratch[base + comptime!(BlockField::MinX as usize)] = min_x;
            scratch[base + comptime!(BlockField::MaxX as usize)] = max_x;
            scratch[base + comptime!(BlockField::MinY as usize)] = min_y;
            scratch[base + comptime!(BlockField::MaxY as usize)] = max_y;
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
    cash: Ticks,
    chosen: u32,
    empty_score: Ticks,
) -> bool {
    let base = 6 * config.actors as usize + block as usize * FIELDS;
    let stocked_price = scratch[base + comptime!(BlockField::StockedPrice as usize)];
    let empty_price = scratch[base + comptime!(BlockField::EmptyPrice as usize)];
    let min_x = scratch[base + comptime!(BlockField::MinX as usize)];
    let max_x = scratch[base + comptime!(BlockField::MaxX as usize)];
    let min_y = scratch[base + comptime!(BlockField::MinY as usize)];
    let max_y = scratch[base + comptime!(BlockField::MaxY as usize)];
    let distance = max(x, min_x) - min(x, max_x) + max(y, min_y) - min(y, max_y);
    let delivery = times32(
        max(
            ceildiv(
                distance,
                config.shipment_speed * config.railway.speed_multiplier,
            ),
            1,
        ),
        config.delivery_penalty,
    );
    let lower_score = plus(ticks(empty_price, 0), delivery).value;
    let stock_possible = stocked_price != u32::MAX && !lt(cash, ticks(stocked_price, 0));
    let empty_improves = chosen == NO_ACTOR
        && empty_price != u32::MAX
        && !lt(cash, ticks(empty_price, 0))
        && lt(lower_score, empty_score);
    distance > config.search_radius || !(stock_possible || empty_improves)
}

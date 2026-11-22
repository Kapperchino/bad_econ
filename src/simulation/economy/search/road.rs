use super::*;

pub(in crate::simulation) const SIDE: u32 = 32;
const CAPACITY: u32 = 128;
const STATIONS: usize = 256;
const RECORD: u32 = CAPACITY + 1;

#[cube]
fn nearest(start: u32, end: u32, coordinate: u32) -> u32 {
    max(start, coordinate) - min(end, coordinate)
}

#[cube]
fn farthest_axis(start: u32, end: u32, coordinate: u32) -> u32 {
    max(
        max(start, coordinate) - min(start, coordinate),
        max(end, coordinate) - min(end, coordinate),
    )
}

#[cube]
fn base(config: &EconomyConfig, tile: u32) -> u32 {
    17 * config.actors + empty::LIMIT + tile * comptime!(RECORD)
}

#[cube]
pub(super) fn prepare(
    actors: &ActorTable<'_>,
    config: &EconomyConfig,
    spatial: &[u32],
    scratch: &[u32],
    searches: &mut [u32],
    control: &[Atomic<u32>],
) {
    let tile = CUBE_POS as u32;
    let count = spatial[1usize];
    let start = base(config, tile);
    let active = count > 0
        && count <= empty::LIMIT
        && control[SEARCHES_WORD].load() > config.actors / 16
        && empty::stocked_count(config, scratch, count) <= 8;
    if UNIT_POS == 0 {
        searches[start as usize] = u32::MAX;
    }
    if active {
        let span = ceildiv(config.map_size, comptime!(SIDE));
        let x = (tile % comptime!(SIDE)) * span;
        let y = (tile / comptime!(SIDE)) * span;
        let last_x = min(x + span - 1, config.map_size - 1);
        let last_y = min(y + span - 1, config.map_size - 1);
        let max_days = max(ceildiv(config.search_radius, config.shipment_speed), 1);
        let surcharge = u64::cast_from(max_days)
            * (u64::cast_from(config.delivery_penalty)
                + u64::cast_from(config.starting_wage)
                    * comptime!(crate::domain::railway::CART_BILLABLE_MINUTES_PER_UNIT_DAY as u64));
        let mut station_floor = Shared::<[u32]>::new_slice(STATIONS);
        if comptime!(config.railway.routes_enabled == Feature::Enabled)
            && config.railway.nodes > 0
            && config.railway.nodes <= comptime!(STATIONS as u32)
        {
            let mut near_station = Shared::<[u32]>::new_slice(STATIONS);
            let mut farthest = u32::new(comptime!(i64::from(u32::MAX)));
            let mut station = UNIT_POS;
            while station < config.railway.nodes {
                let sx = config.railway.stations[station as usize * 2];
                let sy = config.railway.stations[station as usize * 2 + 1];
                near_station[station as usize] = nearest(x, last_x, sx) + nearest(y, last_y, sy);
                farthest = min(
                    farthest,
                    farthest_axis(x, last_x, sx) + farthest_axis(y, last_y, sy),
                );
                station += CUBE_DIM;
            }
            let reachable = parallel::reduce::<u32>(farthest, comptime!(Reduction::Minimum));
            sync_cube();
            let mut origin = UNIT_POS;
            while origin < config.railway.nodes {
                let mut access = u32::new(comptime!(i64::from(u32::MAX)));
                for dest in 0..config.railway.nodes {
                    let last =
                        ((origin * config.railway.nodes + dest + 1) * config.ring - 1) as usize;
                    if near_station[dest as usize] <= reachable
                        && config.railway.quotes[last] < config.railway.company_count
                    {
                        access = min(access, near_station[dest as usize]);
                    }
                }
                station_floor[origin as usize] = access;
                origin += CUBE_DIM;
            }
            sync_cube();
        }
        let mut best = u64::new(-1);
        let mut member = UNIT_POS;
        while member < count {
            let supply = index::Supply::load(scratch, member);
            if supply.producer && !supply.offered && supply.quote > 0 {
                let seller = spatial[(2 + member) as usize];
                let sx = actors.x(seller);
                let sy = actors.y(seller);
                let far = farthest_axis(x, last_x, sx) + farthest_axis(y, last_y, sy);
                if far <= config.search_radius {
                    let days = max(ceildiv(far, config.shipment_speed), 1);
                    let price = cartage::delivered_price(config, supply.quote, far, days);
                    let score = price + days * config.delivery_penalty;
                    if u64::cast_from(supply.quote) + surcharge <= u64::cast_from(u32::MAX) {
                        best = min(best, (u64::cast_from(price) << 32) | u64::cast_from(score));
                    }
                }
            }
            member += CUBE_DIM;
        }
        let bound = parallel::reduce::<u64>(best, comptime!(Reduction::Minimum));
        let mut stored = 0u32;
        let mut block = 0u32;
        while block < count {
            let member = block + UNIT_POS;
            let mut candidate = false;
            if member < count {
                let supply = index::Supply::load(scratch, member);
                if supply.producer && !supply.offered && supply.quote > 0 {
                    let seller = spatial[(2 + member) as usize];
                    let sx = actors.x(seller);
                    let sy = actors.y(seller);
                    let near = nearest(x, last_x, sx) + nearest(y, last_y, sy);
                    let mut days = max(ceildiv(near, config.shipment_speed), 1);
                    let mut price = cartage::delivered_price(config, supply.quote, near, days);
                    if comptime!(config.railway.routes_enabled == Feature::Enabled)
                        && config.railway.nodes > 0
                        && config.railway.nodes <= comptime!(STATIONS as u32)
                    {
                        let origin = config.railway.access[seller as usize * 2];
                        let access = station_floor[origin as usize];
                        if access < u32::MAX {
                            let floor = ceildiv(
                                access + config.railway.access[seller as usize * 2 + 1],
                                config.shipment_speed,
                            ) + 1;
                            let far = farthest_axis(x, last_x, sx) + farthest_axis(y, last_y, sy);
                            if floor < max(ceildiv(far, config.shipment_speed), 1) {
                                days = min(days, floor);
                                price = supply.quote;
                            }
                        }
                    }
                    let score = price + days * config.delivery_penalty;
                    let max_cost = u64::cast_from(supply.quote) + surcharge;
                    candidate = near <= config.search_radius
                        && (max_cost > u64::cast_from(u32::MAX)
                            || price < u32::cast_from(bound >> 32)
                            || score <= u32::cast_from(bound));
                }
            }
            let prefix = parallel::scan_prefix(
                u32::cast_from(candidate),
                comptime!(parallel::ScanMode::Plane),
            );
            let slot = stored + prefix.inclusive;
            if candidate && slot <= comptime!(CAPACITY) {
                searches[(start + slot) as usize] = member;
            }
            stored += prefix.total;
            block += CUBE_DIM;
        }
        if UNIT_POS == 0 {
            searches[start as usize] = stored;
        }
    }
}

#[cube]
pub(super) fn candidates(config: &EconomyConfig, x: u32, y: u32) -> u32 {
    let span = ceildiv(config.map_size, comptime!(SIDE));
    base(config, (y / span) * comptime!(SIDE) + x / span)
}

#[cube]
pub(super) fn available(config: &EconomyConfig, searches: &[u32], start: u32, buyer: u32) -> bool {
    searches[start as usize] <= comptime!(CAPACITY)
        && (config.railway.nodes <= comptime!(STATIONS as u32)
            || !railway::serves_buyer(config, buyer))
}

#[cube]
pub(super) fn query(
    actors: &ActorTable<'_>,
    config: &EconomyConfig,
    spatial: &[u32],
    scratch: &[u32],
    searches: &[u32],
    start: u32,
    buyer: u32,
    rotation: u32,
    cash: Ticks,
    buyer_access: u32,
) -> empty::Offer {
    let count = spatial[1usize];
    let offset = (rotation + buyer) % max(count, 1);
    let mut best = u64::new(-1);
    let mut chosen = u32::new(comptime!(i64::from(NO_ACTOR)));
    let mut trip = 0u32;
    let inbound = railway::serves_buyer(config, buyer);
    for slot in 0..searches[start as usize] {
        let member = searches[(start + 1 + slot) as usize];
        let seller = spatial[(2 + member) as usize];
        let distance = actors.distance(buyer, seller);
        let mut days = max(ceildiv(distance, config.shipment_speed), 1);
        if inbound {
            days = railway::delivery_days(
                config,
                railway::DeliveryRequest {
                    buyer,
                    seller,
                    trip: distance,
                    buyer_access,
                },
            );
        }
        let price = cartage::delivered_price(
            config,
            index::Supply::load(scratch, member).quote,
            distance,
            days,
        );
        let score = price + days * config.delivery_penalty;
        let rank = select(member >= offset, member - offset, member + count - offset);
        let key = (u64::cast_from(score) << 32) | u64::cast_from(rank);
        if distance <= config.search_radius && !lt(cash, ticks(price, 0)) && key < best {
            best = key;
            chosen = seller;
            trip = distance;
        }
    }
    empty::Offer {
        actor: chosen,
        trip,
    }
}

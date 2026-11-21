use super::*;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum SearchOrder {
    Compact,
    Spatial,
}

#[derive(Clone, Copy)]
#[repr(u32)]
enum ScanPopulation {
    Original,
    Stocked,
}

#[cube]
pub(super) fn search(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    spatial: &[u32],
    proposals: &mut [u32],
    searches: &mut [u32],
    successes: &[Atomic<u32>],
    scratch: &[u32],
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
    #[comptime] suppliers: SupplierSearch,
    #[comptime] order: SearchOrder,
) {
    let job = ABSOLUTE_POS as u32;
    let count = match comptime!(order) {
        SearchOrder::Compact => control[SEARCHES_WORD].load(),
        SearchOrder::Spatial => config.actors,
    };
    if job < count && control[LATCH_WORD].load() == 0 {
        let id = match comptime!(order) {
            SearchOrder::Compact => searches[job as usize],
            SearchOrder::Spatial => {
                let cells = config.spatial_side * config.spatial_side;
                let base =
                    3 * (cells + 1 + config.actors) + ceildiv(config.actors, DISCOVERY_BLOCK) + 1;
                spatial[(base + job) as usize]
            }
        };
        let active = comptime!(order == SearchOrder::Compact)
            || searches[(config.actors + id) as usize] != 0
            || searches[(2 * config.actors + id) as usize] != 0;
        if active {
            let cells = config.spatial_side * config.spatial_side;
            let indexed = comptime!(matches!(parameters.market, MarketFlag::Consumer(_)))
                && spatial[cells as usize] <= empty::LIMIT;
            let wanted = ticks(
                searches[(config.actors + id) as usize],
                searches[(2 * config.actors + id) as usize],
            );
            let purchasing_power = ticks(
                searches[(3 * config.actors + id) as usize],
                searches[(4 * config.actors + id) as usize],
            );
            let mut buyer_access = 0u32;
            if comptime!(config.railway.routes_enabled == Feature::Enabled)
                && config.railway.nodes > 0
            {
                buyer_access = config.railway.access[id as usize * 2 + 1];
            }
            let px = actors[ActorLayout::X * (actors.len() / ActorLayout::WORDS) + id as usize];
            let py = actors[ActorLayout::Y * (actors.len() / ActorLayout::WORDS) + id as usize];
            let radius = config.search_radius;
            let whole_map = comptime!(matches!(suppliers, SupplierSearch::SingleCell { .. }));
            let max_days = max(ceildiv(radius, config.shipment_speed), 1);
            let maximum_delivery_cost = plus(
                times32(max_days, config.delivery_penalty),
                ticks(cartage::rate(config, max_days), 0),
            )
            .value;
            let scaled_speed = times32(config.shipment_speed, config.railway.speed_multiplier);
            let maximum_rail_days = ceildiv(u32::MAX, max(scaled_speed.extract(LOW_WORD), 1));
            let cx = if comptime!(whole_map) {
                u32::cast_from(0u32)
            } else {
                px / radius
            };
            let cy = if comptime!(whole_map) {
                u32::cast_from(0u32)
            } else {
                py / radius
            };
            let side = config.spatial_side;
            let mut chosen = u32::new(comptime!(i64::from(NO_ACTOR)));
            let mut trip = 0u32;
            let mut best = u32::new(comptime!(i64::from(u32::MAX)));
            let mut considered = 0u32;
            let mut seen_supply = false;
            let mut empty_seller = u32::new(comptime!(i64::from(NO_ACTOR)));
            let mut empty_trip = 0u32;
            let mut empty_score = u32::new(comptime!(i64::from(u32::MAX)));
            let mut stocked = spatial[cells as usize];
            let mut affordable = true;
            if indexed {
                stocked = empty::stocked_count(config, scratch, spatial[cells as usize]);
                affordable =
                    empty::affordable(config, scratch, spatial[cells as usize], purchasing_power);
            }
            let population = select(
                indexed
                    && stocked * 2 < spatial[cells as usize]
                    && empty::compacted(config, spatial[cells as usize]),
                comptime!(ScanPopulation::Stocked as u32),
                comptime!(ScanPopulation::Original as u32),
            );
            let mut y = select(stocked > 0 && affordable, max(cy, 1u32) - 1, side);
            while y <= min(cy + 1, side - 1) && considered < DISCOVERY_CANDIDATES {
                let mut x = max(cx, 1u32) - 1;
                while x <= min(cx + 1, side - 1) && considered < DISCOVERY_CANDIDATES {
                    let cell = y * side + x;
                    let begin = spatial[cell as usize];
                    let end = spatial[(cell + 1) as usize];
                    let original_length = end - begin;
                    let mut length = original_length;
                    let mut cursor =
                        (parameters.settlement_rotation + id) % max(original_length, 1);
                    if population == comptime!(ScanPopulation::Stocked as u32) {
                        length = stocked;
                        cursor = empty::stock_cursor(config, &*searches, stocked, cursor);
                    }
                    let limit = begin + supplier_samples(parameters, length);
                    let mut stride = 1u32;
                    if comptime!(parameters.market == MarketFlag::Household)
                        && length > HOUSEHOLD_SEARCH_SAMPLES
                    {
                        let offset = (id + parameters.settlement_rotation) % (length / 2);
                        stride = searches[(index::HOUSEHOLD_STRIDE_FIELD * config.actors
                            + begin
                            + offset) as usize];
                    }
                    let mut j = begin;
                    let nearest = if comptime!(whole_map) {
                        u32::cast_from(0u32)
                    } else {
                        max(px, x * radius) - min(px, (x + 1) * radius - 1) + max(py, y * radius)
                            - min(py, (y + 1) * radius - 1)
                    };
                    if nearest > radius {
                        j = end;
                    }
                    while j < limit && considered < DISCOVERY_CANDIDATES {
                        let previous = j;
                        let rotated = begin + cursor;
                        let block = rotated / index::BLOCK;
                        let mut skipped = 0u32;
                        if stride == 1 && population == comptime!(ScanPopulation::Original as u32) {
                            let unavailable = if indexed {
                                index::skip_stocked(
                                    scratch,
                                    config,
                                    block,
                                    px,
                                    py,
                                    purchasing_power,
                                )
                            } else {
                                index::skip(
                                    scratch,
                                    config,
                                    block,
                                    px,
                                    py,
                                    buyer_access,
                                    purchasing_power,
                                    chosen,
                                    ticks(
                                        empty_score,
                                        select(empty_seller == NO_ACTOR, u32::MAX, 0),
                                    ),
                                    u32::cast_from(
                                        seen_supply
                                            || comptime!(
                                                parameters.market != MarketFlag::Household
                                            ),
                                    ),
                                )
                            };
                            skipped = select(
                                unavailable,
                                min(min((block + 1) * index::BLOCK, end) - rotated, limit - j),
                                0,
                            );
                        }
                        if skipped == 0 {
                            let width = select(j == begin, 8u32, 32u32);
                            let mut offset = 0u32;
                            while offset < min(width, limit - j)
                                && considered < DISCOVERY_CANDIDATES
                            {
                                let position = cursor + offset;
                                let member = match comptime!(parameters.market) {
                                    MarketFlag::Household => {
                                        begin + (cursor + offset * stride) % max(length, 1)
                                    }
                                    _ => {
                                        let slot =
                                            select(position < length, position, position - length);
                                        let mut member = begin + slot;
                                        if population == comptime!(ScanPopulation::Stocked as u32) {
                                            member = searches[(17 * config.actors + slot) as usize];
                                        }
                                        member
                                    }
                                };
                                let firm = spatial[(side * side + 1 + member) as usize];
                                let supply = index::Supply::load(scratch, member);
                                let distance =
                                    if comptime!(parameters.market == MarketFlag::Household) {
                                        let seller_x = searches[(index::HOUSEHOLD_X_FIELD
                                            * config.actors
                                            + member)
                                            as usize];
                                        let seller_y = searches[(index::HOUSEHOLD_Y_FIELD
                                            * config.actors
                                            + member)
                                            as usize];
                                        max(px, seller_x) - min(px, seller_x) + max(py, seller_y)
                                            - min(py, seller_y)
                                    } else {
                                        distance(actors, id, firm)
                                    };
                                if comptime!(parameters.market == MarketFlag::Household) {
                                    seen_supply =
                                        seen_supply || (distance <= radius && supply.offered);
                                }
                                let mut viable = firm != id
                                    && distance <= radius
                                    && supply.quote > 0
                                    && !lt(purchasing_power, ticks(supply.quote, 0))
                                    && !equal(wanted, zero());
                                let animal_days = ceildiv(distance, config.shipment_speed);
                                let road_price = plus(
                                    ticks(supply.quote, 0),
                                    ticks(cartage::rate(config, animal_days), 0),
                                )
                                .value;
                                if viable && lt(purchasing_power, road_price) {
                                    let days = railway::delivery_days(
                                        config,
                                        railway::DeliveryRequest {
                                            buyer: id,
                                            seller: firm,
                                            trip: distance,
                                            buyer_access,
                                        },
                                    );
                                    viable = !lt(
                                        purchasing_power,
                                        ticks(
                                            cartage::delivered_price(
                                                config,
                                                supply.quote,
                                                distance,
                                                days,
                                            ),
                                            0,
                                        ),
                                    );
                                }
                                let mut relevant = viable && supply.offered;
                                if viable
                                    && supply.producer
                                    && !supply.offered
                                    && chosen == NO_ACTOR
                                    && !indexed
                                {
                                    let days =
                                        minimum_delivery_days(config, buyer_access, firm, distance);
                                    let lower_score = supply.quote + days * config.delivery_penalty;
                                    relevant =
                                        empty_seller == NO_ACTOR || lower_score < empty_score;
                                }
                                let counted = u32::cast_from(relevant && supply.offered);
                                if relevant && supply.offered && chosen != NO_ACTOR {
                                    let maximum_score =
                                        plus(ticks(supply.quote, 0), maximum_delivery_cost).value;
                                    if maximum_score.extract(HIGH_WORD) == 0 {
                                        let mut days = minimum_delivery_days(
                                            config,
                                            buyer_access,
                                            firm,
                                            distance,
                                        );
                                        if comptime!(
                                            config.railway.routes_enabled == Feature::Enabled
                                        ) && config.railway.nodes > 0
                                        {
                                            let access_days = ceildiv(
                                                buyer_access
                                                    + config.railway.access[firm as usize * 2 + 1],
                                                config.shipment_speed,
                                            );
                                            let bounded = scaled_speed.extract(HIGH_WORD) == 0
                                                && scaled_speed.extract(LOW_WORD) > 0
                                                && access_days <= u32::MAX - maximum_rail_days;
                                            days = select(bounded, days, 0);
                                        }
                                        let floor = supply.quote + days * config.delivery_penalty;
                                        relevant = floor < best;
                                    }
                                }
                                if relevant {
                                    let days = railway::delivery_days(
                                        config,
                                        railway::DeliveryRequest {
                                            buyer: id,
                                            seller: firm,
                                            trip: distance,
                                            buyer_access,
                                        },
                                    );
                                    let delivered = cartage::delivered_price(
                                        config,
                                        supply.quote,
                                        distance,
                                        days,
                                    );
                                    let score = delivered + days * config.delivery_penalty;
                                    if !supply.offered
                                        && (empty_seller == NO_ACTOR || score < empty_score)
                                    {
                                        empty_seller = firm;
                                        empty_trip = distance;
                                        empty_score = score;
                                    }
                                    if supply.offered && (chosen == NO_ACTOR || score < best) {
                                        chosen = firm;
                                        trip = distance;
                                        best = score;
                                    }
                                }
                                considered += counted;
                                offset += 1;
                            }
                            j += width;
                        }
                        j += skipped;
                        let next = cursor + (j - previous) * stride;
                        if stride == 1 {
                            cursor = select(next < length, next, next - length);
                        } else {
                            cursor = next % max(length, 1);
                        }
                    }
                    x += 1;
                }
                y += 1;
            }
            if indexed && chosen == NO_ACTOR && affordable {
                let mut offer = empty::Offer {
                    actor: u32::new(comptime!(i64::from(NO_ACTOR))),
                    trip: 0u32,
                };
                let cacheable = cells == 1
                    && empty::cache_fits(config, &*searches, spatial[cells as usize])
                    && (stocked == 0
                        || parameters.traded_good() != comptime!(Good::Bread as u32)
                            && empty::compacted(config, spatial[cells as usize]));
                if cacheable {
                    let cached = empty::cached_query(
                        actors,
                        config,
                        spatial,
                        scratch,
                        searches,
                        parameters,
                        id,
                        px,
                        py,
                        buyer_access,
                        purchasing_power,
                    );
                    offer.actor = cached.actor;
                    offer.trip = cached.trip;
                } else {
                    let queried = empty::query(
                        actors,
                        config,
                        spatial,
                        scratch,
                        parameters,
                        id,
                        px,
                        py,
                        buyer_access,
                        purchasing_power,
                    );
                    offer.actor = queried.actor;
                    offer.trip = queried.trip;
                }
                empty_seller = offer.actor;
                empty_trip = offer.trip;
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
                let days = railway::delivery_days(
                    config,
                    railway::DeliveryRequest {
                        buyer: id,
                        seller: chosen,
                        trip,
                        buyer_access,
                    },
                );
                let delivered = cartage::delivered_price(config, supply.quote, trip, days);
                let affordable = minimum(wanted, divide(purchasing_power, max(delivered, 1)));
                quantity = select(
                    missing_supply,
                    affordable,
                    minimum(affordable, supply.available),
                );
            }
            if comptime!(parameters.market == MarketFlag::Household) {
                let mut failure = u32::cast_from(missing_supply);
                if !equal(wanted, zero()) && chosen == NO_ACTOR {
                    failure = select(seen_supply || equal(purchasing_power, zero()), 2u32, 1u32);
                }
                set32(
                    state,
                    config,
                    comptime!(Field::FoodAccess as u32),
                    id,
                    failure,
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
                record_wide_order(control, parameters, quantity);
            }
        }
    }
}

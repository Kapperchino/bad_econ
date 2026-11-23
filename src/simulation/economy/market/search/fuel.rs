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
impl SearchInput {
    pub(super) fn match_fuel(&mut self) {
        let order = comptime!(match self.program.stage {
            SearchStage::SearchSpatialFuel => SearchOrder::Spatial,
            _ => SearchOrder::Compact,
        });
        let mut view = EconomyView::new(&mut self.state, &self.actors, &self.config);
        let economy = &mut view;
        let parameters = &self.parameters;
        let spatial: &[u32] = &self.spatial;
        let proposals: &mut [u32] = &mut self.proposals;
        let searches: &mut [u32] = &mut self.searches;
        let successes: &[Atomic<u32>] = &self.successes;
        let scratch: &[u32] = &self.scratch;
        let control: &[Atomic<u32>] = &self.control;
        let suppliers = comptime!(self.program.suppliers);
        let collection = comptime!(self.program.collection);
        let config = economy.ledger.config;
        let job = ABSOLUTE_POS as u32;
        let jobs = control[SEARCHES_WORD].load();
        let spatial_order = comptime!(order == SearchOrder::Spatial) && jobs > config.actors / 8;
        let count = select(spatial_order, config.actors, jobs);
        if job < count && control[LATCH_WORD].load() == 0 {
            let id = if spatial_order {
                let cells = config.spatial_side * config.spatial_side;
                let base =
                    3 * (cells + 1 + config.actors) + ceildiv(config.actors, DISCOVERY_BLOCK) + 1;
                spatial[(base + job) as usize]
            } else {
                searches[job as usize]
            };
            let active = !spatial_order
                || searches[(config.actors + id) as usize] != 0
                || searches[(2 * config.actors + id) as usize] != 0;
            if active {
                let cells = config.spatial_side * config.spatial_side;
                let indexed = comptime!(matches!(parameters.market, MarketFlag::Consumer(_)))
                    && spatial[cells as usize] <= empty::LIMIT;
                let demand = BuyerDemand::load(&*searches, config.actors, id);
                let wanted = demand.wanted;
                let purchasing_power = demand.purchasing_power;
                let inbound = config.railway.serves_buyer(id, config.ring);
                let location = BuyerLocation::new(&economy.actors, config, id);
                let buyer_access = location.access;
                let px = location.x;
                let py = location.y;
                let radius = config.search_radius;
                let columns = SearchAxis::for_suppliers(px, config, suppliers);
                let rows = SearchAxis::for_suppliers(py, config, suppliers);
                let side = config.spatial_side;
                let mut chosen = u32::new(comptime!(i64::from(NO_ACTOR)));
                let mut trip = 0u32;
                let mut price = 0u32;
                let mut available = zero();
                let mut best = u32::new(comptime!(i64::from(u32::MAX)));
                let mut considered = 0u32;
                let mut seen_supply = false;
                let mut empty_seller = u32::new(comptime!(i64::from(NO_ACTOR)));
                let mut empty_trip = 0u32;
                let mut empty_price = 0u32;
                let mut empty_score = u32::new(comptime!(i64::from(u32::MAX)));
                let empty_suppliers =
                    empty::EmptySuppliers::new(config, spatial, scratch, spatial[cells as usize]);
                let mut stocked = spatial[cells as usize];
                let mut affordable = true;
                if indexed {
                    stocked = empty_suppliers.stocked_count();
                    affordable = empty_suppliers.affordable(purchasing_power);
                }
                let population = select(
                    indexed && stocked * 2 < spatial[cells as usize] && empty_suppliers.compacted(),
                    comptime!(ScanPopulation::Stocked as u32),
                    comptime!(ScanPopulation::Original as u32),
                );
                let mut y = select(stocked > 0 && affordable, rows.first, side);
                while y <= rows.last && considered < DISCOVERY_CANDIDATES {
                    let mut x = columns.first;
                    while x <= columns.last && considered < DISCOVERY_CANDIDATES {
                        let cell = y * side + x;
                        let begin = spatial[cell as usize];
                        let end = spatial[(cell + 1) as usize];
                        let original_length = end - begin;
                        let mut length = original_length;
                        let mut cursor =
                            (parameters.settlement_rotation + id) % max(original_length, 1);
                        if population == comptime!(ScanPopulation::Stocked as u32) {
                            length = stocked;
                            cursor = empty_suppliers.stock_cursor(&*searches, stocked, cursor);
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
                        if columns.distance_to(x) + rows.distance_to(y) > radius {
                            j = end;
                        }
                        while j < limit && considered < DISCOVERY_CANDIDATES {
                            let previous = j;
                            let rotated = begin + cursor;
                            let block = rotated / index::BLOCK;
                            let mut skipped = 0u32;
                            if stride == 1
                                && population == comptime!(ScanPopulation::Original as u32)
                            {
                                let unavailable = if indexed {
                                    index::SupplierBlock::new(scratch, config, block)
                                        .skip_stocked(&location, purchasing_power)
                                } else {
                                    index::SupplierBlock::new(scratch, config, block).skip(
                                        &location,
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
                                            let slot = select(
                                                position < length,
                                                position,
                                                position - length,
                                            );
                                            let mut member = begin + slot;
                                            if population
                                                == comptime!(ScanPopulation::Stocked as u32)
                                            {
                                                member =
                                                    searches[(17 * config.actors + slot) as usize];
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
                                            max(px, seller_x) - min(px, seller_x)
                                                + max(py, seller_y)
                                                - min(py, seller_y)
                                        } else {
                                            economy.actors.distance(id, firm)
                                        };
                                    if comptime!(parameters.market == MarketFlag::Household) {
                                        seen_supply =
                                            seen_supply || (distance <= radius && supply.offered);
                                    }
                                    let empty = !indexed && supply.producer && chosen == NO_ACTOR;
                                    if (supply.offered || empty)
                                        && firm != id
                                        && distance <= radius
                                        && supply.quote > 0
                                        && !lt(purchasing_power, ticks(supply.quote, 0))
                                        && !equal(wanted, zero())
                                    {
                                        let mut days =
                                            max(ceildiv(distance, config.shipment_speed), 1);
                                        if inbound {
                                            days = railway::DeliveryRequest {
                                                buyer: id,
                                                seller: firm,
                                                trip: distance,
                                                buyer_access,
                                            }
                                            .days(config);
                                        }
                                        let delivered = cartage::delivered_price(
                                            config,
                                            supply.quote,
                                            distance,
                                            days,
                                        );
                                        let affordable = !lt(purchasing_power, ticks(delivered, 0));
                                        let score = delivered + days * config.delivery_penalty;
                                        if affordable && supply.offered {
                                            considered += 1;
                                            if chosen == NO_ACTOR || score < best {
                                                chosen = firm;
                                                trip = distance;
                                                best = score;
                                                price = delivered;
                                                available = supply.available;
                                            }
                                        }
                                        if affordable
                                            && !supply.offered
                                            && (empty_seller == NO_ACTOR || score < empty_score)
                                        {
                                            empty_seller = firm;
                                            empty_trip = distance;
                                            empty_score = score;
                                            empty_price = delivered;
                                        }
                                    }
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
                        && empty_suppliers.cache_fits(&*searches)
                        && (stocked == 0
                            || parameters.traded_good() != comptime!(Good::Bread as u32)
                                && empty_suppliers.compacted());
                    let mut region = 0u32;
                    let mut local = false;
                    if comptime!(collection == OrderCollection::ConsumerMask)
                        && cells == 1
                        && config.actors >= 262_144
                        && stocked <= 8
                    {
                        region = road::candidates(config, px, py);
                        local = road::available(config, &*searches, region, id);
                    }
                    let buyer = empty::Buyer::new(
                        config,
                        id,
                        location,
                        purchasing_power,
                        parameters.settlement_rotation,
                    );
                    if local {
                        let selected = road::query(
                            &economy.actors,
                            config,
                            spatial,
                            scratch,
                            &*searches,
                            region,
                            id,
                            parameters.settlement_rotation,
                            purchasing_power,
                            buyer_access,
                        );
                        offer.actor = selected.actor;
                        offer.trip = selected.trip;
                    } else if cacheable {
                        let cached = empty_suppliers.cached_query(
                            &economy.actors,
                            searches,
                            &buyer,
                            parameters.traded_good(),
                        );
                        offer.actor = cached.actor;
                        offer.trip = cached.trip;
                    } else {
                        let queried = empty_suppliers.query(&economy.actors, &buyer);
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
                    if missing_supply {
                        price = empty_price;
                        if indexed {
                            let terms = household_production::TradeFields::new(
                                &economy.actors,
                                chosen,
                                parameters.traded_good(),
                            );
                            let quote = economy.ledger.number_at(terms.price, chosen);
                            let mut days = max(ceildiv(trip, config.shipment_speed), 1);
                            if inbound {
                                days = railway::DeliveryRequest {
                                    buyer: id,
                                    seller: chosen,
                                    trip,
                                    buyer_access,
                                }
                                .days(config);
                            }
                            price = cartage::delivered_price(config, quote, trip, days);
                        }
                    }
                    let affordable = minimum(wanted, divide(purchasing_power, max(price, 1)));
                    quantity = select(missing_supply, affordable, minimum(affordable, available));
                    if comptime!(collection == OrderCollection::ConsumerMask) && !missing_supply {
                        filter::sample_order(
                            &economy.ledger,
                            successes,
                            parameters,
                            id,
                            chosen,
                            quantity,
                        );
                    }
                }
                if comptime!(parameters.market == MarketFlag::Household) {
                    let result = FoodSearchResult {
                        seller: chosen,
                        seen_supply,
                        used_fallback: missing_supply,
                    };
                    result.record(&mut economy.ledger, id, wanted, purchasing_power);
                }
                MarketProposal {
                    peer: chosen,
                    quantity,
                    trip,
                }
                .store(proposals, id);
                if chosen != NO_ACTOR {
                    if comptime!(collection != OrderCollection::ConsumerMask) {
                        record_success(successes, parameters, id);
                    }
                    record_wide_order(control, parameters, quantity);
                }
            }
        }
    }
}

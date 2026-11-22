use super::{ranking::MatchSelection, *};

#[cube]
impl SearchInput {
    pub(super) fn match_suppliers(&mut self) {
        let source = comptime!(match self.program.stage {
            SearchStage::SearchIndexed => index::SupplySource::Index,
            SearchStage::SearchGrain => index::SupplySource::Grain,
            _ => index::SupplySource::State,
        });
        let mut view = EconomyView::new(&mut self.state, &self.actors, &self.config);
        let economy = &mut view;
        let parameters = &self.parameters;
        let spatial: &mut [u32] = &mut self.spatial;
        let proposals: &mut [u32] = &mut self.proposals;
        let searches: &mut [u32] = &mut self.searches;
        let successes: &[Atomic<u32>] = &self.successes;
        let scratch: &[u32] = &self.scratch;
        let control: &[Atomic<u32>] = &self.control;
        let suppliers = comptime!(self.program.suppliers);
        let collection = comptime!(self.program.collection);
        let config = economy.ledger.config;
        let jobs = orders::job_count(control, SEARCHES_WORD);
        let mut job = CUBE_POS as u32;
        while job < jobs {
            let id = searches[job as usize];
            let demand = BuyerDemand::load(&*searches, config.actors, id);
            let wanted = demand.wanted;
            let purchasing_power = demand.purchasing_power;
            let mut input_target = zero();
            if comptime!(matches!(parameters.market, MarketFlag::Input(_))) {
                let capacity = economy.actors.capacity(id);
                input_target = minimum(
                    wanted,
                    times32(capacity, economy.ledger.number(Field::InputTicks, id)),
                );
            }
            let location = BuyerLocation::new(&economy.actors, config, id);
            let buyer_access = location.access;
            let px = location.x;
            let py = location.y;
            let radius = config.search_radius;
            let whole_map = comptime!(matches!(suppliers, SupplierSearch::SingleCell { .. }));
            let maximum_delivery_cost = times32(
                max(ceildiv(radius, config.shipment_speed), 1),
                config.delivery_penalty,
            );
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
            let buyer = MatchingBuyer {
                id,
                demand,
                input_target,
                access: buyer_access,
                maximum_delivery_cost,
            };
            let mut selection = MatchSelection::new(comptime!(parameters.market));
            let lanes = comptime!(match source {
                index::SupplySource::Index => crate::simulation::ops::MAX_WORKGROUP_SIZE as usize,
                _ => 32usize,
            });
            let mut candidates = CandidateTile::new(lanes, comptime!(parameters.market));
            let mut y = max(cy, 1u32) - 1;
            if comptime!(source == index::SupplySource::Grain)
                && grain::tiled_dealer(&*economy, &*spatial, control, jobs, id, purchasing_power)
            {
                let trade = grain::dealer_trade(
                    &*economy,
                    &*spatial,
                    &*searches,
                    control,
                    parameters,
                    job,
                    id,
                    wanted,
                    purchasing_power,
                    buyer_access,
                );
                selection.proposal.peer = trade.seller;
                selection.proposal.quantity = trade.quantity;
                selection.proposal.trip = trade.trip;
                y = side;
            } else if comptime!(source == index::SupplySource::Grain)
                && (!household_production::grain_dealer(&*economy, id)
                    || lt(
                        purchasing_power,
                        ticks(control[INPUT_MINIMUM_QUOTE_WORD].load(), 0),
                    ))
            {
                let trade = grain::search(
                    &*economy,
                    &*spatial,
                    &*searches,
                    control,
                    parameters,
                    id,
                    wanted,
                    purchasing_power,
                    buyer_access,
                );
                selection.proposal.peer = trade.seller;
                selection.proposal.quantity = trade.quantity;
                selection.proposal.trip = trade.trip;
                y = side;
            }
            if comptime!(
                source == index::SupplySource::Index
                    && matches!(parameters.market, MarketFlag::Input(_))
            ) && parameters.traded_good() == Good::Bread as u32
                && lt(
                    purchasing_power,
                    ticks(control[INPUT_MINIMUM_QUOTE_WORD].load(), 0),
                )
            {
                y = side;
            }
            while y <= min(cy + 1, side - 1) && selection.considered < DISCOVERY_CANDIDATES {
                let mut x = max(cx, 1u32) - 1;
                while x <= min(cx + 1, side - 1) && selection.considered < DISCOVERY_CANDIDATES {
                    let cell = y * side + x;
                    let begin = spatial[cell as usize];
                    let end = spatial[(cell + 1) as usize];
                    let length = end - begin;
                    let limit = begin + supplier_samples(parameters, length);
                    let stride = supplier_stride(&economy.actors, parameters, id, length);
                    let mut cursor = (parameters.settlement_rotation + id) % max(length, 1);
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
                    while j < limit && selection.considered < DISCOVERY_CANDIDATES {
                        let previous = j;
                        let mut skipped = 0u32;
                        if comptime!(
                            matches!(
                                parameters.market,
                                MarketFlag::Consumer(_) | MarketFlag::Household
                            ) || source == index::SupplySource::Index
                        ) && stride == 1
                        {
                            let rotated = begin + cursor;
                            let block = rotated / index::BLOCK;
                            let unavailable = index::SupplierBlock::new(scratch, config, block)
                                .skip(
                                    &location,
                                    purchasing_power,
                                    selection.proposal.peer,
                                    selection.fallback_score,
                                    u32::cast_from(
                                        selection.seen_supply
                                            || comptime!(
                                                parameters.market != MarketFlag::Household
                                            ),
                                    ),
                                );
                            skipped = select(
                                unavailable,
                                min(min((block + 1) * index::BLOCK, end) - rotated, end - j),
                                0,
                            );
                        }
                        if comptime!(
                            source == index::SupplySource::Index
                                && matches!(parameters.market, MarketFlag::Input(_))
                        ) && stride == 1
                            && selection.proposal.peer != NO_ACTOR
                            && input_target.extract(HIGH_WORD) == 0
                        {
                            let rotated = begin + cursor;
                            let block = rotated / index::BLOCK;
                            let supplier_block = index::SupplierBlock::new(scratch, config, block);
                            let upper = ticks(supplier_block.maximum_available(), 0);
                            let mut inferior = lt(upper, selection.proposal.quantity);
                            if equal(upper, selection.proposal.quantity) {
                                let price_floor = supplier_block
                                    .partial_input_price_floor(&location, maximum_delivery_cost);
                                inferior = price_floor >= selection.score.extract(LOW_WORD);
                            }
                            inferior = inferior && lt(upper, input_target);
                            skipped = max(
                                skipped,
                                select(
                                    inferior,
                                    min(min((block + 1) * index::BLOCK, end) - rotated, limit - j),
                                    0,
                                ),
                            );
                        }
                        if parameters.traded_good() == Good::Bread as u32 && stride == 1 {
                            let rotated = begin + cursor;
                            let block = rotated / DISCOVERY_BLOCK;
                            let quote =
                                spatial[(3 * (side * side + 1 + config.actors) + block) as usize];
                            let unaffordable = select(
                                lt(purchasing_power, ticks(quote, 0)),
                                min(min((block + 1) * DISCOVERY_BLOCK, end) - rotated, end - j),
                                0,
                            );
                            let visibility = select(unaffordable > skipped, unaffordable, 0);
                            skipped = max(skipped, unaffordable);
                            let mut offset = 0u32;
                            while offset < visibility && !selection.seen_supply {
                                let mut offered = 0u32;
                                if offset + UNIT_POS < visibility {
                                    let seller = spatial
                                        [(side * side + 1 + rotated + offset + UNIT_POS) as usize];
                                    let supply = index::Supply::for_search(
                                        &*economy,
                                        scratch,
                                        parameters,
                                        rotated + offset + UNIT_POS,
                                        seller,
                                        source,
                                    );
                                    offered = u32::cast_from(
                                        economy.actors.distance(id, seller) <= radius
                                            && supply.offered,
                                    );
                                }
                                selection.seen_supply =
                                    parallel::reduce::<u32>(offered, comptime!(Reduction::Sum)) > 0;
                                offset += CUBE_DIM;
                            }
                        }
                        if skipped == 0 {
                            let initial_width = select(
                                comptime!(matches!(parameters.market, MarketFlag::Input(_)))
                                    && parameters.traded_good() == Good::Grain as u32,
                                32u32,
                                8u32,
                            );
                            let width = select(j == begin, initial_width, CUBE_DIM);
                            let lane = UNIT_POS as usize;
                            candidates.clear_faults(lane, comptime!(parameters.market));
                            let window = CandidateWindow {
                                width,
                                start: j,
                                limit,
                            };
                            let mut offer = Candidate::empty();
                            let mut firm = u32::new(comptime!(i64::from(NO_ACTOR)));
                            if window.includes_lane() {
                                let rotated = begin + (cursor + UNIT_POS * stride) % max(length, 1);
                                firm = spatial[(side * side + 1 + rotated) as usize];
                                let supply = index::Supply::for_search(
                                    &*economy, scratch, parameters, rotated, firm, source,
                                );
                                offer.replace(candidates.evaluate(
                                    &*economy, parameters, &buyer, &selection, firm, &supply,
                                    source,
                                ));
                            }
                            candidates.store(lane, &offer, firm);
                            sync_cube();
                            let mut cooperative = false;
                            if comptime!(source == index::SupplySource::Index) {
                                cooperative = CUBE_DIM >= 32;
                                selection.rank_indexed(&candidates, &offer, parameters, &window);
                            }
                            if comptime!(
                                source != index::SupplySource::Index
                                    && parameters.scan
                                        == crate::simulation::parallel::ScanMode::Plane
                            ) {
                                cooperative = CUBE_DIM <= PLANE_DIM;
                                if cooperative {
                                    selection.rank_plane(
                                        &candidates,
                                        &offer,
                                        parameters,
                                        control,
                                        &window,
                                    );
                                }
                            }
                            if !cooperative {
                                selection.rank_serial(
                                    &mut candidates,
                                    parameters,
                                    control,
                                    &window,
                                );
                            }
                            sync_cube();
                            j += width;
                        }
                        j += skipped;
                        cursor = (cursor + (j - previous) * stride) % max(length, 1);
                    }
                    x += 1;
                }
                y += 1;
            }
            if UNIT_POS == 0 {
                let mut missing_supply = false;
                if selection.proposal.peer == NO_ACTOR && selection.fallback.peer != NO_ACTOR {
                    selection.proposal.peer = selection.fallback.peer;
                    selection.proposal.quantity = selection.fallback.quantity;
                    selection.proposal.trip = selection.fallback.trip;
                    missing_supply = true;
                }
                if comptime!(matches!(parameters.market, MarketFlag::Consumer(_)))
                    && selection.proposal.peer != NO_ACTOR
                {
                    let supply = index::Supply::new(
                        &*economy,
                        selection.proposal.peer,
                        parameters.traded_good(),
                    );
                    let days = railway::DeliveryRequest {
                        buyer: id,
                        seller: selection.proposal.peer,
                        trip: selection.proposal.trip,
                        buyer_access,
                    }
                    .days(config);
                    let delivered = cartage::delivered_price(
                        config,
                        supply.quote,
                        selection.proposal.trip,
                        days,
                    );
                    let affordable = minimum(wanted, divide(purchasing_power, max(delivered, 1)));
                    selection.proposal.quantity = select(
                        missing_supply,
                        affordable,
                        minimum(affordable, supply.available),
                    );
                }
                if comptime!(parameters.market == MarketFlag::Household) {
                    let mut failure = u32::cast_from(missing_supply);
                    if !equal(wanted, zero()) && selection.proposal.peer == NO_ACTOR {
                        failure = select(
                            selection.seen_supply || equal(purchasing_power, zero()),
                            2u32,
                            1u32,
                        );
                    }
                    economy.ledger.set_number(Field::FoodAccess, id, failure);
                }
                selection.proposal.store(proposals, id);
                if selection.proposal.peer != NO_ACTOR {
                    let delivery = railway::DeliveryRequest {
                        buyer: id,
                        seller: selection.proposal.peer,
                        trip: selection.proposal.trip,
                        buyer_access,
                    };
                    if comptime!(
                        matches!(parameters.market, MarketFlag::Input(_))
                            && config.railway.routes_enabled == Feature::Enabled
                    ) && control[INPUT_RAIL_WORD].load() == 0
                        && delivery.may_book(config)
                    {
                        control[INPUT_RAIL_WORD].store(1);
                    }
                    if comptime!(collection == OrderCollection::InputTile) {
                        let slot = control[CONSTRUCTED_EVENTS_WORD].fetch_add(1);
                        if slot < comptime!(sorting::INPUT_TILE as u32) {
                            searches
                                [(comptime!(INPUT_BUYERS_FIELD) * config.actors + slot) as usize] =
                                id;
                        }
                    }
                    record_success(successes, parameters, id);
                    record_wide_order(control, parameters, selection.proposal.quantity);
                }
            }
            sync_cube();
            job += CUBE_COUNT as u32;
        }
    }
}

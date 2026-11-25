use super::demand::FoodSearchResult;
use crate::simulation::economy::*;

#[derive(CubeType)]
pub(in crate::simulation::economy) struct MarketProposal {
    pub peer: u32,
    pub quantity: Ticks,
    pub trip: u32,
}

#[cube]
impl MarketProposal {
    pub fn payment(
        economy: &EconomyView<'_>,
        control: &[Atomic<u32>],
        id: u32,
        #[comptime] market: MarketFlag,
    ) -> Self {
        let config = economy.ledger.config;
        let mut quantity = zero();
        let recipient = match comptime!(market) {
            MarketFlag::Payroll => {
                quantity = economy.ledger.value(Field::Due, id);
                economy.ledger.number(Field::Employer, id) - 1
            }
            MarketFlag::Distribution => {
                let kind = economy.actors.kind(id);
                if !kind.is(comptime!(ActorKind::Household)) {
                    let cash_reserve = select(
                        kind.bits() == comptime!(Good::Housing as u32),
                        zero(),
                        ticks(config.reserve_low, config.reserve_high),
                    );
                    let reserve = maximum(
                        cash_reserve,
                        economy.ledger.value(Field::OperatingReserve, id),
                    );
                    let surplus = positive(economy.ledger.value(Field::Cash, id), reserve);
                    quantity =
                        ratio(surplus, config.distribution_rate, POLICY_SCALE).record(control);
                }
                economy.actors.owner(id)
            }
            _ => u32::new(comptime!(i64::from(NO_ACTOR))),
        };
        MarketProposal {
            peer: select(equal(quantity, zero()), NO_ACTOR, recipient),
            quantity,
            trip: 0,
        }
    }

    pub fn store(&self, proposals: &mut [u32], actor: u32) {
        let base = actor as usize * ProposalLayout::WORDS;
        proposals[base + ProposalLayout::ACTOR] = self.peer;
        proposals[base + ProposalLayout::QUANTITY_LOW] = self.quantity.extract(LOW_WORD);
        proposals[base + ProposalLayout::QUANTITY_HIGH] = self.quantity.extract(HIGH_WORD);
        proposals[base + ProposalLayout::DISTANCE] = self.trip;
    }
}

#[allow(
    clippy::collapsible_match,
    reason = "CubeCL uses explicit runtime branches"
)]
#[cube]
pub(in crate::simulation::economy) fn propose(
    economy: &mut EconomyView<'_>,
    spatial: &[u32],
    proposals: &mut [u32],
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
    invocation: u32,
    #[comptime] labor_round: employment::LaborRound,
) -> u32 {
    let config = economy.ledger.config;
    let mut chosen = u32::new(comptime!(i64::from(NO_ACTOR)));
    if invocation < config.actors {
        let market = comptime!(parameters.market);
        let mut id = invocation;
        if comptime!(market == MarketFlag::Household && parameters.discovery.indexed()) {
            let base = config.query_members();
            id = spatial[(base + invocation) as usize];
        }
        let mut quantity = ticks(0u32, 0u32);
        let mut trip = 0u32;
        let mut search = false;
        let good = parameters.traded_good();
        let mut wanted = ticks(0u32, 0u32);
        let mut seen_supply = false;
        let mut empty_seller = u32::new(comptime!(i64::from(NO_ACTOR)));
        let mut empty_quantity = zero();
        let mut empty_trip = 0u32;
        let mut empty_score = ticks(u32::MAX, u32::MAX);
        let mut missing_supply = false;
        let mut purchasing_power = economy.ledger.value(Field::Cash, id);
        match market {
            MarketFlag::Labor => {
                let labor = employment::LaborSearch::new(
                    &*economy,
                    control,
                    parameters,
                    id,
                    comptime!(labor_round),
                );
                chosen = labor.peer;
                quantity = labor.quantity;
                trip = labor.trip;
                search = labor.vacancies;
            }
            MarketFlag::Input(_) | MarketFlag::Consumer(_) | MarketFlag::Household => {
                let order = PurchaseRequest::new(
                    economy,
                    spatial,
                    control,
                    parameters,
                    id,
                    comptime!(pricing::QuoteSource::FromLedger),
                );
                search = order.search;
                wanted = order.wanted;
                purchasing_power = order.purchasing_power;
            }
            MarketFlag::Payroll | MarketFlag::Distribution => {
                let payment = MarketProposal::payment(&*economy, control, id, comptime!(market));
                chosen = payment.peer;
                quantity = payment.quantity;
                if comptime!(market == MarketFlag::Distribution) && chosen != NO_ACTOR {
                    let estate = economy.ledger.number(Field::EstateSettled, chosen);
                    if estate
                        == comptime!(crate::domain::enterprise::EstateStatus::Unclaimed as u32)
                    {
                        chosen = comptime!(NO_ACTOR);
                    }
                }
            }
        }

        if comptime!(labor_round != employment::LaborRound::Retained) && search {
            let columns = search::SearchAxis::around(economy.actors.x(id), config);
            let rows = search::SearchAxis::around(economy.actors.y(id), config);
            let side = config.spatial_side;
            let mut best = ticks(u32::MAX, u32::MAX);
            if comptime!(market == MarketFlag::Labor) {
                best = ticks(0u32, 0u32);
            }
            let mut considered = 0u32;
            let supplier_limit = parameters.supplier_limit();
            let mut y = rows.first;
            if comptime!(matches!(market, MarketFlag::Input(_))) {
                let local = search::local::staple(
                    &*economy,
                    spatial,
                    control,
                    parameters,
                    id,
                    wanted,
                    purchasing_power,
                );
                if (comptime!(parameters.market == MarketFlag::Input(InputMarket::Grain))
                    && pricing::distributor(&economy.ledger, id))
                    || local.peer != NO_ACTOR
                {
                    chosen = local.peer;
                    quantity = local.quantity;
                    trip = local.trip;
                    y = side;
                }
            }
            while y <= rows.last && considered < supplier_limit {
                let mut x = columns.first;
                while x <= columns.last && considered < supplier_limit {
                    let cell = y * side + x;
                    let begin = search::supplier_start(&*economy, spatial, parameters, id, cell);
                    let mut j = begin;
                    let length = spatial[(cell + 1) as usize] - begin;
                    let limit = begin + search::supplier_samples(config, parameters, length);
                    let stride = search::supplier_stride(&economy.actors, parameters, id, length);
                    let mut cursor = (parameters.settlement_rotation + id) % max(length, 1);
                    if columns.distance_to(x) + rows.distance_to(y) > config.search_radius {
                        j = spatial[(cell + 1) as usize];
                    }
                    while j < limit && considered < supplier_limit {
                        let previous = j;
                        let rotated = begin + cursor;
                        let mut skipped = 0u32;
                        if parameters.traded_good() == Good::Bread as u32
                            && comptime!(parameters.discovery.indexed())
                            && stride == 1
                        {
                            let block = rotated / DISCOVERY_BLOCK;
                            let base = 3 * (side * side + 1 + config.actors);
                            let quote = spatial[(base + block) as usize];
                            let cash = purchasing_power;
                            if lt(cash, ticks(quote, 0)) {
                                skipped = min(
                                    min((block + 1) * DISCOVERY_BLOCK, begin + length) - rotated,
                                    begin + length - j,
                                );
                            }
                            let mut offset = 0;
                            while offset < skipped && !seen_supply {
                                let seller = spatial[(side * side + 1 + rotated + offset) as usize];
                                seen_supply = economy.actors.distance(id, seller)
                                    <= config.search_radius
                                    && household_production::offers(&*economy, seller, good);
                                offset += 1;
                            }
                        }
                        if skipped == 0 {
                            let firm = spatial[(side * side + 1u32 + rotated) as usize];
                            let dist = economy.actors.distance(id, firm);
                            let mut accessible = dist <= config.search_radius;
                            if comptime!(matches!(market, MarketFlag::Input(_))) {
                                accessible = accessible
                                    && household_production::counterparty(
                                        &*economy,
                                        id,
                                        firm,
                                        parameters.traded_good(),
                                    );
                            }
                            if accessible {
                                let route = match comptime!(market) {
                                    MarketFlag::Labor => railway::FreightRoute::by_road(
                                        &economy.actors,
                                        config,
                                        id,
                                        firm,
                                    ),
                                    _ => railway::FreightRequest::new(
                                        &economy.actors,
                                        config,
                                        id,
                                        firm,
                                    )
                                    .preferred(config),
                                };
                                let terms = household_production::TradeFields::new(
                                    &economy.actors,
                                    economy.ledger.config,
                                    firm,
                                    good,
                                );
                                let quote = economy.ledger.number_at(terms.price, firm);
                                let quote = match comptime!(market) {
                                    MarketFlag::Labor
                                    | MarketFlag::Input(InputMarket::Construction) => quote,
                                    _ => cartage::delivered_price(config, quote, dist, route.days),
                                };
                                let mut inspect = true;
                                if comptime!(market != MarketFlag::Labor) {
                                    let cash = purchasing_power;
                                    inspect = !lt(cash, ticks(max(quote, 1), 0));
                                    if !inspect {
                                        seen_supply = seen_supply
                                            || household_production::offers(&*economy, firm, good);
                                    }
                                }
                                if inspect {
                                    let mut eligible = accessible;

                                    let days = select(
                                        comptime!(
                                            market == MarketFlag::Input(InputMarket::Construction)
                                        ),
                                        0,
                                        route.days,
                                    );
                                    let mut score =
                                        ticks(quote + days * config.delivery_penalty, 0u32);
                                    let amount = if comptime!(market == MarketFlag::Labor) {
                                        let offer =
                                            search::OfferRequest::labor(&*economy, id, firm)
                                                .evaluate(&*economy, control, parameters);
                                        eligible = offer.eligible;
                                        score = offer.score;
                                        offer.quantity
                                    } else {
                                        eligible = eligible
                                            && search::accepts_input_quote(
                                                &*economy, control, parameters, id, quote,
                                            );
                                        let affordable = minimum(
                                            wanted,
                                            divide(purchasing_power, max(quote, 1u32)),
                                        );
                                        let offered =
                                            household_production::offers(&*economy, firm, good);
                                        seen_supply = seen_supply || (eligible && offered);
                                        let known_producer = !economy
                                            .actors
                                            .kind(firm)
                                            .is(comptime!(ActorKind::Household))
                                            && (economy.actors.kind(firm).output() == good
                                                || pricing::distributor(&economy.ledger, firm));
                                        if eligible
                                            && firm != id
                                            && known_producer
                                            && !offered
                                            && quote > 0
                                            && !equal(affordable, zero())
                                            && lt(score, empty_score)
                                        {
                                            empty_seller = firm;
                                            empty_quantity = affordable;
                                            empty_trip = dist;
                                            empty_score = score;
                                        }
                                        let amount = minimum(
                                            minimum(
                                                wanted,
                                                positive(
                                                    economy.ledger.value_at(
                                                        household_production::trade_stock_field(
                                                            good,
                                                        ),
                                                        firm,
                                                    ),
                                                    household_production::reserve(
                                                        &*economy, firm, good,
                                                    ),
                                                ),
                                            ),
                                            divide(purchasing_power, max(quote, 1u32)),
                                        );
                                        eligible = eligible
                                            && firm != id
                                            && household_production::offers(&*economy, firm, good)
                                            && quote > 0u32
                                            && !equal(amount, ticks(0u32, 0u32));
                                        score = search::supply_score(
                                            &*economy, parameters, id, wanted, amount, score,
                                        );
                                        amount
                                    };
                                    let mut improves = lt(score, best);
                                    if comptime!(market == MarketFlag::Labor) {
                                        improves = lt(best, score);
                                    }
                                    improves = improves || chosen == NO_ACTOR;
                                    considered += u32::cast_from(
                                        eligible
                                            && (comptime!(!matches!(market, MarketFlag::Input(_)))
                                                || score.extract(HIGH_WORD) == 0),
                                    );
                                    if eligible && improves {
                                        chosen = firm;
                                        quantity = amount;
                                        trip = dist;
                                        best = score;
                                    }
                                }
                            }
                            j += 1;
                        }
                        j += skipped;
                        cursor = (cursor + (j - previous) * stride) % max(length, 1);
                    }
                    x += 1;
                }
                y += 1;
            }
        }
        if chosen == NO_ACTOR && empty_seller != NO_ACTOR {
            chosen = empty_seller;
            quantity = empty_quantity;
            trip = empty_trip;
            missing_supply = true;
        }
        if comptime!(market == MarketFlag::Household) {
            let cash = economy.ledger.value(Field::Cash, id);
            let result = FoodSearchResult {
                seller: chosen,
                seen_supply,
                used_fallback: missing_supply,
            };
            result.record(&mut economy.ledger, id, wanted, cash);
        }
        MarketProposal {
            peer: chosen,
            quantity,
            trip,
        }
        .store(proposals, id);
    }
    chosen
}

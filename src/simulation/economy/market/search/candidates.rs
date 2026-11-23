use super::{ranking::MatchSelection, *};

#[derive(CubeType, Clone, Copy)]
pub(in crate::simulation::economy) struct Candidate {
    pub eligible: bool,
    pub seen_supply: bool,
    pub empty: bool,
    pub quantity: Ticks,
    pub score: Ticks,
    pub trip: u32,
}

#[cube]
impl Candidate {
    pub(super) fn empty() -> Self {
        Candidate {
            eligible: false,
            seen_supply: false,
            empty: false,
            quantity: zero(),
            score: zero(),
            trip: 0,
        }
    }

    pub(super) fn replace(&mut self, other: Candidate) {
        self.eligible = other.eligible;
        self.seen_supply = other.seen_supply;
        self.empty = other.empty;
        self.quantity = other.quantity;
        self.score = other.score;
        self.trip = other.trip;
    }
}

#[derive(CubeType)]
pub(super) struct CandidateTile {
    pub eligible: Shared<[u32]>,
    pub seen_supply: Shared<[u32]>,
    pub empty: Shared<[u32]>,
    pub quantity: Shared<[Ticks]>,
    pub score: Shared<[Ticks]>,
    pub trip: Shared<[u32]>,
    pub firm: Shared<[u32]>,
    faults: Shared<[Atomic<u32>]>,
    visited: Shared<u32>,
    saw_supply: Shared<u32>,
}

#[cube]
impl CandidateTile {
    pub(super) fn new(#[comptime] lanes: usize, #[comptime] market: MarketFlag) -> Self {
        CandidateTile {
            eligible: Shared::<[u32]>::new_slice(lanes),
            seen_supply: Shared::<[u32]>::new_slice(lanes),
            empty: Shared::<[u32]>::new_slice(lanes),
            quantity: Shared::<[Ticks]>::new_slice(lanes),
            score: Shared::<[Ticks]>::new_slice(lanes),
            trip: Shared::<[u32]>::new_slice(lanes),
            firm: Shared::<[u32]>::new_slice(lanes),
            faults: Shared::<[Atomic<u32>]>::new_slice(comptime!(match market {
                MarketFlag::Labor => lanes * CONTROL_WORDS,
                _ => 1usize,
            })),
            visited: Shared::<u32>::new(),
            saw_supply: Shared::<u32>::new(),
        }
    }

    pub(super) fn store(&mut self, lane: usize, offer: &Candidate, firm: u32) {
        self.eligible[lane] = u32::cast_from(offer.eligible);
        self.seen_supply[lane] = u32::cast_from(offer.seen_supply);
        self.empty[lane] = u32::cast_from(offer.empty);
        self.quantity[lane] = offer.quantity;
        self.score[lane] = offer.score;
        self.trip[lane] = offer.trip;
        self.firm[lane] = firm;
    }

    pub(super) fn clear_faults(&self, lane: usize, #[comptime] market: MarketFlag) {
        if comptime!(market == MarketFlag::Labor) {
            self.faults[lane * CONTROL_WORDS + crate::simulation::storage::flags::FAULT_WORD]
                .store(0);
        }
    }

    pub(super) fn faults(&self, lane: usize) -> u32 {
        self.faults[lane * CONTROL_WORDS + crate::simulation::storage::flags::FAULT_WORD].load()
    }

    pub(super) fn publish_search(&mut self, considered: u32, saw_supply: u32) {
        *self.visited = considered;
        *self.saw_supply = saw_supply;
    }

    pub(super) fn considered(&self) -> u32 {
        *self.visited
    }

    pub(super) fn saw_supply(&self) -> bool {
        *self.saw_supply != 0
    }

    pub(super) fn evaluate(
        &self,
        economy: &EconomyView<'_>,
        parameters: &EconomyParameters,
        buyer: &MatchingBuyer,
        selection: &MatchSelection,
        firm: u32,
        supply: &index::Supply,
        #[comptime] source: index::SupplySource,
    ) -> Candidate {
        let config = economy.ledger.config;
        let fault_start = UNIT_POS as usize
            * comptime!(match parameters.market {
                MarketFlag::Labor => CONTROL_WORDS,
                _ => 0usize,
            });
        let fault_words = comptime!(match parameters.market {
            MarketFlag::Labor => CONTROL_WORDS,
            _ => 1usize,
        });
        let mut offer = Candidate::empty();
        let mut relevant = true;
        if comptime!(
            source == index::SupplySource::Index
                && matches!(parameters.market, MarketFlag::Input(_))
        ) && selection.proposal.peer != NO_ACTOR
            && buyer.input_target.extract(HIGH_WORD) == 0
        {
            relevant = !(lt(supply.available, buyer.input_target)
                && lt(supply.available, selection.proposal.quantity));
        }
        let mut trip = 0u32;
        if relevant {
            trip = economy.actors.distance(buyer.id, firm);
            if comptime!(matches!(
                parameters.market,
                MarketFlag::Household | MarketFlag::Consumer(_)
            )) {
                let quote = supply.quote;
                let days = minimum_delivery_days(config, buyer.access, firm, trip);
                let lower_score =
                    plus(ticks(quote, 0), times32(days, config.delivery_penalty)).value;
                relevant = supply.offered
                    || (selection.proposal.peer == NO_ACTOR
                        && quote > 0
                        && lt(lower_score, selection.fallback_score));
            }
            if comptime!(
                source == index::SupplySource::Index
                    && matches!(parameters.market, MarketFlag::Input(_))
            ) && selection.proposal.peer != NO_ACTOR
                && buyer.input_target.extract(HIGH_WORD) == 0
                && lt(supply.available, buyer.input_target)
                && equal(supply.available, selection.proposal.quantity)
            {
                let days = minimum_delivery_days(config, buyer.access, firm, trip);
                let lower = plus(
                    ticks(supply.quote, 0),
                    times32(days, config.delivery_penalty),
                )
                .value;
                let upper = plus(ticks(supply.quote, 0), buyer.maximum_delivery_cost).value;
                let price_floor = select(upper.extract(HIGH_WORD) == 0, lower.extract(LOW_WORD), 0);
                relevant = price_floor < selection.score.extract(LOW_WORD);
            }
        }
        if relevant {
            let request = OfferRequest {
                buyer: buyer.id,
                seller: firm,
                trip,
                buyer_access: buyer.access,
                demand: BuyerDemand {
                    wanted: buyer.demand.wanted,
                    purchasing_power: buyer.demand.purchasing_power,
                },
            };
            offer.replace(match comptime!(parameters.market) {
                MarketFlag::Consumer(_) => request.consumer(config, supply),
                _ => request.with_supply(
                    economy,
                    &self.faults[fault_start..fault_start + fault_words],
                    parameters,
                    supply,
                ),
            });
        }
        offer
    }
}

#[cube]
pub(super) fn minimum_delivery_days(
    config: &EconomyConfig,
    buyer_access: u32,
    seller: u32,
    trip: u32,
) -> u32 {
    let mut days = max(ceildiv(trip, config.shipment_speed), 1);
    if comptime!(config.railway.routes_enabled == Feature::Enabled) && config.railway.nodes > 0 {
        let access = buyer_access + config.railway.access[seller as usize * 2 + 1];
        days = min(days, ceildiv(access, config.shipment_speed) + 1);
    }
    days
}

#[cube]
pub(in crate::simulation::economy) fn supplier_stride(
    actors: &ActorTable<'_>,
    parameters: &EconomyParameters,
    buyer: u32,
    length: u32,
) -> u32 {
    let mut stride = 1u32;
    let sampled =
        comptime!(parameters.market == MarketFlag::Household) && length > HOUSEHOLD_SEARCH_SAMPLES;
    let industrial = comptime!(matches!(parameters.market, MarketFlag::Input(_)))
        && !actors.kind(buyer).is(comptime!(ActorKind::Household));
    if (sampled || industrial) && length > 1 {
        stride = coprime_stride(
            length,
            (buyer + parameters.settlement_rotation) % max(length / 2, 1),
        );
    }
    stride
}

#[cube]
pub(in crate::simulation::economy) fn coprime_stride(length: u32, offset: u32) -> u32 {
    let mut stride = max(length / 2 + offset, 1);
    let mut divisor = 0u32;
    while divisor != 1 {
        let mut remaining = length;
        divisor = stride;
        while remaining > 0 {
            let next = divisor % remaining;
            divisor = remaining;
            remaining = next;
        }
        stride += u32::cast_from(divisor != 1);
    }
    stride
}

#[cube]
pub(in crate::simulation::economy) fn supplier_samples(
    parameters: &EconomyParameters,
    length: u32,
) -> u32 {
    match comptime!(parameters.market) {
        MarketFlag::Household => min(length, HOUSEHOLD_SEARCH_SAMPLES),
        _ => length,
    }
}

#[cube]
pub(in crate::simulation::economy) fn supply_score(
    economy: &EconomyView<'_>,
    parameters: &EconomyParameters,
    buyer: u32,
    wanted: Ticks,
    amount: Ticks,
    price: Ticks,
) -> Ticks {
    let mut score = price;
    if comptime!(matches!(parameters.market, MarketFlag::Input(_)))
        && !economy
            .actors
            .kind(buyer)
            .is(comptime!(ActorKind::Household))
    {
        let capacity = economy.actors.capacity(buyer);
        let daily = times32(capacity, economy.ledger.number(Field::InputTicks, buyer));
        let missing = positive(minimum(wanted, daily), amount);
        score = ticks(price.extract(LOW_WORD), missing.extract(LOW_WORD));
    }
    score
}

#[derive(CubeType)]
pub(in crate::simulation::economy) struct OfferRequest {
    pub(super) buyer: u32,
    pub(super) seller: u32,
    pub(super) trip: u32,
    pub(super) buyer_access: u32,
    pub(super) demand: BuyerDemand,
}

#[cube]
impl OfferRequest {
    pub(in crate::simulation::economy) fn labor(
        economy: &EconomyView<'_>,
        buyer: u32,
        seller: u32,
    ) -> Self {
        let config = economy.ledger.config;
        let trip = economy.actors.distance(buyer, seller);
        let mut buyer_access = 0u32;
        if comptime!(config.railway.routes_enabled == Feature::Enabled) && config.railway.nodes > 0
        {
            buyer_access = config.railway.access[buyer as usize * 2 + 1];
        }
        OfferRequest {
            buyer,
            seller,
            trip,
            buyer_access,
            demand: BuyerDemand {
                wanted: zero(),
                purchasing_power: zero(),
            },
        }
    }

    pub(in crate::simulation::economy) fn evaluate(
        &self,
        economy: &EconomyView<'_>,
        control: &[Atomic<u32>],
        parameters: &EconomyParameters,
    ) -> Candidate {
        let supply = index::Supply::new(economy, self.seller, parameters.traded_good());
        self.with_supply(economy, control, parameters, &supply)
    }

    pub(super) fn with_supply(
        &self,
        economy: &EconomyView<'_>,
        control: &[Atomic<u32>],
        parameters: &EconomyParameters,
        supply: &index::Supply,
    ) -> Candidate {
        let id = self.buyer;
        let firm = self.seller;
        let dist = self.trip;
        let buyer_access = self.buyer_access;
        let wanted = self.demand.wanted;
        let purchasing_power = self.demand.purchasing_power;
        let config = economy.ledger.config;
        let market = comptime!(parameters.market);
        let mut eligible = false;
        let mut seen_supply = false;
        let mut empty_seller = u32::new(comptime!(i64::from(NO_ACTOR)));
        let mut empty_quantity = zero();
        let empty_score = ticks(u32::MAX, u32::MAX);
        let mut score = zero();
        let mut amount = zero();
        let mut accessible = dist <= config.search_radius;
        if comptime!(market == MarketFlag::Input(InputMarket::Grain)) {
            accessible = accessible && household_production::grain_counterparty(economy, id, firm);
        }
        if accessible {
            let mut wage = economy.ledger.number(Field::Wage, firm);
            let quote = supply.quote;
            let mut inspect = true;
            if comptime!(market != MarketFlag::Labor) {
                let cash = purchasing_power;
                inspect = !lt(cash, ticks(max(quote, 1), 0));
                if !inspect {
                    seen_supply = seen_supply || supply.offered;
                }
            }
            if inspect {
                eligible = true;
                let days = match comptime!(market) {
                    MarketFlag::Labor => max(ceildiv(dist, config.shipment_speed), 1),
                    _ => railway::DeliveryRequest {
                        buyer: id,
                        seller: firm,
                        trip: dist,
                        buyer_access,
                    }
                    .days(config),
                };
                let quote = match comptime!(market) {
                    MarketFlag::Labor => quote,
                    _ => cartage::delivered_price(config, quote, dist, days),
                };
                score = ticks(quote + days * config.delivery_penalty, 0u32);
                amount = ticks(1u32, 0u32);
                if comptime!(market == MarketFlag::Labor) {
                    if comptime!(config.bargaining.enabled == Feature::Enabled) {
                        let members = employment::payroll_members(&economy.actors, config, id);
                        let negotiation = negotiate(
                            NegotiationRequest {
                                prior_gross: divide(
                                    economy.ledger.value(Field::GrossIncome, id),
                                    members,
                                ),
                                household_cash: divide(
                                    economy.ledger.value(Field::Cash, id),
                                    members,
                                ),
                                funded_per_packet: divide(
                                    economy.ledger.value(Field::Cash, firm),
                                    max(economy.ledger.number(Field::Slots, firm), 1),
                                ),
                                candidate_wage: wage,
                                payable_minutes: economy.ledger.number(Field::LaborMinutes, firm),
                            },
                            config,
                            control,
                        );
                        eligible = eligible
                            && negotiation.status == comptime!(NegotiationStatus::Agreed as u32);
                        wage = negotiation.wage.extract(LOW_WORD);
                        amount = negotiation.wage;
                    }
                    let team = employment::team(&economy.ledger, id);
                    let slots = economy.ledger.number(Field::Slots, firm);
                    let shifts = min(slots, team);
                    if comptime!(config.cohorts == Feature::Enabled) {
                        amount = ticks(amount.extract(LOW_WORD), shifts);
                    }
                    let duration = economy.ledger.number(Field::LaborMinutes, firm)
                        + ceildiv(dist, config.commute_speed) * COMMUTE_LEGS;
                    eligible = eligible
                        && slots > 0
                        && economy.ledger.number(Field::Time, id) / team >= duration;
                    score = divide(
                        times32(
                            wage * economy.ledger.number(Field::LaborMinutes, firm),
                            POLICY_SCALE,
                        ),
                        duration,
                    );
                    if comptime!(config.home_mode == Feature::Enabled) {
                        let conditions = 500
                            + economy.ledger.number(Field::EquipmentCondition, firm) / 2
                            + economy.ledger.number(Field::SafetyProvision, firm) / 4;
                        score = ratio(score, conditions, 1000).record(control);
                        let need = economy.ledger.number(Field::FoodNeed, id);
                        let bread = economy.ledger.number(Field::Bread, id);
                        let missing = max(need, bread) - bread;
                        let preparation = food::HomeKitchen::new(&economy.ledger, id)
                            .work(config, control, missing);
                        let minutes = preparation.minutes;
                        let time = economy.ledger.number(Field::Time, id);
                        let committed = duration * shifts;
                        let remaining = max(time, committed) - committed
                            + economy.ledger.number(Field::HomeTime, id);
                        let self_provision = missing > 0
                            && !lt(economy.ledger.value(Field::Grain, id), preparation.grain);
                        let paid = times32(
                            wage,
                            economy.ledger.number(Field::LaborMinutes, firm) * shifts,
                        );
                        let food_cost =
                            times32(missing, economy.ledger.number(Field::FoodPrice, id));
                        eligible = eligible
                            && !(self_provision && remaining < minutes && lt(paid, food_cost));
                    }
                    let incumbent = economy.ledger.number(Field::Job, id) == firm + 1;
                    let doubled = plus(score, score).record(control);
                    score = ticks(
                        doubled.extract(LOW_WORD) | u32::cast_from(incumbent),
                        doubled.extract(HIGH_WORD),
                    );
                } else {
                    let offered = supply.offered;
                    let stocked = minimum(wanted, supply.available);
                    let payment = times(stocked, max(quote, 1u32));
                    let funded = payment.faults.is_empty() && !lt(purchasing_power, payment.value);
                    let mut affordable = stocked;
                    if !offered || !funded {
                        affordable = minimum(wanted, divide(purchasing_power, max(quote, 1u32)));
                    }
                    seen_supply = seen_supply || (eligible && offered);
                    let known_producer = supply.producer;
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
                    }
                    amount = minimum(affordable, supply.available);
                    eligible = eligible
                        && firm != id
                        && offered
                        && quote > 0u32
                        && !equal(amount, ticks(0u32, 0u32));
                    score = supply_score(economy, parameters, id, wanted, amount, score);
                }
            }
        }
        let empty = empty_seller != NO_ACTOR;
        Candidate {
            eligible,
            seen_supply,
            empty,
            quantity: select(empty, empty_quantity, amount),
            score,
            trip: dist,
        }
    }

    fn consumer(&self, config: &EconomyConfig, supply: &index::Supply) -> Candidate {
        let id = self.buyer;
        let firm = self.seller;
        let dist = self.trip;
        let buyer_access = self.buyer_access;
        let wanted = self.demand.wanted;
        let purchasing_power = self.demand.purchasing_power;
        let mut eligible = false;
        let mut seen_supply = false;
        let mut empty = false;
        let mut score = zero();
        if dist <= config.search_radius {
            seen_supply = supply.offered;
            let quote = supply.quote;
            if !lt(purchasing_power, ticks(max(quote, 1), 0)) {
                let days = railway::DeliveryRequest {
                    buyer: id,
                    seller: firm,
                    trip: dist,
                    buyer_access,
                }
                .days(config);
                let quote = cartage::delivered_price(config, quote, dist, days);
                score = ticks(quote + days * config.delivery_penalty, 0);
                let usable = firm != id
                    && supply.quote > 0
                    && !lt(purchasing_power, ticks(quote, 0))
                    && !equal(wanted, zero());
                eligible = usable && supply.offered;
                empty = usable && supply.producer && !supply.offered;
            }
        }
        Candidate {
            eligible,
            seen_supply,
            empty,
            quantity: zero(),
            score,
            trip: dist,
        }
    }
}

#[derive(CubeType)]
pub(super) struct BuyerLocation {
    pub x: u32,
    pub y: u32,
    pub access: u32,
}

#[cube]
impl BuyerLocation {
    pub(super) fn new(actors: &ActorTable<'_>, config: &EconomyConfig, buyer: u32) -> Self {
        let mut access = 0u32;
        if comptime!(config.railway.routes_enabled == Feature::Enabled) && config.railway.nodes > 0
        {
            access = config.railway.access[buyer as usize * 2 + 1];
        }
        BuyerLocation {
            x: actors.x(buyer),
            y: actors.y(buyer),
            access,
        }
    }

    pub(super) fn distance_to_bounds(&self, min_x: u32, max_x: u32, min_y: u32, max_y: u32) -> u32 {
        max(self.x, min_x) - min(self.x, max_x) + max(self.y, min_y) - min(self.y, max_y)
    }
}

#[derive(CubeType)]
pub(super) struct MatchingBuyer {
    pub id: u32,
    pub demand: BuyerDemand,
    pub input_target: Ticks,
    pub access: u32,
    pub maximum_delivery_cost: Ticks,
}

#[derive(CubeType)]
pub(super) struct CandidateWindow {
    pub width: u32,
    pub start: u32,
    pub limit: u32,
}

#[cube]
impl CandidateWindow {
    pub(super) fn includes_lane(&self) -> bool {
        UNIT_POS < self.width && self.start + UNIT_POS < self.limit
    }

    pub(super) fn slots(&self) -> usize {
        min(self.width as usize, (self.limit - self.start) as usize)
    }
}

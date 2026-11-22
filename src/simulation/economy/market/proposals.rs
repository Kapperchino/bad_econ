use crate::simulation::economy::*;

#[derive(CubeType)]
pub(in crate::simulation::economy) struct MarketProposal {
    pub peer: u32,
    pub quantity: Ticks,
    pub trip: u32,
}

#[cube]
impl MarketProposal {
    pub fn store(&self, proposals: &mut [u32], actor: u32) {
        let base = actor as usize * ProposalLayout::WORDS;
        proposals[base + ProposalLayout::ACTOR] = self.peer;
        proposals[base + ProposalLayout::QUANTITY_LOW] = self.quantity.extract(LOW_WORD);
        proposals[base + ProposalLayout::QUANTITY_HIGH] = self.quantity.extract(HIGH_WORD);
        proposals[base + ProposalLayout::DISTANCE] = self.trip;
    }
}

#[derive(CubeType)]
pub(in crate::simulation::economy) struct InputDemand {
    pub search: bool,
    pub wanted: Ticks,
}

#[cube]
pub(in crate::simulation::economy) fn trial_packet(
    ledger: &Ledger<'_>,
    firm: u32,
    good: u32,
) -> Ticks {
    let config = ledger.config;
    let empty = comptime!(
        config.economy == crate::domain::Economy::Industrial
            && config.home_mode == Feature::Enabled
    ) && ledger.number(Field::EnterpriseModel, firm) == 0
        && ledger.number(Field::EnterpriseStatus, firm)
            == comptime!(crate::domain::enterprise::EnterpriseState::Operating as u32)
        && !farming::FarmSite::new(firm, good).is_farm(ledger)
        && equal(ledger.value_at(good, firm), zero())
        && equal(ledger.value(Field::Pipeline, firm), zero());
    ticks(u32::cast_from(empty), 0)
}

#[cube]
impl InputDemand {
    pub(super) fn new(
        economy: &mut EconomyView<'_>,
        control: &[Atomic<u32>],
        id: u32,
        k: &KernelActorKind,
        good: u32,
    ) -> Self {
        let config = economy.ledger.config;
        let pricing = pricing::FirmPricing::new(id, k.output(), control);
        let mut search = !k.is(comptime!(ActorKind::Household))
            && (economy.ledger.number(Field::FirstInput, id) == good
                || economy.ledger.number(Field::SecondInput, id) == good);
        let mut wanted = ticks(
            economy.actors.capacity(id)
                * economy.ledger.number(Field::InputTicks, id)
                * consumption::input_coverage(config, k.bits()),
            0u32,
        );
        if economy.ledger.number(Field::EnterpriseModel, id) == 1 {
            search = good == k.output();
            let forecast = maximum(economy.ledger.value(Field::Demand, id), ticks(1, 0));
            wanted = pricing.distributor_target(&*economy, forecast);
        }
        if !k.is(comptime!(ActorKind::Household))
            && economy.ledger.number(Field::EnterpriseModel, id) == 0
            && comptime!(config.home_mode == Feature::Enabled)
        {
            let horizon = config.coverage + config.ring;
            let packets = divide(
                plus(
                    times(economy.ledger.value(Field::Demand, id), horizon).record(control),
                    ticks(economy.ledger.number(Field::OutputTicks, id) - 1, 0),
                )
                .record(control),
                max(economy.ledger.number(Field::OutputTicks, id), 1),
            );
            let capacity = economy.actors.capacity(id);
            let profitable = comptime!(config.economy != crate::domain::Economy::Industrial)
                || config.price_response == 0
                || pricing.covers_batch(&economy.ledger);
            wanted = times(
                minimum(
                    maximum(
                        select(profitable, packets, zero()),
                        trial_packet(&economy.ledger, id, k.output()),
                    ),
                    times32(capacity, horizon),
                ),
                economy.ledger.number(Field::InputTicks, id),
            )
            .record(control);
        }
        if k.is(comptime!(ActorKind::Household))
            && comptime!(config.home_mode == Feature::Enabled)
            && good == comptime!(Good::Grain as u32)
            && (comptime!(config.economy == crate::domain::Economy::Agrarian)
                || economy.ledger.number(Field::LandPlots, id) > 0)
            && economy.ledger.number(Field::LivingMembers, id) > 0
        {
            search = true;
            let near_term = times32(
                economy.ledger.number(Field::FoodNeed, id),
                config.coverage + config.ring,
            );
            let reserve = economy.ledger.value(Field::GrainReserve, id);
            let quoting = economy.ledger.number(Field::GrainQuote, id) > 0;
            wanted = select(quoting, minimum(near_term, reserve), near_term);
        }
        let investment = economy.ledger.number(Field::InvestmentGood, id) > 0;
        if k.is(comptime!(ActorKind::Household))
            && investment
            && comptime!(config.property_mode == Feature::Disabled)
            && economy.ledger.number(Field::LivingMembers, id) > 0
            && comptime!(config.home_mode == Feature::Enabled)
            && (good == comptime!(Good::Tools as u32) || good == comptime!(Good::Wood as u32))
        {
            search = true;
            let members = crate::simulation::enterprise::founder_shares(&*economy, id);
            wanted = times32(
                members,
                select(good == comptime!(Good::Tools as u32), 2u32, 5u32),
            );
        }
        if !k.is(comptime!(ActorKind::Household))
            && comptime!(
                config.home_mode == Feature::Enabled && config.property_mode == Feature::Disabled
            )
            && good == comptime!(Good::Wood as u32)
            && economy.ledger.number(Field::Slots, id) == 0
            && economy.ledger.number(Field::Plan, id) > 0
            && !lt(economy.ledger.value(Field::Cash, id), ticks(10_000, 0))
        {
            wanted = select(search, maximum(wanted, ticks(5, 0)), ticks(5, 0));
            search = true;
        }
        if !k.is(comptime!(ActorKind::Household))
            && k.bits() != comptime!(Good::Railway as u32)
            && good == comptime!(Good::Tools as u32)
            && comptime!(config.home_mode == Feature::Enabled)
        {
            wanted = select(search, maximum(wanted, ticks(2, 0)), ticks(2, 0));
            search = true;
        }
        if good == comptime!(Good::Construction as u32) {
            let building = economy.ledger.number(Field::ProjectStatus, id)
                == comptime!(crate::domain::property::ProjectStatus::Building as u32);
            let work = economy.ledger.value(Field::ConstructionRemaining, id);
            wanted = select(
                building,
                select(search, maximum(wanted, work), work),
                wanted,
            );
            search = search || building;
            if k.bits() == comptime!(Good::Housing as u32) {
                wanted = plus(wanted, landlords::maintenance(&economy.ledger, id)).record(control);
                search = true;
            }
        }
        let rail = railway::demand(economy, id, good);
        InputDemand {
            search: search || !equal(rail, zero()),
            wanted: maximum(wanted, rail),
        }
    }
}

#[allow(
    clippy::collapsible_match,
    reason = "CubeCL uses explicit runtime branches"
)]
#[cube]
pub(in crate::simulation::economy) fn propose(
    economy: &mut EconomyView<'_>,
    spatial: &mut [u32],
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
            let cells = config.spatial_side * config.spatial_side;
            let base =
                3 * (cells + 1 + config.actors) + ceildiv(config.actors, DISCOVERY_BLOCK) + 1;
            id = spatial[(base + invocation) as usize];
        }
        let k = economy.actors.kind(id);
        let mut quantity = ticks(0u32, 0u32);
        let mut trip = 0u32;
        let mut search = false;
        let mut good = 0u32;
        let mut wanted = ticks(0u32, 0u32);
        let mut seen_supply = false;
        let mut empty_seller = u32::new(comptime!(i64::from(NO_ACTOR)));
        let mut empty_quantity = zero();
        let mut empty_trip = 0u32;
        let mut empty_score = ticks(u32::MAX, u32::MAX);
        let mut missing_supply = false;
        let mut purchasing_power = economy.ledger.value(Field::Cash, id);
        if comptime!(matches!(market, MarketFlag::Input(_))) {
            purchasing_power =
                enterprises::FirmPolicy::new(id, control).purchasing_power(&*economy);
        }
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
            MarketFlag::Input(_) => {
                good = parameters.traded_good();
                let demand = InputDemand::new(economy, control, id, &k, good);
                search = demand.search;
                wanted = demand.wanted;
            }
            MarketFlag::Consumer(_) => {
                good = parameters.traded_good();
                let purchase = consumption::HouseholdPurchase::new(
                    &*economy, &*spatial, control, parameters, id,
                );
                if good == comptime!(Good::Coal as u32) || good == comptime!(Good::Wood as u32) {
                    wanted = purchase.fuel_target(good);
                    let held = plus(
                        economy.ledger.value_at(good, id),
                        economy
                            .ledger
                            .value_at(TRANSIT_FIELDS as u32 + good - 1, id),
                    )
                    .record(control);
                    search = k.is(comptime!(ActorKind::Household))
                        && economy.ledger.number(Field::LivingMembers, id) > 0
                        && !lt(
                            economy.ledger.value(Field::Bread, id),
                            ticks(food::HouseholdRation::new(&economy.ledger, id).minimum(), 0),
                        )
                        && lt(held, wanted)
                        && !equal(purchasing_power, zero());
                    if search {
                        search = purchase.preferred_fuel(good) == good;
                    }
                } else {
                    let interval = config.consumption_intervals[good as usize - 1];
                    let until_due =
                        (interval - (parameters.day + id) % max(interval, 1)) % max(interval, 1);
                    search = k.is(comptime!(ActorKind::Household))
                        && economy.ledger.number(Field::LivingMembers, id) > 0
                        && (until_due <= config.coverage + config.ring
                            || consumption::scaled_luxury(config, good));
                    let reserve = ticks(
                        food::HouseholdRation::new(&economy.ledger, id).reserve(config)
                            * (config.coverage + config.ring - 1),
                        0,
                    );
                    let food = plus(
                        economy.ledger.value(Field::Bread, id),
                        economy.ledger.value_at(
                            TRANSIT_FIELDS as u32 + comptime!(Good::Bread as u32) - 1,
                            id,
                        ),
                    )
                    .record(control);
                    let cash_reserve =
                        times(reserve, economy.ledger.number(Field::FoodPrice, id)).record(control);
                    search = search
                        && !lt(food, reserve)
                        && lt(cash_reserve, economy.ledger.value(Field::Cash, id));
                    purchasing_power = purchase.budget(good);
                    search = search && !equal(purchasing_power, zero());
                    wanted = ticks(1, 0);
                    if search {
                        wanted = purchase.durable_target(good, purchasing_power);
                    }
                }
            }
            MarketFlag::Household => {
                good = comptime!(Good::Bread as u32);
                search = k.is(comptime!(ActorKind::Household));
                wanted = ticks(
                    food::HouseholdRation::new(&economy.ledger, id).purchase(config, parameters)
                        * (config.coverage + config.ring - 1u32),
                    0u32,
                );
            }
            MarketFlag::Payroll => {
                quantity = economy.ledger.value(Field::Due, id);
                if !equal(quantity, ticks(0u32, 0u32)) {
                    chosen = economy.ledger.number(Field::Employer, id) - 1u32;
                }
            }
            MarketFlag::Distribution => {
                if !k.is(comptime!(ActorKind::Household)) {
                    quantity = ratio(
                        positive(
                            economy.ledger.value(Field::Cash, id),
                            maximum(
                                select(
                                    k.bits() == comptime!(Good::Housing as u32),
                                    zero(),
                                    ticks(config.reserve_low, config.reserve_high),
                                ),
                                economy.ledger.value(Field::OperatingReserve, id),
                            ),
                        ),
                        config.distribution_rate,
                        POLICY_SCALE,
                    )
                    .record(control);
                    let owner = economy.actors.owner(id);
                    let estate = economy.ledger.number(Field::EstateSettled, owner);
                    if !equal(quantity, zero())
                        && estate
                            != comptime!(crate::domain::enterprise::EstateStatus::Unclaimed as u32)
                    {
                        chosen = owner;
                    }
                }
            }
        }
        if good > 0u32 {
            wanted = positive(
                wanted,
                plus(
                    economy.ledger.value_at(good, id),
                    economy
                        .ledger
                        .value_at(TRANSIT_FIELDS as u32 + good - 1, id),
                )
                .record(control),
            );
        }
        search = search
            && economy.ledger.number(Field::Time, id)
                >= select(comptime!(market == MarketFlag::Labor), 1u32, 0);
        if parameters.traded_good() > 0 {
            purchasing_power = minimum(
                purchasing_power,
                positive(
                    economy.ledger.value(Field::Cash, id),
                    economy.ledger.value(Field::CartagePayable, id),
                ),
            );
        }
        if good > 0 {
            search = search && !equal(wanted, zero()) && !equal(purchasing_power, zero());
        }
        if comptime!(labor_round != employment::LaborRound::Retained) && search {
            let px = economy.actors.x(id);
            let py = economy.actors.y(id);
            let radius = config.search_radius;
            let cx = px / radius;
            let cy = py / radius;
            let side = config.spatial_side;
            let mut best = ticks(u32::MAX, u32::MAX);
            if comptime!(market == MarketFlag::Labor) {
                best = ticks(0u32, 0u32);
            }
            let mut considered = 0u32;
            let mut y = max(cy, 1u32) - 1u32;
            while y <= min(cy + 1u32, side - 1u32) && considered < DISCOVERY_CANDIDATES {
                let mut x = max(cx, 1u32) - 1u32;
                while x <= min(cx + 1u32, side - 1u32) && considered < DISCOVERY_CANDIDATES {
                    let cell = y * side + x;
                    let mut j = spatial[cell as usize];
                    let begin = j;
                    let length = spatial[(cell + 1) as usize] - begin;
                    let limit = begin + search::supplier_samples(parameters, length);
                    let stride = search::supplier_stride(&economy.actors, parameters, id, length);
                    let mut cursor = (parameters.settlement_rotation + id) % max(length, 1);
                    let nearest = max(px, x * radius) - min(px, (x + 1) * radius - 1)
                        + max(py, y * radius)
                        - min(py, (y + 1) * radius - 1);
                    if nearest > radius {
                        j = spatial[(cell + 1) as usize];
                    }
                    while j < limit && considered < DISCOVERY_CANDIDATES {
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
                            if comptime!(market == MarketFlag::Input(InputMarket::Grain)) {
                                accessible = accessible
                                    && household_production::grain_counterparty(
                                        &*economy, id, firm,
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
                                    firm,
                                    good,
                                );
                                let quote = economy.ledger.number_at(terms.price, firm);
                                let quote = match comptime!(market) {
                                    MarketFlag::Labor => quote,
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
                                    let mut eligible = dist <= config.search_radius;

                                    let mut score =
                                        ticks(quote + route.days * config.delivery_penalty, 0u32);
                                    let amount = if comptime!(market == MarketFlag::Labor) {
                                        let offer =
                                            search::OfferRequest::labor(&*economy, id, firm)
                                                .evaluate(&*economy, control, parameters);
                                        eligible = offer.eligible;
                                        score = offer.score;
                                        offer.quantity
                                    } else {
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
                                            && economy.actors.kind(firm).output() == good;
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
                                                    economy.ledger.value_at(good, firm),
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
            let mut failure = u32::cast_from(missing_supply);
            if !equal(wanted, zero()) && chosen == NO_ACTOR {
                failure = select(seen_supply || equal(cash, zero()), 2u32, 1u32);
            }
            economy.ledger.set_number(Field::FoodAccess, id, failure);
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

use super::*;
use crate::simulation::configuration::Configuration;
use crate::simulation::{
    flags::InputMarket,
    markets::{LATCH_WORD, SEARCHES_WORD},
};
mod empty;
mod filter;
mod fuel;
pub(in crate::simulation) mod grain;
mod index;
mod road;
pub(in crate::simulation) use road::SIDE as ROAD_TILE_SIDE;

pub(in crate::simulation) const INPUT_MINIMUM_QUOTE_WORD: usize = CONTROL_WORDS + 7;
pub(in crate::simulation) const INPUT_RAIL_WORD: usize = markets::WIDE_ORDERS_WORD;
pub(in crate::simulation) const INPUT_BUYERS_FIELD: u32 = 12;
pub(super) const GRAIN_FIRMS_WORD: usize = CONTROL_WORDS + 8;

#[cfg(test)]
mod tests;

#[derive(Clone, Copy)]
#[repr(u32)]
enum Preparation {
    Demand,
    Inventory,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PreparationRound {
    First,
    Capture,
    Selected,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(in crate::simulation) enum StatusReset {
    Full,
    Active,
}

#[cube]
fn prepare(
    economy: &mut EconomyView<'_>,
    spatial: &mut [u32],
    proposals: &mut [u32],
    control: &[Atomic<u32>],
    searches: &mut [u32],
    successes: &[Atomic<u32>],
    scratch: &mut [u32],
    parameters: &EconomyParameters,
    invocation: u32,
    #[comptime] round: PreparationRound,
    #[comptime] status_reset: StatusReset,
) {
    let config = economy.ledger.config;
    let mut id = invocation;
    let mut searching = false;
    let mut eligible = false;
    let masks = ceildiv(config.actors, 32);
    if invocation < masks + ceildiv(config.actors, markets::COMPACTION_BLOCK) {
        successes[invocation as usize].store(0);
    }
    let count = match comptime!(round) {
        PreparationRound::Selected => control[markets::PREPARATION_ACTORS_WORD].load(),
        _ => config.actors,
    };
    if invocation < count {
        if comptime!(round == PreparationRound::Selected) {
            id = searches[(5 * config.actors + invocation) as usize];
        }
        if comptime!(status_reset == StatusReset::Full) {
            KernelEventStatus::new(comptime!(EventStatus::Inactive))
                .store(scratch, 4 * config.actors + invocation);
        }
        let market = comptime!(parameters.market);
        if comptime!(
            matches!(market, MarketFlag::Consumer(_)) && round == PreparationRound::Capture
        ) {
            searches[(14 * config.actors + id) as usize] = 0;
        }
        if comptime!(matches!(
            market,
            MarketFlag::Consumer(_) | MarketFlag::Household
        )) {
            searches[(config.actors + id) as usize] = 0;
            searches[(2 * config.actors + id) as usize] = 0;
        }
        let k = economy.actors.kind(id);
        let mut chosen = u32::new(comptime!(i64::from(NO_ACTOR)));
        let mut quantity = ticks(0u32, 0u32);
        let mut trip = 0u32;
        let mut search = false;
        let mut good = 0u32;
        let mut wanted = ticks(0u32, 0u32);
        let household_demand = match comptime!(parameters.market) {
            MarketFlag::Input(_) => {
                let grain = parameters.traded_good() == Good::Grain as u32
                    && comptime!(config.home_mode == Feature::Enabled)
                    && (comptime!(config.economy == crate::domain::Economy::Agrarian)
                        || economy.ledger.number(Field::LandPlots, id) > 0)
                    && economy.ledger.number(Field::LivingMembers, id) > 0;
                let investment = (parameters.traded_good() == Good::Tools as u32
                    || parameters.traded_good() == Good::Wood as u32)
                    && comptime!(
                        config.home_mode == Feature::Enabled
                            && config.property_mode == Feature::Disabled
                    )
                    && economy.ledger.number(Field::InvestmentGood, id) > 0
                    && economy.ledger.number(Field::LivingMembers, id) > 0;
                let construction = parameters.traded_good() == Good::Construction as u32
                    && economy.ledger.number(Field::ProjectStatus, id)
                        == comptime!(crate::domain::property::ProjectStatus::Building as u32);
                u32::cast_from(grain || investment || construction)
            }
            _ => u32::new(1),
        };
        let preparation = match comptime!(round) {
            PreparationRound::Selected => {
                u32::new(comptime!(i64::from(Preparation::Demand as u32)))
            }
            _ => select(
                household_demand == 0
                    && k.is(comptime!(ActorKind::Household))
                    && economy.ledger.number(Field::EnterpriseModel, id) == 0,
                comptime!(Preparation::Inventory as u32),
                comptime!(Preparation::Demand as u32),
            ),
        };
        if comptime!(round == PreparationRound::Capture) {
            eligible = preparation == comptime!(Preparation::Demand as u32);
        }
        match preparation {
            0 => {
                let mut purchasing_power = economy.ledger.value(Field::Cash, id);
                if comptime!(matches!(market, MarketFlag::Input(_))) {
                    purchasing_power = enterprises::purchasing_power(&*economy, control, id);
                }
                match market {
                    MarketFlag::Labor => {
                        let labor = employment::LaborSearch::new(
                            &*economy,
                            control,
                            parameters,
                            id,
                            comptime!(employment::LaborRound::FromClock),
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
                        if comptime!(matches!(
                            market,
                            MarketFlag::Consumer(Good::Coal | Good::Wood)
                        )) {
                            wanted = consumption::fuel_purchase_target(
                                &economy.ledger,
                                control,
                                parameters,
                                id,
                                good,
                            );
                            let held = add_checked(
                                control,
                                economy.ledger.value_at(good, id),
                                economy
                                    .ledger
                                    .value_at(TRANSIT_FIELDS as u32 + good - 1, id),
                            );
                            search = k.is(comptime!(ActorKind::Household))
                                && economy.ledger.number(Field::LivingMembers, id) > 0
                                && !lt(
                                    economy.ledger.value(Field::Bread, id),
                                    ticks(food::minimum_daily(&economy.ledger, id), 0),
                                )
                                && lt(held, wanted)
                                && !equal(purchasing_power, zero());
                            if search {
                                search = consumption::preferred_fuel(
                                    &*economy, &*spatial, parameters, id, good,
                                ) == good;
                            }
                        } else {
                            let interval = config.consumption_intervals[good as usize - 1];
                            let until_due = (interval - (parameters.day + id) % max(interval, 1))
                                % max(interval, 1);
                            search = k.is(comptime!(ActorKind::Household))
                                && economy.ledger.number(Field::LivingMembers, id) > 0
                                && (until_due <= config.coverage + config.ring
                                    || comptime!(
                                        matches!(
                                            market,
                                            MarketFlag::Consumer(
                                                Good::LuxuryClothes | Good::LuxuryFurniture
                                            )
                                        ) && config.food_income
                                            == crate::domain::food::FoodIncome::PurchasingPower
                                    ));
                            let reserve = ticks(
                                food::reserve_daily(&economy.ledger, id)
                                    * (config.coverage + config.ring - 1),
                                0,
                            );
                            let food = add_checked(
                                control,
                                economy.ledger.value(Field::Bread, id),
                                economy.ledger.value_at(
                                    TRANSIT_FIELDS as u32 + comptime!(Good::Bread as u32) - 1,
                                    id,
                                ),
                            );
                            let cash_reserve = multiply(
                                control,
                                reserve,
                                economy.ledger.number(Field::FoodPrice, id),
                            );
                            search = search
                                && !lt(food, reserve)
                                && lt(cash_reserve, economy.ledger.value(Field::Cash, id));
                            if comptime!(matches!(
                                market,
                                MarketFlag::Consumer(Good::LuxuryClothes | Good::LuxuryFurniture)
                            )) {
                                purchasing_power =
                                    consumption::budget(&economy.ledger, control, id, good);
                            }
                            search = search && !equal(purchasing_power, zero());
                            wanted = ticks(consumption::consumers(&economy.ledger, id), 0);
                            if comptime!(matches!(
                                market,
                                MarketFlag::Consumer(Good::LuxuryClothes | Good::LuxuryFurniture)
                            )) && search
                            {
                                wanted = consumption::durable_target(
                                    &*economy,
                                    &*spatial,
                                    control,
                                    parameters,
                                    id,
                                    good,
                                    purchasing_power,
                                );
                            }
                        }
                    }
                    MarketFlag::Household => {
                        good = comptime!(Good::Bread as u32);
                        search = k.is(comptime!(ActorKind::Household));
                        wanted = ticks(
                            food::purchase_daily(&economy.ledger, parameters, id)
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
                            quantity = scaled(
                                control,
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
                            );
                            if !equal(quantity, ticks(0u32, 0u32)) {
                                chosen = economy.actors.owner(id);
                            }
                        }
                    }
                }
                if good > 0u32 {
                    wanted = positive(
                        wanted,
                        add_checked(
                            control,
                            economy.ledger.value_at(good, id),
                            economy
                                .ledger
                                .value_at(TRANSIT_FIELDS as u32 + good - 1, id),
                        ),
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
                searching = search;
                if comptime!(
                    round == PreparationRound::Capture
                        && market == MarketFlag::Input(InputMarket::Grain)
                ) {
                    eligible = select(k.is(comptime!(ActorKind::Household)), searching, eligible);
                }
                if comptime!(
                    round == PreparationRound::Capture && matches!(market, MarketFlag::Consumer(_))
                ) {
                    eligible = searching;
                }
                if comptime!(
                    round == PreparationRound::Capture
                        && matches!(market, MarketFlag::Consumer(Good::Coal | Good::Wood))
                        && config.food_income == crate::domain::food::FoodIncome::PurchasingPower
                ) && !searching
                    && k.is(comptime!(ActorKind::Household))
                {
                    let comfort = consumption::fuel_target(&economy.ledger, control, id, good);
                    let held = add_checked(
                        control,
                        economy.ledger.value_at(good, id),
                        economy
                            .ledger
                            .value_at(TRANSIT_FIELDS as u32 + good - 1, id),
                    );
                    eligible = economy.ledger.number(Field::LivingMembers, id) > 0
                        && !lt(
                            economy.ledger.value(Field::Bread, id),
                            ticks(food::minimum_daily(&economy.ledger, id), 0),
                        )
                        && !equal(purchasing_power, zero())
                        && lt(held, comfort);
                }
                if searching {
                    searches[(config.actors + id) as usize] = wanted.extract(LOW_WORD);
                    searches[(2 * config.actors + id) as usize] = wanted.extract(HIGH_WORD);
                    searches[(3 * config.actors + id) as usize] =
                        purchasing_power.extract(LOW_WORD);
                    searches[(4 * config.actors + id) as usize] =
                        purchasing_power.extract(HIGH_WORD);
                }
                if comptime!(market == MarketFlag::Household) {
                    let failure = select(
                        equal(wanted, zero()),
                        0u32,
                        select(equal(purchasing_power, zero()), 2u32, 1u32),
                    );
                    economy.ledger.set_number(Field::FoodAccess, id, failure);
                }
            }
            _ => {
                if comptime!(matches!(
                    round,
                    PreparationRound::First | PreparationRound::Capture
                )) {
                    let inactive_good = parameters.traded_good();
                    add_checked(
                        control,
                        economy.ledger.value_at(inactive_good, id),
                        economy
                            .ledger
                            .value_at(TRANSIT_FIELDS as u32 + inactive_good - 1, id),
                    );
                }
            }
        }
        if comptime!(matches!(
            market,
            MarketFlag::Labor | MarketFlag::Payroll | MarketFlag::Distribution
        )) {
            proposals[id as usize * ProposalLayout::WORDS + ProposalLayout::ACTOR] = chosen;
            proposals[id as usize * ProposalLayout::WORDS + ProposalLayout::QUANTITY_LOW] =
                quantity.extract(LOW_WORD);
            proposals[id as usize * ProposalLayout::WORDS + ProposalLayout::QUANTITY_HIGH] =
                quantity.extract(HIGH_WORD);
            proposals[id as usize * ProposalLayout::WORDS + ProposalLayout::DISTANCE] = trip;
        }
    }
    let mut packed = u32::cast_from(searching);
    if comptime!(round == PreparationRound::Capture) {
        packed |= u32::cast_from(eligible) << 16;
    }
    let prefix = parallel::scan_prefix(packed, comptime!(parameters.scan));
    let mut searching_total = prefix.total;
    let mut searching_prefix = prefix.inclusive;
    if comptime!(round == PreparationRound::Capture) {
        searching_total &= 0xffff;
        searching_prefix &= 0xffff;
    }
    let mut base = Shared::<u32>::new();
    let mut eligible_base = Shared::<u32>::new();
    if UNIT_POS == 0 {
        *base = match searching_total {
            0 => 0u32,
            _ => control[SEARCHES_WORD].fetch_add(searching_total),
        };
        if comptime!(round == PreparationRound::Capture) {
            let total = prefix.total >> 16;
            *eligible_base = match total {
                0 => 0u32,
                _ => control[markets::PREPARATION_ACTORS_WORD].fetch_add(total),
            };
        }
    }
    sync_cube();
    if searching {
        searches[(*base + searching_prefix - 1) as usize] = id;
    }
    if comptime!(round == PreparationRound::Capture) && eligible {
        searches[(5 * config.actors + *eligible_base + (prefix.inclusive >> 16) - 1) as usize] = id;
    }
}

#[derive(CubeType, Clone, Copy)]
pub(super) struct Candidate {
    pub eligible: bool,
    pub seen_supply: bool,
    pub empty: bool,
    pub quantity: Ticks,
    pub score: Ticks,
    pub trip: u32,
}

#[cube]
impl Candidate {
    fn replace(&mut self, other: Candidate) {
        self.eligible = other.eligible;
        self.seen_supply = other.seen_supply;
        self.empty = other.empty;
        self.quantity = other.quantity;
        self.score = other.score;
        self.trip = other.trip;
    }
}

#[derive(CubeType)]
struct CandidateTile {
    eligible: Shared<[u32]>,
    seen_supply: Shared<[u32]>,
    empty: Shared<[u32]>,
    quantity: Shared<[Ticks]>,
    score: Shared<[Ticks]>,
    trip: Shared<[u32]>,
    firm: Shared<[u32]>,
}

#[cube]
impl CandidateTile {
    fn new(#[comptime] lanes: usize) -> Self {
        CandidateTile {
            eligible: Shared::<[u32]>::new_slice(lanes),
            seen_supply: Shared::<[u32]>::new_slice(lanes),
            empty: Shared::<[u32]>::new_slice(lanes),
            quantity: Shared::<[Ticks]>::new_slice(lanes),
            score: Shared::<[Ticks]>::new_slice(lanes),
            trip: Shared::<[u32]>::new_slice(lanes),
            firm: Shared::<[u32]>::new_slice(lanes),
        }
    }

    fn store(&mut self, lane: usize, offer: &Candidate, firm: u32) {
        self.eligible[lane] = u32::cast_from(offer.eligible);
        self.seen_supply[lane] = u32::cast_from(offer.seen_supply);
        self.empty[lane] = u32::cast_from(offer.empty);
        self.quantity[lane] = offer.quantity;
        self.score[lane] = offer.score;
        self.trip[lane] = offer.trip;
        self.firm[lane] = firm;
    }
}

#[cube]
fn minimum_delivery_days(config: &EconomyConfig, buyer_access: u32, seller: u32, trip: u32) -> u32 {
    let mut days = max(ceildiv(trip, config.shipment_speed), 1);
    if comptime!(config.railway.routes_enabled == Feature::Enabled) && config.railway.nodes > 0 {
        let access = buyer_access + config.railway.access[seller as usize * 2 + 1];
        days = min(days, ceildiv(access, config.shipment_speed) + 1);
    }
    days
}

#[cube]
pub(super) fn supplier_stride(
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
pub(super) fn coprime_stride(length: u32, offset: u32) -> u32 {
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
pub(super) fn supplier_samples(parameters: &EconomyParameters, length: u32) -> u32 {
    match comptime!(parameters.market) {
        MarketFlag::Household => min(length, HOUSEHOLD_SEARCH_SAMPLES),
        _ => length,
    }
}

#[cube]
pub(super) fn supply_score(
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

#[cube]
pub(super) fn candidate(
    economy: &EconomyView<'_>,
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
    id: u32,
    firm: u32,
    wanted: Ticks,
    purchasing_power: Ticks,
) -> Candidate {
    let config = economy.ledger.config;
    let supply = index::Supply::new(economy, firm, parameters.traded_good());
    let trip = economy.actors.distance(id, firm);
    let mut buyer_access = 0u32;
    if comptime!(config.railway.routes_enabled == Feature::Enabled) && config.railway.nodes > 0 {
        buyer_access = config.railway.access[id as usize * 2 + 1];
    }
    candidate_supply(
        economy,
        control,
        parameters,
        id,
        firm,
        trip,
        buyer_access,
        wanted,
        purchasing_power,
        &supply,
    )
}

#[cube]
fn candidate_supply(
    economy: &EconomyView<'_>,
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
    id: u32,
    firm: u32,
    dist: u32,
    buyer_access: u32,
    wanted: Ticks,
    purchasing_power: Ticks,
    supply: &index::Supply,
) -> Candidate {
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
                _ => railway::delivery_days(
                    config,
                    railway::DeliveryRequest {
                        buyer: id,
                        seller: firm,
                        trip: dist,
                        buyer_access,
                    },
                ),
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
                            household_cash: divide(economy.ledger.value(Field::Cash, id), members),
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
                    score = scaled(control, score, conditions, 1000);
                    let need = economy.ledger.number(Field::FoodNeed, id);
                    let bread = economy.ledger.number(Field::Bread, id);
                    let missing = max(need, bread) - bread;
                    let preparation = food::household_work(&economy.ledger, control, id, missing);
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
                    let food_cost = times32(missing, economy.ledger.number(Field::FoodPrice, id));
                    eligible =
                        eligible && !(self_provision && remaining < minutes && lt(paid, food_cost));
                }
                let incumbent = economy.ledger.number(Field::Job, id) == firm + 1;
                let doubled = add_checked(control, score, score);
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

#[cube]
fn consumer_candidate(
    config: &EconomyConfig,
    id: u32,
    firm: u32,
    dist: u32,
    buyer_access: u32,
    wanted: Ticks,
    purchasing_power: Ticks,
    supply: &index::Supply,
) -> Candidate {
    let mut eligible = false;
    let mut seen_supply = false;
    let mut empty = false;
    let mut score = zero();
    if dist <= config.search_radius {
        seen_supply = supply.offered;
        let quote = supply.quote;
        if !lt(purchasing_power, ticks(max(quote, 1), 0)) {
            let days = railway::delivery_days(
                config,
                railway::DeliveryRequest {
                    buyer: id,
                    seller: firm,
                    trip: dist,
                    buyer_access,
                },
            );
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

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(in crate::simulation) enum SearchStage {
    Index,
    EmptySort,
    EmptyBounds,
    Prepare,
    PrepareCandidates,
    PrepareSelected,
    Search,
    SearchIndexed,
    GrainFirms,
    GrainDealerTiles,
    SearchGrain,
    #[cfg_attr(not(test), allow(dead_code))]
    SearchFuel,
    #[cfg_attr(not(test), allow(dead_code))]
    SearchHousehold,
    SearchSpatialFuel,
    FilterEmpty,
    FilterBuyers,
    StockLimits,
    RoadCandidates,
    FinishFiltered,
    Compact,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(super) enum SupplierSearch {
    Inactive,
    SingleCell { shipment_speed: u32 },
    Neighborhood { shipment_speed: u32 },
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(in crate::simulation) enum OrderCollection {
    Bitmap,
    InputTile,
    ConsumerMask,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(in crate::simulation) struct SearchProgram {
    collection: OrderCollection,
    stage: SearchStage,
    suppliers: SupplierSearch,
}

impl SearchProgram {
    pub(in crate::simulation) fn collecting(self, collection: OrderCollection) -> Self {
        Self { collection, ..self }
    }
}

impl SearchStage {
    pub(in crate::simulation) fn program(self, config: &Configuration) -> SearchProgram {
        let shipment_speed = config.shipment_speed;
        let suppliers = match self {
            Self::Search
            | Self::SearchGrain
            | Self::SearchIndexed
            | Self::SearchFuel
            | Self::SearchHousehold
            | Self::SearchSpatialFuel => {
                match config.spatial_side == 1 && config.map_size <= config.search_radius {
                    true => SupplierSearch::SingleCell { shipment_speed },
                    false => SupplierSearch::Neighborhood { shipment_speed },
                }
            }
            _ => SupplierSearch::Inactive,
        };
        SearchProgram {
            collection: OrderCollection::Bitmap,
            stage: self,
            suppliers,
        }
    }
}

pub(in crate::simulation) fn empty_bounds_invocations(
    members: usize,
    mode: parallel::ScanMode,
) -> usize {
    let lanes = match mode {
        parallel::ScanMode::Plane => 32,
        parallel::ScanMode::Shared => 1,
    };
    members
        .min(empty::LIMIT as usize)
        .div_ceil(empty::BLOCK as usize)
        * lanes
}

#[derive(CubeType, CubeLaunch)]
pub(in crate::simulation) struct SearchInput {
    pub state: Box<[u32]>,
    pub actors: Box<[u32]>,
    pub config: EconomyConfig,
    pub spatial: Box<[u32]>,
    pub proposals: Box<[u32]>,
    pub searches: Box<[u32]>,
    pub successes: Box<[Atomic<u32>]>,
    pub scratch: Box<[u32]>,
    pub control: Box<[Atomic<u32>]>,
    pub parameters: EconomyParameters,
    #[cube(comptime)]
    pub status_reset: StatusReset,
    #[cube(comptime)]
    pub program: SearchProgram,
}

pub(in crate::simulation) struct MarketSearch;
pub(in crate::simulation) struct Search;

impl System for MarketSearch {
    type Input = SearchInput;
    type Passes = gpu_passes![Search];
}

#[cube]
impl GpuPass<SearchInput> for Search {
    type Output = ();

    fn execute(mut input: SearchInput) {
        match comptime!(input.program.suppliers) {
            SupplierSearch::SingleCell { shipment_speed } => {
                input.config.spatial_side = 1;
                input.config.shipment_speed = shipment_speed;
            }
            SupplierSearch::Neighborhood { shipment_speed } => {
                input.config.shipment_speed = shipment_speed;
            }
            SupplierSearch::Inactive => (),
        }
        input.parameters.refresh(&mut input.config);
        let mut economy = EconomyView::new(&mut input.state, &input.actors, &input.config);
        if input.control[LATCH_WORD].load() == 0 {
            match comptime!(input.program.stage) {
                SearchStage::Index => index::build(
                    &economy,
                    &mut input.spatial,
                    &mut input.scratch,
                    &mut input.searches,
                    &input.successes,
                    &input.control,
                    &input.parameters,
                    comptime!(input.program.collection),
                ),
                SearchStage::EmptySort => empty::sort(
                    &input.config,
                    &input.spatial,
                    &mut input.scratch,
                    &mut input.searches,
                    comptime!(input.parameters.scan),
                ),
                SearchStage::EmptyBounds => empty::bounds(
                    &economy.actors,
                    &input.config,
                    &input.spatial,
                    &mut input.scratch,
                    comptime!(input.parameters.scan),
                ),
                SearchStage::Prepare
                | SearchStage::PrepareCandidates
                | SearchStage::PrepareSelected => {
                    let mut group = CUBE_POS as u32;
                    let groups = match comptime!(input.program.stage) {
                        SearchStage::PrepareSelected => {
                            let clear = ceildiv(input.config.actors, 32)
                                + ceildiv(input.config.actors, markets::COMPACTION_BLOCK);
                            ceildiv(
                                max(
                                    clear,
                                    input.control[markets::PREPARATION_ACTORS_WORD].load(),
                                ),
                                CUBE_DIM,
                            )
                        }
                        _ => CUBE_COUNT as u32,
                    };
                    while group < groups {
                        prepare(
                            &mut economy,
                            &mut input.spatial,
                            &mut input.proposals,
                            &input.control,
                            &mut input.searches,
                            &input.successes,
                            &mut input.scratch,
                            &input.parameters,
                            group * CUBE_DIM + UNIT_POS,
                            comptime!(match input.program.stage {
                                SearchStage::PrepareCandidates => PreparationRound::Capture,
                                SearchStage::PrepareSelected => PreparationRound::Selected,
                                _ => PreparationRound::First,
                            }),
                            comptime!(input.status_reset),
                        );
                        group += CUBE_COUNT as u32;
                        sync_cube();
                    }
                }
                SearchStage::GrainDealerTiles => grain::dealer_tiles(
                    &economy,
                    &input.spatial,
                    &mut input.searches,
                    &input.control,
                    &input.parameters,
                ),
                SearchStage::GrainFirms => grain::collect_firms(
                    &economy,
                    &input.spatial,
                    &mut input.searches,
                    &input.control,
                ),
                SearchStage::Search | SearchStage::SearchGrain | SearchStage::SearchIndexed => {
                    search(
                        &mut economy,
                        &mut input.spatial,
                        &mut input.proposals,
                        &mut input.searches,
                        &input.successes,
                        &input.scratch,
                        &input.control,
                        &input.parameters,
                        comptime!(input.program.suppliers),
                        comptime!(input.program.collection),
                        comptime!(match input.program.stage {
                            SearchStage::SearchIndexed => index::SupplySource::Index,
                            SearchStage::SearchGrain => index::SupplySource::Grain,
                            _ => index::SupplySource::State,
                        }),
                    )
                }
                SearchStage::SearchFuel
                | SearchStage::SearchHousehold
                | SearchStage::SearchSpatialFuel => fuel::search(
                    &mut economy,
                    &input.spatial,
                    &mut input.proposals,
                    &mut input.searches,
                    &input.successes,
                    &input.scratch,
                    &input.control,
                    &input.parameters,
                    comptime!(input.program.suppliers),
                    comptime!(input.program.collection),
                    comptime!(match input.program.stage {
                        SearchStage::SearchSpatialFuel => fuel::SearchOrder::Spatial,
                        _ => fuel::SearchOrder::Compact,
                    }),
                ),
                SearchStage::RoadCandidates => road::prepare(
                    &economy.actors,
                    &input.config,
                    &input.spatial,
                    &input.scratch,
                    &mut input.searches,
                    &input.control,
                ),
                SearchStage::StockLimits => filter::stock_limits(
                    &economy.ledger,
                    &input.spatial,
                    &input.scratch,
                    &input.successes,
                ),
                SearchStage::FilterBuyers => filter::buyers(
                    &economy,
                    &input.proposals,
                    &input.searches,
                    &input.successes,
                    &input.control,
                    &input.parameters,
                ),
                SearchStage::FilterEmpty => filter::orders(
                    &economy,
                    &input.proposals,
                    &input.successes,
                    &input.control,
                    &input.parameters,
                ),
                SearchStage::FinishFiltered => filter::finish(
                    &input.config,
                    &input.spatial,
                    &input.successes,
                    &input.control,
                    &input.parameters,
                    comptime!(input.program.collection),
                ),
                SearchStage::Compact => {
                    if input.control[crate::simulation::flags::FAULT_WORD].load() == 0 {
                        compact(
                            &input.proposals,
                            &input.searches,
                            &input.successes,
                            &mut input.scratch,
                            &input.control,
                            &input.parameters,
                            ABSOLUTE_POS as u32,
                        );
                        if comptime!(matches!(
                            input.parameters.market,
                            MarketFlag::Consumer(_) | MarketFlag::Household
                        )) {
                            let job = ABSOLUTE_POS as u32;
                            if job < input.control[SEARCHES_WORD].load() {
                                let actor = input.searches[job as usize];
                                let peer = input.proposals[actor as usize * ProposalLayout::WORDS
                                    + ProposalLayout::ACTOR];
                                if peer != NO_ACTOR {
                                    let mut status = super::clearing::buyer_settlement(
                                        &mut economy,
                                        &mut input.proposals,
                                        &input.control,
                                        &input.parameters,
                                        actor,
                                        peer,
                                        comptime!(super::scheduler::Posting::CheckActor),
                                    );
                                    if comptime!(
                                        input.parameters.empty_orders
                                            == crate::simulation::parameters::EmptyOrderMode::FeedbackOnly
                                    ) {
                                        let position = (actor + input.parameters.count
                                            - input.parameters.settlement_rotation)
                                            % input.parameters.count;
                                        let kept = input.successes[(position / 32) as usize].load()
                                            & (1u32 << (position % 32));
                                        if kept == 0 {
                                            status.replace(super::clearing::empty_seller_settlement(
                                                &mut economy,
                                                &input.proposals,
                                                &input.control,
                                                &input.parameters,
                                                actor,
                                                peer,
                                                &status,
                                            ));
                                        }
                                    }
                                    status.store(
                                        &mut input.scratch,
                                        4 * input.parameters.count + actor,
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[cube]
fn search(
    economy: &mut EconomyView<'_>,
    spatial: &mut [u32],
    proposals: &mut [u32],
    searches: &mut [u32],
    successes: &[Atomic<u32>],
    scratch: &[u32],
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
    #[comptime] suppliers: SupplierSearch,
    #[comptime] collection: OrderCollection,
    #[comptime] source: index::SupplySource,
) {
    let config = economy.ledger.config;
    let jobs = markets::job_count(control, SEARCHES_WORD);
    let mut job = CUBE_POS as u32;
    while job < jobs {
        let id = searches[job as usize];
        let wanted = ticks(
            searches[(config.actors + id) as usize],
            searches[(2 * config.actors + id) as usize],
        );
        let purchasing_power = ticks(
            searches[(3 * config.actors + id) as usize],
            searches[(4 * config.actors + id) as usize],
        );
        let mut input_target = zero();
        if comptime!(matches!(parameters.market, MarketFlag::Input(_))) {
            let capacity = economy.actors.capacity(id);
            input_target = minimum(
                wanted,
                times32(capacity, economy.ledger.number(Field::InputTicks, id)),
            );
        }
        let mut buyer_access = 0u32;
        if comptime!(config.railway.routes_enabled == Feature::Enabled) && config.railway.nodes > 0
        {
            buyer_access = config.railway.access[id as usize * 2 + 1];
        }
        let px = economy.actors.x(id);
        let py = economy.actors.y(id);
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
        let mut chosen = u32::new(comptime!(i64::from(NO_ACTOR)));
        let mut quantity = zero();
        let mut trip = 0u32;
        let mut best = ticks(u32::MAX, u32::MAX);
        if comptime!(parameters.market == MarketFlag::Labor) {
            best = zero();
        }
        let mut considered = 0u32;
        let mut seen_supply = false;
        let mut empty_seller = u32::new(comptime!(i64::from(NO_ACTOR)));
        let mut empty_quantity = zero();
        let mut empty_trip = 0u32;
        let mut empty_score = ticks(u32::MAX, u32::MAX);
        let lanes = comptime!(match source {
            index::SupplySource::Index => crate::simulation::ops::MAX_WORKGROUP_SIZE as usize,
            _ => 32usize,
        });
        let mut candidates = CandidateTile::new(lanes);
        let candidate_faults =
            Shared::<[Atomic<u32>]>::new_slice(comptime!(match parameters.market {
                MarketFlag::Labor => lanes * CONTROL_WORDS,
                _ => 1usize,
            }));
        let fault_words = comptime!(match parameters.market {
            MarketFlag::Labor => CONTROL_WORDS,
            _ => 1usize,
        });
        let mut visited = Shared::<u32>::new();
        let mut supply = Shared::<u32>::new();
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
            chosen = trade.seller;
            quantity = trade.quantity;
            trip = trade.trip;
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
            chosen = trade.seller;
            quantity = trade.quantity;
            trip = trade.trip;
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
        while y <= min(cy + 1, side - 1) && considered < DISCOVERY_CANDIDATES {
            let mut x = max(cx, 1u32) - 1;
            while x <= min(cx + 1, side - 1) && considered < DISCOVERY_CANDIDATES {
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
                while j < limit && considered < DISCOVERY_CANDIDATES {
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
                        let unavailable = index::skip(
                            scratch,
                            config,
                            block,
                            px,
                            py,
                            buyer_access,
                            purchasing_power,
                            chosen,
                            empty_score,
                            u32::cast_from(
                                seen_supply
                                    || comptime!(parameters.market != MarketFlag::Household),
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
                        && chosen != NO_ACTOR
                        && input_target.extract(HIGH_WORD) == 0
                    {
                        let rotated = begin + cursor;
                        let block = rotated / index::BLOCK;
                        let upper = ticks(index::maximum_available(scratch, config, block), 0);
                        let mut inferior = lt(upper, quantity);
                        if equal(upper, quantity) {
                            let price_floor = index::partial_input_price_floor(
                                scratch,
                                config,
                                block,
                                px,
                                py,
                                buyer_access,
                                maximum_delivery_cost,
                            );
                            inferior = price_floor >= best.extract(LOW_WORD);
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
                        while offset < visibility && !seen_supply {
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
                                    economy.actors.distance(id, seller) <= radius && supply.offered,
                                );
                            }
                            seen_supply =
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
                        let fault_start = lane
                            * comptime!(match parameters.market {
                                MarketFlag::Labor => CONTROL_WORDS,
                                _ => 0usize,
                            });
                        if comptime!(parameters.market == MarketFlag::Labor) {
                            candidate_faults[fault_start + crate::simulation::flags::FAULT_WORD]
                                .store(0);
                        }
                        let mut offer = Candidate {
                            eligible: false,
                            seen_supply: false,
                            empty: false,
                            quantity: zero(),
                            score: zero(),
                            trip: 0,
                        };
                        let mut firm = u32::new(comptime!(i64::from(NO_ACTOR)));
                        if UNIT_POS < width && j + UNIT_POS < limit {
                            let rotated = begin + (cursor + UNIT_POS * stride) % max(length, 1);
                            firm = spatial[(side * side + 1 + rotated) as usize];
                            let supply = index::Supply::for_search(
                                &*economy, scratch, parameters, rotated, firm, source,
                            );
                            let mut relevant = true;
                            if comptime!(
                                source == index::SupplySource::Index
                                    && matches!(parameters.market, MarketFlag::Input(_))
                            ) && chosen != NO_ACTOR
                                && input_target.extract(HIGH_WORD) == 0
                            {
                                relevant = !(lt(supply.available, input_target)
                                    && lt(supply.available, quantity));
                            }
                            let mut trip = 0u32;
                            if relevant {
                                trip = economy.actors.distance(id, firm);
                                if comptime!(matches!(
                                    parameters.market,
                                    MarketFlag::Household | MarketFlag::Consumer(_)
                                )) {
                                    let quote = supply.quote;
                                    let days =
                                        minimum_delivery_days(config, buyer_access, firm, trip);
                                    let lower_score = plus(
                                        ticks(quote, 0),
                                        times32(days, config.delivery_penalty),
                                    )
                                    .value;
                                    relevant = supply.offered
                                        || (chosen == NO_ACTOR
                                            && quote > 0
                                            && lt(lower_score, empty_score));
                                }
                                if comptime!(
                                    source == index::SupplySource::Index
                                        && matches!(parameters.market, MarketFlag::Input(_))
                                ) && chosen != NO_ACTOR
                                    && input_target.extract(HIGH_WORD) == 0
                                    && lt(supply.available, input_target)
                                    && equal(supply.available, quantity)
                                {
                                    let days =
                                        minimum_delivery_days(config, buyer_access, firm, trip);
                                    let lower = plus(
                                        ticks(supply.quote, 0),
                                        times32(days, config.delivery_penalty),
                                    )
                                    .value;
                                    let upper =
                                        plus(ticks(supply.quote, 0), maximum_delivery_cost).value;
                                    let price_floor = select(
                                        upper.extract(HIGH_WORD) == 0,
                                        lower.extract(LOW_WORD),
                                        0,
                                    );
                                    relevant = price_floor < best.extract(LOW_WORD);
                                }
                            }
                            if relevant {
                                offer.replace(match comptime!(parameters.market) {
                                    MarketFlag::Consumer(_) => consumer_candidate(
                                        config,
                                        id,
                                        firm,
                                        trip,
                                        buyer_access,
                                        wanted,
                                        purchasing_power,
                                        &supply,
                                    ),
                                    _ => candidate_supply(
                                        &*economy,
                                        &candidate_faults[fault_start..fault_start + fault_words],
                                        parameters,
                                        id,
                                        firm,
                                        trip,
                                        buyer_access,
                                        wanted,
                                        purchasing_power,
                                        &supply,
                                    ),
                                });
                            }
                        }
                        candidates.store(lane, &offer, firm);
                        sync_cube();
                        let mut cooperative = false;
                        if comptime!(source == index::SupplySource::Index) {
                            cooperative = CUBE_DIM >= 32;
                            let eligible = u32::cast_from(offer.eligible);
                            let score = offer.score;
                            let counted = eligible
                                * u32::cast_from(
                                    comptime!(!matches!(parameters.market, MarketFlag::Input(_)))
                                        || score.extract(HIGH_WORD) == 0,
                                );
                            let improving =
                                eligible * u32::cast_from(chosen == NO_ACTOR || lt(score, best));
                            let prefix = parallel::scan_prefix(
                                counted | (improving << 16),
                                comptime!(parameters.scan),
                            );
                            let remaining = DISCOVERY_CANDIDATES - considered;
                            let included = u32::cast_from(
                                UNIT_POS < width
                                    && j + UNIT_POS < limit
                                    && (prefix.inclusive & 0xffff) - counted < remaining,
                            );
                            if comptime!(parameters.market == MarketFlag::Household) {
                                seen_supply = seen_supply
                                    || parallel::reduce::<u32>(
                                        u32::cast_from(offer.seen_supply) * included,
                                        comptime!(Reduction::Sum),
                                    ) > 0;
                            }
                            if prefix.total >> 16 > 0 {
                                let winner = group_winner(
                                    CandidateRank {
                                        active: improving * included,
                                        score,
                                    },
                                    comptime!(parameters.scan),
                                );
                                if winner != NO_ACTOR {
                                    let slot = winner as usize;
                                    chosen = candidates.firm[slot];
                                    quantity = candidates.quantity[slot];
                                    trip = candidates.trip[slot];
                                    best = candidates.score[slot];
                                }
                            }
                            if chosen == NO_ACTOR {
                                let fallback = group_winner(
                                    CandidateRank {
                                        active: u32::cast_from(offer.empty) * included,
                                        score,
                                    },
                                    comptime!(parameters.scan),
                                );
                                if fallback != NO_ACTOR {
                                    let slot = fallback as usize;
                                    let score = candidates.score[slot];
                                    if lt(score, empty_score) {
                                        empty_seller = candidates.firm[slot];
                                        empty_quantity = candidates.quantity[slot];
                                        empty_trip = candidates.trip[slot];
                                        empty_score = score;
                                    }
                                }
                            }
                            considered += min(prefix.total & 0xffff, remaining);
                        }
                        if comptime!(
                            source != index::SupplySource::Index
                                && parameters.scan == crate::simulation::parallel::ScanMode::Plane
                        ) {
                            cooperative = CUBE_DIM <= PLANE_DIM;
                            if cooperative {
                                let eligible = u32::cast_from(offer.eligible);
                                let counted = eligible
                                    * u32::cast_from(
                                        comptime!(!matches!(
                                            parameters.market,
                                            MarketFlag::Input(_)
                                        )) || offer.score.extract(HIGH_WORD) == 0,
                                    );
                                let preceding = plane_exclusive_sum(counted);
                                let visited = UNIT_POS < width
                                    && j + UNIT_POS < limit
                                    && preceding < DISCOVERY_CANDIDATES - considered;
                                let included = u32::cast_from(visited);
                                let total = plane_sum(counted * included);
                                seen_supply = seen_supply
                                    || plane_sum(u32::cast_from(offer.seen_supply) * included) > 0;
                                if comptime!(parameters.market == MarketFlag::Labor) {
                                    let faults = candidate_faults
                                        [fault_start + crate::simulation::flags::FAULT_WORD]
                                        .load();
                                    if visited && faults != 0 {
                                        control[crate::simulation::flags::FAULT_WORD]
                                            .fetch_or(faults);
                                    }
                                }
                                let score = offer.score;
                                let mut order = score;
                                if comptime!(parameters.market == MarketFlag::Labor) {
                                    order =
                                        ticks(!score.extract(LOW_WORD), !score.extract(HIGH_WORD));
                                }
                                let winner = plane_winner(CandidateRank {
                                    active: eligible * included,
                                    score: order,
                                })
                                .lane;
                                let fallback = plane_winner(CandidateRank {
                                    active: u32::cast_from(offer.empty) * included,
                                    score,
                                })
                                .lane;
                                if winner != NO_ACTOR {
                                    let slot = winner as usize;
                                    let score = candidates.score[slot];
                                    let mut improves = lt(score, best);
                                    if comptime!(parameters.market == MarketFlag::Labor) {
                                        improves = lt(best, score);
                                    }
                                    if improves || chosen == NO_ACTOR {
                                        chosen = candidates.firm[slot];
                                        quantity = candidates.quantity[slot];
                                        trip = candidates.trip[slot];
                                        best = score;
                                    }
                                }
                                if fallback != NO_ACTOR {
                                    let slot = fallback as usize;
                                    let score = candidates.score[slot];
                                    if lt(score, empty_score) {
                                        empty_seller = candidates.firm[slot];
                                        empty_quantity = candidates.quantity[slot];
                                        empty_trip = candidates.trip[slot];
                                        empty_score = score;
                                    }
                                }
                                considered += total;
                            }
                        }
                        if !cooperative {
                            if UNIT_POS == 0 {
                                let mut slot = 0usize;
                                while slot < min(width as usize, (limit - j) as usize)
                                    && considered < DISCOVERY_CANDIDATES
                                {
                                    if comptime!(parameters.market == MarketFlag::Labor) {
                                        let faults = candidate_faults[slot * CONTROL_WORDS
                                            + crate::simulation::flags::FAULT_WORD]
                                            .load();
                                        if faults != 0 {
                                            control[crate::simulation::flags::FAULT_WORD]
                                                .fetch_or(faults);
                                        }
                                    }
                                    seen_supply = seen_supply || candidates.seen_supply[slot] != 0;
                                    let amount = candidates.quantity[slot];
                                    let score = candidates.score[slot];
                                    let distance = candidates.trip[slot];
                                    let firm = candidates.firm[slot];
                                    if candidates.empty[slot] != 0 && lt(score, empty_score) {
                                        empty_seller = firm;
                                        empty_quantity = amount;
                                        empty_trip = distance;
                                        empty_score = score;
                                    }
                                    let eligible = candidates.eligible[slot] != 0;
                                    let mut improves = lt(score, best);
                                    if comptime!(parameters.market == MarketFlag::Labor) {
                                        improves = lt(best, score);
                                    }
                                    if eligible && (improves || chosen == NO_ACTOR) {
                                        chosen = firm;
                                        quantity = amount;
                                        trip = distance;
                                        best = score;
                                    }
                                    considered += u32::cast_from(
                                        eligible
                                            && (comptime!(!matches!(
                                                parameters.market,
                                                MarketFlag::Input(_)
                                            )) || score.extract(HIGH_WORD) == 0),
                                    );
                                    slot += 1;
                                }
                                *visited = considered;
                                *supply = u32::cast_from(seen_supply);
                            }
                            sync_cube();
                            considered = *visited;
                            seen_supply = *supply != 0;
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
            if chosen == NO_ACTOR && empty_seller != NO_ACTOR {
                chosen = empty_seller;
                quantity = empty_quantity;
                trip = empty_trip;
                missing_supply = true;
            }
            if comptime!(matches!(parameters.market, MarketFlag::Consumer(_))) && chosen != NO_ACTOR
            {
                let supply = index::Supply::new(&*economy, chosen, parameters.traded_good());
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
                economy.ledger.set_number(Field::FoodAccess, id, failure);
            }
            proposals[id as usize * ProposalLayout::WORDS + ProposalLayout::ACTOR] = chosen;
            proposals[id as usize * ProposalLayout::WORDS + ProposalLayout::QUANTITY_LOW] =
                quantity.extract(LOW_WORD);
            proposals[id as usize * ProposalLayout::WORDS + ProposalLayout::QUANTITY_HIGH] =
                quantity.extract(HIGH_WORD);
            proposals[id as usize * ProposalLayout::WORDS + ProposalLayout::DISTANCE] = trip;
            if chosen != NO_ACTOR {
                if comptime!(
                    matches!(parameters.market, MarketFlag::Input(_))
                        && config.railway.routes_enabled == Feature::Enabled
                ) && control[INPUT_RAIL_WORD].load() == 0
                    && railway::may_book(config, id, chosen, trip, buyer_access)
                {
                    control[INPUT_RAIL_WORD].store(1);
                }
                if comptime!(collection == OrderCollection::InputTile) {
                    let slot = control[CONSTRUCTED_EVENTS_WORD].fetch_add(1);
                    if slot < comptime!(super::sorting::INPUT_TILE as u32) {
                        searches[(comptime!(INPUT_BUYERS_FIELD) * config.actors + slot) as usize] =
                            id;
                    }
                }
                record_success(successes, parameters, id);
                record_wide_order(control, parameters, quantity);
            }
        }
        sync_cube();
        job += CUBE_COUNT as u32;
    }
}

#[cube]
pub(super) fn record_success(
    successes: &[Atomic<u32>],
    parameters: &EconomyParameters,
    actor: u32,
) {
    let position = (actor + parameters.count - parameters.settlement_rotation) % parameters.count;
    let masks = ceildiv(parameters.count, 32);
    successes[(position / 32) as usize].fetch_or(1u32 << (position % 32));
    successes[(masks + position / markets::COMPACTION_BLOCK) as usize].fetch_add(1);
}

#[cube]
pub(super) fn record_wide_order(
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
    quantity: Ticks,
) {
    if comptime!(
        parameters.empty_orders == crate::simulation::parameters::EmptyOrderMode::FeedbackOnly
    ) && (quantity.extract(HIGH_WORD) > 0 || equal(quantity, zero()))
    {
        control[markets::WIDE_ORDERS_WORD].fetch_or(select(
            quantity.extract(HIGH_WORD) > 0,
            1u32,
            2u32,
        ));
    }
}

#[cube]
pub(super) fn compact(
    proposals: &[u32],
    searches: &[u32],
    successes: &[Atomic<u32>],
    scratch: &mut [u32],
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
    job: u32,
) {
    let count = parameters.count;
    let masks = ceildiv(count, 32);
    let blocks = ceildiv(count, markets::COMPACTION_BLOCK);
    if job == 0 {
        control[CONSTRUCTED_EVENTS_WORD].store(successes[(masks + blocks - 1) as usize].load());
    }
    if job < control[SEARCHES_WORD].load() {
        let actor = searches[job as usize];
        let peer = proposals[actor as usize * ProposalLayout::WORDS + ProposalLayout::ACTOR];
        if peer != NO_ACTOR {
            KernelEventStatus::new(comptime!(EventStatus::Inactive))
                .store(scratch, 4 * count + actor);
            let position = (actor + count - parameters.settlement_rotation) % count;
            let block = position / markets::COMPACTION_BLOCK;
            let word = position / 32;
            let kept = successes[word as usize].load();
            let bit = 1u32 << (position % 32);
            if kept & bit != 0 {
                let mut slot = 0u32;
                if block > 0 {
                    slot = successes[(masks + block - 1) as usize].load();
                }
                for preceding in block * (markets::COMPACTION_BLOCK / 32)..word {
                    slot += successes[preceding as usize].load().count_ones();
                }
                slot += (kept & (bit - 1)).count_ones();
                scratch[slot as usize] = peer;
                scratch[(count + slot) as usize] = actor;
            }
        }
    }
}

#[derive(CubeType, Clone, Copy)]
struct CandidateRank {
    active: u32,
    score: Ticks,
}

#[derive(CubeType)]
struct CandidateTree {
    scores: Shared<[u64]>,
    lanes: Shared<[u32]>,
}

#[cube]
impl workgroup::TreeStorage for CandidateTree {
    fn merge(&mut self, destination: usize, source: usize) {
        let preferred = self.scores[source] < self.scores[destination]
            || (self.scores[source] == self.scores[destination]
                && self.lanes[source] < self.lanes[destination]);
        if preferred {
            self.scores[destination] = self.scores[source];
            self.lanes[destination] = self.lanes[source];
        }
    }
}

#[cube]
fn group_winner(rank: CandidateRank, #[comptime] mode: parallel::ScanMode) -> u32 {
    let mut winner = u32::new(comptime!(i64::from(NO_ACTOR)));
    if comptime!(mode == parallel::ScanMode::Plane) {
        let local = plane_winner(rank);
        if CUBE_DIM <= PLANE_DIM {
            winner = local.lane;
        } else {
            let mut scores = WorkgroupTree::planes::<u64>();
            let mut lanes = WorkgroupTree::planes::<u32>();
            if UNIT_POS_PLANE == 0 {
                scores[PLANE_POS as usize] = local.score;
                lanes[PLANE_POS as usize] = local.lane;
            }
            sync_cube();
            let mut best = u64::new(-1);
            for plane in 0..ceildiv(CUBE_DIM, PLANE_DIM) {
                let score = scores[plane as usize];
                let lane = lanes[plane as usize];
                if score < best || score == best && lane < winner {
                    best = score;
                    winner = lane;
                }
            }
            sync_cube();
        }
    } else {
        let score = u64::cast_from(rank.score.extract(LOW_WORD))
            | (u64::cast_from(rank.score.extract(HIGH_WORD)) << 32);
        let mut tree = CandidateTree {
            scores: WorkgroupTree::lanes::<u64>(select(rank.active > 0, score, u64::MAX)),
            lanes: WorkgroupTree::lanes::<u32>(select(rank.active > 0, UNIT_POS, NO_ACTOR)),
        };
        WorkgroupTree::reduce::<CandidateTree>(&mut tree);
        winner = tree.lanes[0usize];
        sync_cube();
    }
    winner
}

#[derive(CubeType)]
struct RankedLane {
    lane: u32,
    score: u64,
}

#[cube]
fn plane_winner(rank: CandidateRank) -> RankedLane {
    let high = select(rank.active > 0, rank.score.extract(HIGH_WORD), u32::MAX);
    let minimum_high = plane_min(high);
    let low = select(
        rank.active > 0 && high == minimum_high,
        rank.score.extract(LOW_WORD),
        u32::MAX,
    );
    let minimum_low = plane_min(low);
    RankedLane {
        lane: plane_min(select(
            rank.active > 0 && high == minimum_high && low == minimum_low,
            UNIT_POS,
            NO_ACTOR,
        )),
        score: u64::cast_from(minimum_low) | (u64::cast_from(minimum_high) << 32),
    }
}

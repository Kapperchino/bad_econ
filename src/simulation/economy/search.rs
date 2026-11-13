use super::*;
use crate::simulation::markets::{LATCH_WORD, SEARCHES_WORD};
mod empty;
mod fuel;
mod index;

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
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
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
        if comptime!(market == MarketFlag::Consumer && round == PreparationRound::Capture) {
            searches[(14 * config.actors + id) as usize] = 0;
        }
        if comptime!(market == MarketFlag::Household && parameters.discovery.indexed()) {
            let cells = config.spatial_side * config.spatial_side;
            let base =
                3 * (cells + 1 + config.actors) + ceildiv(config.actors, DISCOVERY_BLOCK) + 1;
            id = spatial[(base + invocation) as usize];
        }
        if comptime!(market == MarketFlag::Consumer)
            && (parameters.traded_good() == comptime!(Good::Coal as u32)
                || parameters.traded_good() == comptime!(Good::Wood as u32))
        {
            searches[(config.actors + id) as usize] = 0;
            searches[(2 * config.actors + id) as usize] = 0;
        }
        let k = kind(actors, id);
        let mut chosen = u32::new(comptime!(i64::from(NO_ACTOR)));
        let mut quantity = ticks(0u32, 0u32);
        let trip = 0u32;
        let mut search = false;
        let mut good = 0u32;
        let mut wanted = ticks(0u32, 0u32);
        let seen_supply = false;
        let missing_supply = false;
        let household_demand = match comptime!(parameters.market) {
            MarketFlag::Input(_) => {
                let grain = parameters.traded_good() == Good::Grain as u32
                    && comptime!(config.home_mode == Feature::Enabled)
                    && (comptime!(config.economy == crate::domain::Economy::Agrarian)
                        || number(state, config, comptime!(Field::LandPlots as u32), id) > 0)
                    && number(state, config, comptime!(Field::LivingMembers as u32), id) > 0;
                let investment = (parameters.traded_good() == Good::Tools as u32
                    || parameters.traded_good() == Good::Wood as u32)
                    && comptime!(
                        config.home_mode == Feature::Enabled
                            && config.property_mode == Feature::Disabled
                    )
                    && number(state, config, comptime!(Field::InvestmentGood as u32), id) > 0
                    && number(state, config, comptime!(Field::LivingMembers as u32), id) > 0;
                let construction = parameters.traded_good() == Good::Construction as u32
                    && number(state, config, comptime!(Field::ProjectStatus as u32), id)
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
                    && number(state, config, comptime!(Field::EnterpriseModel as u32), id) == 0,
                comptime!(Preparation::Inventory as u32),
                comptime!(Preparation::Demand as u32),
            ),
        };
        if comptime!(round == PreparationRound::Capture) {
            eligible = preparation == comptime!(Preparation::Demand as u32);
        }
        match preparation {
            0 => {
                let mut purchasing_power = value(state, config, comptime!(Field::Cash as u32), id);
                if comptime!(matches!(market, MarketFlag::Input(_))) {
                    purchasing_power =
                        enterprises::purchasing_power(state, actors, config, control, id);
                }
                match market {
                    MarketFlag::Labor => {
                        search = k.is(comptime!(ActorKind::Household))
                            && number(state, config, comptime!(Field::Employed as u32), id) == 0u32
                            && (comptime!(config.economy == crate::domain::Economy::Agrarian)
                                || number(state, config, comptime!(Field::LandPlots as u32), id)
                                    == 0);
                    }
                    MarketFlag::Input(_) => {
                        good = parameters.traded_good();
                        search = !k.is(comptime!(ActorKind::Household))
                            && (number(state, config, comptime!(Field::FirstInput as u32), id)
                                == good
                                || number(state, config, comptime!(Field::SecondInput as u32), id)
                                    == good);
                        wanted = ticks(
                            actors[ActorLayout::CAPACITY * (actors.len() / ActorLayout::WORDS)
                                + id as usize]
                                * number(state, config, comptime!(Field::InputTicks as u32), id)
                                * config.coverage,
                            0u32,
                        );
                        if number(state, config, comptime!(Field::EnterpriseModel as u32), id) == 1
                        {
                            search = good == k.output();
                            let forecast = maximum(
                                value(state, config, comptime!(Field::Demand as u32), id),
                                ticks(1, 0),
                            );
                            wanted = multiply(
                                control,
                                forecast,
                                number(state, config, comptime!(Field::StockCoverage as u32), id)
                                    + config.ring,
                            );
                        }
                        if !k.is(comptime!(ActorKind::Household))
                            && number(state, config, comptime!(Field::EnterpriseModel as u32), id)
                                == 0
                            && comptime!(config.home_mode == Feature::Enabled)
                        {
                            let horizon = config.coverage + config.ring;
                            let packets = divide(
                                add_checked(
                                    control,
                                    multiply(
                                        control,
                                        value(state, config, comptime!(Field::Demand as u32), id),
                                        horizon,
                                    ),
                                    ticks(
                                        number(
                                            state,
                                            config,
                                            comptime!(Field::OutputTicks as u32),
                                            id,
                                        ) - 1,
                                        0,
                                    ),
                                ),
                                max(
                                    number(state, config, comptime!(Field::OutputTicks as u32), id),
                                    1,
                                ),
                            );
                            let capacity = actors[ActorLayout::CAPACITY
                                * (actors.len() / ActorLayout::WORDS)
                                + id as usize];
                            wanted = multiply(
                                control,
                                minimum(packets, times32(capacity, horizon)),
                                number(state, config, comptime!(Field::InputTicks as u32), id),
                            );
                        }
                        if k.is(comptime!(ActorKind::Household))
                            && comptime!(config.home_mode == Feature::Enabled)
                            && good == comptime!(Good::Grain as u32)
                            && (comptime!(config.economy == crate::domain::Economy::Agrarian)
                                || number(state, config, comptime!(Field::LandPlots as u32), id)
                                    > 0)
                            && number(state, config, comptime!(Field::LivingMembers as u32), id) > 0
                        {
                            search = true;
                            wanted = ticks(
                                number(state, config, comptime!(Field::FoodNeed as u32), id)
                                    * (config.coverage + config.ring),
                                0,
                            );
                        }
                        let investment =
                            number(state, config, comptime!(Field::InvestmentGood as u32), id) > 0;
                        if k.is(comptime!(ActorKind::Household))
                            && investment
                            && comptime!(config.property_mode == Feature::Disabled)
                            && number(state, config, comptime!(Field::LivingMembers as u32), id) > 0
                            && comptime!(config.home_mode == Feature::Enabled)
                            && (good == comptime!(Good::Tools as u32)
                                || good == comptime!(Good::Wood as u32))
                        {
                            search = true;
                            wanted =
                                ticks(select(good == comptime!(Good::Tools as u32), 2u32, 5u32), 0);
                        }
                        if !k.is(comptime!(ActorKind::Household))
                            && comptime!(
                                config.home_mode == Feature::Enabled
                                    && config.property_mode == Feature::Disabled
                            )
                            && good == comptime!(Good::Wood as u32)
                            && number(state, config, comptime!(Field::Slots as u32), id) == 0
                            && number(state, config, comptime!(Field::Plan as u32), id) > 0
                            && !lt(
                                value(state, config, comptime!(Field::Cash as u32), id),
                                ticks(10_000, 0),
                            )
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
                            let building =
                                number(state, config, comptime!(Field::ProjectStatus as u32), id)
                                    == comptime!(
                                        crate::domain::property::ProjectStatus::Building as u32
                                    );
                            let work = value(
                                state,
                                config,
                                comptime!(Field::ConstructionRemaining as u32),
                                id,
                            );
                            wanted = select(
                                building,
                                select(search, maximum(wanted, work), work),
                                wanted,
                            );
                            search = search || building;
                            if k.bits() == comptime!(Good::Housing as u32) {
                                wanted = add_checked(
                                    control,
                                    wanted,
                                    landlords::maintenance(state, config, id),
                                );
                                search = true;
                            }
                        }
                    }
                    MarketFlag::Consumer => {
                        good = parameters.traded_good();
                        if good == comptime!(Good::Coal as u32)
                            || good == comptime!(Good::Wood as u32)
                        {
                            wanted = consumption::fuel_target(state, config, control, id, good);
                            search = k.is(comptime!(ActorKind::Household))
                                && number(
                                    state,
                                    config,
                                    comptime!(Field::LivingMembers as u32),
                                    id,
                                ) > 0
                                && !lt(
                                    value(state, config, comptime!(Field::Bread as u32), id),
                                    ticks(
                                        number(
                                            state,
                                            config,
                                            comptime!(Field::FoodNeed as u32),
                                            id,
                                        ),
                                        0,
                                    ),
                                );
                        } else {
                            let interval = config.consumption_intervals[good as usize - 1];
                            let until_due = (interval - (parameters.day + id) % max(interval, 1))
                                % max(interval, 1);
                            search = k.is(comptime!(ActorKind::Household))
                                && number(
                                    state,
                                    config,
                                    comptime!(Field::LivingMembers as u32),
                                    id,
                                ) > 0
                                && until_due <= config.coverage + config.ring;
                            let reserve = ticks(
                                number(state, config, comptime!(Field::FoodNeed as u32), id)
                                    * (config.coverage + config.ring - 1),
                                0,
                            );
                            let food = add_checked(
                                control,
                                value(state, config, comptime!(Field::Bread as u32), id),
                                value(
                                    state,
                                    config,
                                    TRANSIT_FIELDS as u32 + comptime!(Good::Bread as u32) - 1,
                                    id,
                                ),
                            );
                            let cash_reserve = multiply(
                                control,
                                reserve,
                                number(state, config, comptime!(Field::Price as u32), id),
                            );
                            search = search
                                && !lt(food, reserve)
                                && lt(
                                    cash_reserve,
                                    value(state, config, comptime!(Field::Cash as u32), id),
                                );
                            purchasing_power =
                                consumption::budget(state, config, control, id, good);
                            search = search && !equal(purchasing_power, zero());
                            wanted = ticks(1, 0);
                        }
                    }
                    MarketFlag::Household => {
                        good = comptime!(Good::Bread as u32);
                        search = k.is(comptime!(ActorKind::Household));
                        wanted = ticks(
                            number(state, config, comptime!(Field::FoodNeed as u32), id)
                                * (config.coverage + config.ring - 1u32),
                            0u32,
                        );
                    }
                    MarketFlag::Payroll => {
                        quantity = value(state, config, comptime!(Field::Due as u32), id);
                        if !equal(quantity, ticks(0u32, 0u32)) {
                            chosen =
                                number(state, config, comptime!(Field::Employer as u32), id) - 1u32;
                        }
                    }
                    MarketFlag::Distribution => {
                        if !k.is(comptime!(ActorKind::Household)) {
                            quantity = scaled(
                                control,
                                positive(
                                    value(state, config, comptime!(Field::Cash as u32), id),
                                    maximum(
                                        select(
                                            k.bits() == comptime!(Good::Housing as u32),
                                            zero(),
                                            ticks(config.reserve_low, config.reserve_high),
                                        ),
                                        value(
                                            state,
                                            config,
                                            comptime!(Field::OperatingReserve as u32),
                                            id,
                                        ),
                                    ),
                                ),
                                config.distribution_rate,
                                POLICY_SCALE,
                            );
                            if !equal(quantity, ticks(0u32, 0u32)) {
                                chosen = actors[ActorLayout::OWNER
                                    * (actors.len() / ActorLayout::WORDS)
                                    + id as usize];
                            }
                        }
                    }
                }
                if comptime!(matches!(market, MarketFlag::Input(_))) {
                    let rail = railway::demand(state, actors, config, id, good);
                    wanted = maximum(wanted, rail);
                    search = search || !equal(rail, zero());
                }
                if good > 0u32 {
                    wanted = positive(
                        wanted,
                        add_checked(
                            control,
                            value(state, config, good, id),
                            value(state, config, TRANSIT_FIELDS as u32 + good - 1, id),
                        ),
                    );
                }
                search = search
                    && number(state, config, comptime!(Field::Time as u32), id)
                        >= select(comptime!(market == MarketFlag::Labor), 1u32, 0);
                if good > 0 {
                    search = search && !equal(wanted, zero()) && !equal(purchasing_power, zero());
                }
                searching = search;
                if comptime!(round == PreparationRound::Capture && market == MarketFlag::Consumer) {
                    eligible = searching;
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
                    let cash = value(state, config, comptime!(Field::Cash as u32), id);
                    let mut failure = u32::cast_from(missing_supply);
                    if !equal(wanted, zero()) && chosen == NO_ACTOR {
                        failure = select(seen_supply || equal(cash, zero()), 2u32, 1u32);
                    }
                    set32(
                        state,
                        config,
                        comptime!(Field::FoodAccess as u32),
                        id,
                        failure,
                    );
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
                        value(state, config, inactive_good, id),
                        value(state, config, TRANSIT_FIELDS as u32 + inactive_good - 1, id),
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
    actors: &[u32],
    parameters: &EconomyParameters,
    buyer: u32,
    length: u32,
) -> u32 {
    let mut stride = 1u32;
    let sampled =
        comptime!(parameters.market == MarketFlag::Household) && length > HOUSEHOLD_SEARCH_SAMPLES;
    let industrial = comptime!(matches!(parameters.market, MarketFlag::Input(_)))
        && !kind(actors, buyer).is(comptime!(ActorKind::Household));
    if (sampled || industrial) && length > 1 {
        stride = max(
            length / 2 + (buyer + parameters.settlement_rotation) % max(length / 2, 1),
            1,
        );
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
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    parameters: &EconomyParameters,
    buyer: u32,
    wanted: Ticks,
    amount: Ticks,
    price: Ticks,
) -> Ticks {
    let mut score = price;
    if comptime!(matches!(parameters.market, MarketFlag::Input(_)))
        && !kind(actors, buyer).is(comptime!(ActorKind::Household))
    {
        let capacity =
            actors[ActorLayout::CAPACITY * (actors.len() / ActorLayout::WORDS) + buyer as usize];
        let daily = times32(
            capacity,
            number(state, config, comptime!(Field::InputTicks as u32), buyer),
        );
        let missing = positive(minimum(wanted, daily), amount);
        score = ticks(price.extract(LOW_WORD), missing.extract(LOW_WORD));
    }
    score
}

#[cube]
pub(super) fn candidate(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
    id: u32,
    firm: u32,
    wanted: Ticks,
    purchasing_power: Ticks,
) -> Candidate {
    let supply = index::Supply::new(state, actors, config, firm, parameters.traded_good());
    let trip = distance(actors, id, firm);
    let mut buyer_access = 0u32;
    if comptime!(config.railway.routes_enabled == Feature::Enabled) && config.railway.nodes > 0 {
        buyer_access = config.railway.access[id as usize * 2 + 1];
    }
    candidate_supply(
        state,
        actors,
        config,
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
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
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
    let market = comptime!(parameters.market);
    let mut eligible = false;
    let mut seen_supply = false;
    let mut empty_seller = u32::new(comptime!(i64::from(NO_ACTOR)));
    let mut empty_quantity = zero();
    let empty_score = ticks(u32::MAX, u32::MAX);
    let mut score = zero();
    let mut amount = zero();
    if dist <= config.search_radius {
        let mut wage = number(state, config, comptime!(Field::Wage as u32), firm);
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
            score = ticks(quote + days * config.delivery_penalty, 0u32);
            amount = ticks(1u32, 0u32);
            if comptime!(market == MarketFlag::Labor) {
                if comptime!(config.bargaining.enabled == Feature::Enabled) {
                    let negotiation = negotiate(
                        NegotiationRequest {
                            prior_gross: value(
                                state,
                                config,
                                comptime!(Field::GrossIncome as u32),
                                id,
                            ),
                            household_cash: value(state, config, comptime!(Field::Cash as u32), id),
                            funded_per_packet: divide(
                                value(state, config, comptime!(Field::Cash as u32), firm),
                                max(
                                    number(state, config, comptime!(Field::Slots as u32), firm),
                                    1,
                                ),
                            ),
                            candidate_wage: wage,
                            payable_minutes: number(
                                state,
                                config,
                                comptime!(Field::LaborMinutes as u32),
                                firm,
                            ),
                        },
                        config,
                        control,
                    );
                    eligible = eligible
                        && negotiation.status == comptime!(NegotiationStatus::Agreed as u32);
                    wage = negotiation.wage.extract(LOW_WORD);
                    amount = negotiation.wage;
                }
                let duration = number(state, config, comptime!(Field::LaborMinutes as u32), firm)
                    + ceildiv(dist, config.commute_speed) * COMMUTE_LEGS;
                eligible = eligible
                    && number(state, config, comptime!(Field::Slots as u32), firm) > 0u32
                    && number(state, config, comptime!(Field::Time as u32), id) >= duration;
                score = divide(
                    times32(
                        wage * number(state, config, comptime!(Field::LaborMinutes as u32), firm),
                        POLICY_SCALE,
                    ),
                    duration,
                );
                if comptime!(config.home_mode == Feature::Enabled) {
                    let conditions = 500
                        + number(
                            state,
                            config,
                            comptime!(Field::EquipmentCondition as u32),
                            firm,
                        ) / 2
                        + number(
                            state,
                            config,
                            comptime!(Field::SafetyProvision as u32),
                            firm,
                        ) / 4;
                    score = scaled(control, score, conditions, 1000);
                    let need = number(state, config, comptime!(Field::FoodNeed as u32), id);
                    let bread = number(state, config, comptime!(Field::Bread as u32), id);
                    let missing = max(need, bread) - bread;
                    let minutes =
                        missing * (config.home.milling_minutes + config.home.baking_minutes);
                    let time = number(state, config, comptime!(Field::Time as u32), id);
                    let remaining = max(time, duration) - duration
                        + number(state, config, comptime!(Field::HomeTime as u32), id);
                    let self_provision = missing > 0
                        && !lt(
                            value(state, config, comptime!(Field::Grain as u32), id),
                            ticks(missing, 0),
                        );
                    let paid = times32(
                        wage,
                        number(state, config, comptime!(Field::LaborMinutes as u32), firm),
                    );
                    let food_cost = times32(
                        missing,
                        number(state, config, comptime!(Field::Price as u32), id),
                    );
                    eligible =
                        eligible && !(self_provision && remaining < minutes && lt(paid, food_cost));
                }
            } else {
                let affordable = minimum(wanted, divide(purchasing_power, max(quote, 1u32)));
                let offered = supply.offered;
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
                score = supply_score(state, actors, config, parameters, id, wanted, amount, score);
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
            score = ticks(quote + days * config.delivery_penalty, 0);
            let usable = firm != id && quote > 0 && !equal(wanted, zero());
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
    SearchFuel,
    SearchHousehold,
    SearchSpatialFuel,
    Compact,
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
    pub stage: SearchStage,
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
        input.parameters.refresh(&mut input.config);
        if input.control[LATCH_WORD].load() == 0 {
            match comptime!(input.stage) {
                SearchStage::Index => index::build(
                    &mut input.state,
                    &input.actors,
                    &input.config,
                    &mut input.spatial,
                    &mut input.scratch,
                    &input.parameters,
                ),
                SearchStage::EmptySort => empty::sort(
                    &input.config,
                    &input.spatial,
                    &mut input.scratch,
                    &mut input.searches,
                ),
                SearchStage::EmptyBounds => empty::bounds(
                    &input.actors,
                    &input.config,
                    &input.spatial,
                    &mut input.scratch,
                ),
                SearchStage::Prepare
                | SearchStage::PrepareCandidates
                | SearchStage::PrepareSelected => prepare(
                    &mut input.state,
                    &input.actors,
                    &input.config,
                    &mut input.spatial,
                    &mut input.proposals,
                    &input.control,
                    &mut input.searches,
                    &input.successes,
                    &mut input.scratch,
                    &input.parameters,
                    ABSOLUTE_POS as u32,
                    comptime!(match input.stage {
                        SearchStage::PrepareCandidates => PreparationRound::Capture,
                        SearchStage::PrepareSelected => PreparationRound::Selected,
                        _ => PreparationRound::First,
                    }),
                    comptime!(input.status_reset),
                ),
                SearchStage::Search => search(
                    &mut input.state,
                    &input.actors,
                    &input.config,
                    &mut input.spatial,
                    &mut input.proposals,
                    &input.searches,
                    &input.successes,
                    &input.scratch,
                    &input.control,
                    &input.parameters,
                ),
                SearchStage::SearchFuel
                | SearchStage::SearchHousehold
                | SearchStage::SearchSpatialFuel => fuel::search(
                    &mut input.state,
                    &input.actors,
                    &input.config,
                    &input.spatial,
                    &mut input.proposals,
                    &mut input.searches,
                    &input.successes,
                    &input.scratch,
                    &input.control,
                    &input.parameters,
                    comptime!(match input.stage {
                        SearchStage::SearchSpatialFuel => fuel::SearchOrder::Spatial,
                        _ => fuel::SearchOrder::Compact,
                    }),
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
                            MarketFlag::Consumer | MarketFlag::Household
                        )) {
                            let job = ABSOLUTE_POS as u32;
                            if job < input.control[SEARCHES_WORD].load() {
                                let actor = input.searches[job as usize];
                                let peer = input.proposals[actor as usize * ProposalLayout::WORDS
                                    + ProposalLayout::ACTOR];
                                if peer != NO_ACTOR {
                                    super::clearing::buyer_settlement(
                                        &mut input.state,
                                        &input.actors,
                                        &input.config,
                                        &mut input.proposals,
                                        &input.control,
                                        &input.parameters,
                                        actor,
                                        peer,
                                        comptime!(super::scheduler::Posting::CheckActor),
                                    )
                                    .store(&mut input.scratch, 4 * input.parameters.count + actor);
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
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    spatial: &mut [u32],
    proposals: &mut [u32],
    searches: &[u32],
    successes: &[Atomic<u32>],
    scratch: &[u32],
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
) {
    let job = CUBE_POS as u32;
    if job < control[SEARCHES_WORD].load() && control[LATCH_WORD].load() == 0 {
        let id = searches[job as usize];
        let wanted = ticks(
            searches[(config.actors + id) as usize],
            searches[(2 * config.actors + id) as usize],
        );
        let purchasing_power = ticks(
            searches[(3 * config.actors + id) as usize],
            searches[(4 * config.actors + id) as usize],
        );
        let mut buyer_access = 0u32;
        if comptime!(config.railway.routes_enabled == Feature::Enabled) && config.railway.nodes > 0
        {
            buyer_access = config.railway.access[id as usize * 2 + 1];
        }
        let px = actors[ActorLayout::X * (actors.len() / ActorLayout::WORDS) + id as usize];
        let py = actors[ActorLayout::Y * (actors.len() / ActorLayout::WORDS) + id as usize];
        let radius = config.search_radius;
        let cx = px / radius;
        let cy = py / radius;
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
        let mut candidates = Shared::<[u32]>::new_slice(32usize * 9);
        let candidate_faults = Shared::<[Atomic<u32>]>::new_slice(32usize * CONTROL_WORDS);
        let mut visited = Shared::<u32>::new();
        let mut supply = Shared::<u32>::new();
        let mut y = max(cy, 1u32) - 1;
        while y <= min(cy + 1, side - 1) && considered < DISCOVERY_CANDIDATES {
            let mut x = max(cx, 1u32) - 1;
            while x <= min(cx + 1, side - 1) && considered < DISCOVERY_CANDIDATES {
                let cell = y * side + x;
                let begin = spatial[cell as usize];
                let end = spatial[(cell + 1) as usize];
                let length = end - begin;
                let limit = begin + supplier_samples(parameters, length);
                let stride = supplier_stride(actors, parameters, id, length);
                let mut cursor = (parameters.settlement_rotation + id) % max(length, 1);
                let mut j = begin;
                let nearest = max(px, x * radius) - min(px, (x + 1) * radius - 1)
                    + max(py, y * radius)
                    - min(py, (y + 1) * radius - 1);
                if nearest > radius {
                    j = end;
                }
                while j < limit && considered < DISCOVERY_CANDIDATES {
                    let previous = j;
                    let mut skipped = 0u32;
                    if comptime!(matches!(
                        parameters.market,
                        MarketFlag::Consumer | MarketFlag::Household
                    )) && stride == 1
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
                                let supply = match comptime!(parameters.market) {
                                    MarketFlag::Consumer | MarketFlag::Household => {
                                        index::Supply::load(
                                            scratch,
                                            config,
                                            rotated + offset + UNIT_POS,
                                        )
                                    }
                                    _ => index::Supply::new(
                                        state,
                                        actors,
                                        config,
                                        seller,
                                        comptime!(Good::Bread as u32),
                                    ),
                                };
                                offered = u32::cast_from(
                                    distance(actors, id, seller) <= radius && supply.offered,
                                );
                            }
                            seen_supply = parallel::reduce(offered, comptime!(Reduction::Sum)) > 0;
                            offset += 32;
                        }
                    }
                    if skipped == 0 {
                        let initial_width = select(
                            comptime!(matches!(parameters.market, MarketFlag::Input(_)))
                                && parameters.traded_good() == Good::Grain as u32,
                            32u32,
                            8u32,
                        );
                        let width = select(j == begin, initial_width, 32u32);
                        let lane = UNIT_POS as usize;
                        let fault_start = lane * CONTROL_WORDS;
                        if comptime!(parameters.market == MarketFlag::Labor) {
                            candidate_faults[fault_start + crate::simulation::flags::FAULT_WORD]
                                .store(0);
                        }
                        candidates[lane * 9] = 0;
                        candidates[lane * 9 + 1] = 0;
                        candidates[lane * 9 + 2] = 0;
                        if UNIT_POS < width && j + UNIT_POS < limit {
                            let rotated = begin + (cursor + UNIT_POS * stride) % max(length, 1);
                            let firm = spatial[(side * side + 1 + rotated) as usize];
                            let supply = match comptime!(parameters.market) {
                                MarketFlag::Consumer | MarketFlag::Household => {
                                    index::Supply::load(scratch, config, rotated)
                                }
                                _ => index::Supply::new(
                                    state,
                                    actors,
                                    config,
                                    firm,
                                    parameters.traded_good(),
                                ),
                            };
                            let trip = distance(actors, id, firm);
                            let mut relevant = true;
                            if comptime!(matches!(
                                parameters.market,
                                MarketFlag::Household | MarketFlag::Consumer
                            )) {
                                let quote = supply.quote;
                                let days = minimum_delivery_days(config, buyer_access, firm, trip);
                                let lower_score =
                                    plus(ticks(quote, 0), times32(days, config.delivery_penalty))
                                        .value;
                                relevant = supply.offered
                                    || (chosen == NO_ACTOR
                                        && quote > 0
                                        && lt(lower_score, empty_score));
                            }
                            if relevant {
                                let offer = match comptime!(parameters.market) {
                                    MarketFlag::Consumer => consumer_candidate(
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
                                        state,
                                        actors,
                                        config,
                                        &candidate_faults[fault_start..fault_start + CONTROL_WORDS],
                                        parameters,
                                        id,
                                        firm,
                                        trip,
                                        buyer_access,
                                        wanted,
                                        purchasing_power,
                                        &supply,
                                    ),
                                };
                                candidates[lane * 9] = u32::cast_from(offer.eligible);
                                candidates[lane * 9 + 1] = u32::cast_from(offer.seen_supply);
                                candidates[lane * 9 + 2] = u32::cast_from(offer.empty);
                                candidates[lane * 9 + 3] = offer.quantity.extract(LOW_WORD);
                                candidates[lane * 9 + 4] = offer.quantity.extract(HIGH_WORD);
                                candidates[lane * 9 + 5] = offer.score.extract(LOW_WORD);
                                candidates[lane * 9 + 6] = offer.score.extract(HIGH_WORD);
                                candidates[lane * 9 + 7] = offer.trip;
                                candidates[lane * 9 + 8] = firm;
                            }
                        }
                        sync_cube();
                        let mut cooperative = false;
                        if comptime!(
                            parameters.scan == crate::simulation::parallel::ScanMode::Plane
                        ) {
                            cooperative = PLANE_DIM == 32;
                            if cooperative {
                                let eligible = candidates[lane * 9];
                                let counted = eligible
                                    * u32::cast_from(
                                        comptime!(!matches!(
                                            parameters.market,
                                            MarketFlag::Input(_)
                                        )) || candidates[lane * 9 + 6] == 0,
                                    );
                                let preceding = plane_exclusive_sum(counted);
                                let visited = UNIT_POS < width
                                    && j + UNIT_POS < limit
                                    && preceding < DISCOVERY_CANDIDATES - considered;
                                let included = u32::cast_from(visited);
                                let total = plane_sum(counted * included);
                                seen_supply = seen_supply
                                    || plane_sum(candidates[lane * 9 + 1] * included) > 0;
                                if comptime!(parameters.market == MarketFlag::Labor) {
                                    let faults = candidate_faults
                                        [fault_start + crate::simulation::flags::FAULT_WORD]
                                        .load();
                                    if visited && faults != 0 {
                                        control[crate::simulation::flags::FAULT_WORD]
                                            .fetch_or(faults);
                                    }
                                }
                                let score =
                                    ticks(candidates[lane * 9 + 5], candidates[lane * 9 + 6]);
                                let mut order = score;
                                if comptime!(parameters.market == MarketFlag::Labor) {
                                    order =
                                        ticks(!score.extract(LOW_WORD), !score.extract(HIGH_WORD));
                                }
                                let winner = plane_winner(CandidateRank {
                                    active: eligible * included,
                                    score: order,
                                });
                                let fallback = plane_winner(CandidateRank {
                                    active: candidates[lane * 9 + 2] * included,
                                    score,
                                });
                                if winner != NO_ACTOR {
                                    let slot = winner as usize;
                                    let score =
                                        ticks(candidates[slot * 9 + 5], candidates[slot * 9 + 6]);
                                    let mut improves = lt(score, best);
                                    if comptime!(parameters.market == MarketFlag::Labor) {
                                        improves = lt(best, score);
                                    }
                                    if improves || chosen == NO_ACTOR {
                                        chosen = candidates[slot * 9 + 8];
                                        quantity = ticks(
                                            candidates[slot * 9 + 3],
                                            candidates[slot * 9 + 4],
                                        );
                                        trip = candidates[slot * 9 + 7];
                                        best = score;
                                    }
                                }
                                if fallback != NO_ACTOR {
                                    let slot = fallback as usize;
                                    let score =
                                        ticks(candidates[slot * 9 + 5], candidates[slot * 9 + 6]);
                                    if lt(score, empty_score) {
                                        empty_seller = candidates[slot * 9 + 8];
                                        empty_quantity = ticks(
                                            candidates[slot * 9 + 3],
                                            candidates[slot * 9 + 4],
                                        );
                                        empty_trip = candidates[slot * 9 + 7];
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
                                    seen_supply = seen_supply || candidates[slot * 9 + 1] != 0;
                                    let amount =
                                        ticks(candidates[slot * 9 + 3], candidates[slot * 9 + 4]);
                                    let score =
                                        ticks(candidates[slot * 9 + 5], candidates[slot * 9 + 6]);
                                    let distance = candidates[slot * 9 + 7];
                                    let firm = candidates[slot * 9 + 8];
                                    if candidates[slot * 9 + 2] != 0 && lt(score, empty_score) {
                                        empty_seller = firm;
                                        empty_quantity = amount;
                                        empty_trip = distance;
                                        empty_score = score;
                                    }
                                    let eligible = candidates[slot * 9] != 0;
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
            if comptime!(parameters.market == MarketFlag::Consumer) && chosen != NO_ACTOR {
                let supply =
                    index::Supply::new(state, actors, config, chosen, parameters.traded_good());
                let affordable = minimum(wanted, divide(purchasing_power, max(supply.quote, 1)));
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
            }
        }
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
            let mut slot = 0u32;
            if block > 0 {
                slot = successes[(masks + block - 1) as usize].load();
            }
            for preceding in block * (markets::COMPACTION_BLOCK / 32)..word {
                slot += successes[preceding as usize].load().count_ones();
            }
            let mask = (1u32 << (position % 32)) - 1;
            slot += (successes[word as usize].load() & mask).count_ones();
            scratch[slot as usize] = peer;
            scratch[(count + slot) as usize] = actor;
        }
    }
}

#[derive(CubeType, Clone, Copy)]
struct CandidateRank {
    active: u32,
    score: Ticks,
}

#[cube]
fn plane_winner(rank: CandidateRank) -> u32 {
    let high = select(rank.active > 0, rank.score.extract(HIGH_WORD), u32::MAX);
    let minimum_high = plane_min(high);
    let low = select(
        rank.active > 0 && high == minimum_high,
        rank.score.extract(LOW_WORD),
        u32::MAX,
    );
    let minimum_low = plane_min(low);
    plane_min(select(
        rank.active > 0 && high == minimum_high && low == minimum_low,
        UNIT_POS,
        NO_ACTOR,
    ))
}

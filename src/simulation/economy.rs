use super::{
    flags::{
        ActorKind, COMPLETED_WORD, CONSTRUCTED_EVENTS_WORD, CONTROL_WORDS, EventStatus, FaultFlags,
        KernelActorKind, KernelAuthorization, KernelEventStatus, KernelFaults, MarketFlag,
        PROGRESS_WORD,
    },
    households::{KernelFatigue, KernelObservation, ObservationKind, next_fatigue, observed_blend},
    layout::*,
    markets::{self, LATCH_WORD, RANGES_WORD},
    negotiation::{NegotiationRequest, NegotiationStatus, legal_grid_rate, negotiate},
    parallel::{self, Reduction},
    parameters::{Discovery, EconomyParameters},
    payroll::PayrollAccrual,
    policy::*,
    scheduler::apply,
    words::*,
};
use crate::simulation::configuration::{EconomyConfig, Feature};
use crate::{
    domain::{
        BASE_FIELDS, CONSUMPTION_FIELDS, Field, Good, PRODUCTION_FIELDS, Phase, TRANSIT_FIELDS,
        USED_FIELDS,
    },
    ledger::{AuthorizationFlags, EMPTY, LEG_WORDS, LegLayout, MAX_LEGS},
};
use cubecl::prelude::*;
pub(super) mod audit;
mod clearing;
pub(super) mod enterprises;
mod household_production;
pub(super) mod search;
use super::scheduler::Posting;
use clearing::{PeerBalances, PeerOffer, actor_settlement};

#[cube]
fn at(config: &EconomyConfig, field: u32, actor: u32) -> u32 {
    field * config.actors + actor
}

#[cube]
fn value(state: &mut [u32], config: &EconomyConfig, field: u32, actor: u32) -> Ticks {
    load(state, at(config, field, actor))
}

#[cube]
fn number(state: &mut [u32], config: &EconomyConfig, field: u32, actor: u32) -> u32 {
    value(state, config, field, actor).extract(LOW_WORD)
}

#[cube]
fn set(state: &mut [u32], config: &EconomyConfig, field: u32, actor: u32, number: Ticks) {
    store(state, at(config, field, actor), number);
}

#[cube]
fn set32(state: &mut [u32], config: &EconomyConfig, field: u32, actor: u32, number: u32) {
    set(state, config, field, actor, ticks(number, 0u32));
}

#[cube]
fn add_checked(control: &[Atomic<u32>], a: Ticks, b: Ticks) -> Ticks {
    let result = plus(a, b);
    result.faults.record(control);
    result.value
}

#[cube]
fn multiply(control: &[Atomic<u32>], a: Ticks, b: u32) -> Ticks {
    let result = times(a, b);
    result.faults.record(control);
    result.value
}

#[cube]
fn increase(
    state: &mut [u32],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
    field: u32,
    actor: u32,
    amount: Ticks,
) {
    let next = add_checked(control, value(state, config, field, actor), amount);
    set(state, config, field, actor, next);
}

#[cube]
fn kind(actors: &[u32], actor: u32) -> KernelActorKind {
    KernelActorKind::load(
        actors[ActorLayout::KIND * (actors.len() / ActorLayout::WORDS) + actor as usize],
    )
}

#[cube]
fn distance(actors: &[u32], a: u32, b: u32) -> u32 {
    let ax = actors[ActorLayout::X * (actors.len() / ActorLayout::WORDS) + a as usize];
    let ay = actors[ActorLayout::Y * (actors.len() / ActorLayout::WORDS) + a as usize];
    let bx = actors[ActorLayout::X * (actors.len() / ActorLayout::WORDS) + b as usize];
    let by = actors[ActorLayout::Y * (actors.len() / ActorLayout::WORDS) + b as usize];
    max(ax, bx) - min(ax, bx) + max(ay, by) - min(ay, by)
}

#[allow(
    clippy::collapsible_match,
    clippy::manual_is_multiple_of,
    reason = "CubeCL uses explicit branches and integer remainder"
)]
#[cube]
fn local(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
    id: u32,
) {
    if id < config.actors {
        let k = kind(actors, id);
        let phase = comptime!(parameters.phase);
        match phase {
            Phase::Opening => {
                for good in 1u32..Good::COUNT as u32 + 1 {
                    let slot = BASE_FIELDS as u32
                        + (good - 1) * config.ring
                        + parameters.day % config.ring;
                    let delivery = value(state, config, slot, id);
                    increase(state, config, control, good, id, delivery);
                    let transit_field = TRANSIT_FIELDS as u32 + good - 1;
                    let available = value(state, config, transit_field, id);
                    if lt(available, delivery) {
                        KernelFaults::new(comptime!(FaultFlags::TRANSIT_UNDERFLOW)).record(control);
                    }
                    set(state, config, transit_field, id, minus(available, delivery));
                    let total = value(state, config, comptime!(Field::Transit as u32), id);
                    set(
                        state,
                        config,
                        comptime!(Field::Transit as u32),
                        id,
                        minus(total, delivery),
                    );
                    set32(state, config, slot, id, 0u32);
                }
                set32(state, config, comptime!(Field::Employed as u32), id, 0u32);
                set32(state, config, comptime!(Field::Employer as u32), id, 0u32);
                set32(state, config, comptime!(Field::Sales as u32), id, 0u32);
                set32(state, config, comptime!(Field::Unfilled as u32), id, 0u32);
                set32(state, config, comptime!(Field::Shortage as u32), id, 0u32);
                set32(state, config, comptime!(Field::Purchased as u32), id, 0u32);
                set32(state, config, comptime!(Field::Hours as u32), id, 0u32);
                set32(
                    state,
                    config,
                    comptime!(Field::HomeMinutes as u32),
                    id,
                    0u32,
                );
                set32(
                    state,
                    config,
                    comptime!(Field::ResearchMinutes as u32),
                    id,
                    0u32,
                );
            }
            Phase::Observation => {}
            Phase::Proposal => {
                if !k.is(comptime!(ActorKind::Household)) {
                    if comptime!(config.home_mode == Feature::Enabled) {
                        enterprises::available_method(state, actors, config, control, id);
                    }
                    let desired = proposed_starts(
                        KernelStarts {
                            coverage: config.coverage,
                            demand: value(state, config, comptime!(Field::Demand as u32), id),
                            finished: value(state, config, k.output(), id),
                            pipeline: value(state, config, comptime!(Field::Pipeline as u32), id),
                        },
                        control,
                    );
                    let quotient = divide(
                        add_checked(
                            control,
                            desired,
                            ticks(
                                number(state, config, comptime!(Field::OutputTicks as u32), id)
                                    - 1u32,
                                0u32,
                            ),
                        ),
                        number(state, config, comptime!(Field::OutputTicks as u32), id),
                    );
                    let mut planned = minimum(
                        quotient,
                        ticks(
                            actors[ActorLayout::GEOGRAPHIC_CAPACITY
                                * (actors.len() / ActorLayout::WORDS)
                                + id as usize],
                            0u32,
                        ),
                    );
                    for input in 0u32..2 {
                        let good = number(
                            state,
                            config,
                            comptime!(Field::FirstInput as u32) + input,
                            id,
                        );
                        if good > 0 {
                            planned = minimum(
                                planned,
                                divide(
                                    value(state, config, good, id),
                                    number(state, config, comptime!(Field::InputTicks as u32), id),
                                ),
                            );
                        }
                    }
                    let mut planning_wage =
                        number(state, config, comptime!(Field::Wage as u32), id);
                    if comptime!(config.bargaining.enabled == Feature::Enabled) {
                        planning_wage = legal_grid_rate(config);
                    }
                    planned = minimum(
                        planned,
                        divide(
                            value(state, config, comptime!(Field::Cash as u32), id),
                            planning_wage
                                * number(state, config, comptime!(Field::LaborMinutes as u32), id),
                        ),
                    );
                    let distributor =
                        number(state, config, comptime!(Field::EnterpriseModel as u32), id) == 1;
                    if distributor {
                        let stock = value(state, config, k.output(), id);
                        let demand = maximum(
                            value(state, config, comptime!(Field::Demand as u32), id),
                            ticks(1, 0),
                        );
                        let units = max(
                            number(state, config, comptime!(Field::OutputTicks as u32), id),
                            1,
                        );
                        let shifts = divide(
                            add_checked(control, minimum(stock, demand), ticks(units - 1, 0)),
                            units,
                        );
                        let funded = divide(
                            value(state, config, comptime!(Field::Cash as u32), id),
                            planning_wage
                                * number(state, config, comptime!(Field::LaborMinutes as u32), id),
                        );
                        let capacity = ticks(
                            actors[ActorLayout::CAPACITY * (actors.len() / ActorLayout::WORDS)
                                + id as usize],
                            0,
                        );
                        planned = minimum(minimum(shifts, funded), capacity);
                    }
                    if comptime!(config.home_mode == Feature::Enabled) {
                        let status =
                            number(state, config, comptime!(Field::EnterpriseStatus as u32), id);
                        planned = select(
                            status
                                == comptime!(
                                    crate::domain::enterprise::EnterpriseState::Operating as u32
                                ),
                            planned,
                            zero(),
                        );
                    }
                    set(state, config, comptime!(Field::Slots as u32), id, planned);
                    set(state, config, comptime!(Field::Plan as u32), id, planned);
                }
            }
            Phase::WorkPayment => {
                if number(state, config, comptime!(Field::EnterpriseModel as u32), id) == 1 {
                    let services = number(state, config, comptime!(Field::Pipeline as u32), id);
                    set32(
                        state,
                        config,
                        comptime!(Field::RetailOpen as u32),
                        id,
                        services,
                    );
                    set32(state, config, comptime!(Field::Pipeline as u32), id, 0);
                }
                if k.is(comptime!(ActorKind::Household))
                    && comptime!(config.home_mode == Feature::Enabled)
                    && number(state, config, comptime!(Field::LivingMembers as u32), id) > 0
                {
                    enterprises::expand(state, actors, config, control, id);
                    enterprises::construct_home(state, config, control, id);
                    household_production::produce(state, config, control, id, parameters.day);
                }
            }
            Phase::PhysicalClose => match k.bits() {
                ActorKind::HOUSEHOLD_BITS => {
                    let consumed = minimum(
                        value(state, config, comptime!(Field::Bread as u32), id),
                        ticks(
                            number(state, config, comptime!(Field::FoodNeed as u32), id),
                            0u32,
                        ),
                    );
                    let remaining = minus(
                        value(state, config, comptime!(Field::Bread as u32), id),
                        consumed,
                    );
                    set(state, config, comptime!(Field::Bread as u32), id, remaining);
                    increase(
                        state,
                        config,
                        control,
                        comptime!(Field::Consumed as u32),
                        id,
                        consumed,
                    );
                    increase(
                        state,
                        config,
                        control,
                        CONSUMPTION_FIELDS as u32 + comptime!(Good::Bread as u32) - 1,
                        id,
                        consumed,
                    );
                    if comptime!(config.economy == crate::domain::Economy::Industrial) {
                        for offset in 0..Good::COUNT {
                            let interval = config.consumption_intervals[offset];
                            if interval > 1
                                && (parameters.day + id) % max(interval, 1) == 0
                                && number(state, config, comptime!(Field::LivingMembers as u32), id)
                                    > 0
                            {
                                let good = offset as u32 + 1;
                                let stock = value(state, config, good, id);
                                let used = minimum(stock, ticks(1, 0));
                                set(state, config, good, id, minus(stock, used));
                                increase(
                                    state,
                                    config,
                                    control,
                                    CONSUMPTION_FIELDS as u32 + offset as u32,
                                    id,
                                    used,
                                );
                            }
                        }
                    }
                    let shortage = number(state, config, comptime!(Field::FoodNeed as u32), id)
                        - consumed.extract(LOW_WORD);
                    set32(
                        state,
                        config,
                        comptime!(Field::Shortage as u32),
                        id,
                        shortage,
                    );
                    let mut fatigue = min(
                        number(state, config, comptime!(Field::Fatigue as u32), id)
                            + config.work_fatigue,
                        POLICY_SCALE,
                    );
                    if number(state, config, comptime!(Field::Employed as u32), id) == 0u32 {
                        fatigue = max(
                            number(state, config, comptime!(Field::Fatigue as u32), id),
                            config.rest_recovery,
                        ) - config.rest_recovery;
                    }
                    if comptime!(config.household_policy == Feature::Enabled) {
                        let previous_fatigue =
                            number(state, config, comptime!(Field::Fatigue as u32), id);
                        let exertion = number(state, config, comptime!(Field::Hours as u32), id)
                            + number(state, config, comptime!(Field::ResearchMinutes as u32), id);
                        fatigue = next_fatigue(KernelFatigue {
                            fatigue: previous_fatigue,
                            response: config.fatigue_response,
                            exertion_minutes: exertion,
                            rest_minutes: config.rest_minutes,
                        });
                    }
                    set32(state, config, comptime!(Field::Fatigue as u32), id, fatigue);
                }
                _ => {
                    let output = value(state, config, comptime!(Field::Pipeline as u32), id);
                    increase(state, config, control, k.output(), id, output);
                    increase(
                        state,
                        config,
                        control,
                        comptime!(Field::Produced as u32),
                        id,
                        output,
                    );
                    set32(state, config, comptime!(Field::Pipeline as u32), id, 0u32);
                }
            },
            Phase::AccountsDecisions => {
                if k.is(comptime!(ActorKind::Household))
                    && comptime!(config.home_mode == Feature::Enabled)
                {
                    enterprises::household_quote(state, config, id);
                    enterprises::investment(state, config, id);
                }
                if !k.is(comptime!(ActorKind::Household)) {
                    if comptime!(config.home_mode == Feature::Enabled) {
                        let packets = number(state, config, comptime!(Field::Plan as u32), id)
                            - number(state, config, comptime!(Field::Slots as u32), id);
                        enterprises::maintain(state, config, control, id, packets);
                        enterprises::lifecycle(state, config, id);
                    }
                    let old = value(state, config, comptime!(Field::Demand as u32), id);
                    let signal = add_checked(
                        control,
                        value(state, config, comptime!(Field::Sales as u32), id),
                        value(state, config, comptime!(Field::Unfilled as u32), id),
                    );
                    let demand = observed_blend(
                        old,
                        KernelObservation {
                            kind: comptime!(ObservationKind::Observed as u32),
                            value: signal,
                        },
                        config.forecast_weight,
                        control,
                    );
                    let excess = positive(value(state, config, k.output(), id), demand);
                    let unfilled = value(state, config, comptime!(Field::Unfilled as u32), id);
                    let positive_sign = !lt(unfilled, excess);
                    let magnitude = select(
                        positive_sign,
                        minus(unfilled, excess),
                        minus(excess, unfilled),
                    );
                    let denominator = maximum(demand, ticks(1, 0));
                    let scaled = fraction_index(magnitude, denominator) * config.price_response
                        / POLICY_SCALE;
                    let table_index = select(
                        positive_sign,
                        PRICE_INDEX_LIMIT + scaled,
                        PRICE_INDEX_LIMIT - scaled,
                    );
                    let candidate = divide(
                        add_checked(
                            control,
                            multiply(
                                control,
                                value(state, config, comptime!(Field::Price as u32), id),
                                config.price_table[table_index as usize],
                            ),
                            ticks(select(positive_sign, PRICE_SCALE - 1, 0), 0u32),
                        ),
                        PRICE_SCALE,
                    );
                    set32(
                        state,
                        config,
                        comptime!(Field::Price as u32),
                        id,
                        min(
                            max(candidate.extract(LOW_WORD), config.minimum_price),
                            config.maximum_price,
                        ),
                    );
                    set(state, config, comptime!(Field::Demand as u32), id, demand);
                    enterprises::operating_plan(state, config, control, id);

                    let cap = divide(
                        divide(
                            value(state, config, comptime!(Field::Cash as u32), id),
                            number(state, config, comptime!(Field::LaborMinutes as u32), id),
                        ),
                        max(
                            number(state, config, comptime!(Field::Plan as u32), id),
                            1u32,
                        ),
                    );
                    if number(state, config, comptime!(Field::EnterpriseModel as u32), id) == 1 {
                        let replacement =
                            number(state, config, comptime!(Field::ReplacementCost as u32), id);
                        let margin =
                            number(state, config, comptime!(Field::RetailMargin as u32), id);
                        let labor = number(state, config, comptime!(Field::Wage as u32), id)
                            * number(state, config, comptime!(Field::LaborMinutes as u32), id);
                        let quote = replacement * (1000 + margin) / 1000
                            + ceildiv(
                                labor,
                                max(
                                    number(state, config, comptime!(Field::OutputTicks as u32), id),
                                    1,
                                ),
                            );
                        set32(
                            state,
                            config,
                            comptime!(Field::Price as u32),
                            id,
                            min(max(quote, config.minimum_price), config.maximum_price),
                        );
                    }
                    let wage = number(state, config, comptime!(Field::Wage as u32), id);
                    let unstaffed = number(state, config, comptime!(Field::Slots as u32), id) > 0;
                    let increment = config.wage_increment;
                    let offered = select(
                        unstaffed,
                        wage + increment,
                        max(wage, increment + MINIMUM_WAGE) - increment,
                    );
                    set32(
                        state,
                        config,
                        comptime!(Field::Wage as u32),
                        id,
                        min(
                            max(
                                minimum(ticks(offered, 0u32), cap).extract(LOW_WORD),
                                MINIMUM_WAGE,
                            ),
                            MAXIMUM_WAGE,
                        ),
                    );
                }
            }
            _ => {}
        }
    }
}

#[allow(
    clippy::collapsible_match,
    reason = "CubeCL uses explicit runtime branches"
)]
#[cube]
fn propose(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    spatial: &mut [u32],
    proposals: &mut [u32],
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
    invocation: u32,
) {
    if invocation < config.actors {
        let market = comptime!(parameters.market);
        let mut id = invocation;
        if comptime!(market == MarketFlag::Household && parameters.discovery.indexed()) {
            let cells = config.spatial_side * config.spatial_side;
            let base =
                3 * (cells + 1 + config.actors) + ceildiv(config.actors, DISCOVERY_BLOCK) + 1;
            id = spatial[(base + invocation) as usize];
        }
        let k = kind(actors, id);
        let mut chosen = u32::new(comptime!(i64::from(NO_ACTOR)));
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
        let mut purchasing_power = value(state, config, comptime!(Field::Cash as u32), id);
        if comptime!(matches!(market, MarketFlag::Input(_))) {
            purchasing_power = enterprises::purchasing_power(state, actors, config, control, id);
        }
        match market {
            MarketFlag::Labor => {
                search = k.is(comptime!(ActorKind::Household))
                    && number(state, config, comptime!(Field::Employed as u32), id) == 0u32;
            }
            MarketFlag::Input(_) => {
                good = comptime!(market.good().expect("input market") as u32);
                search = !k.is(comptime!(ActorKind::Household))
                    && (number(state, config, comptime!(Field::FirstInput as u32), id) == good
                        || number(state, config, comptime!(Field::SecondInput as u32), id) == good);
                wanted = ticks(
                    actors
                        [ActorLayout::CAPACITY * (actors.len() / ActorLayout::WORDS) + id as usize]
                        * number(state, config, comptime!(Field::InputTicks as u32), id)
                        * config.coverage,
                    0u32,
                );
                if number(state, config, comptime!(Field::EnterpriseModel as u32), id) == 1 {
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
                    && number(state, config, comptime!(Field::EnterpriseModel as u32), id) == 0
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
                                number(state, config, comptime!(Field::OutputTicks as u32), id) - 1,
                                0,
                            ),
                        ),
                        max(
                            number(state, config, comptime!(Field::OutputTicks as u32), id),
                            1,
                        ),
                    );
                    wanted = multiply(
                        control,
                        packets,
                        number(state, config, comptime!(Field::InputTicks as u32), id),
                    );
                }
                if k.is(comptime!(ActorKind::Household))
                    && comptime!(config.home_mode == Feature::Enabled)
                    && good == comptime!(Good::Grain as u32)
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
                    && number(state, config, comptime!(Field::LivingMembers as u32), id) > 0
                    && comptime!(config.home_mode == Feature::Enabled)
                    && (good == comptime!(Good::Tools as u32)
                        || good == comptime!(Good::Wood as u32))
                {
                    search = true;
                    wanted = ticks(select(good == comptime!(Good::Tools as u32), 2u32, 5u32), 0);
                }
                if !k.is(comptime!(ActorKind::Household))
                    && comptime!(config.home_mode == Feature::Enabled)
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
                    && good == comptime!(Good::Tools as u32)
                    && comptime!(config.home_mode == Feature::Enabled)
                {
                    wanted = select(search, maximum(wanted, ticks(2, 0)), ticks(2, 0));
                    search = true;
                }
            }
            MarketFlag::Consumer(_) => {
                good = comptime!(market.good().expect("consumer market") as u32);
                let interval = config.consumption_intervals[good as usize - 1];
                let until_due =
                    (interval - (parameters.day + id) % max(interval, 1)) % max(interval, 1);
                search = k.is(comptime!(ActorKind::Household))
                    && number(state, config, comptime!(Field::LivingMembers as u32), id) > 0
                    && until_due <= config.coverage + config.ring;
                let reserve = ticks(
                    number(state, config, comptime!(Field::FoodNeed as u32), id)
                        * (config.coverage + config.ring),
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
                wanted = ticks(1, 0);
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
                    chosen = number(state, config, comptime!(Field::Employer as u32), id) - 1u32;
                }
            }
            MarketFlag::Distribution => {
                if !k.is(comptime!(ActorKind::Household)) {
                    quantity = scaled(
                        control,
                        positive(
                            value(state, config, comptime!(Field::Cash as u32), id),
                            maximum(
                                ticks(config.reserve_low, config.reserve_high),
                                value(state, config, comptime!(Field::OperatingReserve as u32), id),
                            ),
                        ),
                        config.distribution_rate,
                        POLICY_SCALE,
                    );
                    if !equal(quantity, ticks(0u32, 0u32)) {
                        chosen = actors[ActorLayout::OWNER * (actors.len() / ActorLayout::WORDS)
                            + id as usize];
                    }
                }
            }
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
        if search {
            let px = actors[ActorLayout::X * (actors.len() / ActorLayout::WORDS) + id as usize];
            let py = actors[ActorLayout::Y * (actors.len() / ActorLayout::WORDS) + id as usize];
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
                    let nearest = max(px, x * radius) - min(px, (x + 1) * radius - 1)
                        + max(py, y * radius)
                        - min(py, (y + 1) * radius - 1);
                    if nearest > radius {
                        j = spatial[(cell + 1) as usize];
                    }
                    while j < spatial[(cell + 1) as usize] && considered < DISCOVERY_CANDIDATES {
                        let begin = spatial[cell as usize];
                        let length = spatial[(cell + 1) as usize] - begin;
                        let rotated = begin + (j - begin + parameters.day + id) % max(length, 1);
                        let mut skipped = 0u32;
                        if comptime!(
                            market.good() == Some(Good::Bread) && parameters.discovery.indexed()
                        ) {
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
                                seen_supply = distance(actors, id, seller) <= config.search_radius
                                    && household_production::offers(
                                        state, actors, config, seller, good,
                                    );
                                offset += 1;
                            }
                        }
                        if skipped == 0 {
                            let firm = spatial[(side * side + 1u32 + rotated) as usize];
                            let dist = distance(actors, id, firm);
                            if dist <= config.search_radius {
                                let mut wage =
                                    number(state, config, comptime!(Field::Wage as u32), firm);
                                let quote =
                                    number(state, config, comptime!(Field::Price as u32), firm);
                                let mut inspect = true;
                                if comptime!(market != MarketFlag::Labor) {
                                    let cash = purchasing_power;
                                    inspect = !lt(cash, ticks(max(quote, 1), 0));
                                    if !inspect {
                                        seen_supply = seen_supply
                                            || household_production::offers(
                                                state, actors, config, firm, good,
                                            );
                                    }
                                }
                                if inspect {
                                    let mut eligible = dist <= config.search_radius;

                                    let mut score = ticks(
                                        quote
                                            + max(ceildiv(dist, config.shipment_speed), 1u32)
                                                * config.delivery_penalty,
                                        0u32,
                                    );
                                    let mut amount = ticks(1u32, 0u32);
                                    if comptime!(market == MarketFlag::Labor) {
                                        if comptime!(config.bargaining.enabled == Feature::Enabled)
                                        {
                                            let negotiation = negotiate(
                                                NegotiationRequest {
                                                    prior_gross: value(
                                                        state,
                                                        config,
                                                        comptime!(Field::GrossIncome as u32),
                                                        id,
                                                    ),
                                                    household_cash: value(
                                                        state,
                                                        config,
                                                        comptime!(Field::Cash as u32),
                                                        id,
                                                    ),
                                                    funded_per_packet: divide(
                                                        value(
                                                            state,
                                                            config,
                                                            comptime!(Field::Cash as u32),
                                                            firm,
                                                        ),
                                                        max(
                                                            number(
                                                                state,
                                                                config,
                                                                comptime!(Field::Slots as u32),
                                                                firm,
                                                            ),
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
                                                && negotiation.status
                                                    == comptime!(NegotiationStatus::Agreed as u32);
                                            wage = negotiation.wage.extract(LOW_WORD);
                                            amount = negotiation.wage;
                                        }
                                        let duration = number(
                                            state,
                                            config,
                                            comptime!(Field::LaborMinutes as u32),
                                            firm,
                                        ) + ceildiv(dist, config.commute_speed)
                                            * COMMUTE_LEGS;
                                        eligible = eligible
                                            && number(
                                                state,
                                                config,
                                                comptime!(Field::Slots as u32),
                                                firm,
                                            ) > 0u32
                                            && number(
                                                state,
                                                config,
                                                comptime!(Field::Time as u32),
                                                id,
                                            ) >= duration;
                                        score = divide(
                                            times32(
                                                wage * number(
                                                    state,
                                                    config,
                                                    comptime!(Field::LaborMinutes as u32),
                                                    firm,
                                                ),
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
                                            let need = number(
                                                state,
                                                config,
                                                comptime!(Field::FoodNeed as u32),
                                                id,
                                            );
                                            let bread = number(
                                                state,
                                                config,
                                                comptime!(Field::Bread as u32),
                                                id,
                                            );
                                            let missing = max(need, bread) - bread;
                                            let minutes = missing
                                                * (config.home.milling_minutes
                                                    + config.home.baking_minutes);
                                            let time = number(
                                                state,
                                                config,
                                                comptime!(Field::Time as u32),
                                                id,
                                            );
                                            let remaining = max(time, duration) - duration
                                                + number(
                                                    state,
                                                    config,
                                                    comptime!(Field::HomeTime as u32),
                                                    id,
                                                );
                                            let self_provision = missing > 0
                                                && !lt(
                                                    value(
                                                        state,
                                                        config,
                                                        comptime!(Field::Grain as u32),
                                                        id,
                                                    ),
                                                    ticks(missing, 0),
                                                );
                                            let paid = times32(
                                                wage,
                                                number(
                                                    state,
                                                    config,
                                                    comptime!(Field::LaborMinutes as u32),
                                                    firm,
                                                ),
                                            );
                                            let food_cost = times32(
                                                missing,
                                                number(
                                                    state,
                                                    config,
                                                    comptime!(Field::Price as u32),
                                                    id,
                                                ),
                                            );
                                            eligible = eligible
                                                && !(self_provision
                                                    && remaining < minutes
                                                    && lt(paid, food_cost));
                                        }
                                    } else {
                                        let affordable = minimum(
                                            wanted,
                                            divide(purchasing_power, max(quote, 1u32)),
                                        );
                                        let offered = household_production::offers(
                                            state, actors, config, firm, good,
                                        );
                                        seen_supply = seen_supply || (eligible && offered);
                                        let known_producer = !kind(actors, firm)
                                            .is(comptime!(ActorKind::Household))
                                            && kind(actors, firm).output() == good;
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
                                        amount = minimum(
                                            minimum(
                                                wanted,
                                                positive(
                                                    value(state, config, good, firm),
                                                    household_production::reserve(
                                                        state, actors, config, firm, good,
                                                    ),
                                                ),
                                            ),
                                            divide(purchasing_power, max(quote, 1u32)),
                                        );
                                        eligible = eligible
                                            && firm != id
                                            && household_production::offers(
                                                state, actors, config, firm, good,
                                            )
                                            && quote > 0u32
                                            && !equal(amount, ticks(0u32, 0u32));
                                    }
                                    let mut improves = lt(score, best);
                                    if comptime!(market == MarketFlag::Labor) {
                                        improves = lt(best, score);
                                    }
                                    improves = improves || chosen == NO_ACTOR;
                                    considered += u32::cast_from(eligible);
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
        proposals[id as usize * ProposalLayout::WORDS + ProposalLayout::ACTOR] = chosen;
        proposals[id as usize * ProposalLayout::WORDS + ProposalLayout::QUANTITY_LOW] =
            quantity.extract(LOW_WORD);
        proposals[id as usize * ProposalLayout::WORDS + ProposalLayout::QUANTITY_HIGH] =
            quantity.extract(HIGH_WORD);
        proposals[id as usize * ProposalLayout::WORDS + ProposalLayout::DISTANCE] = trip;
    }
}

#[derive(CubeType, Clone, Copy)]
struct KernelLeg {
    field: u32,
    actor: u32,
    debit: Ticks,
    credit: Ticks,
    authorization: KernelAuthorization,
    guard: Ticks,
}

#[derive(CubeType, Clone, Copy)]
struct EventBuilder {
    position: usize,
}
#[cube]
impl EventBuilder {
    fn new(event: u32) -> EventBuilder {
        EventBuilder {
            position: event as usize * EVENT_WORDS,
        }
    }
    fn leg(&mut self, config: &EconomyConfig, bundles: &mut [u32], leg: KernelLeg) {
        let base = self.position;
        bundles[base + LegLayout::COORDINATE] = at(config, leg.field, leg.actor);
        bundles[base + LegLayout::DEBIT_LOW] = leg.debit.extract(LOW_WORD);
        bundles[base + LegLayout::DEBIT_HIGH] = leg.debit.extract(HIGH_WORD);
        bundles[base + LegLayout::CREDIT_LOW] = leg.credit.extract(LOW_WORD);
        bundles[base + LegLayout::CREDIT_HIGH] = leg.credit.extract(HIGH_WORD);
        bundles[base + LegLayout::AUTHORIZATION] = leg.authorization.bits();
        bundles[base + LegLayout::GUARD_LOW] = leg.guard.extract(LOW_WORD);
        bundles[base + LegLayout::GUARD_HIGH] = leg.guard.extract(HIGH_WORD);
        self.position += LEG_WORDS;
    }

    fn transfer(
        &mut self,
        config: &EconomyConfig,
        bundles: &mut [u32],
        field: u32,
        actor: u32,
        debit: Ticks,
        credit: Ticks,
    ) {
        self.leg(
            config,
            bundles,
            KernelLeg {
                field,
                actor,
                debit,
                credit,
                authorization: KernelAuthorization::new(comptime!(AuthorizationFlags::empty())),
                guard: ticks(0u32, 0u32),
            },
        );
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum EventStorage {
    Buffer,
    Local,
}

#[cube]
fn construct(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    proposals: &mut [u32],
    bundles: &mut [u32],
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
    id: u32,
    #[comptime] storage: EventStorage,
) {
    if id < parameters.count {
        let event_id = match comptime!(storage) {
            EventStorage::Buffer => id,
            EventStorage::Local => 0,
        };
        for slot in 0u32..MAX_LEGS as u32 {
            for word in 0..LEG_WORDS {
                bundles[event_id as usize * EVENT_WORDS + slot as usize * LEG_WORDS + word] =
                    select(word == LegLayout::COORDINATE, EMPTY, 0u32);
            }
        }
        let actor = id + parameters.offset;
        let other = proposals[actor as usize * ProposalLayout::WORDS + ProposalLayout::ACTOR];
        let quantity = ticks(
            proposals[actor as usize * ProposalLayout::WORDS + ProposalLayout::QUANTITY_LOW],
            proposals[actor as usize * ProposalLayout::WORDS + ProposalLayout::QUANTITY_HIGH],
        );
        let trip = proposals[actor as usize * ProposalLayout::WORDS + ProposalLayout::DISTANCE];
        let zero = ticks(0u32, 0u32);
        let mut event = EventBuilder::new(event_id);
        if other != NO_ACTOR {
            if comptime!(storage == EventStorage::Buffer) {
                control[CONSTRUCTED_EVENTS_WORD].fetch_add(1u32);
            }
            match comptime!(parameters.market) {
                MarketFlag::Labor => {
                    let mut offered_wage =
                        number(state, config, comptime!(Field::Wage as u32), other);
                    if comptime!(config.bargaining.enabled == Feature::Enabled) {
                        offered_wage = quantity.extract(LOW_WORD);
                    }
                    let wage = times32(
                        offered_wage,
                        number(state, config, comptime!(Field::LaborMinutes as u32), other),
                    );
                    event.transfer(
                        config,
                        bundles,
                        comptime!(Field::Cash as u32),
                        other,
                        wage,
                        zero,
                    );
                    event.transfer(
                        config,
                        bundles,
                        comptime!(Field::Escrow as u32),
                        other,
                        zero,
                        wage,
                    );
                    event.transfer(
                        config,
                        bundles,
                        comptime!(Field::Slots as u32),
                        other,
                        ticks(1u32, 0u32),
                        zero,
                    );
                    event.transfer(
                        config,
                        bundles,
                        comptime!(Field::Time as u32),
                        actor,
                        ticks(
                            number(state, config, comptime!(Field::LaborMinutes as u32), other)
                                + ceildiv(trip, config.commute_speed) * COMMUTE_LEGS,
                            0u32,
                        ),
                        zero,
                    );
                    event.transfer(
                        config,
                        bundles,
                        comptime!(Field::Employer as u32),
                        actor,
                        zero,
                        ticks(other + 1u32, 0u32),
                    );
                    event.transfer(
                        config,
                        bundles,
                        comptime!(Field::Due as u32),
                        actor,
                        zero,
                        wage,
                    );
                    event.leg(
                        config,
                        bundles,
                        KernelLeg {
                            field: comptime!(Field::Employed as u32),
                            actor,
                            debit: zero,
                            credit: ticks(1u32, 0u32),
                            authorization: KernelAuthorization::new(comptime!(
                                AuthorizationFlags::EQUAL
                            )),
                            guard: zero,
                        },
                    );
                    event.transfer(
                        config,
                        bundles,
                        comptime!(Field::Hours as u32),
                        actor,
                        zero,
                        ticks(
                            number(state, config, comptime!(Field::LaborMinutes as u32), other),
                            0u32,
                        ),
                    );
                    event.transfer(
                        config,
                        bundles,
                        comptime!(Field::Pipeline as u32),
                        other,
                        zero,
                        ticks(
                            number(state, config, comptime!(Field::OutputTicks as u32), other),
                            0u32,
                        ),
                    );
                    event.leg(
                        config,
                        bundles,
                        KernelLeg {
                            field: comptime!(Field::Wage as u32),
                            actor: other,
                            debit: zero,
                            credit: zero,
                            authorization: KernelAuthorization::new(comptime!(
                                AuthorizationFlags::EQUAL
                            )),
                            guard: value(state, config, comptime!(Field::Wage as u32), other),
                        },
                    );
                    let mut total_used = zero;
                    for input in 0u32..2 {
                        let good = number(
                            state,
                            config,
                            comptime!(Field::FirstInput as u32) + input,
                            other,
                        );
                        if good > 0 {
                            let amount =
                                value(state, config, comptime!(Field::InputTicks as u32), other);
                            event.transfer(config, bundles, good, other, amount, zero);
                            event.transfer(
                                config,
                                bundles,
                                USED_FIELDS as u32 + good - 1,
                                other,
                                zero,
                                amount,
                            );
                            total_used = add_checked(control, total_used, amount);
                        }
                    }
                    event.transfer(
                        config,
                        bundles,
                        comptime!(Field::Used as u32),
                        other,
                        zero,
                        total_used,
                    );
                }
                MarketFlag::Payroll => {
                    let mut net = quantity;
                    let prior = value(state, config, comptime!(Field::GrossIncome as u32), actor);
                    event.leg(
                        config,
                        bundles,
                        KernelLeg {
                            field: comptime!(Field::GrossIncome as u32),
                            actor,
                            debit: zero,
                            credit: quantity,
                            authorization: KernelAuthorization::new(comptime!(
                                AuthorizationFlags::EQUAL
                            )),
                            guard: prior,
                        },
                    );
                    if comptime!(config.payroll == Feature::Enabled) {
                        let accrued = add_checked(control, prior, quantity);
                        let before = PayrollAccrual::new(prior, config, control);
                        let after = PayrollAccrual::new(accrued, config, control);
                        net = minus(after.net, before.net);
                        event.leg(
                            config,
                            bundles,
                            KernelLeg {
                                field: comptime!(Field::PayrollReserve as u32),
                                actor,
                                debit: before.reserve,
                                credit: after.reserve,
                                authorization: KernelAuthorization::new(comptime!(
                                    AuthorizationFlags::EQUAL
                                )),
                                guard: before.reserve,
                            },
                        );
                        event.transfer(
                            config,
                            bundles,
                            comptime!(Field::TaxTreasury as u32),
                            actor,
                            zero,
                            minus(after.tax, before.tax),
                        );
                        event.transfer(
                            config,
                            bundles,
                            comptime!(Field::ChargeTreasury as u32),
                            actor,
                            zero,
                            minus(after.charge, before.charge),
                        );
                    }
                    event.transfer(
                        config,
                        bundles,
                        comptime!(Field::Escrow as u32),
                        other,
                        quantity,
                        zero,
                    );
                    event.transfer(
                        config,
                        bundles,
                        comptime!(Field::Due as u32),
                        actor,
                        quantity,
                        zero,
                    );
                    event.transfer(
                        config,
                        bundles,
                        comptime!(Field::Cash as u32),
                        actor,
                        zero,
                        net,
                    );
                    event.leg(
                        config,
                        bundles,
                        KernelLeg {
                            field: comptime!(Field::Employer as u32),
                            actor,
                            debit: zero,
                            credit: zero,
                            authorization: KernelAuthorization::new(comptime!(
                                AuthorizationFlags::EQUAL
                            )),
                            guard: ticks(other + 1u32, 0u32),
                        },
                    );
                }
                MarketFlag::Distribution => {
                    event.transfer(
                        config,
                        bundles,
                        comptime!(Field::Cash as u32),
                        actor,
                        quantity,
                        zero,
                    );
                    event.transfer(
                        config,
                        bundles,
                        comptime!(Field::Cash as u32),
                        other,
                        zero,
                        quantity,
                    );
                }
                _ => {
                    let good = comptime!(parameters.market.good().expect("goods market") as u32);
                    let cost = multiply(
                        control,
                        quantity,
                        number(state, config, comptime!(Field::Price as u32), other),
                    );
                    let slot = BASE_FIELDS as u32
                        + (good - 1) * config.ring
                        + (parameters.day + max(ceildiv(trip, config.shipment_speed), 1u32))
                            % config.ring;
                    event.transfer(
                        config,
                        bundles,
                        comptime!(Field::Cash as u32),
                        actor,
                        cost,
                        zero,
                    );
                    event.transfer(
                        config,
                        bundles,
                        comptime!(Field::Cash as u32),
                        other,
                        zero,
                        cost,
                    );
                    let retained =
                        household_production::reserve(state, actors, config, other, good);
                    event.leg(
                        config,
                        bundles,
                        KernelLeg {
                            field: good,
                            actor: other,
                            debit: quantity,
                            credit: zero,
                            authorization: KernelAuthorization::new(comptime!(
                                AuthorizationFlags::AT_LEAST
                            )),
                            guard: add_checked(control, retained, quantity),
                        },
                    );
                    event.transfer(config, bundles, slot, actor, zero, quantity);
                    for input in 0u32..2 {
                        if number(
                            state,
                            config,
                            comptime!(Field::FirstInput as u32) + input,
                            actor,
                        ) == good
                        {
                            let cost_field = comptime!(Field::FirstInputCost as u32) + input;
                            event.transfer(
                                config,
                                bundles,
                                cost_field,
                                actor,
                                value(state, config, cost_field, actor),
                                value(state, config, comptime!(Field::Price as u32), other),
                            );
                        }
                    }
                    if number(
                        state,
                        config,
                        comptime!(Field::EnterpriseModel as u32),
                        actor,
                    ) == 1
                    {
                        event.transfer(
                            config,
                            bundles,
                            comptime!(Field::ReplacementCost as u32),
                            actor,
                            value(
                                state,
                                config,
                                comptime!(Field::ReplacementCost as u32),
                                actor,
                            ),
                            value(state, config, comptime!(Field::Price as u32), other),
                        );
                    }
                    event.transfer(
                        config,
                        bundles,
                        TRANSIT_FIELDS as u32 + good - 1,
                        actor,
                        zero,
                        quantity,
                    );
                    event.transfer(
                        config,
                        bundles,
                        comptime!(Field::Transit as u32),
                        actor,
                        zero,
                        quantity,
                    );
                    event.transfer(
                        config,
                        bundles,
                        comptime!(Field::Sales as u32),
                        other,
                        zero,
                        quantity,
                    );
                    event.transfer(
                        config,
                        bundles,
                        comptime!(Field::Purchased as u32),
                        actor,
                        zero,
                        quantity,
                    );
                    event.leg(
                        config,
                        bundles,
                        KernelLeg {
                            field: comptime!(Field::Price as u32),
                            actor: other,
                            debit: zero,
                            credit: zero,
                            authorization: KernelAuthorization::new(comptime!(
                                AuthorizationFlags::EQUAL
                            )),
                            guard: value(state, config, comptime!(Field::Price as u32), other),
                        },
                    );
                }
            }
        }
    }
}

#[cube]
fn summarize(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    bundles: &mut [u32],
    control: &[Atomic<u32>],
    id: u32,
) {
    let groups = ceildiv(config.actors, SummaryLayout::ACTORS_PER_GROUP);
    if id < groups {
        let mut totals = Array::<Ticks>::new(SummaryLayout::METRICS);
        for metric in 0..SummaryLayout::METRICS {
            totals[metric] = zero();
        }
        totals[SummaryLayout::MINIMUM_WAGE as usize] = ticks(u32::MAX, u32::MAX);
        totals[SummaryLayout::MINIMUM_PRICE as usize] = ticks(u32::MAX, u32::MAX);
        for good in 0u32..Good::COUNT as u32 {
            totals[(SummaryLayout::WAGE_MINIMA + good) as usize] = ticks(u32::MAX, u32::MAX);
            totals[(SummaryLayout::PRICE_MINIMA + good) as usize] = ticks(u32::MAX, u32::MAX);
        }
        let count = config.actors;
        for offset in 0..ceildiv(count - id, groups) {
            let actor = id + offset * groups;
            totals[SummaryLayout::CASH as usize] = add_checked(
                control,
                totals[SummaryLayout::CASH as usize],
                add_checked(
                    control,
                    value(state, config, comptime!(Field::Cash as u32), actor),
                    value(state, config, comptime!(Field::Escrow as u32), actor),
                ),
            );
            let payroll_reserve = value(
                state,
                config,
                comptime!(Field::PayrollReserve as u32),
                actor,
            );
            let tax = value(state, config, comptime!(Field::TaxTreasury as u32), actor);
            let charge = value(
                state,
                config,
                comptime!(Field::ChargeTreasury as u32),
                actor,
            );
            totals[SummaryLayout::PAYROLL_RESERVES as usize] = add_checked(
                control,
                totals[SummaryLayout::PAYROLL_RESERVES as usize],
                payroll_reserve,
            );
            totals[SummaryLayout::INCOME_TAX as usize] =
                add_checked(control, totals[SummaryLayout::INCOME_TAX as usize], tax);
            totals[SummaryLayout::EMPLOYEE_CHARGES as usize] = add_checked(
                control,
                totals[SummaryLayout::EMPLOYEE_CHARGES as usize],
                charge,
            );
            totals[SummaryLayout::CASH as usize] = add_checked(
                control,
                totals[SummaryLayout::CASH as usize],
                add_checked(control, payroll_reserve, add_checked(control, tax, charge)),
            );
            if comptime!(config.payroll == Feature::Enabled) {
                let accrued = PayrollAccrual::new(
                    value(state, config, comptime!(Field::GrossIncome as u32), actor),
                    config,
                    control,
                );
                if !equal(payroll_reserve, accrued.reserve)
                    || !equal(tax, accrued.tax)
                    || !equal(charge, accrued.charge)
                {
                    KernelFaults::new(comptime!(FaultFlags::PAYROLL_MISMATCH)).record(control);
                }
            }
            for g in 0u32..Good::COUNT as u32 {
                totals[(SummaryLayout::INVENTORIES + g) as usize] = add_checked(
                    control,
                    totals[(SummaryLayout::INVENTORIES + g) as usize],
                    value(state, config, g + 1u32, actor),
                );
            }
            if !equal(
                value(state, config, comptime!(Field::Escrow as u32), actor),
                zero(),
            ) || !equal(
                value(state, config, comptime!(Field::Pipeline as u32), actor),
                zero(),
            ) || !equal(
                value(state, config, comptime!(Field::Due as u32), actor),
                zero(),
            ) {
                KernelFaults::new(comptime!(FaultFlags::UNSETTLED_COMMITMENTS)).record(control);
            }
            let k = kind(actors, actor);
            match k.bits() {
                ActorKind::HOUSEHOLD_BITS => {
                    totals[SummaryLayout::EMPLOYED as usize] = add_checked(
                        control,
                        totals[SummaryLayout::EMPLOYED as usize],
                        value(state, config, comptime!(Field::Employed as u32), actor),
                    );
                    totals[SummaryLayout::SHORTAGES as usize] = add_checked(
                        control,
                        totals[SummaryLayout::SHORTAGES as usize],
                        value(state, config, comptime!(Field::Shortage as u32), actor),
                    );
                    totals[SummaryLayout::CONSUMED as usize] = add_checked(
                        control,
                        totals[SummaryLayout::CONSUMED as usize],
                        value(state, config, comptime!(Field::Consumed as u32), actor),
                    );
                    let cash = value(state, config, comptime!(Field::Cash as u32), actor);
                    let mut bin = WORD_BITS - cash.extract(LOW_WORD).leading_zeros();
                    if cash.extract(HIGH_WORD) > 0u32 {
                        bin = TICK_BITS - cash.extract(HIGH_WORD).leading_zeros();
                    }
                    control[(SummaryLayout::WEALTH_START as u32
                        + min(bin, SummaryLayout::WEALTH_BINS as u32 - 1))
                        as usize]
                        .fetch_add(1u32);
                    let x = actors
                        [ActorLayout::X * (actors.len() / ActorLayout::WORDS) + actor as usize]
                        * SummaryLayout::REGION_SIDE as u32
                        / config.map_size;
                    let y = actors
                        [ActorLayout::Y * (actors.len() / ActorLayout::WORDS) + actor as usize]
                        * SummaryLayout::REGION_SIDE as u32
                        / config.map_size;
                    control[(SummaryLayout::REGIONS_START as u32
                        + y * SummaryLayout::REGION_SIDE as u32
                        + x) as usize]
                        .fetch_add(number(
                            state,
                            config,
                            comptime!(Field::Shortage as u32),
                            actor,
                        ));
                }
                _ => {
                    totals[(SummaryLayout::INVENTORIES + k.output() - 1) as usize] = add_checked(
                        control,
                        totals[(SummaryLayout::INVENTORIES + k.output() - 1) as usize],
                        value(state, config, comptime!(Field::Pipeline as u32), actor),
                    );
                    totals[(SummaryLayout::PRODUCED + k.output() - 1) as usize] = add_checked(
                        control,
                        totals[(SummaryLayout::PRODUCED + k.output() - 1) as usize],
                        value(state, config, comptime!(Field::Produced as u32), actor),
                    );
                    totals[SummaryLayout::DISTRIBUTORS as usize] = add_checked(
                        control,
                        totals[SummaryLayout::DISTRIBUTORS as usize],
                        ticks(
                            u32::cast_from(
                                number(
                                    state,
                                    config,
                                    comptime!(Field::EnterpriseModel as u32),
                                    actor,
                                ) == 1,
                            ),
                            0,
                        ),
                    );
                    totals[SummaryLayout::CLOSURES as usize] = add_checked(
                        control,
                        totals[SummaryLayout::CLOSURES as usize],
                        ticks(
                            u32::cast_from(
                                number(
                                    state,
                                    config,
                                    comptime!(Field::EnterpriseStatus as u32),
                                    actor,
                                ) == 4,
                            ),
                            0,
                        ),
                    );
                    let good = k.output() - 1;
                    let wage = value(state, config, comptime!(Field::Wage as u32), actor);
                    let price = value(state, config, comptime!(Field::Price as u32), actor);
                    totals[(SummaryLayout::FIRMS_BY_GOOD + good) as usize] = add_checked(
                        control,
                        totals[(SummaryLayout::FIRMS_BY_GOOD + good) as usize],
                        ticks(1, 0),
                    );
                    totals[(SummaryLayout::WAGE_MINIMA + good) as usize] =
                        minimum(totals[(SummaryLayout::WAGE_MINIMA + good) as usize], wage);
                    totals[(SummaryLayout::WAGE_MAXIMA + good) as usize] =
                        maximum(totals[(SummaryLayout::WAGE_MAXIMA + good) as usize], wage);
                    totals[(SummaryLayout::WAGE_TOTALS + good) as usize] = add_checked(
                        control,
                        totals[(SummaryLayout::WAGE_TOTALS + good) as usize],
                        wage,
                    );
                    totals[(SummaryLayout::WAGE_SQUARES + good) as usize] = add_checked(
                        control,
                        totals[(SummaryLayout::WAGE_SQUARES + good) as usize],
                        times32(wage.extract(LOW_WORD), wage.extract(LOW_WORD)),
                    );
                    totals[(SummaryLayout::PRICE_MINIMA + good) as usize] =
                        minimum(totals[(SummaryLayout::PRICE_MINIMA + good) as usize], price);
                    totals[(SummaryLayout::PRICE_MAXIMA + good) as usize] =
                        maximum(totals[(SummaryLayout::PRICE_MAXIMA + good) as usize], price);
                    totals[(SummaryLayout::PRICE_TOTALS + good) as usize] = add_checked(
                        control,
                        totals[(SummaryLayout::PRICE_TOTALS + good) as usize],
                        price,
                    );
                    totals[(SummaryLayout::PRICE_SQUARES + good) as usize] = add_checked(
                        control,
                        totals[(SummaryLayout::PRICE_SQUARES + good) as usize],
                        times32(price.extract(LOW_WORD), price.extract(LOW_WORD)),
                    );
                    totals[SummaryLayout::MINIMUM_WAGE as usize] = minimum(
                        totals[SummaryLayout::MINIMUM_WAGE as usize],
                        value(state, config, comptime!(Field::Wage as u32), actor),
                    );
                    totals[SummaryLayout::MAXIMUM_WAGE as usize] = select(
                        lt(
                            totals[SummaryLayout::MAXIMUM_WAGE as usize],
                            value(state, config, comptime!(Field::Wage as u32), actor),
                        ),
                        value(state, config, comptime!(Field::Wage as u32), actor),
                        totals[SummaryLayout::MAXIMUM_WAGE as usize],
                    );
                    totals[SummaryLayout::MINIMUM_PRICE as usize] = minimum(
                        totals[SummaryLayout::MINIMUM_PRICE as usize],
                        value(state, config, comptime!(Field::Price as u32), actor),
                    );
                    totals[SummaryLayout::MAXIMUM_PRICE as usize] = select(
                        lt(
                            totals[SummaryLayout::MAXIMUM_PRICE as usize],
                            value(state, config, comptime!(Field::Price as u32), actor),
                        ),
                        value(state, config, comptime!(Field::Price as u32), actor),
                        totals[SummaryLayout::MAXIMUM_PRICE as usize],
                    );
                }
            }
            let mut transit = ticks(0u32, 0u32);
            for good in 0u32..Good::COUNT as u32 {
                let held = value(state, config, TRANSIT_FIELDS as u32 + good, actor);
                totals[(SummaryLayout::TRANSIT + good) as usize] = add_checked(
                    control,
                    totals[(SummaryLayout::TRANSIT + good) as usize],
                    held,
                );
                let household_output = value(state, config, PRODUCTION_FIELDS as u32 + good, actor);
                totals[(SummaryLayout::HOUSEHOLD_PRODUCTION + good) as usize] = add_checked(
                    control,
                    totals[(SummaryLayout::HOUSEHOLD_PRODUCTION + good) as usize],
                    household_output,
                );
                totals[(SummaryLayout::PRODUCED + good) as usize] = add_checked(
                    control,
                    totals[(SummaryLayout::PRODUCED + good) as usize],
                    household_output,
                );
                totals[(SummaryLayout::USED + good) as usize] = add_checked(
                    control,
                    totals[(SummaryLayout::USED + good) as usize],
                    value(state, config, USED_FIELDS as u32 + good, actor),
                );
                totals[(SummaryLayout::CONSUMPTION + good) as usize] = add_checked(
                    control,
                    totals[(SummaryLayout::CONSUMPTION + good) as usize],
                    value(state, config, CONSUMPTION_FIELDS as u32 + good, actor),
                );
                let mut in_ring = zero();
                for slot in 0u32..config.ring {
                    in_ring = add_checked(
                        control,
                        in_ring,
                        value(
                            state,
                            config,
                            BASE_FIELDS as u32 + good * config.ring + slot,
                            actor,
                        ),
                    );
                }
                if !equal(in_ring, held) {
                    KernelFaults::new(comptime!(FaultFlags::TRANSIT_MISMATCH)).record(control);
                }
                transit = add_checked(control, transit, held);
            }
            if !equal(
                transit,
                value(state, config, comptime!(Field::Transit as u32), actor),
            ) {
                KernelFaults::new(comptime!(FaultFlags::TRANSIT_MISMATCH)).record(control);
            }
        }
        for metric in 0u32..SummaryLayout::METRICS as u32 {
            let base = (id * SummaryLayout::METRICS as u32 + metric) * TICK_WORDS as u32;
            bundles[(base) as usize] = totals[(metric) as usize].extract(LOW_WORD);
            bundles[(base + 1u32) as usize] = totals[(metric) as usize].extract(HIGH_WORD);
        }
    }
}

#[cube]
fn inspect(
    state: &mut [u32],
    config: &EconomyConfig,
    proposals: &mut [u32],
    parameters: &EconomyParameters,
    id: u32,
) {
    if id < config.fields {
        let item = value(state, config, id, parameters.offset);
        proposals[(id * TICK_WORDS as u32) as usize] = item.extract(LOW_WORD);
        proposals[(id * TICK_WORDS as u32 + 1u32) as usize] = item.extract(HIGH_WORD);
    }
}

#[enumflags2::bitflags]
#[repr(u32)]
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum EconomyStage {
    ClearFeedback,
    RecordFeedback,
    ApplyFeedback,
    IndexCount,
    IndexOffsets,
    IndexFill,
    IndexPrices,
    Local,
    Propose,
    MarketActivity,
    Construct,
    Summarize,
    Inspect,
    GroupInitialize,
    GroupSort,
    GroupMerge,
    GroupExecute,
    GroupPrepare,
    GroupFinalize,
    MarketStart,
    MarketBoundary,
    GroupRanges,
    GroupCompactCount,
    GroupCompactFill,
    GroupRadixLocal,
    GroupRadixScatter,
    SearchPrepare,
}

#[cube(launch)]
pub fn economy(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    spatial: &mut [u32],
    proposals: &mut [u32],
    bundles: &mut [u32],
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
    #[comptime] stage: EconomyStage,
) {
    let id = ABSOLUTE_POS as u32;
    let guarded = comptime!(
        parameters.discovery == Discovery::Indexed
            && matches!(
                stage,
                EconomyStage::IndexCount
                    | EconomyStage::IndexOffsets
                    | EconomyStage::IndexFill
                    | EconomyStage::IndexPrices
                    | EconomyStage::Propose
                    | EconomyStage::GroupInitialize
                    | EconomyStage::GroupPrepare
                    | EconomyStage::GroupSort
                    | EconomyStage::GroupMerge
                    | EconomyStage::GroupRanges
                    | EconomyStage::SearchPrepare
                    | EconomyStage::GroupCompactCount
                    | EconomyStage::GroupCompactFill
                    | EconomyStage::GroupRadixLocal
                    | EconomyStage::GroupRadixScatter
                    | EconomyStage::GroupExecute
                    | EconomyStage::GroupFinalize
            )
    );
    if !guarded || control[LATCH_WORD].load() == 0 {
        match stage {
            EconomyStage::MarketStart | EconomyStage::MarketBoundary => {
                if id == 0 {
                    control[LATCH_WORD].store(control[crate::simulation::flags::FAULT_WORD].load());
                    if comptime!(stage == EconomyStage::MarketStart) {
                        control[CONSTRUCTED_EVENTS_WORD].store(0);
                        control[RANGES_WORD].store(0);
                        control[markets::SEARCHES_WORD].store(0);
                    }
                }
            }
            EconomyStage::SearchPrepare => search::prepare(
                state, actors, config, spatial, proposals, control, bundles, parameters, id,
            ),
            EconomyStage::GroupCompactCount => {
                markets::compact_count(proposals, bundles, parameters, id)
            }
            EconomyStage::GroupCompactFill => {
                markets::compact_fill(proposals, bundles, control, parameters, id)
            }
            EconomyStage::GroupRadixLocal => markets::radix_local(bundles, control, parameters, id),
            EconomyStage::GroupRadixScatter => {
                markets::radix_scatter(bundles, control, parameters, id)
            }
            EconomyStage::GroupRanges => markets::ranges(bundles, control, parameters, id),
            EconomyStage::ClearFeedback => {
                if id < control.len() as u32 {
                    control[id as usize].store(0);
                }
            }
            EconomyStage::RecordFeedback => {
                record_feedback(state, config, proposals, bundles, control, parameters, id)
            }
            EconomyStage::ApplyFeedback => apply_feedback(state, config, control, id),
            EconomyStage::IndexCount => {
                index_count(state, actors, config, spatial, bundles, parameters, id)
            }
            EconomyStage::IndexOffsets => index_offsets(config, spatial, bundles, parameters, id),
            EconomyStage::IndexFill => index_fill(config, spatial, bundles, parameters, id),
            EconomyStage::IndexPrices => index_prices(state, config, spatial),
            EconomyStage::Local => local(state, actors, config, control, parameters, id),
            EconomyStage::Propose => propose(
                state, actors, config, spatial, proposals, control, parameters, id,
            ),
            EconomyStage::MarketActivity => {
                let mut active = 0u32;
                if id < config.actors {
                    active = u32::cast_from(
                        proposals[id as usize * ProposalLayout::WORDS + ProposalLayout::ACTOR]
                            != NO_ACTOR,
                    );
                }
                let total = parallel::reduce(active, comptime!(Reduction::Sum));
                if UNIT_POS == 0 && total != 0 {
                    control[CONSTRUCTED_EVENTS_WORD].fetch_add(total);
                }
            }
            EconomyStage::Construct => construct(
                state,
                actors,
                config,
                proposals,
                bundles,
                control,
                parameters,
                id,
                comptime!(EventStorage::Buffer),
            ),
            EconomyStage::GroupInitialize => {
                group_initialize(proposals, bundles, control, parameters, id)
            }
            EconomyStage::GroupSort => group_sort(bundles, control, parameters, id),
            EconomyStage::GroupMerge => group_merge(bundles, control, parameters, id),
            EconomyStage::GroupExecute => group_execute(
                state, actors, config, proposals, bundles, control, parameters, id,
            ),
            EconomyStage::GroupPrepare => actor_settlement(
                state,
                actors,
                config,
                proposals,
                bundles,
                control,
                parameters,
                id,
                comptime!(Posting::CheckActor),
            ),
            EconomyStage::GroupFinalize => actor_settlement(
                state,
                actors,
                config,
                proposals,
                bundles,
                control,
                parameters,
                id,
                comptime!(Posting::ApplyActor),
            ),
            EconomyStage::Summarize => summarize(state, actors, config, bundles, control, id),
            EconomyStage::Inspect => inspect(state, config, proposals, parameters, id),
        }
    }
}

#[cube]
fn discoverable(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    parameters: &EconomyParameters,
    firm: u32,
) -> bool {
    let market = comptime!(parameters.market);
    match market {
        MarketFlag::Labor => number(state, config, comptime!(Field::Slots as u32), firm) > 0,
        MarketFlag::Input(_) | MarketFlag::Consumer(_) | MarketFlag::Household => {
            let good = comptime!(market.good().expect("goods market") as u32);
            let distributor = number(
                state,
                config,
                comptime!(Field::EnterpriseModel as u32),
                firm,
            ) == 1;
            let k = kind(actors, firm);
            let status = number(
                state,
                config,
                comptime!(Field::EnterpriseStatus as u32),
                firm,
            );
            let producer = !k.is(comptime!(ActorKind::Household))
                && k.output() == good
                && status
                    < comptime!(crate::domain::enterprise::EnterpriseState::Liquidating as u32);
            let mut available =
                producer || household_production::offers(state, actors, config, firm, good);
            if comptime!(matches!(market, MarketFlag::Input(_))) {
                available = available && !distributor;
            }
            available
        }
        MarketFlag::Payroll | MarketFlag::Distribution => false,
    }
}

#[cube]
fn group_initialize(
    proposals: &mut [u32],
    scratch: &mut [u32],
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
    id: u32,
) {
    let count = parameters.count;
    let mut peer = u32::new(comptime!(i64::from(NO_ACTOR)));
    if id < count {
        peer = proposals
            [(parameters.offset + id) as usize * ProposalLayout::WORDS + ProposalLayout::ACTOR];
        KernelEventStatus::new(comptime!(EventStatus::Inactive)).store(scratch, 4 * count + id);
    }
    let prefix = parallel::prefix_sum(u32::cast_from(peer != NO_ACTOR));
    let mut first = Shared::<u32>::new();
    if UNIT_POS == 0 {
        *first = control[CONSTRUCTED_EVENTS_WORD].fetch_add(prefix.total);
    }
    sync_cube();
    if peer != NO_ACTOR {
        let slot = *first + prefix.inclusive - 1;
        scratch[slot as usize] = peer;
        scratch[(count + slot) as usize] = id;
    }
}

#[cube]
fn group_sort(
    scratch: &mut [u32],
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
    id: u32,
) {
    let count = parameters.count;
    let active = markets::active(control, parameters);
    let lane = UNIT_POS as usize;
    let mut keys = Shared::<[u32]>::new_slice(256usize);
    let mut items = Shared::<[u32]>::new_slice(256usize);
    let mut key = u32::new(comptime!(i64::from(NO_ACTOR)));
    let mut item = 0u32;
    if id < active {
        key = scratch[id as usize];
        item = scratch[(count + id) as usize];
    }
    keys[lane] = key;
    items[lane] = item;
    sync_cube();
    let rotation = parameters.settlement_rotation % max(count, 1);
    let mut width = 2u32;
    while width <= CUBE_DIM {
        let mut stride = width / 2;
        while stride > 0 {
            let other = (UNIT_POS ^ stride) as usize;
            let other_key = keys[other];
            let other_item = items[other];
            let priority = (item + count - rotation) % max(count, 1);
            let other_priority = (other_item + count - rotation) % max(count, 1);
            let precedes = other_key < key || (other_key == key && other_priority < priority);
            let lower = (UNIT_POS & width == 0) == (UNIT_POS & stride == 0);
            if precedes == lower {
                key = other_key;
                item = other_item;
            }
            sync_cube();
            keys[lane] = key;
            items[lane] = item;
            sync_cube();
            stride /= 2;
        }
        width *= 2;
    }
    if id < active {
        scratch[id as usize] = key;
        scratch[(count + id) as usize] = item;
    }
}

#[cube]
fn group_merge(
    scratch: &mut [u32],
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
    id: u32,
) {
    let count = parameters.count;
    let active = markets::active(control, parameters);
    if id < active {
        let width = parameters.sort_width;
        let source = parameters.sort_source;
        let destination = 2 * count - source;
        let base = id / (width * 2) * width * 2;
        let middle = min(base + width, active);
        let end = min(base + width * 2, active);
        let first = id < middle;
        let start = select(first, middle, base);
        let stop = select(first, end, middle);
        let key = scratch[(source + id) as usize];
        let item = scratch[(source + count + id) as usize];
        let mut low = start;
        let mut high = stop;
        while low < high {
            let probe = (low + high) / 2;
            let other_key = scratch[(source + probe) as usize];
            let other_item = scratch[(source + count + probe) as usize];
            let rotation = parameters.settlement_rotation % max(count, 1);
            let priority = (item + count - rotation) % max(count, 1);
            let other_priority = (other_item + count - rotation) % max(count, 1);
            let precedes = other_key < key || (other_key == key && other_priority < priority);
            low = select(precedes, probe + 1, low);
            high = select(precedes, high, probe);
        }
        let target = destination + base + id - select(first, base, middle) + low - start;
        scratch[target as usize] = key;
        scratch[(target + count) as usize] = item;
    }
}

#[cube]
fn group_execute(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    proposals: &mut [u32],
    scratch: &mut [u32],
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
    id: u32,
) {
    let count = parameters.count;
    let active_count = markets::active(control, parameters);
    let mut invocation = id;
    if comptime!(parameters.discovery == Discovery::Indexed) {
        invocation = active_count;
        if id < control[RANGES_WORD].load() {
            invocation = scratch[(5 * count + id) as usize];
        }
    }
    let id = invocation;
    if id < active_count {
        let source = parameters.sort_source;
        let key = scratch[(source + id) as usize];
        let mut first = id == 0;
        if id > 0 {
            first = scratch[(source + id - 1) as usize] != key;
        }
        if first {
            let low = id;
            let mut cursor = low;
            let mut active = true;
            let mut event = Array::<u32>::new(EVENT_WORDS);
            let mut peer = PeerBalances::new(state, actors, config, parameters, key);
            while cursor < active_count && active {
                active = scratch[(source + cursor) as usize] == key;
                if active {
                    let actor = scratch[(source + count + cursor) as usize];
                    if comptime!(
                        parameters.market == MarketFlag::Payroll
                            || parameters.market == MarketFlag::Distribution
                    ) {
                        construct(
                            state,
                            actors,
                            config,
                            proposals,
                            &mut event[..],
                            control,
                            parameters,
                            actor,
                            comptime!(EventStorage::Local),
                        );
                        apply(state, &event[..], control, 0).store(scratch, 4 * count + actor);
                    } else {
                        let prior = KernelEventStatus::load(scratch, 4 * count + actor);
                        peer.offer(
                            state,
                            config,
                            proposals,
                            control,
                            parameters,
                            PeerOffer {
                                actor: actor + parameters.offset,
                                peer: key,
                                status: prior,
                            },
                        )
                        .store(scratch, 4 * count + actor);
                    }
                    cursor += 1;
                }
            }
            control[PROGRESS_WORD].fetch_max(cursor - low);
            control[COMPLETED_WORD].fetch_add(cursor - low);
            peer.store(state, actors, config, parameters, key);
        }
    }
}
#[cube]
fn index_count(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    spatial: &mut [u32],
    scratch: &mut [u32],
    parameters: &EconomyParameters,
    member: u32,
) {
    let cells = config.spatial_side * config.spatial_side;
    let segment = index_segment(config, parameters);
    let members = spatial[(segment + cells) as usize];
    let mut eligible = 0u32;
    if member < members {
        let firm = spatial[(segment + cells + 1 + member) as usize];
        eligible = u32::cast_from(discoverable(state, actors, config, parameters, firm));
    }
    let prefix = parallel::scan_prefix(eligible, comptime!(parameters.scan));
    if member < config.actors {
        scratch[member as usize] = prefix.inclusive;
    }
    let block = CUBE_POS as u32;
    if UNIT_POS == 0 && block < ceildiv(config.actors, CUBE_DIM) {
        scratch[(config.actors + block) as usize] = prefix.total;
    }
}

#[cube]
fn index_segment(config: &EconomyConfig, parameters: &EconomyParameters) -> u32 {
    let cells = config.spatial_side * config.spatial_side;
    (cells + 1 + config.actors)
        * comptime!(if parameters.market.good() == Some(Good::Bread) {
            1u32
        } else {
            2u32
        })
}

#[cube]
fn index_offset(config: &EconomyConfig, scratch: &mut [u32], members: u32) -> u32 {
    let mut offset = 0u32;
    if members > 0 {
        let member = members - 1;
        offset = scratch[member as usize];
        let block = member / CUBE_DIM;
        if block > 0 {
            offset += scratch[(config.actors + block - 1) as usize];
        }
    }
    offset
}

#[cube]
fn index_offsets(
    config: &EconomyConfig,
    spatial: &mut [u32],
    scratch: &mut [u32],
    parameters: &EconomyParameters,
    cell: u32,
) {
    let cells = config.spatial_side * config.spatial_side;
    if cell <= cells {
        let segment = index_segment(config, parameters);
        let member = spatial[(segment + cell) as usize];
        spatial[cell as usize] = index_offset(config, scratch, member);
    }
}

#[cube]
fn index_fill(
    config: &EconomyConfig,
    spatial: &mut [u32],
    scratch: &mut [u32],
    parameters: &EconomyParameters,
    member: u32,
) {
    let cells = config.spatial_side * config.spatial_side;
    let segment = index_segment(config, parameters);
    let members = spatial[(segment + cells) as usize];
    if member < members {
        let mut previous = 0u32;
        if UNIT_POS > 0 {
            previous = scratch[member as usize - 1];
        }
        if scratch[member as usize] > previous {
            let offset = index_offset(config, scratch, member + 1);
            spatial[(cells + offset) as usize] = spatial[(segment + cells + 1 + member) as usize];
        }
    }
}

#[cube]
fn index_prices(state: &mut [u32], config: &EconomyConfig, spatial: &mut [u32]) {
    let cells = config.spatial_side * config.spatial_side;
    let count = spatial[cells as usize];
    let block = CUBE_POS as u32;
    let mut quote = u32::new(comptime!(i64::from(u32::MAX)));
    let mut member = block * DISCOVERY_BLOCK + UNIT_POS;
    while member < min((block + 1) * DISCOVERY_BLOCK, count) {
        let actor = spatial[(cells + 1 + member) as usize];
        quote = min(
            quote,
            number(state, config, comptime!(Field::Price as u32), actor),
        );
        member += CUBE_DIM;
    }
    let minimum = parallel::reduce(quote, comptime!(Reduction::Minimum));
    if UNIT_POS == 0 && block * DISCOVERY_BLOCK < count {
        let base = 3 * (cells + 1 + config.actors);
        spatial[(base + block) as usize] = minimum;
    }
}

#[cube]
fn scaled(control: &[Atomic<u32>], value: Ticks, numerator: u32, denominator: u32) -> Ticks {
    let result = ratio(value, numerator, denominator);
    result.faults.record(control);
    result.value
}
#[allow(
    clippy::collapsible_if,
    reason = "Guard the storage read before indexing a partial workgroup"
)]
#[cube]
fn record_feedback(
    state: &mut [u32],
    config: &EconomyConfig,
    proposals: &mut [u32],
    outcomes: &mut [u32],
    feedback: &[Atomic<u32>],
    parameters: &EconomyParameters,
    id: u32,
) {
    if id < parameters.count && feedback[CONSTRUCTED_EVENTS_WORD].load() == 0 {
        let status = KernelEventStatus::load(outcomes, parameters.status_start + id);
        if status.contains(comptime!(EventStatus::Rejected)) {
            let actor = parameters.offset + id;
            if comptime!(parameters.market == MarketFlag::Household)
                && number(state, config, comptime!(Field::FoodAccess as u32), actor) == 0
            {
                set32(state, config, comptime!(Field::FoodAccess as u32), actor, 3);
            }
            let firm = proposals[actor as usize * ProposalLayout::WORDS + ProposalLayout::ACTOR];
            let base = CONTROL_WORDS as u32 + firm * TICK_WORDS as u32;
            let quantity =
                proposals[actor as usize * ProposalLayout::WORDS + ProposalLayout::QUANTITY_LOW];
            let old = feedback[base as usize].fetch_add(quantity);
            let carry = u32::cast_from(quantity > u32::MAX - old);
            let high = proposals
                [actor as usize * ProposalLayout::WORDS + ProposalLayout::QUANTITY_HIGH]
                + carry;
            let prior = feedback[(base + 1) as usize].fetch_add(high);
            if high > u32::MAX - prior {
                KernelFaults::new(comptime!(FaultFlags::DEMAND_OVERFLOW)).record(feedback);
            }
        }
    }
}
#[cube]
fn apply_feedback(state: &mut [u32], config: &EconomyConfig, feedback: &[Atomic<u32>], id: u32) {
    if id < config.actors {
        let base = CONTROL_WORDS as u32 + id * TICK_WORDS as u32;
        set(
            state,
            config,
            comptime!(Field::Unfilled as u32),
            id,
            ticks(
                feedback[base as usize].load(),
                feedback[(base + 1) as usize].load(),
            ),
        );
    }
}

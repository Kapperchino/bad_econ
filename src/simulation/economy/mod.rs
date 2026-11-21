pub(super) mod markets;
pub(super) mod scheduler;

use super::{
    flags::{
        ActorKind, COMPLETED_WORD, CONSTRUCTED_EVENTS_WORD, CONTROL_WORDS, EventStatus, FaultFlags,
        KernelActorKind, KernelAuthorization, KernelEventStatus, KernelFaults, MarketFlag,
        PROGRESS_WORD,
    },
    households::{KernelFatigue, KernelObservation, ObservationKind, next_fatigue, observed_blend},
    layout::*,
    markets::{LATCH_WORD, RANGE_SLOTS_WORD},
    negotiation::{NegotiationRequest, NegotiationStatus, legal_grid_rate, negotiate},
    parallel::{
        self, Reduction,
        workgroup::{self, TickTree, WorkgroupTree},
    },
    parameters::{Discovery, EconomyParameters},
    pass::GpuPass,
    payroll::PayrollAccrual,
    policy::*,
    scheduler::apply,
    system::{AsBufferArg, System, gpu_passes},
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
pub(super) mod batch;
mod clearing;
mod consumption;
mod employment;
pub(super) mod enterprises;
mod farming;
pub(in crate::simulation) mod food;
mod household_production;
pub(super) mod indexing;
pub(in crate::simulation) mod landlords;
pub(super) mod mobility;
pub(super) mod opening;
mod pricing;
pub(in crate::simulation) mod property;
pub(in crate::simulation) mod railway;
mod rent;
pub(super) mod search;
pub(super) mod sorting;
mod systems;
use super::scheduler::Posting;
use clearing::{PeerBalances, PeerOffer, actor_settlement};
pub(super) use systems::{EconomyInputLaunch, EconomySystem};

#[cube]
fn at(config: &EconomyConfig, field: u32, actor: u32) -> u32 {
    field * config.state_stride + actor
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
                    let cash = value(state, config, comptime!(Field::Cash as u32), id);
                    let rent = value(state, config, comptime!(Field::RentalIncome as u32), id);
                    set(
                        state,
                        config,
                        comptime!(Field::BusinessOpeningCash as u32),
                        id,
                        cash,
                    );
                    set(
                        state,
                        config,
                        comptime!(Field::BusinessOpeningRent as u32),
                        id,
                        rent,
                    );
                }
                if !k.is(comptime!(ActorKind::Household))
                    && k.bits() != comptime!(Good::Housing as u32)
                    && k.bits() != comptime!(Good::Railway as u32)
                {
                    if comptime!(config.home_mode == Feature::Enabled) {
                        enterprises::available_method(state, actors, config, control, id);
                    }
                    let mut demand = value(state, config, comptime!(Field::Demand as u32), id);
                    if k.bits() <= comptime!(Good::Bread as u32) {
                        demand = farming::production_target(control, demand);
                    }
                    let mut desired = proposed_starts(
                        KernelStarts {
                            coverage: u32::cast_from(config.coverage),
                            demand,
                            finished: value(state, config, k.output(), id),
                            pipeline: value(state, config, comptime!(Field::Pipeline as u32), id),
                        },
                        control,
                    );
                    if farming::farm(state, config, id, k.output()) {
                        desired =
                            farming::starts(state, config, control, id, k.output(), parameters.day);
                    }
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
                    let budget = employment::ShiftBudget::new(state, config, control, id);
                    let funded = divide(budget.cash, budget.pay);
                    planned = minimum(planned, funded);
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
                        let capacity = ticks(
                            actors[ActorLayout::CAPACITY * (actors.len() / ActorLayout::WORDS)
                                + id as usize],
                            0,
                        );
                        planned = minimum(minimum(shifts, funded), capacity);
                    }
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
                    set(state, config, comptime!(Field::Slots as u32), id, planned);
                    set(state, config, comptime!(Field::Plan as u32), id, planned);
                }
            }
            Phase::WorkPayment => {
                if k.is(comptime!(ActorKind::Household)) {
                    employment::finish_day(state, config, actors, id);
                }
                if number(state, config, comptime!(Field::EnterpriseModel as u32), id) == 1 {
                    let services = value(state, config, comptime!(Field::Pipeline as u32), id);
                    let services = property::scaled_output(state, config, control, id, services);
                    set(
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
                    if comptime!(config.property_mode == Feature::Disabled) {
                        enterprises::construct_home(state, config, control, id);
                    }
                    let rural = comptime!(config.economy == crate::domain::Economy::Agrarian)
                        || number(state, config, comptime!(Field::LandPlots as u32), id) > 0;
                    let ration = times32(
                        number(state, config, comptime!(Field::NutritionNeed as u32), id),
                        number(state, config, comptime!(Field::FoodPrice as u32), id),
                    );
                    let unfunded = lt(
                        value(state, config, comptime!(Field::Cash as u32), id),
                        ration,
                    );
                    if rural || unfunded {
                        household_production::produce(state, config, control, id, parameters.day);
                    }
                }
            }
            Phase::PhysicalClose => match k.bits() {
                ActorKind::HOUSEHOLD_BITS => {
                    let income = observed_blend(
                        value(state, config, comptime!(Field::IncomeForecast as u32), id),
                        KernelObservation {
                            kind: comptime!(ObservationKind::Observed as u32),
                            value: multiply(
                                control,
                                positive(
                                    value(state, config, comptime!(Field::GrossIncome as u32), id),
                                    value(
                                        state,
                                        config,
                                        comptime!(Field::OpeningGrossIncome as u32),
                                        id,
                                    ),
                                ),
                                POLICY_SCALE,
                            ),
                        },
                        POLICY_SCALE / 30,
                        control,
                    );
                    set(
                        state,
                        config,
                        comptime!(Field::IncomeForecast as u32),
                        id,
                        income,
                    );
                    if config.heating_per_person > 0 {
                        consumption::heat(state, config, control, id);
                    }
                    let bread = value(state, config, comptime!(Field::Bread as u32), id);
                    let minimum_ration =
                        value(state, config, comptime!(Field::NutritionNeed as u32), id);
                    let reserve =
                        food::consumption_reserve(state, config, control, id, parameters.day);
                    let available = maximum(minimum_ration, positive(bread, reserve));
                    let consumed = minimum(
                        minimum(bread, available),
                        value(state, config, comptime!(Field::FoodNeed as u32), id),
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
                                && ((parameters.day + id) % max(interval, 1) == 0
                                    || consumption::scaled_luxury(config, offset as u32 + 1))
                                && number(state, config, comptime!(Field::LivingMembers as u32), id)
                                    > 0
                            {
                                let good = offset as u32 + 1;
                                let stock = value(state, config, good, id);
                                let members = consumption::consumers(state, config, id);
                                let used = consumption::durable_use(
                                    config,
                                    members,
                                    stock,
                                    good,
                                    parameters.day + id,
                                );
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
                    let previous_fatigue =
                        number(state, config, comptime!(Field::Fatigue as u32), id);
                    let mut fatigue = min(previous_fatigue + config.work_fatigue, POLICY_SCALE);
                    if number(state, config, comptime!(Field::Hours as u32), id) == 0u32 {
                        fatigue =
                            max(previous_fatigue, config.rest_recovery) - config.rest_recovery;
                    }
                    if comptime!(config.household_policy == Feature::Enabled) {
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
                    let mut output = value(state, config, comptime!(Field::Pipeline as u32), id);
                    if farming::farm(state, config, id, k.output()) {
                        output = farming::harvest(
                            state,
                            config,
                            control,
                            id,
                            parameters.day,
                            output,
                            k.output(),
                        );
                    }
                    increase(
                        state,
                        config,
                        control,
                        comptime!(Field::UnscaledProduced as u32),
                        id,
                        output,
                    );
                    let output = property::scaled_output(state, config, control, id, output);
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
                if !k.is(comptime!(ActorKind::Household)) {
                    let business = u32::cast_from(k.bits() == comptime!(Good::Housing as u32))
                        + 2 * u32::cast_from(k.bits() == comptime!(Good::Railway as u32));
                    let revenue = match business {
                        2 => zero(),
                        1 => minus(
                            value(state, config, comptime!(Field::RentalIncome as u32), id),
                            value(
                                state,
                                config,
                                comptime!(Field::BusinessOpeningRent as u32),
                                id,
                            ),
                        ),
                        _ => multiply(
                            control,
                            value(state, config, comptime!(Field::Sales as u32), id),
                            number(state, config, comptime!(Field::Price as u32), id),
                        ),
                    };
                    let cash = value(state, config, comptime!(Field::Cash as u32), id);
                    set(
                        state,
                        config,
                        comptime!(Field::BusinessClosingCash as u32),
                        id,
                        cash,
                    );
                    let rail_revenue =
                        value(state, config, comptime!(Field::RailRevenue as u32), id);
                    set(
                        state,
                        config,
                        comptime!(Field::BusinessRevenue as u32),
                        id,
                        add_checked(control, revenue, rail_revenue),
                    );
                }
                if k.is(comptime!(ActorKind::Household))
                    && comptime!(config.home_mode == Feature::Enabled)
                {
                    enterprises::household_quote(state, config, control, id);
                }
                if !k.is(comptime!(ActorKind::Household))
                    && k.bits() != comptime!(Good::Housing as u32)
                    && k.bits() != comptime!(Good::Railway as u32)
                {
                    if comptime!(config.home_mode == Feature::Enabled) {
                        let packets = number(state, config, comptime!(Field::Plan as u32), id)
                            - number(state, config, comptime!(Field::Slots as u32), id);
                        enterprises::maintain(state, config, control, id, packets);
                    }
                    if comptime!(config.home_mode == Feature::Enabled)
                        || number(state, config, comptime!(Field::EnterpriseModel as u32), id)
                            == comptime!(
                                crate::domain::enterprise::EnterpriseModel::Distributor as u32
                            )
                    {
                        enterprises::lifecycle(state, actors, config, control, id, parameters.day);
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
                    let target = pricing::inventory_target(
                        state,
                        config,
                        control,
                        id,
                        k.output(),
                        demand,
                        parameters.day,
                    );
                    let excess = positive(value(state, config, k.output(), id), target);
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
                    let response = pricing::response(
                        state,
                        config,
                        control,
                        id,
                        k.output(),
                        excess,
                        config.price_table[table_index as usize],
                    );
                    pricing::adjust(state, config, control, id, response);
                    set(state, config, comptime!(Field::Demand as u32), id, demand);
                    enterprises::operating_plan(state, actors, config, control, id);

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

#[derive(CubeType)]
struct InputDemand {
    search: bool,
    wanted: Ticks,
}

#[cube]
impl InputDemand {
    fn new(
        state: &mut [u32],
        actors: &[u32],
        config: &EconomyConfig,
        control: &[Atomic<u32>],
        id: u32,
        k: &KernelActorKind,
        good: u32,
    ) -> Self {
        let mut search = !k.is(comptime!(ActorKind::Household))
            && (number(state, config, comptime!(Field::FirstInput as u32), id) == good
                || number(state, config, comptime!(Field::SecondInput as u32), id) == good);
        let mut wanted = ticks(
            actors[ActorLayout::CAPACITY * (actors.len() / ActorLayout::WORDS) + id as usize]
                * number(state, config, comptime!(Field::InputTicks as u32), id)
                * consumption::input_coverage(config, k.bits()),
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
                number(state, config, comptime!(Field::StockCoverage as u32), id) + config.ring,
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
            let capacity =
                actors[ActorLayout::CAPACITY * (actors.len() / ActorLayout::WORDS) + id as usize];
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
                || number(state, config, comptime!(Field::LandPlots as u32), id) > 0)
            && number(state, config, comptime!(Field::LivingMembers as u32), id) > 0
        {
            search = true;
            wanted = ticks(
                number(state, config, comptime!(Field::FoodNeed as u32), id)
                    * (config.coverage + config.ring),
                0,
            );
        }
        let investment = number(state, config, comptime!(Field::InvestmentGood as u32), id) > 0;
        if k.is(comptime!(ActorKind::Household))
            && investment
            && comptime!(config.property_mode == Feature::Disabled)
            && number(state, config, comptime!(Field::LivingMembers as u32), id) > 0
            && comptime!(config.home_mode == Feature::Enabled)
            && (good == comptime!(Good::Tools as u32) || good == comptime!(Good::Wood as u32))
        {
            search = true;
            wanted = ticks(select(good == comptime!(Good::Tools as u32), 2u32, 5u32), 0);
        }
        if !k.is(comptime!(ActorKind::Household))
            && comptime!(
                config.home_mode == Feature::Enabled && config.property_mode == Feature::Disabled
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
            let building = number(state, config, comptime!(Field::ProjectStatus as u32), id)
                == comptime!(crate::domain::property::ProjectStatus::Building as u32);
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
                wanted = add_checked(control, wanted, landlords::maintenance(state, config, id));
                search = true;
            }
        }
        let rail = railway::demand(state, actors, config, id, good);
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
fn propose(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    spatial: &mut [u32],
    proposals: &mut [u32],
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
    invocation: u32,
    #[comptime] labor_round: employment::LaborRound,
) -> u32 {
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
        let k = kind(actors, id);
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
                let labor = employment::LaborSearch::new(
                    state,
                    actors,
                    config,
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
                let demand = InputDemand::new(state, actors, config, control, id, &k, good);
                search = demand.search;
                wanted = demand.wanted;
            }
            MarketFlag::Consumer(_) => {
                good = parameters.traded_good();
                if good == comptime!(Good::Coal as u32) || good == comptime!(Good::Wood as u32) {
                    wanted = consumption::fuel_purchase_target(
                        state, config, control, parameters, id, good,
                    );
                    let held = add_checked(
                        control,
                        value(state, config, good, id),
                        value(state, config, TRANSIT_FIELDS as u32 + good - 1, id),
                    );
                    search = k.is(comptime!(ActorKind::Household))
                        && number(state, config, comptime!(Field::LivingMembers as u32), id) > 0
                        && !lt(
                            value(state, config, comptime!(Field::Bread as u32), id),
                            ticks(food::minimum_daily(state, config, id), 0),
                        )
                        && lt(held, wanted)
                        && !equal(purchasing_power, zero());
                    if search {
                        search = consumption::preferred_fuel(
                            state, actors, config, &*spatial, parameters, id, good,
                        ) == good;
                    }
                } else {
                    let interval = config.consumption_intervals[good as usize - 1];
                    let until_due =
                        (interval - (parameters.day + id) % max(interval, 1)) % max(interval, 1);
                    search = k.is(comptime!(ActorKind::Household))
                        && number(state, config, comptime!(Field::LivingMembers as u32), id) > 0
                        && (until_due <= config.coverage + config.ring
                            || consumption::scaled_luxury(config, good));
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
                        number(state, config, comptime!(Field::FoodPrice as u32), id),
                    );
                    search = search
                        && !lt(food, reserve)
                        && lt(
                            cash_reserve,
                            value(state, config, comptime!(Field::Cash as u32), id),
                        );
                    purchasing_power = consumption::budget(state, config, control, id, good);
                    search = search && !equal(purchasing_power, zero());
                    wanted = ticks(1, 0);
                    if search {
                        wanted = consumption::durable_target(
                            state,
                            actors,
                            config,
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
                    food::purchase_daily(state, config, parameters, id)
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
                                select(
                                    k.bits() == comptime!(Good::Housing as u32),
                                    zero(),
                                    ticks(config.reserve_low, config.reserve_high),
                                ),
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
        if comptime!(labor_round != employment::LaborRound::Retained) && search {
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
                    let begin = j;
                    let length = spatial[(cell + 1) as usize] - begin;
                    let limit = begin + search::supplier_samples(parameters, length);
                    let stride = search::supplier_stride(actors, parameters, id, length);
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
                                let route = match comptime!(market) {
                                    MarketFlag::Labor => {
                                        railway::animal_route(actors, config, id, firm)
                                    }
                                    _ => railway::route(actors, config, id, firm),
                                };
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

                                    let mut score =
                                        ticks(quote + route.days * config.delivery_penalty, 0u32);
                                    let amount = if comptime!(market == MarketFlag::Labor) {
                                        let offer = search::candidate(
                                            state,
                                            actors,
                                            config,
                                            control,
                                            parameters,
                                            id,
                                            firm,
                                            zero(),
                                            zero(),
                                        );
                                        eligible = offer.eligible;
                                        score = offer.score;
                                        offer.quantity
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
                                        let amount = minimum(
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
                                        score = search::supply_score(
                                            state, actors, config, parameters, id, wanted, amount,
                                            score,
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
    chosen
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
        let actor = id + parameters.offset;
        let other = proposals[actor as usize * ProposalLayout::WORDS + ProposalLayout::ACTOR];
        let mut path = railway::animal_route(actors, config, actor, actor);
        if parameters.traded_good() > 0 && other < config.actors {
            let planned = railway::animal_route(actors, config, actor, other);
            path.replace(&planned);
        }
        construct_route(
            state,
            actors,
            config,
            proposals,
            bundles,
            control,
            parameters,
            &path,
            id,
            comptime!(storage),
        );
    }
}

#[cube]
fn construct_route(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    proposals: &mut [u32],
    bundles: &mut [u32],
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
    route: &railway::FreightRoute,
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
                    let shifts = employment::shifts(config, quantity);
                    let labor = number(state, config, comptime!(Field::LaborMinutes as u32), other);
                    let mut offered_wage =
                        number(state, config, comptime!(Field::Wage as u32), other);
                    if comptime!(config.bargaining.enabled == Feature::Enabled) {
                        offered_wage = quantity.extract(LOW_WORD);
                    }
                    let wage = times32(offered_wage, labor * shifts);
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
                        ticks(shifts, 0u32),
                        zero,
                    );
                    event.transfer(
                        config,
                        bundles,
                        comptime!(Field::Time as u32),
                        actor,
                        ticks(
                            (labor + ceildiv(trip, config.commute_speed) * COMMUTE_LEGS) * shifts,
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
                            credit: ticks(shifts, 0u32),
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
                        ticks(labor * shifts, 0u32),
                    );
                    event.transfer(
                        config,
                        bundles,
                        comptime!(Field::Pipeline as u32),
                        other,
                        zero,
                        ticks(
                            number(state, config, comptime!(Field::OutputTicks as u32), other)
                                * shifts,
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
                            let amount = multiply(
                                control,
                                value(state, config, comptime!(Field::InputTicks as u32), other),
                                shifts,
                            );
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
                        let members = employment::payroll_members(actors, config, actor);
                        let before = PayrollAccrual::new(prior, members, config, control);
                        let after = PayrollAccrual::new(accrued, members, config, control);
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
                    let good = parameters.traded_good();
                    let cost = multiply(
                        control,
                        quantity,
                        number(state, config, comptime!(Field::Price as u32), other),
                    );
                    let freight = multiply(control, quantity, route.fee);
                    let slot = BASE_FIELDS as u32
                        + (good - 1) * config.ring
                        + (parameters.day + route.days) % config.ring;
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
                        freight,
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
                    if good == comptime!(Good::Bread as u32)
                        && kind(actors, actor).is(comptime!(ActorKind::Household))
                    {
                        event.transfer(
                            config,
                            bundles,
                            comptime!(Field::FoodPrice as u32),
                            actor,
                            value(state, config, comptime!(Field::FoodPrice as u32), actor),
                            value(state, config, comptime!(Field::Price as u32), other),
                        );
                    }
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
            let rent = value(state, config, comptime!(Field::RentTreasury as u32), actor);
            totals[SummaryLayout::RENT_COLLECTED as usize] = add_checked(
                control,
                totals[SummaryLayout::RENT_COLLECTED as usize],
                rent,
            );
            totals[SummaryLayout::RENT_ARREARS as usize] = add_checked(
                control,
                totals[SummaryLayout::RENT_ARREARS as usize],
                value(state, config, comptime!(Field::RentArrears as u32), actor),
            );
            totals[SummaryLayout::CASH as usize] = add_checked(
                control,
                totals[SummaryLayout::CASH as usize],
                value(state, config, comptime!(Field::RentPending as u32), actor),
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
                    employment::payroll_members(actors, config, actor),
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
                    totals[SummaryLayout::HOMELESS_HOUSEHOLDS as usize] = add_checked(
                        control,
                        totals[SummaryLayout::HOMELESS_HOUSEHOLDS as usize],
                        ticks(
                            u32::cast_from(
                                number(
                                    state,
                                    config,
                                    comptime!(Field::LivingMembers as u32),
                                    actor,
                                ) > 0
                                    && number(
                                        state,
                                        config,
                                        comptime!(Field::HousingStatus as u32),
                                        actor,
                                    ) == comptime!(crate::domain::HousingState::Homeless as u32),
                            ),
                            0,
                        ),
                    );
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
                    let members = consumption::consumers(state, config, actor);
                    let cash = divide(
                        value(state, config, comptime!(Field::Cash as u32), actor),
                        max(members, 1),
                    );
                    let mut bin = WORD_BITS - cash.extract(LOW_WORD).leading_zeros();
                    if cash.extract(HIGH_WORD) > 0u32 {
                        bin = TICK_BITS - cash.extract(HIGH_WORD).leading_zeros();
                    }
                    control[(SummaryLayout::WEALTH_START as u32
                        + min(bin, SummaryLayout::WEALTH_BINS as u32 - 1))
                        as usize]
                        .fetch_add(members);
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
                                ) == 1
                                    && number(
                                        state,
                                        config,
                                        comptime!(Field::EnterpriseStatus as u32),
                                        actor,
                                    ) != comptime!(
                                        crate::domain::enterprise::EnterpriseState::Closed as u32
                                    ),
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
                    let status = number(
                        state,
                        config,
                        comptime!(Field::EnterpriseStatus as u32),
                        actor,
                    );
                    let trading = status
                        > comptime!(crate::domain::enterprise::EnterpriseState::Forming as u32)
                        && status
                            < comptime!(
                                crate::domain::enterprise::EnterpriseState::Liquidating as u32
                            );
                    if trading {
                        let good = k.output() - 1;
                        let wage = value(state, config, comptime!(Field::Wage as u32), actor);
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
                        if pricing::stocked(state, config, actor, k.output()) {
                            let price = value(state, config, comptime!(Field::Price as u32), actor);
                            totals[(SummaryLayout::PRICE_COUNTS + good) as usize] = add_checked(
                                control,
                                totals[(SummaryLayout::PRICE_COUNTS + good) as usize],
                                ticks(1, 0),
                            );
                            totals[(SummaryLayout::PRICE_MINIMA + good) as usize] = minimum(
                                totals[(SummaryLayout::PRICE_MINIMA + good) as usize],
                                price,
                            );
                            totals[(SummaryLayout::PRICE_MAXIMA + good) as usize] = maximum(
                                totals[(SummaryLayout::PRICE_MAXIMA + good) as usize],
                                price,
                            );
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
                            totals[SummaryLayout::MINIMUM_PRICE as usize] =
                                minimum(totals[SummaryLayout::MINIMUM_PRICE as usize], price);
                            totals[SummaryLayout::MAXIMUM_PRICE as usize] =
                                maximum(totals[SummaryLayout::MAXIMUM_PRICE as usize], price);
                        }
                    }
                }
            }
            match k.bits() {
                ActorKind::HOUSEHOLD_BITS => {
                    let working = number(state, config, comptime!(Field::Worker as u32), actor)
                        != NO_ACTOR
                        && number(state, config, comptime!(Field::LivingMembers as u32), actor) > 0;
                    let subsistence =
                        comptime!(config.economy == crate::domain::Economy::Industrial)
                            && number(state, config, comptime!(Field::LandPlots as u32), actor) > 0;
                    let category = select(working, select(subsistence, 1u32, 0u32), 2u32);
                    let metric = (SummaryLayout::WORKFORCE + category) as usize;
                    if comptime!(config.cohorts == Feature::Enabled) {
                        let living =
                            number(state, config, comptime!(Field::LivingMembers as u32), actor);
                        let workers = min(
                            living,
                            number(
                                state,
                                config,
                                comptime!(Field::EligibleWorkers as u32),
                                actor,
                            ),
                        );
                        let inactive = comptime!(SummaryLayout::INACTIVE as usize);
                        totals[inactive] =
                            add_checked(control, totals[inactive], ticks(living - workers, 0));
                        totals[metric] = add_checked(control, totals[metric], ticks(workers, 0));
                    } else {
                        totals[metric] = add_checked(control, totals[metric], ticks(1, 0));
                    }
                    totals[SummaryLayout::RURAL_EMPLOYED as usize] = add_checked(
                        control,
                        totals[SummaryLayout::RURAL_EMPLOYED as usize],
                        select(
                            subsistence,
                            value(state, config, comptime!(Field::Employed as u32), actor),
                            zero(),
                        ),
                    );
                    totals[SummaryLayout::HOUSEHOLD_CASH as usize] = add_checked(
                        control,
                        totals[SummaryLayout::HOUSEHOLD_CASH as usize],
                        value(state, config, comptime!(Field::Cash as u32), actor),
                    );
                }
                _ => {
                    let good = k.output() - 1;
                    let status = number(
                        state,
                        config,
                        comptime!(Field::EnterpriseStatus as u32),
                        actor,
                    );
                    if status < 4 {
                        let category = select(
                            k.output() == comptime!(Good::Housing as u32),
                            2u32,
                            number(
                                state,
                                config,
                                comptime!(Field::EnterpriseModel as u32),
                                actor,
                            ),
                        );
                        totals[(SummaryLayout::ACTIVE_BY_GOOD + good) as usize] = add_checked(
                            control,
                            totals[(SummaryLayout::ACTIVE_BY_GOOD + good) as usize],
                            ticks(1, 0),
                        );
                        totals[(SummaryLayout::BUSINESS_TYPES + category) as usize] = add_checked(
                            control,
                            totals[(SummaryLayout::BUSINESS_TYPES + category) as usize],
                            ticks(1, 0),
                        );
                        totals[(SummaryLayout::BUSINESS_STATES + status) as usize] = add_checked(
                            control,
                            totals[(SummaryLayout::BUSINESS_STATES + status) as usize],
                            ticks(1, 0),
                        );
                    }
                    totals[(SummaryLayout::FIRM_CASH + good) as usize] = add_checked(
                        control,
                        totals[(SummaryLayout::FIRM_CASH + good) as usize],
                        value(state, config, comptime!(Field::Cash as u32), actor),
                    );
                    totals[(SummaryLayout::BUSINESS_OPENING + good) as usize] = add_checked(
                        control,
                        totals[(SummaryLayout::BUSINESS_OPENING + good) as usize],
                        value(
                            state,
                            config,
                            comptime!(Field::BusinessOpeningCash as u32),
                            actor,
                        ),
                    );
                    totals[(SummaryLayout::BUSINESS_CLOSING + good) as usize] = add_checked(
                        control,
                        totals[(SummaryLayout::BUSINESS_CLOSING + good) as usize],
                        value(
                            state,
                            config,
                            comptime!(Field::BusinessClosingCash as u32),
                            actor,
                        ),
                    );
                    totals[(SummaryLayout::BUSINESS_REVENUE + good) as usize] = add_checked(
                        control,
                        totals[(SummaryLayout::BUSINESS_REVENUE + good) as usize],
                        value(
                            state,
                            config,
                            comptime!(Field::BusinessRevenue as u32),
                            actor,
                        ),
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
#[repr(u64)]
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
    FirmRent,
    HouseholdRent,
    Propose,
    MarketActivity,
    Construct,
    Summarize,
    Inspect,
    GroupInitialize,
    GroupSort,
    GroupMerge,
    GroupExecute,
    GroupRemainder,
    GroupPrepare,
    GroupFinalize,
    MarketStart,
    InputMarketStart,
    InputMarketPrepare,
    InputMarketRepeat,
    MarketBoundary,
    GroupRanges,
    ProposeCompact,
    ProposeRetainedCompact,
    GroupCompactFill,
    GroupRadixLocal,
    GroupRadixScatter,
    InputQuotes,
}

#[cube]
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
                    | EconomyStage::ProposeCompact
                    | EconomyStage::ProposeRetainedCompact
                    | EconomyStage::GroupCompactFill
                    | EconomyStage::GroupRadixLocal
                    | EconomyStage::GroupRadixScatter
                    | EconomyStage::GroupExecute
                    | EconomyStage::GroupRemainder
                    | EconomyStage::GroupFinalize
            )
    );
    if !guarded || control[LATCH_WORD].load() == 0 {
        match stage {
            EconomyStage::InputQuotes => {
                pricing::refresh(state, actors, config, spatial, control, parameters, id)
            }
            EconomyStage::MarketStart
            | EconomyStage::InputMarketStart
            | EconomyStage::InputMarketPrepare
            | EconomyStage::InputMarketRepeat
            | EconomyStage::MarketBoundary => {
                if id == 0 {
                    control[LATCH_WORD].store(control[crate::simulation::flags::FAULT_WORD].load());
                    if comptime!(stage != EconomyStage::MarketBoundary) {
                        control[CONSTRUCTED_EVENTS_WORD].store(0);
                        control[RANGE_SLOTS_WORD].store(0);
                        control[markets::SEARCHES_WORD].store(0);
                        control[search::INPUT_MINIMUM_QUOTE_WORD].store(u32::MAX);
                        if comptime!(
                            parameters.empty_orders
                                == crate::simulation::parameters::EmptyOrderMode::FeedbackOnly
                        ) {
                            control[markets::WIDE_ORDERS_WORD].store(0);
                        }
                    }
                    if comptime!(matches!(
                        stage,
                        EconomyStage::InputMarketStart | EconomyStage::InputMarketPrepare
                    )) {
                        control[markets::PREPARATION_ACTORS_WORD].store(0);
                    }
                }
                if comptime!(matches!(
                    stage,
                    EconomyStage::InputMarketStart | EconomyStage::InputMarketRepeat
                )) && control[crate::simulation::flags::FAULT_WORD].load() == 0
                {
                    if id < parameters.count {
                        KernelEventStatus::new(comptime!(EventStatus::Inactive))
                            .store(bundles, 4 * parameters.count + id);
                    }
                }
            }
            EconomyStage::ProposeCompact | EconomyStage::ProposeRetainedCompact => {
                let actor = (id + parameters.settlement_rotation) % max(parameters.count, 1);
                let peer = propose(
                    state,
                    actors,
                    config,
                    spatial,
                    proposals,
                    control,
                    parameters,
                    select(id < parameters.count, actor, config.actors),
                    comptime!(match stage {
                        EconomyStage::ProposeRetainedCompact => employment::LaborRound::Retained,
                        _ => employment::LaborRound::Vacancies,
                    }),
                );
                markets::compact_proposal(peer, bundles, parameters, id);
            }
            EconomyStage::GroupCompactFill => {
                if control[crate::simulation::flags::FAULT_WORD].load() == 0 {
                    markets::compact_fill(proposals, bundles, control, parameters, id);
                }
            }
            EconomyStage::GroupRadixLocal => markets::radix_local(bundles, control, parameters, id),
            EconomyStage::GroupRadixScatter => {
                markets::radix_scatter(bundles, control, parameters, id)
            }
            EconomyStage::GroupRanges => {
                if control[crate::simulation::flags::FAULT_WORD].load() == 0 {
                    markets::ranges(bundles, control, config, parameters, id);
                }
            }
            EconomyStage::ClearFeedback => {
                if id < control.len() as u32 {
                    control[id as usize].store(0);
                }
            }
            EconomyStage::RecordFeedback => record_feedback(
                state, config, &*spatial, proposals, bundles, control, parameters, id,
            ),
            EconomyStage::ApplyFeedback => apply_feedback(state, config, control, id),
            EconomyStage::IndexCount => {
                index_count(state, actors, config, spatial, bundles, parameters, id)
            }
            EconomyStage::IndexOffsets => index_offsets(config, spatial, bundles, parameters, id),
            EconomyStage::IndexFill => index_fill(config, spatial, bundles, parameters, id),
            EconomyStage::IndexPrices => index_prices(state, config, spatial),
            EconomyStage::FirmRent => {
                rent::collect(state, actors, config, control, id, rent::Tenants::Firms)
            }
            EconomyStage::HouseholdRent => rent::collect(
                state,
                actors,
                config,
                control,
                id,
                rent::Tenants::Households,
            ),
            EconomyStage::Local => local(state, actors, config, control, parameters, id),
            EconomyStage::Propose => {
                propose(
                    state,
                    actors,
                    config,
                    spatial,
                    proposals,
                    control,
                    parameters,
                    id,
                    comptime!(employment::LaborRound::FromClock),
                );
            }
            EconomyStage::MarketActivity => {
                let mut active = 0u32;
                if id < config.actors {
                    active = u32::cast_from(
                        proposals[id as usize * ProposalLayout::WORDS + ProposalLayout::ACTOR]
                            != NO_ACTOR,
                    );
                }
                let total = parallel::reduce::<u32>(active, comptime!(Reduction::Sum));
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
            EconomyStage::GroupRemainder => group_remainder(
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
            let good = parameters.traded_good();
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
    let prefix = parallel::prefix_sum::<u32>(u32::cast_from(peer != NO_ACTOR));
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
    let range = id;
    let count = parameters.count;
    let active_count = markets::active(control, parameters);
    let mut invocation = id;
    if comptime!(parameters.discovery == Discovery::Indexed) {
        invocation = active_count;
        if id < control[RANGE_SLOTS_WORD].load() {
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
            let deferred = comptime!(
                parameters.market == MarketFlag::Household
                    && parameters.discovery == Discovery::Indexed
            );
            if deferred {
                active = peer.has_supply();
            }
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
                    if deferred {
                        active = peer.has_supply();
                    }
                }
            }
            if deferred {
                scratch[(7 * count + key) as usize] = cursor;
                cursor = low + scratch[(6 * count + range) as usize];
            }
            control[PROGRESS_WORD].fetch_max(cursor - low);
            control[COMPLETED_WORD].fetch_add(cursor - low);
            peer.store(state, actors, config, parameters, key);
        }
    }
}

#[cube]
fn group_remainder(
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
    if id < markets::active(control, parameters) {
        let key = scratch[(parameters.sort_source + id) as usize];
        if id >= scratch[(7 * count + key) as usize] {
            let actor = scratch[(parameters.sort_source + count + id) as usize];
            let prior = KernelEventStatus::load(scratch, 4 * count + actor);
            let mut peer = PeerBalances::new(state, actors, config, parameters, key);
            peer.offer(
                state,
                config,
                proposals,
                control,
                parameters,
                PeerOffer {
                    actor,
                    peer: key,
                    status: prior,
                },
            )
            .store(scratch, 4 * count + actor);
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
    (cells + 1 + config.actors) * select(parameters.traded_good() == Good::Bread as u32, 1u32, 2u32)
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
    let minimum = parallel::reduce::<u32>(quote, comptime!(Reduction::Minimum));
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
    requests: &[u32],
    proposals: &mut [u32],
    outcomes: &mut [u32],
    feedback: &[Atomic<u32>],
    parameters: &EconomyParameters,
    id: u32,
) {
    let count = match comptime!(parameters.discovery) {
        Discovery::Indexed => feedback[markets::FEEDBACK_EVENTS_WORD].load(),
        #[cfg(test)]
        _ => parameters.count,
    };
    if id < count && feedback[CONSTRUCTED_EVENTS_WORD].load() == 0 {
        let index = match comptime!(parameters.discovery) {
            Discovery::Indexed => {
                let source = feedback[markets::FEEDBACK_SOURCE_WORD].load();
                if source == comptime!(markets::FeedbackSource::Searches as u32) {
                    requests[id as usize]
                } else {
                    outcomes[(parameters.sort_source + parameters.count + id) as usize]
                }
            }
            #[cfg(test)]
            _ => id,
        };
        let actor = parameters.offset + index;
        let firm = proposals[actor as usize * ProposalLayout::WORDS + ProposalLayout::ACTOR];
        let status = KernelEventStatus::load(outcomes, parameters.status_start + index);
        if firm != NO_ACTOR && status.contains(comptime!(EventStatus::Rejected)) {
            if comptime!(parameters.market == MarketFlag::Household)
                && number(state, config, comptime!(Field::FoodAccess as u32), actor) == 0
            {
                set32(state, config, comptime!(Field::FoodAccess as u32), actor, 3);
            }
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

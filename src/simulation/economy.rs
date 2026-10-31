use super::{
    flags::{
        ActorKind, CONSTRUCTED_EVENTS_WORD, CONTROL_WORDS, EventStatus, FaultFlags,
        KernelActorKind, KernelAuthorization, KernelEventStatus, KernelFaults, MarketFlag,
    },
    households::{
        KernelCapacity, KernelFatigue, KernelHealth, KernelObservation, ObservationKind,
        Participation, available_minutes, next_fatigue, next_health, observed_blend,
    },
    layout::*,
    negotiation::{
        NegotiationRequest, NegotiationStatus, PolicyLayout, legal_grid_rate, negotiate,
    },
    parameters::EconomyParameters,
    payroll::PayrollAccrual,
    policy::*,
    words::*,
};
use crate::{
    domain::{BASE_FIELDS, Field, Good, Phase},
    ledger::{AuthorizationFlags, EMPTY, LEG_WORDS, LegLayout, MAX_LEGS},
};
use cubecl::prelude::*;

#[cube]
fn at(config: &[u32], field: u32, actor: u32) -> u32 {
    field * config[ConfigLayout::ACTORS] + actor
}

#[cube]
fn value(state: &mut [u32], config: &[u32], field: u32, actor: u32) -> Ticks {
    load(state, at(config, field, actor))
}

#[cube]
fn number(state: &mut [u32], config: &[u32], field: u32, actor: u32) -> u32 {
    value(state, config, field, actor).extract(LOW_WORD)
}

#[cube]
fn set(state: &mut [u32], config: &[u32], field: u32, actor: u32, number: Ticks) {
    store(state, at(config, field, actor), number);
}

#[cube]
fn set32(state: &mut [u32], config: &[u32], field: u32, actor: u32, number: u32) {
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
    config: &[u32],
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
    KernelActorKind::load(actors[actor as usize * ActorLayout::WORDS + ActorLayout::KIND])
}

#[cube]
fn distance(actors: &[u32], a: u32, b: u32) -> u32 {
    let ax = actors[a as usize * ActorLayout::WORDS + ActorLayout::X];
    let ay = actors[a as usize * ActorLayout::WORDS + ActorLayout::Y];
    let bx = actors[b as usize * ActorLayout::WORDS + ActorLayout::X];
    let by = actors[b as usize * ActorLayout::WORDS + ActorLayout::Y];
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
    config: &[u32],
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
    id: u32,
) {
    if id < config[ConfigLayout::ACTORS] {
        let k = kind(actors, id);
        let phase = comptime!(parameters.phase);
        match phase {
            Phase::Opening => {
                let slot = BASE_FIELDS as u32 + parameters.day % config[ConfigLayout::RING];
                let delivery = value(state, config, slot, id);
                match k.bits() {
                    ActorKind::HOUSEHOLD_BITS => {
                        increase(
                            state,
                            config,
                            control,
                            comptime!(Field::Bread as u32),
                            id,
                            delivery,
                        );
                    }
                    ActorKind::MILL_BITS | ActorKind::BAKERY_BITS => {
                        increase(state, config, control, k.input(), id, delivery);
                    }
                    _ => {}
                }
                if lt(
                    value(state, config, comptime!(Field::Transit as u32), id),
                    delivery,
                ) {
                    KernelFaults::new(comptime!(FaultFlags::TRANSIT_UNDERFLOW)).record(control);
                }
                let remaining = minus(
                    value(state, config, comptime!(Field::Transit as u32), id),
                    delivery,
                );
                set(
                    state,
                    config,
                    comptime!(Field::Transit as u32),
                    id,
                    remaining,
                );
                set32(state, config, slot, id, 0u32);
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
                    comptime!(Field::ResearchMinutes as u32),
                    id,
                    0u32,
                );
            }
            Phase::Observation => {
                if k.is(comptime!(ActorKind::Household)) {
                    let capacity = available_minutes(KernelCapacity {
                        participation: comptime!(Participation::Available as u32),
                        capacity_minutes: config[ConfigLayout::WORK_CAP],
                        health: number(state, config, comptime!(Field::Health as u32), id),
                        fatigue: number(state, config, comptime!(Field::Fatigue as u32), id),
                        reserved_minutes: config[ConfigLayout::RESERVED_MINUTES],
                    });
                    set32(state, config, comptime!(Field::Time as u32), id, capacity);
                }
            }
            Phase::Proposal => {
                if !k.is(comptime!(ActorKind::Household)) {
                    let desired = proposed_starts(
                        KernelStarts {
                            coverage: config[ConfigLayout::COVERAGE],
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
                                actors
                                    [id as usize * ActorLayout::WORDS + ActorLayout::OUTPUT_TICKS]
                                    - 1u32,
                                0u32,
                            ),
                        ),
                        actors[id as usize * ActorLayout::WORDS + ActorLayout::OUTPUT_TICKS],
                    );
                    let mut planned = minimum(
                        quotient,
                        ticks(
                            actors[id as usize * ActorLayout::WORDS + ActorLayout::CAPACITY],
                            0u32,
                        ),
                    );
                    if k.processes_input() {
                        planned = minimum(
                            planned,
                            divide(
                                value(state, config, k.input(), id),
                                actors[id as usize * ActorLayout::WORDS + ActorLayout::INPUT_TICKS],
                            ),
                        );
                    }
                    let mut planning_wage =
                        number(state, config, comptime!(Field::Wage as u32), id);
                    if config[ConfigLayout::BARGAINING + PolicyLayout::ENABLED] != 0 {
                        planning_wage =
                            legal_grid_rate(config, comptime!(ConfigLayout::BARGAINING));
                    }
                    planned = minimum(
                        planned,
                        divide(
                            value(state, config, comptime!(Field::Cash as u32), id),
                            planning_wage
                                * actors
                                    [id as usize * ActorLayout::WORDS + ActorLayout::LABOR_MINUTES],
                        ),
                    );
                    set(state, config, comptime!(Field::Slots as u32), id, planned);
                    set(state, config, comptime!(Field::Plan as u32), id, planned);
                }
            }
            Phase::PhysicalClose => match k.bits() {
                ActorKind::HOUSEHOLD_BITS => {
                    let consumed = minimum(
                        value(state, config, comptime!(Field::Bread as u32), id),
                        ticks(config[ConfigLayout::HOUSEHOLD_NEED], 0u32),
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
                    let shortage =
                        config[ConfigLayout::HOUSEHOLD_NEED] - consumed.extract(LOW_WORD);
                    set32(
                        state,
                        config,
                        comptime!(Field::Shortage as u32),
                        id,
                        shortage,
                    );
                    let mut health = min(
                        number(state, config, comptime!(Field::Health as u32), id)
                            + config[ConfigLayout::HEALTH_RECOVERY],
                        POLICY_SCALE,
                    );
                    if shortage > 0u32 {
                        health = max(
                            number(state, config, comptime!(Field::Health as u32), id),
                            config[ConfigLayout::HUNGER_DAMAGE],
                        ) - config[ConfigLayout::HUNGER_DAMAGE];
                    }
                    let mut fatigue = min(
                        number(state, config, comptime!(Field::Fatigue as u32), id)
                            + config[ConfigLayout::WORK_FATIGUE],
                        POLICY_SCALE,
                    );
                    if number(state, config, comptime!(Field::Employed as u32), id) == 0u32 {
                        fatigue = max(
                            number(state, config, comptime!(Field::Fatigue as u32), id),
                            config[ConfigLayout::REST_RECOVERY],
                        ) - config[ConfigLayout::REST_RECOVERY];
                    }
                    if config[ConfigLayout::HOUSEHOLD_POLICY_ENABLED] != 0 {
                        let previous_health =
                            number(state, config, comptime!(Field::Health as u32), id);
                        let previous_fatigue =
                            number(state, config, comptime!(Field::Fatigue as u32), id);
                        let deficit =
                            shortage * POLICY_SCALE / config[ConfigLayout::HOUSEHOLD_NEED];
                        health = next_health(KernelHealth {
                            health: previous_health,
                            fatigue: previous_fatigue,
                            deficit,
                            recovery: config[ConfigLayout::HOUSEHOLD_RECOVERY],
                            deficit_damage: config[ConfigLayout::DEFICIT_DAMAGE],
                            fatigue_damage: config[ConfigLayout::FATIGUE_DAMAGE],
                        });
                        let exertion = number(state, config, comptime!(Field::Hours as u32), id)
                            + number(state, config, comptime!(Field::ResearchMinutes as u32), id);
                        fatigue = next_fatigue(KernelFatigue {
                            fatigue: previous_fatigue,
                            response: config[ConfigLayout::FATIGUE_RESPONSE],
                            exertion_minutes: exertion,
                            rest_minutes: config[ConfigLayout::REST_MINUTES],
                        });
                    }
                    set32(state, config, comptime!(Field::Health as u32), id, health);
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
                if !k.is(comptime!(ActorKind::Household)) {
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
                        config[ConfigLayout::FORECAST_WEIGHT],
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
                    let scaled = fraction_index(magnitude, denominator)
                        * config[ConfigLayout::PRICE_RESPONSE]
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
                                config[(ConfigLayout::PRICE_TABLE as u32 + table_index) as usize],
                            ),
                            ticks(PRICE_SCALE - 1, 0u32),
                        ),
                        PRICE_SCALE,
                    );
                    set32(
                        state,
                        config,
                        comptime!(Field::Price as u32),
                        id,
                        min(
                            max(
                                candidate.extract(LOW_WORD),
                                config[ConfigLayout::MINIMUM_PRICE],
                            ),
                            config[ConfigLayout::MAXIMUM_PRICE],
                        ),
                    );
                    set(state, config, comptime!(Field::Demand as u32), id, demand);
                    let cap = divide(
                        divide(
                            value(state, config, comptime!(Field::Cash as u32), id),
                            actors[id as usize * ActorLayout::WORDS + ActorLayout::LABOR_MINUTES],
                        ),
                        max(
                            number(state, config, comptime!(Field::Plan as u32), id),
                            1u32,
                        ),
                    );
                    let offered = number(state, config, comptime!(Field::Wage as u32), id)
                        + select(
                            number(state, config, comptime!(Field::Slots as u32), id) > 0u32,
                            config[ConfigLayout::WAGE_INCREMENT],
                            0u32,
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
    config: &[u32],
    spatial: &mut [u32],
    proposals: &mut [u32],
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
    id: u32,
) {
    if id < config[ConfigLayout::ACTORS] {
        let market = comptime!(parameters.market);
        let k = kind(actors, id);
        let mut chosen = u32::new(comptime!(i64::from(NO_ACTOR)));
        let mut quantity = ticks(0u32, 0u32);
        let mut trip = 0u32;
        let mut search = false;
        let mut good = 0u32;
        let mut wanted = ticks(0u32, 0u32);
        match market {
            MarketFlag::Labor => {
                search = k.is(comptime!(ActorKind::Household))
                    && number(state, config, comptime!(Field::Employed as u32), id) == 0u32;
            }
            MarketFlag::GrainInput | MarketFlag::FlourInput | MarketFlag::BreadInput => {
                good = comptime!(market.good().expect("input market") as u32);
                search = k.processes_input() && k.input() == good;
                wanted = ticks(
                    actors[id as usize * ActorLayout::WORDS + ActorLayout::CAPACITY]
                        * actors[id as usize * ActorLayout::WORDS + ActorLayout::INPUT_TICKS]
                        * config[ConfigLayout::COVERAGE],
                    0u32,
                );
            }
            MarketFlag::Household => {
                good = comptime!(Good::Bread as u32);
                search = k.is(comptime!(ActorKind::Household));
                wanted = ticks(
                    config[ConfigLayout::HOUSEHOLD_NEED]
                        * (config[ConfigLayout::COVERAGE] + config[ConfigLayout::RING] - 1u32),
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
                            ticks(
                                config[ConfigLayout::RESERVE_LOW],
                                config[ConfigLayout::RESERVE_HIGH],
                            ),
                        ),
                        config[ConfigLayout::DISTRIBUTION_RATE],
                        POLICY_SCALE,
                    );
                    if !equal(quantity, ticks(0u32, 0u32)) {
                        chosen = actors[id as usize * ActorLayout::WORDS + ActorLayout::OWNER];
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
                    value(state, config, comptime!(Field::Transit as u32), id),
                ),
            );
        }
        search = search
            && number(state, config, comptime!(Field::Time as u32), id)
                >= select(comptime!(market == MarketFlag::Labor), 1u32, 0);
        if good > 0 {
            search = search
                && !equal(wanted, zero())
                && !equal(
                    value(state, config, comptime!(Field::Cash as u32), id),
                    zero(),
                );
        }
        if search {
            let cx = actors[id as usize * ActorLayout::WORDS + ActorLayout::X]
                / config[ConfigLayout::SEARCH_RADIUS];
            let cy = actors[id as usize * ActorLayout::WORDS + ActorLayout::Y]
                / config[ConfigLayout::SEARCH_RADIUS];
            let side = config[ConfigLayout::SPATIAL_SIDE];
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
                    while j < spatial[(cell + 1) as usize] && considered < DISCOVERY_CANDIDATES {
                        let firm = spatial[(side * side + 1u32 + j) as usize];
                        let dist = distance(actors, id, firm);
                        let mut wage = number(state, config, comptime!(Field::Wage as u32), firm);
                        let quote = number(state, config, comptime!(Field::Price as u32), firm);
                        let mut eligible = dist <= config[ConfigLayout::SEARCH_RADIUS];
                        let mut score = ticks(
                            quote
                                + max(ceildiv(dist, config[ConfigLayout::SHIPMENT_SPEED]), 1u32)
                                    * config[ConfigLayout::DELIVERY_PENALTY],
                            0u32,
                        );
                        let mut amount = ticks(1u32, 0u32);
                        if comptime!(market == MarketFlag::Labor) {
                            if config[ConfigLayout::BARGAINING + PolicyLayout::ENABLED] != 0 {
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
                                        payable_minutes: actors[firm as usize * ActorLayout::WORDS
                                            + ActorLayout::LABOR_MINUTES],
                                    },
                                    config,
                                    control,
                                    comptime!(ConfigLayout::BARGAINING),
                                );
                                eligible = eligible
                                    && negotiation.status
                                        == comptime!(NegotiationStatus::Agreed as u32);
                                wage = negotiation.wage.extract(LOW_WORD);
                                amount = negotiation.wage;
                            }
                            let duration = actors
                                [firm as usize * ActorLayout::WORDS + ActorLayout::LABOR_MINUTES]
                                + ceildiv(dist, config[ConfigLayout::COMMUTE_SPEED]) * COMMUTE_LEGS;
                            eligible = eligible
                                && number(state, config, comptime!(Field::Slots as u32), firm)
                                    > 0u32
                                && number(state, config, comptime!(Field::Time as u32), id)
                                    >= duration;
                            score = divide(
                                times32(
                                    wage * actors[firm as usize * ActorLayout::WORDS
                                        + ActorLayout::LABOR_MINUTES],
                                    POLICY_SCALE,
                                ),
                                duration,
                            );
                        } else {
                            amount = minimum(
                                minimum(wanted, value(state, config, good, firm)),
                                divide(
                                    value(state, config, comptime!(Field::Cash as u32), id),
                                    max(quote, 1u32),
                                ),
                            );
                            eligible = eligible
                                && kind(actors, firm).output() == good
                                && quote > 0u32
                                && !equal(amount, ticks(0u32, 0u32));
                        }
                        let mut improves = lt(score, best);
                        if comptime!(market == MarketFlag::Labor) {
                            improves = lt(best, score);
                        }
                        improves = improves || (equal(score, best) && firm < chosen);
                        considered += u32::cast_from(eligible);
                        if eligible && improves {
                            chosen = firm;
                            quantity = amount;
                            trip = dist;
                            best = score;
                        }
                        j += 1;
                    }
                    x += 1;
                }
                y += 1;
            }
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
    fn leg(&mut self, config: &[u32], bundles: &mut [u32], leg: KernelLeg) {
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
        config: &[u32],
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

#[cube]
fn construct(
    state: &mut [u32],
    actors: &[u32],
    config: &[u32],
    proposals: &mut [u32],
    bundles: &mut [u32],
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
    id: u32,
) {
    if id < parameters.count {
        for slot in 0u32..MAX_LEGS as u32 {
            for word in 0..LEG_WORDS {
                bundles[id as usize * EVENT_WORDS + slot as usize * LEG_WORDS + word] =
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
        let mut event = EventBuilder::new(id);
        if other != NO_ACTOR {
            control[CONSTRUCTED_EVENTS_WORD].fetch_add(1u32);
            match comptime!(parameters.market) {
                MarketFlag::Labor => {
                    let mut offered_wage =
                        number(state, config, comptime!(Field::Wage as u32), other);
                    if config[ConfigLayout::BARGAINING + PolicyLayout::ENABLED] != 0 {
                        offered_wage = quantity.extract(LOW_WORD);
                    }
                    let wage = times32(
                        offered_wage,
                        actors[other as usize * ActorLayout::WORDS + ActorLayout::LABOR_MINUTES],
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
                            actors
                                [other as usize * ActorLayout::WORDS + ActorLayout::LABOR_MINUTES]
                                + ceildiv(trip, config[ConfigLayout::COMMUTE_SPEED]) * COMMUTE_LEGS,
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
                            actors
                                [other as usize * ActorLayout::WORDS + ActorLayout::LABOR_MINUTES],
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
                            actors[other as usize * ActorLayout::WORDS + ActorLayout::OUTPUT_TICKS],
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
                    if kind(actors, other).processes_input() {
                        event.transfer(
                            config,
                            bundles,
                            kind(actors, other).input(),
                            other,
                            ticks(
                                actors[other as usize * ActorLayout::WORDS
                                    + ActorLayout::INPUT_TICKS],
                                0u32,
                            ),
                            zero,
                        );
                        event.transfer(
                            config,
                            bundles,
                            comptime!(Field::Used as u32),
                            other,
                            zero,
                            ticks(
                                actors[other as usize * ActorLayout::WORDS
                                    + ActorLayout::INPUT_TICKS],
                                0u32,
                            ),
                        );
                    }
                }
                MarketFlag::Payroll => {
                    let mut net = quantity;
                    if config[ConfigLayout::PAYROLL_ENABLED] != 0 {
                        let prior =
                            value(state, config, comptime!(Field::GrossIncome as u32), actor);
                        let accrued = add_checked(control, prior, quantity);
                        let before = PayrollAccrual::new(prior, config, control);
                        let after = PayrollAccrual::new(accrued, config, control);
                        net = minus(after.net, before.net);
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
                        + (parameters.day
                            + max(ceildiv(trip, config[ConfigLayout::SHIPMENT_SPEED]), 1u32))
                            % config[ConfigLayout::RING];
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
                    event.transfer(config, bundles, good, other, quantity, zero);
                    event.transfer(config, bundles, slot, actor, zero, quantity);
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
    config: &[u32],
    bundles: &mut [u32],
    control: &[Atomic<u32>],
    id: u32,
) {
    let groups = ceildiv(
        config[ConfigLayout::ACTORS],
        SummaryLayout::ACTORS_PER_GROUP,
    );
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
        for actor in id * SummaryLayout::ACTORS_PER_GROUP
            ..min(
                (id + 1u32) * SummaryLayout::ACTORS_PER_GROUP,
                config[ConfigLayout::ACTORS],
            )
        {
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
            if config[ConfigLayout::PAYROLL_ENABLED] != 0 {
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
            let mut input = 0u32;
            match k.bits() {
                ActorKind::HOUSEHOLD_BITS => {
                    input = k.input();
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
                    let x = actors[actor as usize * ActorLayout::WORDS + ActorLayout::X]
                        * SummaryLayout::REGION_SIDE as u32
                        / config[ConfigLayout::MAP_SIZE];
                    let y = actors[actor as usize * ActorLayout::WORDS + ActorLayout::Y]
                        * SummaryLayout::REGION_SIDE as u32
                        / config[ConfigLayout::MAP_SIZE];
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
                    if k.processes_input() {
                        input = k.input();
                    }
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
            if input > 0u32 {
                totals[(SummaryLayout::TRANSIT + input - 1u32) as usize] = add_checked(
                    control,
                    totals[(SummaryLayout::TRANSIT + input - 1u32) as usize],
                    value(state, config, comptime!(Field::Transit as u32), actor),
                );
                totals[(SummaryLayout::USED + input - 1u32) as usize] = add_checked(
                    control,
                    totals[(SummaryLayout::USED + input - 1u32) as usize],
                    value(state, config, comptime!(Field::Used as u32), actor),
                );
            }
            let mut transit = ticks(0u32, 0u32);
            for slot in 0u32..config[ConfigLayout::RING] {
                transit = add_checked(
                    control,
                    transit,
                    value(state, config, BASE_FIELDS as u32 + slot, actor),
                );
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

#[enumflags2::bitflags]
#[repr(u32)]
#[derive(Clone, Copy)]
enum ReductionKind {
    Sum,
    Minimum,
    Maximum,
}

#[derive(CubeType, Clone, Copy)]
struct SummaryReduction {
    kind: u32,
}
#[cube]
impl SummaryReduction {
    fn new(metric: u32) -> Self {
        let mut kind = u32::new(comptime!(i64::from(
            enumflags2::BitFlags::from_flag(ReductionKind::Sum).bits()
        )));
        if metric == SummaryLayout::MINIMUM_WAGE
            || metric == SummaryLayout::MINIMUM_PRICE
            || (metric >= SummaryLayout::WAGE_MINIMA
                && metric < SummaryLayout::WAGE_MINIMA + Good::COUNT as u32)
            || (metric >= SummaryLayout::PRICE_MINIMA
                && metric < SummaryLayout::PRICE_MINIMA + Good::COUNT as u32)
        {
            kind = comptime!(enumflags2::BitFlags::from_flag(ReductionKind::Minimum).bits());
        }
        if metric == SummaryLayout::MAXIMUM_WAGE
            || metric == SummaryLayout::MAXIMUM_PRICE
            || (metric >= SummaryLayout::WAGE_MAXIMA
                && metric < SummaryLayout::WAGE_MAXIMA + Good::COUNT as u32)
            || (metric >= SummaryLayout::PRICE_MAXIMA
                && metric < SummaryLayout::PRICE_MAXIMA + Good::COUNT as u32)
        {
            kind = comptime!(enumflags2::BitFlags::from_flag(ReductionKind::Maximum).bits());
        }
        SummaryReduction { kind }
    }
    fn is(&self, #[comptime] kind: ReductionKind) -> bool {
        self.kind == comptime!(enumflags2::BitFlags::from_flag(kind).bits())
    }
    fn combine(&self, control: &[Atomic<u32>], total: Ticks, part: Ticks) -> Ticks {
        let mut result = zero();
        if self.is(comptime!(ReductionKind::Minimum)) {
            result = minimum(total, part);
        }
        if self.is(comptime!(ReductionKind::Maximum)) {
            result = maximum(total, part);
        }
        if self.is(comptime!(ReductionKind::Sum)) {
            result = add_checked(control, total, part);
        }
        result
    }
}

#[cube]
fn reduce(
    config: &[u32],
    proposals: &mut [u32],
    bundles: &mut [u32],
    control: &[Atomic<u32>],
    id: u32,
) {
    if id < SummaryLayout::METRICS as u32 {
        let mut total = ticks(0u32, 0u32);
        let reduction = SummaryReduction::new(id);
        if reduction.is(comptime!(ReductionKind::Minimum)) {
            total = ticks(u32::MAX, u32::MAX);
        }
        for group in 0u32..ceildiv(
            config[ConfigLayout::ACTORS],
            SummaryLayout::ACTORS_PER_GROUP,
        ) {
            let base = (group * SummaryLayout::METRICS as u32 + id) * TICK_WORDS as u32;
            let part = ticks(bundles[(base) as usize], bundles[(base + 1u32) as usize]);
            total = reduction.combine(control, total, part);
        }
        proposals[(id * TICK_WORDS as u32) as usize] = total.extract(LOW_WORD);
        proposals[(id * TICK_WORDS as u32 + 1u32) as usize] = total.extract(HIGH_WORD);
    }
}

#[cube]
fn inspect(
    state: &mut [u32],
    config: &[u32],
    proposals: &mut [u32],
    parameters: &EconomyParameters,
    id: u32,
) {
    if id < config[ConfigLayout::FIELDS] {
        let item = value(state, config, id, parameters.offset);
        proposals[(id * TICK_WORDS as u32) as usize] = item.extract(LOW_WORD);
        proposals[(id * TICK_WORDS as u32 + 1u32) as usize] = item.extract(HIGH_WORD);
    }
}

#[cube]
fn copy_state(state: &mut [u32], proposals: &mut [u32], id: u32) {
    if id < (state.len() as u32 / TICK_WORDS as u32) {
        proposals[(id * TICK_WORDS as u32) as usize] = state[(id * TICK_WORDS as u32) as usize];
        proposals[(id * TICK_WORDS as u32 + 1u32) as usize] =
            state[(id * TICK_WORDS as u32 + HIGH_WORD as u32) as usize];
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
    IndexPrefix,
    IndexFill,
    Local,
    Propose,
    Construct,
    Summarize,
    Reduce,
    Inspect,
    CopyState,
}

#[cube(launch)]
pub fn economy(
    state: &mut [u32],
    actors: &[u32],
    config: &[u32],
    spatial: &mut [u32],
    proposals: &mut [u32],
    bundles: &mut [u32],
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
    #[comptime] stage: EconomyStage,
) {
    let id = ABSOLUTE_POS as u32;
    match stage {
        EconomyStage::ClearFeedback => {
            if id < control.len() as u32 {
                control[id as usize].store(0);
            }
        }
        EconomyStage::RecordFeedback => {
            record_feedback(config, proposals, bundles, control, parameters, id)
        }
        EconomyStage::ApplyFeedback => apply_feedback(state, config, control, id),
        EconomyStage::IndexCount => index_count(state, actors, config, spatial, parameters, id),
        EconomyStage::IndexPrefix => index_prefix(config, spatial, id),
        EconomyStage::IndexFill => index_fill(state, actors, config, spatial, parameters, id),
        EconomyStage::Local => local(state, actors, config, control, parameters, id),
        EconomyStage::Propose => propose(
            state, actors, config, spatial, proposals, control, parameters, id,
        ),
        EconomyStage::Construct => construct(
            state, actors, config, proposals, bundles, control, parameters, id,
        ),
        EconomyStage::Summarize => summarize(state, actors, config, bundles, control, id),
        EconomyStage::Reduce => reduce(config, proposals, bundles, control, id),
        EconomyStage::Inspect => inspect(state, config, proposals, parameters, id),
        EconomyStage::CopyState => copy_state(state, proposals, id),
    }
}

#[cube]
fn discoverable(
    state: &mut [u32],
    actors: &[u32],
    config: &[u32],
    parameters: &EconomyParameters,
    firm: u32,
) -> bool {
    let market = comptime!(parameters.market);
    match market {
        MarketFlag::Labor => number(state, config, comptime!(Field::Slots as u32), firm) > 0,
        MarketFlag::GrainInput
        | MarketFlag::FlourInput
        | MarketFlag::BreadInput
        | MarketFlag::Household => {
            let good = comptime!(market.good().expect("goods market") as u32);
            kind(actors, firm).output() == good && !equal(value(state, config, good, firm), zero())
        }
        MarketFlag::Payroll | MarketFlag::Distribution => false,
    }
}
#[cube]
fn firm_cell(actors: &[u32], config: &[u32], firm: u32) -> u32 {
    actors[firm as usize * ActorLayout::WORDS + ActorLayout::Y]
        / config[ConfigLayout::SEARCH_RADIUS]
        * config[ConfigLayout::SPATIAL_SIDE]
        + actors[firm as usize * ActorLayout::WORDS + ActorLayout::X]
            / config[ConfigLayout::SEARCH_RADIUS]
}
#[cube]
fn index_count(
    state: &mut [u32],
    actors: &[u32],
    config: &[u32],
    spatial: &mut [u32],
    parameters: &EconomyParameters,
    cell: u32,
) {
    if cell < config[ConfigLayout::SPATIAL_SIDE] * config[ConfigLayout::SPATIAL_SIDE] {
        let mut count = 0u32;
        for firm in config[ConfigLayout::HOUSEHOLDS]..config[ConfigLayout::ACTORS] {
            if firm_cell(actors, config, firm) == cell {
                count += u32::cast_from(discoverable(state, actors, config, parameters, firm));
            }
        }
        spatial[cell as usize] = count;
    }
}
#[cube]
fn index_prefix(config: &[u32], spatial: &mut [u32], id: u32) {
    if id == 0 {
        let mut offset = 0u32;
        for cell in 0..config[ConfigLayout::SPATIAL_SIDE] * config[ConfigLayout::SPATIAL_SIDE] {
            let count = spatial[cell as usize];
            spatial[cell as usize] = offset;
            offset += count;
        }
        spatial
            [(config[ConfigLayout::SPATIAL_SIDE] * config[ConfigLayout::SPATIAL_SIDE]) as usize] =
            offset;
    }
}
#[cube]
fn index_fill(
    state: &mut [u32],
    actors: &[u32],
    config: &[u32],
    spatial: &mut [u32],
    parameters: &EconomyParameters,
    cell: u32,
) {
    let cells = config[ConfigLayout::SPATIAL_SIDE] * config[ConfigLayout::SPATIAL_SIDE];
    if cell < cells {
        let mut offset = spatial[cell as usize];
        for firm in config[ConfigLayout::HOUSEHOLDS]..config[ConfigLayout::ACTORS] {
            if firm_cell(actors, config, firm) == cell
                && discoverable(state, actors, config, parameters, firm)
            {
                spatial[(cells + 1 + offset) as usize] = firm;
                offset += 1;
            }
        }
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
    config: &[u32],
    proposals: &mut [u32],
    outcomes: &mut [u32],
    feedback: &[Atomic<u32>],
    parameters: &EconomyParameters,
    id: u32,
) {
    if id < parameters.count {
        let status = KernelEventStatus::load(outcomes, parameters.status_start + id);
        if status.contains(comptime!(EventStatus::Rejected)) {
            let actor = parameters.offset + id;
            let firm = proposals[actor as usize * ProposalLayout::WORDS + ProposalLayout::ACTOR]
                - config[ConfigLayout::HOUSEHOLDS];
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
fn apply_feedback(state: &mut [u32], config: &[u32], feedback: &[Atomic<u32>], id: u32) {
    if id < config[ConfigLayout::FIRMS] {
        let base = CONTROL_WORDS as u32 + id * TICK_WORDS as u32;
        set(
            state,
            config,
            comptime!(Field::Unfilled as u32),
            config[ConfigLayout::HOUSEHOLDS] + id,
            ticks(
                feedback[base as usize].load(),
                feedback[(base + 1) as usize].load(),
            ),
        );
    }
}

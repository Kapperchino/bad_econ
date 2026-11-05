use super::*;
use crate::simulation::markets::{LATCH_WORD, SEARCHES_WORD};

#[cfg(test)]
mod tests;

#[cube]
pub(super) fn prepare(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    spatial: &mut [u32],
    proposals: &mut [u32],
    control: &[Atomic<u32>],
    searches: &mut [u32],
    parameters: &EconomyParameters,
    invocation: u32,
) {
    let mut id = invocation;
    let mut searching = false;
    if invocation < config.actors {
        let market = comptime!(parameters.market);

        if comptime!(market == MarketFlag::Household && parameters.discovery.indexed()) {
            let cells = config.spatial_side * config.spatial_side;
            let base =
                3 * (cells + 1 + config.actors) + ceildiv(config.actors, DISCOVERY_BLOCK) + 1;
            id = spatial[(base + invocation) as usize];
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
        searching = search;
        if searching {
            searches[(config.actors + id) as usize] = wanted.extract(LOW_WORD);
            searches[(2 * config.actors + id) as usize] = wanted.extract(HIGH_WORD);
            searches[(3 * config.actors + id) as usize] = purchasing_power.extract(LOW_WORD);
            searches[(4 * config.actors + id) as usize] = purchasing_power.extract(HIGH_WORD);
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
    let prefix = parallel::scan_prefix(u32::cast_from(searching), comptime!(parameters.scan));
    let mut base = Shared::<u32>::new();
    if UNIT_POS == 0 {
        *base = control[SEARCHES_WORD].fetch_add(prefix.total);
    }
    sync_cube();
    if searching {
        searches[(*base + prefix.inclusive - 1) as usize] = id;
    }
}

#[derive(CubeType, Clone, Copy)]
struct Candidate {
    eligible: bool,
    seen_supply: bool,
    empty: bool,
    quantity: Ticks,
    score: Ticks,
    trip: u32,
}

#[cube]
fn candidate(
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
    let market = comptime!(parameters.market);
    let good = comptime!(parameters.market.good().map_or(0, |good| good as u32));
    let mut eligible = false;
    let mut seen_supply = false;
    let mut empty_seller = u32::new(comptime!(i64::from(NO_ACTOR)));
    let mut empty_quantity = zero();
    let empty_score = ticks(u32::MAX, u32::MAX);
    let mut score = zero();
    let mut amount = zero();
    let dist = distance(actors, id, firm);
    if dist <= config.search_radius {
        let mut wage = number(state, config, comptime!(Field::Wage as u32), firm);
        let quote = number(state, config, comptime!(Field::Price as u32), firm);
        let mut inspect = true;
        if comptime!(market != MarketFlag::Labor) {
            let cash = purchasing_power;
            inspect = !lt(cash, ticks(max(quote, 1), 0));
            if !inspect {
                seen_supply =
                    seen_supply || household_production::offers(state, actors, config, firm, good);
            }
        }
        if inspect {
            eligible = true;

            score = ticks(
                quote + max(ceildiv(dist, config.shipment_speed), 1u32) * config.delivery_penalty,
                0u32,
            );
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
                let offered = household_production::offers(state, actors, config, firm, good);
                seen_supply = seen_supply || (eligible && offered);
                let known_producer = !kind(actors, firm).is(comptime!(ActorKind::Household))
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
                }
                amount = minimum(
                    minimum(
                        wanted,
                        positive(
                            value(state, config, good, firm),
                            household_production::reserve(state, actors, config, firm, good),
                        ),
                    ),
                    divide(purchasing_power, max(quote, 1u32)),
                );
                eligible = eligible
                    && firm != id
                    && household_production::offers(state, actors, config, firm, good)
                    && quote > 0u32
                    && !equal(amount, ticks(0u32, 0u32));
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

#[derive(CubeType, CubeLaunch)]
pub(in crate::simulation) struct SearchInput {
    pub state: Box<[u32]>,
    pub actors: Box<[u32]>,
    pub config: EconomyConfig,
    pub spatial: Box<[u32]>,
    pub proposals: Box<[u32]>,
    pub searches: Box<[u32]>,
    pub control: Box<[Atomic<u32>]>,
    pub parameters: EconomyParameters,
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
        search(
            &mut input.state,
            &input.actors,
            &input.config,
            &mut input.spatial,
            &mut input.proposals,
            &mut input.searches,
            &input.control,
            &input.parameters,
        );
    }
}

#[cube]
fn search(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    spatial: &mut [u32],
    proposals: &mut [u32],
    searches: &mut [u32],
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
                let mut j = begin;
                let nearest = max(px, x * radius) - min(px, (x + 1) * radius - 1)
                    + max(py, y * radius)
                    - min(py, (y + 1) * radius - 1);
                if nearest > radius {
                    j = end;
                }
                while j < end && considered < DISCOVERY_CANDIDATES {
                    let mut skipped = 0u32;
                    if comptime!(parameters.market.good() == Some(Good::Bread)) {
                        let rotated = begin + (j - begin + parameters.day + id) % max(length, 1);
                        let block = rotated / DISCOVERY_BLOCK;
                        let quote =
                            spatial[(3 * (side * side + 1 + config.actors) + block) as usize];
                        if lt(purchasing_power, ticks(quote, 0)) {
                            skipped =
                                min(min((block + 1) * DISCOVERY_BLOCK, end) - rotated, end - j);
                        }
                        let mut offset = 0u32;
                        while offset < skipped && !seen_supply {
                            let mut offered = 0u32;
                            if offset + UNIT_POS < skipped {
                                let seller = spatial
                                    [(side * side + 1 + rotated + offset + UNIT_POS) as usize];
                                offered = u32::cast_from(
                                    distance(actors, id, seller) <= radius
                                        && household_production::offers(
                                            state,
                                            actors,
                                            config,
                                            seller,
                                            comptime!(Good::Bread as u32),
                                        ),
                                );
                            }
                            seen_supply = parallel::reduce(offered, comptime!(Reduction::Sum)) > 0;
                            offset += 32;
                        }
                    }
                    if skipped == 0 {
                        let width = select(j == begin, 8u32, 32u32);
                        let lane = UNIT_POS as usize;
                        let fault_start = lane * CONTROL_WORDS;
                        if comptime!(parameters.market == MarketFlag::Labor) {
                            candidate_faults[fault_start + crate::simulation::flags::FAULT_WORD]
                                .store(0);
                        }
                        candidates[lane * 9] = 0;
                        candidates[lane * 9 + 1] = 0;
                        candidates[lane * 9 + 2] = 0;
                        if UNIT_POS < width && j + UNIT_POS < end {
                            let rotated = begin
                                + (j + UNIT_POS - begin + parameters.day + id) % max(length, 1);
                            let firm = spatial[(side * side + 1 + rotated) as usize];
                            let offer = candidate(
                                state,
                                actors,
                                config,
                                &candidate_faults[fault_start..fault_start + CONTROL_WORDS],
                                parameters,
                                id,
                                firm,
                                wanted,
                                purchasing_power,
                            );
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
                        sync_cube();
                        let mut cooperative = false;
                        if comptime!(
                            parameters.scan == crate::simulation::parallel::ScanMode::Plane
                        ) {
                            cooperative = PLANE_DIM == 32;
                            if cooperative {
                                let eligible = candidates[lane * 9];
                                let preceding = plane_exclusive_sum(eligible);
                                let visited = UNIT_POS < width
                                    && j + UNIT_POS < end
                                    && preceding < DISCOVERY_CANDIDATES - considered;
                                let included = u32::cast_from(visited);
                                let total = plane_sum(eligible * included);
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
                                while slot < min(width as usize, (end - j) as usize)
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
                                    considered += u32::cast_from(eligible);
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

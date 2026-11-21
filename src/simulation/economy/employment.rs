use super::*;
use crate::domain::{HiringTerm, enterprise::EnterpriseState};

#[derive(CubeType)]
pub(super) struct ShiftBudget {
    pub cash: Ticks,
    pub pay: u32,
}

#[cube]
impl ShiftBudget {
    pub fn new(
        state: &mut [u32],
        config: &EconomyConfig,
        control: &[Atomic<u32>],
        firm: u32,
    ) -> Self {
        let cash = positive(
            value(
                state,
                config,
                comptime!(Field::BusinessOpeningCash as u32),
                firm,
            ),
            rent::obligation(state, config, control, firm),
        );
        let wage = match comptime!(config.bargaining.enabled) {
            Feature::Enabled => legal_grid_rate(config),
            Feature::Disabled => number(state, config, comptime!(Field::Wage as u32), firm),
        };
        ShiftBudget {
            cash,
            pay: wage * number(state, config, comptime!(Field::LaborMinutes as u32), firm),
        }
    }
}

#[derive(CubeType)]
struct StandingJob {
    employer: u32,
    days: u32,
    active: bool,
}

#[cube]
impl StandingJob {
    fn load(state: &mut [u32], config: &EconomyConfig, worker: u32) -> Self {
        let employer = number(state, config, comptime!(Field::Job as u32), worker);
        let days = number(
            state,
            config,
            comptime!(Field::JobDaysRemaining as u32),
            worker,
        );
        let status = number(
            state,
            config,
            comptime!(Field::EnterpriseStatus as u32),
            max(employer, 1) - 1,
        );
        StandingJob {
            employer,
            days,
            active: employer > 0
                && days > 0
                && status < comptime!(EnterpriseState::Liquidating as u32),
        }
    }
}

#[derive(CubeType)]
pub(super) struct LaborSearch {
    pub peer: u32,
    pub quantity: Ticks,
    pub trip: u32,
    pub vacancies: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum LaborRound {
    Retained,
    Vacancies,
    FromClock,
}

#[cube]
impl LaborSearch {
    pub fn new(
        state: &mut [u32],
        actors: &[u32],
        config: &EconomyConfig,
        control: &[Atomic<u32>],
        parameters: &EconomyParameters,
        actor: u32,
        #[comptime] round: LaborRound,
    ) -> Self {
        let retained = match comptime!(round) {
            LaborRound::Retained => true,
            LaborRound::Vacancies => false,
            LaborRound::FromClock => parameters.matching_round == RETAINED_LABOR_ROUND as u32,
        };
        let job = StandingJob::load(state, config, actor);
        let available = kind(actors, actor).is(comptime!(ActorKind::Household))
            && number(state, config, comptime!(Field::Employed as u32), actor) == 0
            && number(state, config, comptime!(Field::Time as u32), actor) > 0;
        let rural = number(state, config, comptime!(Field::LandPlots as u32), actor) > 0;
        let industrial = comptime!(config.economy == crate::domain::Economy::Industrial);
        let open = available
            && !retained
            && (!industrial || !rural || parameters.matching_round + 1 == MATCHING_ROUNDS as u32);
        let firm = max(job.employer, 1) - 1;
        let short_time = if open
            && job.active
            && number(state, config, comptime!(Field::Plan as u32), firm) == 0
        {
            let budget = ShiftBudget::new(state, config, control, firm);
            !lt(budget.cash, ticks(budget.pay, 0))
        } else {
            false
        };
        let mut result = LaborSearch {
            peer: u32::new(comptime!(i64::from(NO_ACTOR))),
            quantity: zero(),
            trip: 0,
            vacancies: open && !short_time,
        };
        if retained
            && available
            && job.active
            && number(state, config, comptime!(Field::Slots as u32), firm) > 0
        {
            let offer = search::candidate(
                state,
                actors,
                config,
                control,
                parameters,
                actor,
                firm,
                zero(),
                zero(),
            );
            result.peer = select(offer.eligible, firm, result.peer);
            result.quantity = offer.quantity;
            result.trip = offer.trip;
        }
        result
    }
}

#[cube]
pub(super) fn finish_day(state: &mut [u32], config: &EconomyConfig, actors: &[u32], actor: u32) {
    let employer = number(state, config, comptime!(Field::Employer as u32), actor);
    let job = StandingJob::load(state, config, actor);
    let living = number(state, config, comptime!(Field::LivingMembers as u32), actor) > 0;
    let continuing = job.active && (employer == 0 || employer == job.employer);
    let mut days = select(continuing, job.days, 0);
    if employer > 0 && !continuing {
        let firm = employer - 1;
        let good = kind(actors, firm).output();
        let producer = number(
            state,
            config,
            comptime!(Field::EnterpriseModel as u32),
            firm,
        ) == 0;
        let farm_term = select(
            number(state, config, comptime!(Field::LandPlots as u32), actor) > 0,
            comptime!(HiringTerm::Day as u32),
            comptime!(HiringTerm::Year as u32),
        );
        let regular_term = select(
            producer && good == comptime!(Good::Coal as u32),
            comptime!(HiringTerm::Month as u32),
            comptime!(HiringTerm::Fortnight as u32),
        );
        days = select(
            farming::farm(state, config, firm, good),
            farm_term,
            regular_term,
        );
    }
    let remaining = select(living, max(days, 1) - 1, 0);
    let renewal = living && continuing;
    let current = select(employer > 0, employer, job.employer);
    set32(
        state,
        config,
        comptime!(Field::Job as u32),
        actor,
        select(remaining > 0 || renewal, current, 0),
    );
    set32(
        state,
        config,
        comptime!(Field::JobDaysRemaining as u32),
        actor,
        remaining,
    );
}

#[cube]
pub(super) fn team(state: &mut [u32], config: &EconomyConfig, actor: u32) -> u32 {
    match comptime!(config.cohorts) {
        Feature::Enabled => max(
            number(
                state,
                config,
                comptime!(Field::EligibleWorkers as u32),
                actor,
            ),
            1,
        ),
        Feature::Disabled => u32::new(1),
    }
}

#[cube]
pub(super) fn shifts(config: &EconomyConfig, quantity: Ticks) -> u32 {
    match comptime!(config.cohorts) {
        Feature::Enabled => max(quantity.extract(HIGH_WORD), 1),
        Feature::Disabled => u32::new(1),
    }
}

#[cube]
pub(super) fn payroll_members(actors: &[u32], config: &EconomyConfig, actor: u32) -> u32 {
    match comptime!(config.cohorts) {
        Feature::Enabled => max(
            actors[ActorLayout::CAPACITY * (actors.len() / ActorLayout::WORDS) + actor as usize],
            1,
        ),
        Feature::Disabled => u32::new(1),
    }
}

use super::*;

#[derive(CubeType)]
pub(super) struct LaborSearch {
    pub peer: u32,
    pub quantity: Ticks,
    pub trip: u32,
    pub vacancies: bool,
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
    ) -> Self {
        let retained = parameters.matching_round == RETAINED_LABOR_ROUND as u32;
        let job = number(state, config, comptime!(Field::Job as u32), actor);
        let term = number(
            state,
            config,
            comptime!(Field::JobDaysRemaining as u32),
            actor,
        );
        let available = kind(actors, actor).is(comptime!(ActorKind::Household))
            && number(state, config, comptime!(Field::Employed as u32), actor) == 0
            && number(state, config, comptime!(Field::Time as u32), actor) > 0;
        let rural = number(state, config, comptime!(Field::LandPlots as u32), actor) > 0;
        let industrial = comptime!(config.economy == crate::domain::Economy::Industrial);
        let contracted = industrial
            && job > 0
            && term > 0
            && number(
                state,
                config,
                comptime!(Field::EnterpriseStatus as u32),
                max(job, 1) - 1,
            ) < 4;
        let mut result = LaborSearch {
            peer: u32::new(comptime!(i64::from(NO_ACTOR))),
            quantity: zero(),
            trip: 0,
            vacancies: available
                && !contracted
                && !retained
                && (!industrial
                    || !rural
                    || parameters.matching_round + 1 == MATCHING_ROUNDS as u32),
        };
        if retained && available && contracted {
            let offer = search::candidate(
                state,
                actors,
                config,
                control,
                parameters,
                actor,
                job - 1,
                zero(),
                zero(),
            );
            result.peer = select(offer.eligible, job - 1, result.peer);
            result.quantity = offer.quantity;
            result.trip = offer.trip;
        }
        result
    }
}

#[cube]
pub(super) fn finish_day(
    state: &mut [u32],
    config: &EconomyConfig,
    parameters: &EconomyParameters,
    actor: u32,
) {
    if comptime!(config.economy == crate::domain::Economy::Industrial) {
        let employer = number(state, config, comptime!(Field::Employer as u32), actor);
        let job = number(state, config, comptime!(Field::Job as u32), actor);
        let prior = number(
            state,
            config,
            comptime!(Field::JobDaysRemaining as u32),
            actor,
        );
        let retained = job > 0
            && prior > 0
            && number(state, config, comptime!(Field::LivingMembers as u32), actor) > 0
            && number(
                state,
                config,
                comptime!(Field::EnterpriseStatus as u32),
                max(job, 1) - 1,
            ) < 4;
        let current = select(employer > 0, employer, select(retained, job, 0));
        let term = select(
            current == job && retained,
            prior,
            parameters.employment_term_days,
        );
        let remaining = select(current > 0, max(term, 1) - 1, 0);
        set32(
            state,
            config,
            comptime!(Field::Job as u32),
            actor,
            select(remaining > 0, current, 0),
        );
        set32(
            state,
            config,
            comptime!(Field::JobDaysRemaining as u32),
            actor,
            remaining,
        );
        if comptime!(config.cohorts == Feature::Disabled) && retained && employer == 0 {
            set32(state, config, comptime!(Field::Employer as u32), actor, job);
            set32(state, config, comptime!(Field::Employed as u32), actor, 1);
        }
    }
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

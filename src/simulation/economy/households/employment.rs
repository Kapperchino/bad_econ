use crate::domain::{HiringTerm, enterprise::EnterpriseState};
use crate::simulation::economy::*;

#[derive(CubeType)]
pub(in crate::simulation::economy) struct ShiftBudget {
    pub cash: Ticks,
    pub pay: u32,
}

#[cube]
impl ShiftBudget {
    pub fn new(ledger: &Ledger<'_>, control: &[Atomic<u32>], firm: u32) -> Self {
        let config = ledger.config;
        let cash = positive(
            ledger.value(Field::BusinessOpeningCash, firm),
            rent::obligation(ledger, control, firm),
        );
        let wage = match comptime!(config.bargaining.enabled) {
            Feature::Enabled => legal_grid_rate(config),
            Feature::Disabled => ledger.number(Field::Wage, firm),
        };
        ShiftBudget {
            cash,
            pay: wage * ledger.number(Field::LaborMinutes, firm),
        }
    }
}

#[derive(CubeType)]
struct StandingJob {
    employer: u32,
    days: u32,
    active: bool,
}

#[repr(u32)]
enum OutsideWork {
    NewContract,
    Unavailable,
    Temporary,
}

#[cube]
impl StandingJob {
    fn load(ledger: &Ledger<'_>, worker: u32) -> Self {
        let employer = ledger.number(Field::Job, worker);
        let days = ledger.number(Field::JobDaysRemaining, worker);
        let status = ledger.number(Field::EnterpriseStatus, max(employer, 1) - 1);
        StandingJob {
            employer,
            days,
            active: employer > 0
                && days > 0
                && status < comptime!(EnterpriseState::Liquidating as u32),
        }
    }

    fn outside_work(&self, economy: &EconomyView<'_>, control: &[Atomic<u32>]) -> u32 {
        let config = economy.ledger.config;
        let firm = max(self.employer, 1) - 1;
        let mut work = u32::new(comptime!(OutsideWork::NewContract as i64));
        if self.active && economy.ledger.number(Field::Plan, firm) == 0 {
            let budget = ShiftBudget::new(&economy.ledger, control, firm);
            let seasonal = config.home.harvest_days < 365
                && farming::FarmSite::new(firm, economy.actors.kind(firm).output())
                    .is_farm(&economy.ledger);
            let funded = select(
                seasonal,
                comptime!(OutsideWork::Temporary as u32),
                comptime!(OutsideWork::Unavailable as u32),
            );
            work = select(lt(budget.cash, ticks(budget.pay, 0)), work, funded);
        }
        work
    }
}

#[derive(CubeType)]
pub(in crate::simulation::economy) struct LaborSearch {
    pub peer: u32,
    pub quantity: Ticks,
    pub trip: u32,
    pub vacancies: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(in crate::simulation::economy) enum LaborRound {
    Retained,
    Vacancies,
    FromClock,
}

#[cube]
impl LaborSearch {
    pub fn new(
        economy: &EconomyView<'_>,
        control: &[Atomic<u32>],
        parameters: &EconomyParameters,
        actor: u32,
        #[comptime] round: LaborRound,
    ) -> Self {
        let config = economy.ledger.config;
        let retained = match comptime!(round) {
            LaborRound::Retained => true,
            LaborRound::Vacancies => false,
            LaborRound::FromClock => parameters.matching_round == RETAINED_LABOR_ROUND as u32,
        };
        let job = StandingJob::load(&economy.ledger, actor);
        let available = economy
            .actors
            .kind(actor)
            .is(comptime!(ActorKind::Household))
            && economy.ledger.number(Field::Employed, actor) == 0
            && economy.ledger.number(Field::Time, actor) > 0;
        let rural = economy.ledger.number(Field::LandPlots, actor) > 0;
        let industrial = comptime!(config.economy == crate::domain::Economy::Industrial);
        let open = available
            && !retained
            && (!industrial || !rural || parameters.matching_round + 1 == MATCHING_ROUNDS as u32);
        let firm = max(job.employer, 1) - 1;
        let vacancies = if open {
            job.outside_work(economy, control) != comptime!(OutsideWork::Unavailable as u32)
        } else {
            false
        };
        let mut result = LaborSearch {
            peer: u32::new(comptime!(i64::from(NO_ACTOR))),
            quantity: zero(),
            trip: 0,
            vacancies,
        };
        if retained && available && job.active && economy.ledger.number(Field::Slots, firm) > 0 {
            let offer = search::OfferRequest::labor(economy, actor, firm)
                .evaluate(economy, control, parameters);
            result.peer = select(offer.eligible, firm, result.peer);
            result.quantity = offer.quantity;
            result.trip = offer.trip;
        }
        result
    }
}

#[cube]
pub(in crate::simulation::economy) fn finish_day(
    economy: &mut EconomyView<'_>,
    control: &[Atomic<u32>],
    actor: u32,
) {
    let employer = economy.ledger.number(Field::Employer, actor);
    let job = StandingJob::load(&economy.ledger, actor);
    let living = economy.ledger.number(Field::LivingMembers, actor) > 0;
    let continuing = job.active
        && (employer == 0
            || employer == job.employer
            || job.outside_work(&*economy, control) == comptime!(OutsideWork::Temporary as u32));
    let mut days = select(continuing, job.days, 0);
    if employer > 0 && !continuing {
        let firm = employer - 1;
        let good = economy.actors.kind(firm).output();
        let producer = economy.ledger.number(Field::EnterpriseModel, firm) == 0;
        let farm_term = select(
            economy.ledger.number(Field::LandPlots, actor) > 0,
            comptime!(HiringTerm::Day as u32),
            comptime!(HiringTerm::Year as u32),
        );
        let regular_term = select(
            producer && good == comptime!(Good::Coal as u32),
            comptime!(HiringTerm::Month as u32),
            comptime!(HiringTerm::Fortnight as u32),
        );
        days = select(
            farming::FarmSite::new(firm, good).is_farm(&economy.ledger),
            farm_term,
            regular_term,
        );
    }
    let remaining = select(living, max(days, 1) - 1, 0);
    let renewal = living && continuing;
    let current = select(continuing, job.employer, employer);
    economy.ledger.set_number(
        Field::Job,
        actor,
        select(remaining > 0 || renewal, current, 0),
    );
    economy
        .ledger
        .set_number(Field::JobDaysRemaining, actor, remaining);
}

#[cube]
pub(in crate::simulation::economy) fn team(ledger: &Ledger<'_>, actor: u32) -> u32 {
    let config = ledger.config;
    match comptime!(config.cohorts) {
        Feature::Enabled => max(ledger.number(Field::EligibleWorkers, actor), 1),
        Feature::Disabled => u32::new(1),
    }
}

#[cube]
pub(in crate::simulation::economy) fn shifts(config: &EconomyConfig, quantity: Ticks) -> u32 {
    match comptime!(config.cohorts) {
        Feature::Enabled => max(quantity.extract(HIGH_WORD), 1),
        Feature::Disabled => u32::new(1),
    }
}

#[cube]
pub(in crate::simulation::economy) fn payroll_members(
    actors: &ActorTable<'_>,
    config: &EconomyConfig,
    actor: u32,
) -> u32 {
    match comptime!(config.cohorts) {
        Feature::Enabled => max(actors.capacity(actor), 1),
        Feature::Disabled => u32::new(1),
    }
}

#[cube]
pub(in crate::simulation::economy) fn adjust_wage(ledger: &mut Ledger<'_>, firm: u32) {
    let config = ledger.config;
    let increment = config.wage_increment;
    if increment > 0 {
        let cap = divide(
            divide(
                ledger.value(Field::Cash, firm),
                ledger.number(Field::LaborMinutes, firm),
            ),
            max(ledger.number(Field::Plan, firm), 1u32),
        );
        let wage = ledger.number(Field::Wage, firm);
        let unstaffed = ledger.number(Field::Slots, firm) > 0;
        let offered = select(
            unstaffed,
            wage + increment,
            max(wage, increment + MINIMUM_WAGE) - increment,
        );
        ledger.set_number(
            Field::Wage,
            firm,
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

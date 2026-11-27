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

#[repr(u32)]
enum WageChange {
    Hold,
    Cut,
    Raise,
}

#[cube]
fn standing_wage(ledger: &Ledger<'_>, firm: u32) -> u32 {
    max(
        ledger.number(Field::UncappedWage, firm),
        ledger.number(Field::Wage, firm),
    )
}

#[cube]
fn offer_wage(ledger: &mut Ledger<'_>, firm: u32, rate: u32, cap: Ticks) {
    let offered = max(minimum(ticks(rate, 0), cap).extract(LOW_WORD), MINIMUM_WAGE);
    if comptime!(ledger.config.economy == crate::domain::Economy::Industrial) {
        ledger.set_number(Field::UncappedWage, firm, select(offered < rate, rate, 0));
    }
    ledger.set_number(Field::Wage, firm, offered);
}

#[cube]
pub(in crate::simulation::economy) fn post_wage(
    ledger: &mut Ledger<'_>,
    control: &[Atomic<u32>],
    firm: u32,
) {
    let config = ledger.config;
    let minutes = ledger.number(Field::LaborMinutes, firm);
    if comptime!(config.economy == crate::domain::Economy::Industrial)
        && config.wage_increment > 0
        && minutes > 0
    {
        let cash = positive(
            ledger.value(Field::Cash, firm),
            rent::obligation(&*ledger, control, firm),
        );
        let rate = standing_wage(&*ledger, firm);
        offer_wage(ledger, firm, rate, divide(cash, minutes));
    }
}

#[cube]
pub(in crate::simulation::economy) fn bid_for_vacancies(
    ledger: &mut Ledger<'_>,
    control: &[Atomic<u32>],
    firm: u32,
    good: u32,
    day: u32,
    previous_plan: u32,
    previous_vacancies: u32,
) {
    let config = ledger.config;
    let plan = ledger.number(Field::Plan, firm);
    let increment = config.wage_increment;
    if comptime!(config.economy == crate::domain::Economy::Industrial)
        && increment > 0
        && previous_vacancies > 0
        && plan > previous_plan - previous_vacancies
    {
        let minutes = ledger.number(Field::LaborMinutes, firm);
        let wage = min(standing_wage(&*ledger, firm) + increment, MAXIMUM_WAGE);
        let budget = ShiftBudget {
            cash: positive(
                ledger.value(Field::Cash, firm),
                rent::obligation(&*ledger, control, firm),
            ),
            pay: wage * minutes,
        };
        let funded =
            farming::FarmProduction::new(ledger, control, farming::FarmSite::new(firm, good), day)
                .funded_shifts(&budget);
        let rise = times32(increment * minutes, plan);
        let spare = positive(
            ledger.value(Field::Cash, firm),
            ledger.value(Field::OperatingReserve, firm),
        );
        if !lt(spare, rise) && !lt(funded, ticks(plan, 0)) {
            ledger.set_number(Field::UncappedWage, firm, 0);
            ledger.set_number(Field::Wage, firm, wage);
        }
    }
}

#[cube]
pub(in crate::simulation::economy) fn adjust_wage(
    ledger: &mut Ledger<'_>,
    control: &[Atomic<u32>],
    firm: u32,
    good: u32,
    day: u32,
    excess: Ticks,
) {
    let config = ledger.config;
    let increment = config.wage_increment;
    if increment > 0 {
        let plan = ledger.number(Field::Plan, firm);
        let industrial = comptime!(config.economy == crate::domain::Economy::Industrial);
        let mut cash = ledger.value(Field::Cash, firm);
        let mut cut = true;
        if comptime!(config.economy == crate::domain::Economy::Industrial) {
            cash = positive(cash, rent::obligation(&*ledger, control, firm));
            let seasonal_idle = plan == 0
                && config.home.harvest_days < 365
                && farming::FarmSite::new(firm, good).is_farm(&*ledger);
            let batch = ledger.value(Field::OutputTicks, firm);
            cut = ledger.number(Field::Slots, firm) == 0
                && (day + firm) % 30 == 0
                && !seasonal_idle
                && !equal(excess, zero())
                && equal(ledger.value(Field::Unfilled, firm), zero())
                && lt(ledger.value(Field::Sales, firm), batch)
                && lt(ledger.value(Field::Demand, firm), batch);
        }
        let cap = divide(
            divide(cash, ledger.number(Field::LaborMinutes, firm)),
            select(industrial, 1, max(plan, 1)),
        );
        let wage = if comptime!(config.economy == crate::domain::Economy::Industrial) {
            standing_wage(&*ledger, firm)
        } else {
            ledger.number(Field::Wage, firm)
        };
        let change = select(
            !industrial && ledger.number(Field::Slots, firm) > 0,
            comptime!(WageChange::Raise as u32),
            select(
                cut,
                comptime!(WageChange::Cut as u32),
                comptime!(WageChange::Hold as u32),
            ),
        );
        let offered = match change {
            1 => max(wage, increment + MINIMUM_WAGE) - increment,
            2 => wage + increment,
            _ => wage,
        };
        let rate = min(offered, MAXIMUM_WAGE);
        offer_wage(ledger, firm, rate, cap);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{Checkpoint, Experiment, Market, Scenario};
    use crate::simulation::{Simulation, tests::device};
    use anyhow::{Context, Result};

    #[derive(Clone, Copy, Debug)]
    enum WageDecision {
        Hiring,
        Review { days_late: u32 },
    }

    #[derive(Clone, Copy, Debug)]
    struct WageCase {
        decision: WageDecision,
        good: Good,
        changes: &'static [(Field, u64)],
        wage: u32,
    }

    #[test]
    fn industrial_wage_recovers_from_a_temporary_cash_shortfall_on_gpu() -> Result<()> {
        let scenario = Scenario::industrial(24, 42, Experiment::Concentrated)?;
        let device = device(64)?;
        let mut initial = scenario.initial()?;
        let mill = initial
            .actors
            .iter()
            .position(|actor| actor.kind == Good::Steel as u32)
            .context("generated steel mill is missing")?;
        let minutes = initial.get(Field::LaborMinutes, mill);
        let initial_cash = initial.get(Field::Cash, mill);
        let arrears = 30;
        let cash = minutes * 5
            + u64::from(scenario.config().rent.map_or(0, |rent| rent.firm_daily))
            + arrears;
        let transferred = initial_cash - cash;
        initial.set(Field::Wage, mill, 17);
        initial.set(Field::Cash, mill, cash);
        initial.set(Field::Cash, 0, initial.get(Field::Cash, 0) + transferred);
        initial.set(Field::RentArrears, mill, arrears);
        let mut sim =
            Simulation::from_checkpoint(Checkpoint::new(&scenario, &initial)?, device.clone())?;
        let posting = sim.parameters(Phase::Proposal, Market::Labor, 0, initial.actors.len())?;
        sim.economy(
            EconomyStage::Local,
            initial.actors.len(),
            &posting,
            &sim.work.proposals,
        );
        sim.fault_check()?;
        let mut stressed = sim.download()?;
        assert_eq!(stressed.get(Field::Wage, mill), 5);
        assert_eq!(stressed.get(Field::UncappedWage, mill), 17);
        stressed.set(Field::Cash, mill, initial_cash);
        stressed.set(Field::Cash, 0, stressed.get(Field::Cash, 0) - transferred);
        stressed.set(Field::Slots, mill, 0);
        let saved = serde_json::to_vec(&Checkpoint::new(&scenario, &stressed)?)?;
        let mut restored = Simulation::from_checkpoint(serde_json::from_slice(&saved)?, device)?;
        restored.economy(
            EconomyStage::Local,
            initial.actors.len(),
            &posting,
            &restored.work.proposals,
        );
        restored.fault_check()?;
        let recovered = restored.download()?;
        assert_eq!(recovered.get(Field::Wage, mill), 17);
        assert_eq!(recovered.get(Field::UncappedWage, mill), 0);
        Ok(())
    }

    #[test]
    fn industrial_wage_reviews_respect_orders_seasons_and_saved_increments_on_gpu() -> Result<()> {
        let generated = Scenario::industrial(24, 42, Experiment::Concentrated)?;
        assert_eq!(generated.config().behavior.wage_increment, 1);
        let device = device(64)?;
        for increment in [1, 0, 3] {
            let mut config = generated.config().clone();
            config.behavior.wage_increment = increment;
            config.forecast_weight = 0;
            let scenario = Scenario::new(config)?;
            let opening = scenario.initial()?;
            let mut sim =
                Simulation::from_checkpoint(Checkpoint::new(&scenario, &opening)?, device.clone())?;
            let funded = WageCase {
                decision: WageDecision::Hiring,
                good: Good::Steel,
                changes: &[],
                wage: 12 + increment,
            };
            let cut = WageCase {
                decision: WageDecision::Review { days_late: 0 },
                wage: 12 - increment,
                ..funded
            };
            let shortfall = WageCase {
                changes: &[
                    (Field::Plan, 0),
                    (Field::Cash, 140),
                    (Field::RentArrears, 30),
                ],
                wage: if increment == 0 { 12 } else { 5 },
                ..cut
            };
            let hiring_hold = WageCase { wage: 12, ..funded };
            let review_hold = WageCase { wage: 12, ..cut };
            let mut cases = vec![funded, cut];
            match increment {
                1 => cases.extend([
                    shortfall,
                    WageCase {
                        changes: &[(Field::Slots, 0)],
                        ..hiring_hold
                    },
                    WageCase {
                        changes: &[(Field::Steel, 10_000)],
                        ..hiring_hold
                    },
                    WageCase {
                        changes: &[(Field::Plan, 1000)],
                        ..hiring_hold
                    },
                    WageCase {
                        changes: &[(Field::OperatingReserve, 1_000_000)],
                        ..hiring_hold
                    },
                    WageCase {
                        decision: WageDecision::Review { days_late: 1 },
                        ..review_hold
                    },
                    WageCase {
                        changes: &[(Field::Slots, 1)],
                        ..review_hold
                    },
                    WageCase {
                        changes: &[(Field::Sales, 10)],
                        ..review_hold
                    },
                    WageCase {
                        changes: &[(Field::Demand, 10)],
                        ..review_hold
                    },
                    WageCase {
                        changes: &[(Field::Unfilled, 1)],
                        ..review_hold
                    },
                    WageCase {
                        changes: &[(Field::Steel, 0)],
                        ..review_hold
                    },
                    WageCase {
                        good: Good::Flax,
                        changes: &[(Field::Plan, 0)],
                        ..review_hold
                    },
                    WageCase {
                        changes: &[
                            (Field::Slots, 1),
                            (Field::Cash, 300),
                            (Field::RentArrears, 90),
                        ],
                        wage: 10,
                        ..review_hold
                    },
                ]),
                0 => cases.push(shortfall),
                _ => (),
            }
            for case in cases {
                let mut initial = opening.clone();
                let firm = initial
                    .actors
                    .iter()
                    .position(|actor| actor.kind == case.good as u32)
                    .context("generated wage-test firm is missing")?;
                let defaults: &[(Field, u64)] = match case.decision {
                    WageDecision::Hiring => &[
                        (Field::Slots, 1),
                        (Field::Plan, 1),
                        (Field::Cash, 1_000_000),
                        (Field::OperatingReserve, 0),
                        (Field::Steel, 0),
                        (Field::Demand, 10),
                        (Field::Iron, 1000),
                        (Field::Coal, 1000),
                    ],
                    WageDecision::Review { .. } => &[
                        (Field::Cash, 100_000),
                        (Field::Plan, 2),
                        (Field::Slots, 0),
                        (Field::Demand, 0),
                        (Field::OutputTicks, 10),
                        (Field::LaborMinutes, 20),
                        (Field::inventory(case.good), 10_000),
                    ],
                };
                initial.set(Field::Wage, firm, 12);
                for &(field, value) in defaults.iter().chain(case.changes) {
                    initial.set(field, firm, value);
                }
                let rows = initial.actors.len();
                let stride = sim.config.state_stride as usize;
                let values = crate::simulation::tensor::state_values(&initial.values, rows, stride);
                sim.device.write(&sim.state, bytemuck::cast_slice(&values));
                let params = match case.decision {
                    WageDecision::Hiring => {
                        sim.parameters(Phase::Proposal, Market::Labor, 0, rows)?
                    }
                    WageDecision::Review { days_late } => {
                        let mut params =
                            sim.parameters(Phase::AccountsDecisions, Market::Labor, 0, rows)?;
                        params.day = 330 - firm as u32 % 30 + days_late;
                        params
                    }
                };
                sim.economy(EconomyStage::Local, rows, &params, &sim.work.proposals);
                sim.fault_check()?;
                let word = (Field::Wage as usize * stride + firm) * TICK_WORDS + LOW_WORD;
                assert_eq!(
                    sim.device.read_range(&sim.state, word, 1)?[0],
                    case.wage,
                    "{case:?}, increment {increment}"
                );
            }
        }
        Ok(())
    }
}

use super::{
    DemographyPolicy, PersonLayout as P, PopulationModel, Stage,
    systems::{DemographyInput, DemographyInputExpand},
};
use crate::simulation::economy::{
    configuration::Feature,
    households::food::{BreadDemand, HomeKitchen},
};
use crate::{
    domain::{
        Field,
        demography::{DAYS_PER_YEAR, PROBABILITY_SCALE},
    },
    simulation::storage::{state::*, words::*},
};
use cubecl::prelude::*;

#[cube]
fn band(age: u32) -> u32 {
    let mut result = 4;
    if age < 65 * DAYS_PER_YEAR {
        result = 3;
    }
    if age < 45 * DAYS_PER_YEAR {
        result = 2;
    }
    if age < 15 * DAYS_PER_YEAR {
        result = 1;
    }
    if age < DAYS_PER_YEAR {
        result = 0;
    }
    result
}

#[cube]
fn probability_draw(high: f32, low: f32) -> Ticks {
    let high = u64::cast_from(high * 16777216.0) * comptime!(PROBABILITY_SCALE);
    let low = u64::cast_from(low * 16777216.0) * comptime!(PROBABILITY_SCALE);
    let value = (high >> 24) + ((((high & 0xffffffu64) << 24) + low) >> 48);
    ticks(value as u32, (value >> 32) as u32)
}

#[cube]
pub(super) fn rate(annual_ppm: u32) -> Ticks {
    let daily = u64::cast_from(annual_ppm) * 1_000_000u64 / comptime!(DAYS_PER_YEAR as u64);
    ticks(u32::cast_from(daily), u32::cast_from(daily >> 32))
}

#[repr(u32)]
enum FamilySupport {
    Secure,
    Unfunded,
    Deprived,
    Unsettled,
}

#[cube]
pub(super) fn family_support(ledger: &Ledger<'_>, household: u32, food: u32) -> u32 {
    let config = ledger.config;
    let mut support = u32::new(comptime!(FamilySupport::Secure as i64));
    if comptime!(config.economy == crate::domain::Economy::Industrial) {
        let plots = ledger.number(Field::LandPlots, household);
        let bread = ledger
            .stock_with_transit(comptime!(crate::domain::Good::Bread as u32), household)
            .value;
        let flour = minimum(
            times32(food, 90),
            ledger
                .stock_with_transit(comptime!(crate::domain::Good::Flour as u32), household)
                .value,
        );
        let baked = select(
            HomeKitchen::available(ledger, household),
            HomeKitchen::new(ledger, household).bread_from_flour(flour),
            zero(),
        );
        let reserves = divide(plus(minimum(bread, times32(food, 90)), baked).value, 90);
        let purchased = positive(ticks(food, 0), reserves);
        let food_cost = times(
            purchased,
            max(ledger.number(Field::FoodPrice, household), 1),
        )
        .value;
        let cost = plus(
            food_cost,
            times32(
                select(plots > 0, 0, config.household_rent),
                select(
                    comptime!(config.cohorts == Feature::Enabled),
                    ledger.number(Field::LivingMembers, household),
                    1,
                ),
            ),
        )
        .value;
        let income = divide(ledger.value(Field::IncomeForecast, household), 1000);
        let savings = divide(ledger.value(Field::Cash, household), 90);
        let available = plus(income, savings).value;
        support = select(
            lt(available, cost),
            comptime!(FamilySupport::Unfunded as u32),
            support,
        );
        let arrears = ledger.value(Field::RentArrears, household);
        let housing = ledger.number(Field::HousingStatus, household);
        let unsettled = !equal(arrears, zero())
            || housing == comptime!(crate::domain::HousingState::Homeless as u32);
        support = select(
            unsettled,
            comptime!(FamilySupport::Unsettled as u32),
            support,
        );
    }
    support
}

#[derive(CubeType)]
pub(super) struct OccupationalExposure {
    pub minutes: u32,
    pub condition: u32,
    pub safety: u32,
    pub fatigue: u32,
    pub method: u32,
}

#[cube]
impl OccupationalExposure {
    pub(super) fn unprotected(minutes: u32, fatigue: u32) -> Self {
        OccupationalExposure {
            minutes,
            condition: 1000,
            safety: 0,
            fatigue,
            method: 0,
        }
    }

    pub(super) fn risk(&self, rate_ppm: u32, reference: u32) -> Ticks {
        let base = divide(
            times(times32(rate_ppm, 1_000_000), self.minutes).value,
            reference,
        );
        let equipment = 1000 + (1000 - min(self.condition, 1000)) * 2;
        let tired = 1000 + self.fatigue + (max(self.minutes, 480) - 480) * 1000 / 480;
        let mechanized = select(self.method > 0, 1250u32, 1000u32);
        let maintained = ratio(base, equipment, 1000).value;
        let exertion = ratio(maintained, tired, 1000).value;
        let method_risk = ratio(exertion, mechanized, 1000).value;
        ratio(method_risk, 1000 - min(self.safety, 900), 1000).value
    }
}

#[cube]
pub(super) fn famine_exposure(prior: u32, deficit: u32, recovery: u32) -> u32 {
    let reduction = recovery * (1000 - deficit) / 1000;
    min(max(prior + deficit, reduction) - reduction, 365_000)
}

#[cube]
pub(super) fn famine_risk(exposure: u32, daily_ppm: u32, vulnerability: u32) -> Ticks {
    let severity = min(exposure, 90_000);
    let base = ratio(times32(daily_ppm, 1_000_000), severity, 30_000).value;
    let duration = ratio(base, severity, 30_000).value;
    ratio(duration, vulnerability, 1000).value
}

#[cube]
pub(super) fn cold_exposure(prior: u32, deficit: u32, recovery: u32) -> u32 {
    select(
        deficit > 0,
        min(prior + deficit, 365_000),
        max(prior, recovery) - recovery,
    )
}

#[cube]
pub(super) fn cold_risk(exposure: u32, daily_ppm: u32, vulnerability: u32) -> Ticks {
    let mut risk = zero();
    if exposure > 0 {
        let severity = min(exposure, 28_000);
        let base = ratio(times32(daily_ppm, 1_000_000), severity, 7_000).value;
        let duration = ratio(base, severity, 7_000).value;
        risk = ratio(duration, vulnerability, 1000).value;
    }
    risk
}

#[derive(CubeType)]
struct MortalityRisks {
    background: Ticks,
    occupational: Ticks,
    famine: Ticks,
    cold: Ticks,
}

#[cube]
impl MortalityRisks {
    fn cause(&self, draw: Ticks) -> u32 {
        let first = self.background;
        let second = plus(first, self.occupational).value;
        let third = plus(second, self.famine).value;
        let total = minimum(
            plus(third, self.cold).value,
            ticks(
                comptime!(PROBABILITY_SCALE as u32),
                comptime!((PROBABILITY_SCALE >> 32) as u32),
            ),
        );
        let mut cause = 0;
        if lt(draw, total) {
            cause = 4;
        }
        if lt(draw, third) {
            cause = 3;
        }
        if lt(draw, second) {
            cause = 2;
        }
        if lt(draw, first) {
            cause = 1;
        }
        cause
    }
}

#[derive(CubeType)]
struct HouseholdProfile {
    living: u32,
    demand: u64,
    worker: u32,
    available: u32,
    workers: u32,
    shortest_day: u32,
    household_time: u32,
}

#[derive(CubeType)]
struct PersonDay<'a> {
    words: &'a mut [u32],
    base: usize,
}

#[cube]
impl<'a> PersonDay<'a> {
    fn new(words: &'a mut [u32], index: u32) -> Self {
        PersonDay::<'a> {
            words,
            base: index as usize * P::WORDS,
        }
    }

    fn number(&self, #[comptime] field: usize) -> u32 {
        self.words[self.base + field]
    }

    fn set(&mut self, #[comptime] field: usize, value: u32) {
        self.words[self.base + field] = value;
    }

    fn identity(&self) -> Ticks {
        read_tick_words(&*self.words, self.base + P::IDENTITY)
    }
}

#[derive(CubeType)]
struct HouseholdDay<'a> {
    economy: EconomyView<'a>,
    people: &'a mut [u32],
    policy: &'a DemographyPolicy,
    births: &'a mut [u32],
    samples: &'a [f32],
    day: u32,
    household: u32,
    first: u32,
    count: u32,
}

#[cube]
impl<'a> HouseholdDay<'a> {
    fn new(input: &'a mut DemographyInput, household: u32) -> Self {
        HouseholdDay::<'a> {
            economy: EconomyView::new(&mut input.state, &input.actors, &input.config),
            people: &mut input.people,
            policy: &input.policy,
            births: &mut input.births,
            samples: &input.samples,
            day: input.day,
            household,
            first: input.ranges[household as usize * 2],
            count: input.ranges[household as usize * 2 + 1],
        }
    }

    fn profile(
        &mut self,
        #[comptime] stage: Stage,
        #[comptime] model: PopulationModel,
    ) -> HouseholdProfile {
        let config = self.economy.ledger.config;
        let mut profile = HouseholdProfile {
            living: 0,
            demand: 0,
            worker: u32::MAX,
            available: 0,
            workers: 0,
            shortest_day: config.work_cap,
            household_time: 0,
        };
        if comptime!(stage == Stage::Observation) {
            self.births[self.household as usize] = 0;
            self.economy
                .ledger
                .set_number(Field::SplitPerson, self.household, 0);
        }
        if comptime!(model == PopulationModel::Cohorts) {
            profile.living = self
                .economy
                .ledger
                .number(Field::LivingMembers, self.household);
            for band in 0..5u32 {
                profile.demand += u64::cast_from(
                    self.economy
                        .ledger
                        .number_at(comptime!(Field::Infants as u32) + band, self.household),
                ) * u64::cast_from(self.policy.food[band as usize]);
            }
            profile.demand *= u64::cast_from(config.household_need);
            profile.workers = self
                .economy
                .ledger
                .number(Field::YoungAdults, self.household)
                + self
                    .economy
                    .ledger
                    .number(Field::OlderAdults, self.household);
            profile.shortest_day = min(
                config.work_cap
                    * (1000 - self.economy.ledger.number(Field::Fatigue, self.household))
                    / 1000,
                1440 - config.reserved_minutes,
            );
            profile.household_time = profile.workers * profile.shortest_day;
            profile.worker = select(profile.workers > 0, self.household, u32::MAX);
            profile.workers = select(profile.shortest_day > 0, profile.workers, 0);
        }
        for offset in 0..self.count {
            let index = self.first + (offset + self.day % max(self.count, 1)) % max(self.count, 1);
            let mut person = PersonDay::new(self.people, index);
            if comptime!(stage == Stage::Observation) {
                person.set(P::EMPLOYER, 0);
                person.set(P::WORK, 0);
            }
            if person.number(P::CAUSE) == 0 {
                let age = person.number(P::AGE);
                profile.living += 1;
                profile.demand +=
                    u64::cast_from(self.policy.food[band(age) as usize] * config.household_need);
                let eligible = age >= self.policy.parameters.minimum_age
                    && age < self.policy.parameters.retirement_age;
                let capacity = min(
                    config.work_cap * (1000 - person.number(P::FATIGUE)) / 1000,
                    1440 - config.reserved_minutes,
                );
                profile.household_time += select(eligible, capacity, 0);
                if eligible && capacity > 0 {
                    profile.workers += 1;
                    profile.shortest_day = min(profile.shortest_day, capacity);
                }
                if eligible && profile.worker == u32::MAX {
                    profile.worker = index;
                    profile.available = capacity;
                }
            }
        }
        profile
    }

    #[allow(
        clippy::manual_div_ceil,
        reason = "CubeCL lowers explicit integer division"
    )]
    fn observe(&mut self, profile: &HouseholdProfile) {
        let config = self.economy.ledger.config;
        self.economy
            .ledger
            .set_number(Field::LivingMembers, self.household, profile.living);
        self.economy.ledger.set_number(
            Field::HeatingNeed,
            self.household,
            profile.living * config.heating_per_person,
        );
        self.economy
            .ledger
            .set_number(Field::HeatingServed, self.household, 0);
        self.economy
            .ledger
            .set_number(Field::Worker, self.household, profile.worker);
        let mut available = profile.available;
        if comptime!(config.cohorts == Feature::Enabled) {
            available = profile.shortest_day * profile.workers;
            self.economy
                .ledger
                .set_number(Field::EligibleWorkers, self.household, profile.workers);
        }
        let moving = self.economy.ledger.number(Field::MoveDue, self.household) > 0;
        self.economy.ledger.set_number(
            Field::HouseholdBudget,
            self.household,
            select(moving, 0, profile.household_time),
        );
        self.economy
            .ledger
            .set_number(Field::Time, self.household, select(moving, 0, available));
        self.economy.ledger.set_number(
            Field::HomeTime,
            self.household,
            select(moving, 0, profile.household_time - available),
        );
        let gross = self
            .economy
            .ledger
            .value(Field::GrossIncome, self.household);
        let previous = self
            .economy
            .ledger
            .value(Field::OpeningGrossIncome, self.household);
        let salary = positive(gross, previous);
        self.economy
            .ledger
            .set(Field::OpeningGrossIncome, self.household, gross);
        self.plan_food(
            salary,
            profile.living,
            ((profile.demand + 999u64) / 1000u64) as u32,
        );
        self.economy.ledger.set_number(
            Field::NutritionNeed,
            self.household,
            select(
                config.food_medium_salary > 0,
                profile.living * config.household_need,
                ((profile.demand + 999u64) / 1000u64) as u32,
            ),
        );
    }

    fn after_labor(&mut self, profile: &HouseholdProfile) {
        let config = self.economy.ledger.config;
        let gross = self
            .economy
            .ledger
            .value(Field::GrossIncome, self.household);
        let opening = self
            .economy
            .ledger
            .value(Field::OpeningGrossIncome, self.household);
        let salary = positive(gross, opening);
        self.plan_food(
            salary,
            profile.living,
            profile.living * config.household_need,
        );
    }

    fn requirements(&mut self, profile: &HouseholdProfile) {
        let config = self.economy.ledger.config;
        let work = self.economy.ledger.number(Field::Hours, self.household)
            + self
                .economy
                .ledger
                .number(Field::CartageMinutes, self.household)
            + self
                .economy
                .ledger
                .number(Field::HomeMinutes, self.household);
        let additional = u64::cast_from(work)
            * u64::cast_from(self.policy.parameters.work_food)
            * u64::cast_from(config.household_need)
            / 480u64;
        let total = profile.demand
            + additional
            + u64::cast_from(
                self.economy
                    .ledger
                    .number(Field::FoodFraction, self.household),
            );
        let gross = self
            .economy
            .ledger
            .value(Field::GrossIncome, self.household);
        let opening = self
            .economy
            .ledger
            .value(Field::OpeningGrossIncome, self.household);
        let salary = positive(gross, opening);
        self.plan_food(salary, profile.living, (total / 1000u64) as u32);
        self.economy.ledger.set_number(
            Field::NutritionNeed,
            self.household,
            select(
                config.food_medium_salary > 0,
                profile.living * config.household_need,
                (total / 1000u64) as u32,
            ),
        );
        self.economy.ledger.set_number(
            Field::FoodFraction,
            self.household,
            (total % 1000u64) as u32,
        );
    }

    fn plan_food(&mut self, salary: Ticks, members: u32, minimum: u32) {
        let ledger = &mut self.economy.ledger;
        let target = BreadDemand {
            gross_income: salary,
            daily_savings: divide(
                ledger.value(Field::Cash, self.household),
                HomeKitchen::budget_days(&*ledger, self.household, self.day),
            ),
            expected_income: ledger.value(Field::IncomeForecast, self.household),
            minimum,
            members,
            price: ledger.number(Field::FoodPrice, self.household),
        }
        .target(ledger.config);
        ledger.set_number(Field::FoodNeed, self.household, target);
    }

    #[allow(
        clippy::manual_is_multiple_of,
        clippy::manual_range_contains,
        clippy::manual_div_ceil,
        reason = "CubeCL lowers explicit runtime guards"
    )]
    fn individual_mortality(&mut self, mut living: u32) {
        let config = self.economy.ledger.config;
        let target = self.economy.ledger.number(Field::FoodNeed, self.household);
        let consumed = target - self.economy.ledger.number(Field::Shortage, self.household);
        let need = self
            .economy
            .ledger
            .number(Field::NutritionNeed, self.household);
        let deficit = (need - min(need, consumed)) * 1000 / max(need, 1);
        let freezing_need = self
            .economy
            .ledger
            .number(Field::LivingMembers, self.household)
            * (max(config.heating_per_person, 2) - 2);
        let served = self
            .economy
            .ledger
            .number(Field::HeatingServed, self.household);
        let cold_deficit =
            (freezing_need - min(freezing_need, served)) * 1000 / max(freezing_need, 1);
        let mut supported_births = zero();
        if self.policy.parameters.births > 0 {
            let infant_food = select(
                config.food_medium_salary > 0,
                config.household_need,
                (self.policy.food[0usize] * config.household_need + 999) / 1000,
            );
            let support = family_support(&self.economy.ledger, self.household, need + infant_food);
            let support = select(
                comptime!(config.economy == crate::domain::Economy::Industrial)
                    && max(deficit, cold_deficit) > 0,
                comptime!(FamilySupport::Deprived as u32),
                support,
            );
            supported_births = select(
                support == comptime!(FamilySupport::Secure as u32),
                rate(self.policy.parameters.births),
                zero(),
            );
        }
        let selected = self.economy.ledger.number(Field::Worker, self.household);
        let employer = self.economy.ledger.number(Field::Employer, self.household);
        let minutes = self.economy.ledger.number(Field::Hours, self.household);
        let home = self
            .economy
            .ledger
            .number(Field::HomeMinutes, self.household);
        let mut home_remaining = home;
        let mut cartage_remaining = self
            .economy
            .ledger
            .number(Field::CartageMinutes, self.household);
        let mut paid_workers = self.economy.ledger.number(Field::Employed, self.household);
        let shift = minutes / max(paid_workers, 1);
        let mut delegate = zero();
        for offset in 0..self.count {
            let first = select(selected == u32::MAX, self.first, selected + 1);
            let index = self.first + (first - self.first + offset) % max(self.count, 1);
            let mut person = PersonDay::new(self.people, index);
            if person.number(P::CAUSE) == 0 {
                let age = person.number(P::AGE);
                let group = band(age);
                let mut work = select(index == selected, minutes, 0);
                let eligible = age >= self.policy.parameters.minimum_age
                    && age < self.policy.parameters.retirement_age;
                let capacity = min(
                    config.work_cap * (1000 - person.number(P::FATIGUE)) / 1000,
                    1440 - config.reserved_minutes,
                );
                if comptime!(config.cohorts == Feature::Enabled) {
                    let hired = paid_workers > 0 && shift > 0 && eligible && capacity >= shift;
                    work = select(hired, shift, 0);
                    paid_workers -= u32::cast_from(hired);
                }
                let home_work =
                    select(eligible, min(home_remaining, max(capacity, work) - work), 0);
                home_remaining -= home_work;
                let cartage = select(
                    eligible,
                    min(cartage_remaining, max(capacity, work) - work - home_work),
                    0,
                );
                cartage_remaining -= cartage;
                let prior = person.number(P::EXPOSURE);
                let exposure = famine_exposure(prior, deficit, self.policy.parameters.recovery);
                person.set(P::EXPOSURE, exposure);
                let sex_and_cold_exposure = person.number(P::SEX_AND_COLD_EXPOSURE);
                let sex = sex_and_cold_exposure & 1;
                let cold_exposure = cold_exposure(
                    sex_and_cold_exposure >> 1,
                    cold_deficit,
                    self.policy.parameters.recovery,
                );
                person.set(P::SEX_AND_COLD_EXPOSURE, sex | (cold_exposure << 1));
                person.set(P::EMPLOYER, select(work > 0, employer, 0));
                person.set(P::WORK, work + home_work + cartage);
                let background = rate(self.policy.background[group as usize]);
                let mut occupational = zero();
                if employer > 0 && work > 0 {
                    let firm = employer - 1;
                    let good = self.economy.actors.kind(firm).bits();
                    occupational = OccupationalExposure {
                        minutes: work,
                        condition: self.economy.ledger.number(Field::EquipmentCondition, firm),
                        safety: self.economy.ledger.number(Field::SafetyProvision, firm),
                        fatigue: person.number(P::FATIGUE),
                        method: self.economy.ledger.number(Field::ActiveMethod, firm),
                    }
                    .risk(
                        self.policy.occupational[good as usize - 1],
                        self.policy.parameters.reference_minutes,
                    );
                }
                if home_work > 0 {
                    let household_risk =
                        OccupationalExposure::unprotected(home_work, person.number(P::FATIGUE))
                            .risk(
                                self.policy.occupational[0],
                                self.policy.parameters.reference_minutes,
                            );
                    occupational = plus(occupational, household_risk).value;
                }
                if cartage > 0 {
                    occupational = plus(
                        occupational,
                        OccupationalExposure::unprotected(cartage, person.number(P::FATIGUE)).risk(
                            self.policy.occupational
                                [comptime!(crate::domain::Good::Railway as usize - 1)],
                            self.policy.parameters.reference_minutes,
                        ),
                    )
                    .value;
                }
                let famine = famine_risk(
                    exposure,
                    self.policy.parameters.famine,
                    self.policy.vulnerability[group as usize],
                );
                let draw = probability_draw(
                    self.samples[index as usize * 5],
                    self.samples[index as usize * 5 + 1],
                );
                let cold = cold_risk(
                    cold_exposure,
                    self.policy.parameters.cold,
                    self.policy.cold_vulnerability[group as usize],
                );
                let cause = MortalityRisks {
                    background,
                    occupational,
                    famine,
                    cold,
                }
                .cause(draw);
                person.set(P::CAUSE, cause);
                living -= u32::cast_from(cause > 0);
                if comptime!(config.cohorts == Feature::Enabled) && eligible && cause == 0 {
                    delegate = person.identity();
                }
                let independent = age >= 18 * DAYS_PER_YEAR
                    && age < 19 * DAYS_PER_YEAR
                    && index != selected
                    && cause == 0;
                if independent && (self.day + person.number(P::IDENTITY)) % 365 == 0 && living > 1 {
                    self.economy
                        .ledger
                        .set(Field::SplitPerson, self.household, person.identity());
                }
                person.set(P::DEATH_DAY, select(cause > 0, self.day + 1, 0));
                person.set(P::AGE, person.number(P::AGE) + 1);
                let fatigue = person.number(P::FATIGUE);
                let exertion = person.number(P::WORK) * config.work_fatigue / 480;
                person.set(
                    P::FATIGUE,
                    min(
                        max(fatigue + exertion, config.rest_recovery) - config.rest_recovery,
                        1000,
                    ),
                );
                let cooldown = person.number(P::COOLDOWN);
                person.set(P::COOLDOWN, max(cooldown, 1) - 1);
                let fertile = age >= 18 * DAYS_PER_YEAR
                    && age < 45 * DAYS_PER_YEAR
                    && sex == 0
                    && cooldown == 0
                    && cause == 0;
                let birth_draw = probability_draw(
                    self.samples[index as usize * 5 + 2],
                    self.samples[index as usize * 5 + 3],
                );
                let fertility = select(
                    comptime!(config.economy == crate::domain::Economy::Industrial)
                        && max(exposure, cold_exposure) > 0,
                    zero(),
                    supported_births,
                );
                if fertile && lt(birth_draw, fertility) && self.births[self.household as usize] == 0
                {
                    self.births[self.household as usize] =
                        1 + u32::cast_from(self.samples[index as usize * 5 + 4] >= 0.5);
                    person.set(P::COOLDOWN, self.policy.parameters.spacing);
                }
            }
        }
        if comptime!(config.cohorts == Feature::Enabled) {
            self.economy
                .ledger
                .set(Field::CohortDelegate, self.household, delegate);
        }
        self.economy
            .ledger
            .set_number(Field::LivingMembers, self.household, living);
    }
}

#[cube]
impl DemographyInput {
    pub(super) fn advance(&mut self) {
        let household = ABSOLUTE_POS as u32;
        if household < self.config.actors {
            let stage = comptime!(self.stage);
            let model = comptime!(self.model);
            let mut day = HouseholdDay::new(self, household);
            let profile = day.profile(stage, model);
            match stage {
                Stage::Observation => day.observe(&profile),
                Stage::Diet => day.after_labor(&profile),
                Stage::Requirements => day.requirements(&profile),
                Stage::Mortality => match model {
                    PopulationModel::Cohorts => super::cohorts::advance(
                        &mut day.economy,
                        day.policy,
                        day.samples,
                        household,
                    ),
                    PopulationModel::Individuals => day.individual_mortality(profile.living),
                },
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::system::AsBufferArg;
    use anyhow::Result;

    #[cube(launch)]
    fn probe(input: &[u32], output: &mut [u32]) {
        let index = ABSOLUTE_POS;
        if index * 8 < input.len() {
            let base = index * 8;
            let risk = OccupationalExposure {
                minutes: input[base + 1],
                condition: input[base + 2],
                safety: input[base + 3],
                fatigue: input[base + 4],
                method: input[base + 5],
            }
            .risk(input[base], 144_000);
            let draw = probability_draw(
                (input[base + 6] >> 8) as f32 / 16777216.0,
                (input[base + 7] >> 8) as f32 / 16777216.0,
            );
            let daily = rate(input[base]);
            output[index * 6] = risk.extract(0usize);
            output[index * 6 + 1] = risk.extract(1usize);
            output[index * 6 + 2] = draw.extract(0usize);
            output[index * 6 + 3] = draw.extract(1usize);
            output[index * 6 + 4] = daily.extract(0usize);
            output[index * 6 + 5] = daily.extract(1usize);
        }
    }

    #[cube(launch)]
    fn famine_probe(input: &[u32], output: &mut [u32]) {
        let index = ABSOLUTE_POS;
        if index * 4 < input.len() {
            let base = index * 4;
            let exposure = famine_exposure(input[base], input[base + 1], 500);
            let risk = famine_risk(exposure, 20_000, input[base + 2]);
            output[base] = exposure;
            output[base + 1] = risk.extract(0usize);
            output[base + 2] = risk.extract(1usize);
            output[base + 3] = MortalityRisks {
                background: ticks(10, 0),
                occupational: ticks(20, 0),
                famine: ticks(30, 0),
                cold: zero(),
            }
            .cause(ticks(input[base + 3], 0));
        }
    }

    #[test]
    fn famine_duration_feeding_age_and_single_cause_boundaries_on_gpu() -> Result<()> {
        let device = crate::simulation::tests::device(64)?;
        let cases = [
            [0, 0, 1000, 0],
            [30_000, 0, 1000, 9],
            [30_000, 500, 1000, 10],
            [30_000, 1000, 1000, 29],
            [60_000, 1000, 1000, 30],
            [30_000, 1000, 2000, 59],
            [365_000, 1000, 1000, 60],
            [10, 0, 1000, u32::MAX],
        ];
        let input = device.tensor(bytemuck::cast_slice(&cases));
        let output = device.empty(cases.len() * 4);
        famine_probe::launch(
            &device.client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(64),
            input.buffer_arg(),
            output.buffer_arg(),
        );
        let words = device.read(&output)?;
        let rows = words.chunks_exact(4).collect::<Vec<_>>();
        let risk = |index: usize| u64::from(rows[index][1]) | (u64::from(rows[index][2]) << 32);
        assert_eq!(
            rows.iter().map(|row| row[0]).collect::<Vec<_>>(),
            [0, 29_500, 30_250, 31_000, 61_000, 31_000, 365_000, 0]
        );
        assert_eq!(risk(0), 0);
        assert_eq!(risk(7), 0);
        assert!(risk(1) < risk(2));
        assert!(risk(2) < risk(3));
        assert!(risk(3) < risk(4));
        assert_eq!(risk(5), risk(3) * 2);
        assert_eq!(
            rows.iter().map(|row| row[3]).collect::<Vec<_>>(),
            [1, 1, 2, 2, 3, 3, 0, 0]
        );
        Ok(())
    }

    #[test]
    fn occupational_units_rare_risks_safety_and_cubek_draws_on_gpu() -> Result<()> {
        let device = crate::simulation::tests::device(64)?;
        let cases = [
            [100, 0, 1000, 0, 0, 0, 0, 0],
            [100, 1, 1000, 0, 0, 0, u32::MAX, u32::MAX],
            [100, 480, 1000, 0, 0, 0, 1234, 5678],
            [100, 960, 1000, 0, 0, 0, u32::MAX, 0],
            [100, 480, 500, 0, 0, 0, 0, u32::MAX],
            [100, 480, 1000, 750, 0, 0, 17, 912912],
            [1280, 480, 1000, 0, 0, 0, 8128, 113782],
            [100, 480, 1000, 0, 500, 0, 22, 55],
            [100, 480, 1000, 0, 0, 1, 99999, 11],
            [0, 480, 1000, 0, 0, 0, 0, 0],
            [4294, 480, 1000, 0, 0, 0, 1, 2],
            [4295, 480, 1000, 0, 0, 0, 3, 4],
            [20_000, 480, 1000, 0, 0, 0, 5, 6],
            [1_000_000, 480, 1000, 0, 0, 0, 7, 8],
            [u32::MAX, 480, 1000, 0, 0, 0, 9, 10],
        ];
        let input = device.tensor(bytemuck::cast_slice(&cases));
        let output = device.empty(cases.len() * 6);
        probe::launch(
            &device.client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(64),
            input.buffer_arg(),
            output.buffer_arg(),
        );
        let words = device.read(&output)?;
        let risks = words
            .chunks_exact(6)
            .map(|row| u64::from(row[0]) | (u64::from(row[1]) << 32))
            .collect::<Vec<_>>();
        assert_eq!(risks[0], 0);
        assert_eq!(risks[1], 100 * 1_000_000 / 144_000);
        assert!(risks[1] > 0);
        assert!(risks[3] > risks[2]);
        assert!(risks[4] > risks[2]);
        assert!(risks[5] < risks[2]);
        assert!(risks[6] > risks[2]);
        assert!(risks[7] > risks[2]);
        assert!(risks[8] > risks[2]);
        for (case, row) in cases.iter().zip(words.chunks_exact(6)) {
            let random = (u64::from(case[6] >> 8) << 24) | u64::from(case[7] >> 8);
            let expected = (u128::from(random) * u128::from(PROBABILITY_SCALE)) >> 48;
            let actual = u64::from(row[2]) | (u64::from(row[3]) << 32);
            assert_eq!(u128::from(actual), expected);
            let expected_rate = u128::from(case[0]) * 1_000_000 / u128::from(DAYS_PER_YEAR);
            let actual_rate = u64::from(row[4]) | (u64::from(row[5]) << 32);
            assert_eq!(
                u128::from(actual_rate),
                expected_rate,
                "annual rate {}",
                case[0]
            );
        }
        Ok(())
    }
}

#[cfg(test)]
#[cube(launch)]
pub(super) fn original_grow(
    source: &[u32],
    target: &mut [u32],
    old_ranges: &[u32],
    new_ranges: &[u32],
    births: &[u32],
    next_low: u32,
    next_high: u32,
) {
    let household = ABSOLUTE_POS;
    if household < births.len() {
        let start = old_ranges[household * 2];
        let count = old_ranges[household * 2 + 1];
        let destination = new_ranges[household * 2];
        for person in 0..count {
            for word in 0..P::WORDS {
                target[(destination + person) as usize * P::WORDS + word] =
                    source[(start + person) as usize * P::WORDS + word];
            }
        }
        if births[household] > 0 {
            let base = (destination + count) as usize * P::WORDS;
            for word in 0..P::WORDS {
                target[base + word] = 0;
            }
            let identity = plus(ticks(next_low, next_high), ticks(destination - start, 0)).value;
            target[base + P::IDENTITY] = identity.extract(0usize);
            target[base + P::IDENTITY + 1] = identity.extract(1usize);
            target[base + 2] = household as u32;
            target[base + P::SEX_AND_COLD_EXPOSURE] = births[household] - 1;
        }
    }
}

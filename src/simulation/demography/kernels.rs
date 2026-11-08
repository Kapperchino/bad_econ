use super::{DemographyPolicy, OBSERVATION_METRICS, OBSERVATION_SHARDS, PersonLayout as P, Stage};
use crate::simulation::configuration::EconomyConfig;
use crate::{
    domain::{
        Field,
        demography::{DAYS_PER_YEAR, PROBABILITY_SCALE},
    },
    simulation::{layout::ActorLayout as A, words::*},
};
use cubecl::prelude::*;

#[cube]
fn get(state: &mut [u32], config: &EconomyConfig, field: u32, actor: u32) -> u32 {
    load(state, field * config.actors + actor).extract(0usize)
}

#[cube]
fn put(state: &mut [u32], config: &EconomyConfig, field: u32, actor: u32, value: u32) {
    store(state, field * config.actors + actor, ticks(value, 0));
}

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
fn rate(annual_ppm: u32) -> Ticks {
    divide(times32(annual_ppm, 1_000_000), DAYS_PER_YEAR)
}

#[cube]
fn occupation(
    rate_ppm: u32,
    minutes: u32,
    reference: u32,
    condition: u32,
    safety: u32,
    fatigue: u32,
    method: u32,
) -> Ticks {
    let base = divide(
        times(times32(rate_ppm, 1_000_000), minutes).value,
        reference,
    );
    let equipment = 1000 + (1000 - min(condition, 1000)) * 2;
    let tired = 1000 + fatigue + (max(minutes, 480) - 480) * 1000 / 480;
    let mechanized = select(method > 0, 1250u32, 1000u32);
    let maintained = ratio(base, equipment, 1000).value;
    let exertion = ratio(maintained, tired, 1000).value;
    let method_risk = ratio(exertion, mechanized, 1000).value;
    ratio(method_risk, 1000 - min(safety, 900), 1000).value
}

#[cube]
fn famine_exposure(prior: u32, deficit: u32, recovery: u32) -> u32 {
    let reduction = recovery * (1000 - deficit) / 1000;
    min(max(prior + deficit, reduction) - reduction, 365_000)
}

#[cube]
fn famine_risk(exposure: u32, daily_ppm: u32, vulnerability: u32) -> Ticks {
    let severity = min(exposure, 90_000);
    let base = ratio(times32(daily_ppm, 1_000_000), severity, 30_000).value;
    let duration = ratio(base, severity, 30_000).value;
    ratio(duration, vulnerability, 1000).value
}

#[cube]
fn death_cause(draw: Ticks, background: Ticks, occupational: Ticks, famine: Ticks) -> u32 {
    let first = background;
    let second = plus(first, occupational).value;
    let total = minimum(
        plus(second, famine).value,
        ticks(
            comptime!(PROBABILITY_SCALE as u32),
            comptime!((PROBABILITY_SCALE >> 32) as u32),
        ),
    );
    let mut cause = 0;
    if lt(draw, total) {
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

#[allow(
    clippy::collapsible_if,
    clippy::manual_is_multiple_of,
    clippy::manual_range_contains,
    clippy::manual_div_ceil,
    reason = "CubeCL lowers explicit runtime guards"
)]
#[cube]
pub fn step(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    people: &mut [u32],
    ranges: &[u32],
    policy: &DemographyPolicy,
    births: &mut [u32],
    samples: &[f32],
    day: u32,
    #[comptime] stage: Stage,
) {
    let household = ABSOLUTE_POS as u32;
    if household < config.actors {
        let start = ranges[household as usize * 2];
        let count = ranges[household as usize * 2 + 1];
        let mut living = 0;
        let mut demand = 0;
        let mut worker = u32::new(comptime!(i64::from(u32::MAX)));
        let mut available = 0;
        let mut household_time = 0;
        if comptime!(stage == Stage::Observation) {
            births[household as usize] = 0;
            put(
                state,
                config,
                comptime!(Field::SplitPerson as u32),
                household,
                0,
            );
        }
        for offset in 0..count {
            let index = start + (offset + day % max(count, 1)) % max(count, 1);
            let base = index as usize * P::WORDS;
            if comptime!(stage == Stage::Observation) {
                people[base + P::EMPLOYER] = 0;
                people[base + P::WORK] = 0;
            }
            if people[base + P::CAUSE] == 0 {
                let age = people[base + P::AGE];
                living += 1;
                demand += policy.food[band(age) as usize] * config.household_need;
                let eligible =
                    age >= policy.parameters.minimum_age && age < policy.parameters.retirement_age;
                let capacity = min(
                    config.work_cap * (1000 - people[base + P::FATIGUE]) / 1000,
                    1440 - config.reserved_minutes,
                );
                household_time += select(eligible, capacity, 0);
                if eligible && worker == u32::MAX {
                    worker = index;
                    available = min(
                        config.work_cap * (1000 - people[base + P::FATIGUE]) / 1000,
                        1440 - config.reserved_minutes,
                    );
                }
                if comptime!(stage == Stage::Observation) {
                    people[base + P::EMPLOYER] = 0;
                    people[base + P::WORK] = 0;
                }
            }
        }
        if comptime!(stage == Stage::Observation) {
            put(
                state,
                config,
                comptime!(Field::LivingMembers as u32),
                household,
                living,
            );
            put(
                state,
                config,
                comptime!(Field::Worker as u32),
                household,
                worker,
            );
            let moving = get(state, config, comptime!(Field::MoveDue as u32), household) > 0;
            put(
                state,
                config,
                comptime!(Field::HouseholdBudget as u32),
                household,
                select(moving, 0, household_time),
            );
            put(
                state,
                config,
                comptime!(Field::Time as u32),
                household,
                select(moving, 0, available),
            );
            put(
                state,
                config,
                comptime!(Field::HomeTime as u32),
                household,
                select(moving, 0, household_time - available),
            );
            put(
                state,
                config,
                comptime!(Field::FoodNeed as u32),
                household,
                (demand + 999) / 1000,
            );
        }
        if comptime!(stage == Stage::Requirements) {
            let work = get(state, config, comptime!(Field::Hours as u32), household)
                + get(
                    state,
                    config,
                    comptime!(Field::HomeMinutes as u32),
                    household,
                );
            let additional = work * policy.parameters.work_food * config.household_need / 480;
            let total = demand
                + additional
                + get(
                    state,
                    config,
                    comptime!(Field::FoodFraction as u32),
                    household,
                );
            put(
                state,
                config,
                comptime!(Field::FoodNeed as u32),
                household,
                total / 1000,
            );
            put(
                state,
                config,
                comptime!(Field::FoodFraction as u32),
                household,
                total % 1000,
            );
        }
        if comptime!(stage == Stage::Mortality) {
            let need = get(state, config, comptime!(Field::FoodNeed as u32), household);
            let deficit = get(state, config, comptime!(Field::Shortage as u32), household) * 1000
                / max(need, 1);
            let selected = get(state, config, comptime!(Field::Worker as u32), household);
            let employer = get(state, config, comptime!(Field::Employer as u32), household);
            let minutes = get(state, config, comptime!(Field::Hours as u32), household);
            let home = get(
                state,
                config,
                comptime!(Field::HomeMinutes as u32),
                household,
            );
            let mut home_remaining = home;
            for offset in 0..count {
                let first = select(selected == u32::MAX, start, selected + 1);
                let index = start + (first - start + offset) % max(count, 1);
                let base = index as usize * P::WORDS;
                if people[base + P::CAUSE] == 0 {
                    let age = people[base + P::AGE];
                    let group = band(age);
                    let work = select(index == selected, minutes, 0);
                    let eligible = age >= policy.parameters.minimum_age
                        && age < policy.parameters.retirement_age;
                    let capacity = min(
                        config.work_cap * (1000 - people[base + P::FATIGUE]) / 1000,
                        1440 - config.reserved_minutes,
                    );
                    let home_work =
                        select(eligible, min(home_remaining, max(capacity, work) - work), 0);
                    home_remaining -= home_work;
                    let prior = people[base + P::EXPOSURE];
                    let exposure = famine_exposure(prior, deficit, policy.parameters.recovery);
                    people[base + P::EXPOSURE] = exposure;
                    people[base + P::EMPLOYER] = select(work > 0, employer, 0);
                    people[base + P::WORK] = work + home_work;
                    let background = rate(policy.background[group as usize]);
                    let mut occupational = zero();
                    if employer > 0 && work > 0 {
                        let firm = employer - 1;
                        let good = actors[A::KIND * (actors.len() / A::WORDS) + firm as usize];
                        occupational = occupation(
                            policy.occupational[good as usize - 1],
                            work,
                            policy.parameters.reference_minutes,
                            get(
                                state,
                                config,
                                comptime!(Field::EquipmentCondition as u32),
                                firm,
                            ),
                            get(
                                state,
                                config,
                                comptime!(Field::SafetyProvision as u32),
                                firm,
                            ),
                            people[base + P::FATIGUE],
                            get(state, config, comptime!(Field::ActiveMethod as u32), firm),
                        );
                    }
                    if home_work > 0 {
                        let household_risk = occupation(
                            policy.occupational[0],
                            home_work,
                            policy.parameters.reference_minutes,
                            1000,
                            0,
                            people[base + P::FATIGUE],
                            0,
                        );
                        occupational = plus(occupational, household_risk).value;
                    }
                    let famine = famine_risk(
                        exposure,
                        policy.parameters.famine,
                        policy.vulnerability[group as usize],
                    );
                    let draw = probability_draw(
                        samples[index as usize * 5],
                        samples[index as usize * 5 + 1],
                    );
                    let cause = death_cause(draw, background, occupational, famine);
                    people[base + P::CAUSE] = cause;
                    living -= u32::cast_from(cause > 0);
                    let independent = age >= 18 * DAYS_PER_YEAR
                        && age < 19 * DAYS_PER_YEAR
                        && index != selected
                        && cause == 0;
                    if independent && (day + people[base + P::IDENTITY]) % 365 == 0 && living > 1 {
                        store(
                            state,
                            comptime!(Field::SplitPerson as u32) * config.actors + household,
                            ticks(people[base + P::IDENTITY], people[base + P::IDENTITY + 1]),
                        );
                    }
                    people[base + P::DEATH_DAY] = select(cause > 0, day + 1, 0);
                    people[base + P::AGE] += 1;
                    let fatigue = people[base + P::FATIGUE];
                    let exertion = people[base + P::WORK] * config.work_fatigue / 480;
                    people[base + P::FATIGUE] = min(
                        max(fatigue + exertion, config.rest_recovery) - config.rest_recovery,
                        1000,
                    );
                    let cooldown = people[base + P::COOLDOWN];
                    people[base + P::COOLDOWN] = max(cooldown, 1) - 1;
                    let fertile = age >= 18 * DAYS_PER_YEAR
                        && age < 45 * DAYS_PER_YEAR
                        && people[base + P::SEX] == 0
                        && cooldown == 0
                        && cause == 0;
                    let birth_draw = probability_draw(
                        samples[index as usize * 5 + 2],
                        samples[index as usize * 5 + 3],
                    );
                    if fertile
                        && lt(birth_draw, rate(policy.parameters.births))
                        && births[household as usize] == 0
                    {
                        births[household as usize] =
                            1 + u32::cast_from(samples[index as usize * 5 + 4] >= 0.5);
                        people[base + P::COOLDOWN] = policy.parameters.spacing;
                    }
                }
            }
            put(
                state,
                config,
                comptime!(Field::LivingMembers as u32),
                household,
                living,
            );
        }
    }
}

#[cube]
pub fn observe(
    people: &[u32],
    state: &mut [u32],
    config: &EconomyConfig,
    policy: &DemographyPolicy,
    output: &[Atomic<u32>],
    count: u32,
) {
    let index = ABSOLUTE_POS;
    if index < count as usize {
        let base = index * P::WORDS;
        let shard = index % OBSERVATION_SHARDS * OBSERVATION_METRICS;
        let cause = people[base + P::CAUSE];
        if cause == 0 {
            output[shard].fetch_add(1);
            let age = people[base + P::AGE];
            let household = people[base + P::HOUSEHOLD];
            let selected =
                get(state, config, comptime!(Field::Worker as u32), household) == index as u32;
            let firm = get(
                state,
                config,
                comptime!(Field::InvestmentFirm as u32),
                household,
            );
            let owner = firm > 0
                && get(
                    state,
                    config,
                    comptime!(Field::EnterpriseStatus as u32),
                    max(firm, 1) - 1,
                ) < 4;
            let farmer = get(state, config, comptime!(Field::LandPlots as u32), household) > 0;
            let dependent =
                age < policy.parameters.minimum_age || age >= policy.parameters.retirement_age;
            let mut category = select(selected, 3u32, 5u32);
            category = select(people[base + P::EMPLOYER] > 0, 1u32, category);
            category = select(selected && farmer, 0u32, category);
            category = select(selected && owner, 2u32, category);
            category = select(dependent, 4u32, category);
            output[shard + 8 + category as usize].fetch_add(1);
            output[shard + 1].fetch_add(u32::cast_from(
                age < policy.parameters.minimum_age || age >= policy.parameters.retirement_age,
            ));
        } else {
            output[shard + (cause + 1) as usize].fetch_add(1);
        }
        output[shard + 5].fetch_add(people[base + P::WORK]);
        let exposure = people[base + P::EXPOSURE];
        let previous = output[shard + 6].fetch_add(exposure);
        output[shard + 7].fetch_add(u32::cast_from(exposure > u32::MAX - previous));
    }
}

#[cube]
pub fn observation_totals(sums: &[u64], output: &mut [u32]) {
    let metric = ABSOLUTE_POS;
    if metric < OBSERVATION_METRICS && metric != 7 {
        output[metric] = sums[metric] as u32;
        if metric == 6 {
            output[7usize] = ((sums[6usize] >> 32) + sums[7usize]) as u32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::Device;
    use anyhow::Result;
    use burn_wgpu::CubeTensor;

    #[cube(launch)]
    fn probe(input: &[u32], output: &mut [u32]) {
        let index = ABSOLUTE_POS;
        if index * 8 < input.len() {
            let base = index * 8;
            let risk = occupation(
                input[base],
                input[base + 1],
                144_000,
                input[base + 2],
                input[base + 3],
                input[base + 4],
                input[base + 5],
            );
            let draw = probability_draw(
                (input[base + 6] >> 8) as f32 / 16777216.0,
                (input[base + 7] >> 8) as f32 / 16777216.0,
            );
            output[index * 4] = risk.extract(0usize);
            output[index * 4 + 1] = risk.extract(1usize);
            output[index * 4 + 2] = draw.extract(0usize);
            output[index * 4 + 3] = draw.extract(1usize);
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
            output[base + 3] = death_cause(
                ticks(input[base + 3], 0),
                ticks(10, 0),
                ticks(20, 0),
                ticks(30, 0),
            );
        }
    }

    #[test]
    fn famine_duration_feeding_age_and_single_cause_boundaries_on_gpu() -> Result<()> {
        let device = Device::headless(64)?;
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
        let argument = |tensor: &CubeTensor| tensor.clone().into_buffer_arg();
        famine_probe::launch(
            &device.client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(64),
            argument(&input),
            argument(&output),
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
        let device = Device::headless(64)?;
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
        ];
        let input = device.tensor(bytemuck::cast_slice(&cases));
        let output = device.empty(cases.len() * 4);
        let argument = |tensor: &CubeTensor| tensor.clone().into_buffer_arg();
        probe::launch(
            &device.client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(64),
            argument(&input),
            argument(&output),
        );
        let words = device.read(&output)?;
        let risks = words
            .chunks_exact(4)
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
        for (case, row) in cases.iter().zip(words.chunks_exact(4)) {
            let random = (u64::from(case[6] >> 8) << 24) | u64::from(case[7] >> 8);
            let expected = (u128::from(random) * u128::from(PROBABILITY_SCALE)) >> 48;
            let actual = u64::from(row[2]) | (u64::from(row[3]) << 32);
            assert_eq!(u128::from(actual), expected);
        }
        Ok(())
    }
}

#[cube]
pub(super) fn grow(
    source: &[Vector<u32, Const<4>>],
    target: &mut [Vector<u32, Const<4>>],
    old_ranges: &[u32],
    new_ranges: &[u32],
    people: u32,
) {
    let index = ABSOLUTE_POS;
    if index < people as usize * (P::WORDS / 4) {
        let person = index / (P::WORDS / 4);
        let household = source[person * (P::WORDS / 4)].extract(P::HOUSEHOLD) as usize;
        let offset = new_ranges[household * 2] - old_ranges[household * 2];
        target[index + offset as usize * (P::WORDS / 4)] = source[index];
    }
}

#[cube]
pub(super) fn newborns(
    target: &mut [Vector<u32, Const<4>>],
    old_ranges: &[u32],
    new_ranges: &[u32],
    births: &[u32],
    next_low: u32,
    next_high: u32,
) {
    let household = ABSOLUTE_POS;
    if household < births.len() && births[household] > 0 {
        let start = old_ranges[household * 2];
        let count = old_ranges[household * 2 + 1];
        let destination = new_ranges[household * 2];
        let base = (destination + count) as usize * (P::WORDS / 4);
        let identity = plus(ticks(next_low, next_high), ticks(destination - start, 0)).value;
        let mut header = Vector::<u32, Const<4>>::new(0);
        header.insert(P::IDENTITY, identity.extract(0usize));
        header.insert(P::IDENTITY + 1, identity.extract(1usize));
        header.insert(P::HOUSEHOLD, household as u32);
        let mut status = Vector::<u32, Const<4>>::new(0);
        status.insert(P::SEX - 4, births[household] - 1);
        target[base] = header;
        target[base + 1] = status;
        target[base + 2] = Vector::new(0);
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
            target[base + P::SEX] = births[household] - 1;
        }
    }
}

#[cube]
pub(super) fn birth_prefix(births: &[u32], scratch: &mut [u32]) {
    let actor = ABSOLUTE_POS;
    let mut present = 0u32;
    if actor < births.len() {
        present = u32::cast_from(births[actor] > 0);
    }
    let prefix = crate::simulation::parallel::prefix_sum(present);
    if actor < births.len() {
        scratch[actor] = prefix.inclusive;
    }
    if UNIT_POS == 0 {
        scratch[births.len() + CUBE_POS] = prefix.total;
    }
}

#[cube]
pub(super) fn birth_ranges(old: &[u32], births: &[u32], scratch: &[u32], ranges: &mut [u32]) {
    let actor = ABSOLUTE_POS;
    if actor < births.len() {
        let birth = u32::cast_from(births[actor] > 0);
        let mut offset = scratch[actor] - birth;
        if CUBE_POS > 0 {
            offset += scratch[births.len() + CUBE_POS - 1];
        }
        ranges[actor * 2] = old[actor * 2] + offset;
        ranges[actor * 2 + 1] = old[actor * 2 + 1] + birth;
    }
}

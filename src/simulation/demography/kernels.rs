use super::{OBSERVATION_SHARDS, PersonLayout as P, PolicyLayout as D, Stage};
use crate::{
    domain::{
        Field,
        demography::{DAYS_PER_YEAR, PROBABILITY_SCALE},
    },
    simulation::{
        layout::{ActorLayout as A, ConfigLayout as C},
        words::*,
    },
};
use cubecl::prelude::*;

#[cube]
fn get(state: &mut [u32], config: &[u32], field: u32, actor: u32) -> u32 {
    load(state, field * config[C::ACTORS] + actor).extract(0usize)
}

#[cube]
fn put(state: &mut [u32], config: &[u32], field: u32, actor: u32, value: u32) {
    store(state, field * config[C::ACTORS] + actor, ticks(value, 0));
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
fn mix(value: u32) -> u32 {
    let a = (value ^ (value >> 16)) * 0x7feb352du32;
    let b = (a ^ (a >> 15)) * 0x846ca68bu32;
    b ^ (b >> 16)
}

#[cube]
fn random(people: &mut [u32], policy: &[u32], base: usize, day: u32, purpose: u32) -> Ticks {
    let a = mix(people[base + P::IDENTITY] ^ policy[D::SEED] ^ mix(day) ^ purpose);
    let b = mix(people[base + P::IDENTITY + 1] ^ policy[D::SEED + 1] ^ mix(a + 0x9e3779b9u32));
    ticks(mix(a ^ b), mix(b + a))
}

#[cube]
fn probability_draw(random: Ticks) -> Ticks {
    let low = times32(random.extract(0usize), comptime!(PROBABILITY_SCALE as u32));
    let cross_a = times32(random.extract(1usize), comptime!(PROBABILITY_SCALE as u32));
    let cross_b = times32(
        random.extract(0usize),
        comptime!((PROBABILITY_SCALE >> 32) as u32),
    );
    let high = times32(
        random.extract(1usize),
        comptime!((PROBABILITY_SCALE >> 32) as u32),
    );
    let middle = plus(
        ticks(low.extract(1usize), 0),
        ticks(cross_a.extract(0usize), 0),
    )
    .value;
    let middle = plus(middle, ticks(cross_b.extract(0usize), 0)).value;
    let upper = plus(high, ticks(cross_a.extract(1usize), 0)).value;
    let upper = plus(upper, ticks(cross_b.extract(1usize), 0)).value;
    plus(upper, ticks(middle.extract(1usize), 0)).value
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
#[cube(launch)]
pub fn step(
    state: &mut [u32],
    actors: &[u32],
    config: &[u32],
    people: &mut [u32],
    ranges: &[u32],
    policy: &[u32],
    births: &mut [u32],
    day: u32,
    #[comptime] stage: Stage,
) {
    let household = ABSOLUTE_POS as u32;
    if household < config[C::ACTORS] {
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
                demand += policy[D::FOOD + band(age) as usize] * config[C::HOUSEHOLD_NEED];
                let eligible = age >= policy[D::MINIMUM_AGE] && age < policy[D::RETIREMENT_AGE];
                let capacity = min(
                    config[C::WORK_CAP] * (1000 - people[base + P::FATIGUE]) / 1000,
                    1440 - config[C::RESERVED_MINUTES],
                );
                household_time += select(eligible, capacity, 0);
                if eligible && worker == u32::MAX {
                    worker = index;
                    available = min(
                        config[C::WORK_CAP] * (1000 - people[base + P::FATIGUE]) / 1000,
                        1440 - config[C::RESERVED_MINUTES],
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
            let additional = work * policy[D::WORK_FOOD] * config[C::HOUSEHOLD_NEED] / 480;
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
                    let eligible = age >= policy[D::MINIMUM_AGE] && age < policy[D::RETIREMENT_AGE];
                    let capacity = min(
                        config[C::WORK_CAP] * (1000 - people[base + P::FATIGUE]) / 1000,
                        1440 - config[C::RESERVED_MINUTES],
                    );
                    let home_work =
                        select(eligible, min(home_remaining, max(capacity, work) - work), 0);
                    home_remaining -= home_work;
                    let prior = people[base + P::EXPOSURE];
                    let exposure = famine_exposure(prior, deficit, policy[D::RECOVERY]);
                    people[base + P::EXPOSURE] = exposure;
                    people[base + P::EMPLOYER] = select(work > 0, employer, 0);
                    people[base + P::WORK] = work + home_work;
                    let background = rate(policy[D::BACKGROUND + group as usize]);
                    let mut occupational = zero();
                    if employer > 0 && work > 0 {
                        let firm = employer - 1;
                        let good = actors[A::KIND * (actors.len() / A::WORDS) + firm as usize];
                        occupational = occupation(
                            policy[D::OCCUPATIONAL + good as usize - 1],
                            work,
                            policy[D::REFERENCE_MINUTES],
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
                            policy[D::OCCUPATIONAL],
                            home_work,
                            policy[D::REFERENCE_MINUTES],
                            1000,
                            0,
                            people[base + P::FATIGUE],
                            0,
                        );
                        occupational = plus(occupational, household_risk).value;
                    }
                    let famine = famine_risk(
                        exposure,
                        policy[D::FAMINE],
                        policy[D::VULNERABILITY + group as usize],
                    );
                    let draw = probability_draw(random(people, policy, base, day, 1));
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
                            comptime!(Field::SplitPerson as u32) * config[C::ACTORS] + household,
                            ticks(people[base + P::IDENTITY], people[base + P::IDENTITY + 1]),
                        );
                    }
                    people[base + P::DEATH_DAY] = select(cause > 0, day + 1, 0);
                    people[base + P::AGE] += 1;
                    let fatigue = people[base + P::FATIGUE];
                    let exertion = people[base + P::WORK] * config[C::WORK_FATIGUE] / 480;
                    people[base + P::FATIGUE] = min(
                        max(fatigue + exertion, config[C::REST_RECOVERY])
                            - config[C::REST_RECOVERY],
                        1000,
                    );
                    let cooldown = people[base + P::COOLDOWN];
                    people[base + P::COOLDOWN] = max(cooldown, 1) - 1;
                    let fertile = age >= 18 * DAYS_PER_YEAR
                        && age < 45 * DAYS_PER_YEAR
                        && people[base + P::SEX] == 0
                        && cooldown == 0
                        && cause == 0;
                    let birth_draw = probability_draw(random(people, policy, base, day, 2));
                    if fertile
                        && lt(birth_draw, rate(policy[D::BIRTHS]))
                        && births[household as usize] == 0
                    {
                        births[household as usize] =
                            1 + (random(people, policy, base, day, 3).extract(0usize) & 1);
                        people[base + P::COOLDOWN] = policy[D::SPACING];
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

#[cube(launch)]
pub fn copy(source: &[u32], target: &mut [u32]) {
    let index = ABSOLUTE_POS;
    if index < source.len() {
        target[index] = source[index];
    }
}

#[cube(launch)]
pub fn observe(people: &[u32], policy: &[u32], output: &[Atomic<u32>], count: u32) {
    let index = ABSOLUTE_POS;
    if index < count as usize {
        let base = index * P::WORDS;
        let shard = index % OBSERVATION_SHARDS * 8;
        let cause = people[base + P::CAUSE];
        if cause == 0 {
            output[shard].fetch_add(1);
            let age = people[base + P::AGE];
            output[shard + 1].fetch_add(u32::cast_from(
                age < policy[D::MINIMUM_AGE] || age >= policy[D::RETIREMENT_AGE],
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

#[cube(launch)]
pub fn reduce_observation(shards: &[u32], output: &mut [u32]) {
    let metric = ABSOLUTE_POS;
    if metric < 7 {
        let mut total = zero();
        for shard in 0..OBSERVATION_SHARDS {
            let base = shard * 8 + metric;
            let mut high = 0;
            if metric == 6 {
                high = shards[base + 1];
            }
            total = plus(total, ticks(shards[base], high)).value;
        }
        output[metric] = total.extract(0usize);
        if metric == 6 {
            output[7usize] = total.extract(1usize);
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
            let draw = probability_draw(ticks(input[base + 6], input[base + 7]));
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
        let argument = |tensor: &CubeTensor| unsafe {
            BufferArg::from_raw_parts(tensor.handle.clone(), tensor.meta.shape()[0])
        };
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
    fn occupational_units_rare_risks_safety_and_full_width_draws_on_gpu() -> Result<()> {
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
        let argument = |tensor: &CubeTensor| unsafe {
            BufferArg::from_raw_parts(tensor.handle.clone(), tensor.meta.shape()[0])
        };
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
            let random = u64::from(case[6]) | (u64::from(case[7]) << 32);
            let expected = (u128::from(random) * u128::from(PROBABILITY_SCALE)) >> 64;
            let actual = u64::from(row[2]) | (u64::from(row[3]) << 32);
            assert_eq!(u128::from(actual), expected);
        }
        Ok(())
    }
}

#[cube(launch)]
pub(super) fn grow(
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

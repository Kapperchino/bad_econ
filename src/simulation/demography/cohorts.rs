use super::{DemographyPolicy, kernels::*};
use crate::domain::{
    Field,
    demography::{DAYS_PER_YEAR, PROBABILITY_SCALE, PopulationObservation},
};
use crate::simulation::{
    Device,
    configuration::{Configuration, EconomyConfig},
    layout::ActorLayout,
    system::AsBufferArg,
    words::*,
};
use anyhow::Result;
use burn_wgpu::CubeTensor;
use cubecl::prelude::*;

#[cube]
fn wide(value: Ticks) -> u64 {
    u64::cast_from(value.extract(0usize)) | (u64::cast_from(value.extract(1usize)) << 32)
}

#[cube]
fn events(count: u32, probability: u64, seed: u32, stream: u32) -> u32 {
    let mut draw = seed + stream * 0x9e3779b9u32;
    draw = (draw ^ (draw >> 16)) * 0x85ebca6bu32;
    draw = (draw ^ (draw >> 13)) * 0xc2b2ae35u32;
    draw ^= draw >> 16;
    let scale = comptime!(PROBABILITY_SCALE);
    let threshold = u64::cast_from(draw) * (scale >> 16) >> 16;
    let expected = u64::cast_from(count) * min(probability, scale);
    (expected / scale) as u32 + u32::cast_from(threshold < expected % scale)
}

#[cube]
fn share(amount: u32, subgroup: u32, population: u32, seed: u32, stream: u32) -> u32 {
    let proportion = u64::cast_from(subgroup) * comptime!(PROBABILITY_SCALE)
        / u64::cast_from(max(population, 1));
    clamp(
        events(amount, proportion, seed, stream),
        amount - min(amount, population - subgroup),
        min(amount, subgroup),
    )
}

#[cube]
pub(super) fn advance(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    policy: &DemographyPolicy,
    samples: &[f32],
    row: u32,
) {
    let living = get(state, config, comptime!(Field::LivingMembers as u32), row);
    if living > 0 {
        let seed = u32::cast_from(samples[row as usize] * 16777216.0);
        let food = get(state, config, comptime!(Field::NutritionNeed as u32), row);
        let eaten = get(state, config, comptime!(Field::FoodNeed as u32), row)
            - get(state, config, comptime!(Field::Shortage as u32), row);
        let deficit = (food - min(food, eaten)) * 1000 / max(food, 1);
        let heat = living * (max(config.heating_per_person, 2) - 2);
        let supplied = get(state, config, comptime!(Field::HeatingServed as u32), row);
        let freezing = (heat - min(heat, supplied)) * 1000 / max(heat, 1);
        let hunger = famine_exposure(
            get(
                state,
                config,
                comptime!(Field::CohortFamineExposure as u32),
                row,
            ),
            deficit,
            policy.parameters.recovery,
        );
        let cold = cold_exposure(
            get(
                state,
                config,
                comptime!(Field::CohortColdExposure as u32),
                row,
            ),
            freezing,
            policy.parameters.recovery,
        );
        put(
            state,
            config,
            comptime!(Field::CohortFamineExposure as u32),
            row,
            hunger,
        );
        put(
            state,
            config,
            comptime!(Field::CohortColdExposure as u32),
            row,
            cold,
        );

        let paid = get(state, config, comptime!(Field::Employed as u32), row);
        let employer = get(state, config, comptime!(Field::Employer as u32), row);
        let adults = get(state, config, comptime!(Field::YoungAdults as u32), row)
            + get(state, config, comptime!(Field::OlderAdults as u32), row);
        let fatigue = get(state, config, comptime!(Field::Fatigue as u32), row);
        let mut occupational = 0u64;
        if employer > 0 && paid > 0 {
            let firm = employer - 1;
            let good =
                actors[ActorLayout::KIND * (actors.len() / ActorLayout::WORDS) + firm as usize];
            let shift = get(state, config, comptime!(Field::Hours as u32), row) / paid;
            occupational = wide(occupation(
                policy.occupational[good as usize - 1],
                shift,
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
                fatigue,
                get(state, config, comptime!(Field::ActiveMethod as u32), firm),
            )) * u64::cast_from(paid)
                / u64::cast_from(max(adults, 1));
        }
        let home = get(state, config, comptime!(Field::HomeMinutes as u32), row) / max(adults, 1);
        let cartage =
            get(state, config, comptime!(Field::CartageMinutes as u32), row) / max(adults, 1);
        occupational += wide(occupation(
            policy.occupational[comptime!(crate::domain::Good::Railway as usize - 1)],
            cartage,
            policy.parameters.reference_minutes,
            1000,
            0,
            fatigue,
            0,
        ));
        occupational += wide(occupation(
            policy.occupational[0],
            home,
            policy.parameters.reference_minutes,
            1000,
            0,
            fatigue,
            0,
        ));
        let supported = family_support(state, config, row, food + config.household_need) == 0
            && (comptime!(config.economy == crate::domain::Economy::Agrarian)
                || max(hunger, cold) == 0);
        let mut survivors = 0u32;
        let mut births = 0u32;
        let mut incoming = 0u32;
        let mut incoming_female = 0u32;
        let minimum = policy.parameters.minimum_age;
        let retirement = policy.parameters.retirement_age;
        let middle = min(45 * DAYS_PER_YEAR, retirement);
        for band in 0..5u32 {
            let field = comptime!(Field::Infants as u32) + band;
            let count = get(state, config, field, row);
            let mut remaining = count;
            let mut cumulative = 0u64;
            let mut allocated = 0u32;
            for cause in 0..4u32 {
                let probability = match cause {
                    0 => wide(rate(policy.background[band as usize])),
                    1 => select(band == 2 || band == 3, occupational, 0u64),
                    2 => wide(famine_risk(
                        hunger,
                        policy.parameters.famine,
                        policy.vulnerability[band as usize],
                    )),
                    _ => wide(cold_risk(
                        cold,
                        policy.parameters.cold,
                        policy.cold_vulnerability[band as usize],
                    )),
                };
                cumulative += probability;
                let total = events(count, cumulative, seed, band);
                let died = total - allocated;
                allocated = total;
                let coordinate =
                    (comptime!(Field::BackgroundDeaths as u32) + cause) * config.state_stride + row;
                let deaths = plus(load(state, coordinate), ticks(died, 0)).value;
                store(state, coordinate, deaths);
                remaining -= died;
            }
            let female_field = comptime!(Field::FemaleInfants as u32) + band;
            let female = get(state, config, female_field, row);
            let surviving_female = share(remaining, female, count, seed, band + 12);
            let duration = match band {
                0 => DAYS_PER_YEAR,
                1 => minimum - DAYS_PER_YEAR,
                2 => middle - minimum,
                _ => max(retirement - middle, 1),
            };
            let aging = select(
                band < 4,
                events(
                    remaining,
                    comptime!(PROBABILITY_SCALE) / u64::cast_from(duration),
                    seed,
                    band + 5,
                ),
                0,
            );
            let aging_female = share(aging, surviving_female, remaining, seed, band + 17);
            let retained_female = surviving_female - aging_female;
            let female_occupants = retained_female + incoming_female;
            if band == 2 {
                let recovering = get(
                    state,
                    config,
                    comptime!(Field::RecoveringMothers as u32),
                    row,
                );
                let retained = share(retained_female, recovering, female, seed, 22);
                let recovered = events(
                    retained,
                    comptime!(PROBABILITY_SCALE) / u64::cast_from(policy.parameters.spacing),
                    seed,
                    23,
                );
                let waiting = retained - recovered;
                let fertile_days = middle - max(minimum, 18 * DAYS_PER_YEAR);
                let fertility = wide(rate(policy.parameters.births)) * u64::cast_from(fertile_days)
                    / u64::cast_from(middle - minimum);
                births = select(
                    supported,
                    events(retained_female - waiting, fertility, seed, 11),
                    0,
                );
                put(
                    state,
                    config,
                    comptime!(Field::RecoveringMothers as u32),
                    row,
                    waiting + births,
                );
            }
            let occupants = remaining - aging + incoming;
            let bypass = band == 3 && retirement == middle;
            put(state, config, field, row, select(bypass, 0, occupants));
            put(
                state,
                config,
                female_field,
                row,
                select(bypass, 0, female_occupants),
            );
            incoming_female = select(bypass, aging_female + female_occupants, aging_female);
            incoming = select(bypass, aging + occupants, aging);
            survivors += remaining;
        }
        let infants = get(state, config, comptime!(Field::Infants as u32), row) + births;
        put(
            state,
            config,
            comptime!(Field::Infants as u32),
            row,
            infants,
        );
        let female_infants = get(state, config, comptime!(Field::FemaleInfants as u32), row)
            + events(births, comptime!(PROBABILITY_SCALE / 2), seed, 24);
        put(
            state,
            config,
            comptime!(Field::FemaleInfants as u32),
            row,
            female_infants,
        );
        let coordinate = comptime!(Field::CohortBirths as u32) * config.state_stride + row;
        let born = plus(load(state, coordinate), ticks(births, 0)).value;
        store(state, coordinate, born);
        put(
            state,
            config,
            comptime!(Field::LivingMembers as u32),
            row,
            survivors + births,
        );
        let workers = get(state, config, comptime!(Field::YoungAdults as u32), row)
            + get(state, config, comptime!(Field::OlderAdults as u32), row);
        put(
            state,
            config,
            comptime!(Field::CohortDelegate as u32),
            row,
            u32::cast_from(workers > 0),
        );
    }
}

const METRICS: usize = 16;

#[cube(launch)]
fn observe(state: &mut [u32], config: &EconomyConfig, output: &mut [u64], #[comptime] lanes: u32) {
    let row = ABSOLUTE_POS as u32;
    let mut totals = Array::<u64>::new(METRICS);
    #[unroll]
    for metric in 0..METRICS {
        totals[metric] = 0;
    }
    if row < config.actors {
        let living = get(state, config, comptime!(Field::LivingMembers as u32), row);
        let adults = get(state, config, comptime!(Field::YoungAdults as u32), row)
            + get(state, config, comptime!(Field::OlderAdults as u32), row);
        let investment = get(state, config, comptime!(Field::InvestmentFirm as u32), row);
        let owner = u32::cast_from(
            adults > 0
                && investment > 0
                && get(
                    state,
                    config,
                    comptime!(Field::EnterpriseStatus as u32),
                    max(investment, 1) - 1,
                ) < 4,
        );
        let farmers = select(
            get(state, config, comptime!(Field::LandPlots as u32), row) > 0,
            adults - owner,
            0,
        );
        let paid = min(
            get(state, config, comptime!(Field::Employed as u32), row),
            adults - owner - farmers,
        );
        totals[0usize] = u64::cast_from(living);
        totals[1usize] = u64::cast_from(living - adults);
        for metric in 0..5u32 {
            totals[2 + metric as usize] = wide(load(
                state,
                (comptime!(Field::CohortBirths as u32) + metric) * config.state_stride + row,
            ));
        }
        totals[7usize] = u64::cast_from(get(state, config, comptime!(Field::Hours as u32), row))
            + u64::cast_from(select(
                get(state, config, comptime!(Field::HouseholdBudget as u32), row) > 0,
                get(state, config, comptime!(Field::CartageMinutes as u32), row),
                0,
            ))
            + u64::cast_from(get(
                state,
                config,
                comptime!(Field::HomeMinutes as u32),
                row,
            ));
        totals[8usize] = u64::cast_from(living)
            * u64::cast_from(get(
                state,
                config,
                comptime!(Field::CohortFamineExposure as u32),
                row,
            ));
        totals[9usize] = u64::cast_from(living)
            * u64::cast_from(get(
                state,
                config,
                comptime!(Field::CohortColdExposure as u32),
                row,
            ));
        totals[10usize] = u64::cast_from(farmers);
        totals[11usize] = u64::cast_from(paid);
        totals[12usize] = u64::cast_from(owner);
        totals[13usize] = u64::cast_from(adults - owner - farmers - paid);
        totals[14usize] = u64::cast_from(living - adults);
        totals[15usize] = 0u64;
    }
    #[unroll]
    for metric in 0..METRICS {
        let value = totals[metric];
        if comptime!(lanes > 1) {
            let low = plane_sum(u32::cast_from(value) & 0xffff);
            let middle = plane_sum(u32::cast_from(value) >> 16);
            let high = plane_sum(u32::cast_from(value >> 32));
            if UNIT_POS_PLANE == 0 && row < config.actors {
                output[(row / comptime!(lanes)) as usize * METRICS + metric] = u64::cast_from(low)
                    + (u64::cast_from(middle) << 16)
                    + (u64::cast_from(high) << 32);
            }
        } else if row < config.actors {
            output[row as usize * METRICS + metric] = value;
        }
    }
}

pub(super) fn observation(
    device: &Device,
    state: &CubeTensor,
    config: &Configuration,
    initial_population: u64,
) -> Result<PopulationObservation> {
    let rows = config.actors as usize;
    let hardware = &device.client.properties().hardware;
    let lanes = match (
        crate::simulation::parallel::ScanMode::new(device),
        hardware.plane_size_min,
        hardware.plane_size_max,
    ) {
        (crate::simulation::parallel::ScanMode::Plane, 32, 32)
            if device.workgroup.is_multiple_of(32) =>
        {
            32u32
        }
        _ => 1,
    };
    let partials = rows.div_ceil(lanes as usize);
    let totals = device.empty(partials * METRICS * 2);
    observe::launch(
        &device.client,
        device.count(rows),
        CubeDim::new_1d(device.workgroup),
        state.buffer_arg(),
        config.launch(),
        totals.buffer_arg(),
        lanes,
    );
    let dtype = <u64 as Scalar>::elem_type_native();
    let input = cubecl::std::tensor::TensorHandle::new_contiguous(
        [partials, METRICS],
        totals.handle,
        dtype,
    );
    let sums = crate::simulation::reduction::reduce(
        device,
        &input,
        0,
        cubek::reduce::components::instructions::ReduceOperationConfig::Sum,
        dtype,
    )?;
    let bytes = device.client.read_one(sums.handle)?;
    let totals: &[u64] = bytemuck::cast_slice(&bytes);
    let observation = PopulationObservation {
        living: totals[0],
        dependents: totals[1],
        births: totals[2],
        deaths: totals[3..7].try_into()?,
        work_minutes: totals[7],
        famine_exposure: totals[8],
        cold_exposure: totals[9],
        occupations: totals[10..16].try_into()?,
    };
    let classified = observation.living + observation.deaths.iter().sum::<u64>();
    match initial_population + observation.births == classified
        && observation.occupations.iter().sum::<u64>() == observation.living
    {
        true => Ok(observation),
        false => Err(anyhow::anyhow!("cohort population does not balance")),
    }
}

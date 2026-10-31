use super::{
    DAY_MINUTES, HOUSEHOLD_SCALE, NORMAL_WORK_MINUTES, ObservationKind, Participation,
    RequestLayout, ResultLayout, ServiceLayout,
};
use crate::simulation::{
    layout::{HIGH_WORD, LOW_WORD},
    words::*,
};
use cubecl::prelude::*;

#[derive(CubeType, Clone, Copy)]
pub struct KernelDeficit {
    pub lower: u32,
    pub upper: u32,
}

#[cube]
fn product96(value: Ticks, weight: u32) -> Vector<u32, Const<3>> {
    let low = times32(value.extract(LOW_WORD), weight);
    let high = times32(value.extract(HIGH_WORD), weight);
    let middle = low.extract(HIGH_WORD) + high.extract(LOW_WORD);
    let mut result = Vector::<u32, Const<3>>::new(low.extract(LOW_WORD));
    result.insert(1usize, middle);
    result.insert(
        2usize,
        high.extract(HIGH_WORD) + u32::cast_from(middle < low.extract(HIGH_WORD)),
    );
    result
}

#[cube]
fn wide_le(left: Vector<u32, Const<3>>, right: Vector<u32, Const<3>>) -> bool {
    left.extract(2usize) < right.extract(2usize)
        || (left.extract(2usize) == right.extract(2usize)
            && left.extract(1usize) < right.extract(1usize))
        || (left.extract(2usize) == right.extract(2usize)
            && left.extract(1usize) == right.extract(1usize)
            && left.extract(0usize) <= right.extract(0usize))
}

#[cube]
pub fn service_deficit(need: Ticks, delivered: Ticks, weight: u32) -> KernelDeficit {
    let mut lower = 0u32;
    let mut upper = 0u32;
    if !equal(need, zero()) {
        let numerator = product96(positive(need, delivered), weight);
        let mut high = weight + 1;
        for _ in 0..10u32 {
            let middle = (lower + high) / 2;
            let covered = wide_le(product96(need, middle), numerator);
            lower = select(covered, middle, lower);
            high = select(covered, high, middle);
        }
        let exact = wide_le(numerator, product96(need, lower));
        upper = lower + u32::cast_from(!exact);
    }
    KernelDeficit { lower, upper }
}

#[derive(CubeType, Clone, Copy)]
pub struct KernelCapacity {
    pub participation: u32,
    pub capacity_minutes: u32,
    pub health: u32,
    pub fatigue: u32,
    pub reserved_minutes: u32,
}

#[cube]
pub fn available_minutes(request: KernelCapacity) -> u32 {
    let physical = request.capacity_minutes * request.health * (HOUSEHOLD_SCALE - request.fatigue)
        / (HOUSEHOLD_SCALE * HOUSEHOLD_SCALE);
    let clock = DAY_MINUTES - min(request.reserved_minutes, DAY_MINUTES);
    select(
        request.participation == comptime!(Participation::Available as u32),
        min(physical, clock),
        0,
    )
}

#[derive(CubeType, Clone, Copy)]
pub struct KernelHealth {
    pub health: u32,
    pub fatigue: u32,
    pub deficit: u32,
    pub recovery: u32,
    pub deficit_damage: u32,
    pub fatigue_damage: u32,
}

#[cube]
pub fn next_health(request: KernelHealth) -> u32 {
    let scale_squared = HOUSEHOLD_SCALE * HOUSEHOLD_SCALE;
    let recovery =
        request.recovery * (HOUSEHOLD_SCALE - request.health) * (HOUSEHOLD_SCALE - request.deficit);
    let damage = (request.deficit_damage * request.deficit
        + request.fatigue_damage * request.fatigue)
        * HOUSEHOLD_SCALE;
    let restored = request.health * scale_squared + recovery;
    min(
        (max(restored, damage) - damage) / scale_squared,
        HOUSEHOLD_SCALE,
    )
}

#[derive(CubeType, Clone, Copy)]
pub struct KernelFatigue {
    pub fatigue: u32,
    pub response: u32,
    pub exertion_minutes: u32,
    pub rest_minutes: u32,
}

#[cube]
pub fn next_fatigue(request: KernelFatigue) -> u32 {
    let tired = request.fatigue * NORMAL_WORK_MINUTES + request.response * request.exertion_minutes;
    let rested = request.response * request.rest_minutes;
    min(
        (max(tired, rested) - rested) / NORMAL_WORK_MINUTES,
        HOUSEHOLD_SCALE,
    )
}

#[derive(CubeType, Clone, Copy)]
pub struct KernelObservation {
    pub kind: u32,
    pub value: Ticks,
}

#[cube]
pub fn observed_blend(
    previous: Ticks,
    observation: KernelObservation,
    weight: u32,
    control: &[Atomic<u32>],
) -> Ticks {
    let mut output = previous;
    if observation.kind == comptime!(ObservationKind::Observed as u32) {
        let baseline = minimum(previous, observation.value);
        let difference = minus(maximum(previous, observation.value), baseline);
        let selected_weight = select(
            lt(observation.value, previous),
            HOUSEHOLD_SCALE - weight,
            weight,
        );
        let delta = ratio(difference, selected_weight, HOUSEHOLD_SCALE);
        let increased = plus(baseline, delta.value);
        delta.faults.union(increased.faults).record(control);
        output = increased.value;
    }
    output
}

#[cube]
fn read(input: &[u32], offset: usize) -> Ticks {
    ticks(input[offset], input[offset + 1])
}

#[cube(launch)]
pub fn evaluate(
    requests: &[u32],
    services: &[u32],
    output: &mut [u32],
    control: &[Atomic<u32>],
    count: u32,
) {
    let index = ABSOLUTE_POS;
    if index < count as usize {
        let base = index * RequestLayout::WORDS;
        let target = index * ResultLayout::WORDS;
        let mut lower = 0u32;
        let mut upper = 0u32;
        let start = requests[base + RequestLayout::SERVICE_OFFSET];
        let end = start + requests[base + RequestLayout::SERVICE_COUNT];
        for service in start..end {
            let source = service as usize * ServiceLayout::WORDS;
            let deficit = service_deficit(
                read(services, source + ServiceLayout::NEED),
                read(services, source + ServiceLayout::DELIVERED),
                services[source + ServiceLayout::WEIGHT],
            );
            lower += deficit.lower;
            upper += deficit.upper;
        }
        let available = available_minutes(KernelCapacity {
            participation: requests[base + RequestLayout::PARTICIPATION],
            capacity_minutes: requests[base + RequestLayout::CAPACITY],
            health: requests[base + RequestLayout::HEALTH],
            fatigue: requests[base + RequestLayout::FATIGUE],
            reserved_minutes: requests[base + RequestLayout::RESERVED],
        });
        let health = next_health(KernelHealth {
            health: requests[base + RequestLayout::HEALTH],
            fatigue: requests[base + RequestLayout::FATIGUE],
            deficit: lower,
            recovery: requests[base + RequestLayout::RECOVERY],
            deficit_damage: requests[base + RequestLayout::DEFICIT_DAMAGE],
            fatigue_damage: requests[base + RequestLayout::FATIGUE_DAMAGE],
        });
        let fatigue = next_fatigue(KernelFatigue {
            fatigue: requests[base + RequestLayout::FATIGUE],
            response: requests[base + RequestLayout::FATIGUE_RESPONSE],
            exertion_minutes: requests[base + RequestLayout::EXERTION],
            rest_minutes: requests[base + RequestLayout::REST],
        });
        let forecast = observed_blend(
            read(requests, base + RequestLayout::FORECAST),
            KernelObservation {
                kind: requests[base + RequestLayout::OBSERVATION],
                value: read(requests, base + RequestLayout::SIGNAL),
            },
            requests[base + RequestLayout::WEIGHT],
            control,
        );
        output[target + ResultLayout::FORECAST + LOW_WORD] = forecast.extract(LOW_WORD);
        output[target + ResultLayout::FORECAST + HIGH_WORD] = forecast.extract(HIGH_WORD);
        output[target + ResultLayout::DEFICIT_LOWER] = lower;
        output[target + ResultLayout::DEFICIT_UPPER] = upper;
        output[target + ResultLayout::AVAILABLE] = available;
        output[target + ResultLayout::HEALTH] = health;
        output[target + ResultLayout::FATIGUE] = fatigue;
    }
}

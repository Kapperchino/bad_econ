use super::{BARGAINING_SCALE, RequestLayout, ResultLayout};
use crate::simulation::{
    flags::KernelFaults,
    layout::{HIGH_WORD, LOW_WORD},
    pass::GpuPass,
    system::{System, gpu_passes},
    words::*,
};
use cubecl::prelude::*;

#[cube]
fn add(left: Ticks, right: Ticks, control: &[Atomic<u32>]) -> Ticks {
    let result = plus(left, right);
    result.faults.record(control);
    result.value
}

#[cube]
fn ceil_ratio(value: Ticks, numerator: u32, denominator: u32, control: &[Atomic<u32>]) -> Ticks {
    let floor = ratio(value, numerator, denominator);
    floor.faults.record(control);
    let remainder = minus(value, times(divide(value, denominator), denominator).value);
    let fraction = times32(remainder.extract(LOW_WORD), numerator);
    let residual = minus(
        fraction,
        times(divide(fraction, denominator), denominator).value,
    );
    add(
        floor.value,
        ticks(u32::cast_from(!equal(residual, zero())), 0),
        control,
    )
}

#[cube]
fn grid_floor(value: Ticks, quantum: u32, control: &[Atomic<u32>]) -> Ticks {
    let result = times(divide(value, quantum), quantum);
    result.faults.record(control);
    result.value
}

#[cube]
fn grid_ceil(value: Ticks, quantum: u32, control: &[Atomic<u32>]) -> Ticks {
    let result = times(ceil_ratio(value, 1, quantum, control), quantum);
    result.faults.record(control);
    result.value
}

#[cube]
pub fn inverse_bracket(
    threshold: Ticks,
    net_at_threshold: Ticks,
    slope: u32,
    target: Ticks,
    control: &[Atomic<u32>],
) -> Ticks {
    add(
        threshold,
        ceil_ratio(
            positive(target, net_at_threshold),
            BARGAINING_SCALE,
            slope,
            control,
        ),
        control,
    )
}

#[derive(CubeType, Clone, Copy)]
pub struct KernelWageBounds {
    pub legal: Ticks,
    pub gross_needed: Ticks,
    pub funded: Ticks,
    pub payable_hours: u32,
    pub employer_charge: u32,
    pub quantum: u32,
}

#[derive(CubeType, Clone, Copy)]
pub struct WageQuote {
    pub minimum: Ticks,
    pub maximum: Ticks,
}

#[cube]
pub fn wage_bounds(request: KernelWageBounds, control: &[Atomic<u32>]) -> WageQuote {
    let floor = grid_ceil(
        maximum(
            request.legal,
            ceil_ratio(request.gross_needed, 1, request.payable_hours, control),
        ),
        request.quantum,
        control,
    );
    let cap = ratio(
        request.funded,
        BARGAINING_SCALE,
        (BARGAINING_SCALE + request.employer_charge) * request.payable_hours,
    );
    cap.faults.record(control);
    WageQuote {
        minimum: floor,
        maximum: grid_floor(cap.value, request.quantum, control),
    }
}

#[cube]
pub fn employer_ceiling(
    receipts: Ticks,
    nonlabor: Ticks,
    contingency: Ticks,
    budget: Ticks,
    control: &[Atomic<u32>],
) -> Ticks {
    minimum(
        positive(receipts, add(nonlabor, contingency, control)),
        budget,
    )
}

#[derive(CubeType, Clone, Copy)]
pub struct KernelHousehold {
    pub cash_cost: Ticks,
    pub funded_other: Ticks,
    pub participating_adults: u32,
    pub normal_day: u32,
    pub cover_millidays: u32,
    pub work_cost: Ticks,
}

#[derive(CubeType, Clone, Copy)]
pub struct HouseholdQuote {
    pub anchor: Ticks,
    pub required_net: Ticks,
}

#[cube]
pub fn household_quote(request: KernelHousehold, control: &[Atomic<u32>]) -> HouseholdQuote {
    let base = positive(request.cash_cost, request.funded_other);
    let hours = request.participating_adults * request.normal_day;
    let anchor = ceil_ratio(base, 1, hours, control);
    let refusal = select(
        equal(request.cash_cost, zero()),
        42_000,
        21_000 + 2 * min(request.cover_millidays, 10_500),
    );
    let required = ceil_ratio(base, refusal, 35_000 * hours, control);
    HouseholdQuote {
        anchor,
        required_net: add(required, request.work_cost, control),
    }
}

#[derive(CubeType, Clone, Copy)]
pub struct KernelBargainingFactors {
    pub mandate: u32,
    pub withdrawal: u32,
    pub alternatives: u32,
    pub concentration: u32,
    pub replacements: u32,
}

#[cube]
pub fn bargaining_weight(factors: KernelBargainingFactors) -> u32 {
    let positive_score = 7 * BARGAINING_SCALE
        + 5 * factors.mandate
        + 4 * factors.withdrawal
        + 3 * factors.alternatives;
    let negative_score = 4 * factors.concentration + 3 * factors.replacements;
    max(
        BARGAINING_SCALE,
        positive_score - min(positive_score, negative_score),
    ) / 20
}

#[cube]
pub fn second_offer(
    first: Ticks,
    desired: Ticks,
    weight: u32,
    minimum_wage: Ticks,
    maximum_wage: Ticks,
    quantum: u32,
    control: &[Atomic<u32>],
) -> Ticks {
    let difference = minus(maximum(first, desired), minimum(first, desired));
    let upward = ratio(difference, weight, BARGAINING_SCALE);
    upward.faults.record(control);
    let downward = ceil_ratio(difference, weight, BARGAINING_SCALE, control);
    let rising = plus(first, upward.value);
    let mut faults = KernelFaults::none();
    if !lt(desired, first) {
        faults.insert(rising.faults);
    }
    faults.record(control);
    let blended = select(lt(desired, first), minus(first, downward), rising.value);
    minimum(
        maximum(grid_floor(blended, quantum, control), minimum_wage),
        maximum_wage,
    )
}

#[cube]
fn read(input: &[u32], offset: usize) -> Ticks {
    ticks(input[offset], input[offset + 1])
}

#[cube]
fn write(output: &mut [u32], offset: usize, value: Ticks) {
    output[offset] = value.extract(LOW_WORD);
    output[offset + 1] = value.extract(HIGH_WORD);
}

#[derive(CubeType, CubeLaunch)]
pub(super) struct BargainingInput {
    pub requests: Box<[u32]>,
    pub output: Box<[u32]>,
    pub control: Box<[Atomic<u32>]>,
    pub count: u32,
}

pub(super) struct BargainingEvaluation;
pub(super) struct EvaluateBargaining;

impl System for BargainingEvaluation {
    type Input = BargainingInput;
    type Passes = gpu_passes![EvaluateBargaining];
}

#[cube]
impl GpuPass<BargainingInput> for EvaluateBargaining {
    type Output = ();

    fn execute(mut input: BargainingInput) {
        evaluate(
            &input.requests,
            &mut input.output,
            &input.control,
            input.count,
        );
    }
}

#[cube]
fn evaluate(input: &[u32], output: &mut [u32], control: &[Atomic<u32>], count: u32) {
    let index = ABSOLUTE_POS;
    if index < count as usize {
        let base = index * RequestLayout::WORDS;
        let target = index * ResultLayout::WORDS;
        let gross = inverse_bracket(
            read(input, base + RequestLayout::THRESHOLD),
            read(input, base + RequestLayout::NET_THRESHOLD),
            input[base + RequestLayout::SLOPE],
            read(input, base + RequestLayout::TARGET),
            control,
        );
        let required_gross = positive(gross, read(input, base + RequestLayout::PRIOR));
        let wages = wage_bounds(
            KernelWageBounds {
                legal: read(input, base + RequestLayout::LEGAL),
                gross_needed: required_gross,
                funded: read(input, base + RequestLayout::FUNDED),
                payable_hours: input[base + RequestLayout::HOURS],
                employer_charge: input[base + RequestLayout::EMPLOYER_CHARGE],
                quantum: input[base + RequestLayout::QUANTUM],
            },
            control,
        );
        let ceiling = employer_ceiling(
            read(input, base + RequestLayout::RECEIPTS),
            read(input, base + RequestLayout::NONLABOR),
            read(input, base + RequestLayout::CONTINGENCY),
            read(input, base + RequestLayout::BUDGET),
            control,
        );
        let household = household_quote(
            KernelHousehold {
                cash_cost: read(input, base + RequestLayout::CASH_COST),
                funded_other: read(input, base + RequestLayout::FUNDED_OTHER),
                participating_adults: input[base + RequestLayout::ADULTS],
                normal_day: input[base + RequestLayout::NORMAL_DAY],
                cover_millidays: input[base + RequestLayout::COVER],
                work_cost: read(input, base + RequestLayout::WORK_COST),
            },
            control,
        );
        let weight = bargaining_weight(KernelBargainingFactors {
            mandate: input[base + RequestLayout::MANDATE],
            withdrawal: input[base + RequestLayout::WITHDRAWAL],
            alternatives: input[base + RequestLayout::ALTERNATIVES],
            concentration: input[base + RequestLayout::CONCENTRATION],
            replacements: input[base + RequestLayout::REPLACEMENTS],
        });
        let available = !lt(wages.maximum, wages.minimum);
        let mut second = zero();
        if available {
            second = second_offer(
                read(input, base + RequestLayout::FIRST),
                read(input, base + RequestLayout::DESIRED),
                weight,
                wages.minimum,
                wages.maximum,
                input[base + RequestLayout::QUANTUM],
                control,
            );
        }
        write(output, target + ResultLayout::GROSS, gross);
        write(
            output,
            target + ResultLayout::REQUIRED_GROSS,
            required_gross,
        );
        write(output, target + ResultLayout::MINIMUM_WAGE, wages.minimum);
        write(output, target + ResultLayout::MAXIMUM_WAGE, wages.maximum);
        write(output, target + ResultLayout::PAYROLL_CEILING, ceiling);
        write(output, target + ResultLayout::ANCHOR, household.anchor);
        write(
            output,
            target + ResultLayout::REQUIRED_NET,
            household.required_net,
        );
        write(output, target + ResultLayout::SECOND, second);
        output[target + ResultLayout::WEIGHT] = weight;
        output[target + ResultLayout::AVAILABLE] = u32::cast_from(available);
    }
}

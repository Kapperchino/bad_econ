use super::{BARGAINING_SCALE, RequestLayout, ResultLayout};
use crate::simulation::{
    pass::GpuPass,
    storage::{flags::KernelFaults, words::*},
    system::{System, gpu_passes},
};
use cubecl::prelude::*;

#[derive(CubeType)]
struct WageArithmetic<'a> {
    control: &'a [Atomic<u32>],
}

#[cube]
impl<'a> WageArithmetic<'a> {
    fn new(control: &'a [Atomic<u32>]) -> Self {
        WageArithmetic::<'a> { control }
    }

    fn add(&self, left: Ticks, right: Ticks) -> Ticks {
        plus(left, right).record(self.control)
    }

    fn ceil_ratio(&self, value: Ticks, numerator: u32, denominator: u32) -> Ticks {
        CheckedTicks::ceiling_ratio(value, numerator, denominator).record(self.control)
    }

    fn grid_floor(&self, value: Ticks, quantum: u32) -> Ticks {
        times(divide(value, quantum), quantum).record(self.control)
    }

    fn grid_ceil(&self, value: Ticks, quantum: u32) -> Ticks {
        times(self.ceil_ratio(value, 1, quantum), quantum).record(self.control)
    }
}

#[cube]
pub fn inverse_bracket(
    threshold: Ticks,
    net_at_threshold: Ticks,
    slope: u32,
    target: Ticks,
    control: &[Atomic<u32>],
) -> Ticks {
    let arithmetic = WageArithmetic::new(control);
    arithmetic.add(
        threshold,
        arithmetic.ceil_ratio(positive(target, net_at_threshold), BARGAINING_SCALE, slope),
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
impl KernelWageBounds {
    pub fn quote(&self, control: &[Atomic<u32>]) -> WageQuote {
        let arithmetic = WageArithmetic::new(control);
        let floor = arithmetic.grid_ceil(
            maximum(
                self.legal,
                arithmetic.ceil_ratio(self.gross_needed, 1, self.payable_hours),
            ),
            self.quantum,
        );
        let cap = ratio(
            self.funded,
            BARGAINING_SCALE,
            (BARGAINING_SCALE + self.employer_charge) * self.payable_hours,
        );
        cap.faults.record(control);
        WageQuote {
            minimum: floor,
            maximum: arithmetic.grid_floor(cap.value, self.quantum),
        }
    }
}

#[cube]
pub fn wage_bounds(request: KernelWageBounds, control: &[Atomic<u32>]) -> WageQuote {
    request.quote(control)
}

#[cube]
pub fn employer_ceiling(
    receipts: Ticks,
    nonlabor: Ticks,
    contingency: Ticks,
    budget: Ticks,
    control: &[Atomic<u32>],
) -> Ticks {
    let arithmetic = WageArithmetic::new(control);
    minimum(
        positive(receipts, arithmetic.add(nonlabor, contingency)),
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
impl KernelHousehold {
    pub fn quote(&self, control: &[Atomic<u32>]) -> HouseholdQuote {
        let arithmetic = WageArithmetic::new(control);
        let base = positive(self.cash_cost, self.funded_other);
        let hours = self.participating_adults * self.normal_day;
        let anchor = arithmetic.ceil_ratio(base, 1, hours);
        let refusal = select(
            equal(self.cash_cost, zero()),
            42_000,
            21_000 + 2 * min(self.cover_millidays, 10_500),
        );
        let required = arithmetic.ceil_ratio(base, refusal, 35_000 * hours);
        HouseholdQuote {
            anchor,
            required_net: arithmetic.add(required, self.work_cost),
        }
    }
}

#[cube]
pub fn household_quote(request: KernelHousehold, control: &[Atomic<u32>]) -> HouseholdQuote {
    request.quote(control)
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
impl KernelBargainingFactors {
    pub fn weight(&self) -> u32 {
        let positive_score =
            7 * BARGAINING_SCALE + 5 * self.mandate + 4 * self.withdrawal + 3 * self.alternatives;
        let negative_score = 4 * self.concentration + 3 * self.replacements;
        max(
            BARGAINING_SCALE,
            positive_score - min(positive_score, negative_score),
        ) / 20
    }
}

#[cube]
pub fn bargaining_weight(factors: KernelBargainingFactors) -> u32 {
    factors.weight()
}

#[cube]
impl WageQuote {
    pub fn second_offer(
        &self,
        first: Ticks,
        desired: Ticks,
        weight: u32,
        quantum: u32,
        control: &[Atomic<u32>],
    ) -> Ticks {
        let arithmetic = WageArithmetic::new(control);
        let difference = minus(maximum(first, desired), minimum(first, desired));
        let upward = ratio(difference, weight, BARGAINING_SCALE);
        upward.faults.record(control);
        let downward = arithmetic.ceil_ratio(difference, weight, BARGAINING_SCALE);
        let rising = plus(first, upward.value);
        let mut faults = KernelFaults::none();
        if !lt(desired, first) {
            faults.insert(rising.faults);
        }
        faults.record(control);
        let blended = select(lt(desired, first), minus(first, downward), rising.value);
        minimum(
            maximum(arithmetic.grid_floor(blended, quantum), self.minimum),
            self.maximum,
        )
    }
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
    WageQuote {
        minimum: minimum_wage,
        maximum: maximum_wage,
    }
    .second_offer(first, desired, weight, quantum, control)
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
        let index = ABSOLUTE_POS;
        if index < input.count as usize {
            let base = index * RequestLayout::WORDS;
            let target = index * ResultLayout::WORDS;
            let gross = inverse_bracket(
                read_tick_words(&input.requests, base + RequestLayout::THRESHOLD),
                read_tick_words(&input.requests, base + RequestLayout::NET_THRESHOLD),
                input.requests[base + RequestLayout::SLOPE],
                read_tick_words(&input.requests, base + RequestLayout::TARGET),
                &input.control,
            );
            let required_gross = positive(
                gross,
                read_tick_words(&input.requests, base + RequestLayout::PRIOR),
            );
            let wages = KernelWageBounds {
                legal: read_tick_words(&input.requests, base + RequestLayout::LEGAL),
                gross_needed: required_gross,
                funded: read_tick_words(&input.requests, base + RequestLayout::FUNDED),
                payable_hours: input.requests[base + RequestLayout::HOURS],
                employer_charge: input.requests[base + RequestLayout::EMPLOYER_CHARGE],
                quantum: input.requests[base + RequestLayout::QUANTUM],
            }
            .quote(&input.control);
            let ceiling = employer_ceiling(
                read_tick_words(&input.requests, base + RequestLayout::RECEIPTS),
                read_tick_words(&input.requests, base + RequestLayout::NONLABOR),
                read_tick_words(&input.requests, base + RequestLayout::CONTINGENCY),
                read_tick_words(&input.requests, base + RequestLayout::BUDGET),
                &input.control,
            );
            let household = KernelHousehold {
                cash_cost: read_tick_words(&input.requests, base + RequestLayout::CASH_COST),
                funded_other: read_tick_words(&input.requests, base + RequestLayout::FUNDED_OTHER),
                participating_adults: input.requests[base + RequestLayout::ADULTS],
                normal_day: input.requests[base + RequestLayout::NORMAL_DAY],
                cover_millidays: input.requests[base + RequestLayout::COVER],
                work_cost: read_tick_words(&input.requests, base + RequestLayout::WORK_COST),
            }
            .quote(&input.control);
            let weight = KernelBargainingFactors {
                mandate: input.requests[base + RequestLayout::MANDATE],
                withdrawal: input.requests[base + RequestLayout::WITHDRAWAL],
                alternatives: input.requests[base + RequestLayout::ALTERNATIVES],
                concentration: input.requests[base + RequestLayout::CONCENTRATION],
                replacements: input.requests[base + RequestLayout::REPLACEMENTS],
            }
            .weight();
            let available = !lt(wages.maximum, wages.minimum);
            let second = if available {
                wages.second_offer(
                    read_tick_words(&input.requests, base + RequestLayout::FIRST),
                    read_tick_words(&input.requests, base + RequestLayout::DESIRED),
                    weight,
                    input.requests[base + RequestLayout::QUANTUM],
                    &input.control,
                )
            } else {
                zero()
            };
            write_tick_words(&mut input.output, target + ResultLayout::GROSS, gross);
            write_tick_words(
                &mut input.output,
                target + ResultLayout::REQUIRED_GROSS,
                required_gross,
            );
            write_tick_words(
                &mut input.output,
                target + ResultLayout::MINIMUM_WAGE,
                wages.minimum,
            );
            write_tick_words(
                &mut input.output,
                target + ResultLayout::MAXIMUM_WAGE,
                wages.maximum,
            );
            write_tick_words(
                &mut input.output,
                target + ResultLayout::PAYROLL_CEILING,
                ceiling,
            );
            write_tick_words(
                &mut input.output,
                target + ResultLayout::ANCHOR,
                household.anchor,
            );
            write_tick_words(
                &mut input.output,
                target + ResultLayout::REQUIRED_NET,
                household.required_net,
            );
            write_tick_words(&mut input.output, target + ResultLayout::SECOND, second);
            input.output[target + ResultLayout::WEIGHT] = weight;
            input.output[target + ResultLayout::AVAILABLE] = u32::cast_from(available);
        }
    }
}

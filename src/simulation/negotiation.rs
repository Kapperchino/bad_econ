use super::{
    bargaining::{
        KernelBargainingFactors, KernelWageBounds, bargaining_weight, second_offer, wage_bounds,
    },
    configuration::Feature,
    layout::LOW_WORD,
    policy::MAXIMUM_WAGE,
    words::*,
};
use crate::domain::{PAYROLL_SCALE, bargaining::BargainingPolicy};
use crate::simulation::configuration::EconomyConfig;
use cubecl::prelude::*;

#[derive(CubeType, CubeLaunch, Clone, Copy, Default)]
pub struct BargainingConfig {
    #[cube(comptime)]
    pub enabled: Feature,
    pub legal_minimum: u32,
    pub quantum: u32,
    pub daily_cash_cost: u32,
    pub funded_other: u32,
    pub participating_adults: u32,
    pub normal_day_minutes: u32,
    pub work_cost_per_minute: u32,
    pub desired_wage: u32,
    pub mandate: u32,
    pub withdrawal: u32,
    pub alternatives: u32,
    pub concentration: u32,
    pub replacements: u32,
}
impl BargainingConfig {
    pub fn new(policy: Option<&BargainingPolicy>) -> Self {
        match policy {
            None => Self::default(),
            Some(policy) => Self {
                enabled: Feature::Enabled,
                legal_minimum: policy.legal_minimum,
                quantum: policy.quantum,
                daily_cash_cost: policy.daily_cash_cost,
                funded_other: policy.funded_other,
                participating_adults: policy.participating_adults,
                normal_day_minutes: policy.normal_day_minutes,
                work_cost_per_minute: policy.work_cost_per_minute,
                desired_wage: policy.desired_wage,
                mandate: policy.mandate,
                withdrawal: policy.withdrawal,
                alternatives: policy.alternatives,
                concentration: policy.concentration,
                replacements: policy.replacements,
            },
        }
    }
    pub fn launch(self) -> BargainingConfigLaunch {
        BargainingConfigLaunch {
            enabled: self.enabled,
            legal_minimum: self.legal_minimum,
            quantum: self.quantum,
            daily_cash_cost: self.daily_cash_cost,
            funded_other: self.funded_other,
            participating_adults: self.participating_adults,
            normal_day_minutes: self.normal_day_minutes,
            work_cost_per_minute: self.work_cost_per_minute,
            desired_wage: self.desired_wage,
            mandate: self.mandate,
            withdrawal: self.withdrawal,
            alternatives: self.alternatives,
            concentration: self.concentration,
            replacements: self.replacements,
        }
    }
}

#[derive(CubeType, Clone, Copy)]
pub struct NegotiationRequest {
    pub prior_gross: Ticks,
    pub household_cash: Ticks,
    pub funded_per_packet: Ticks,
    pub candidate_wage: u32,
    pub payable_minutes: u32,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NegotiationStatus {
    Unavailable,
    Agreed,
}

#[derive(CubeType, Clone, Copy)]
pub struct NegotiationQuote {
    pub wage: Ticks,
    pub minimum: Ticks,
    pub maximum: Ticks,
    pub required_net_rate: Ticks,
    pub gross_needed: Ticks,
    pub status: u32,
}

#[derive(CubeType, Clone, Copy)]
struct NetAmount {
    whole: Ticks,
    remainder: u32,
}

#[cube]
impl NetAmount {
    fn ratio(numerator: Ticks, denominator: u32) -> Self {
        let whole = divide(numerator, denominator);
        NetAmount {
            whole,
            remainder: minus(numerator, times(whole, denominator).value).extract(LOW_WORD),
        }
    }

    fn accrued(gross: Ticks, slope: u32, denominator: u32, control: &[Atomic<u32>]) -> Self {
        let units = divide(gross, PAYROLL_SCALE);
        let remainder = minus(gross, times(units, PAYROLL_SCALE).value).extract(LOW_WORD);
        let numerator = remainder * slope;
        let product = times(units, slope);
        let whole = plus(product.value, ticks(numerator / PAYROLL_SCALE, 0));
        product.faults.union(whole.faults).record(control);
        NetAmount {
            whole: whole.value,
            remainder: (numerator % PAYROLL_SCALE) * (denominator / PAYROLL_SCALE),
        }
    }

    fn covered_by(&self, other: &NetAmount) -> bool {
        lt(self.whole, other.whole)
            || (equal(self.whole, other.whole) && self.remainder <= other.remainder)
    }

    fn deduct(&mut self, other: &NetAmount, denominator: u32) {
        let borrow = u32::cast_from(self.remainder < other.remainder);
        self.whole = minus(minus(self.whole, other.whole), ticks(borrow, 0));
        self.remainder = select(
            self.remainder < other.remainder,
            denominator - (other.remainder - self.remainder),
            self.remainder - other.remainder,
        );
    }

    fn gross(&self, slope: u32, denominator: u32, control: &[Atomic<u32>]) -> Ticks {
        let coefficient = slope * (denominator / PAYROLL_SCALE);
        let units = divide(self.whole, coefficient);
        let remainder = minus(self.whole, times(units, coefficient).value).extract(LOW_WORD);
        let numerator = plus(times32(remainder, denominator), ticks(self.remainder, 0));
        let fraction = divide(numerator.value, coefficient);
        let residual = minus(numerator.value, times(fraction, coefficient).value);
        let product = times(units, denominator);
        let sum = plus(product.value, fraction);
        let result = plus(
            sum.value,
            ticks(u32::cast_from(!equal(residual, zero())), 0),
        );
        numerator
            .faults
            .union(product.faults)
            .union(sum.faults)
            .union(result.faults)
            .record(control);
        result.value
    }
}

#[repr(u32)]
#[derive(Clone, Copy)]
enum IncomeSearch {
    Pending,
    Resolved,
}

#[cube]
fn required_gross(
    prior: Ticks,
    mut need: NetAmount,
    denominator: u32,
    config: &EconomyConfig,
    control: &[Atomic<u32>],
) -> Ticks {
    let mut result = zero();
    let mut lower = zero();
    let mut rate = config.tax_base_rate;
    let charge = config.employee_charge;
    let mut status = u32::new(comptime!(IncomeSearch::Pending as i64));
    #[unroll]
    for bracket in 0..config.tax_brackets.len() {
        let bracket = config.tax_brackets.index(bracket);
        let upper = ticks(bracket.low, bracket.high);
        let start = maximum(prior, lower);
        if status == comptime!(IncomeSearch::Pending as u32) && lt(start, upper) {
            let span = minus(upper, start);
            let slope = PAYROLL_SCALE - rate - charge;
            let available = NetAmount::accrued(span, slope, denominator, control);
            let mut used = span;
            if need.covered_by(&available) {
                used = need.gross(slope, denominator, control);
                status = comptime!(IncomeSearch::Resolved as u32);
            }
            if status == comptime!(IncomeSearch::Pending as u32) {
                need.deduct(&available, denominator);
            }
            let sum = plus(result, used);
            sum.faults.record(control);
            result = sum.value;
        }
        lower = upper;
        rate = bracket.rate;
    }
    if status == comptime!(IncomeSearch::Pending as u32) {
        let remaining = need.gross(PAYROLL_SCALE - rate - charge, denominator, control);
        let sum = plus(result, remaining);
        sum.faults.record(control);
        result = sum.value;
    }
    plus(prior, result).faults.record(control);
    result
}

#[cube]
pub fn legal_grid_rate(config: &EconomyConfig) -> u32 {
    let quantum = config.bargaining.quantum;
    ceildiv(config.bargaining.legal_minimum, quantum) * quantum
}

#[cube]
pub fn negotiate(
    request: NegotiationRequest,
    config: &EconomyConfig,
    control: &[Atomic<u32>],
) -> NegotiationQuote {
    let cash_cost = config.bargaining.daily_cash_cost;
    let mut cover = 10_500u32;
    if cash_cost > 0 {
        let bounded_cash = minimum(request.household_cash, times32(cash_cost, 11));
        let scaled = ratio(bounded_cash, 1000, cash_cost);
        scaled.faults.record(control);
        cover = min(scaled.value.extract(LOW_WORD), 10_500);
    }
    let base = cash_cost - min(cash_cost, config.bargaining.funded_other);
    let refusal = select(cash_cost == 0, 42_000, 21_000 + 2 * cover);
    let denominator =
        35_000 * config.bargaining.participating_adults * config.bargaining.normal_day_minutes;
    let rate_numerator = times32(base, refusal);
    let rate = NetAmount::ratio(rate_numerator, denominator);
    let displayed = plus(
        rate.whole,
        ticks(
            u32::cast_from(rate.remainder > 0) + config.bargaining.work_cost_per_minute,
            0,
        ),
    );
    let numerator = times(rate_numerator, request.payable_minutes);
    let mut payment = NetAmount::ratio(numerator.value, denominator);
    let required = plus(
        payment.whole,
        times32(
            config.bargaining.work_cost_per_minute,
            request.payable_minutes,
        ),
    );
    displayed
        .faults
        .union(numerator.faults)
        .union(required.faults)
        .record(control);
    payment.whole = required.value;
    let gross = required_gross(request.prior_gross, payment, denominator, config, control);
    let quote = wage_bounds(
        KernelWageBounds {
            legal: ticks(config.bargaining.legal_minimum, 0),
            gross_needed: gross,
            funded: request.funded_per_packet,
            payable_hours: request.payable_minutes,
            employer_charge: 0,
            quantum: config.bargaining.quantum,
        },
        control,
    );
    let quantum = config.bargaining.quantum;
    let ceiling = minimum(quote.maximum, ticks(MAXIMUM_WAGE / quantum * quantum, 0));
    let weight = bargaining_weight(KernelBargainingFactors {
        mandate: config.bargaining.mandate,
        withdrawal: config.bargaining.withdrawal,
        alternatives: config.bargaining.alternatives,
        concentration: config.bargaining.concentration,
        replacements: config.bargaining.replacements,
    });
    let desired = maximum(
        quote.minimum,
        ticks(
            select(
                config.bargaining.desired_wage == 0,
                request.candidate_wage,
                config.bargaining.desired_wage,
            ),
            0,
        ),
    );
    let mut wage = zero();
    let mut status = u32::new(comptime!(NegotiationStatus::Unavailable as i64));
    if !lt(ceiling, quote.minimum) {
        wage = second_offer(
            ticks(request.candidate_wage, 0),
            desired,
            weight,
            quote.minimum,
            ceiling,
            quantum,
            control,
        );
        status = comptime!(NegotiationStatus::Agreed as u32);
    }
    NegotiationQuote {
        wage,
        minimum: quote.minimum,
        maximum: ceiling,
        required_net_rate: displayed.value,
        gross_needed: gross,
        status,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::configuration::{Configuration, TaxBracket};
    use crate::simulation::layout::HIGH_WORD;
    use crate::{
        ledger::TickLayout,
        simulation::{
            Device,
            flags::{CONTROL_WORDS, FAULT_WORD},
        },
    };
    use anyhow::Result;
    use bytemuck::{Pod, Zeroable};
    use std::mem::offset_of;

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Pod, Zeroable)]
    struct IncomeCase {
        prior: TickLayout,
        whole: TickLayout,
        remainder: u32,
        denominator: u32,
    }
    impl IncomeCase {
        const WORDS: usize = size_of::<Self>() / size_of::<u32>();
        const PRIOR: usize = offset_of!(Self, prior) / size_of::<u32>();
        const WHOLE: usize = offset_of!(Self, whole) / size_of::<u32>();
        const REMAINDER: usize = offset_of!(Self, remainder) / size_of::<u32>();
        const DENOMINATOR: usize = offset_of!(Self, denominator) / size_of::<u32>();
    }

    #[cube(launch)]
    fn income_probe(
        input: &[u32],
        output: &mut [u32],
        config: &EconomyConfig,
        control: &[Atomic<u32>],
    ) {
        let id = ABSOLUTE_POS;
        if id < input.len() / IncomeCase::WORDS {
            let base = id * IncomeCase::WORDS;
            let prior = ticks(
                input[base + IncomeCase::PRIOR],
                input[base + IncomeCase::PRIOR + 1],
            );
            let need = NetAmount {
                whole: ticks(
                    input[base + IncomeCase::WHOLE],
                    input[base + IncomeCase::WHOLE + 1],
                ),
                remainder: input[base + IncomeCase::REMAINDER],
            };
            let gross = required_gross(
                prior,
                need,
                input[base + IncomeCase::DENOMINATOR],
                config,
                control,
            );
            output[id * 2] = gross.extract(LOW_WORD);
            output[id * 2 + 1] = gross.extract(HIGH_WORD);
        }
    }

    struct TaxFixture {
        thresholds: [u64; 4],
        rates: [u32; 5],
        employee_charge: u32,
    }
    impl TaxFixture {
        fn numerator(&self, gross: u64) -> u128 {
            let base = u128::from(gross) * u128::from(1000 - self.rates[0] - self.employee_charge);
            self.thresholds.iter().zip(self.rates.windows(2)).fold(
                base as i128,
                |net, (threshold, rates)| {
                    net - i128::from(gross.saturating_sub(*threshold))
                        * (i128::from(rates[1]) - i128::from(rates[0]))
                },
            ) as u128
        }

        fn configuration(&self, device: &Device) -> Result<Configuration> {
            let scenario =
                crate::domain::Scenario::baseline(1, 42, crate::domain::Experiment::Baseline)?;
            let mut config = Configuration::new(device, &scenario, &scenario.initial()?);
            config.tax_base_rate = self.rates[0];
            config.employee_charge = self.employee_charge;
            config.tax_brackets = self
                .thresholds
                .iter()
                .zip(&self.rates[1..])
                .map(|(&threshold, &rate)| TaxBracket {
                    low: threshold as u32,
                    high: (threshold >> 32) as u32,
                    rate,
                })
                .collect();
            Ok(config)
        }

        fn expected(&self, case: &IncomeCase) -> u64 {
            let prior = u64::from(case.prior);
            let denominator = u128::from(case.denominator);
            let required = (u128::from(u64::from(case.whole)) * denominator
                + u128::from(case.remainder))
                * 1000;
            (0..=1000)
                .find(|gross| {
                    (self.numerator(prior + gross) - self.numerator(prior)) * denominator
                        >= required
                })
                .expect("fixture target has an affordable result")
        }
    }

    fn check(device: Device) -> Result<()> {
        let fixture = TaxFixture {
            thresholds: [100, 500, u64::from(u32::MAX) + 100, u64::MAX - 1000],
            rates: [100, 300, 200, 900, 100],
            employee_charge: 50,
        };
        let priors = (0..=110)
            .chain(490..=510)
            .chain([
                u64::from(u32::MAX) + 90,
                u64::from(u32::MAX) + 100,
                u64::MAX - 1010,
                u64::MAX - 1000,
            ])
            .collect::<Vec<_>>();
        let mut cases = Vec::new();
        for prior in priors {
            for whole in 0..=25u64 {
                for denominator in [1000, 35_000, 4_294_955_000u32] {
                    for remainder in [0, denominator / 3, denominator - 1] {
                        cases.push(IncomeCase {
                            prior: prior.into(),
                            whole: whole.into(),
                            remainder,
                            denominator,
                        });
                    }
                }
            }
        }
        let input = device.tensor(bytemuck::cast_slice(&cases));
        let output = device.empty(cases.len() * 2);
        let config = fixture.configuration(&device)?;
        let control = device.tensor(&[0; CONTROL_WORDS]);
        let argument = |tensor: &burn_wgpu::CubeTensor| tensor.clone().into_buffer_arg();
        income_probe::launch(
            &device.client,
            device.count(cases.len()),
            CubeDim::new_1d(device.workgroup),
            argument(&input),
            argument(&output),
            config.launch(),
            argument(&control),
        );
        let actual = device.read(&output)?;
        assert_eq!(device.read(&control)?[FAULT_WORD], 0);
        for (case, words) in cases.iter().zip(actual.chunks_exact(2)) {
            let value = u64::from(words[0]) | (u64::from(words[1]) << 32);
            assert_eq!(value, fixture.expected(case), "{case:?}");
        }
        Ok(())
    }

    #[test]
    fn exact_bracket_and_fraction_inversion_on_gpu() -> Result<()> {
        check(Device::headless(64)?)
    }
}

use super::{
    flags::{FaultFlags, KernelFaults},
    layout::*,
    words::*,
};
use crate::domain::PAYROLL_SCALE;
use crate::simulation::configuration::EconomyConfig;
use cubecl::prelude::*;

#[derive(CubeType, Clone, Copy)]
struct ScaledAmount {
    whole: Ticks,
    remainder: u32,
}

#[cube]
impl ScaledAmount {
    fn new(amount: Ticks, rate: u32, control: &[Atomic<u32>]) -> Self {
        let whole = ratio(amount, rate, PAYROLL_SCALE);
        whole.faults.record(control);
        let units = divide(amount, PAYROLL_SCALE);
        let remaining = minus(amount, times(units, PAYROLL_SCALE).value);
        ScaledAmount {
            whole: whole.value,
            remainder: remaining.extract(LOW_WORD) * rate % PAYROLL_SCALE,
        }
    }

    fn add(&mut self, other: ScaledAmount, control: &[Atomic<u32>]) {
        let fraction = self.remainder + other.remainder;
        let sum = plus(self.whole, other.whole);
        let carried = plus(sum.value, ticks(fraction / PAYROLL_SCALE, 0));
        sum.faults.union(carried.faults).record(control);
        self.whole = carried.value;
        self.remainder = fraction % PAYROLL_SCALE;
    }
}

#[derive(CubeType, Clone, Copy)]
pub struct PayrollAccrual {
    pub net: Ticks,
    pub tax: Ticks,
    pub charge: Ticks,
    pub reserve: Ticks,
}

#[cube]
impl PayrollAccrual {
    pub fn new(gross: Ticks, config: &EconomyConfig, control: &[Atomic<u32>]) -> Self {
        let mut tax = ScaledAmount {
            whole: zero(),
            remainder: 0,
        };
        let mut lower = zero();
        let mut rate = config.tax_base_rate;
        #[unroll]
        for bracket in 0..config.tax_brackets.len() {
            let bracket = config.tax_brackets.index(bracket);
            let upper = ticks(bracket.low, bracket.high);
            let taxable = positive(minimum(gross, upper), lower);
            tax.add(ScaledAmount::new(taxable, rate, control), control);
            lower = upper;
            rate = bracket.rate;
        }
        tax.add(
            ScaledAmount::new(positive(gross, lower), rate, control),
            control,
        );
        let charge = ScaledAmount::new(gross, config.employee_charge, control);
        let fractional = tax.remainder + charge.remainder;
        let reserve = ticks(ceildiv(fractional, PAYROLL_SCALE), 0);
        let retained = plus(tax.whole, charge.whole);
        let withheld = plus(retained.value, reserve);
        retained.faults.union(withheld.faults).record(control);
        if lt(gross, withheld.value) {
            KernelFaults::new(comptime!(FaultFlags::PAYROLL_MISMATCH)).record(control);
        }
        PayrollAccrual {
            net: minus(gross, withheld.value),
            tax: tax.whole,
            charge: charge.whole,
            reserve,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::configuration::{Configuration, TaxBracket};
    use crate::simulation::{
        flags::{CONTROL_WORDS, FAULT_WORD},
        policy::{KernelStarts, proposed_starts},
    };
    use anyhow::Result;

    #[cube(launch)]
    fn teaching_cases(
        gross: &[u32],
        output: &mut [u32],
        config: &EconomyConfig,
        control: &[Atomic<u32>],
    ) {
        let id = ABSOLUTE_POS;
        if id < gross.len() / TICK_WORDS {
            let income = ticks(gross[id * TICK_WORDS], gross[id * TICK_WORDS + HIGH_WORD]);
            store(
                output,
                id as u32,
                PayrollAccrual::new(income, config, control).net,
            );
        }
        if id == 0 {
            let starts = proposed_starts(
                KernelStarts {
                    coverage: 4,
                    demand: ticks(10, 0),
                    finished: ticks(20, 0),
                    pipeline: ticks(5, 0),
                },
                control,
            );
            store(output, 3, starts);
        }
    }

    #[test]
    fn lean_payroll_and_proposed_starts_teaching_values_on_gpu() -> Result<()> {
        let device = crate::simulation::tests::device(64)?;
        let scenario =
            crate::domain::Scenario::baseline(1, 42, crate::domain::Experiment::Baseline)?;
        let mut config = Configuration::new(&device, &scenario, &scenario.initial()?);
        config.tax_base_rate = 100;
        config.employee_charge = 50;
        config.tax_brackets = vec![TaxBracket {
            low: 10_000,
            high: 0,
            rate: 300,
        }];
        let gross = device.tensor(&[8000, 0, 10_000, 0, 11_080, 0]);
        let output = device.empty(4 * TICK_WORDS);
        let control = device.tensor(&[0; CONTROL_WORDS]);
        let argument = |tensor: &burn_wgpu::CubeTensor| tensor.clone().into_buffer_arg();
        teaching_cases::launch(
            &device.client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(device.workgroup),
            argument(&gross),
            argument(&output),
            config.launch(),
            argument(&control),
        );
        let actual = device.read(&output)?;
        assert_eq!(actual, [6800, 0, 8500, 0, 9202, 0, 15, 0]);
        assert_eq!(actual[4] - actual[0], 2402);
        assert_eq!(device.read(&control)?[FAULT_WORD], 0);
        Ok(())
    }
}

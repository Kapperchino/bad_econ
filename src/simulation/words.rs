use super::{
    flags::{FaultFlags, KernelFaults},
    layout::*,
    policy::{PRICE_INDEX_BITS, PRICE_INDEX_LIMIT},
};
use cubecl::prelude::*;

pub type Ticks = Vector<u32, Const<TICK_WORDS>>;
#[derive(CubeType, Clone, Copy)]
pub struct CheckedTicks {
    pub value: Ticks,
    pub faults: KernelFaults,
}

#[cube]
pub fn ticks(low: u32, high: u32) -> Ticks {
    let mut result = Ticks::new(low);
    result.insert(HIGH_WORD, high);
    result
}
#[cube]
pub fn zero() -> Ticks {
    ticks(0, 0)
}
#[cube]
pub fn equal(a: Ticks, b: Ticks) -> bool {
    a.extract(LOW_WORD) == b.extract(LOW_WORD) && a.extract(HIGH_WORD) == b.extract(HIGH_WORD)
}
#[cube]
pub fn lt(a: Ticks, b: Ticks) -> bool {
    a.extract(HIGH_WORD) < b.extract(HIGH_WORD)
        || (a.extract(HIGH_WORD) == b.extract(HIGH_WORD)
            && a.extract(LOW_WORD) < b.extract(LOW_WORD))
}
#[cube]
pub fn plus(a: Ticks, b: Ticks) -> CheckedTicks {
    let low = a.extract(LOW_WORD) + b.extract(LOW_WORD);
    let carry = u32::cast_from(low < a.extract(LOW_WORD));
    let high = a.extract(HIGH_WORD) + b.extract(HIGH_WORD);
    let result = high + carry;
    let mut faults = KernelFaults::none();
    if high < a.extract(HIGH_WORD) || result < high {
        faults.insert(KernelFaults::new(comptime!(
            FaultFlags::ARITHMETIC_OVERFLOW
        )));
    }
    CheckedTicks {
        value: ticks(low, result),
        faults,
    }
}
#[cube]
pub fn minus(a: Ticks, b: Ticks) -> Ticks {
    ticks(
        a.extract(LOW_WORD) - b.extract(LOW_WORD),
        a.extract(HIGH_WORD)
            - b.extract(HIGH_WORD)
            - u32::cast_from(a.extract(LOW_WORD) < b.extract(LOW_WORD)),
    )
}
#[cube]
pub fn minimum(a: Ticks, b: Ticks) -> Ticks {
    ticks(
        select(lt(a, b), a.extract(LOW_WORD), b.extract(LOW_WORD)),
        select(lt(a, b), a.extract(HIGH_WORD), b.extract(HIGH_WORD)),
    )
}
#[cube]
pub fn maximum(a: Ticks, b: Ticks) -> Ticks {
    ticks(
        select(lt(a, b), b.extract(LOW_WORD), a.extract(LOW_WORD)),
        select(lt(a, b), b.extract(HIGH_WORD), a.extract(HIGH_WORD)),
    )
}
#[cube]
pub fn positive(a: Ticks, b: Ticks) -> Ticks {
    {
        let result = minus(a, b);
        ticks(
            select(lt(a, b), 0, result.extract(LOW_WORD)),
            select(lt(a, b), 0, result.extract(HIGH_WORD)),
        )
    }
}
#[cube]
pub fn times32(a: u32, b: u32) -> Ticks {
    let al = a & HALF_WORD_MASK;
    let ah = a >> HALF_WORD_BITS;
    let bl = b & HALF_WORD_MASK;
    let bh = b >> HALF_WORD_BITS;
    let low = al * bl;
    let middle1 = ah * bl + (low >> HALF_WORD_BITS);
    let middle2 = al * bh + (middle1 & HALF_WORD_MASK);
    ticks(
        (middle2 << HALF_WORD_BITS) | (low & HALF_WORD_MASK),
        ah * bh + (middle1 >> HALF_WORD_BITS) + (middle2 >> HALF_WORD_BITS),
    )
}
#[cube]
pub fn times(a: Ticks, b: u32) -> CheckedTicks {
    let low = times32(a.extract(LOW_WORD), b);
    let high = times32(a.extract(HIGH_WORD), b);
    let sum = low.extract(HIGH_WORD) + high.extract(LOW_WORD);
    let mut faults = KernelFaults::none();
    if high.extract(HIGH_WORD) > 0 || sum < low.extract(HIGH_WORD) {
        faults.insert(KernelFaults::new(comptime!(
            FaultFlags::ARITHMETIC_OVERFLOW
        )));
    }
    CheckedTicks {
        value: ticks(low.extract(LOW_WORD), sum),
        faults,
    }
}
#[cube]
pub fn divide(a: Ticks, b: u32) -> Ticks {
    match b {
        1 => a,
        _ => divide_nonunit(a, b),
    }
}

#[cube]
fn divide_nonunit(a: Ticks, b: u32) -> Ticks {
    let mut quotient = zero();
    if a.extract(HIGH_WORD) == 0 && b != 0 {
        quotient = ticks(a.extract(LOW_WORD) / b, 0);
    } else {
        let mut remainder = zero();
        for step in 0..TICK_BITS {
            let bit = (TICK_BITS - 1) - step;
            let digit = (select(bit >= WORD_BITS, a.extract(HIGH_WORD), a.extract(LOW_WORD))
                >> (bit & WORD_TOP_BIT))
                & 1;
            remainder = ticks(
                (remainder.extract(LOW_WORD) << 1) | digit,
                (remainder.extract(HIGH_WORD) << 1) | (remainder.extract(LOW_WORD) >> WORD_TOP_BIT),
            );
            if !lt(remainder, ticks(b, 0)) {
                remainder = minus(remainder, ticks(b, 0));
                quotient.insert(
                    HIGH_WORD,
                    quotient.extract(HIGH_WORD)
                        | select(bit >= WORD_BITS, 1u32 << (bit & WORD_TOP_BIT), 0),
                );
                quotient.insert(
                    LOW_WORD,
                    quotient.extract(LOW_WORD)
                        | select(bit < WORD_BITS, 1u32 << (bit & WORD_TOP_BIT), 0),
                );
            }
        }
    }
    quotient
}
#[cube]
pub fn ratio(a: Ticks, numerator: u32, denominator: u32) -> CheckedTicks {
    let quotient = divide(a, denominator);
    let remainder = minus(a, times(quotient, denominator).value);
    let whole = times(quotient, numerator);
    let fraction = divide(times32(remainder.extract(LOW_WORD), numerator), denominator);
    let result = plus(whole.value, fraction);
    CheckedTicks {
        value: result.value,
        faults: whole.faults.union(result.faults),
    }
}
#[cube]
pub fn fraction_index(numerator: Ticks, denominator: Ticks) -> u32 {
    let bounded = minimum(numerator, denominator);
    let mut remainder = bounded;
    let mut result = 0u32;
    #[unroll]
    for _step in 0..PRICE_INDEX_BITS {
        let carry = remainder.extract(HIGH_WORD) >> WORD_TOP_BIT;
        remainder = ticks(
            remainder.extract(LOW_WORD) << 1,
            (remainder.extract(HIGH_WORD) << 1) | (remainder.extract(LOW_WORD) >> WORD_TOP_BIT),
        );
        let digit = carry != 0 || !lt(remainder, denominator);
        let reduced = minus(remainder, denominator);
        remainder = select(digit, reduced, remainder);
        result = result * 2 + u32::cast_from(digit);
    }
    select(equal(bounded, denominator), PRICE_INDEX_LIMIT, result)
}
#[allow(
    clippy::manual_is_multiple_of,
    reason = "CubeCL supports integer remainder"
)]
#[cube]
pub fn ceildiv(a: u32, b: u32) -> u32 {
    a / b + u32::cast_from(a % b != 0)
}
#[cube]
pub fn load(values: &mut [u32], coordinate: u32) -> Ticks {
    ticks(
        values[(coordinate * TICK_WORDS as u32) as usize],
        values[(coordinate * TICK_WORDS as u32 + HIGH_WORD as u32) as usize],
    )
}
#[cube]
pub fn store(values: &mut [u32], coordinate: u32, value: Ticks) {
    values[(coordinate * TICK_WORDS as u32) as usize] = value.extract(LOW_WORD);
    values[(coordinate * TICK_WORDS as u32 + HIGH_WORD as u32) as usize] = value.extract(HIGH_WORD);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        arithmetic::DrawKey,
        ledger::TickLayout,
        simulation::{Device, ops::TEST_WORKGROUP},
    };
    use anyhow::Result;
    use bytemuck::{Pod, Zeroable};
    use std::mem::offset_of;

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Pod, Zeroable)]
    struct ProbeInput {
        left: TickLayout,
        numerator: u32,
        denominator: u32,
        right: TickLayout,
    }
    impl ProbeInput {
        const LEFT: usize = offset_of!(Self, left) / WORD_BYTES;
        const NUMERATOR: usize = offset_of!(Self, numerator) / WORD_BYTES;
        const DENOMINATOR: usize = offset_of!(Self, denominator) / WORD_BYTES;
        const RIGHT: usize = offset_of!(Self, right) / WORD_BYTES;
        const WORDS: usize = size_of::<Self>() / WORD_BYTES;
    }
    #[repr(C)]
    #[derive(Clone, Copy, Pod, Zeroable)]
    struct ProbeOutput {
        sum: TickLayout,
        sum_faults: u32,
        product: TickLayout,
        product_faults: u32,
        quotient: TickLayout,
        ratio: TickLayout,
        ratio_faults: u32,
        fraction: u32,
        difference: TickLayout,
    }
    impl ProbeOutput {
        const SUM: usize = offset_of!(Self, sum) / WORD_BYTES;
        const SUM_FAULTS: usize = offset_of!(Self, sum_faults) / WORD_BYTES;
        const PRODUCT: usize = offset_of!(Self, product) / WORD_BYTES;
        const PRODUCT_FAULTS: usize = offset_of!(Self, product_faults) / WORD_BYTES;
        const QUOTIENT: usize = offset_of!(Self, quotient) / WORD_BYTES;
        const RATIO: usize = offset_of!(Self, ratio) / WORD_BYTES;
        const RATIO_FAULTS: usize = offset_of!(Self, ratio_faults) / WORD_BYTES;
        const FRACTION: usize = offset_of!(Self, fraction) / WORD_BYTES;
        const DIFFERENCE: usize = offset_of!(Self, difference) / WORD_BYTES;
        const WORDS: usize = size_of::<Self>() / WORD_BYTES;
    }

    #[cube(launch)]
    fn probe(input: &[u32], output: &mut [u32]) {
        let id = ABSOLUTE_POS;
        if id < input.len() / ProbeInput::WORDS {
            let base = id * ProbeInput::WORDS;
            let a = ticks(
                input[base + ProbeInput::LEFT + LOW_WORD],
                input[base + ProbeInput::LEFT + HIGH_WORD],
            );
            let numerator = input[base + ProbeInput::NUMERATOR];
            let denominator = input[base + ProbeInput::DENOMINATOR];
            let b = ticks(
                input[base + ProbeInput::RIGHT + LOW_WORD],
                input[base + ProbeInput::RIGHT + HIGH_WORD],
            );
            let sum = plus(a, b);
            let product = times(a, numerator);
            let quotient = divide(a, denominator);
            let scaled = ratio(a, numerator, denominator);
            let difference = minus(a, b);
            let target = id * ProbeOutput::WORDS;
            output[target + ProbeOutput::SUM + LOW_WORD] = sum.value.extract(LOW_WORD);
            output[target + ProbeOutput::SUM + HIGH_WORD] = sum.value.extract(HIGH_WORD);
            output[target + ProbeOutput::SUM_FAULTS] = sum.faults.bits();
            output[target + ProbeOutput::PRODUCT + LOW_WORD] = product.value.extract(LOW_WORD);
            output[target + ProbeOutput::PRODUCT + HIGH_WORD] = product.value.extract(HIGH_WORD);
            output[target + ProbeOutput::PRODUCT_FAULTS] = product.faults.bits();
            output[target + ProbeOutput::QUOTIENT + LOW_WORD] = quotient.extract(LOW_WORD);
            output[target + ProbeOutput::QUOTIENT + HIGH_WORD] = quotient.extract(HIGH_WORD);
            output[target + ProbeOutput::RATIO + LOW_WORD] = scaled.value.extract(LOW_WORD);
            output[target + ProbeOutput::RATIO + HIGH_WORD] = scaled.value.extract(HIGH_WORD);
            output[target + ProbeOutput::RATIO_FAULTS] = scaled.faults.bits();
            output[target + ProbeOutput::FRACTION] = fraction_index(a, b);
            output[target + ProbeOutput::DIFFERENCE + LOW_WORD] = difference.extract(LOW_WORD);
            output[target + ProbeOutput::DIFFERENCE + HIGH_WORD] = difference.extract(HIGH_WORD);
        }
    }

    fn check(device: Device) -> Result<()> {
        let edges = [
            0,
            1,
            u32::MAX as u64,
            1u64 << WORD_BITS,
            (1u64 << (TICK_BITS - 1)) - 1,
            1u64 << (TICK_BITS - 1),
            u64::MAX - 1,
            u64::MAX,
        ];
        let cases = (0..512u64)
            .map(|index| {
                let draw = DrawKey {
                    seed: 81,
                    day: 0,
                    phase: 0,
                    actor: index,
                    purpose: 0,
                    index: 0,
                }
                .draw();
                let a = match index {
                    _ if (index as usize) < edges.len().pow(2) => {
                        edges[index as usize % edges.len()]
                    }
                    _ if index % 2 == 0 => u64::from(draw as u32),
                    _ => draw,
                };
                let b = match index {
                    _ if (index as usize) < edges.len().pow(2) => {
                        edges[index as usize / edges.len()].max(1)
                    }
                    _ => draw.rotate_left(19).max(1),
                };
                let n = match index % 4 {
                    0 => 1000,
                    1 => 1,
                    2 => u32::MAX,
                    _ => draw as u32,
                };
                let d = match index % 4 {
                    0 => 1000,
                    1 => u32::MAX,
                    2 => 1,
                    _ => (draw >> WORD_BITS) as u32 | 1,
                };
                ProbeInput {
                    left: a.into(),
                    numerator: n,
                    denominator: d,
                    right: b.into(),
                }
            })
            .collect::<Vec<_>>();
        let input = device.tensor(bytemuck::cast_slice(&cases));
        let output = device.empty(cases.len() * ProbeOutput::WORDS);
        probe::launch(
            &device.client,
            device.count(cases.len()),
            CubeDim::new_1d(TEST_WORKGROUP),
            input.clone().into_buffer_arg(),
            output.clone().into_buffer_arg(),
        );
        let actual = device.read(&output)?;
        for (case, output) in cases
            .iter()
            .zip(bytemuck::cast_slice::<u32, ProbeOutput>(&actual))
        {
            let a = u64::from(case.left);
            let b = u64::from(case.right);
            let n = case.numerator;
            let d = case.denominator;
            let wide_sum = u128::from(a) + u128::from(b);
            let wide_product = u128::from(a) * u128::from(n);
            let wide_ratio = wide_product / u128::from(d);
            let faults = |wide| match wide > u128::from(u64::MAX) {
                true => FaultFlags::ARITHMETIC_OVERFLOW,
                false => FaultFlags::empty(),
            };
            assert_eq!(u64::from(output.sum), wide_sum as u64, "sum {case:?}");
            assert_eq!(FaultFlags::new(output.sum_faults)?, faults(wide_sum));
            assert_eq!(
                u64::from(output.product),
                wide_product as u64,
                "product {case:?}"
            );
            assert_eq!(
                FaultFlags::new(output.product_faults)?,
                faults(wide_product)
            );
            assert_eq!(
                u64::from(output.quotient),
                a / u64::from(d),
                "division {case:?}"
            );
            assert_eq!(u64::from(output.ratio), wide_ratio as u64, "ratio {case:?}");
            assert_eq!(FaultFlags::new(output.ratio_faults)?, faults(wide_ratio));
            assert_eq!(
                output.fraction,
                (u128::from(a.min(b)) * u128::from(PRICE_INDEX_LIMIT) / u128::from(b)) as u32,
                "fraction {case:?}"
            );
            assert_eq!(
                u64::from(output.difference),
                a.wrapping_sub(b),
                "subtraction {case:?}"
            );
        }
        Ok(())
    }

    #[test]
    fn portable_words_match_wide_oracle_on_gpu() -> Result<()> {
        check(crate::simulation::tests::device(TEST_WORKGROUP)?)
    }
}

use cubecl::prelude::*;

pub type Ticks = Vector<u32, Const<2>>;
#[derive(CubeType, Clone, Copy)]
pub struct CheckedTicks {
    pub value: Ticks,
    pub overflow: u32,
}

#[cube]
pub fn ticks(low: u32, high: u32) -> Ticks {
    let mut result = Ticks::new(low);
    result.insert(1usize, high);
    result
}
#[cube]
pub fn zero() -> Ticks {
    ticks(0, 0)
}
#[cube]
pub fn equal(a: Ticks, b: Ticks) -> bool {
    a.extract(0usize) == b.extract(0usize) && a.extract(1usize) == b.extract(1usize)
}
#[cube]
pub fn lt(a: Ticks, b: Ticks) -> bool {
    a.extract(1usize) < b.extract(1usize)
        || (a.extract(1usize) == b.extract(1usize) && a.extract(0usize) < b.extract(0usize))
}
#[cube]
pub fn plus(a: Ticks, b: Ticks) -> CheckedTicks {
    let low = a.extract(0usize) + b.extract(0usize);
    let carry = u32::cast_from(low < a.extract(0usize));
    let high = a.extract(1usize) + b.extract(1usize);
    let result = high + carry;
    CheckedTicks {
        value: ticks(low, result),
        overflow: u32::cast_from(high < a.extract(1usize) || result < high),
    }
}
#[cube]
pub fn minus(a: Ticks, b: Ticks) -> Ticks {
    ticks(
        a.extract(0usize) - b.extract(0usize),
        a.extract(1usize)
            - b.extract(1usize)
            - u32::cast_from(a.extract(0usize) < b.extract(0usize)),
    )
}
#[cube]
pub fn minimum(a: Ticks, b: Ticks) -> Ticks {
    ticks(
        select(lt(a, b), a.extract(0usize), b.extract(0usize)),
        select(lt(a, b), a.extract(1usize), b.extract(1usize)),
    )
}
#[cube]
pub fn maximum(a: Ticks, b: Ticks) -> Ticks {
    ticks(
        select(lt(a, b), b.extract(0usize), a.extract(0usize)),
        select(lt(a, b), b.extract(1usize), a.extract(1usize)),
    )
}
#[cube]
pub fn positive(a: Ticks, b: Ticks) -> Ticks {
    {
        let result = minus(a, b);
        ticks(
            select(lt(a, b), 0, result.extract(0usize)),
            select(lt(a, b), 0, result.extract(1usize)),
        )
    }
}
#[cube]
pub fn times32(a: u32, b: u32) -> Ticks {
    let al = a & 65535;
    let ah = a >> 16;
    let bl = b & 65535;
    let bh = b >> 16;
    let low = al * bl;
    let middle1 = ah * bl + (low >> 16);
    let middle2 = al * bh + (middle1 & 65535);
    ticks(
        (middle2 << 16) | (low & 65535),
        ah * bh + (middle1 >> 16) + (middle2 >> 16),
    )
}
#[cube]
pub fn times(a: Ticks, b: u32) -> CheckedTicks {
    let low = times32(a.extract(0usize), b);
    let high = times32(a.extract(1usize), b);
    let sum = low.extract(1usize) + high.extract(0usize);
    CheckedTicks {
        value: ticks(low.extract(0usize), sum),
        overflow: u32::cast_from(high.extract(1usize) > 0 || sum < low.extract(1usize)),
    }
}
#[cube]
pub fn divide(a: Ticks, b: u32) -> Ticks {
    let mut quotient = zero();
    let mut remainder = zero();
    for step in 0..64u32 {
        let bit = 63 - step;
        let digit = (select(bit >= 32, a.extract(1usize), a.extract(0usize)) >> (bit & 31)) & 1;
        remainder = ticks(
            (remainder.extract(0usize) << 1) | digit,
            (remainder.extract(1usize) << 1) | (remainder.extract(0usize) >> 31),
        );
        if !lt(remainder, ticks(b, 0)) {
            remainder = minus(remainder, ticks(b, 0));
            quotient.insert(
                1usize,
                quotient.extract(1usize) | select(bit >= 32, 1u32 << (bit & 31), 0),
            );
            quotient.insert(
                0usize,
                quotient.extract(0usize) | select(bit < 32, 1u32 << (bit & 31), 0),
            );
        }
    }
    quotient
}
#[cube]
pub fn ratio(a: Ticks, numerator: u32, denominator: u32) -> CheckedTicks {
    let quotient = divide(a, denominator);
    let remainder = minus(a, times(quotient, denominator).value);
    let whole = times(quotient, numerator);
    let fraction = divide(times32(remainder.extract(0usize), numerator), denominator);
    let result = plus(whole.value, fraction);
    CheckedTicks {
        value: result.value,
        overflow: whole.overflow | result.overflow,
    }
}
#[cube]
pub fn fraction_index(numerator: Ticks, denominator: Ticks) -> u32 {
    let bounded = minimum(numerator, denominator);
    let mut remainder = bounded;
    let mut result = 0u32;
    for _step in 0..5u32 {
        let carry = remainder.extract(1usize) >> 31;
        remainder = ticks(
            remainder.extract(0usize) << 1,
            (remainder.extract(1usize) << 1) | (remainder.extract(0usize) >> 31),
        );
        let digit = carry != 0 || !lt(remainder, denominator);
        let reduced = minus(remainder, denominator);
        remainder = select(digit, reduced, remainder);
        result = result * 2 + u32::cast_from(digit);
    }
    select(equal(bounded, denominator), 32, result)
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
        values[(coordinate * 2) as usize],
        values[(coordinate * 2 + 1) as usize],
    )
}
#[cube]
pub fn store(values: &mut [u32], coordinate: u32, value: Ticks) {
    values[(coordinate * 2) as usize] = value.extract(0usize);
    values[(coordinate * 2 + 1) as usize] = value.extract(1usize);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{arithmetic::DrawKey, simulation::Device};
    use anyhow::Result;

    #[cube(launch)]
    fn probe(input: &[u32], output: &mut [u32]) {
        let id = ABSOLUTE_POS;
        if id < input.len() / 6 {
            let base = id * 6;
            let a = ticks(input[base], input[base + 1]);
            let numerator = input[base + 2];
            let denominator = input[base + 3];
            let b = ticks(input[base + 4], input[base + 5]);
            let sum = plus(a, b);
            let product = times(a, numerator);
            let quotient = divide(a, denominator);
            let scaled = ratio(a, numerator, denominator);
            let difference = minus(a, b);
            let target = id * 14;
            output[target] = sum.value.extract(0usize);
            output[target + 1] = sum.value.extract(1usize);
            output[target + 2] = sum.overflow;
            output[target + 3] = product.value.extract(0usize);
            output[target + 4] = product.value.extract(1usize);
            output[target + 5] = product.overflow;
            output[target + 6] = quotient.extract(0usize);
            output[target + 7] = quotient.extract(1usize);
            output[target + 8] = scaled.value.extract(0usize);
            output[target + 9] = scaled.value.extract(1usize);
            output[target + 10] = scaled.overflow;
            output[target + 11] = fraction_index(a, b);
            output[target + 12] = difference.extract(0usize);
            output[target + 13] = difference.extract(1usize);
        }
    }

    fn check(device: Device) -> Result<()> {
        let edges = [
            0,
            1,
            u32::MAX as u64,
            1u64 << 32,
            (1u64 << 63) - 1,
            1u64 << 63,
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
                    0..64 => edges[index as usize % 8],
                    _ => draw,
                };
                let b = match index {
                    0..64 => edges[index as usize / 8].max(1),
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
                    _ => (draw >> 32) as u32 | 1,
                };
                (a, b, n, d)
            })
            .collect::<Vec<_>>();
        let words = cases
            .iter()
            .flat_map(|&(a, b, n, d)| {
                [a as u32, (a >> 32) as u32, n, d, b as u32, (b >> 32) as u32]
            })
            .collect::<Vec<_>>();
        let input = device.tensor(&words);
        let output = device.empty(cases.len() * 14);
        probe::launch(
            &device.client,
            CubeCount::Static(8, 1, 1),
            CubeDim::new_1d(64),
            unsafe { BufferArg::from_raw_parts(input.handle.clone(), words.len()) },
            unsafe { BufferArg::from_raw_parts(output.handle.clone(), cases.len() * 14) },
        );
        let actual = device.read(&output)?;
        for (case, words) in cases.into_iter().zip(actual.chunks_exact(14)) {
            let (a, b, n, d) = case;
            let wide_sum = u128::from(a) + u128::from(b);
            let wide_product = u128::from(a) * u128::from(n);
            let wide_ratio = wide_product / u128::from(d);
            let value = |index: usize| u64::from(words[index]) | u64::from(words[index + 1]) << 32;
            assert_eq!(value(0), wide_sum as u64, "sum {case:?}");
            assert_eq!(words[2] != 0, wide_sum > u128::from(u64::MAX));
            assert_eq!(value(3), wide_product as u64, "product {case:?}");
            assert_eq!(words[5] != 0, wide_product > u128::from(u64::MAX));
            assert_eq!(value(6), a / u64::from(d), "division {case:?}");
            assert_eq!(value(8), wide_ratio as u64, "ratio {case:?}");
            assert_eq!(words[10] != 0, wide_ratio > u128::from(u64::MAX));
            assert_eq!(
                words[11],
                (u128::from(a.min(b)) * 32 / u128::from(b)) as u32,
                "fraction {case:?}"
            );
            assert_eq!(value(12), a.wrapping_sub(b), "subtraction {case:?}");
        }
        Ok(())
    }

    #[test]
    fn portable_words_match_wide_oracle_on_cpu() -> Result<()> {
        check(Device::cpu(64)?)
    }

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn portable_words_match_wide_oracle_on_gpu() -> Result<()> {
        check(Device::headless(64)?)
    }
}

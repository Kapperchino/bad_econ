use crate::simulation::{
    flags::{FaultFlags, KernelFaults},
    layout::{HIGH_WORD, LOW_WORD, TICK_WORDS},
    words::*,
};
use cubecl::prelude::*;

pub const ENTRY_WORDS: usize = 3;
pub const AVAILABLE: u32 = 1;
pub const INADMISSIBLE: u32 = 2;
pub const UNRESOLVED: u32 = 3;
pub const INAPPLICABLE: u32 = 4;
const INTERVAL_WORDS: usize = 4;

#[derive(CubeType, Clone, Copy)]
pub(super) struct Bounds {
    pub(super) lower: Ticks,
    pub(super) upper: Ticks,
}

#[cube]
fn read(input: &[u32], offset: usize) -> Ticks {
    ticks(input[offset], input[offset + 1])
}

#[cube]
pub(super) fn interval(input: &[u32], index: usize) -> Bounds {
    let base = index * INTERVAL_WORDS;
    Bounds {
        lower: read(input, base),
        upper: read(input, base + TICK_WORDS),
    }
}

#[cube]
pub(super) fn write(output: &mut [u32], index: usize, bounds: Bounds) {
    let base = index * INTERVAL_WORDS;
    output[base] = bounds.lower.extract(LOW_WORD);
    output[base + 1] = bounds.lower.extract(HIGH_WORD);
    output[base + 2] = bounds.upper.extract(LOW_WORD);
    output[base + 3] = bounds.upper.extract(HIGH_WORD);
}

#[cube]
pub(super) fn sum(left: Ticks, right: Ticks, control: &[Atomic<u32>]) -> Ticks {
    let result = plus(left, right);
    result.faults.record(control);
    result.value
}

#[cube]
pub(super) fn scaled(
    value: Ticks,
    numerator: u32,
    denominator: u32,
    control: &[Atomic<u32>],
) -> Bounds {
    let lower = ratio(value, numerator, denominator);
    lower.faults.record(control);
    let remainder = minus(value, times(divide(value, denominator), denominator).value);
    let product = times32(remainder.extract(LOW_WORD), numerator);
    let remainder = minus(
        product,
        times(divide(product, denominator), denominator).value,
    );
    Bounds {
        lower: lower.value,
        upper: sum(
            lower.value,
            ticks(u32::cast_from(!equal(remainder, zero())), 0),
            control,
        ),
    }
}

#[cube]
fn row_product(
    offsets: &[u32],
    entries: &[u32],
    input: &[u32],
    row: usize,
    control: &[Atomic<u32>],
) -> Bounds {
    let mut lower = zero();
    let mut upper = zero();
    for entry in offsets[row]..offsets[row + 1] {
        let base = entry as usize * ENTRY_WORDS;
        let value = interval(input, entries[base] as usize);
        let numerator = entries[base + 1];
        let denominator = entries[base + 2];
        lower = sum(
            lower,
            scaled(value.lower, numerator, denominator, control).lower,
            control,
        );
        upper = sum(
            upper,
            scaled(value.upper, numerator, denominator, control).upper,
            control,
        );
    }
    Bounds { lower, upper }
}

#[cube(launch)]
pub fn matvec(
    offsets: &[u32],
    entries: &[u32],
    input: &[u32],
    output: &mut [u32],
    control: &[Atomic<u32>],
    rows: u32,
) {
    let row = ABSOLUTE_POS;
    if row < rows as usize {
        write(
            output,
            row,
            row_product(offsets, entries, input, row, control),
        );
    }
}

#[cube(launch)]
pub fn vector_add(
    left: &[u32],
    right: &[u32],
    output: &mut [u32],
    control: &[Atomic<u32>],
    length: u32,
) {
    let index = ABSOLUTE_POS;
    if index < length as usize {
        let a = interval(left, index);
        let b = interval(right, index);
        write(
            output,
            index,
            Bounds {
                lower: sum(a.lower, b.lower, control),
                upper: sum(a.upper, b.upper, control),
            },
        );
    }
}

#[cube]
fn product(left: Ticks, right: Ticks) -> Vector<u32, Const<4>> {
    let a = times32(left.extract(LOW_WORD), right.extract(LOW_WORD));
    let b = times32(left.extract(LOW_WORD), right.extract(HIGH_WORD));
    let c = times32(left.extract(HIGH_WORD), right.extract(LOW_WORD));
    let d = times32(left.extract(HIGH_WORD), right.extract(HIGH_WORD));
    let first = a.extract(HIGH_WORD) + b.extract(LOW_WORD);
    let second = first + c.extract(LOW_WORD);
    let carry = u32::cast_from(first < a.extract(HIGH_WORD)) + u32::cast_from(second < first);
    let third = b.extract(HIGH_WORD) + c.extract(HIGH_WORD);
    let fourth = third + d.extract(LOW_WORD);
    let fifth = fourth + carry;
    let top = d.extract(HIGH_WORD)
        + u32::cast_from(third < b.extract(HIGH_WORD))
        + u32::cast_from(fourth < third)
        + u32::cast_from(fifth < fourth);
    let mut result = Vector::<u32, Const<4>>::new(a.extract(LOW_WORD));
    result.insert(1usize, second);
    result.insert(2usize, fifth);
    result.insert(3usize, top);
    result
}

#[cube]
pub(super) fn divide_product(
    left: Ticks,
    right: Ticks,
    denominator: Ticks,
    control: &[Atomic<u32>],
) -> Bounds {
    let numerator = product(left, right);
    let mut quotient = zero();
    let mut remainder = zero();
    for step in 0..128u32 {
        let bit = 127 - step;
        let word = select(
            bit >= 96,
            numerator.extract(3usize),
            select(
                bit >= 64,
                numerator.extract(2usize),
                select(
                    bit >= 32,
                    numerator.extract(1usize),
                    numerator.extract(0usize),
                ),
            ),
        );
        let digit = (word >> (bit % 32)) & 1;
        let carry = remainder.extract(HIGH_WORD) >> 31;
        remainder = ticks(
            (remainder.extract(LOW_WORD) << 1) | digit,
            (remainder.extract(HIGH_WORD) << 1) | (remainder.extract(LOW_WORD) >> 31),
        );
        let included = carry != 0 || !lt(remainder, denominator);
        remainder = select(included, minus(remainder, denominator), remainder);
        if included && bit >= 64 {
            KernelFaults::new(comptime!(FaultFlags::ARITHMETIC_OVERFLOW)).record(control);
        }
        let low = quotient.extract(LOW_WORD) | select(included && bit < 32, 1u32 << (bit % 32), 0);
        let high =
            quotient.extract(HIGH_WORD) | select(included && bit / 32 == 1, 1u32 << (bit % 32), 0);
        quotient = ticks(low, high);
    }
    Bounds {
        lower: quotient,
        upper: sum(
            quotient,
            ticks(u32::cast_from(!equal(remainder, zero())), 0),
            control,
        ),
    }
}

#[cube(launch)]
pub fn vector_dot(
    left: &[u32],
    right: &[u32],
    output: &mut [u32],
    control: &[Atomic<u32>],
    length: u32,
    scale: u32,
) {
    if ABSOLUTE_POS == 0 {
        let mut lower = zero();
        let mut upper = zero();
        for index in 0..length {
            let a = interval(left, index as usize);
            let b = interval(right, index as usize);
            lower = sum(
                lower,
                divide_product(a.lower, b.lower, ticks(scale, 0), control).lower,
                control,
            );
            upper = sum(
                upper,
                divide_product(a.upper, b.upper, ticks(scale, 0), control).upper,
                control,
            );
        }
        write(output, 0, Bounds { lower, upper });
    }
}

#[cube(launch)]
pub fn normalize(
    input: &[u32],
    output: &mut [u32],
    control: &[Atomic<u32>],
    length: u32,
    reference: u32,
    scale: u32,
) {
    let index = ABSOLUTE_POS;
    if index < length as usize {
        let divisor = interval(input, reference as usize);
        if equal(divisor.lower, zero()) {
            control[0].store(UNRESOLVED);
        }
        if !equal(divisor.lower, zero()) {
            let value = interval(input, index);
            let mut lower =
                divide_product(value.lower, ticks(scale, 0), divisor.upper, control).lower;
            let mut upper =
                divide_product(value.upper, ticks(scale, 0), divisor.lower, control).upper;
            if index == reference as usize {
                lower = ticks(scale, 0);
                upper = ticks(scale, 0);
            }
            write(output, index, Bounds { lower, upper });
        }
    }
}

#[cube(launch)]
pub fn certificate_rows(
    offsets: &[u32],
    entries: &[u32],
    weights: &[u32],
    output: &mut [u32],
    control: &[Atomic<u32>],
    rows: u32,
    numerator: u32,
    denominator: u32,
) {
    let row = ABSOLUTE_POS;
    if row < rows as usize {
        let mut lower = zero();
        let mut upper = zero();
        for entry in offsets[row]..offsets[row + 1] {
            let base = entry as usize * ENTRY_WORDS;
            let part = scaled(
                ticks(0, weights[entries[base] as usize]),
                entries[base + 1],
                entries[base + 2],
                control,
            );
            lower = sum(lower, part.lower, control);
            upper = sum(upper, part.upper, control);
        }
        let target = scaled(ticks(0, weights[row]), numerator, denominator, control).lower;
        output[row * 2] = u32::cast_from(!lt(target, upper));
        output[row * 2 + 1] = u32::cast_from(!lt(lower, ticks(0, weights[row])));
    }
}

#[cube(launch)]
pub fn certificate_reduce(rows: &[u32], control: &[Atomic<u32>], count: u32) {
    if ABSOLUTE_POS == 0 {
        let mut certified = true;
        let mut inadmissible = true;
        for row in 0..count {
            certified = certified && rows[row as usize * 2] != 0;
            inadmissible = inadmissible && rows[row as usize * 2 + 1] != 0;
        }
        let status = select(
            certified,
            AVAILABLE,
            select(inadmissible, INADMISSIBLE, UNRESOLVED),
        );
        control[0].store(status);
    }
}

#[cube(launch)]
pub fn iterate(
    offsets: &[u32],
    entries: &[u32],
    rhs: &[u32],
    previous: &[u32],
    next: &mut [u32],
    control: &[Atomic<u32>],
    rows: u32,
) {
    let row = ABSOLUTE_POS;
    if row < rows as usize && control[0].load() == AVAILABLE {
        let product = row_product(offsets, entries, previous, row, control);
        let forcing = interval(rhs, row);
        let value = sum(forcing.lower, product.lower, control);
        write(
            next,
            row,
            Bounds {
                lower: value,
                upper: value,
            },
        );
    }
}

#[cube(launch)]
pub fn residual(
    offsets: &[u32],
    entries: &[u32],
    rhs: &[u32],
    estimate: &[u32],
    weights: &[u32],
    output: &mut [u32],
    control: &[Atomic<u32>],
    rows: u32,
) {
    let row = ABSOLUTE_POS;
    if row < rows as usize && control[0].load() == AVAILABLE {
        let product = row_product(offsets, entries, estimate, row, control);
        let forcing = interval(rhs, row);
        let lower = sum(forcing.lower, product.lower, control);
        let upper = sum(forcing.upper, product.upper, control);
        let approximate = interval(estimate, row).lower;
        let difference = maximum(positive(approximate, lower), positive(upper, approximate));
        let weighted = scaled(difference, 1, weights[row], control).upper;
        output[row * TICK_WORDS] = weighted.extract(LOW_WORD);
        output[row * TICK_WORDS + 1] = weighted.extract(HIGH_WORD);
    }
}

#[cube(launch)]
pub fn error_bound(
    residuals: &[u32],
    error: &mut [u32],
    control: &[Atomic<u32>],
    count: u32,
    numerator: u32,
    denominator: u32,
    tolerance_low: u32,
    tolerance_high: u32,
) {
    if ABSOLUTE_POS == 0 && control[0].load() == AVAILABLE {
        let mut norm = zero();
        for row in 0..count {
            norm = maximum(norm, read(residuals, row as usize * TICK_WORDS));
        }
        let bound = scaled(norm, denominator, denominator - numerator, control).upper;
        error[0] = bound.extract(LOW_WORD);
        error[1] = bound.extract(HIGH_WORD);
        if lt(ticks(tolerance_low, tolerance_high), bound) {
            control[0].store(UNRESOLVED);
        }
    }
}

#[cube(launch)]
pub fn solution_bounds(
    estimate: &[u32],
    weights: &[u32],
    error: &[u32],
    output: &mut [u32],
    control: &[Atomic<u32>],
    rows: u32,
) {
    let row = ABSOLUTE_POS;
    if row < rows as usize && control[0].load() == AVAILABLE {
        let bound = times(read(error, 0), weights[row]);
        bound.faults.record(control);
        let value = interval(estimate, row).lower;
        write(
            output,
            row,
            Bounds {
                lower: positive(value, bound.value),
                upper: sum(value, bound.value, control),
            },
        );
    }
}

#[cube]
fn signed_difference(output: &mut [u32], offset: usize, left: Ticks, right: Ticks) {
    let negative = lt(left, right);
    let magnitude = select(negative, minus(right, left), minus(left, right));
    output[offset] = u32::cast_from(negative);
    output[offset + 1] = magnitude.extract(LOW_WORD);
    output[offset + 2] = magnitude.extract(HIGH_WORD);
}

#[cube(launch)]
pub fn comparator(
    requirement: &[u32],
    output: &mut [u32],
    control: &[Atomic<u32>],
    living_low: u32,
    living_high: u32,
    scale: u32,
) {
    if ABSOLUTE_POS == 0 {
        let living = ticks(living_low, living_high);
        let value = interval(requirement, 0);
        signed_difference(output, 0, living, value.upper);
        signed_difference(output, 3, living, value.lower);
        if equal(value.lower, zero()) {
            control[0].store(select(equal(value.upper, zero()), INAPPLICABLE, UNRESOLVED));
        }
        if !equal(value.lower, zero()) {
            let lower = divide_product(living, ticks(scale, 0), value.upper, control).lower;
            let upper = divide_product(living, ticks(scale, 0), value.lower, control).upper;
            signed_difference(output, 6, lower, ticks(scale, 0));
            signed_difference(output, 9, upper, ticks(scale, 0));
        }
    }
}

#[cube(launch)]
pub fn dense_from_sparse(
    offsets: &[u32],
    entries: &[u32],
    output: &mut [u32],
    control: &[Atomic<u32>],
    rows: u32,
    columns: u32,
    scale: u32,
) {
    let index = ABSOLUTE_POS;
    if index < rows as usize * columns as usize {
        let row = index / columns as usize;
        let column = index % columns as usize;
        let mut lower = zero();
        let mut upper = zero();
        for entry in offsets[row]..offsets[row + 1] {
            let base = entry as usize * ENTRY_WORDS;
            if entries[base] as usize == column {
                let coefficient = scaled(
                    ticks(scale, 0),
                    entries[base + 1],
                    entries[base + 2],
                    control,
                );
                lower = coefficient.lower;
                upper = coefficient.upper;
            }
        }
        write(output, index, Bounds { lower, upper });
    }
}

#[cube(launch)]
pub fn dense_product(
    left: &[u32],
    right: &[u32],
    output: &mut [u32],
    control: &[Atomic<u32>],
    rows: u32,
    inner: u32,
    columns: u32,
    scale: u32,
) {
    let index = ABSOLUTE_POS;
    if index < rows as usize * columns as usize {
        let row = index / columns as usize;
        let column = index % columns as usize;
        let mut lower = zero();
        let mut upper = zero();
        for k in 0..inner {
            let a = interval(left, row * inner as usize + k as usize);
            let b = interval(right, k as usize * columns as usize + column);
            lower = sum(
                lower,
                divide_product(a.lower, b.lower, ticks(scale, 0), control).lower,
                control,
            );
            upper = sum(
                upper,
                divide_product(a.upper, b.upper, ticks(scale, 0), control).upper,
                control,
            );
        }
        write(output, index, Bounds { lower, upper });
    }
}

#[cube(launch)]
pub fn dense_transpose(input: &[u32], output: &mut [u32], rows: u32, columns: u32) {
    let index = ABSOLUTE_POS;
    if index < rows as usize * columns as usize {
        let row = index / columns as usize;
        let column = index % columns as usize;
        write(output, column * rows as usize + row, interval(input, index));
    }
}

#[cube(launch)]
pub fn dense_units(
    input: &[u32],
    row_factors: &[u32],
    column_factors: &[u32],
    output: &mut [u32],
    control: &[Atomic<u32>],
    rows: u32,
    columns: u32,
) {
    let index = ABSOLUTE_POS;
    if index < rows as usize * columns as usize {
        let row = index / columns as usize;
        let column = index % columns as usize;
        let value = interval(input, index);
        let lower = scaled(
            value.lower,
            row_factors[row * 2],
            row_factors[row * 2 + 1],
            control,
        )
        .lower;
        let upper = scaled(
            value.upper,
            row_factors[row * 2],
            row_factors[row * 2 + 1],
            control,
        )
        .upper;
        write(
            output,
            index,
            Bounds {
                lower: scaled(
                    lower,
                    column_factors[column * 2],
                    column_factors[column * 2 + 1],
                    control,
                )
                .lower,
                upper: scaled(
                    upper,
                    column_factors[column * 2],
                    column_factors[column * 2 + 1],
                    control,
                )
                .upper,
            },
        );
    }
}

#[cube(launch)]
pub fn dense_column(input: &[u32], output: &mut [u32], rows: u32, columns: u32, column: u32) {
    let row = ABSOLUTE_POS;
    if row < rows as usize {
        write(
            output,
            row * columns as usize + column as usize,
            interval(input, row),
        );
    }
}

#[cube(launch)]
pub fn signed_vector(positive: &[u32], negative: &[u32], output: &mut [u32], length: u32) {
    let index = ABSOLUTE_POS;
    if index < length as usize {
        let a = interval(positive, index);
        let b = interval(negative, index);
        signed_difference(output, index * 6, a.lower, b.upper);
        signed_difference(output, index * 6 + 3, a.upper, b.lower);
    }
}

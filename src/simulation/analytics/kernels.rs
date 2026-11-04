use crate::simulation::flags::{FaultFlags, KernelFaults};
use cubecl::prelude::*;

pub const ENTRY_WORDS: usize = 3;
pub const AVAILABLE: u32 = 1;
pub const INADMISSIBLE: u32 = 2;
pub const UNRESOLVED: u32 = 3;
pub const INAPPLICABLE: u32 = 4;

#[derive(CubeType, Clone, Copy)]
pub(super) struct Bounds {
    pub(super) lower: f32,
    pub(super) upper: f32,
}

#[cube]
pub(super) fn interval(input: &[f32], index: usize) -> Bounds {
    Bounds {
        lower: input[index],
        upper: input[input.len() / 2 + index],
    }
}

#[cube]
pub(super) fn write(output: &mut [f32], index: usize, bounds: Bounds) {
    output[index] = bounds.lower;
    output[output.len() / 2 + index] = bounds.upper;
}

#[cube]
pub(super) fn positive(left: f32, right: f32) -> f32 {
    max(left - right, 0.0)
}

#[cube]
pub(super) fn sum(left: f32, right: f32, control: &[Atomic<u32>]) -> f32 {
    checked(left + right, control)
}

#[cube]
pub(super) fn scaled(
    value: f32,
    numerator: u32,
    denominator: u32,
    control: &[Atomic<u32>],
) -> Bounds {
    let value = checked(value * numerator as f32 / denominator as f32, control);
    Bounds {
        lower: value,
        upper: value,
    }
}

#[cube]
pub(super) fn divide_product(
    left: f32,
    right: f32,
    denominator: f32,
    control: &[Atomic<u32>],
) -> Bounds {
    let value = checked(left * right / denominator, control);
    Bounds {
        lower: value,
        upper: value,
    }
}

#[cube]
fn checked_vector<N: Size>(value: Vector<f32, N>, control: &[Atomic<u32>]) -> Vector<f32, N> {
    #[unroll]
    for lane in 0..N::value() {
        checked(value.extract(lane), control);
    }
    value
}

#[cube]
pub fn rescale<N: Size>(
    values: &mut [Vector<f32, N>],
    control: &[Atomic<u32>],
    scale: f32,
    length: u32,
) {
    let index = ABSOLUTE_POS;
    if index < length as usize / N::value() {
        values[index] = checked_vector(values[index] / Vector::new(scale), control);
    }
}

#[cube]
pub fn vector_add<N: Size>(
    left: &[Vector<f32, N>],
    right: &[Vector<f32, N>],
    output: &mut [Vector<f32, N>],
    control: &[Atomic<u32>],
    length: u32,
) {
    let index = ABSOLUTE_POS;
    if index < length as usize * 2 / N::value() {
        output[index] = checked_vector(left[index] + right[index], control);
    }
}

#[cube]
pub fn normalize(
    input: &[f32],
    output: &mut [f32],
    control: &[Atomic<u32>],
    length: u32,
    reference: u32,
    scale: u32,
) {
    let index = ABSOLUTE_POS;
    if index < length as usize {
        let divisor = interval(input, reference as usize);
        if divisor.lower == 0.0 {
            control[0].store(UNRESOLVED);
        }
        if divisor.lower > 0.0 {
            let value = interval(input, index);
            write(
                output,
                index,
                Bounds {
                    lower: select(
                        index == reference as usize,
                        scale as f32,
                        divide_product(value.lower, scale as f32, divisor.upper, control).lower,
                    ),
                    upper: select(
                        index == reference as usize,
                        scale as f32,
                        divide_product(value.upper, scale as f32, divisor.lower, control).upper,
                    ),
                },
            );
        }
    }
}

#[cube]
pub fn certificate_rows(
    product: &[f32],
    weights: &[f32],
    output: &mut [u32],
    rows: u32,
    numerator: u32,
    denominator: u32,
) {
    let row = ABSOLUTE_POS;
    if row < rows as usize {
        let value = interval(product, row);
        let weight = interval(weights, row).lower;
        output[row * 2] = u32::cast_from(
            (numerator as f32) < denominator as f32
                && value.upper <= weight * numerator as f32 / denominator as f32,
        );
        output[row * 2 + 1] = u32::cast_from(value.lower >= weight);
    }
}

#[cube]
pub fn certificate_status(flags: &[u32], control: &[Atomic<u32>]) {
    if ABSOLUTE_POS == 0 {
        control[0].store(select(
            flags[0] != 0,
            AVAILABLE,
            select(flags[1] != 0, INADMISSIBLE, UNRESOLVED),
        ));
    }
}

#[cube]
pub fn residual(product: &[f32], estimate: &[f32], weights: &[u32], output: &mut [f32], rows: u32) {
    let row = ABSOLUTE_POS;
    if row < rows as usize {
        let product = interval(product, row);
        let approximate = interval(estimate, row);
        output[row] = max(
            f32::abs(product.lower - approximate.lower),
            f32::abs(product.upper - approximate.upper),
        ) / weights[row] as f32;
    }
}

#[cube]
pub fn error_bound(
    norm: &[f32],
    error: &mut [f32],
    control: &[Atomic<u32>],
    numerator: u32,
    denominator: u32,
    tolerance: f32,
) {
    if ABSOLUTE_POS == 0 {
        let bound = checked(
            max(norm[0], 0.0) * denominator as f32 / (denominator - numerator) as f32,
            control,
        );
        error[0] = bound;
        if control[0].load() == AVAILABLE && bound > tolerance {
            control[0].store(UNRESOLVED);
        }
    }
}

#[cube]
pub fn solution_bounds(
    estimate: &[f32],
    weights: &[u32],
    error: &[f32],
    output: &mut [f32],
    control: &[Atomic<u32>],
    rows: u32,
) {
    let row = ABSOLUTE_POS;
    if row < rows as usize && control[0].load() == AVAILABLE {
        let bound = checked(error[0] * weights[row] as f32, control);
        let value = interval(estimate, row);
        write(
            output,
            row,
            Bounds {
                lower: max(value.lower - bound, 0.0),
                upper: sum(value.upper, bound, control),
            },
        );
    }
}

#[cube]
fn signed_difference(
    output: &mut [u32],
    control: &[Atomic<u32>],
    offset: usize,
    left: f32,
    right: f32,
    #[comptime] rounding: Rounding,
) {
    let difference = match rounding {
        Rounding::Lower => f32::floor(left - right),
        Rounding::Upper => f32::ceil(left - right),
    };
    let magnitude = u64::cast_from(checked(f32::abs(difference), control));
    output[offset] = u32::cast_from(difference < 0.0);
    output[offset + 1] = magnitude as u32;
    output[offset + 2] = (magnitude >> 32) as u32;
}

#[cube]
pub fn comparator(
    requirement: &[f32],
    output: &mut [u32],
    control: &[Atomic<u32>],
    living_low: u32,
    living_high: u32,
    scale: u32,
) {
    if ABSOLUTE_POS == 0 {
        let living = (u64::cast_from(living_low) | (u64::cast_from(living_high) << 32)) as f32;
        let value = interval(requirement, 0);
        signed_difference(output, control, 0, living, value.upper, Rounding::Lower);
        signed_difference(output, control, 3, living, value.lower, Rounding::Upper);
        if value.lower == 0.0 {
            control[0].store(select(value.upper == 0.0, INAPPLICABLE, UNRESOLVED));
        }
        if value.lower > 0.0 {
            signed_difference(
                output,
                control,
                6,
                living * scale as f32 / value.upper,
                scale as f32,
                Rounding::Lower,
            );
            signed_difference(
                output,
                control,
                9,
                living * scale as f32 / value.lower,
                scale as f32,
                Rounding::Upper,
            );
        }
    }
}

#[cube]
pub fn dense_units(
    input: &[f32],
    row_factors: &[u32],
    column_factors: &[u32],
    output: &mut [f32],
    control: &[Atomic<u32>],
    rows: u32,
    columns: u32,
) {
    let index = ABSOLUTE_POS;
    if index < rows as usize * columns as usize {
        let row = index / columns as usize;
        let column = index % columns as usize;
        let factor = row_factors[row * 2] as f32 / row_factors[row * 2 + 1] as f32
            * column_factors[column * 2] as f32
            / column_factors[column * 2 + 1] as f32;
        let value = interval(input, index);
        write(
            output,
            index,
            Bounds {
                lower: checked(value.lower * factor, control),
                upper: checked(value.upper * factor, control),
            },
        );
    }
}

#[cube]
pub fn dense_column(input: &[f32], output: &mut [f32], rows: u32, columns: u32, column: u32) {
    let row = ABSOLUTE_POS;
    if row < rows as usize {
        write(
            output,
            row * columns as usize + column as usize,
            interval(input, row),
        );
    }
}

#[cube]
pub fn signed_vector(
    positive: &[f32],
    negative: &[f32],
    output: &mut [u32],
    control: &[Atomic<u32>],
    length: u32,
) {
    let index = ABSOLUTE_POS;
    if index < length as usize {
        let a = interval(positive, index);
        let b = interval(negative, index);
        signed_difference(
            output,
            control,
            index * 6,
            a.lower,
            b.upper,
            Rounding::Lower,
        );
        signed_difference(
            output,
            control,
            index * 6 + 3,
            a.upper,
            b.lower,
            Rounding::Upper,
        );
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum Rounding {
    Lower,
    Upper,
}

#[cube]
#[allow(
    clippy::manual_range_contains,
    reason = "CubeCL kernels require scalar comparisons"
)]
pub(super) fn checked(value: f32, control: &[Atomic<u32>]) -> f32 {
    if !(value >= 0.0 && value < 18446744073709551616.0) {
        KernelFaults::new(comptime!(FaultFlags::ARITHMETIC_OVERFLOW)).record(control);
    }
    value
}

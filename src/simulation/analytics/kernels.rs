use crate::simulation::storage::flags::{FaultFlags, KernelFaults};
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
impl Bounds {
    pub fn read(input: &[f32], index: usize) -> Self {
        Bounds {
            lower: input[index],
            upper: input[input.len() / 2 + index],
        }
    }

    pub fn store(&self, output: &mut [f32], index: usize) {
        output[index] = self.lower;
        output[output.len() / 2 + index] = self.upper;
    }

    pub fn quotient(&self, divisor: &Bounds, scale: u32, control: &[Atomic<u32>]) -> Bounds {
        Bounds {
            lower: divide_product(self.lower, scale as f32, divisor.upper, control).lower,
            upper: divide_product(self.upper, scale as f32, divisor.lower, control).upper,
        }
    }

    pub fn product_quotient(
        &self,
        right: Bounds,
        divisor: &Bounds,
        control: &[Atomic<u32>],
    ) -> Bounds {
        Bounds {
            lower: divide_product(self.lower, right.lower, divisor.upper, control).lower,
            upper: divide_product(self.upper, right.upper, divisor.lower, control).upper,
        }
    }

    pub fn absolute_difference(&self, right: &Bounds) -> Bounds {
        Bounds {
            lower: max(
                positive(self.lower, right.upper),
                positive(right.lower, self.upper),
            ),
            upper: max(
                positive(self.upper, right.lower),
                positive(right.upper, self.lower),
            ),
        }
    }
}

#[cube]
fn positive(left: f32, right: f32) -> f32 {
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
pub(super) fn checked_vector<N: Size>(
    value: Vector<f32, N>,
    control: &[Atomic<u32>],
) -> Vector<f32, N> {
    #[unroll]
    for lane in 0..N::value() {
        checked(value.extract(lane), control);
    }
    value
}

#[derive(CubeType)]
pub(super) struct SignedBoundsWriter<'a> {
    output: &'a mut [u32],
    control: &'a [Atomic<u32>],
}

#[cube]
impl<'a> SignedBoundsWriter<'a> {
    pub fn new(output: &'a mut [u32], control: &'a [Atomic<u32>]) -> Self {
        SignedBoundsWriter::<'a> { output, control }
    }

    pub fn difference(
        &mut self,
        offset: usize,
        left: f32,
        right: f32,
        #[comptime] rounding: Rounding,
    ) {
        let difference = match rounding {
            Rounding::Lower => f32::floor(left - right),
            Rounding::Upper => f32::ceil(left - right),
        };
        let magnitude = u64::cast_from(checked(f32::abs(difference), self.control));
        self.output[offset] = u32::cast_from(difference < 0.0);
        self.output[offset + 1] = magnitude as u32;
        self.output[offset + 2] = (magnitude >> 32) as u32;
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(super) enum Rounding {
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

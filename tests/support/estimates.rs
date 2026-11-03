use bad_econ::simulation::analytics::{Diagnostic, Interval, Sign, SignedInterval, SignedTicks};

pub trait Estimate {
    fn components(self) -> Vec<f64>;
}

impl Estimate for f64 {
    fn components(self) -> Vec<f64> {
        vec![self]
    }
}

impl Estimate for i128 {
    fn components(self) -> Vec<f64> {
        vec![self as f64]
    }
}

impl Estimate for Interval {
    fn components(self) -> Vec<f64> {
        assert!(self.lower <= self.upper);
        vec![self.lower as f64, self.upper as f64]
    }
}

impl Estimate for SignedTicks {
    fn components(self) -> Vec<f64> {
        vec![match self.sign {
            Sign::NonNegative => self.magnitude as f64,
            Sign::Negative => -(self.magnitude as f64),
        }]
    }
}

impl Estimate for SignedInterval {
    fn components(self) -> Vec<f64> {
        let lower = self.lower.components()[0];
        let upper = self.upper.components()[0];
        assert!(lower <= upper);
        vec![lower, upper]
    }
}

impl<T: Estimate> Estimate for Vec<T> {
    fn components(self) -> Vec<f64> {
        self.into_iter().flat_map(Estimate::components).collect()
    }
}

impl<T: Estimate, const N: usize> Estimate for [T; N] {
    fn components(self) -> Vec<f64> {
        self.into_iter().flat_map(Estimate::components).collect()
    }
}

impl<T: Estimate + std::fmt::Debug> Estimate for Diagnostic<T> {
    fn components(self) -> Vec<f64> {
        match self {
            Self::Available(value) => value.components(),
            other => panic!("expected an available estimate, got {other:?}"),
        }
    }
}

pub fn assert_close(actual: impl Estimate, expected: impl Estimate) {
    let actual = actual.components();
    let expected = expected.components();
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.into_iter().zip(expected) {
        let tolerance = 1.0 + 8.0 * f64::from(f32::EPSILON) * expected.abs();
        assert!(
            (actual - expected).abs() <= tolerance,
            "{actual} differs from {expected} by more than {tolerance}"
        );
    }
}

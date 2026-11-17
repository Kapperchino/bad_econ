use bad_econ::simulation::analytics::{Interval, Sign, SignedInterval, SignedTicks};

pub fn assert_close(actual: f64, expected: f64) {
    let tolerance = 1.0 + 8.0 * f64::from(f32::EPSILON) * expected.abs();
    assert!(
        (actual - expected).abs() <= tolerance,
        "{actual} differs from {expected} by more than {tolerance}"
    );
}

pub fn assert_ticks(actual: &[Interval], expected: &[u64]) {
    assert_eq!(actual.len(), expected.len());
    for (interval, &ticks) in actual.iter().zip(expected) {
        assert!(interval.lower <= interval.upper);
        assert_close(interval.lower as f64, ticks as f64);
        assert_close(interval.upper as f64, ticks as f64);
    }
}

pub fn assert_signed_ticks(actual: &[SignedInterval], expected: &[i64]) {
    let signed = |ticks: SignedTicks| match ticks.sign {
        Sign::NonNegative => ticks.magnitude as f64,
        Sign::Negative => -(ticks.magnitude as f64),
    };
    assert_eq!(actual.len(), expected.len());
    for (interval, &ticks) in actual.iter().zip(expected) {
        assert_close(signed(interval.lower), ticks as f64);
        assert_close(signed(interval.upper), ticks as f64);
    }
}

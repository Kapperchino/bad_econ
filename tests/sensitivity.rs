#![cfg(feature = "compute")]
use crate::{estimates, support};

use anyhow::{Result, anyhow};
use bad_econ::simulation::{Device, analytics::*};

fn sparse(rows: usize, columns: usize, values: &[i32], denominator: u32) -> Result<SparseMatrix> {
    SparseMatrix::new(
        rows,
        columns,
        values
            .iter()
            .enumerate()
            .map(|(index, &value)| Coefficient {
                row: index / columns,
                column: index % columns,
                numerator: value.unsigned_abs(),
                denominator,
            })
            .collect(),
    )
}

fn matrix(
    device: &Device,
    rows: usize,
    columns: usize,
    values: &[i32],
    denominator: u32,
    scale: u32,
) -> Result<DeviceMatrix> {
    DeviceMatrix::new(device, &sparse(rows, columns, values, denominator)?, scale)
}

fn direction(
    device: &Device,
    rows: usize,
    columns: usize,
    values: &[i32],
    denominator: u32,
    scale: u32,
) -> Result<DeviceSignedMatrix> {
    DeviceSignedMatrix::difference(
        matrix(
            device,
            rows,
            columns,
            &values.iter().map(|&value| value.max(0)).collect::<Vec<_>>(),
            denominator,
            scale,
        )?,
        matrix(
            device,
            rows,
            columns,
            &values.iter().map(|&value| value.min(0)).collect::<Vec<_>>(),
            denominator,
            scale,
        )?,
    )
}

fn signed(ticks: SignedTicks) -> i128 {
    match ticks.sign {
        Sign::NonNegative => i128::from(ticks.magnitude),
        Sign::Negative => -i128::from(ticks.magnitude),
    }
}

fn signed_approximate(values: &[SignedInterval], expected: &[i64]) {
    assert_eq!(values.len(), expected.len());
    for (actual, &expected) in values.iter().zip(expected) {
        estimates::assert_close(signed(actual.lower), i128::from(expected));
        estimates::assert_close(signed(actual.upper), i128::from(expected));
    }
}

fn signed_contains(value: SignedInterval, numerator: i128, denominator: i128) {
    assert!(signed(value.lower) * denominator <= numerator);
    assert!(signed(value.upper) * denominator >= numerator);
}

fn contains(value: Interval, numerator: u64, denominator: u64) {
    assert!(u128::from(value.lower) * u128::from(denominator) <= u128::from(numerator));
    assert!(u128::from(value.upper) * u128::from(denominator) >= u128::from(numerator));
}

fn available<T>(value: Diagnostic<T>) -> Result<T> {
    match value {
        Diagnostic::Available(value) => Ok(value),
        _ => Err(anyhow!("expected available analytical diagnostic")),
    }
}

#[test]
fn lean_inverse_labor_coefficient_and_reproduction_derivatives() -> Result<()> {
    let device = support::device(64)?;
    let scale = 24_000;
    let inverse = matrix(&device, 2, 2, &[2, 2, 0, 1], 1, scale)?;
    let labor = matrix(&device, 1, 2, &[2, 3], 1, scale)?;
    let goods_change = direction(&device, 2, 2, &[0, 1, 0, 0], 4, scale)?;
    let direct_change = direction(&device, 1, 2, &[0, -1], 1, scale)?;
    signed_approximate(
        &Sensitivity::inverse(&device, &inverse, &goods_change)?.read(&device)?,
        &[0, 12_000, 0, 0],
    );
    let labor_change = Sensitivity::labor(
        &device,
        &labor,
        &inverse,
        LaborDirection {
            goods: &goods_change,
            direct_labor: &direct_change,
        },
    )?;
    signed_approximate(&labor_change.read(&device)?, &[0, -12_000]);
    let coefficient = Sensitivity::coefficient(
        &device,
        &labor,
        &inverse,
        CoefficientDirection {
            input: 0,
            output: 1,
        },
    )?;
    estimates::assert_close(
        coefficient.read(&device)?,
        [
            Interval { lower: 0, upper: 0 },
            Interval {
                lower: 48_000,
                upper: 48_000,
            },
        ],
    );
    let reproduction = matrix(&device, 2, 1, &[3, 2], 24, scale)?;
    let reproduction_change = direction(&device, 2, 1, &[0, 1], 2, scale)?;
    let training_change = direction(&device, 1, 1, &[-1], 4, scale)?;
    let change = Sensitivity::reproduction(
        &device,
        ReproductionDirection {
            labor: &labor,
            reproduction: &reproduction,
            labor_change: &labor_change,
            reproduction_change: &reproduction_change,
            training_change: &training_change,
        },
    )?;
    signed_approximate(&change.read(&device)?, &[29_000]);
    let conversion_inverse = matrix(&device, 1, 1, &[4], 1, scale)?;
    let beta = DeviceVector::from_ticks(&device, &[96_000])?;
    signed_approximate(
        &Sensitivity::conversion(&device, &conversion_inverse, &beta, &change)?.read(&device)?,
        &[464_000],
    );
    Ok(())
}

#[test]
fn lean_asymmetric_conversion_uses_transposed_direction() -> Result<()> {
    let device = support::device(64)?;
    let inverse = matrix(&device, 2, 2, &[2, 0, 0, 1], 1, 1000)?;
    let beta = DeviceVector::from_ticks(&device, &[2000, 1000])?;
    let change = direction(&device, 2, 2, &[1, 2, -1, 0], 4, 1000)?;
    signed_approximate(
        &Sensitivity::conversion(&device, &inverse, &beta, &change)?.read(&device)?,
        &[500, 1000],
    );
    Ok(())
}

#[test]
fn lean_normalized_and_scalar_derivatives_keep_negative_directions() -> Result<()> {
    let device = support::device(64)?;
    let beta = DeviceVector::from_ticks(&device, &[2000, 3000, 1000])?;
    let beta_change = DeviceSignedVector::difference(
        DeviceVector::from_ticks(&device, &[1000, 0, 1000])?,
        DeviceVector::from_ticks(&device, &[0, 1000, 0])?,
    )?;
    let alpha_change = available(Sensitivity::normalized(
        &device,
        &beta,
        &beta_change,
        0,
        1000,
    )?)?;
    signed_approximate(&alpha_change.read(&device)?, &[0, -1250, 250]);
    let labor = matrix(&device, 3, 2, &[2, 3, 1, 4, 5, 0], 1, 1000)?;
    let labor_change = direction(&device, 3, 2, &[2, 0, 0, -2, 1, 0], 2, 1000)?;
    let alpha = available(beta.normalize(&device, 0, 1000)?)?;
    signed_approximate(
        &Sensitivity::scalar_values(&device, &labor, &alpha, &labor_change, &alpha_change)?
            .read(&device)?,
        &[1250, -6500],
    );
    Ok(())
}

#[test]
fn lean_finite_labor_difference_is_not_an_infinitesimal() -> Result<()> {
    let device = support::device(64)?;
    let scale = 6000;
    let old_goods = matrix(&device, 2, 2, &[2, 4, 0, 0], 4, scale)?;
    let new_goods = matrix(&device, 2, 2, &[1, 2, 0, 0], 4, scale)?;
    let old_direct_labor = matrix(&device, 1, 2, &[1, 1], 1, scale)?;
    let new_direct_labor = matrix(&device, 1, 2, &[1, 2], 1, scale)?;
    let old_labor = matrix(&device, 1, 2, &[2, 3], 1, scale)?;
    let new_inverse = matrix(&device, 2, 2, &[4, 2, 0, 3], 3, scale)?;
    signed_approximate(
        &Sensitivity::finite_labor_change(
            &device,
            FiniteLaborChange {
                old_goods: &old_goods,
                new_goods: &new_goods,
                old_direct_labor: &old_direct_labor,
                new_direct_labor: &new_direct_labor,
                old_labor: &old_labor,
                new_inverse: &new_inverse,
            },
        )?
        .read(&device)?,
        &[-4000, -2000],
    );
    Ok(())
}

#[test]
fn lean_residual_bound_uses_absolute_signed_residual() -> Result<()> {
    let device = support::device(64)?;
    let inverse = matrix(&device, 2, 2, &[2, 2, 0, 1], 1, 1000)?;
    let residual = DeviceSignedVector::difference(
        DeviceVector::from_ticks(&device, &[0, 1000])?,
        DeviceVector::from_ticks(&device, &[1000, 0])?,
    )?;
    estimates::assert_close(
        Sensitivity::residual_error(&device, &inverse, &residual)?.read(&device)?,
        [
            Interval {
                lower: 4000,
                upper: 4000,
            },
            Interval {
                lower: 1000,
                upper: 1000,
            },
        ],
    );
    Ok(())
}

#[test]
fn normalized_derivatives_round_f32_estimates_outward() -> Result<()> {
    let device = support::device(64)?;
    let unit = 1u64 << 40;
    let beta = DeviceVector::from_ticks(&device, &[unit * 3, unit * 2])?;
    let change = DeviceSignedVector::difference(
        DeviceVector::from_ticks(&device, &[unit, 0])?,
        DeviceVector::from_ticks(&device, &[0, unit])?,
    )?;
    let result =
        available(Sensitivity::normalized(&device, &beta, &change, 0, 1000)?)?.read(&device)?;
    signed_approximate(&result[..1], &[0]);
    signed_contains(result[1], -5000, 9);
    assert!(signed(result[1].upper) - signed(result[1].lower) <= 3);
    let reference = DeviceVector::from_ticks(&device, &[1])?;
    let reference_change = DeviceSignedVector::difference(
        DeviceVector::from_ticks(&device, &[1 << 60])?,
        DeviceVector::from_ticks(&device, &[0])?,
    )?;
    signed_approximate(
        &available(Sensitivity::normalized(
            &device,
            &reference,
            &reference_change,
            0,
            1000,
        )?)?
        .read(&device)?,
        &[0],
    );
    Ok(())
}

#[test]
fn lean_weighted_neumann_tail_contains_true_labor_and_counts_retained_terms() -> Result<()> {
    let device = support::device(64)?;
    let goods = sparse(2, 2, &[2, 0, 0, 1], 4)?;
    let labor = matrix(&device, 2, 2, &[1, 2, 3, 1], 1, 16_000)?;
    let certificate = Contraction::new(1, 2, vec![1, 2])?;
    let report = available(LaborTruncation::new(
        &device,
        &goods,
        &labor,
        &certificate,
        NeumannTerms::new(3)?,
    )?)?
    .read(&device)?;
    assert_eq!(report.terms, 3);
    assert_eq!(report.weighted_tail, 20_000);
    for (&partial, expected) in report.partial.iter().zip([28_000, 42_000, 84_000, 21_000]) {
        assert_eq!(
            partial,
            Interval {
                lower: expected,
                upper: expected
            }
        );
    }
    for (&bound, numerator) in report
        .enclosure
        .iter()
        .zip([96_000, 128_000, 288_000, 64_000])
    {
        contains(bound, numerator, 3);
    }
    let zero = available(LaborTruncation::new(
        &device,
        &goods,
        &labor,
        &certificate,
        NeumannTerms::new(0)?,
    )?)?
    .read(&device)?;
    assert_eq!(zero.weighted_tail, 160_000);
    assert!(
        zero.partial
            .iter()
            .all(|value| *value == Interval { lower: 0, upper: 0 })
    );
    Ok(())
}

#[test]
fn weighted_witness_certifies_goods_with_row_sum_above_one() -> Result<()> {
    let device = support::device(64)?;
    let goods = sparse(2, 2, &[1, 2, 0, 0], 2)?;
    let labor = matrix(&device, 1, 2, &[1, 1], 1, 256_000)?;
    let report = available(LaborTruncation::new(
        &device,
        &goods,
        &labor,
        &Contraction::new(3, 4, vec![4, 1])?,
        NeumannTerms::new(4)?,
    )?)?
    .read(&device)?;
    assert_eq!(report.weighted_tail, 1_620_000);
    contains(report.enclosure[0], 512_000, 1);
    contains(report.enclosure[1], 768_000, 1);
    let zero_goods = SparseMatrix::new(2, 2, Vec::new())?;
    let exact = available(LaborTruncation::new(
        &device,
        &zero_goods,
        &labor,
        &Contraction::new(0, 1, vec![4, 1])?,
        NeumannTerms::new(1)?,
    )?)?
    .read(&device)?;
    assert_eq!(exact.weighted_tail, 0);
    assert_eq!(exact.partial, exact.enclosure);
    Ok(())
}

#[test]
fn lean_truncation_accepts_signed_direct_labor_matrices() -> Result<()> {
    let device = support::device(64)?;
    let goods = sparse(2, 2, &[2, 0, 0, 1], 4)?;
    let labor = direction(&device, 1, 2, &[1, -2], 1, 16_000)?;
    let report = available(SignedLaborTruncation::new(
        &device,
        &goods,
        &labor,
        &Contraction::new(1, 2, vec![1, 2])?,
        NeumannTerms::new(3)?,
    )?)?
    .read(&device)?;
    signed_approximate(&report.partial, &[28_000, -42_000]);
    signed_contains(report.enclosure[0], 32_000, 1);
    signed_contains(report.enclosure[1], -128_000, 3);
    assert_eq!(report.weighted_tail, 20_000);
    Ok(())
}

#[test]
fn truncation_rejects_false_certificates_and_reports_bounded_overflow() -> Result<()> {
    let device = support::device(64)?;
    let identity = sparse(1, 1, &[1], 1)?;
    let labor = matrix(&device, 1, 1, &[1], 1, 1000)?;
    assert!(matches!(
        LaborTruncation::new(
            &device,
            &identity,
            &labor,
            &Contraction::new(1, 2, vec![1])?,
            NeumannTerms::new(1)?,
        )?,
        Diagnostic::Inadmissible
    ));
    assert!(matches!(
        LaborTruncation::new(
            &device,
            &sparse(1, 1, &[1], 2)?,
            &labor,
            &Contraction::new(1, 4, vec![1])?,
            NeumannTerms::new(1)?,
        )?,
        Diagnostic::UnresolvedAccuracy
    ));
    let enormous = DeviceMatrix::new(
        &device,
        &SparseMatrix::new(
            1,
            1,
            vec![Coefficient {
                row: 0,
                column: 0,
                numerator: u32::MAX,
                denominator: 1,
            }],
        )?,
        1 << 31,
    )?;
    assert!(
        LaborTruncation::new(
            &device,
            &SparseMatrix::new(1, 1, Vec::new())?,
            &enormous,
            &Contraction::new(0, 1, vec![2])?,
            NeumannTerms::new(1)?,
        )
        .is_err()
    );
    assert!(NeumannTerms::new(4097).is_err());
    Ok(())
}

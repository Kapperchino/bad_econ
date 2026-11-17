use super::{estimates, support};

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
    estimates::assert_signed_ticks(
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
    estimates::assert_signed_ticks(&labor_change.read(&device)?, &[0, -12_000]);
    let coefficient = Sensitivity::coefficient(
        &device,
        &labor,
        &inverse,
        CoefficientDirection {
            input: 0,
            output: 1,
        },
    )?;
    estimates::assert_ticks(&coefficient.read(&device)?, &[0, 48_000]);
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
    estimates::assert_signed_ticks(&change.read(&device)?, &[29_000]);
    let conversion_inverse = matrix(&device, 1, 1, &[4], 1, scale)?;
    let beta = DeviceVector::from_ticks(&device, &[96_000])?;
    estimates::assert_signed_ticks(
        &Sensitivity::conversion(&device, &conversion_inverse, &beta, &change)?.read(&device)?,
        &[464_000],
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
    estimates::assert_signed_ticks(&alpha_change.read(&device)?, &[0, -1250, 250]);
    let labor = matrix(&device, 3, 2, &[2, 3, 1, 4, 5, 0], 1, 1000)?;
    let labor_change = direction(&device, 3, 2, &[2, 0, 0, -2, 1, 0], 2, 1000)?;
    let alpha = available(beta.normalize(&device, 0, 1000)?)?;
    estimates::assert_signed_ticks(
        &Sensitivity::scalar_values(&device, &labor, &alpha, &labor_change, &alpha_change)?
            .read(&device)?,
        &[1250, -6500],
    );
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

use super::{estimates, fixtures::dense_matrix, support};

use anyhow::{Result, anyhow};
use bad_econ::simulation::analytics::*;

fn dense(
    rows: usize,
    columns: usize,
    numerators: &[u32],
    denominators: &[u32],
) -> Result<SparseMatrix> {
    dense_matrix(
        rows,
        columns,
        numerators.iter().copied(),
        denominators.iter().copied(),
    )
}

fn grain_goods() -> Result<SparseMatrix> {
    dense(2, 2, &[1, 1, 0, 0], &[2, 1, 1, 1])
}
fn grain_labor() -> Result<SparseMatrix> {
    dense(1, 2, &[1, 1], &[1, 1])
}
fn grain_reproduction() -> Result<SparseMatrix> {
    dense(2, 1, &[1, 1], &[8, 12])
}
fn grain_training() -> Result<SparseMatrix> {
    dense(1, 1, &[1], &[4])
}

fn available(values: Diagnostic<ApproximateValues>) -> Result<ApproximateValues> {
    match values {
        Diagnostic::Available(values) => Ok(values),
        diagnostic => Err(anyhow!("unexpected diagnostic: {diagnostic:?}")),
    }
}

fn contains(interval: Interval, numerator: u64, denominator: u64) {
    assert!(u128::from(interval.lower) * u128::from(denominator) <= u128::from(numerator));
    assert!(u128::from(interval.upper) * u128::from(denominator) >= u128::from(numerator));
}

#[test]
fn lean_coupled_step_uses_old_blocks_and_normalizes_both_blocks() -> Result<()> {
    let device = support::device(64)?;
    let labor = dense(2, 2, &[1, 2, 0, 1], &[1, 1, 1, 1])?;
    let reproduction = dense(2, 2, &[1, 1, 0, 0], &[4, 8, 1, 1])?;
    let training = dense(2, 2, &[0, 0, 1, 0], &[1, 1, 4, 1])?;
    let coupled = SparseMatrix::coupled(&grain_goods()?, &labor, &reproduction, &training)?;
    let step = DeviceOperator::new(&device, &coupled).step(
        &device,
        &DeviceVector::from_ticks(&device, &[0, 0, 1000, 1000])?,
        &DeviceVector::from_ticks(&device, &[2000, 3000, 5000, 7000])?,
    )?;
    estimates::assert_ticks(&step.read(&device)?, &[6000, 19000, 3250, 1250]);
    let coupled = SparseMatrix::coupled(
        &SparseMatrix::new(2, 2, vec![])?,
        &labor,
        &reproduction,
        &training,
    )?;
    let solution = DeviceOperator::new(&device, &coupled).solve(
        &device,
        &DeviceVector::from_ticks(&device, &[0, 0, 1000, 1000])?,
        &Contraction::new(9, 10, vec![240, 648, 200, 140])?,
        SolveBudget {
            iterations: 100,
            weighted_tolerance: 10,
        },
    )?;
    let report = available(solution.read(&device)?)?;
    for (interval, numerator) in report.values.iter().zip([40_000, 108_000, 40_000, 28_000]) {
        contains(*interval, numerator, 23);
    }
    let reference = DeviceVector::from_ticks(&device, &[40_000, 108_000, 40_000, 28_000])?;
    let normalized = match reference.normalize(&device, 2, 1000)? {
        Diagnostic::Available(value) => value,
        _ => Err(anyhow!("normalization unavailable"))?,
    };
    estimates::assert_ticks(&normalized.read(&device)?, &[1000, 2700, 1000, 700]);
    Ok(())
}

#[test]
fn lean_residual_certificate_is_for_the_actual_operator() -> Result<()> {
    let device = support::device(64)?;
    let matrix = grain_goods()?;
    let rhs = DeviceVector::from_ticks(&device, &[1000, 1000])?;
    let approximate = DeviceVector::from_ticks(&device, &[1900, 2900])?;
    let bound = DeviceOperator::new(&device, &matrix.transpose()).solve_from(
        &device,
        &rhs,
        &approximate,
        &Contraction::new(1, 2, vec![1, 2])?,
        SolveBudget {
            iterations: 0,
            weighted_tolerance: 100,
        },
    )?;
    let report = available(bound.read(&device)?)?;
    assert_eq!(report.weighted_error, 100);
    assert_eq!(
        report.values,
        [
            Interval {
                lower: 1800,
                upper: 2000
            },
            Interval {
                lower: 2700,
                upper: 3100
            }
        ]
    );
    let wrong = DeviceOperator::new(&device, &matrix.transpose()).solve(
        &device,
        &rhs,
        &Contraction::new(5, 6, vec![6, 2])?,
        SolveBudget {
            iterations: 32,
            weighted_tolerance: 100,
        },
    )?;
    assert!(matches!(
        wrong.read(&device)?,
        Diagnostic::UnresolvedAccuracy
    ));
    let too_tight = DeviceOperator::new(&device, &matrix.transpose()).solve_from(
        &device,
        &rhs,
        &approximate,
        &Contraction::new(1, 2, vec![1, 2])?,
        SolveBudget {
            iterations: 0,
            weighted_tolerance: 99,
        },
    )?;
    assert!(matches!(
        too_tight.read(&device)?,
        Diagnostic::UnresolvedAccuracy
    ));
    Ok(())
}

#[test]
fn serialized_analysis_request_runs_complete_device_account() -> Result<()> {
    let device = support::device(64)?;
    let request = AnalysisRequest {
        context: ContextVersion::default(),
        scale: 1000,
        goods: grain_goods()?,
        labor: grain_labor()?,
        weights: vec![1000],
        final_demand: vec![0, 2000],
        goods_certificate: Contraction::new(5, 6, vec![6, 2])?,
        value_certificate: Contraction::new(1, 2, vec![1, 2])?,
        budget: SolveBudget {
            iterations: 128,
            weighted_tolerance: 30,
        },
        reference_occupation: 0,
        research: Some(ResearchRequest {
            inputs: dense(2, 1, &[1, 1], &[2, 1])?,
            hours: dense(1, 1, &[2], &[1])?,
            attempts: vec![1000],
        }),
        reproduction: Some(ReproductionRequest {
            goods: grain_reproduction()?,
            training: grain_training()?,
            certificate: Contraction::new(20, 21, vec![14, 21, 6])?,
        }),
        comparator: Some(ComparatorRequest {
            context: ComparatorContext {
                population: "workers".into(),
                classification: "single occupation".into(),
                period: "example".into(),
                convention: "supplied weight".into(),
            },
            living: 15000,
        }),
        valuation: None,
    };
    let restored: AnalysisRequest = serde_json::from_str(&serde_json::to_string(&request)?)?;
    let report = restored.run(&device)?;
    match &report.plan_value {
        Diagnostic::Available(total) => contains(total[0], 12000, 1),
        diagnostic => Err(anyhow!("plan total unavailable: {diagnostic:?}"))?,
    }
    assert!(matches!(
        report.normalized_coupled,
        Diagnostic::Available(_)
    ));
    match &report.residual_scale {
        Diagnostic::Available(scale) => contains(scale[0], 250, 1),
        diagnostic => Err(anyhow!("residual scale unavailable: {diagnostic:?}"))?,
    }
    assert!(matches!(report.comparator, Diagnostic::Available(_)));
    let json = serde_json::to_value(&report)?;
    assert_eq!(json["profile"], ANALYTICAL_PROFILE);
    assert!(json["goods_values"]["Available"]["values"].is_array());
    assert!(json["residual_scale"]["Available"].is_array());
    let mut invalid = serde_json::to_value(&request)?;
    invalid["goods"]["entries"][0]["denominator"] = serde_json::json!(0);
    assert!(serde_json::from_value::<AnalysisRequest>(invalid).is_err());
    Ok(())
}

#[test]
fn cubek_rectangular_products_and_transposes_match_f64_reference() -> Result<()> {
    let device = support::device(64)?;
    let rows = 5;
    let inner = 7;
    let columns = 3;
    let scale = 1000;
    let left = (0..rows * inner)
        .map(|index| (index % 11) as u32)
        .collect::<Vec<_>>();
    let right = (0..inner * columns)
        .map(|index| (index % 5) as u32)
        .collect::<Vec<_>>();
    let a = DeviceMatrix::new(
        &device,
        &dense(rows, inner, &left, &vec![17; left.len()])?,
        scale,
    )?;
    let b = DeviceMatrix::new(
        &device,
        &dense(inner, columns, &right, &vec![13; right.len()])?,
        scale,
    )?;
    let product = a.product(&device, &b)?;
    let actual = product.read(&device)?;
    let transposed = product.transpose(&device)?.read(&device)?;
    for row in 0..rows {
        for column in 0..columns {
            let expected = (0..inner)
                .map(|k| {
                    f64::from(left[row * inner + k]) / 17.0 * f64::from(right[k * columns + column])
                        / 13.0
                })
                .sum::<f64>()
                * f64::from(scale);
            let value = actual[row * columns + column];
            estimates::assert_close(value.lower as f64, expected);
            estimates::assert_close(value.upper as f64, expected);
            assert_eq!(transposed[column * rows + row], value);
        }
    }
    Ok(())
}

#[test]
fn vector_arithmetic_preserves_partial_workgroups_empty_shapes_and_lane_faults() -> Result<()> {
    for workgroup in [32, 256] {
        let device = support::device(workgroup)?;
        for length in [0, 1, 2, 3, 31, 32, 33, 63, 64, 65, 257] {
            let left = (0..length)
                .map(|index| index as u64 % 17)
                .collect::<Vec<_>>();
            let right = (0..length)
                .map(|index| index as u64 % 11)
                .collect::<Vec<_>>();
            let a = DeviceVector::from_ticks(&device, &left)?;
            let b = DeviceVector::from_ticks(&device, &right)?;
            let expected = left
                .iter()
                .zip(&right)
                .map(|(a, b)| Interval {
                    lower: a + b,
                    upper: a + b,
                })
                .collect::<Vec<_>>();
            assert_eq!(a.add(&device, &b)?.read(&device)?, expected);
            let dot = left.iter().zip(&right).map(|(a, b)| a * b).sum();
            assert_eq!(
                a.dot(&device, &b, 1)?.read(&device)?,
                [Interval {
                    lower: dot,
                    upper: dot
                }]
            );
        }
        for lane in 0..4 {
            let mut values = [1; 4];
            values[lane] = 1 << 63;
            let values = DeviceVector::from_ticks(&device, &values)?;
            assert!(values.add(&device, &values).is_err());
        }
    }
    Ok(())
}

#[test]
fn iterative_solver_retains_transient_overflow_and_recovers_on_the_next_solve() -> Result<()> {
    for workgroup in [32, 256] {
        let device = support::device(workgroup)?;
        let operator = DeviceOperator::new(&device, &dense(2, 2, &[0, 2, 0, 0], &[1; 4])?);
        let initial = DeviceVector::from_ticks(&device, &[0, 1 << 63])?;
        let before = initial.read(&device)?;
        let zero = DeviceVector::from_ticks(&device, &[0, 0])?;
        let certificate = Contraction::new(1, 2, vec![4, 1])?;
        let budget = SolveBudget {
            iterations: 2,
            weighted_tolerance: 0,
        };
        assert!(operator.step(&device, &zero, &initial).is_err());
        assert!(
            operator
                .solve_from(&device, &zero, &initial, &certificate, budget)
                .is_err()
        );
        assert_eq!(initial.read(&device)?, before);
        let recovered = available(
            operator
                .solve(&device, &zero, &certificate, budget)?
                .read(&device)?,
        )?;
        assert_eq!(recovered.values, [Interval { lower: 0, upper: 0 }; 2]);
        assert_eq!(recovered.weighted_error, 0);
    }
    Ok(())
}

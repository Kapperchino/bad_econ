#![cfg(feature = "compute")]
use crate::{estimates, support};

use anyhow::{Result, anyhow};
use bad_econ::simulation::{Device, analytics::*};

fn dense(
    rows: usize,
    columns: usize,
    numerators: &[u32],
    denominators: &[u32],
) -> Result<SparseMatrix> {
    SparseMatrix::new(
        rows,
        columns,
        numerators
            .iter()
            .zip(denominators)
            .enumerate()
            .map(|(index, (&numerator, &denominator))| Coefficient {
                row: index / columns,
                column: index % columns,
                numerator,
                denominator,
            })
            .collect(),
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

fn approximate(vector: &DeviceVector, device: &Device, expected: &[u64]) -> Result<()> {
    estimates::assert_close(
        vector.read(device)?,
        expected
            .iter()
            .map(|&value| Interval {
                lower: value,
                upper: value,
            })
            .collect::<Vec<_>>(),
    );
    Ok(())
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
    approximate(&step, &device, &[6000, 19000, 3250, 1250])?;
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
    approximate(&normalized, &device, &[1000, 2700, 1000, 700])?;
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
fn goods_remain_available_when_optional_closure_is_inadmissible() -> Result<()> {
    let device = support::device(64)?;
    let zero = SparseMatrix::new(1, 1, vec![])?;
    let one = dense(1, 1, &[1], &[1])?;
    let goods = DeviceOperator::new(&device, &zero).solve(
        &device,
        &DeviceVector::from_ticks(&device, &[1000])?,
        &Contraction::new(0, 1, vec![1])?,
        SolveBudget {
            iterations: 1,
            weighted_tolerance: 0,
        },
    )?;
    estimates::assert_close(
        available(goods.read(&device)?)?.values,
        [Interval {
            lower: 1000,
            upper: 1000,
        }],
    );
    let coupled = SparseMatrix::coupled(&zero, &one, &one, &zero)?;
    let closure = DeviceOperator::new(&device, &coupled).solve(
        &device,
        &DeviceVector::from_ticks(&device, &[0, 1000])?,
        &Contraction::new(1, 2, vec![1, 1])?,
        SolveBudget {
            iterations: 10,
            weighted_tolerance: 1,
        },
    )?;
    assert!(matches!(closure.read(&device)?, Diagnostic::Inadmissible));
    let zeros = DeviceVector::from_ticks(&device, &[0, 1000])?;
    match zeros.normalize(&device, 1, 1000)? {
        Diagnostic::Available(value) => approximate(&value, &device, &[0, 1000])?,
        _ => Err(anyhow!("zero goods values must be allowed"))?,
    }
    assert!(matches!(
        zeros.normalize(&device, 0, 1000)?,
        Diagnostic::UnresolvedAccuracy
    ));
    Ok(())
}

#[test]
fn analytical_products_quantize_ticks_and_reject_out_of_range_estimates() -> Result<()> {
    let device = support::device(64)?;
    assert!(DeviceVector::from_ticks(&device, &[u64::MAX]).is_err());
    let left = DeviceVector::from_ticks(&device, &[1 << 63])?;
    let one = DeviceVector::from_ticks(&device, &[1])?;
    approximate(&left.dot(&device, &one, 1)?, &device, &[1 << 63])?;
    let two = DeviceVector::from_ticks(&device, &[2])?;
    assert!(left.dot(&device, &two, 1).is_err());
    assert!(left.add(&device, &left).is_err());
    let quantized = DeviceVector::from_ticks(&device, &[(1 << 24) + 1])?;
    assert_eq!(
        quantized.read(&device)?,
        [Interval {
            lower: 1 << 24,
            upper: 1 << 24
        }]
    );
    let wide = DeviceVector::from_ticks(&device, &[1u64 << 40])?;
    approximate(&wide.dot(&device, &wide, 1 << 20)?, &device, &[1u64 << 60])?;
    let thirds = DeviceOperator::new(&device, &dense(1, 1, &[1], &[3])?).apply(&device, &one)?;
    assert_eq!(thirds.read(&device)?, [Interval { lower: 0, upper: 1 }]);
    Ok(())
}

#[test]
fn empty_accounts_have_vacuous_certificates_and_exact_zero_hours() -> Result<()> {
    let device = support::device(64)?;
    let goods = SparseMatrix::new(0, 0, Vec::new())?;
    let labor = SparseMatrix::new(2, 0, Vec::new())?;
    let certificate = Contraction::new(0, 1, Vec::new())?;
    let budget = SolveBudget {
        iterations: 16,
        weighted_tolerance: 0,
    };
    let operator = DeviceOperator::new(&device, &goods);
    let solution = operator.solve(
        &device,
        &DeviceVector::from_ticks(&device, &[])?,
        &certificate,
        budget,
    )?;
    let report = available(solution.read(&device)?)?;
    assert!(report.values.is_empty());
    assert_eq!(report.weighted_error, 0);
    assert_eq!(report.iterations, 0);
    let account = AccountModel::new(goods.clone(), labor.clone())?.upload(&device)?;
    match solution.values(&device)? {
        Diagnostic::Available(values) => {
            approximate(
                &account.occupational_hours(&device, values)?,
                &device,
                &[0, 0],
            )?;
        }
        _ => Err(anyhow!("empty solution unavailable"))?,
    }
    match operator.inverse(&device, 1000, &certificate, budget)? {
        Diagnostic::Available(inverse) => {
            assert_eq!((inverse.rows(), inverse.columns()), (0, 0));
            assert!(inverse.read(&device)?.is_empty());
        }
        _ => Err(anyhow!("empty inverse unavailable"))?,
    }
    match LaborTruncation::new(
        &device,
        &goods,
        &DeviceMatrix::new(&device, &labor, 1000)?,
        &certificate,
        NeumannTerms::new(3)?,
    )? {
        Diagnostic::Available(truncation) => {
            let report = truncation.read(&device)?;
            assert!(report.partial.is_empty());
            assert!(report.enclosure.is_empty());
            assert_eq!(report.weighted_tail, 0);
        }
        _ => Err(anyhow!("empty truncation unavailable"))?,
    }
    assert!(
        DeviceOperator::new(&device, &labor)
            .inverse(&device, 1000, &certificate, budget)
            .is_err()
    );
    assert!(
        operator
            .inverse(&device, 1000, &Contraction::new(0, 1, vec![1])?, budget)
            .is_err()
    );
    Ok(())
}

#[test]
fn account_api_keeps_optional_closure_and_research_separate() -> Result<()> {
    let device = support::device(64)?;
    let model = AccountModel::new(grain_goods()?, grain_labor()?)?
        .with_reproduction(grain_reproduction()?, grain_training()?)?
        .with_research(dense(2, 1, &[1, 1], &[2, 1])?, dense(1, 1, &[2], &[1])?)?;
    let account = model.upload(&device)?;
    let goods = DeviceVector::from_ticks(&device, &[2000, 3000])?;
    let weights = DeviceVector::from_ticks(&device, &[1000])?;
    let research = match account.research_values(&device, &goods, &weights)? {
        Diagnostic::Available(values) => values,
        _ => Err(anyhow!("research account unavailable"))?,
    };
    approximate(&research, &device, &[6000])?;
    let y = DeviceVector::from_ticks(&device, &[0, 2000])?;
    let z = DeviceVector::from_ticks(&device, &[1000])?;
    approximate(
        &DeviceAccounts::plan_value(&device, &goods, &research, &y, &z, 1000)?,
        &device,
        &[12_000],
    )?;
    let plan = match account.plan(
        &device,
        &y,
        &z,
        &Contraction::new(5, 6, vec![6, 2])?,
        SolveBudget {
            iterations: 32,
            weighted_tolerance: 6,
        },
    )? {
        Diagnostic::Available(plan) => plan,
        _ => Err(anyhow!("plan unavailable"))?,
    };
    match plan.occupational_hours {
        Diagnostic::Available(hours) => contains(hours.read(&device)?[0], 12_000, 1),
        _ => Err(anyhow!("plan hours unavailable"))?,
    }
    Ok(())
}

#[test]
fn labeled_comparators_do_not_fabricate_zero_requirement_rates() -> Result<()> {
    let device = support::device(64)?;
    let context = ComparatorContext {
        population: "workers".into(),
        classification: "one occupation".into(),
        period: "day 1".into(),
        convention: "external weights".into(),
    };
    let requirement = DeviceVector::from_ticks(&device, &[8000])?;
    let report = comparator(&device, context.clone(), 12_000, &requirement, 1000)?;
    assert_eq!(report.context, context);
    assert_eq!(
        report.surplus.lower,
        SignedTicks {
            sign: Sign::NonNegative,
            magnitude: 4000
        }
    );
    estimates::assert_close(
        report.rate,
        Diagnostic::Available(SignedInterval {
            lower: SignedTicks {
                sign: Sign::NonNegative,
                magnitude: 500,
            },
            upper: SignedTicks {
                sign: Sign::NonNegative,
                magnitude: 500,
            },
        }),
    );
    let deficit = comparator(&device, context.clone(), 4000, &requirement, 1000)?;
    assert_eq!(
        deficit.surplus.lower,
        SignedTicks {
            sign: Sign::Negative,
            magnitude: 4000
        }
    );
    estimates::assert_close(
        deficit.rate,
        Diagnostic::Available(SignedInterval {
            lower: SignedTicks {
                sign: Sign::Negative,
                magnitude: 500,
            },
            upper: SignedTicks {
                sign: Sign::Negative,
                magnitude: 500,
            },
        }),
    );
    assert!(matches!(
        comparator(
            &device,
            context.clone(),
            1000,
            &DeviceVector::from_ticks(&device, &[0])?,
            1000
        )?
        .rate,
        Diagnostic::Inapplicable
    ));
    Ok(())
}

#[test]
fn approximate_inverse_and_lifetime_conventions_are_preserved() -> Result<()> {
    let device = support::device(64)?;
    let inverse = match DeviceOperator::new(&device, &grain_goods()?).inverse(
        &device,
        1000,
        &Contraction::new(5, 6, vec![6, 2])?,
        SolveBudget {
            iterations: 32,
            weighted_tolerance: 6,
        },
    )? {
        Diagnostic::Available(inverse) => inverse,
        _ => Err(anyhow!("inverse unavailable"))?,
    };
    for (value, expected) in inverse
        .read(&device)?
        .into_iter()
        .zip([2000, 2000, 0, 1000])
    {
        contains(value, expected, 1);
    }
    let subsistence = DeviceMatrix::new(&device, &dense(1, 2, &[1, 2], &[1, 1])?, 1000)?;
    let education = DeviceMatrix::new(&device, &dense(1, 2, &[4, 8], &[1, 1])?, 1000)?;
    let per_hour = DeviceMatrix::per_hour_goods(&device, &subsistence, &education, &[2, 4])?;
    estimates::assert_close(
        per_hour.read(&device)?,
        [
            Interval {
                lower: 3000,
                upper: 3000,
            },
            Interval {
                lower: 4000,
                upper: 4000,
            },
        ],
    );
    assert!(education.per_hour_training(&device, &[0, 4]).is_err());
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
fn comparator_uses_positive_fractional_denominators() -> Result<()> {
    let device = support::device(64)?;
    let requirement = DeviceOperator::new(&device, &dense(1, 1, &[1], &[3])?)
        .apply(&device, &DeviceVector::from_ticks(&device, &[1])?)?;
    let context = ComparatorContext {
        population: "workers".into(),
        classification: "one".into(),
        period: "day".into(),
        convention: "external weights".into(),
    };
    let report = comparator(&device, context, 10, &requirement, 1000)?;
    let rate = match report.rate {
        Diagnostic::Available(rate) => rate,
        other => panic!("unexpected fractional comparator: {other:?}"),
    };
    estimates::assert_close(
        rate,
        SignedInterval {
            lower: SignedTicks {
                sign: Sign::NonNegative,
                magnitude: 29_000,
            },
            upper: SignedTicks {
                sign: Sign::NonNegative,
                magnitude: 29_000,
            },
        },
    );
    Ok(())
}

#[test]
fn wide_products_match_independent_f64_fixtures_with_f32_tolerance() -> Result<()> {
    let device = support::device(64)?;
    for index in 0..48u64 {
        let draw = bad_econ::arithmetic::DrawKey {
            seed: 901,
            day: 0,
            phase: 0,
            actor: index,
            purpose: 0,
            index: 0,
        }
        .draw();
        let left = match index % 3 {
            0 => u64::from(draw as u32),
            1 => draw & ((1u64 << 40) - 1),
            _ => 1 << 60,
        };
        let right = match index % 3 {
            0 => draw.rotate_left(17),
            1 => draw.rotate_left(29) & ((1u64 << 40) - 1),
            _ => 1,
        };
        let divisor = match index % 3 {
            0 => u32::MAX,
            1 => (draw as u32).max(1 << 20),
            _ => 1,
        };
        let actual = DeviceVector::from_ticks(&device, &[left])?
            .dot(
                &device,
                &DeviceVector::from_ticks(&device, &[right])?,
                divisor,
            )?
            .read(&device)?[0];
        let expected = left as f64 * right as f64 / f64::from(divisor);
        estimates::assert_close(actual.lower as f64, expected);
        estimates::assert_close(actual.upper as f64, expected);
    }
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
    let empty = DeviceVector::from_ticks(&device, &[])?;
    assert_eq!(
        empty.dot(&device, &empty, 1)?.read(&device)?,
        [Interval { lower: 0, upper: 0 }]
    );
    Ok(())
}

#[test]
fn contraction_rounded_to_one_does_not_admit_an_identity_operator() -> Result<()> {
    let device = support::device(64)?;
    let solution = DeviceOperator::new(&device, &dense(1, 1, &[1], &[1])?).solve(
        &device,
        &DeviceVector::from_ticks(&device, &[0])?,
        &Contraction::new(u32::MAX - 1, u32::MAX, vec![1])?,
        SolveBudget {
            iterations: 0,
            weighted_tolerance: 0,
        },
    )?;
    assert!(matches!(solution.read(&device)?, Diagnostic::Inadmissible));
    Ok(())
}

#[test]
fn vector_arithmetic_preserves_partial_workgroups_empty_shapes_and_lane_faults() -> Result<()> {
    for workgroup in [32, 64, 128, 256] {
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
fn iterative_solver_matches_gpu_steps_and_preserves_aliased_inputs() -> Result<()> {
    for workgroup in [32, 64, 128, 256] {
        let device = support::device(workgroup)?;
        for rows in [0, 1, 2, 3, 32, 33, 65] {
            let entries = (0..rows)
                .flat_map(|row| match rows {
                    1 => vec![Coefficient {
                        row,
                        column: row,
                        numerator: 1,
                        denominator: 2,
                    }],
                    _ => [row, (row + 1) % rows]
                        .map(|column| Coefficient {
                            row,
                            column,
                            numerator: 1,
                            denominator: 4,
                        })
                        .to_vec(),
                })
                .collect();
            let operator = DeviceOperator::new(&device, &SparseMatrix::new(rows, rows, entries)?);
            let values = (0..rows).map(|row| 10 + row as u64 % 7).collect::<Vec<_>>();
            let input = DeviceVector::from_ticks(&device, &values)?;
            let before = input.read(&device)?;
            let certificate = Contraction::new(1, 2, vec![1; rows])?;
            for iterations in [0, 1, 2, 7] {
                let mut expected = DeviceVector::from_ticks(&device, &values)?;
                for _ in 0..iterations {
                    expected = operator.step(&device, &input, &expected)?;
                }
                let reference = available(
                    operator
                        .solve_from(
                            &device,
                            &input,
                            &expected,
                            &certificate,
                            SolveBudget {
                                iterations: 0,
                                weighted_tolerance: u64::MAX,
                            },
                        )?
                        .read(&device)?,
                )?;
                let actual = available(
                    operator
                        .solve_from(
                            &device,
                            &input,
                            &input,
                            &certificate,
                            SolveBudget {
                                iterations,
                                weighted_tolerance: u64::MAX,
                            },
                        )?
                        .read(&device)?,
                )?;
                assert_eq!(
                    actual.values, reference.values,
                    "{workgroup} {rows} {iterations}"
                );
                assert_eq!(actual.weighted_error, reference.weighted_error);
                assert_eq!(
                    actual.iterations,
                    match rows {
                        0 => 0,
                        _ => iterations,
                    }
                );
                assert_eq!(input.read(&device)?, before);
            }
        }
    }
    Ok(())
}

#[test]
fn iterative_solver_retains_transient_overflow_and_recovers_on_the_next_solve() -> Result<()> {
    for workgroup in [32, 64, 128, 256] {
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

#[test]
fn solver_preserves_certificate_overflow_without_any_iterations() -> Result<()> {
    for workgroup in [32, 64, 128, 256] {
        let device = support::device(workgroup)?;
        let operator = DeviceOperator::new(&device, &dense(1, 1, &[u32::MAX], &[1])?);
        let weights = DeviceVector::from_ticks(&device, &[u64::from(u32::MAX)])?;
        let zero = DeviceVector::from_ticks(&device, &[0])?;
        assert!(operator.apply(&device, &weights).is_err());
        assert!(
            operator
                .solve(
                    &device,
                    &zero,
                    &Contraction::new(1, 2, vec![u32::MAX])?,
                    SolveBudget {
                        iterations: 0,
                        weighted_tolerance: 0,
                    },
                )
                .is_err()
        );
    }
    Ok(())
}

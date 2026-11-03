#![cfg(feature = "compute")]
use crate::support;

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

fn available(values: Diagnostic<CertifiedValues>) -> Result<CertifiedValues> {
    match values {
        Diagnostic::Available(values) => Ok(values),
        diagnostic => Err(anyhow!("unexpected diagnostic: {diagnostic:?}")),
    }
}

fn exact(vector: &DeviceVector, device: &Device, expected: &[u64]) -> Result<()> {
    assert_eq!(
        vector.read(device)?,
        expected
            .iter()
            .map(|&value| Interval {
                lower: value,
                upper: value
            })
            .collect::<Vec<_>>()
    );
    Ok(())
}

fn contains(interval: Interval, numerator: u64, denominator: u64) {
    assert!(u128::from(interval.lower) * u128::from(denominator) <= u128::from(numerator));
    assert!(u128::from(interval.upper) * u128::from(denominator) >= u128::from(numerator));
}

#[test]
fn lean_sparse_accounts_research_and_plan_totals() -> Result<()> {
    let device = support::device(64)?;
    let transpose = DeviceOperator::new(&device, &grain_goods()?.transpose());
    let labor = DeviceOperator::new(&device, &grain_labor()?);
    let forcing = DeviceOperator::new(&device, &grain_labor()?.transpose())
        .apply(&device, &DeviceVector::exact(&device, &[1000])?)?;
    let values = transpose.solve(
        &device,
        &forcing,
        &Contraction::new(1, 2, vec![2, 4])?,
        SolveBudget {
            iterations: 32,
            weighted_tolerance: 2,
        },
    )?;
    let report = available(values.read(&device)?)?;
    contains(report.values[0], 2000, 1);
    contains(report.values[1], 3000, 1);
    let goods_values = DeviceVector::exact(&device, &[2000, 3000])?;
    let j = dense(2, 1, &[1, 1], &[2, 1])?;
    let f = dense(1, 1, &[2], &[1])?;
    let attempts = DeviceVector::exact(&device, &[1000])?;
    let final_demand = DeviceVector::exact(&device, &[0, 2000])?;
    let research = DeviceOperator::new(&device, &j.transpose())
        .apply(&device, &goods_values)?
        .add(
            &device,
            &DeviceOperator::new(&device, &f.transpose())
                .apply(&device, &DeviceVector::exact(&device, &[1000])?)?,
        )?;
    exact(&research, &device, &[6000])?;
    let total = goods_values
        .dot(&device, &final_demand, 1000)?
        .add(&device, &research.dot(&device, &attempts, 1000)?)?;
    exact(&total, &device, &[12_000])?;
    let gross_demand = final_demand.add(
        &device,
        &DeviceOperator::new(&device, &j).apply(&device, &attempts)?,
    )?;
    let gross = DeviceOperator::new(&device, &grain_goods()?).solve(
        &device,
        &gross_demand,
        &Contraction::new(5, 6, vec![6, 2])?,
        SolveBudget {
            iterations: 32,
            weighted_tolerance: 6,
        },
    )?;
    let requirements = available(gross.read(&device)?)?;
    contains(requirements.values[0], 7000, 1);
    contains(requirements.values[1], 3000, 1);
    let hours = labor
        .apply(&device, &DeviceVector::exact(&device, &[7000, 3000])?)?
        .add(
            &device,
            &DeviceOperator::new(&device, &f).apply(&device, &attempts)?,
        )?;
    exact(&hours, &device, &[12_000])?;
    Ok(())
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
        &DeviceVector::exact(&device, &[0, 0, 1000, 1000])?,
        &DeviceVector::exact(&device, &[2000, 3000, 5000, 7000])?,
    )?;
    exact(&step, &device, &[6000, 19000, 3250, 1250])?;
    let coupled = SparseMatrix::coupled(
        &SparseMatrix::new(2, 2, vec![])?,
        &labor,
        &reproduction,
        &training,
    )?;
    let solution = DeviceOperator::new(&device, &coupled).solve(
        &device,
        &DeviceVector::exact(&device, &[0, 0, 1000, 1000])?,
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
    let reference = DeviceVector::exact(&device, &[40_000, 108_000, 40_000, 28_000])?;
    let normalized = match reference.normalize(&device, 2, 1000)? {
        Diagnostic::Available(value) => value,
        _ => Err(anyhow!("normalization unavailable"))?,
    };
    exact(&normalized, &device, &[1000, 2700, 1000, 700])?;
    Ok(())
}

#[test]
fn lean_residual_certificate_is_for_the_actual_operator() -> Result<()> {
    let device = support::device(64)?;
    let matrix = grain_goods()?;
    let rhs = DeviceVector::exact(&device, &[1000, 1000])?;
    let approximate = DeviceVector::exact(&device, &[1900, 2900])?;
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
        &DeviceVector::exact(&device, &[1000])?,
        &Contraction::new(0, 1, vec![1])?,
        SolveBudget {
            iterations: 1,
            weighted_tolerance: 0,
        },
    )?;
    assert_eq!(
        available(goods.read(&device)?)?.values,
        [Interval {
            lower: 1000,
            upper: 1000
        }]
    );
    let coupled = SparseMatrix::coupled(&zero, &one, &one, &zero)?;
    let closure = DeviceOperator::new(&device, &coupled).solve(
        &device,
        &DeviceVector::exact(&device, &[0, 1000])?,
        &Contraction::new(1, 2, vec![1, 1])?,
        SolveBudget {
            iterations: 10,
            weighted_tolerance: 1,
        },
    )?;
    assert!(matches!(closure.read(&device)?, Diagnostic::Inadmissible));
    let zeros = DeviceVector::exact(&device, &[0, 1000])?;
    match zeros.normalize(&device, 1, 1000)? {
        Diagnostic::Available(value) => exact(&value, &device, &[0, 1000])?,
        _ => Err(anyhow!("zero goods values must be allowed"))?,
    }
    assert!(matches!(
        zeros.normalize(&device, 0, 1000)?,
        Diagnostic::UnresolvedAccuracy
    ));
    Ok(())
}

#[test]
fn analytical_products_preserve_wide_ticks_and_reject_overflow() -> Result<()> {
    let device = support::device(64)?;
    let left = DeviceVector::exact(&device, &[u64::MAX])?;
    let one = DeviceVector::exact(&device, &[1])?;
    exact(&left.dot(&device, &one, 1)?, &device, &[u64::MAX])?;
    let two = DeviceVector::exact(&device, &[2])?;
    assert!(left.dot(&device, &two, 1).is_err());
    let wide = DeviceVector::exact(&device, &[1u64 << 40])?;
    exact(&wide.dot(&device, &wide, 1 << 20)?, &device, &[1u64 << 60])?;
    let thirds = DeviceOperator::new(&device, &dense(1, 1, &[1], &[3])?).apply(&device, &one)?;
    assert_eq!(thirds.read(&device)?, [Interval { lower: 0, upper: 1 }]);
    Ok(())
}

#[test]
fn analytical_admission_and_context_versions_reject_invalid_inputs() -> Result<()> {
    assert!(dense(1, 1, &[1], &[0]).is_err());
    assert!(Contraction::new(1, 1, vec![1]).is_err());
    assert!(Contraction::new(1, 2, vec![0]).is_err());
    let version = ContextVersion {
        techniques: 0,
        units: 0,
        classification: 0,
        access: 0,
        reproduction: 0,
        occupations: 0,
        reference_occupation: 0,
        research_catalogue: 0,
        capability_graph: 0,
        numerical_profile: 1,
    };
    let cache = CachedAccount::new(version, 7);
    assert_eq!(cache.get(version), Some(&7));
    assert_eq!(
        cache.get(ContextVersion {
            access: 1,
            ..version
        }),
        None
    );
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
        &DeviceVector::exact(&device, &[])?,
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
            exact(
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
fn grain_coupled_fixed_point_fixture() -> Result<()> {
    let device = support::device(64)?;
    let matrix = SparseMatrix::coupled(
        &grain_goods()?,
        &grain_labor()?,
        &grain_reproduction()?,
        &grain_training()?,
    )?;
    let step = DeviceOperator::new(&device, &matrix).step(
        &device,
        &DeviceVector::exact(&device, &[0, 0, 1000])?,
        &DeviceVector::exact(&device, &[8000, 12000, 4000])?,
    )?;
    exact(&step, &device, &[8000, 12000, 4000])?;
    Ok(())
}

#[test]
fn account_api_keeps_optional_closure_and_research_separate() -> Result<()> {
    let device = support::device(64)?;
    let model = AccountModel::new(grain_goods()?, grain_labor()?)?
        .with_reproduction(grain_reproduction()?, grain_training()?)?
        .with_research(dense(2, 1, &[1, 1], &[2, 1])?, dense(1, 1, &[2], &[1])?)?;
    let account = model.upload(&device)?;
    let goods = DeviceVector::exact(&device, &[2000, 3000])?;
    let weights = DeviceVector::exact(&device, &[1000])?;
    let research = match account.research_values(&device, &goods, &weights)? {
        Diagnostic::Available(values) => values,
        _ => Err(anyhow!("research account unavailable"))?,
    };
    exact(&research, &device, &[6000])?;
    let y = DeviceVector::exact(&device, &[0, 2000])?;
    let z = DeviceVector::exact(&device, &[1000])?;
    exact(
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
    let requirement = DeviceVector::exact(&device, &[8000])?;
    let report = comparator(&device, context.clone(), 12_000, &requirement, 1000)?;
    assert_eq!(report.context, context);
    assert_eq!(
        report.surplus.lower,
        SignedTicks {
            sign: Sign::NonNegative,
            magnitude: 4000
        }
    );
    assert_eq!(
        report.rate,
        Diagnostic::Available(SignedInterval {
            lower: SignedTicks {
                sign: Sign::NonNegative,
                magnitude: 500
            },
            upper: SignedTicks {
                sign: Sign::NonNegative,
                magnitude: 500
            },
        })
    );
    let deficit = comparator(&device, context.clone(), 4000, &requirement, 1000)?;
    assert_eq!(
        deficit.surplus.lower,
        SignedTicks {
            sign: Sign::Negative,
            magnitude: 4000
        }
    );
    assert_eq!(
        deficit.rate,
        Diagnostic::Available(SignedInterval {
            lower: SignedTicks {
                sign: Sign::Negative,
                magnitude: 500
            },
            upper: SignedTicks {
                sign: Sign::Negative,
                magnitude: 500
            },
        })
    );
    assert!(matches!(
        comparator(
            &device,
            context.clone(),
            1000,
            &DeviceVector::exact(&device, &[0])?,
            1000
        )?
        .rate,
        Diagnostic::Inapplicable
    ));
    Ok(())
}

#[test]
fn retained_boundary_units_and_inventory_accounts_execute_on_device() -> Result<()> {
    let device = support::device(64)?;
    let matrix = |numerator, denominator| {
        DeviceMatrix::new(&device, &dense(1, 1, &[numerator], &[denominator])?, 36_000)
    };
    let retained_goods = matrix(1, 4)?;
    let retained_from_outside = matrix(1, 2)?;
    let outside_from_retained = matrix(1, 3)?;
    let outside_inverse = matrix(4, 3)?;
    let retained_labor = matrix(1, 1)?;
    let outside_labor = matrix(2, 1)?;
    let retained_reproduction = matrix(1, 8)?;
    let outside_reproduction = matrix(1, 6)?;
    let training = matrix(1, 4)?;
    let reduced = BoundaryReduction::new(
        &device,
        BoundaryBlocks {
            retained_goods: &retained_goods,
            retained_from_outside: &retained_from_outside,
            outside_from_retained: &outside_from_retained,
            outside_inverse: &outside_inverse,
            retained_labor: &retained_labor,
            outside_labor: &outside_labor,
            retained_reproduction: &retained_reproduction,
            outside_reproduction: &outside_reproduction,
            training: &training,
        },
    )?;
    assert_eq!(
        reduced.goods.read(&device)?,
        [Interval {
            lower: 17000,
            upper: 17000
        }]
    );
    assert_eq!(
        reduced.labor.read(&device)?,
        [Interval {
            lower: 68000,
            upper: 68000
        }]
    );
    assert_eq!(
        reduced.reproduction.read(&device)?,
        [Interval {
            lower: 8500,
            upper: 8500
        }]
    );
    assert_eq!(
        reduced.training.read(&device)?,
        [Interval {
            lower: 25000,
            upper: 25000
        }]
    );
    exact(
        &BoundaryReduction::effective_demand(
            &device,
            &DeviceVector::exact(&device, &[36_000])?,
            &DeviceVector::exact(&device, &[108_000])?,
            &outside_inverse,
            &retained_from_outside,
        )?,
        &device,
        &[108_000],
    )?;
    let g = DeviceMatrix::new(&device, &dense(1, 2, &[2, 3], &[1, 1])?, 1000)?;
    let transformed = g.scale_units(
        &device,
        &[UnitFactor::new(1, 1)?],
        &[UnitFactor::new(2, 1)?, UnitFactor::new(1, 2)?],
    )?;
    assert_eq!(
        transformed.read(&device)?,
        [
            Interval {
                lower: 4000,
                upper: 4000
            },
            Interval {
                lower: 1500,
                upper: 1500
            }
        ]
    );
    exact(
        &transformed.apply(&device, &DeviceVector::exact(&device, &[1000, 4000])?)?,
        &device,
        &[10000],
    )?;
    exact(
        &g.apply(&device, &DeviceVector::exact(&device, &[2000, 2000])?)?,
        &device,
        &[10000],
    )?;
    let opening = DeviceVector::exact(&device, &[3000, 0])?;
    let closing = DeviceVector::exact(&device, &[2000, 0])?;
    let production = DeviceVector::exact(&device, &[4000, 1000])?;
    let final_demand = DeviceVector::exact(&device, &[1000, 1000])?;
    let research_inputs = DeviceVector::exact(&device, &[1000, 0])?;
    let research_hours = DeviceVector::exact(&device, &[2000])?;
    let inventory = InventoryAccount::new(
        &device,
        &DeviceOperator::new(&device, &grain_goods()?),
        &DeviceOperator::new(&device, &grain_labor()?),
        &g,
        InventoryFlows {
            opening: &opening,
            closing: &closing,
            production: &production,
            final_demand: &final_demand,
            research_inputs: &research_inputs,
            direct_research_hours: &research_hours,
        },
    )?;
    assert!(
        inventory
            .physical_residual
            .read(&device)?
            .iter()
            .all(|value| value.lower.magnitude == 0 && value.upper.magnitude == 0)
    );
    exact(&inventory.actual_hours, &device, &[7000])?;
    assert_eq!(
        inventory.reconciled_hours.read(&device)?[0].lower,
        SignedTicks {
            sign: Sign::NonNegative,
            magnitude: 7000
        }
    );
    Ok(())
}

#[test]
fn certified_inverse_and_lifetime_conventions_are_preserved() -> Result<()> {
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
    assert_eq!(
        per_hour.read(&device)?,
        [
            Interval {
                lower: 3000,
                upper: 3000
            },
            Interval {
                lower: 4000,
                upper: 4000
            }
        ]
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
fn comparator_distinguishes_unresolved_denominator_from_zero_requirement() -> Result<()> {
    let device = support::device(64)?;
    let requirement = DeviceOperator::new(&device, &dense(1, 1, &[1], &[3])?)
        .apply(&device, &DeviceVector::exact(&device, &[1])?)?;
    let context = ComparatorContext {
        population: "workers".into(),
        classification: "one".into(),
        period: "day".into(),
        convention: "external weights".into(),
    };
    let report = comparator(&device, context, 10, &requirement, 1000)?;
    assert!(matches!(report.rate, Diagnostic::UnresolvedAccuracy));
    Ok(())
}

#[test]
fn wide_interval_products_match_independent_u128_fixtures() -> Result<()> {
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
            _ => u64::MAX,
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
        let actual = DeviceVector::exact(&device, &[left])?
            .dot(&device, &DeviceVector::exact(&device, &[right])?, divisor)?
            .read(&device)?[0];
        let wide = u128::from(left) * u128::from(right);
        assert_eq!(actual.lower, (wide / u128::from(divisor)) as u64);
        assert_eq!(actual.upper, wide.div_ceil(u128::from(divisor)) as u64);
    }
    Ok(())
}

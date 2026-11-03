#![cfg(feature = "compute")]
use crate::support;

use anyhow::Result;
use bad_econ::simulation::{Device, analytics::*};

fn sparse(rows: usize, columns: usize, values: &[u32], denominator: u32) -> Result<SparseMatrix> {
    SparseMatrix::new(
        rows,
        columns,
        values
            .iter()
            .enumerate()
            .map(|(index, &numerator)| Coefficient {
                row: index / columns,
                column: index % columns,
                numerator,
                denominator,
            })
            .collect(),
    )
}

fn matrix(
    device: &Device,
    rows: usize,
    columns: usize,
    values: &[u32],
    denominator: u32,
    scale: u32,
) -> Result<DeviceMatrix> {
    DeviceMatrix::new(device, &sparse(rows, columns, values, denominator)?, scale)
}

fn exact(vector: &DeviceVector, device: &Device, values: &[u64]) -> Result<()> {
    assert_eq!(
        vector.read(device)?,
        values
            .iter()
            .map(|&value| Interval {
                lower: value,
                upper: value
            })
            .collect::<Vec<_>>()
    );
    Ok(())
}

fn signed_exact(vector: &DeviceSignedVector, device: &Device, values: &[i64]) -> Result<()> {
    let decode = |value: SignedTicks| match value.sign {
        Sign::NonNegative => i128::from(value.magnitude),
        Sign::Negative => -i128::from(value.magnitude),
    };
    let actual = vector.read(device)?;
    assert_eq!(actual.len(), values.len());
    for (actual, &value) in actual.iter().zip(values) {
        assert_eq!(decode(actual.lower), i128::from(value));
        assert_eq!(decode(actual.upper), i128::from(value));
    }
    Ok(())
}

#[test]
fn lean_inventory_horizon_telescopes_research_and_signed_stock_changes() -> Result<()> {
    let device = support::device(64)?;
    let context = ContextVersion::default();
    let goods = DeviceOperator::new(&device, &sparse(2, 2, &[1, 2, 0, 0], 2)?);
    let direct = DeviceOperator::new(&device, &sparse(1, 2, &[1, 1], 1)?);
    let labor = matrix(&device, 1, 2, &[2, 3], 1, 1000)?;
    let initial = DeviceVector::exact(&device, &[3000, 0])?;
    let middle = DeviceVector::exact(&device, &[2000, 0])?;
    let terminal = DeviceVector::exact(&device, &[1000, 0])?;
    let production_a = DeviceVector::exact(&device, &[4000, 1000])?;
    let production_b = DeviceVector::exact(&device, &[6000, 2000])?;
    let zero_goods = DeviceVector::exact(&device, &[0, 0])?;
    let demand_a = DeviceVector::exact(&device, &[1000, 1000])?;
    let demand_b = DeviceVector::exact(&device, &[1000, 0])?;
    let inputs_a = DeviceVector::exact(&device, &[1000, 0])?;
    let inputs_b = DeviceVector::exact(&device, &[0, 1000])?;
    let hours_a = DeviceVector::exact(&device, &[2000])?;
    let hours_b = DeviceVector::exact(&device, &[1000])?;
    let zero_hours = DeviceVector::exact(&device, &[0])?;
    let horizon = InventoryHorizon::new(
        &device,
        InventoryRegime {
            context,
            goods: &goods,
            direct_labor: &direct,
            labor_requirements: &labor,
        },
        &initial,
        &[
            InventoryStage {
                context,
                closing: &middle,
                production: &production_a,
                final_demand: &demand_a,
                research_inputs: &inputs_a,
                direct_research_hours: &hours_a,
            },
            InventoryStage {
                context,
                closing: &middle,
                production: &production_b,
                final_demand: &demand_a,
                research_inputs: &inputs_b,
                direct_research_hours: &hours_b,
            },
            InventoryStage {
                context,
                closing: &terminal,
                production: &zero_goods,
                final_demand: &demand_b,
                research_inputs: &zero_goods,
                direct_research_hours: &zero_hours,
            },
        ],
    )?;
    assert_eq!(horizon.stages, 3);
    exact(&horizon.actual_hours, &device, &[16_000])?;
    exact(&horizon.final_and_project_hours, &device, &[20_000])?;
    signed_exact(&horizon.accumulated_stock_adjustment, &device, &[-4000])?;
    signed_exact(&horizon.terminal_stock_adjustment, &device, &[-4000])?;
    signed_exact(&horizon.reconciled_hours, &device, &[16_000])?;
    signed_exact(&horizon.reconciliation_residual, &device, &[0])?;
    for residual in &horizon.physical_residuals {
        signed_exact(residual, &device, &[0, 0])?;
    }
    Ok(())
}

#[test]
fn horizon_empty_context_changes_and_broken_balances_are_distinct() -> Result<()> {
    let device = support::device(64)?;
    let context = ContextVersion::default();
    let changed = ContextVersion {
        techniques: 1,
        ..context
    };
    let goods = DeviceOperator::new(&device, &SparseMatrix::new(1, 1, Vec::new())?);
    let direct = DeviceOperator::new(&device, &sparse(1, 1, &[1], 1)?);
    let labor = matrix(&device, 1, 1, &[1], 1, 1000)?;
    let regime = || InventoryRegime {
        context,
        goods: &goods,
        direct_labor: &direct,
        labor_requirements: &labor,
    };
    let opening = DeviceVector::exact(&device, &[1000])?;
    let zero = DeviceVector::exact(&device, &[0])?;
    let empty = InventoryHorizon::new(&device, regime(), &opening, &[])?;
    exact(&empty.actual_hours, &device, &[0])?;
    signed_exact(&empty.reconciled_hours, &device, &[0])?;
    let stage = |context| InventoryStage {
        context,
        closing: &opening,
        production: &zero,
        final_demand: &opening,
        research_inputs: &zero,
        direct_research_hours: &zero,
    };
    assert!(InventoryHorizon::new(&device, regime(), &opening, &[stage(changed)]).is_err());
    let broken = InventoryHorizon::new(&device, regime(), &opening, &[stage(context)])?;
    signed_exact(&broken.physical_residuals[0], &device, &[-1000])?;
    exact(&broken.actual_hours, &device, &[0])?;
    signed_exact(&broken.reconciled_hours, &device, &[1000])?;
    signed_exact(&broken.reconciliation_residual, &device, &[-1000])?;
    assert!(
        InventoryHorizon::new(&device, regime(), &DeviceVector::exact(&device, &[])?, &[]).is_err()
    );
    Ok(())
}

#[test]
fn lean_boundary_demand_retains_outside_direct_labor() -> Result<()> {
    let device = support::device(64)?;
    let scale = 684_000;
    let retained = DeviceVector::exact(&device, &[684_000])?;
    let outside = DeviceVector::exact(&device, &[2_052_000])?;
    let retained_labor = matrix(&device, 1, 1, &[68], 19, scale)?;
    let outside_direct_labor = matrix(&device, 1, 1, &[2], 1, scale)?;
    let outside_inverse = matrix(&device, 1, 1, &[4], 3, scale)?;
    let retained_from_outside = matrix(&device, 1, 1, &[1], 2, scale)?;
    let account = BoundaryDemandAccount::new(
        &device,
        BoundaryDemand {
            retained: &retained,
            outside: &outside,
            retained_labor: &retained_labor,
            outside_direct_labor: &outside_direct_labor,
            outside_inverse: &outside_inverse,
            retained_from_outside: &retained_from_outside,
        },
    )?;
    exact(&account.outside_direct_hours, &device, &[5_472_000])?;
    exact(&account.effective_retained_demand, &device, &[2_052_000])?;
    exact(&account.total_hours, &device, &[12_816_000])?;
    let full_outside_labor = outside_direct_labor
        .add(
            &device,
            &retained_labor.product(&device, &retained_from_outside)?,
        )?
        .product(&device, &outside_inverse)?;
    assert_eq!(
        account.total_hours.read(&device)?,
        retained_labor
            .apply(&device, &retained)?
            .add(&device, &full_outside_labor.apply(&device, &outside)?)?
            .read(&device)?
    );
    let retained_goods = matrix(&device, 1, 1, &[1], 4, scale)?;
    let outside_from_retained = matrix(&device, 1, 1, &[1], 3, scale)?;
    let retained_direct = matrix(&device, 1, 1, &[1], 1, scale)?;
    let retained_reproduction = matrix(&device, 1, 1, &[1], 8, scale)?;
    let outside_reproduction = matrix(&device, 1, 1, &[1], 6, scale)?;
    let training = matrix(&device, 1, 1, &[1], 4, scale)?;
    let reduction = BoundaryReduction::new(
        &device,
        BoundaryBlocks {
            retained_goods: &retained_goods,
            retained_from_outside: &retained_from_outside,
            outside_from_retained: &outside_from_retained,
            outside_inverse: &outside_inverse,
            retained_labor: &retained_direct,
            outside_labor: &outside_direct_labor,
            retained_reproduction: &retained_reproduction,
            outside_reproduction: &outside_reproduction,
            training: &training,
        },
    )?;
    assert_eq!(
        retained_labor.read(&device)?,
        reduction
            .labor
            .add(&device, &retained_labor.product(&device, &reduction.goods)?)?
            .read(&device)?
    );
    assert_eq!(
        retained_labor
            .product(&device, &reduction.reproduction)?
            .add(&device, &reduction.training)?
            .read(&device)?,
        retained_labor
            .product(&device, &retained_reproduction)?
            .add(
                &device,
                &full_outside_labor.product(&device, &outside_reproduction)?
            )?
            .add(&device, &training)?
            .read(&device)?
    );
    Ok(())
}

#[test]
fn lean_augmented_inverse_and_terminal_activities_use_both_blocks() -> Result<()> {
    let device = support::device(64)?;
    let scale = 1000;
    let inverse = matrix(&device, 1, 1, &[2], 1, scale)?;
    let reproduction = matrix(&device, 1, 1, &[1], 8, scale)?;
    let direct = matrix(&device, 1, 1, &[1], 1, scale)?;
    let augmented = AugmentedInverse::new(
        &device,
        AugmentedBlocks {
            goods_inverse: &inverse,
            reproduction: &reproduction,
            direct_labor: &direct,
            reproduction_inverse: &inverse,
        },
    )?;
    for (block, expected) in [
        (&augmented.goods_from_goods, 3000),
        (&augmented.goods_from_labor, 500),
        (&augmented.labor_from_goods, 4000),
        (&augmented.labor_from_labor, 2000),
    ] {
        assert_eq!(
            block.read(&device)?,
            vec![Interval {
                lower: expected,
                upper: expected
            }]
        );
    }
    let rhs = DeviceVector::exact(&device, &[1000])?;
    let result = augmented.apply(&device, &rhs, &rhs)?;
    exact(&result.goods, &device, &[3500])?;
    exact(&result.labor, &device, &[6000])?;
    let a = matrix(&device, 1, 1, &[1], 2, scale)?;
    let t = matrix(&device, 1, 1, &[1], 4, scale)?;
    assert_eq!(
        result.goods.read(&device)?,
        a.apply(&device, &result.goods)?
            .add(&device, &reproduction.apply(&device, &result.labor)?)?
            .add(&device, &rhs)?
            .read(&device)?
    );
    assert_eq!(
        result.labor.read(&device)?,
        direct
            .apply(&device, &result.goods)?
            .add(&device, &t.apply(&device, &result.labor)?)?
            .add(&device, &rhs)?
            .read(&device)?
    );
    let terminal = matrix(&device, 1, 2, &[3, 4], 1, scale)?.add(
        &device,
        &direct
            .product(&device, &inverse)?
            .product(&device, &matrix(&device, 1, 2, &[1, 2], 1, scale)?)?,
    )?;
    assert_eq!(
        terminal.read(&device)?,
        vec![
            Interval {
                lower: 5000,
                upper: 5000
            },
            Interval {
                lower: 8000,
                upper: 8000
            }
        ]
    );
    Ok(())
}

#[test]
fn lean_physical_and_occupational_unit_covariance_preserves_accounts() -> Result<()> {
    let device = support::device(64)?;
    let scale = 1_080_000;
    let g = matrix(&device, 2, 2, &[6, 10, 18, 10], 3, scale)?;
    let h = matrix(&device, 2, 2, &[1, 2, 3, 1], 1, scale)?;
    let a = matrix(&device, 2, 2, &[2, 1, 0, 1], 4, scale)?;
    let n = matrix(&device, 2, 2, &[6, 2, 0, 4], 3, scale)?;
    let f = matrix(&device, 2, 2, &[2, 1, 1, 2], 8, scale)?;
    let t = matrix(&device, 2, 2, &[1, 0, 2, 1], 8, scale)?;
    let physical = [UnitFactor::new(2, 1)?, UnitFactor::new(1, 2)?];
    let physical_inverse = physical.map(UnitFactor::inverse);
    let occupational = [UnitFactor::new(3, 1)?, UnitFactor::new(2, 1)?];
    let occupational_inverse = occupational.map(UnitFactor::inverse);
    let gp = g.scale_units(&device, &occupational_inverse, &physical)?;
    let hp = h.scale_units(&device, &occupational_inverse, &physical)?;
    let ap = a.scale_units(&device, &physical_inverse, &physical)?;
    let np = n.scale_units(&device, &physical_inverse, &physical)?;
    let fp = f.scale_units(&device, &physical_inverse, &occupational)?;
    let tp = t.scale_units(&device, &occupational_inverse, &occupational)?;
    assert_eq!(gp.read(&device)?, hp.product(&device, &np)?.read(&device)?);
    assert_eq!(
        gp.read(&device)?,
        hp.add(&device, &gp.product(&device, &ap)?)?.read(&device)?
    );
    let demand = DeviceVector::exact(&device, &[4 * u64::from(scale), 6 * u64::from(scale)])?;
    let new_demand = demand.scale_units(&device, &physical_inverse)?;
    assert_eq!(
        g.apply(&device, &demand)?.read(&device)?,
        gp.apply(&device, &new_demand)?
            .scale_units(&device, &occupational)?
            .read(&device)?
    );
    let weights = DeviceVector::exact(&device, &[u64::from(scale), 2 * u64::from(scale)])?;
    let new_weights = weights.scale_units(&device, &occupational)?;
    let new_values = gp.transpose(&device)?.apply(&device, &new_weights)?;
    exact(
        &new_values,
        &device,
        &[28 * u64::from(scale), 5 * u64::from(scale)],
    )?;
    let old_m = g.product(&device, &f)?.add(&device, &t)?;
    let new_m = gp.product(&device, &fp)?.add(&device, &tp)?;
    assert_eq!(
        new_m.read(&device)?,
        old_m
            .scale_units(&device, &occupational_inverse, &occupational)?
            .read(&device)?
    );
    let residual = DeviceSignedVector::difference(
        DeviceVector::exact(&device, &[u64::from(scale), 2 * u64::from(scale)])?,
        old_m.transpose(&device)?.apply(&device, &weights)?,
    )?;
    let new_residual = DeviceSignedVector::difference(
        weights.scale_units(&device, &occupational)?,
        new_m.transpose(&device)?.apply(&device, &new_weights)?,
    )?;
    assert_eq!(
        new_residual.read(&device)?,
        residual
            .scale_units(&device, &occupational)?
            .read(&device)?
    );
    match new_weights.normalize(&device, 0, scale)? {
        Diagnostic::Available(rebased) => exact(
            &rebased,
            &device,
            &[u64::from(scale), 4 * u64::from(scale) / 3],
        )?,
        _ => panic!("positive weights must normalize"),
    }
    Ok(())
}

#[test]
fn lean_lifetime_unit_covariance_transforms_subsistence_and_training() -> Result<()> {
    let device = support::device(64)?;
    let scale = 36_000;
    let g = matrix(&device, 2, 2, &[2, 1, 1, 3], 1, scale)?;
    let subsistence = matrix(&device, 2, 2, &[1, 2, 3, 4], 1, scale)?;
    let education = matrix(&device, 2, 2, &[4, 6, 8, 10], 1, scale)?;
    let training = matrix(&device, 2, 2, &[2, 4, 6, 8], 1, scale)?;
    let identity = [UnitFactor::new(1, 1)?; 2];
    let units = [UnitFactor::new(2, 1)?, UnitFactor::new(3, 1)?];
    let inverse_units = units.map(UnitFactor::inverse);
    let f = DeviceMatrix::per_hour_goods(&device, &subsistence, &education, &[4, 12])?;
    let t = training.per_hour_training(&device, &[4, 12])?;
    let fp = DeviceMatrix::per_hour_goods(
        &device,
        &subsistence.scale_units(&device, &identity, &units)?,
        &education,
        &[2, 4],
    )?;
    let tp = training
        .scale_units(&device, &inverse_units, &identity)?
        .per_hour_training(&device, &[2, 4])?;
    assert_eq!(
        fp.read(&device)?,
        f.scale_units(&device, &identity, &units)?.read(&device)?
    );
    assert_eq!(
        tp.read(&device)?,
        t.scale_units(&device, &inverse_units, &units)?
            .read(&device)?
    );
    let m = g.product(&device, &f)?.add(&device, &t)?;
    let mp = g
        .scale_units(&device, &inverse_units, &identity)?
        .product(&device, &fp)?
        .add(&device, &tp)?;
    assert_eq!(
        mp.read(&device)?,
        m.scale_units(&device, &inverse_units, &units)?
            .read(&device)?
    );
    assert!(
        DeviceVector::exact(&device, &[1])?
            .scale_units(&device, &units)
            .is_err()
    );
    Ok(())
}

#[test]
fn lean_conversion_reports_the_positive_residual_scale_with_normalized_weights() -> Result<()> {
    let device = support::device(64)?;
    let beta = DeviceVector::exact(&device, &[2000, 1000])?;
    let kappa = match beta.reciprocal_reference(&device, 0, 1000)? {
        Diagnostic::Available(value) => value,
        _ => panic!("positive reference must admit a reciprocal"),
    };
    exact(&kappa, &device, &[500])?;
    let alpha = match beta.normalize(&device, 0, 1000)? {
        Diagnostic::Available(value) => value,
        _ => panic!("positive reference must normalize"),
    };
    exact(&alpha, &device, &[1000, 500])?;
    let m = DeviceOperator::new(&device, &sparse(2, 2, &[1, 0, 0, 0], 2)?);
    let residual = DeviceSignedVector::difference(
        DeviceVector::exact(&device, &[1000, 500])?,
        m.apply(&device, &alpha)?,
    )?;
    signed_exact(&residual, &device, &[500, 500])?;
    let zero = DeviceVector::exact(&device, &[0])?;
    assert!(matches!(
        zero.reciprocal_reference(&device, 0, 1000)?,
        Diagnostic::Inapplicable
    ));
    let interval = DeviceOperator::new(&device, &sparse(1, 1, &[1], 3)?)
        .apply(&device, &DeviceVector::exact(&device, &[1])?)?;
    assert!(matches!(
        interval.reciprocal_reference(&device, 0, 1000)?,
        Diagnostic::UnresolvedAccuracy
    ));
    assert!(beta.reciprocal_reference(&device, 2, 1000).is_err());
    assert!(beta.reciprocal_reference(&device, 0, 0).is_err());
    Ok(())
}

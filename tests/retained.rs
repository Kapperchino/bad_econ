#![cfg(feature = "compute")]
use super::{estimates, support};

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

fn approximate(vector: &DeviceVector, device: &Device, values: &[u64]) -> Result<()> {
    estimates::assert_close(
        vector.read(device)?,
        values
            .iter()
            .map(|&value| Interval {
                lower: value,
                upper: value,
            })
            .collect::<Vec<_>>(),
    );
    Ok(())
}

fn signed_approximate(vector: &DeviceSignedVector, device: &Device, values: &[i64]) -> Result<()> {
    let decode = |value: SignedTicks| match value.sign {
        Sign::NonNegative => i128::from(value.magnitude),
        Sign::Negative => -i128::from(value.magnitude),
    };
    let actual = vector.read(device)?;
    assert_eq!(actual.len(), values.len());
    for (actual, &value) in actual.iter().zip(values) {
        estimates::assert_close(decode(actual.lower), i128::from(value));
        estimates::assert_close(decode(actual.upper), i128::from(value));
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
    let initial = DeviceVector::from_ticks(&device, &[3000, 0])?;
    let middle = DeviceVector::from_ticks(&device, &[2000, 0])?;
    let terminal = DeviceVector::from_ticks(&device, &[1000, 0])?;
    let production_a = DeviceVector::from_ticks(&device, &[4000, 1000])?;
    let production_b = DeviceVector::from_ticks(&device, &[6000, 2000])?;
    let zero_goods = DeviceVector::from_ticks(&device, &[0, 0])?;
    let demand_a = DeviceVector::from_ticks(&device, &[1000, 1000])?;
    let demand_b = DeviceVector::from_ticks(&device, &[1000, 0])?;
    let inputs_a = DeviceVector::from_ticks(&device, &[1000, 0])?;
    let inputs_b = DeviceVector::from_ticks(&device, &[0, 1000])?;
    let hours_a = DeviceVector::from_ticks(&device, &[2000])?;
    let hours_b = DeviceVector::from_ticks(&device, &[1000])?;
    let zero_hours = DeviceVector::from_ticks(&device, &[0])?;
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
    approximate(&horizon.actual_hours, &device, &[16_000])?;
    approximate(&horizon.final_and_project_hours, &device, &[20_000])?;
    signed_approximate(&horizon.accumulated_stock_adjustment, &device, &[-4000])?;
    signed_approximate(&horizon.terminal_stock_adjustment, &device, &[-4000])?;
    signed_approximate(&horizon.reconciled_hours, &device, &[16_000])?;
    signed_approximate(&horizon.reconciliation_residual, &device, &[0])?;
    for residual in &horizon.physical_residuals {
        signed_approximate(residual, &device, &[0, 0])?;
    }
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
        estimates::assert_close(
            block.read(&device)?,
            vec![Interval {
                lower: expected,
                upper: expected,
            }],
        );
    }
    let rhs = DeviceVector::from_ticks(&device, &[1000])?;
    let result = augmented.apply(&device, &rhs, &rhs)?;
    approximate(&result.goods, &device, &[3500])?;
    approximate(&result.labor, &device, &[6000])?;
    let a = matrix(&device, 1, 1, &[1], 2, scale)?;
    let t = matrix(&device, 1, 1, &[1], 4, scale)?;
    estimates::assert_close(
        result.goods.read(&device)?,
        a.apply(&device, &result.goods)?
            .add(&device, &reproduction.apply(&device, &result.labor)?)?
            .add(&device, &rhs)?
            .read(&device)?,
    );
    estimates::assert_close(
        result.labor.read(&device)?,
        direct
            .apply(&device, &result.goods)?
            .add(&device, &t.apply(&device, &result.labor)?)?
            .add(&device, &rhs)?
            .read(&device)?,
    );
    let terminal = matrix(&device, 1, 2, &[3, 4], 1, scale)?.add(
        &device,
        &direct
            .product(&device, &inverse)?
            .product(&device, &matrix(&device, 1, 2, &[1, 2], 1, scale)?)?,
    )?;
    estimates::assert_close(
        terminal.read(&device)?,
        vec![
            Interval {
                lower: 5000,
                upper: 5000,
            },
            Interval {
                lower: 8000,
                upper: 8000,
            },
        ],
    );
    Ok(())
}

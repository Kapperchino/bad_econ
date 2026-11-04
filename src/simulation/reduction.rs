use super::{
    Device,
    flags::{FaultFlags, KernelFaults},
    words::{Ticks, ticks},
};
use anyhow::{Context, Result};
use burn_wgpu::CubeTensor;
use cubecl::{prelude::*, std::tensor::TensorHandle};
use cubek::reduce::{
    ReduceDtypes, ReduceStrategy,
    components::instructions::ReduceOperationConfig,
    launch::{RoutineStrategy, VectorizationStrategy},
    routines::{BlueprintStrategy, cube::CubeStrategy, unit::UnitStrategy},
};

pub(super) fn reduce(
    device: &Device,
    input: &TensorHandle,
    axis: usize,
    operation: ReduceOperationConfig,
    dtype: ElemType,
) -> Result<TensorHandle> {
    let mut shape = input.shape().to_vec();
    let length = *shape
        .get(axis)
        .context("reduction axis exceeds tensor rank")?;
    shape[axis] = 1;
    let elements: usize = shape.iter().product();
    let output = match (length, elements) {
        (0, _) | (_, 0) => {
            let identity = match (dtype, operation) {
                (ElemType::Float(_), ReduceOperationConfig::Min) => {
                    u64::from(f32::INFINITY.to_bits())
                }
                (ElemType::Float(_), ReduceOperationConfig::Max) => {
                    u64::from(f32::NEG_INFINITY.to_bits())
                }
                (ElemType::Float(_), ReduceOperationConfig::All | ReduceOperationConfig::Prod) => {
                    u64::from(1.0f32.to_bits())
                }
                (_, ReduceOperationConfig::Min) => u64::MAX,
                (_, ReduceOperationConfig::All | ReduceOperationConfig::Prod) => 1,
                _ => 0,
            };
            let bytes = identity.to_le_bytes();
            let values = bytes[..dtype.size()].repeat(elements.max(1));
            TensorHandle::new_contiguous(shape, device.client.create_from_slice(&values), dtype)
        }
        _ => {
            let output = TensorHandle::new_contiguous(
                shape,
                device.client.empty(elements * dtype.size()),
                dtype,
            );
            cubek::reduce::reduce(
                &device.client,
                input.clone().binding(),
                output.clone().binding(),
                axis,
                ReduceStrategy {
                    routine: match length {
                        0..32 => RoutineStrategy::Unit(BlueprintStrategy::Inferred(UnitStrategy)),
                        _ => RoutineStrategy::Cube(BlueprintStrategy::Inferred(CubeStrategy {
                            use_planes: false,
                        })),
                    },
                    vectorization: VectorizationStrategy {
                        parallel_output_vectorization: true,
                    },
                    autotune_level: Default::default(),
                },
                operation,
                ReduceDtypes {
                    input: input.dtype,
                    output: dtype,
                    accumulation: dtype,
                },
            )?;
            output
        }
    };
    Ok(output)
}

pub(super) fn sum_ticks(
    device: &Device,
    input: &TensorHandle,
    axis: usize,
    control: &CubeTensor,
) -> Result<TensorHandle> {
    let mut shape = input.shape().to_vec();
    let mut strides = input
        .strides()
        .iter()
        .map(|stride| stride * 2)
        .collect::<Vec<_>>();
    shape.push(2);
    strides.push(1);
    let words = TensorHandle::new(
        input.handle.clone(),
        shape,
        strides,
        u32::elem_type_native(),
    );
    let sums = reduce(
        device,
        &words,
        axis,
        ReduceOperationConfig::Sum,
        u64::elem_type_native(),
    )?;
    let mut shape = input.shape().to_vec();
    shape[axis] = 1;
    let elements = shape.iter().product::<usize>();
    let output = TensorHandle::new_contiguous(
        shape,
        device.client.empty(elements.max(1) * size_of::<u64>()),
        u64::elem_type_native(),
    );
    if elements > 0 {
        finish_sum::launch(
            &device.client,
            device.count(elements),
            CubeDim::new_1d(device.workgroup),
            sums.into_arg().into_buffer_arg(),
            output.clone().into_arg().into_buffer_arg(),
            control.clone().into_buffer_arg(),
        );
    }
    Ok(output)
}

pub(super) fn summary(
    device: &Device,
    input: &CubeTensor,
    output: &CubeTensor,
    control: &CubeTensor,
    groups: usize,
) -> Result<()> {
    use super::layout::SummaryLayout as S;
    summary_tiles::launch(
        &device.client,
        device.count(S::METRICS * device.workgroup as usize),
        CubeDim::new_1d(device.workgroup),
        input.clone().into_buffer_arg(),
        output.clone().into_buffer_arg(),
        control.clone().into_buffer_arg(),
        groups as u32,
    );
    Ok(())
}

#[cube(launch)]
fn summary_tiles(input: &[Ticks], output: &mut [Ticks], control: &[Atomic<u32>], groups: u32) {
    use crate::simulation::layout::SummaryLayout as S;
    let metric = CUBE_POS as u32;
    if metric < S::METRICS as u32 {
        let minimum = metric == S::MINIMUM_WAGE
            || metric == S::MINIMUM_PRICE
            || (metric >= S::WAGE_MINIMA
                && metric < S::WAGE_MINIMA + crate::domain::Good::COUNT as u32)
            || (metric >= S::PRICE_MINIMA
                && metric < S::PRICE_MINIMA + crate::domain::Good::COUNT as u32);
        let maximum = metric == S::MAXIMUM_WAGE
            || metric == S::MAXIMUM_PRICE
            || (metric >= S::WAGE_MAXIMA
                && metric < S::WAGE_MAXIMA + crate::domain::Good::COUNT as u32)
            || (metric >= S::PRICE_MAXIMA
                && metric < S::PRICE_MAXIMA + crate::domain::Good::COUNT as u32);
        let mut total = select(minimum, u64::MAX, 0u64);
        let mut group = UNIT_POS;
        while group < groups {
            let words = input[(group * S::METRICS as u32 + metric) as usize];
            let value = u64::cast_from(words.extract(0usize))
                | (u64::cast_from(words.extract(1usize)) << 32);
            let sum = total + value;
            if !minimum && !maximum && sum < total {
                KernelFaults::new(comptime!(FaultFlags::ARITHMETIC_OVERFLOW)).record(control);
            }
            total = select(
                minimum,
                min(total, value),
                select(maximum, max(total, value), sum),
            );
            group += CUBE_DIM;
        }
        let mut shared = Shared::<[u64]>::new_slice(256usize);
        shared[UNIT_POS as usize] = total;
        sync_cube();
        let mut stride = CUBE_DIM / 2;
        while stride > 0 {
            if UNIT_POS < stride {
                let a = shared[UNIT_POS as usize];
                let b = shared[(UNIT_POS + stride) as usize];
                let sum = a + b;
                if !minimum && !maximum && sum < a {
                    KernelFaults::new(comptime!(FaultFlags::ARITHMETIC_OVERFLOW)).record(control);
                }
                shared[UNIT_POS as usize] =
                    select(minimum, min(a, b), select(maximum, max(a, b), sum));
            }
            sync_cube();
            stride /= 2;
        }
        if UNIT_POS == 0 {
            output[metric as usize] = ticks(
                u32::cast_from(shared[0usize]),
                u32::cast_from(shared[0usize] >> 32),
            );
        }
    }
}

#[cfg(test)]
pub(super) fn original_summary(
    device: &Device,
    input: &CubeTensor,
    output: &CubeTensor,
    control: &CubeTensor,
    groups: usize,
) -> Result<()> {
    use super::layout::SummaryLayout as S;
    let input = TensorHandle::new_contiguous(
        [groups, S::METRICS],
        input.handle.clone(),
        u64::elem_type_native(),
    );
    let output =
        TensorHandle::new_contiguous([S::METRICS], output.handle.clone(), u64::elem_type_native());
    let mut start = 0;
    while start < S::METRICS {
        let operation = summary_operation(start as u32);
        let width = (start..S::METRICS)
            .take_while(|&metric| summary_operation(metric as u32) == operation)
            .count();
        let values = TensorHandle::new_contiguous(
            [groups, width],
            device
                .client
                .empty((groups * width).max(1) * size_of::<u64>()),
            u64::elem_type_native(),
        );
        columns::launch(
            &device.client,
            device.count(groups * width),
            CubeDim::new_1d(device.workgroup),
            input.clone().into_arg().into_buffer_arg(),
            values.clone().into_arg().into_buffer_arg(),
            S::METRICS as u32,
            width as u32,
            start as u32,
        );
        let total = match operation {
            ReduceOperationConfig::Sum => sum_ticks(device, &values, 0, control)?,
            _ => reduce(device, &values, 0, operation, u64::elem_type_native())?,
        };
        store_columns::launch(
            &device.client,
            device.count(width),
            CubeDim::new_1d(device.workgroup),
            total.into_arg().into_buffer_arg(),
            output.clone().into_arg().into_buffer_arg(),
            start as u32,
        );
        start += width;
    }
    Ok(())
}

#[cfg(test)]
#[cube(launch)]
fn columns(
    input: &[u64],
    output: &mut [u64],
    #[comptime] input_columns: u32,
    #[comptime] columns: u32,
    start: u32,
) {
    let index = ABSOLUTE_POS;
    if index < output.len() {
        output[index] = input[index / columns as usize * input_columns as usize
            + start as usize
            + index % columns as usize];
    }
}

#[cfg(test)]
#[cube(launch)]
fn store_columns(input: &[u64], output: &mut [u64], start: u32) {
    let index = ABSOLUTE_POS;
    if index < input.len() {
        output[start as usize + index] = input[index];
    }
}

#[cfg(test)]
fn summary_operation(metric: u32) -> ReduceOperationConfig {
    use super::layout::SummaryLayout as S;
    use crate::domain::Good;
    match metric {
        S::MINIMUM_WAGE | S::MINIMUM_PRICE => ReduceOperationConfig::Min,
        S::MAXIMUM_WAGE | S::MAXIMUM_PRICE => ReduceOperationConfig::Max,
        value
            if (S::WAGE_MINIMA..S::WAGE_MINIMA + Good::COUNT as u32).contains(&value)
                || (S::PRICE_MINIMA..S::PRICE_MINIMA + Good::COUNT as u32).contains(&value) =>
        {
            ReduceOperationConfig::Min
        }
        value
            if (S::WAGE_MAXIMA..S::WAGE_MAXIMA + Good::COUNT as u32).contains(&value)
                || (S::PRICE_MAXIMA..S::PRICE_MAXIMA + Good::COUNT as u32).contains(&value) =>
        {
            ReduceOperationConfig::Max
        }
        _ => ReduceOperationConfig::Sum,
    }
}

#[cube(launch)]
fn finish_sum(sums: &[u64], output: &mut [u64], control: &[Atomic<u32>]) {
    let index = ABSOLUTE_POS;
    if index < output.len() {
        let low = sums[index * 2];
        let high = sums[index * 2 + 1] + (low >> 32);
        output[index] = (low & 0xffff_ffffu64) | (high << 32);
        if high > 0xffff_ffffu64 {
            KernelFaults::new(comptime!(FaultFlags::ARITHMETIC_OVERFLOW)).record(control);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_integer_reductions_on_gpu() -> Result<()> {
        let device = Device::headless(64)?;
        let values = [u64::from(u32::MAX), 1, 1 << 40, 7];
        let input = TensorHandle::new_contiguous(
            [2, 2],
            device
                .client
                .create_from_slice(bytemuck::cast_slice(&values)),
            u64::elem_type_native(),
        );
        for operation in [ReduceOperationConfig::Sum, ReduceOperationConfig::Max] {
            let expected = match operation {
                ReduceOperationConfig::Sum => [values[0] + values[2], 8],
                _ => [values[2], 7],
            };
            let output = reduce(&device, &input, 0, operation, u64::elem_type_native())?;
            let bytes = device.client.read_one(output.handle)?;
            assert_eq!(bytemuck::cast_slice::<u8, u64>(&bytes), expected);
        }
        Ok(())
    }

    #[test]
    fn summary_reductions_preserve_wide_totals_extrema_and_overflow_on_gpu() -> Result<()> {
        use super::super::flags::CONTROL_WORDS;
        use super::super::layout::SummaryLayout as S;
        let device = Device::headless(64)?;
        let groups = 257;
        let values = (0..groups * S::METRICS)
            .map(|index| (1u64 << 40) + index as u64)
            .collect::<Vec<_>>();
        let input = device.tensor(bytemuck::cast_slice(&values));
        let output = device.empty(S::WORDS);
        let control = device.tensor(&[0; CONTROL_WORDS]);
        summary(&device, &input, &output, &control, groups)?;
        let actual = device.read(&output)?;
        original_summary(&device, &input, &output, &control, groups)?;
        assert_eq!(actual, device.read(&output)?);
        for metric in 0..S::METRICS {
            let column = (0..groups).map(|group| values[group * S::METRICS + metric]);
            let expected = match summary_operation(metric as u32) {
                ReduceOperationConfig::Min => column.min().unwrap(),
                ReduceOperationConfig::Max => column.max().unwrap(),
                _ => column.sum(),
            };
            assert_eq!(
                u64::from(actual[metric * 2]) | (u64::from(actual[metric * 2 + 1]) << 32),
                expected
            );
        }
        FaultFlags::read(&device.read(&control)?)?.into_result()?;
        let values = [u64::from(u32::MAX), u64::MAX, 1, 1];
        let input = TensorHandle::new_contiguous(
            [2, 2],
            device
                .client
                .create_from_slice(bytemuck::cast_slice(&values)),
            u64::elem_type_native(),
        );
        let output = sum_ticks(&device, &input, 0, &control)?;
        let bytes = device.client.read_one(output.handle)?;
        assert_eq!(bytemuck::cast_slice::<u8, u64>(&bytes), [1 << 32, 0]);
        assert_eq!(
            FaultFlags::read(&device.read(&control)?)?,
            FaultFlags::ARITHMETIC_OVERFLOW
        );
        Ok(())
    }
}

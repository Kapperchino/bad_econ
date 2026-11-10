use crate::simulation::{
    Device,
    flags::{FaultFlags, KernelFaults},
    pass::GpuPass,
    system::{System, gpu_passes},
    words::{Ticks, ticks},
};
use anyhow::{Context, Result};
use burn_wgpu::CubeTensor;
use cubecl::{
    features::Plane,
    ir::{FloatKind, UIntKind},
    prelude::*,
    std::tensor::TensorHandle,
};
use cubek::reduce::{
    ReduceDtypes, ReduceStrategy,
    components::instructions::ReduceOperationConfig,
    launch::{RoutineStrategy, VectorizationStrategy},
    routines::{BlueprintStrategy, cube::CubeStrategy, plane::PlaneStrategy, unit::UnitStrategy},
};

enum ReductionLayout {
    Contiguous,
    Strided,
}

enum ReductionPlanes {
    Available,
    Unavailable,
}

impl ReductionPlanes {
    fn new(device: &Device, dtypes: ReduceDtypes) -> Self {
        let supported_types = [dtypes.input, dtypes.output, dtypes.accumulation]
            .iter()
            .all(|dtype| {
                matches!(
                    dtype,
                    ElemType::Float(FloatKind::F32) | ElemType::UInt(UIntKind::U32)
                )
            });
        match supported_types
            && device
                .client
                .properties()
                .features
                .plane
                .contains(Plane::Ops)
        {
            true => Self::Available,
            false => Self::Unavailable,
        }
    }
}

struct ReductionShape {
    output: Vec<usize>,
    length: usize,
    elements: usize,
    layout: ReductionLayout,
}

impl ReductionShape {
    fn new(input: &TensorHandle, axis: usize) -> Result<Self> {
        let mut output = input.shape().to_vec();
        let length = *output
            .get(axis)
            .context("reduction axis exceeds tensor rank")?;
        output[axis] = 1;
        let elements = output.iter().product();
        let layout = match input.strides()[axis] {
            1 => ReductionLayout::Contiguous,
            _ => ReductionLayout::Strided,
        };
        Ok(Self {
            output,
            length,
            elements,
            layout,
        })
    }

    fn strategy(&self, device: &Device, dtypes: ReduceDtypes) -> ReduceStrategy {
        let plane_size = device.client.properties().hardware.plane_size_max.max(1) as usize;
        let planes = ReductionPlanes::new(device, dtypes);
        let routine = match (self.length, &self.layout, planes) {
            (length, _, _) if length < plane_size => {
                RoutineStrategy::Unit(BlueprintStrategy::Inferred(UnitStrategy))
            }
            (_, ReductionLayout::Contiguous, ReductionPlanes::Available)
                if self.elements >= plane_size =>
            {
                RoutineStrategy::Plane(BlueprintStrategy::Inferred(PlaneStrategy {
                    independent: false,
                }))
            }
            (_, _, ReductionPlanes::Available) => {
                RoutineStrategy::Cube(BlueprintStrategy::Inferred(CubeStrategy {
                    use_planes: true,
                }))
            }
            (_, _, ReductionPlanes::Unavailable) => {
                RoutineStrategy::Cube(BlueprintStrategy::Inferred(CubeStrategy {
                    use_planes: false,
                }))
            }
        };
        ReduceStrategy {
            routine,
            vectorization: VectorizationStrategy {
                parallel_output_vectorization: true,
            },
            autotune_level: Default::default(),
        }
    }
}

pub(in crate::simulation) fn reduce(
    device: &Device,
    input: &TensorHandle,
    axis: usize,
    operation: ReduceOperationConfig,
    dtype: ElemType,
) -> Result<TensorHandle> {
    let shape = ReductionShape::new(input, axis)?;
    let dtypes = ReduceDtypes {
        input: input.dtype,
        output: dtype,
        accumulation: dtype,
    };
    let output = match (shape.length, shape.elements) {
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
            let values = bytes[..dtype.size()].repeat(shape.elements.max(1));
            TensorHandle::new_contiguous(
                shape.output,
                device.client.create_from_slice(&values),
                dtype,
            )
        }
        _ => {
            let strategy = shape.strategy(device, dtypes);
            let output = TensorHandle::new_contiguous(
                shape.output,
                device.client.empty(shape.elements * dtype.size()),
                dtype,
            );
            cubek::reduce::reduce(
                &device.client,
                input.clone().binding(),
                output.clone().binding(),
                axis,
                strategy,
                operation,
                dtypes,
            )?;
            output
        }
    };
    Ok(output)
}

pub(in crate::simulation) fn sum_ticks(
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
        WideSum::launch(
            device,
            device.count(elements),
            CubeDim::new_1d(device.workgroup),
            SumInputLaunch {
                sums: sums.into_arg().into_buffer_arg(),
                output: output.clone().into_arg().into_buffer_arg(),
                control: control.clone().into_buffer_arg(),
            },
        );
    }
    Ok(output)
}

pub(in crate::simulation) fn summary(
    device: &Device,
    input: &CubeTensor,
    output: &CubeTensor,
    control: &CubeTensor,
    groups: &CubeTensor,
) -> Result<()> {
    use crate::simulation::layout::SummaryLayout as S;
    SummaryReduction::launch(
        device,
        device.count(S::METRICS * device.workgroup as usize),
        CubeDim::new_1d(device.workgroup),
        SummaryInputLaunch {
            values: input.clone().into_buffer_arg(),
            output: output.clone().into_buffer_arg(),
            control: control.clone().into_buffer_arg(),
            groups: groups.clone().into_buffer_arg(),
        },
    );
    Ok(())
}

#[derive(CubeType, CubeLaunch)]
struct SummaryInput {
    values: Box<[Ticks]>,
    output: Box<[Ticks]>,
    control: Box<[Atomic<u32>]>,
    groups: Box<[u32]>,
}

struct SummaryReduction;
struct SummarizeTiles;

impl System for SummaryReduction {
    type Input = SummaryInput;
    type Passes = gpu_passes![SummarizeTiles];
}

#[cube]
impl GpuPass<SummaryInput> for SummarizeTiles {
    type Output = ();

    fn execute(mut input: SummaryInput) {
        summary_tiles(
            &input.values,
            &mut input.output,
            &input.control,
            &input.groups,
        );
    }
}

#[cube]
fn summary_tiles(input: &[Ticks], output: &mut [Ticks], control: &[Atomic<u32>], groups: &[u32]) {
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
        while group < groups[metric as usize] {
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
pub(in crate::simulation) fn original_summary(
    device: &Device,
    input: &CubeTensor,
    output: &CubeTensor,
    control: &CubeTensor,
    groups: usize,
) -> Result<()> {
    use crate::simulation::layout::SummaryLayout as S;
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
    use crate::domain::Good;
    use crate::simulation::layout::SummaryLayout as S;
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

#[derive(CubeType, CubeLaunch)]
struct SumInput {
    sums: Box<[u64]>,
    output: Box<[u64]>,
    control: Box<[Atomic<u32>]>,
}

struct WideSum;
struct FinishSum;

impl System for WideSum {
    type Input = SumInput;
    type Passes = gpu_passes![FinishSum];
}

#[cube]
impl GpuPass<SumInput> for FinishSum {
    type Output = ();

    fn execute(mut input: SumInput) {
        finish_sum(&input.sums, &mut input.output, &input.control);
    }
}

#[cube]
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

    #[derive(Clone, Copy)]
    enum InputLayout {
        Contiguous,
        Strided,
    }

    struct FloatReductionCase {
        outputs: usize,
        length: usize,
        layout: InputLayout,
    }

    #[test]
    fn float_reductions_match_reference_across_layouts_and_tails_on_gpu() -> Result<()> {
        let device = crate::simulation::tests::device(64)?;
        let cases = [
            FloatReductionCase {
                outputs: 1,
                length: 17,
                layout: InputLayout::Contiguous,
            },
            FloatReductionCase {
                outputs: 1,
                length: 4097,
                layout: InputLayout::Contiguous,
            },
            FloatReductionCase {
                outputs: 257,
                length: 127,
                layout: InputLayout::Contiguous,
            },
            FloatReductionCase {
                outputs: 64,
                length: 513,
                layout: InputLayout::Contiguous,
            },
            FloatReductionCase {
                outputs: 3,
                length: 129,
                layout: InputLayout::Strided,
            },
            FloatReductionCase {
                outputs: 257,
                length: 33,
                layout: InputLayout::Strided,
            },
        ];
        for case in cases {
            let strides = match case.layout {
                InputLayout::Contiguous => [case.length, 1],
                InputLayout::Strided => [1, case.outputs],
            };
            let values = (0..case.outputs * case.length)
                .map(|index| (index % 97) as f32 / 101.0 - 0.25)
                .collect::<Vec<_>>();
            let input = TensorHandle::new(
                device
                    .client
                    .create_from_slice(bytemuck::cast_slice(&values)),
                [case.outputs, case.length],
                strides,
                f32::elem_type_native(),
            );
            for operation in [ReduceOperationConfig::Sum, ReduceOperationConfig::Max] {
                let output = reduce(&device, &input, 1, operation, f32::elem_type_native())?;
                let bytes = device.client.read_one(output.handle)?;
                let actual = bytemuck::cast_slice::<u8, f32>(&bytes);
                for (row, &actual) in actual.iter().enumerate().take(case.outputs) {
                    let column = (0..case.length)
                        .map(|index| values[row * strides[0] + index * strides[1]] as f64);
                    let expected = match operation {
                        ReduceOperationConfig::Max => column.fold(f64::NEG_INFINITY, f64::max),
                        _ => column.sum::<f64>(),
                    };
                    let tolerance = match operation {
                        ReduceOperationConfig::Max => 0.0,
                        _ => expected.abs().max(1.0) * 2e-5,
                    };
                    assert!(
                        (f64::from(actual) - expected).abs() <= tolerance,
                        "{operation:?}, {} outputs, length {}, row {row}: {actual} versus {expected}",
                        case.outputs,
                        case.length
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn widened_integer_reductions_preserve_carries_and_overflow_on_gpu() -> Result<()> {
        use crate::simulation::flags::CONTROL_WORDS;
        let device = crate::simulation::tests::device(64)?;
        let length = 1025;
        let columns = 7;
        let values = (0..length * columns)
            .map(|index| (1u64 << 40) + u64::from(u32::MAX) - index as u64)
            .collect::<Vec<_>>();
        let input = TensorHandle::new_contiguous(
            [length, columns],
            device
                .client
                .create_from_slice(bytemuck::cast_slice(&values)),
            u64::elem_type_native(),
        );
        let control = device.tensor(&[0; CONTROL_WORDS]);
        let sums = sum_ticks(&device, &input, 0, &control)?;
        let bytes = device.client.read_one(sums.handle)?;
        let actual = bytemuck::cast_slice::<u8, u64>(&bytes);
        for (column, &actual) in actual.iter().enumerate().take(columns) {
            let expected = (0..length)
                .map(|row| values[row * columns + column])
                .sum::<u64>();
            assert_eq!(actual, expected);
        }
        FaultFlags::read(&device.read(&control)?)?.into_result()?;
        for operation in [ReduceOperationConfig::Min, ReduceOperationConfig::Max] {
            let extrema = reduce(&device, &input, 0, operation, u64::elem_type_native())?;
            let bytes = device.client.read_one(extrema.handle)?;
            let actual = bytemuck::cast_slice::<u8, u64>(&bytes);
            for (column, &actual) in actual.iter().enumerate().take(columns) {
                let source = (0..length).map(|row| values[row * columns + column]);
                let expected = match operation {
                    ReduceOperationConfig::Min => source.min().unwrap(),
                    _ => source.max().unwrap(),
                };
                assert_eq!(actual, expected);
            }
        }
        let mut overflowing = vec![0u64; length * columns];
        overflowing[0] = u64::MAX;
        overflowing[columns] = 1;
        let input = TensorHandle::new_contiguous(
            [length, columns],
            device
                .client
                .create_from_slice(bytemuck::cast_slice(&overflowing)),
            u64::elem_type_native(),
        );
        let sums = sum_ticks(&device, &input, 0, &control)?;
        let bytes = device.client.read_one(sums.handle)?;
        assert_eq!(bytemuck::cast_slice::<u8, u64>(&bytes)[0], 0);
        assert_eq!(
            FaultFlags::read(&device.read(&control)?)?,
            FaultFlags::ARITHMETIC_OVERFLOW
        );
        Ok(())
    }

    #[test]
    fn strided_all_reductions_observe_tail_flags_on_gpu() -> Result<()> {
        let device = crate::simulation::tests::device(64)?;
        for length in [17, 129, 4097] {
            let mut values = vec![1u32; length * 2];
            values[(length - 1) * 2] = 0;
            let input = TensorHandle::new_contiguous(
                [length, 2],
                device
                    .client
                    .create_from_slice(bytemuck::cast_slice(&values)),
                u32::elem_type_native(),
            );
            let output = reduce(
                &device,
                &input,
                0,
                ReduceOperationConfig::All,
                u32::elem_type_native(),
            )?;
            let bytes = device.client.read_one(output.handle)?;
            assert_eq!(bytemuck::cast_slice::<u8, u32>(&bytes), [0, 1]);
        }
        Ok(())
    }

    #[test]
    fn reductions_preserve_empty_identities_and_zero_output_shapes_on_gpu() -> Result<()> {
        let device = crate::simulation::tests::device(64)?;
        let input =
            TensorHandle::new_contiguous([0, 3], device.client.empty(4), f32::elem_type_native());
        for operation in [
            ReduceOperationConfig::Sum,
            ReduceOperationConfig::Prod,
            ReduceOperationConfig::All,
            ReduceOperationConfig::Min,
            ReduceOperationConfig::Max,
        ] {
            let output = reduce(&device, &input, 0, operation, f32::elem_type_native())?;
            assert_eq!(output.shape().to_vec(), [1, 3]);
            let bytes = device.client.read_one(output.handle)?;
            let expected = match operation {
                ReduceOperationConfig::Min => f32::INFINITY,
                ReduceOperationConfig::Max => f32::NEG_INFINITY,
                ReduceOperationConfig::All | ReduceOperationConfig::Prod => 1.0,
                _ => 0.0,
            };
            assert_eq!(bytemuck::cast_slice::<u8, f32>(&bytes), [expected; 3]);
        }
        let input =
            TensorHandle::new_contiguous([3, 0], device.client.empty(4), u32::elem_type_native());
        let output = reduce(
            &device,
            &input,
            0,
            ReduceOperationConfig::Sum,
            u64::elem_type_native(),
        )?;
        assert_eq!(output.shape().to_vec(), [1, 0]);
        Ok(())
    }

    #[test]
    fn summary_reductions_preserve_wide_totals_extrema_and_overflow_on_gpu() -> Result<()> {
        use crate::simulation::flags::CONTROL_WORDS;
        use crate::simulation::layout::SummaryLayout as S;
        let device = crate::simulation::tests::device(64)?;
        let groups = 257;
        let values = (0..groups * S::METRICS)
            .map(|index| (1u64 << 40) + index as u64)
            .collect::<Vec<_>>();
        let input = device.tensor(bytemuck::cast_slice(&values));
        let output = device.empty(S::WORDS);
        let control = device.tensor(&[0; CONTROL_WORDS]);
        let group_counts = device.tensor(&vec![groups as u32; S::METRICS]);
        summary(&device, &input, &output, &control, &group_counts)?;
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

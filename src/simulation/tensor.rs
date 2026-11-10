use super::{
    Device,
    pass::GpuPass,
    system::{System, gpu_passes},
};
use anyhow::{Result, anyhow};
use burn::tensor::{DType, Shape};
use burn_wgpu::CubeTensor;
use cubecl::prelude::*;
use cubek::{
    matmul::{definition::MatmulElems, strategy::Strategy},
    std::InputBinding,
};

pub(super) fn vector_width(device: &Device, elements: usize) -> usize {
    device
        .client
        .io_optimized_vector_sizes(size_of::<f32>())
        .filter(|&width| width <= 4 && elements.max(1).is_multiple_of(width))
        .max()
        .unwrap_or(1)
}

pub(super) fn floats(device: &Device, shape: impl Into<Shape>, values: &[f32]) -> CubeTensor {
    let handle = match values.len() {
        0 => device.client.empty(size_of::<f32>()),
        _ => device
            .client
            .create_from_slice(bytemuck::cast_slice(values)),
    };
    CubeTensor::new_contiguous(
        device.client.clone(),
        device.cube.clone(),
        shape.into(),
        handle,
        DType::F32,
    )
}

pub(super) fn empty(device: &Device, shape: impl Into<Shape>) -> CubeTensor {
    let shape = shape.into();
    let handle = device
        .client
        .empty(shape.num_elements().max(1) * size_of::<f32>());
    CubeTensor::new_contiguous(
        device.client.clone(),
        device.cube.clone(),
        shape,
        handle,
        DType::F32,
    )
}

pub(super) fn reshape(tensor: &CubeTensor, shape: impl Into<Shape>) -> CubeTensor {
    CubeTensor::new_contiguous(
        tensor.client.clone(),
        tensor.device.clone(),
        shape.into(),
        tensor.handle.clone(),
        tensor.dtype,
    )
}

pub(super) fn matmul(
    device: &Device,
    left: &CubeTensor,
    right: &CubeTensor,
    output: &CubeTensor,
) -> Result<()> {
    let inner = left.meta.shape()[left.meta.num_dims() - 1];
    match output.meta.num_elements() {
        0 => Ok(()),
        _ if inner == 0 => {
            device.client.write(
                &output.handle,
                burn::tensor::Bytes::from_elems(vec![0.0f32; output.meta.num_elements()]),
            );
            Ok(())
        }
        _ => cubek::matmul::launch::launch_ref(
            &Strategy::Auto,
            &device.client,
            InputBinding::new(left.clone().binding(), f32::elem_type_native()),
            InputBinding::new(right.clone().binding(), f32::elem_type_native()),
            output.clone().binding(),
            &mut MatmulElems::from_single_dtype(f32::elem_type_native()),
        )
        .map_err(|error| anyhow!("Cubek matrix multiplication: {error:?}")),
    }
}

pub(super) fn uniform(device: &Device, shape: impl Into<Shape>, seed: u64) -> Result<CubeTensor> {
    let output = empty(device, shape);
    if output.meta.num_elements() > 0 {
        cubek::random::with_seed(seed, || {
            cubek::random::random_uniform(
                &device.client,
                0.0,
                1.0,
                output.clone().binding(),
                f32::elem_type_native(),
            )
        })?;
    }
    Ok(output)
}

pub(super) fn copy(device: &Device, source: &CubeTensor, target: &CubeTensor) {
    cubecl::std::tensor::copy_into(
        &device.client,
        source.clone().binding(),
        target.clone().binding(),
        u32::elem_type_native(),
    );
}

pub(super) fn copy_contiguous(device: &Device, source: &CubeTensor, target: &CubeTensor) {
    match source.meta.num_elements().is_multiple_of(4) {
        true => ContiguousCopy::launch(
            device,
            device.count(source.meta.num_elements().div_ceil(32)),
            CubeDim::new_1d(device.workgroup),
            CopyInputLaunch {
                source: source.clone().into_buffer_arg(),
                target: target.clone().into_buffer_arg(),
                copies: 8,
            },
        ),
        false => copy(device, source, target),
    }
}

pub(super) enum SnapshotCopy {
    Full,
    Sparse { nonzero: CubeTensor },
}

impl SnapshotCopy {
    pub(super) fn new(device: &Device, words: usize) -> Self {
        let properties = device.client.properties();
        match words >= 1024
            && words.is_multiple_of(4)
            && properties
                .features
                .plane
                .contains(cubecl::features::Plane::Ops)
            && properties.hardware.plane_size_min >= 32
        {
            true => Self::Sparse {
                nonzero: device.tensor(&vec![1; words.div_ceil(1024)]),
            },
            false => Self::Full,
        }
    }

    pub(super) fn save(&self, device: &Device, source: &CubeTensor, target: &CubeTensor) {
        match self {
            Self::Full => copy_contiguous(device, source, target),
            Self::Sparse { nonzero } => SparseCopy::launch(
                device,
                cubecl::calculate_cube_count_elemwise(
                    &device.client,
                    source.meta.num_elements().div_ceil(32),
                    CubeDim::new_1d(32),
                ),
                CubeDim::new_1d(32),
                SnapshotInputLaunch {
                    source: source.clone().into_buffer_arg(),
                    target: target.clone().into_buffer_arg(),
                    nonzero: nonzero.clone().into_buffer_arg(),
                },
            ),
        }
    }
}

#[derive(CubeType, CubeLaunch)]
struct SnapshotInput {
    source: Box<[Vector<u32, Const<4>>]>,
    target: Box<[Vector<u32, Const<4>>]>,
    nonzero: Box<[u32]>,
}

struct SparseCopy;
struct SparseWords;

impl System for SparseCopy {
    type Input = SnapshotInput;
    type Passes = gpu_passes![SparseWords];
}

#[cube]
impl GpuPass<SnapshotInput> for SparseWords {
    type Output = ();

    fn execute(mut input: SnapshotInput) {
        if CUBE_POS < input.nonzero.len() {
            let mut values = Array::<Vector<u32, Const<4>>>::new(8usize);
            let mut local = 0u32;
            #[unroll]
            for part in 0..8usize {
                let index = CUBE_POS * 256 + part * 32 + UNIT_POS as usize;
                let mut value = Vector::<u32, Const<4>>::new(0);
                if index < input.source.len() {
                    value = input.source[index];
                }
                values[part] = value;
                local |= value.extract(0usize)
                    | value.extract(1usize)
                    | value.extract(2usize)
                    | value.extract(3usize);
            }
            let current = plane_max(local);
            let mut prior = 0u32;
            if UNIT_POS == 0 {
                prior = input.nonzero[CUBE_POS];
                input.nonzero[CUBE_POS] = u32::cast_from(current != 0);
            }
            let previous = plane_max(prior);
            if (current | previous) != 0 {
                #[unroll]
                for part in 0..8usize {
                    let index = CUBE_POS * 256 + part * 32 + UNIT_POS as usize;
                    if index < input.source.len() {
                        input.target[index] = values[part];
                    }
                }
            }
        }
    }
}

#[derive(CubeType, CubeLaunch)]
struct CopyInput {
    source: Box<[Vector<u32, Const<4>>]>,
    target: Box<[Vector<u32, Const<4>>]>,
    #[cube(comptime)]
    copies: u32,
}

struct ContiguousCopy;
struct CopyWords;

impl System for ContiguousCopy {
    type Input = CopyInput;
    type Passes = gpu_passes![CopyWords];
}

#[cube]
impl GpuPass<CopyInput> for CopyWords {
    type Output = ();

    fn execute(mut input: CopyInput) {
        let first = ABSOLUTE_POS as u32 / CUBE_DIM * CUBE_DIM * comptime!(input.copies) + UNIT_POS;
        #[unroll]
        for part in 0..comptime!(input.copies) {
            let index = (first + part * CUBE_DIM) as usize;
            if index < input.source.len() {
                input.target[index] = input.source[index];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contiguous_gpu_copy_preserves_words_across_dispatch_boundaries() -> Result<()> {
        for workgroup in [32, 256] {
            let device = crate::simulation::tests::device(workgroup)?;
            for count in [1, 2, 3, 4, 20, 255, 256, 260, 2044, 2048, 2052, 65_540] {
                let values = (0..count)
                    .map(|index| (index as u32).wrapping_mul(2_654_435_761))
                    .collect::<Vec<_>>();
                let source = device.tensor(&values);
                let target = device.tensor(&vec![u32::MAX; count]);
                copy_contiguous(&device, &source, &target);
                assert_eq!(device.read(&target)?, values);
            }
        }
        Ok(())
    }

    #[test]
    fn sparse_gpu_snapshot_preserves_transitions_and_restore() -> Result<()> {
        let device = crate::simulation::tests::device(64)?;
        for count in [
            1, 3, 4, 1020, 1024, 1028, 2048, 2052, 4092, 4096, 4100, 65_540, 65_541,
        ] {
            let snapshot = SnapshotCopy::new(&device, count);
            let audited = device.tensor(&vec![u32::MAX; count]);
            let initial = vec![0; count];
            let populated = (0..count)
                .map(|index| match index / 1024 % 3 {
                    0 => (index as u32).wrapping_mul(2_654_435_761),
                    _ => 0,
                })
                .collect::<Vec<_>>();
            let changed = (0..count)
                .map(|index| match index / 1024 % 3 {
                    1 => u32::MAX - index as u32,
                    _ => 0,
                })
                .collect::<Vec<_>>();
            for values in [&initial, &populated, &changed, &initial, &populated] {
                let source = device.tensor(values);
                snapshot.save(&device, &source, &audited);
                assert_eq!(device.read(&audited)?, *values);
                let restored = device.tensor(&vec![17; count]);
                copy_contiguous(&device, &audited, &restored);
                assert_eq!(device.read(&restored)?, *values);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod copy_profile {
    use super::*;

    #[test]
    #[ignore]
    fn profile_contiguous_copy_on_gpu() -> Result<()> {
        let words = 128 * 1024 * 1024;
        for workgroup in [64, 256] {
            let device = crate::simulation::tests::device(workgroup)?;
            let source = device.tensor(&vec![17; words]);
            let target = device.empty(words);
            for copies in [1, 8, 32, 128] {
                let launch = || {
                    ContiguousCopy::launch(
                        &device,
                        device.count(words.div_ceil(4 * copies as usize)),
                        CubeDim::new_1d(workgroup),
                        CopyInputLaunch {
                            source: source.clone().into_buffer_arg(),
                            target: target.clone().into_buffer_arg(),
                            copies,
                        },
                    )
                };
                launch();
                device.sync()?;
                let start = std::time::Instant::now();
                for _ in 0..5 {
                    launch();
                    device.sync()?;
                }
                eprintln!(
                    "COPYBENCH {workgroup} {copies} {:.3}",
                    start.elapsed().as_secs_f64() * 200.0
                );
                assert_eq!(device.read_range(&target, words - 4, 4)?, [17; 4]);
            }
        }
        Ok(())
    }
}

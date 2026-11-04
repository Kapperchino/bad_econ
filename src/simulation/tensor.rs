use super::Device;
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

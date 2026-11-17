use super::{Device, pass::GpuPass};
use burn_wgpu::CubeTensor;
use cubecl::prelude::*;
use std::marker::PhantomData;

pub struct PassList<First, Next> {
    first: PhantomData<First>,
    next: PhantomData<Next>,
}

pub struct End;

#[macro_export]
macro_rules! gpu_passes {
    () => {
        $crate::simulation::system::End
    };
    ($first:ty $(, $next:ty)* $(,)?) => {
        $crate::simulation::system::PassList<$first, $crate::gpu_passes![$($next),*]>
    };
}

pub use crate::gpu_passes;

#[cube]
impl<Input: CubeType> GpuPass<Input> for End {
    type Output = Input;

    fn execute(input: Input) -> Input {
        input
    }
}

#[cube]
impl<Input: CubeType, First: GpuPass<Input>, Next: GpuPass<First::Output>> GpuPass<Input>
    for PassList<First, Next>
{
    type Output = Next::Output;

    fn execute(input: Input) -> Self::Output {
        Next::execute(First::execute(input))
    }
}

pub(super) trait AsBufferArg {
    fn buffer_arg(&self) -> BufferArg;
}

impl AsBufferArg for CubeTensor {
    fn buffer_arg(&self) -> BufferArg {
        assert!(
            !self.meta.is_tiled(),
            "kernel buffer arguments require untiled tensors"
        );
        TensorBinding {
            handle: self.handle.clone().binding(),
            shape: self.meta.shape.clone(),
            strides: self.meta.strides.clone(),
            tiling: self.meta.tiling,
        }
        .into_buffer_arg()
    }
}

pub trait System: Sized + Send + Sync + 'static {
    type Input: LaunchArg;
    type Passes: GpuPass<Self::Input, Output = ()>;

    fn launch_elements(
        device: &Device,
        elements: usize,
        input: <Self::Input as LaunchArg>::RuntimeArg,
    ) {
        Self::launch(
            device,
            device.count(elements),
            CubeDim::new_1d(device.workgroup),
            input,
        );
    }

    fn launch_single(device: &Device, input: <Self::Input as LaunchArg>::RuntimeArg) {
        Self::launch(
            device,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            input,
        );
    }

    fn launch(
        device: &Device,
        count: CubeCount,
        dimensions: CubeDim,
        input: <Self::Input as LaunchArg>::RuntimeArg,
    ) {
        launch_system::launch::<Self>(&device.client, count, dimensions, input);
    }
}

#[cube(launch)]
fn launch_system<S: System>(input: S::Input) {
    S::Passes::execute(input);
}

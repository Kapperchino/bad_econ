use crate::simulation::{Device, pass::GpuPass};
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

pub(in crate::simulation) trait AsBufferArg {
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

pub enum Dispatch {
    Grid(CubeCount),
    Bounded {
        counts: BufferArg,
        offset: u32,
        max_groups: u32,
    },
}

impl From<CubeCount> for Dispatch {
    fn from(count: CubeCount) -> Self {
        Self::Grid(count)
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
        count: impl Into<Dispatch>,
        dimensions: CubeDim,
        input: <Self::Input as LaunchArg>::RuntimeArg,
    ) {
        match count.into() {
            Dispatch::Grid(count) => {
                launch_system::launch::<Self>(&device.client, count, dimensions, input)
            }
            Dispatch::Bounded {
                counts,
                offset,
                max_groups,
            } => {
                let grid = CubeCount::Static(
                    max_groups.clamp(1, 65535),
                    max_groups.div_ceil(65535).max(1),
                    1,
                );
                launch_bounded::launch::<Self>(
                    &device.client,
                    grid,
                    dimensions,
                    input,
                    counts,
                    offset,
                );
            }
        }
    }
}

#[cube(launch)]
fn launch_system<S: System>(input: S::Input) {
    S::Passes::execute(input);
}

#[cube(launch)]
fn launch_bounded<S: System>(input: S::Input, counts: &[u32], offset: u32) {
    if CUBE_POS_X < counts[offset as usize] && CUBE_POS_Y < counts[offset as usize + 1] {
        S::Passes::execute(input);
    }
}

use super::{Device, pass::GpuPass};
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

pub trait System: Sized + Send + Sync + 'static {
    type Input: LaunchArg;
    type Passes: GpuPass<Self::Input, Output = ()>;

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

use cubecl::prelude::*;

#[cube]
pub trait GpuPass<Input: CubeType>: Send + Sync + 'static {
    type Output: CubeType;

    fn execute(input: Input) -> Self::Output;
}

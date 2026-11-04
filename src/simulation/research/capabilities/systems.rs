use super::{ClosureDirection, initialize, propagate};
use crate::simulation::{
    pass::GpuPass,
    system::{System, gpu_passes},
};
use cubecl::prelude::*;

#[derive(CubeType, CubeLaunch)]
pub(super) struct ClosureInitializationInput {
    pub(super) seed: Box<[u32]>,
    pub(super) mask: Box<[u32]>,
    pub(super) output: Box<[u32]>,
    pub(super) nodes: u32,
    #[cube(comptime)]
    pub(super) direction: ClosureDirection,
}

pub(super) struct ClosureInitialization;
pub(super) struct InitializeClosure;

impl System for ClosureInitialization {
    type Input = ClosureInitializationInput;
    type Passes = gpu_passes![InitializeClosure];
}

#[cube]
impl GpuPass<ClosureInitializationInput> for InitializeClosure {
    type Output = ();

    fn execute(mut input: ClosureInitializationInput) {
        initialize(
            &input.seed,
            &input.mask,
            &mut input.output,
            input.nodes,
            comptime!(input.direction),
        );
    }
}

#[derive(CubeType, CubeLaunch)]
pub(super) struct ClosurePropagationInput {
    pub(super) seed: Box<[u32]>,
    pub(super) mask: Box<[u32]>,
    pub(super) offsets: Box<[u32]>,
    pub(super) sources: Box<[u32]>,
    pub(super) previous: Box<[u32]>,
    pub(super) next: Box<[u32]>,
    pub(super) changed: Box<[Atomic<u32>]>,
    pub(super) nodes: u32,
    #[cube(comptime)]
    pub(super) direction: ClosureDirection,
}

pub(super) struct ClosurePropagation;
pub(super) struct PropagateClosure;

impl System for ClosurePropagation {
    type Input = ClosurePropagationInput;
    type Passes = gpu_passes![PropagateClosure];
}

#[cube]
impl GpuPass<ClosurePropagationInput> for PropagateClosure {
    type Output = ();

    fn execute(mut input: ClosurePropagationInput) {
        propagate(
            &input.seed,
            &input.mask,
            &input.offsets,
            &input.sources,
            &input.previous,
            &mut input.next,
            &input.changed,
            input.nodes,
            comptime!(input.direction),
        );
    }
}

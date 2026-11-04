use super::*;

#[derive(CubeType, CubeLaunch)]
pub(super) struct TransitionInput {
    pub state: Box<[u32]>,
    pub actors: Box<[u32]>,
    pub config: EconomyConfig,
    pub requests: Box<[u32]>,
    pub incoming: Box<[Atomic<u32>]>,
    pub day: u32,
    #[cube(comptime)]
    pub stage: Stage,
}

pub(super) struct RegistryTransitions;
pub(super) struct ApplyTransition;

impl System for RegistryTransitions {
    type Input = TransitionInput;
    type Passes = gpu_passes![ApplyTransition];
}

#[cube]
impl GpuPass<TransitionInput> for ApplyTransition {
    type Output = ();

    fn execute(mut input: TransitionInput) {
        transition(
            &mut input.state,
            &input.actors,
            &input.config,
            &input.requests,
            &input.incoming,
            input.day,
            comptime!(input.stage),
        );
    }
}

#[derive(CubeType, CubeLaunch)]
pub(super) struct CapacityInput {
    pub actors: Box<[u32]>,
    pub requests: Box<[u32]>,
    pub output: Box<[u32]>,
}

pub(super) struct CapacityPatching;
pub(super) struct PatchCapacity;

impl System for CapacityPatching {
    type Input = CapacityInput;
    type Passes = gpu_passes![PatchCapacity];
}

#[cube]
impl GpuPass<CapacityInput> for PatchCapacity {
    type Output = ();

    fn execute(mut input: CapacityInput) {
        capacity_rows(&input.actors, &input.requests, &mut input.output);
    }
}

#[derive(CubeType, CubeLaunch)]
pub(super) struct ActorGrowthInput {
    pub actors: Box<[u32]>,
    pub additions: Box<[u32]>,
    pub output: Box<[u32]>,
}

pub(super) struct ActorGrowth;
pub(super) struct ExtendActors;

impl System for ActorGrowth {
    type Input = ActorGrowthInput;
    type Passes = gpu_passes![ExtendActors];
}

#[cube]
impl GpuPass<ActorGrowthInput> for ExtendActors {
    type Output = ();

    fn execute(mut input: ActorGrowthInput) {
        extend_rows(&input.actors, &input.additions, &mut input.output);
    }
}

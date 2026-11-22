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
            &mut EconomyView::new(&mut input.state, &input.actors, &input.config),
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
        let index = ABSOLUTE_POS;
        if index < input.actors.len() {
            let count = input.requests.len();
            let action = input.requests[index % count];
            let mut value = input.actors[index];
            let field = index / count;
            let capacity = field == A::CAPACITY || (field == A::GEOGRAPHIC_CAPACITY && value > 0);
            if capacity && action >> 24 == 6 {
                value += action & 0x00ffffff;
            }
            input.output[index] = value;
        }
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
        let index = ABSOLUTE_POS;
        if index < input.output.len() {
            let before = input.actors.len() / A::WORDS;
            let added = input.additions.len() / A::WORDS;
            let count = before + added;
            let actor = index % count;
            let field = index / count;
            input.output[index] = if actor < before {
                input.actors[field * before + actor]
            } else {
                input.additions[field * added + actor - before]
            };
        }
    }
}

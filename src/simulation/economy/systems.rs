use super::*;

#[derive(CubeType, CubeLaunch)]
pub(in crate::simulation) struct EconomyInput {
    pub state: Box<[u32]>,
    pub actors: Box<[u32]>,
    pub config: EconomyConfig,
    pub spatial: Box<[u32]>,
    pub proposals: Box<[u32]>,
    pub bundles: Box<[u32]>,
    pub control: Box<[Atomic<u32>]>,
    pub parameters: EconomyParameters,
    #[cube(comptime)]
    pub stage: EconomyStage,
}

pub(in crate::simulation) struct EconomySystem;
pub(in crate::simulation) struct AdvanceEconomy;

impl System for EconomySystem {
    type Input = EconomyInput;
    type Passes = gpu_passes![AdvanceEconomy];
}

#[cube]
impl GpuPass<EconomyInput> for AdvanceEconomy {
    type Output = ();

    fn execute(mut input: EconomyInput) {
        economy(
            &mut input.state,
            &input.actors,
            &input.config,
            &mut input.spatial,
            &mut input.proposals,
            &mut input.bundles,
            &input.control,
            &input.parameters,
            comptime!(input.stage),
        );
    }
}

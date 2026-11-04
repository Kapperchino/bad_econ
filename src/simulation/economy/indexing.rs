use super::*;

#[derive(CubeType, CubeLaunch)]
pub(in crate::simulation) struct IndexInput {
    pub config: EconomyConfig,
    pub spatial: Box<[u32]>,
    pub scratch: Box<[u32]>,
    pub control: Box<[Atomic<u32>]>,
    pub parameters: EconomyParameters,
}

pub(in crate::simulation) struct MarketIndex;
pub(in crate::simulation) struct StoreOffsets;
pub(in crate::simulation) struct StoreMembers;

impl System for MarketIndex {
    type Input = IndexInput;
    type Passes = gpu_passes![StoreOffsets, StoreMembers];
}

#[cube]
impl GpuPass<IndexInput> for StoreOffsets {
    type Output = IndexInput;

    fn execute(mut input: IndexInput) -> IndexInput {
        if input.control[LATCH_WORD].load() == 0 {
            index_offsets(
                &input.config,
                &mut input.spatial,
                &mut input.scratch,
                &input.parameters,
                ABSOLUTE_POS as u32,
            );
        }
        input
    }
}

#[cube]
impl GpuPass<IndexInput> for StoreMembers {
    type Output = ();

    fn execute(mut input: IndexInput) {
        if input.control[LATCH_WORD].load() == 0 {
            index_fill(
                &input.config,
                &mut input.spatial,
                &mut input.scratch,
                &input.parameters,
                ABSOLUTE_POS as u32,
            );
        }
    }
}

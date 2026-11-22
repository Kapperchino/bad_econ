use super::*;

#[derive(CubeType, CubeLaunch)]
pub(super) struct BankInput {
    pub state: Box<[u32]>,
    pub banks: Box<[u32]>,
    pub loans: Box<[u32]>,
    pub output: Box<[u32]>,
    pub decisions: Box<[u32]>,
    pub control: Box<[Atomic<u32>]>,
    pub day: u32,
    pub count: u32,
    #[cube(comptime)]
    pub stage: FinanceStage,
}

pub(super) struct BankAccounting;
pub(super) struct UpdateBanks;

impl System for BankAccounting {
    type Input = BankInput;
    type Passes = gpu_passes![UpdateBanks];
}

#[cube]
impl GpuPass<BankInput> for UpdateBanks {
    type Output = ();

    fn execute(mut input: BankInput) {
        let stage = comptime!(input.stage);
        bank::account(&mut input, stage);
    }
}

#[derive(CubeType, CubeLaunch)]
pub(super) struct SupportNeedsInput {
    pub state: Box<[u32]>,
    pub recipients: Box<[u32]>,
    pub output: Box<[u32]>,
    pub count: u32,
}

pub(super) struct SupportNeeds;
pub(super) struct CalculateSupportNeeds;

impl System for SupportNeeds {
    type Input = SupportNeedsInput;
    type Passes = gpu_passes![CalculateSupportNeeds];
}

#[cube]
impl GpuPass<SupportNeedsInput> for CalculateSupportNeeds {
    type Output = ();

    fn execute(mut input: SupportNeedsInput) {
        support::needs(&mut input);
    }
}

#[derive(CubeType, CubeLaunch)]
pub(super) struct SupportPlanInput {
    pub state: Box<[u32]>,
    pub recipients: Box<[u32]>,
    pub scanned: Box<[u32]>,
    pub output: Box<[u32]>,
    pub control: Box<[Atomic<u32>]>,
    pub count: u32,
    pub treasury: u32,
}

pub(super) struct SupportPlanning;
pub(super) struct PlanSupport;

impl System for SupportPlanning {
    type Input = SupportPlanInput;
    type Passes = gpu_passes![PlanSupport];
}

#[cube]
impl GpuPass<SupportPlanInput> for PlanSupport {
    type Output = ();

    fn execute(mut input: SupportPlanInput) {
        support::plan(&mut input);
    }
}

#[derive(CubeType, CubeLaunch)]
pub(super) struct SupportPostInput {
    pub state: Box<[u32]>,
    pub recipients: Box<[u32]>,
    pub output: Box<[u32]>,
    pub count: u32,
    pub treasury: u32,
}

pub(super) struct SupportPosting;
pub(super) struct PostSupport;

impl System for SupportPosting {
    type Input = SupportPostInput;
    type Passes = gpu_passes![PostSupport];
}

#[cube]
impl GpuPass<SupportPostInput> for PostSupport {
    type Output = ();

    fn execute(mut input: SupportPostInput) {
        support::post(&mut input);
    }
}

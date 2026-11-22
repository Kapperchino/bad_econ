use super::{access::ResearchView, layout::KernelResearchConfig};
use crate::simulation::{
    pass::GpuPass,
    system::{System, gpu_passes},
};
use cubecl::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ResearchStage {
    Opening,
    ProductionReadiness,
    MachineCapacity,
    StockCapacity,
    Strike,
    Maintenance,
    Attempt,
    MachineClose,
    StockClose,
    Summary,
    Audit,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttemptOutcome {
    Idle,
    Ineligible,
    MissingResources,
    NoTime,
    Failed,
    Discovered,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Participation {
    Working,
    Striking,
}

#[derive(CubeType, CubeLaunch)]
pub(super) struct ResearchInput {
    pub main: Box<[u32]>,
    pub state: Box<[u32]>,
    pub config: KernelResearchConfig,
    pub table: Box<[u32]>,
    pub samples: Box<[f32]>,
    pub output: Box<[u32]>,
    pub control: Box<[Atomic<u32>]>,
    #[cube(comptime)]
    pub stage: ResearchStage,
}

pub(super) struct ResearchSystem;
pub(super) struct EvaluateResearch;

impl System for ResearchSystem {
    type Input = ResearchInput;
    type Passes = gpu_passes![EvaluateResearch];
}

#[cube]
impl GpuPass<ResearchInput> for EvaluateResearch {
    type Output = ();

    fn execute(mut input: ResearchInput) {
        let stage = comptime!(input.stage);
        let mut research = ResearchView::new(&mut input);
        let id = ABSOLUTE_POS;
        match stage {
            ResearchStage::Opening => {
                research.select_method(id);
                research.open_machine(id);
                research.open_stock(id);
                research.open_project(id);
            }
            ResearchStage::ProductionReadiness => research.production_readiness(id),
            ResearchStage::MachineCapacity => research.machine_capacity(id),
            ResearchStage::StockCapacity => research.stock_capacity(id),
            ResearchStage::Strike => research.strike(id),
            ResearchStage::Maintenance => research.maintain(id),
            ResearchStage::Attempt => research.research(id),
            ResearchStage::MachineClose => research.close_machine(id),
            ResearchStage::StockClose => research.close_stock(id),
            ResearchStage::Summary => research.summarize(id),
            ResearchStage::Audit => research.audit(id),
        }
    }
}

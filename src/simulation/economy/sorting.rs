use super::markets;
use crate::simulation::{
    Device, parallel,
    parameters::EconomyParameters,
    pass::GpuPass,
    system::{System, gpu_passes},
};
use anyhow::Result;
use burn::tensor::StreamId;
use burn_wgpu::CubeTensor;
use cubecl::{
    client::Graph,
    config::{CubeClRuntimeConfig, RuntimeConfig, profiling::ProfilingLogLevel},
    prelude::*,
};

#[cfg(test)]
mod tests;

pub(in crate::simulation) enum MarketSort {
    Profiled,
    Unrecorded,
    Recorded(Box<RecordedSort>),
}

impl Default for MarketSort {
    fn default() -> Self {
        match CubeClRuntimeConfig::get().profiling.logger.level {
            ProfilingLogLevel::Basic | ProfilingLogLevel::Medium | ProfilingLogLevel::Full => {
                Self::Profiled
            }
            _ => Self::Unrecorded,
        }
    }
}

pub(in crate::simulation) struct RecordedSort {
    graph: Graph,
    plan: SortPlan,
    workgroup: u32,
    thread: std::thread::ThreadId,
    stream: StreamId,
    scratch: CubeTensor,
    control: CubeTensor,
    dispatches: CubeTensor,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct SortPlan {
    count: u32,
    source: u32,
    active: u32,
    discovery: crate::simulation::parameters::Discovery,
    scan: parallel::ScanMode,
}

impl SortPlan {
    fn new(parameters: &EconomyParameters) -> Self {
        Self {
            count: parameters.count,
            source: parameters.sort_source,
            active: parameters.sort_count,
            discovery: parameters.discovery,
            scan: parameters.scan,
        }
    }

    fn output(self) -> u32 {
        match (self.count.max(1).ilog2() + 1).div_ceil(4) % 2 {
            0 => self.source,
            _ => 2 * self.count - self.source,
        }
    }
}

impl RecordedSort {
    fn compatible(
        &self,
        device: &Device,
        scratch: &CubeTensor,
        control: &CubeTensor,
        dispatches: &CubeTensor,
        plan: SortPlan,
    ) -> bool {
        [
            (self.plan, self.workgroup, self.thread, self.stream)
                == (
                    plan,
                    device.workgroup,
                    std::thread::current().id(),
                    StreamId::current(),
                ),
            self.scratch.handle.service == device.client.service_id(),
            same_buffer(&self.scratch, scratch),
            same_buffer(&self.control, control),
            same_buffer(&self.dispatches, dispatches),
        ]
        .into_iter()
        .all(std::convert::identity)
    }
}

fn same_buffer(recorded: &CubeTensor, current: &CubeTensor) -> bool {
    (
        recorded.handle.memory.descriptor(),
        recorded.handle.service,
        recorded.handle.offset_start,
        recorded.handle.offset_end,
        recorded.handle.size(),
        &recorded.meta,
        recorded.dtype,
    ) == (
        current.handle.memory.descriptor(),
        current.handle.service,
        current.handle.offset_start,
        current.handle.offset_end,
        current.handle.size(),
        &current.meta,
        current.dtype,
    )
}

impl MarketSort {
    pub fn run(
        &mut self,
        device: &Device,
        scratch: &CubeTensor,
        control: &CubeTensor,
        dispatches: &CubeTensor,
        parameters: &EconomyParameters,
    ) -> Result<u32> {
        let plan = SortPlan::new(parameters);
        match self {
            Self::Profiled => encode(device, scratch, control, dispatches, parameters),
            Self::Recorded(recorded)
                if recorded.compatible(device, scratch, control, dispatches, plan) =>
            {
                unsafe { recorded.graph.replay() }?;
            }
            _ => {
                *self = Self::Unrecorded;
                let client = &device.client;
                client.graph_prepare()?;
                encode(device, scratch, control, dispatches, parameters);
                client.start_capture()?;
                encode(device, scratch, control, dispatches, parameters);
                let graph = client.stop_capture()?;
                *self = Self::Recorded(Box::new(RecordedSort {
                    graph,
                    plan,
                    workgroup: device.workgroup,
                    thread: std::thread::current().id(),
                    stream: StreamId::current(),
                    scratch: scratch.clone(),
                    control: control.clone(),
                    dispatches: dispatches.clone(),
                }));
            }
        }
        Ok(plan.output())
    }
}

fn encode(
    device: &Device,
    scratch: &CubeTensor,
    control: &CubeTensor,
    dispatches: &CubeTensor,
    parameters: &EconomyParameters,
) {
    let mut parameters = *parameters;
    let count = parameters.count as usize;
    for shift in (0..parameters.count.max(1).ilog2() + 1).step_by(4) {
        parameters.sort_width = shift;
        let launch = |stage| {
            RadixSort::launch(
                device,
                markets::dispatch(dispatches, markets::SORT_DISPATCH),
                CubeDim::new_1d(device.workgroup),
                SortInputLaunch {
                    scratch: scratch.clone().into_buffer_arg(),
                    control: control.clone().into_buffer_arg(),
                    parameters: parameters.launch(),
                    stage,
                },
            );
        };
        launch(SortStage::Local);
        parallel::scan_active(
            device,
            scratch,
            7 * count,
            16 * count.div_ceil(device.workgroup as usize),
            control,
            dispatches,
        );
        launch(SortStage::Scatter);
        parameters.sort_source = 2 * parameters.count - parameters.sort_source;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum SortStage {
    Local,
    Scatter,
}

#[derive(CubeType, CubeLaunch)]
struct SortInput {
    scratch: Box<[u32]>,
    control: Box<[Atomic<u32>]>,
    parameters: EconomyParameters,
    #[cube(comptime)]
    stage: SortStage,
}

struct RadixSort;
struct Sort;

impl System for RadixSort {
    type Input = SortInput;
    type Passes = gpu_passes![Sort];
}

#[cube]
impl GpuPass<SortInput> for Sort {
    type Output = ();

    fn execute(mut input: SortInput) {
        if markets::faults_clear(&input.control) {
            match comptime!(input.stage) {
                SortStage::Local => markets::radix_local(
                    &mut input.scratch,
                    &input.control,
                    &input.parameters,
                    ABSOLUTE_POS as u32,
                ),
                SortStage::Scatter => markets::radix_scatter(
                    &mut input.scratch,
                    &input.control,
                    &input.parameters,
                    ABSOLUTE_POS as u32,
                ),
            }
        }
    }
}

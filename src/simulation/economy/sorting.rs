use super::markets;
use crate::simulation::{
    Device, parallel,
    pass::GpuPass,
    system::{System, gpu_passes},
};
use anyhow::Result;
use burn_wgpu::CubeTensor;
use cubecl::{client::Graph, prelude::*};

#[derive(Default)]
pub(in crate::simulation) enum MarketSort {
    #[default]
    Unrecorded,
    Recorded(RecordedSort),
}

pub(in crate::simulation) struct RecordedSort {
    graph: Graph,
    thread: std::thread::ThreadId,
    _scratch: CubeTensor,
    _control: CubeTensor,
    _dispatches: CubeTensor,
}

impl MarketSort {
    pub fn run(
        &mut self,
        device: &Device,
        scratch: &CubeTensor,
        control: &CubeTensor,
        dispatches: &CubeTensor,
        count: usize,
    ) -> Result<u32> {
        match self {
            Self::Recorded(recorded) if recorded.thread == std::thread::current().id() => {
                unsafe { recorded.graph.replay() }?;
            }
            _ => {
                let client = &device.client;
                let recorded = RecordedSort {
                    graph: {
                        client.graph_prepare()?;
                        encode(device, scratch, control, dispatches, count);
                        client.start_capture()?;
                        encode(device, scratch, control, dispatches, count);
                        client.stop_capture()?
                    },
                    thread: std::thread::current().id(),
                    _scratch: scratch.clone(),
                    _control: control.clone(),
                    _dispatches: dispatches.clone(),
                };
                *self = Self::Recorded(recorded);
            }
        }
        Ok((count.max(1).ilog2() + 1).div_ceil(4) % 2 * 2 * count as u32)
    }
}

fn encode(
    device: &Device,
    scratch: &CubeTensor,
    control: &CubeTensor,
    dispatches: &CubeTensor,
    count: usize,
) {
    let mut source = 0;
    for shift in (0..count.max(1).ilog2() + 1).step_by(4) {
        let launch = |stage| {
            RadixSort::launch(
                device,
                markets::dispatch(dispatches, markets::SORT_DISPATCH),
                CubeDim::new_1d(device.workgroup),
                SortInputLaunch {
                    scratch: scratch.clone().into_buffer_arg(),
                    control: control.clone().into_buffer_arg(),
                    parameters: RadixParametersLaunch {
                        count: count as u32,
                        source,
                        shift,
                        scan: parallel::ScanMode::new(device),
                    },
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
        source = 2 * count as u32 - source;
    }
}

#[derive(CubeType, CubeLaunch)]
pub(super) struct RadixParameters {
    pub count: u32,
    pub source: u32,
    pub shift: u32,
    #[cube(comptime)]
    pub scan: parallel::ScanMode,
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
    parameters: RadixParameters,
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
        if input.control[markets::LATCH_WORD].load() == 0 {
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

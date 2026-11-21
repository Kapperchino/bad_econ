use super::markets;
use crate::simulation::{
    Device,
    market_graphs::same_buffer,
    parallel,
    parameters::EconomyParameters,
    pass::GpuPass,
    system::{AsBufferArg, System, gpu_passes},
    words::ceildiv,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::simulation) enum SortKeys {
    Actors,
    Firms { households: u32 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Radix {
    Nibble,
    Byte,
}

impl Radix {
    fn for_count(count: u32) -> Self {
        match count {
            65_536.. => Self::Byte,
            _ => Self::Nibble,
        }
    }

    fn bits(self) -> u32 {
        match self {
            Self::Nibble => 4,
            Self::Byte => 8,
        }
    }

    fn layout(self, device: &Device) -> parallel::HistogramLayout {
        parallel::HistogramLayout {
            bins: 1u32 << self.bits(),
            source_workgroup: match self {
                Self::Nibble => device.workgroup,
                _ => 256,
            },
        }
    }
}

impl SortKeys {
    pub fn for_market(
        market: crate::domain::Market,
        households: u32,
        home_mode: crate::simulation::configuration::Feature,
    ) -> Self {
        use crate::domain::{Good, Market};
        match market {
            Market::Labor => Self::Firms { households },
            Market::Input(Good::Grain) | Market::Consumer(Good::Grain)
                if home_mode == crate::simulation::configuration::Feature::Enabled =>
            {
                Self::Actors
            }
            Market::Input(Good::Bread)
            | Market::Consumer(Good::Bread)
            | Market::Household
            | Market::Payroll
            | Market::Distribution => Self::Actors,
            Market::Input(_) | Market::Consumer(_) => Self::Firms { households },
        }
    }

    fn base(self) -> u32 {
        match self {
            Self::Actors => 0,
            Self::Firms { households } => households,
        }
    }

    fn span(self, count: u32) -> u32 {
        count.saturating_sub(self.base())
    }
}

pub(in crate::simulation) enum MarketSort {
    Profiled,
    Unrecorded,
    Recorded(Box<SortCache>),
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

#[derive(Default)]
pub(in crate::simulation) struct SortCache {
    actors: SortGraph,
    firms: SortGraph,
}

#[derive(Default)]
enum SortGraph {
    #[default]
    Unrecorded,
    Recorded(Box<RecordedSort>),
}

struct RecordedSort {
    graph: Graph,
    plan: SortPlan,
    workgroup: u32,
    thread: std::thread::ThreadId,
    stream: StreamId,
    scratch: CubeTensor,
    control: CubeTensor,
    dispatches: CubeTensor,
    #[cfg(test)]
    replays: usize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct SortPlan {
    count: u32,
    source: u32,
    active: u32,
    discovery: crate::simulation::parameters::Discovery,
    scan: parallel::ScanMode,
    keys: SortKeys,
    radix: Radix,
}

impl SortPlan {
    fn new(parameters: &EconomyParameters, keys: SortKeys) -> Self {
        Self::with_radix(parameters, keys, Radix::for_count(parameters.count))
    }

    fn with_radix(parameters: &EconomyParameters, keys: SortKeys, radix: Radix) -> Self {
        Self {
            count: parameters.count,
            source: parameters.sort_source,
            active: parameters.sort_count,
            discovery: parameters.discovery,
            scan: parameters.scan,
            keys,
            radix,
        }
    }

    fn output(self) -> u32 {
        match (self.keys.span(self.count).max(1).ilog2() + 1).div_ceil(self.radix.bits()) % 2 {
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

impl SortCache {
    fn run(
        &mut self,
        device: &Device,
        scratch: &CubeTensor,
        control: &CubeTensor,
        dispatches: &CubeTensor,
        parameters: &EconomyParameters,
        keys: SortKeys,
        radix: Radix,
    ) -> Result<()> {
        let graph = match keys {
            SortKeys::Actors => &mut self.actors,
            SortKeys::Firms { .. } => &mut self.firms,
        };
        graph.run(
            device, scratch, control, dispatches, parameters, keys, radix,
        )
    }
}

impl SortGraph {
    fn run(
        &mut self,
        device: &Device,
        scratch: &CubeTensor,
        control: &CubeTensor,
        dispatches: &CubeTensor,
        parameters: &EconomyParameters,
        keys: SortKeys,
        radix: Radix,
    ) -> Result<()> {
        let plan = SortPlan::with_radix(parameters, keys, radix);
        let histogram = parallel::HistogramScan::BlockLocal;
        match self {
            Self::Recorded(recorded)
                if recorded.compatible(device, scratch, control, dispatches, plan) =>
            {
                unsafe { recorded.graph.replay() }?;
                #[cfg(test)]
                {
                    recorded.replays += 1;
                }
            }
            _ => {
                *self = Self::Unrecorded;
                let client = &device.client;
                client.graph_prepare()?;
                encode_keys(
                    device, scratch, control, dispatches, parameters, histogram, keys, radix,
                );
                client.start_capture()?;
                encode_keys(
                    device, scratch, control, dispatches, parameters, histogram, keys, radix,
                );
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
                    #[cfg(test)]
                    replays: 0,
                }));
            }
        }
        Ok(())
    }
}

impl MarketSort {
    pub fn encode_keys(
        device: &Device,
        scratch: &CubeTensor,
        control: &CubeTensor,
        dispatches: &CubeTensor,
        parameters: &EconomyParameters,
        keys: SortKeys,
    ) -> u32 {
        encode_keys(
            device,
            scratch,
            control,
            dispatches,
            parameters,
            parallel::HistogramScan::BlockLocal,
            keys,
            Radix::for_count(parameters.count),
        );
        SortPlan::new(parameters, keys).output()
    }

    #[cfg(test)]
    pub fn run(
        &mut self,
        device: &Device,
        scratch: &CubeTensor,
        control: &CubeTensor,
        dispatches: &CubeTensor,
        parameters: &EconomyParameters,
    ) -> Result<u32> {
        self.run_keys(
            device,
            scratch,
            control,
            dispatches,
            parameters,
            SortKeys::Actors,
        )
    }

    pub fn run_keys(
        &mut self,
        device: &Device,
        scratch: &CubeTensor,
        control: &CubeTensor,
        dispatches: &CubeTensor,
        parameters: &EconomyParameters,
        keys: SortKeys,
    ) -> Result<u32> {
        self.run_radix(
            device,
            scratch,
            control,
            dispatches,
            parameters,
            keys,
            Radix::for_count(parameters.count),
        )
    }

    fn run_radix(
        &mut self,
        device: &Device,
        scratch: &CubeTensor,
        control: &CubeTensor,
        dispatches: &CubeTensor,
        parameters: &EconomyParameters,
        keys: SortKeys,
        radix: Radix,
    ) -> Result<u32> {
        let plan = SortPlan::with_radix(parameters, keys, radix);
        match self {
            Self::Profiled => encode_keys(
                device,
                scratch,
                control,
                dispatches,
                parameters,
                parallel::HistogramScan::BlockLocal,
                keys,
                radix,
            ),
            Self::Recorded(cache) => {
                cache.run(
                    device, scratch, control, dispatches, parameters, keys, radix,
                )?;
            }
            Self::Unrecorded => {
                let mut cache = Box::<SortCache>::default();
                cache.run(
                    device, scratch, control, dispatches, parameters, keys, radix,
                )?;
                *self = Self::Recorded(cache);
            }
        }
        Ok(plan.output())
    }
}

#[cfg(test)]
fn encode(
    device: &Device,
    scratch: &CubeTensor,
    control: &CubeTensor,
    dispatches: &CubeTensor,
    parameters: &EconomyParameters,
    histogram: parallel::HistogramScan,
) {
    encode_keys(
        device,
        scratch,
        control,
        dispatches,
        parameters,
        histogram,
        SortKeys::Actors,
        Radix::Nibble,
    );
}

fn encode_keys(
    device: &Device,
    scratch: &CubeTensor,
    control: &CubeTensor,
    dispatches: &CubeTensor,
    parameters: &EconomyParameters,
    histogram: parallel::HistogramScan,
    keys: SortKeys,
    radix: Radix,
) {
    let mut parameters = *parameters;
    let count = parameters.count as usize;
    let key_span = keys.span(parameters.count).max(1);
    let key_bits = key_span.ilog2() + 1;
    for shift in (0..key_bits).step_by(radix.bits() as usize) {
        let bits = match radix {
            Radix::Byte => radix.bits().min(key_bits - shift),
            _ => radix.bits(),
        };
        let full_bins = 1u32 << bits;
        let bins = match radix {
            Radix::Byte => {
                full_bins.min(full_bins.min(key_span.div_ceil(1u32 << shift)).div_ceil(16) * 16)
            }
            _ => full_bins,
        };
        let layout = parallel::HistogramLayout {
            bins,
            source_workgroup: radix.layout(device).source_workgroup,
        };
        let histogram_count =
            layout.bins as usize * count.div_ceil(layout.source_workgroup as usize);
        let histogram_items = parallel::ActiveScanTile::new(device, histogram_count).items();
        let histogram_span = device.workgroup * histogram_items;
        if radix != Radix::Nibble && (shift == 0 || bins != (1u32 << radix.bits())) {
            prepare_sort_dispatch(device, control, dispatches, layout, histogram_span);
        }
        parameters.sort_width = shift;
        let launch = |stage| {
            RadixSort::launch(
                device,
                markets::dispatch(
                    dispatches,
                    markets::SORT_DISPATCH,
                    count.div_ceil(layout.source_workgroup as usize),
                ),
                CubeDim::new_1d(layout.source_workgroup),
                SortInputLaunch {
                    scratch: scratch.buffer_arg(),
                    control: control.buffer_arg(),
                    parameters: parameters.launch(),
                    stage,
                    histogram,
                    digit_bits: bits,
                    histogram_span,
                    histogram_bound: histogram_count as u32,
                    histogram_bins: bins,
                    key_base: keys.base(),
                },
            );
        };
        launch(SortStage::Local);
        match radix {
            Radix::Nibble => parallel::scan_active(
                device,
                scratch,
                7 * count,
                histogram_count,
                control,
                dispatches,
                histogram,
            ),
            _ => parallel::scan_histogram(
                device,
                scratch,
                7 * count,
                histogram_count,
                control,
                dispatches,
                histogram,
                layout,
            ),
        }
        launch(SortStage::Scatter);
        parameters.sort_source = 2 * parameters.count - parameters.sort_source;
    }
    if radix != Radix::Nibble {
        let native = Radix::Nibble.layout(device);
        let bound = native.bins as usize * count.div_ceil(native.source_workgroup as usize);
        let span = device.workgroup * parallel::ActiveScanTile::new(device, bound).items();
        prepare_sort_dispatch(device, control, dispatches, native, span);
    }
}

#[derive(CubeType, CubeLaunch)]
struct SortDispatchInput {
    control: Box<[Atomic<u32>]>,
    dispatches: Box<[u32]>,
    bins: u32,
    source_workgroup: u32,
    histogram_span: u32,
}

struct SortDispatch;
struct PrepareSort;

impl System for SortDispatch {
    type Input = SortDispatchInput;
    type Passes = gpu_passes![PrepareSort];
}

#[cube]
impl GpuPass<SortDispatchInput> for PrepareSort {
    type Output = ();

    fn execute(mut input: SortDispatchInput) {
        let active = input.control[crate::simulation::flags::CONSTRUCTED_EVENTS_WORD].load();
        let groups = ceildiv(active, input.source_workgroup);
        markets::store_dispatch(
            &mut input.dispatches,
            comptime!(markets::SORT_DISPATCH),
            groups,
        );
        let mut count = input.bins * groups;
        for level in 0..6usize {
            let groups = ceildiv(count, input.histogram_span);
            markets::store_dispatch(
                &mut input.dispatches,
                comptime!(markets::HISTOGRAM_DISPATCH) + level * 3,
                groups,
            );
            count = groups;
        }
    }
}

fn prepare_sort_dispatch(
    device: &Device,
    control: &CubeTensor,
    dispatches: &CubeTensor,
    layout: parallel::HistogramLayout,
    histogram_span: u32,
) {
    SortDispatch::launch_single(
        device,
        SortDispatchInputLaunch {
            control: control.buffer_arg(),
            dispatches: dispatches.buffer_arg(),
            bins: layout.bins,
            source_workgroup: layout.source_workgroup,
            histogram_span,
        },
    );
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
    #[cube(comptime)]
    histogram: parallel::HistogramScan,
    #[cube(comptime)]
    digit_bits: u32,
    #[cube(comptime)]
    histogram_span: u32,
    histogram_bound: u32,
    histogram_bins: u32,
    key_base: u32,
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
                SortStage::Local => markets::radix_local_digits(
                    &mut input.scratch,
                    &input.control,
                    &input.parameters,
                    ABSOLUTE_POS as u32,
                    input.key_base,
                    comptime!(input.digit_bits),
                    input.histogram_bins,
                ),
                SortStage::Scatter => markets::radix_scatter_digits(
                    &mut input.scratch,
                    &input.control,
                    &input.parameters,
                    ABSOLUTE_POS as u32,
                    input.key_base,
                    comptime!(input.histogram),
                    comptime!(input.digit_bits),
                    comptime!(input.histogram_span),
                    input.histogram_bound,
                ),
            }
        }
    }
}

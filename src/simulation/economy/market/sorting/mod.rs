use super::orders;
use crate::simulation::{
    Device,
    economy::parameters::EconomyParameters,
    gpu::tensor::same_buffer,
    parallel,
    pass::GpuPass,
    storage::words::ceildiv,
    system::{AsBufferArg, System, gpu_passes},
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
        home_mode: crate::simulation::economy::configuration::Feature,
    ) -> Self {
        use crate::domain::{Good, Market};
        match market {
            Market::Labor => Self::Firms { households },
            Market::Input(Good::Grain) | Market::Consumer(Good::Grain)
                if home_mode == crate::simulation::economy::configuration::Feature::Enabled =>
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
    discovery: crate::simulation::economy::parameters::Discovery,
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

pub(in crate::simulation) const INPUT_TILE: usize = 2048;

pub(in crate::simulation) fn sort_input_tile(
    device: &Device,
    searches: &CubeTensor,
    proposals: &CubeTensor,
    scratch: &CubeTensor,
    control: &CubeTensor,
    clock: &CubeTensor,
    actors: u32,
) {
    input_tile::launch(
        &device.client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(128),
        searches.buffer_arg(),
        proposals.buffer_arg(),
        scratch.buffer_arg(),
        control.buffer_arg(),
        clock.buffer_arg(),
        actors,
    );
}

#[cube(launch)]
fn input_tile(
    searches: &[u32],
    proposals: &[u32],
    scratch: &mut [u32],
    control: &[Atomic<u32>],
    clock: &[u32],
    count: u32,
) {
    let orders = control[crate::simulation::storage::flags::CONSTRUCTED_EVENTS_WORD].load();
    let rotation = clock[crate::domain::Good::COUNT + 1usize];
    let length = 1u32 << (32 - (max(orders, 1) - 1).leading_zeros());
    let mut keys = Shared::<[u64]>::new_slice(INPUT_TILE);
    let mut slot = UNIT_POS;
    while slot < length {
        let mut key = u64::cast_from(-1i64);
        if slot < orders {
            let actor =
                searches[(comptime!(crate::simulation::economy::market::search::INPUT_BUYERS_FIELD)
                    * count
                    + slot) as usize];
            let peer = proposals[actor as usize
                * crate::simulation::storage::layout::ProposalLayout::WORDS
                + crate::simulation::storage::layout::ProposalLayout::ACTOR];
            crate::simulation::storage::flags::KernelEventStatus::new(comptime!(
                crate::simulation::storage::flags::EventStatus::Inactive
            ))
            .store(scratch, 4 * count + actor);
            let priority = select(
                actor >= rotation,
                actor - rotation,
                actor + count - rotation,
            );
            key = (u64::cast_from(peer) << 32) | u64::cast_from(priority);
        }
        keys[slot as usize] = key;
        slot += CUBE_DIM;
    }
    sync_cube();
    let mut width = 2u32;
    while width <= length {
        let mut step = width / 2;
        while step > 0 {
            let mut pair = UNIT_POS;
            while pair < length / 2 {
                let left = 2 * pair - (pair & (step - 1));
                let right = left + step;
                let a = keys[left as usize];
                let b = keys[right as usize];
                let ascending = left & width == 0;
                keys[left as usize] = select(ascending, min(a, b), max(a, b));
                keys[right as usize] = select(ascending, max(a, b), min(a, b));
                pair += CUBE_DIM;
            }
            sync_cube();
            step /= 2;
        }
        width *= 2;
    }
    let mut slot = UNIT_POS;
    while slot < orders {
        let key = keys[slot as usize];
        let peer = u32::cast_from(key >> 32);
        let rotated = u32::cast_from(key) + rotation;
        let previous = keys[(max(slot, 1) - 1) as usize];
        let first = slot == 0 || u32::cast_from(previous >> 32) != peer;
        scratch[slot as usize] = peer;
        scratch[(count + slot) as usize] = select(rotated < count, rotated, rotated - count);
        scratch[(5 * count + slot) as usize] = select(first, slot, u32::MAX);
        slot += CUBE_DIM;
    }
    if UNIT_POS == 0 {
        control[orders::RANGE_SLOTS_WORD].store(orders);
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
    let capacity = match (parameters.discovery, parameters.sort_count) {
        (crate::simulation::economy::parameters::Discovery::Indexed, 1..) => {
            parameters.sort_count as usize
        }
        _ => count,
    };
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
            layout.bins as usize * capacity.div_ceil(layout.source_workgroup as usize);
        let histogram_items = parallel::ActiveScanTile::new(device, histogram_count).items();
        let histogram_span = device.workgroup * histogram_items;
        if radix != Radix::Nibble && (shift == 0 || bins != (1u32 << radix.bits())) {
            prepare_sort_dispatch(device, control, dispatches, layout, histogram_span);
        }
        parameters.sort_width = shift;
        let launch = |stage| {
            let input = SortInputLaunch {
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
            };
            let groups = capacity.div_ceil(layout.source_workgroup as usize);
            let dim = CubeDim::new_1d(layout.source_workgroup);
            match (&device.cube, groups) {
                (cubecl::Device::Metal(_), 257..) => {
                    sort_blocks::launch(&device.client, CubeCount::Static(256, 1, 1), dim, input)
                }
                _ => RadixSort::launch(
                    device,
                    orders::dispatch(dispatches, orders::SORT_DISPATCH, groups),
                    dim,
                    input,
                ),
            }
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
        let active =
            input.control[crate::simulation::storage::flags::CONSTRUCTED_EVENTS_WORD].load();
        let groups = ceildiv(active, input.source_workgroup);
        orders::store_dispatch(
            &mut input.dispatches,
            comptime!(orders::SORT_DISPATCH),
            groups,
        );
        let mut count = input.bins * groups;
        for level in 0..6usize {
            let groups = ceildiv(count, input.histogram_span);
            orders::store_dispatch(
                &mut input.dispatches,
                comptime!(orders::HISTOGRAM_DISPATCH) + level * 3,
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
        if orders::faults_clear(&input.control) {
            input.sort(ABSOLUTE_POS as u32);
        }
    }
}

#[cube(launch)]
fn sort_blocks(mut input: SortInput) {
    if orders::faults_clear(&input.control) {
        let groups = ceildiv(orders::active(&input.control, &input.parameters), CUBE_DIM);
        let mut group = CUBE_POS as u32;
        while group < groups {
            input.sort(group * CUBE_DIM + UNIT_POS);
            if comptime!(input.stage == SortStage::Local) {
                sync_cube();
            }
            group += CUBE_COUNT as u32;
        }
    }
}

#[cube]
impl SortInput {
    fn sort(&mut self, id: u32) {
        match comptime!(self.stage) {
            SortStage::Local => orders::radix_local_digits(
                &mut self.scratch,
                &self.control,
                &self.parameters,
                id,
                self.key_base,
                comptime!(self.digit_bits),
                self.histogram_bins,
            ),
            SortStage::Scatter => orders::radix_scatter_digits(
                &mut self.scratch,
                &self.control,
                &self.parameters,
                id,
                self.key_base,
                comptime!(self.histogram),
                comptime!(self.digit_bits),
                comptime!(self.histogram_span),
                self.histogram_bound,
            ),
        }
    }
}

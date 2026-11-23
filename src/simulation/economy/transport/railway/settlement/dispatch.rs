use super::*;

#[derive(CubeType, CubeLaunch)]
struct DispatchInput {
    control: Box<[Atomic<u32>]>,
    dispatches: Box<[u32]>,
    workgroup: u32,
    config: EconomyConfig,
    parameters: FreightParameters,
    spatial: Box<[u32]>,
    #[cube(comptime)]
    resource_selection: ResourceSelection,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ResourceSelection {
    AllFirms,
    MarketSuppliers,
}

struct FreightDispatch;
struct PrepareDispatch;

impl System for FreightDispatch {
    type Input = DispatchInput;
    type Passes = gpu_passes![PrepareDispatch];
}

#[cube]
impl GpuPass<DispatchInput> for PrepareDispatch {
    type Output = ();

    fn execute(mut input: DispatchInput) {
        input.parameters.refresh(&input.config);
        let active = input.control[CONSTRUCTED_EVENTS_WORD].load();
        let layout = FreightParticipants::new(
            &input.config,
            &input.spatial,
            input.parameters.good,
            active,
            comptime!(input.resource_selection),
        );
        let resources = layout.sellers + layout.carriers + input.config.railway.region_count;
        orders::store_dispatch(
            &mut input.dispatches,
            RESOURCE_DISPATCH,
            ceildiv(max(resources, 1), input.workgroup),
        );
        orders::store_dispatch(
            &mut input.dispatches,
            INITIALIZE_DISPATCH,
            ceildiv(max(max(resources, active), 1), input.workgroup),
        );
    }
}

#[derive(CubeType, CubeLaunch)]
struct QueueInput {
    progress: Box<[Atomic<u32>]>,
    control: Box<[Atomic<u32>]>,
    dispatches: Box<[u32]>,
    workgroup: u32,
    #[cube(comptime)]
    execution: Execution,
}

struct FreightQueue;
struct PrepareQueue;

impl System for FreightQueue {
    type Input = QueueInput;
    type Passes = gpu_passes![PrepareQueue];
}

#[cube]
impl GpuPass<QueueInput> for PrepareQueue {
    type Output = ();

    fn execute(mut input: QueueInput) {
        let mut groups = 0u32;
        let enabled =
            execution_enabled(&input.progress, &input.control, comptime!(input.execution));
        if enabled {
            let count = input.progress[comptime!(input.execution.wave() as usize)].load();
            input.progress[CLAIM_COUNT].store(0);
            input.progress[CLAIM_COUNT + 1].store(0);
            input.progress[CLAIM_CURSOR].store(0);
            groups = ceildiv(count, input.workgroup);
        }
        orders::store_dispatch(&mut input.dispatches, BOOKING_DISPATCH, groups);
        if comptime!(input.execution == Execution::Buffered(Wave::Retry)) {
            for word in 0..3usize {
                let empty = select(word == 0, 0u32, 1u32);
                input.dispatches[RETRY_RESOURCE_DISPATCH + word] =
                    select(enabled, input.dispatches[RESOURCE_DISPATCH + word], empty);
            }
        }
    }
}

struct FreightSettlement;
pub(super) struct SettleFreight;

impl System for FreightSettlement {
    type Input = FreightSettlementInput;
    type Passes = gpu_passes![SettleFreight];
}

#[cfg(test)]
pub fn launch(
    economy: DeviceEconomy<'_>,
    spatial: &CubeTensor,
    proposals: &CubeTensor,
    scratch: &CubeTensor,
    control: &CubeTensor,
    parameters: &crate::simulation::economy::parameters::EconomyParameters,
    stage: Stage,
) {
    launch_buffered(
        economy,
        spatial,
        proposals,
        scratch,
        control,
        parameters,
        stage,
        Execution::Immediate(Wave::Initial),
        ResourceSelection::AllFirms,
    );
}

pub fn launch_buffered(
    DeviceEconomy {
        device,
        state,
        actors,
        config,
    }: DeviceEconomy<'_>,
    spatial: &CubeTensor,
    proposals: &CubeTensor,
    scratch: &CubeTensor,
    control: &CubeTensor,
    parameters: &crate::simulation::economy::parameters::EconomyParameters,
    stage: Stage,
    execution: Execution,
    resource_selection: ResourceSelection,
) -> usize {
    let buffers = &config.railway.freight;
    if let Stage::Snapshot = stage {
        FreightDispatch::launch_single(
            device,
            DispatchInputLaunch {
                control: control.buffer_arg(),
                dispatches: buffers.dispatches.buffer_arg(),
                workgroup: device.workgroup,
                config: config.launch(),
                parameters: parameters.into(),
                spatial: spatial.buffer_arg(),
                resource_selection,
            },
        );
    }
    match stage {
        Stage::SortClaims => claims::sort_buffered(device, buffers, control, execution),
        Stage::AdmitClaims => {
            admit_claims(device, buffers, control, execution);
            1
        }
        Stage::Queue => {
            FreightQueue::launch_single(
                device,
                QueueInputLaunch {
                    progress: buffers.progress.buffer_arg(),
                    control: control.buffer_arg(),
                    dispatches: buffers.dispatches.buffer_arg(),
                    workgroup: device.workgroup,
                    execution,
                },
            );
            1
        }
        _ => {
            let input = FreightSettlementInputLaunch {
                state: state.buffer_arg(),
                actors: actors.buffer_arg(),
                spatial: spatial.buffer_arg(),
                config: config.launch(),
                proposals: proposals.buffer_arg(),
                scratch: scratch.buffer_arg(),
                control: control.buffer_arg(),
                resources: buffers.resources.buffer_arg(),
                accounts: buffers.accounts.buffer_arg(),
                bookings: buffers.bookings.buffer_arg(),
                queue: buffers.queue.buffer_arg(),
                claims: buffers.claims.buffer_arg(),
                claim_source: claims::settlement_source(buffers),
                claim_offsets: claims::offsets_start(buffers.claim_capacity),
                progress: buffers.progress.buffer_arg(),
                parameters: parameters.into(),
                stage,
                resource_selection,
                execution,
            };
            let dimensions = CubeDim::new_1d(device.workgroup);
            match settlement_dispatch(device, config, parameters, stage, execution) {
                crate::simulation::system::Dispatch::Bounded {
                    counts,
                    offset,
                    max_groups,
                } => {
                    let groups = max_groups.clamp(1, 32);
                    strided_freight::launch(
                        &device.client,
                        CubeCount::Static(groups, 1, 1),
                        dimensions,
                        input,
                        counts,
                        offset,
                    );
                }
                grid => FreightSettlement::launch(device, grid, dimensions, input),
            }
            1 + usize::from(stage == Stage::Snapshot)
        }
    }
}

fn settlement_dispatch(
    device: &Device,
    config: &Configuration,
    parameters: &crate::simulation::economy::parameters::EconomyParameters,
    stage: Stage,
    execution: Execution,
) -> crate::simulation::system::Dispatch {
    let buffers = &config.railway.freight;
    let active_groups = parameters.count.div_ceil(device.workgroup) as usize;
    let seller_bound = match parameters.good {
        good if good == Good::Bread as u32 || good == Good::Grain as u32 => {
            parameters.count as usize
        }
        _ => 0,
    };
    let resource_groups =
        (config.railway.firms as usize + config.railway.region_count as usize + seller_bound)
            .div_ceil(device.workgroup as usize);
    let initialize_groups = resource_groups.max(active_groups);
    match (execution, stage) {
        (Execution::Buffered(_), Stage::EmitClaims) => {
            orders::dispatch(&buffers.dispatches, CLAIM_ACTIVE_DISPATCH, active_groups)
        }
        (Execution::Buffered(_), Stage::Resolve) => orders::dispatch(
            &buffers.dispatches,
            CLAIM_RESOURCE_DISPATCH,
            resource_groups,
        ),
        (_, Stage::Propose | Stage::EmitClaims | Stage::Commit) => {
            orders::dispatch(&buffers.dispatches, BOOKING_DISPATCH, active_groups)
        }
        (Execution::Buffered(Wave::Retry), Stage::Reset | Stage::CountClaims | Stage::Release) => {
            orders::dispatch(
                &buffers.dispatches,
                RETRY_RESOURCE_DISPATCH,
                resource_groups,
            )
        }
        (_, Stage::Initialize) => {
            orders::dispatch(&buffers.dispatches, INITIALIZE_DISPATCH, initialize_groups)
        }
        (_, Stage::ClaimOffsets) => device
            .count((config.actors + config.railway.region_count) as usize + 1)
            .into(),
        _ => orders::dispatch(&buffers.dispatches, RESOURCE_DISPATCH, resource_groups),
    }
}

#[derive(CubeType)]
struct FreightParticipants {
    indexed: bool,
    sellers: u32,
    carriers: u32,
}

#[cube]
impl FreightParticipants {
    fn new(
        config: &EconomyConfig,
        spatial: &[u32],
        good: u32,
        active: u32,
        #[comptime] selection: ResourceSelection,
    ) -> Self {
        let cells = config.spatial_side * config.spatial_side;
        let base = 2 * (cells + 1 + config.actors);
        let firms = spatial[(base + cells) as usize];
        let home = good == comptime!(Good::Bread as u32) || good == comptime!(Good::Grain as u32);
        let suppliers = spatial[cells as usize];
        let indexed = comptime!(selection == ResourceSelection::MarketSuppliers)
            && !home
            && good != comptime!(Good::Railway as u32)
            && suppliers + config.railway.company_count < firms;
        FreightParticipants {
            indexed,
            sellers: select(indexed, suppliers, firms),
            carriers: select(
                indexed,
                config.railway.company_count,
                select(home, active, 0),
            ),
        }
    }
}

#[derive(CubeType, CubeLaunch)]
struct AdmissionInput {
    workgroup: u32,
    tile: u32,
    control: Box<[Atomic<u32>]>,
    progress: Box<[Atomic<u32>]>,
    dispatches: Box<[u32]>,
    capacity: u32,
    #[cube(comptime)]
    execution: Execution,
}

pub(super) fn admit_claims(
    device: &Device,
    buffers: &Buffers,
    control: &CubeTensor,
    execution: Execution,
) {
    FreightAdmission::launch_single(
        device,
        AdmissionInputLaunch {
            control: control.buffer_arg(),
            progress: buffers.progress.buffer_arg(),
            dispatches: buffers.dispatches.buffer_arg(),
            capacity: buffers.claim_capacity as u32,
            workgroup: device.workgroup,
            tile: buffers.sorting.tile() as u32,
            execution,
        },
    );
}

struct FreightAdmission;
struct AdmitFreightClaims;

impl System for FreightAdmission {
    type Input = AdmissionInput;
    type Passes = gpu_passes![AdmitFreightClaims];
}

#[cube]
impl GpuPass<AdmissionInput> for AdmitFreightClaims {
    type Output = ();

    fn execute(mut input: AdmissionInput) {
        let mut count = 0u32;
        if execution_enabled(&input.progress, &input.control, comptime!(input.execution)) {
            let demand = AtomicBalance::at(&input.progress, CLAIM_COUNT).load();
            count = demand.extract(LOW_WORD);
            if comptime!(matches!(input.execution, Execution::Buffered(_)))
                && lt(ticks(input.capacity, 0), demand)
            {
                input.control[crate::simulation::storage::flags::CLAIM_CAPACITY_LOW_WORD]
                    .store(count);
                input.control[crate::simulation::storage::flags::CLAIM_CAPACITY_HIGH_WORD]
                    .store(demand.extract(HIGH_WORD));
                input.control[crate::simulation::storage::flags::FAULT_WORD]
                    .fetch_or(comptime!(FaultFlags::CLAIM_CAPACITY.bits()));
                AtomicBalance::at(&input.progress, CLAIM_COUNT).store(zero());
                count = 0;
            }
        }
        orders::store_dispatch(
            &mut input.dispatches,
            BLOCK_SORT_DISPATCH,
            ceildiv(count, input.tile),
        );
        orders::store_dispatch(
            &mut input.dispatches,
            MERGE_SORT_DISPATCH,
            ceildiv(count, input.workgroup),
        );
        if comptime!(matches!(input.execution, Execution::Buffered(_))) {
            for word in 0..3usize {
                let empty = select(word == 0, 0u32, 1u32);
                let mut active = empty;
                let mut resources = empty;
                if count > 0 {
                    active = input.dispatches[BOOKING_DISPATCH + word];
                    resources = input.dispatches[RESOURCE_DISPATCH + word];
                }
                input.dispatches[CLAIM_ACTIVE_DISPATCH + word] = active;
                input.dispatches[CLAIM_RESOURCE_DISPATCH + word] = resources;
            }
        }
    }
}

#[cube(launch)]
fn strided_freight(mut input: FreightSettlementInput, counts: &[u32], offset: u32) {
    input.parameters.refresh(&input.config);
    let groups = counts[offset as usize] * counts[offset as usize + 1];
    let mut group = CUBE_POS as u32;
    if execution_enabled(&input.progress, &input.control, comptime!(input.execution)) {
        while group < groups {
            input.settle_freight(group * CUBE_DIM + UNIT_POS);
            group += CUBE_COUNT as u32;
        }
    }
}

#[cube]
impl FreightSettlementInput {
    pub(super) fn resource(&self, index: u32) -> u32 {
        let actors = ActorTable::new(&self.actors);
        let cells = self.config.spatial_side * self.config.spatial_side;
        let base = 2 * (cells + 1 + self.config.actors);
        let active = self.parameters.active(&self.control);
        let layout = FreightParticipants::new(
            &self.config,
            &self.spatial,
            self.parameters.good,
            active,
            comptime!(self.resource_selection),
        );
        let mut resource = u32::new(comptime!(i64::from(u32::MAX)));
        if index < layout.sellers {
            resource = self.spatial[(select(layout.indexed, 0, base) + cells + 1 + index) as usize];
        } else if index < layout.sellers + layout.carriers {
            let slot = index - layout.sellers;
            if layout.indexed {
                resource = self.config.railway.companies[slot as usize * COMPANY_WORDS];
            } else {
                let seller = self.scratch[(self.parameters.sort_source + slot) as usize];
                let mut first = true;
                if slot > 0 {
                    first =
                        self.scratch[(self.parameters.sort_source + slot - 1) as usize] != seller;
                }
                if first && actors.kind(seller).is(comptime!(ActorKind::Household)) {
                    resource = seller;
                }
            }
        } else {
            resource = self.config.actors + index - layout.sellers - layout.carriers;
        }
        resource
    }
}

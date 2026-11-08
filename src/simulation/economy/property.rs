use super::*;
use crate::domain::{
    enterprise::EnterpriseState,
    property::{ProjectStatus, PropertyUse},
};
use crate::simulation::{Device, configuration::Configuration};
use burn_wgpu::CubeTensor;

#[cfg(test)]
mod tests;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum PropertyStage {
    Initialize,
    Allocate,
    Construct,
    Consolidate,
}

#[repr(u32)]
enum ConsolidationState {
    Ineligible,
    Building,
    Ready,
}

#[derive(CubeType, CubeLaunch)]
struct PropertyInput {
    state: Box<[u32]>,
    actors: Box<[u32]>,
    spatial: Box<[u32]>,
    config: EconomyConfig,
    control: Box<[Atomic<u32>]>,
    #[cube(comptime)]
    stage: PropertyStage,
}

struct PropertySystem;
struct AllocateProperty;

#[derive(CubeType, CubeLaunch)]
struct ParcelIndexInput {
    actors: Box<[u32]>,
    spatial: Box<[u32]>,
    parcels: Box<[u32]>,
    count: u32,
    source: u32,
    side: u32,
    size: u32,
}

struct ParcelIndex;
struct IndexParcels;

impl System for ParcelIndex {
    type Input = ParcelIndexInput;
    type Passes = gpu_passes![IndexParcels];
}

#[cube]
impl GpuPass<ParcelIndexInput> for IndexParcels {
    type Output = ();

    fn execute(mut input: ParcelIndexInput) {
        let id = ABSOLUTE_POS as u32;
        let cells = input.side * input.side;
        if id < input.count {
            input.parcels[(cells + 1 + id) as usize] = input.spatial[(input.source + id) as usize];
        }
        if id <= cells {
            let mut low = 0;
            let mut high = input.count;
            while low < high {
                let middle = low + (high - low) / 2;
                let actor = input.spatial[(input.source + middle) as usize];
                let x = input.actors[(ActorLayout::X as u32 * input.count + actor) as usize];
                let y = input.actors[(ActorLayout::Y as u32 * input.count + actor) as usize];
                let before = y / input.size * input.side + x / input.size < id;
                low = select(before, middle + 1, low);
                high = select(before, high, middle);
            }
            input.parcels[id as usize] = low;
        }
    }
}

fn parcel_index(
    device: &Device,
    actors: &CubeTensor,
    spatial: &CubeTensor,
    config: &Configuration,
) -> CubeTensor {
    let size = config.map_size.div_ceil(128);
    let side = config.map_size.div_ceil(size);
    let cells = side * side;
    let parcels = device.empty((cells + 1 + config.actors) as usize);
    ParcelIndex::launch(
        device,
        device.count(config.actors.max(cells + 1) as usize),
        CubeDim::new_1d(device.workgroup),
        ParcelIndexInputLaunch {
            actors: actors.clone().into_buffer_arg(),
            spatial: spatial.clone().into_buffer_arg(),
            parcels: parcels.clone().into_buffer_arg(),
            count: config.actors,
            source: 3 * (config.spatial_side * config.spatial_side + 1 + config.actors)
                + config.actors.div_ceil(DISCOVERY_BLOCK)
                + 1,
            side,
            size,
        },
    );
    parcels
}

impl System for PropertySystem {
    type Input = PropertyInput;
    type Passes = gpu_passes![AllocateProperty];
}

#[cube]
impl GpuPass<PropertyInput> for AllocateProperty {
    type Output = ();

    fn execute(mut input: PropertyInput) {
        match comptime!(input.stage) {
            PropertyStage::Initialize => initialize(
                &mut input.state,
                &input.actors,
                &input.config,
                ABSOLUTE_POS as u32,
            ),
            PropertyStage::Allocate => allocate(
                &mut input.state,
                &input.actors,
                &input.config,
                &input.control,
                ABSOLUTE_POS as u32,
            ),
            PropertyStage::Construct => construct(
                &mut input.state,
                &input.actors,
                &input.config,
                &input.control,
                ABSOLUTE_POS as u32,
            ),
            PropertyStage::Consolidate => consolidate(
                &mut input.state,
                &input.actors,
                &input.spatial,
                &input.config,
                &input.control,
            ),
        }
    }
}

pub(in crate::simulation) fn launch(
    device: &Device,
    state: &CubeTensor,
    actors: &CubeTensor,
    spatial: &CubeTensor,
    config: &Configuration,
    control: &CubeTensor,
    phase: Phase,
) {
    let stages: &[PropertyStage] = match (config.property_mode, phase) {
        (Feature::Enabled, Phase::Opening) => &[PropertyStage::Initialize, PropertyStage::Allocate],
        (Feature::Enabled, Phase::PhysicalClose) => &[PropertyStage::Construct],
        (Feature::Disabled, Phase::PhysicalClose) if config.rental_businesses > 0 => {
            &[PropertyStage::Construct]
        }
        _ => &[],
    };
    let consolidation = match (
        config.property_mode,
        phase,
        config.economy,
        config.home_mode,
    ) {
        (
            Feature::Disabled,
            Phase::AccountsDecisions,
            crate::domain::Economy::Industrial,
            Feature::Enabled,
        ) => Some(PropertyStage::Consolidate),
        _ => None,
    };
    for stage in stages.iter().copied().chain(consolidation) {
        let count = match stage {
            PropertyStage::Allocate => config.property_regions.meta.num_elements() / 6,
            PropertyStage::Consolidate => 1,
            _ => config.actors as usize,
        };
        let workgroup = match stage {
            PropertyStage::Consolidate => 256,
            _ => device.workgroup,
        };
        let spatial = match stage {
            PropertyStage::Consolidate => parcel_index(device, actors, spatial, config),
            _ => spatial.clone(),
        };
        PropertySystem::launch(
            device,
            device.count(count),
            CubeDim::new_1d(workgroup),
            PropertyInputLaunch {
                state: state.clone().into_buffer_arg(),
                actors: actors.clone().into_buffer_arg(),
                spatial: spatial.clone().into_buffer_arg(),
                config: config.launch(),
                control: control.clone().into_buffer_arg(),
                stage,
            },
        );
    }
}

#[cube]
fn consolidate(
    state: &mut [u32],
    actors: &[u32],
    spatial: &[u32],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
) {
    for index in 0..config.grain_farms.len() - 1 {
        let farm = config.grain_farms[index];
        let output = number(state, config, comptime!(Field::OutputTicks as u32), farm);
        let labor = number(state, config, comptime!(Field::LaborMinutes as u32), farm);
        let operating = number(
            state,
            config,
            comptime!(Field::EnterpriseStatus as u32),
            farm,
        ) == comptime!(EnterpriseState::Operating as u32);
        let producer = number(
            state,
            config,
            comptime!(Field::EnterpriseModel as u32),
            farm,
        ) == 0;
        let completed = number(
            state,
            config,
            comptime!(Field::ConstructionRemaining as u32),
            farm,
        ) == 0;
        let required = number(
            state,
            config,
            comptime!(Field::PendingCapacity as u32),
            farm,
        );
        let status = select(
            operating
                && producer
                && output * 365 > config.home.annual_yield_per_plot
                && output * config.home.cultivation_minutes > labor,
            select(
                completed,
                comptime!(ConsolidationState::Ready as u32),
                comptime!(ConsolidationState::Building as u32),
            ),
            comptime!(ConsolidationState::Ineligible as u32),
        );
        if status == comptime!(ConsolidationState::Ready as u32) {
            let replacement = divide(
                add_checked(
                    control,
                    times32(required, output * 365),
                    ticks(config.home.annual_yield_per_plot - 1, 0),
                ),
                config.home.annual_yield_per_plot,
            );
            consolidate_farm(
                state,
                actors,
                spatial,
                config,
                control,
                farm,
                replacement.extract(LOW_WORD),
            );
        }
    }
}

#[cube]
fn sum_plots(amount: u64) -> u64 {
    let mut plots = Shared::<[u64]>::new_slice(256usize);
    plots[UNIT_POS as usize] = amount;
    sync_cube();
    let mut stride = CUBE_DIM / 2;
    while stride > 0 {
        if UNIT_POS < stride {
            plots[UNIT_POS as usize] += plots[(UNIT_POS + stride) as usize];
        }
        sync_cube();
        stride /= 2;
    }
    plots[0usize]
}

#[cube]
fn prefix_plots(amount: u64) -> u64 {
    let mut plots = Shared::<[u64]>::new_slice(256usize);
    plots[UNIT_POS as usize] = amount;
    sync_cube();
    let mut stride = 1;
    while stride < CUBE_DIM {
        let index = (UNIT_POS + 1) * stride * 2 - 1;
        if index < CUBE_DIM {
            plots[index as usize] += plots[(index - stride) as usize];
        }
        sync_cube();
        stride *= 2;
    }
    stride = CUBE_DIM / 4;
    while stride > 0 {
        let index = (UNIT_POS + 1) * stride * 2 - 1;
        if index + stride < CUBE_DIM {
            plots[(index + stride) as usize] += plots[index as usize];
        }
        sync_cube();
        stride /= 2;
    }
    plots[UNIT_POS as usize]
}

#[cube]
fn local_plots(state: &mut [u32], actors: &[u32], config: &EconomyConfig, household: u32) -> u64 {
    let plots = number(state, config, comptime!(Field::LandPlots as u32), household);
    u64::cast_from(select(
        kind(actors, household).is(comptime!(ActorKind::Household)),
        plots,
        0,
    ))
}

#[derive(CubeType)]
struct LandNeighborhood {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    side: u32,
}

#[cube]
impl LandNeighborhood {
    fn new(actors: &[u32], config: &EconomyConfig, farm: u32, radius: u32) -> Self {
        let stride = actors.len() / ActorLayout::WORDS;
        let x = actors[ActorLayout::X * stride + farm as usize];
        let y = actors[ActorLayout::Y * stride + farm as usize];
        let size = ceildiv(config.map_size, 128);
        let side = ceildiv(config.map_size, size);
        let lower_x = (max(x, radius) - radius) / size;
        let lower_y = (max(y, radius) - radius) / size;
        let upper_x = min((x + radius) / size, side - 1);
        let upper_y = min((y + radius) / size, side - 1);
        LandNeighborhood {
            x: lower_x,
            y: lower_y,
            width: upper_x - lower_x + 1,
            height: upper_y - lower_y + 1,
            side,
        }
    }

    fn cell(&self, index: u32) -> u32 {
        (self.y + index / self.width) * self.side + self.x + index % self.width
    }

    fn actor(&self, spatial: &[u32], index: u32) -> u32 {
        spatial[(self.side * self.side + 1 + index) as usize]
    }

    fn lower_bound(&self, spatial: &[u32], cell: u32, actor: u32) -> u32 {
        let mut low = spatial[cell as usize];
        let mut high = spatial[cell as usize + 1];
        while low < high {
            let middle = low + (high - low) / 2;
            let before = self.actor(spatial, middle) < actor;
            low = select(before, middle + 1, low);
            high = select(before, high, middle);
        }
        low
    }
}

#[cube]
fn nearby_plots(
    state: &mut [u32],
    actors: &[u32],
    spatial: &[u32],
    config: &EconomyConfig,
    farm: u32,
    radius: u32,
) -> u64 {
    let neighborhood = LandNeighborhood::new(actors, config, farm, radius);
    let mut plots = u64::new(0);
    for index in 0..neighborhood.width * neighborhood.height {
        let cell = neighborhood.cell(index);
        let mut member = spatial[cell as usize] + UNIT_POS;
        let end = spatial[cell as usize + 1];
        while member < end {
            let household = neighborhood.actor(spatial, member);
            let grant = local_plots(state, actors, config, household);
            plots += select(
                distance(actors, farm, household) <= radius,
                grant,
                u64::new(0),
            );
            member += CUBE_DIM;
        }
    }
    sum_plots(plots)
}

#[cube]
fn consolidate_farm(
    state: &mut [u32],
    actors: &[u32],
    spatial: &[u32],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
    farm: u32,
    required: u32,
) {
    if required > 0 {
        let mut low = 0;
        let mut high = min(1u32, config.search_radius);
        let mut available = nearby_plots(state, actors, spatial, config, farm, high);
        while high < config.search_radius && available < u64::cast_from(required) {
            low = high + 1;
            high = min(high * 2, config.search_radius);
            available = nearby_plots(state, actors, spatial, config, farm, high);
        }
        while low < high {
            let radius = low + (high - low) / 2;
            let plots = nearby_plots(state, actors, spatial, config, farm, radius);
            let short = plots < u64::cast_from(required);
            low = select(short, radius + 1, low);
            high = select(short, high, radius);
        }
        let radius = low;
        let neighborhood = LandNeighborhood::new(actors, config, farm, radius);
        let block = ceildiv(config.actors, CUBE_DIM);
        let start = UNIT_POS * block;
        let end = min(start + block, config.actors);
        let mut closer = u64::new(0);
        let mut boundary = u64::new(0);
        for index in 0..neighborhood.width * neighborhood.height {
            let cell = neighborhood.cell(index);
            let first = neighborhood.lower_bound(spatial, cell, start);
            let last = neighborhood.lower_bound(spatial, cell, end);
            for member in first..last {
                let household = neighborhood.actor(spatial, member);
                let plots = local_plots(state, actors, config, household);
                let trip = distance(actors, farm, household);
                closer += select(trip < radius, plots, u64::new(0));
                boundary += select(trip == radius, plots, u64::new(0));
            }
        }
        let preceding = sum_plots(closer) + prefix_plots(boundary) - boundary;
        let mut cutoff = Shared::<[u32]>::new_slice(1usize);
        if UNIT_POS == 0 {
            cutoff[0] = config.actors;
        }
        sync_cube();
        if preceding < u64::cast_from(required) && preceding + boundary >= u64::cast_from(required)
        {
            let mut low = start;
            let mut high = end - 1;
            while low < high {
                let middle = low + (high - low) / 2;
                let mut plots = preceding;
                for index in 0..neighborhood.width * neighborhood.height {
                    let cell = neighborhood.cell(index);
                    let first = neighborhood.lower_bound(spatial, cell, start);
                    let last = neighborhood.lower_bound(spatial, cell, middle + 1);
                    for member in first..last {
                        let household = neighborhood.actor(spatial, member);
                        plots += select(
                            distance(actors, farm, household) == radius,
                            local_plots(state, actors, config, household),
                            u64::new(0),
                        );
                    }
                }
                let short = plots < u64::cast_from(required);
                low = select(short, middle + 1, low);
                high = select(short, high, middle);
            }
            cutoff[0] = low;
        }
        sync_cube();
        let mut acquired = u64::new(0);
        for index in 0..neighborhood.width * neighborhood.height {
            let cell = neighborhood.cell(index);
            let mut member = spatial[cell as usize] + UNIT_POS;
            let end = spatial[cell as usize + 1];
            while member < end {
                let household = neighborhood.actor(spatial, member);
                let plots = local_plots(state, actors, config, household);
                let trip = distance(actors, farm, household);
                let selected = plots > u64::new(0)
                    && (trip < radius || trip == radius && household <= cutoff[0]);
                if selected {
                    increase(
                        state,
                        config,
                        control,
                        comptime!(Field::DisplacedPlots as u32),
                        household,
                        ticks(u32::cast_from(plots), 0),
                    );
                    set32(
                        state,
                        config,
                        comptime!(Field::LandPlots as u32),
                        household,
                        0,
                    );
                    set32(
                        state,
                        config,
                        comptime!(Field::CropWork as u32),
                        household,
                        0,
                    );
                    acquired += plots;
                }
                member += CUBE_DIM;
            }
        }
        let total = sum_plots(acquired);
        sync_storage();
        if UNIT_POS == 0 {
            increase(
                state,
                config,
                control,
                comptime!(Field::OccupiedPlots as u32),
                farm,
                ticks(u32::cast_from(total), u32::cast_from(total >> 32)),
            );
        }
        sync_storage();
    }
}

#[cube]
fn contains(actors: &[u32], config: &EconomyConfig, region: u32, actor: u32) -> bool {
    let base = region as usize * 6;
    let x = actors[ActorLayout::X * (actors.len() / ActorLayout::WORDS) + actor as usize];
    let y = actors[ActorLayout::Y * (actors.len() / ActorLayout::WORDS) + actor as usize];
    x >= config.property_regions[base]
        && y >= config.property_regions[base + 1]
        && x < config.property_regions[base + 2]
        && y < config.property_regions[base + 3]
}

#[cube]
pub(super) fn scaled_output(
    state: &mut [u32],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
    actor: u32,
    output: Ticks,
) -> Ticks {
    let mut scaled = output;
    let plots = number(state, config, comptime!(Field::OccupiedPlots as u32), actor);
    if plots > 1 {
        let mut bonus = 0u32;
        #[unroll]
        for level in 1..comptime!(crate::domain::industry::ProductionScale::MAX_DOUBLINGS + 1) {
            bonus += select(
                plots >= 1u32 << level,
                comptime!(crate::domain::industry::ProductionScale::BONUS_PER_DOUBLING),
                0u32,
            );
        }
        let whole = divide(output, 1000);
        let fraction = minus(output, multiply(control, whole, 1000));
        let carried = number(
            state,
            config,
            comptime!(Field::ScaleRemainder as u32),
            actor,
        );
        let fractional = fraction.extract(LOW_WORD) * bonus + carried;
        let extra = add_checked(
            control,
            multiply(control, whole, bonus),
            ticks(fractional / 1000, 0),
        );
        scaled = add_checked(control, output, extra);
        set32(
            state,
            config,
            comptime!(Field::ScaleRemainder as u32),
            actor,
            fractional % 1000,
        );
    }
    scaled
}

#[cube]
fn initialize(state: &mut [u32], actors: &[u32], config: &EconomyConfig, actor: u32) {
    if actor < config.actors {
        let current = number(
            state,
            config,
            comptime!(Field::PropertyRegion as u32),
            actor,
        );
        let mut selected = current;
        for region in 0..config.property_regions.len() / 6 {
            selected = select(
                current == 0 && contains(actors, config, region as u32, actor),
                region as u32 + 1,
                selected,
            );
        }
        set32(
            state,
            config,
            comptime!(Field::PropertyRegion as u32),
            actor,
            selected,
        );
        let kind = kind(actors, actor).bits();
        let mut usage = u32::new(comptime!(PropertyUse::Factory as i64));
        let natural = comptime!(
            Good::ALL
                .into_iter()
                .filter(|good| good.kind() == crate::domain::GoodKind::Natural)
                .fold(0u32, |mask, good| mask | (1 << good as u32))
        );
        usage = select(
            natural & (1u32 << kind) != 0,
            comptime!(PropertyUse::Farm as u32),
            usage,
        );
        usage = select(
            kind == 0 || kind == comptime!(Good::Housing as u32),
            comptime!(PropertyUse::Housing as u32),
            usage,
        );
        usage = select(
            kind > 0
                && number(
                    state,
                    config,
                    comptime!(Field::EnterpriseModel as u32),
                    actor,
                ) == 1,
            comptime!(PropertyUse::Store as u32),
            usage,
        );
        set32(
            state,
            config,
            comptime!(Field::PropertyUse as u32),
            actor,
            usage,
        );
        let project = number(state, config, comptime!(Field::ProjectStatus as u32), actor);
        let status = number(
            state,
            config,
            comptime!(Field::EnterpriseStatus as u32),
            actor,
        );
        let abandoned = select(
            kind == 0,
            number(state, config, comptime!(Field::EstateSettled as u32), actor) > 0,
            status >= comptime!(EnterpriseState::Liquidating as u32),
        );
        if abandoned
            && (project == comptime!(ProjectStatus::Requested as u32)
                || project == comptime!(ProjectStatus::Building as u32))
        {
            set32(
                state,
                config,
                comptime!(Field::RequestedPlots as u32),
                actor,
                0,
            );
            set32(
                state,
                config,
                comptime!(Field::ReservedPlots as u32),
                actor,
                0,
            );
            set32(
                state,
                config,
                comptime!(Field::ConstructionRemaining as u32),
                actor,
                0,
            );
            set32(
                state,
                config,
                comptime!(Field::ProjectStatus as u32),
                actor,
                comptime!(ProjectStatus::Cancelled as u32),
            );
        }
    }
}

#[cube]
pub(in crate::simulation) fn authorized(
    state: &mut [u32],
    config: &EconomyConfig,
    site: u32,
    owner: u32,
) -> bool {
    let mut allowed = true;
    if comptime!(config.property_mode == Feature::Enabled) {
        let region = number(state, config, comptime!(Field::PropertyRegion as u32), site);
        allowed = false;
        if region > 0 {
            allowed = config.property_regions[(region as usize - 1) * 6 + 5] == owner;
        }
    }
    allowed
}

#[cube]
fn allocate(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
    region: u32,
) {
    if region < config.property_regions.len() as u32 / 6 {
        let capacity = ticks(config.property_regions[region as usize * 6 + 4], 0);
        let mut occupied = zero();
        for actor in 0..config.actors {
            if number(
                state,
                config,
                comptime!(Field::PropertyRegion as u32),
                actor,
            ) == region + 1
            {
                occupied = add_checked(
                    control,
                    occupied,
                    value(state, config, comptime!(Field::OccupiedPlots as u32), actor),
                );
                occupied = add_checked(
                    control,
                    occupied,
                    value(state, config, comptime!(Field::ReservedPlots as u32), actor),
                );
            }
        }
        if lt(capacity, occupied) {
            KernelFaults::new(comptime!(FaultFlags::PROPERTY_CAPACITY)).record(control);
        }
        let mut free = positive(capacity, occupied);
        for actor in 0..config.actors {
            if number(
                state,
                config,
                comptime!(Field::PropertyRegion as u32),
                actor,
            ) == region + 1
            {
                free = reserve_site(state, config, actor, free);
            }
        }
        for actor in 0..config.actors {
            if kind(actors, actor).is(comptime!(ActorKind::Household))
                && contains(actors, config, region, actor)
            {
                free = allocate_plots(state, config, control, actor, free);
            }
        }
    }
}

#[cube]
fn reserve_site(state: &mut [u32], config: &EconomyConfig, actor: u32, available: Ticks) -> Ticks {
    let mut free = available;
    let status = number(state, config, comptime!(Field::ProjectStatus as u32), actor);
    let requested = value(
        state,
        config,
        comptime!(Field::RequestedPlots as u32),
        actor,
    );
    if status == comptime!(ProjectStatus::Requested as u32) && !lt(free, requested) {
        free = minus(free, requested);
        set(
            state,
            config,
            comptime!(Field::ReservedPlots as u32),
            actor,
            requested,
        );
        set32(
            state,
            config,
            comptime!(Field::RequestedPlots as u32),
            actor,
            0,
        );
        set32(
            state,
            config,
            comptime!(Field::ProjectStatus as u32),
            actor,
            comptime!(ProjectStatus::Building as u32),
        );
    }
    free
}

#[cube]
fn allocate_plots(
    state: &mut [u32],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
    actor: u32,
    free: Ticks,
) -> Ticks {
    let plots = value(state, config, comptime!(Field::LandPlots as u32), actor);
    let settled = number(state, config, comptime!(Field::EstateSettled as u32), actor);
    let retained = select(!lt(free, plots) && settled == 0, plots, zero());
    let lost = select(settled == 0, minus(plots, retained), zero());
    set(
        state,
        config,
        comptime!(Field::LandPlots as u32),
        actor,
        retained,
    );
    increase(
        state,
        config,
        control,
        comptime!(Field::DisplacedPlots as u32),
        actor,
        lost,
    );
    let work = value(state, config, comptime!(Field::CropWork as u32), actor);
    set(
        state,
        config,
        comptime!(Field::CropWork as u32),
        actor,
        select(equal(retained, zero()), zero(), work),
    );
    minus(free, retained)
}

#[cube]
fn construct(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
    actor: u32,
) {
    if actor < config.actors {
        let status = number(state, config, comptime!(Field::ProjectStatus as u32), actor);
        let remaining = value(
            state,
            config,
            comptime!(Field::ConstructionRemaining as u32),
            actor,
        );
        let stock = value(state, config, comptime!(Field::Construction as u32), actor);
        let work = minimum(stock, remaining);
        if status == comptime!(ProjectStatus::Building as u32) {
            set(
                state,
                config,
                comptime!(Field::Construction as u32),
                actor,
                minus(stock, work),
            );
            set(
                state,
                config,
                comptime!(Field::ConstructionRemaining as u32),
                actor,
                minus(remaining, work),
            );
            increase(
                state,
                config,
                control,
                USED_FIELDS as u32 + comptime!(Good::Construction as u32) - 1,
                actor,
                work,
            );
            increase(
                state,
                config,
                control,
                comptime!(Field::Used as u32),
                actor,
                work,
            );
            increase(
                state,
                config,
                control,
                comptime!(Field::InstalledCapital as u32),
                actor,
                work,
            );
            if equal(work, remaining) {
                let plots = value(state, config, comptime!(Field::OccupiedPlots as u32), actor);
                let built = value(state, config, comptime!(Field::ReservedPlots as u32), actor);
                let firm = !kind(actors, actor).is(comptime!(ActorKind::Household));
                let workers = number(
                    state,
                    config,
                    comptime!(Field::WorkersPerPlot as u32),
                    actor,
                );
                let developed = add_checked(control, plots, built);
                let target = multiply(control, developed, workers);
                let capacity = actors
                    [ActorLayout::CAPACITY * (actors.len() / ActorLayout::WORDS) + actor as usize];
                let expansion = select(firm, positive(target, ticks(capacity, 0)), zero());
                increase(
                    state,
                    config,
                    control,
                    comptime!(Field::PendingCapacity as u32),
                    actor,
                    expansion,
                );
                set(
                    state,
                    config,
                    comptime!(Field::OccupiedPlots as u32),
                    actor,
                    add_checked(control, plots, built),
                );
                set32(
                    state,
                    config,
                    comptime!(Field::ReservedPlots as u32),
                    actor,
                    0,
                );
                set32(
                    state,
                    config,
                    comptime!(Field::ProjectStatus as u32),
                    actor,
                    comptime!(ProjectStatus::Completed as u32),
                );
                let enterprise = number(
                    state,
                    config,
                    comptime!(Field::EnterpriseStatus as u32),
                    actor,
                );
                set32(
                    state,
                    config,
                    comptime!(Field::EnterpriseStatus as u32),
                    actor,
                    select(
                        firm && enterprise == comptime!(EnterpriseState::Forming as u32),
                        comptime!(EnterpriseState::Operating as u32),
                        enterprise,
                    ),
                );
            }
        }
    }
}

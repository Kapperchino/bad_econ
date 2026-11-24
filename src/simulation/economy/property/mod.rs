pub(in crate::simulation) mod landlords;
pub(in crate::simulation) mod rent;

use crate::domain::{
    enterprise::{EnterpriseModel, EnterpriseState},
    property::{CONSTRUCTION_WORK_PER_PLOT, ProjectStatus, PropertyUse},
};
use crate::simulation::DeviceEconomy;
use crate::simulation::economy::*;
use crate::simulation::{Device, economy::configuration::Configuration};
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
        let actors = ActorTable::with_stride(&input.actors, input.count as usize);
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
                let x = actors.x(actor);
                let y = actors.y(actor);
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
    ParcelIndex::launch_elements(
        device,
        config.actors.max(cells + 1) as usize,
        ParcelIndexInputLaunch {
            actors: actors.buffer_arg(),
            spatial: spatial.buffer_arg(),
            parcels: parcels.buffer_arg(),
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
        let mut economy = EconomyView::new(&mut input.state, &input.actors, &input.config);
        match comptime!(input.stage) {
            PropertyStage::Initialize => {
                PropertySite::new(ABSOLUTE_POS as u32).initialize(&mut economy)
            }
            PropertyStage::Allocate => allocate(&mut economy, &input.control, ABSOLUTE_POS as u32),
            PropertyStage::Construct => {
                PropertySite::new(ABSOLUTE_POS as u32).construct(&mut economy, &input.control)
            }
            PropertyStage::Consolidate => consolidate(&mut economy, &input.spatial, &input.control),
        }
    }
}

pub(in crate::simulation) fn launch(
    DeviceEconomy {
        device,
        state,
        actors,
        config,
    }: DeviceEconomy<'_>,
    spatial: &CubeTensor,
    control: &CubeTensor,
    phase: Phase,
) {
    let stages: &[PropertyStage] = match (config.property_mode, phase) {
        (Feature::Enabled, Phase::Opening) => &[PropertyStage::Initialize, PropertyStage::Allocate],
        (_, Phase::PhysicalClose) => &[PropertyStage::Construct],
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
                state: state.buffer_arg(),
                actors: actors.buffer_arg(),
                spatial: spatial.buffer_arg(),
                config: config.launch(),
                control: control.buffer_arg(),
                stage,
            },
        );
    }
}

#[cube]
fn consolidate(economy: &mut EconomyView<'_>, spatial: &[u32], control: &[Atomic<u32>]) {
    let config = economy.ledger.config;
    for index in 0..config.grain_farms.len() - 1 {
        let farm = config.grain_farms[index];
        let output = economy.ledger.number(Field::OutputTicks, farm);
        let labor = economy.ledger.number(Field::LaborMinutes, farm);
        let operating = economy.ledger.number(Field::EnterpriseStatus, farm)
            == comptime!(EnterpriseState::Operating as u32);
        let producer = economy.ledger.number(Field::EnterpriseModel, farm) == 0;
        let completed = !PropertySite::new(farm).has_construction_project(&economy.ledger);
        let required = economy.ledger.number(Field::PendingCapacity, farm);
        let labor_efficient = if comptime!(
            config.commercial_wheat == crate::domain::food::WheatHusbandry::NorfolkFourCourse
        ) && config.home.harvest_days < 365
        {
            output * 365 * config.home.cultivation_minutes
                > labor
                    * farming::CropCalendar::new(config).paid_days(comptime!(Good::Grain as u32))
        } else {
            output * config.home.cultivation_minutes > labor
        };
        let status = select(
            operating
                && producer
                && output * 365 > config.home.annual_yield_per_plot
                && labor_efficient,
            select(
                completed,
                comptime!(ConsolidationState::Ready as u32),
                comptime!(ConsolidationState::Building as u32),
            ),
            comptime!(ConsolidationState::Ineligible as u32),
        );
        if status == comptime!(ConsolidationState::Ready as u32) {
            let replacement = divide(
                plus(
                    times32(required, output * 365),
                    ticks(config.home.annual_yield_per_plot - 1, 0),
                )
                .record(control),
                config.home.annual_yield_per_plot,
            );
            consolidate_farm(
                economy,
                spatial,
                control,
                farm,
                replacement.extract(LOW_WORD),
            );
        }
    }
}

#[cube]
fn local_plots(economy: &EconomyView<'_>, household: u32) -> u64 {
    let plots = economy.ledger.number(Field::LandPlots, household);
    u64::cast_from(select(
        economy
            .actors
            .kind(household)
            .is(comptime!(ActorKind::Household)),
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
    fn new(actors: &ActorTable<'_>, config: &EconomyConfig, farm: u32, radius: u32) -> Self {
        let x = actors.x(farm);
        let y = actors.y(farm);
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
fn nearby_plots(economy: &EconomyView<'_>, spatial: &[u32], farm: u32, radius: u32) -> u64 {
    let config = economy.ledger.config;
    let neighborhood = LandNeighborhood::new(&economy.actors, config, farm, radius);
    let mut plots = u64::new(0);
    for index in 0..neighborhood.width * neighborhood.height {
        let cell = neighborhood.cell(index);
        let mut member = spatial[cell as usize] + UNIT_POS;
        let end = spatial[cell as usize + 1];
        while member < end {
            let household = neighborhood.actor(spatial, member);
            let grant = local_plots(economy, household);
            plots += select(
                economy.actors.distance(farm, household) <= radius,
                grant,
                u64::new(0),
            );
            member += CUBE_DIM;
        }
    }
    parallel::reduce::<u64>(plots, comptime!(Reduction::Sum))
}

#[cube]
fn consolidate_farm(
    economy: &mut EconomyView<'_>,
    spatial: &[u32],
    control: &[Atomic<u32>],
    farm: u32,
    required: u32,
) {
    let config = economy.ledger.config;
    if required > 0 {
        let mut low = 0;
        let mut high = min(1u32, config.search_radius);
        let mut available = nearby_plots(&*economy, spatial, farm, high);
        while high < config.search_radius && available < u64::cast_from(required) {
            low = high + 1;
            high = min(high * 2, config.search_radius);
            available = nearby_plots(&*economy, spatial, farm, high);
        }
        while low < high {
            let radius = low + (high - low) / 2;
            let plots = nearby_plots(&*economy, spatial, farm, radius);
            let short = plots < u64::cast_from(required);
            low = select(short, radius + 1, low);
            high = select(short, high, radius);
        }
        let radius = low;
        let neighborhood = LandNeighborhood::new(&economy.actors, config, farm, radius);
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
                let plots = local_plots(&*economy, household);
                let trip = economy.actors.distance(farm, household);
                closer += select(trip < radius, plots, u64::new(0));
                boundary += select(trip == radius, plots, u64::new(0));
            }
        }
        let preceding = parallel::reduce::<u64>(closer, comptime!(Reduction::Sum))
            + parallel::prefix_sum::<u64>(boundary).inclusive
            - boundary;
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
                            economy.actors.distance(farm, household) == radius,
                            local_plots(&*economy, household),
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
                let plots = local_plots(&*economy, household);
                let trip = economy.actors.distance(farm, household);
                let selected = plots > u64::new(0)
                    && (trip < radius || trip == radius && household <= cutoff[0]);
                if selected {
                    economy.ledger.increase(
                        control,
                        Field::DisplacedPlots,
                        household,
                        ticks(u32::cast_from(plots), 0),
                    );
                    economy.ledger.set_number(Field::LandPlots, household, 0);
                    economy.ledger.set_number(Field::CropWork, household, 0);
                    PropertySite::new(household).discard_grain(&mut economy.ledger, control);
                    acquired += plots;
                }
                member += CUBE_DIM;
            }
        }
        let total = parallel::reduce::<u64>(acquired, comptime!(Reduction::Sum));
        sync_storage();
        if UNIT_POS == 0 {
            economy.ledger.increase(
                control,
                Field::OccupiedPlots,
                farm,
                ticks(u32::cast_from(total), u32::cast_from(total >> 32)),
            );
        }
        sync_storage();
    }
}

#[cube]
fn contains(actors: &ActorTable<'_>, config: &EconomyConfig, region: u32, actor: u32) -> bool {
    let base = region as usize * 6;
    let x = actors.x(actor);
    let y = actors.y(actor);
    x >= config.property_regions[base]
        && y >= config.property_regions[base + 1]
        && x < config.property_regions[base + 2]
        && y < config.property_regions[base + 3]
}

#[cube]
fn allocate(economy: &mut EconomyView<'_>, control: &[Atomic<u32>], region: u32) {
    let config = economy.ledger.config;
    if region < config.property_regions.len() as u32 / 6 {
        let capacity = ticks(config.property_regions[region as usize * 6 + 4], 0);
        let mut occupied = zero();
        for actor in 0..config.actors {
            if economy.ledger.number(Field::PropertyRegion, actor) == region + 1 {
                occupied = plus(occupied, economy.ledger.value(Field::OccupiedPlots, actor))
                    .record(control);
                occupied = plus(occupied, economy.ledger.value(Field::ReservedPlots, actor))
                    .record(control);
            }
        }
        if lt(capacity, occupied) {
            KernelFaults::new(comptime!(FaultFlags::PROPERTY_CAPACITY)).record(control);
        }
        let mut free = positive(capacity, occupied);
        for actor in 0..config.actors {
            if economy.ledger.number(Field::PropertyRegion, actor) == region + 1 {
                free = PropertySite::new(actor).reserve(&mut economy.ledger, free);
            }
        }
        for actor in 0..config.actors {
            if economy
                .actors
                .kind(actor)
                .is(comptime!(ActorKind::Household))
                && contains(&economy.actors, config, region, actor)
            {
                free = PropertySite::new(actor).allocate(&mut economy.ledger, control, free);
            }
        }
    }
}

#[derive(CubeType)]
pub(in crate::simulation) struct PropertySite {
    actor: u32,
}

#[cube]
impl PropertySite {
    pub fn new(actor: u32) -> Self {
        PropertySite { actor }
    }

    pub(in crate::simulation::economy) fn consume(
        &self,
        ledger: &mut Ledger<'_>,
        control: &[Atomic<u32>],
        good: u32,
        amount: Ticks,
    ) {
        let actor = self.actor;
        let stock = ledger.value_at(good, actor);
        ledger.set_at(good, actor, minus(stock, amount));
        ledger.increase_at(control, USED_FIELDS as u32 + good - 1, actor, amount);
        ledger.increase(control, Field::Used, actor, amount);
    }

    fn discard_grain(&self, ledger: &mut Ledger<'_>, control: &[Atomic<u32>]) {
        let grain = ledger.value(Field::Grain, self.actor);
        self.consume(ledger, control, comptime!(Good::Grain as u32), grain);
    }

    pub fn scale_output(
        &self,
        ledger: &mut Ledger<'_>,
        control: &[Atomic<u32>],
        output: Ticks,
    ) -> Ticks {
        let actor = self.actor;
        let plots = ledger.number(Field::OccupiedPlots, actor);
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
            let fraction = minus(output, times(whole, 1000).record(control));
            let carried = ledger.number(Field::ScaleRemainder, actor);
            let fractional = fraction.extract(LOW_WORD) * bonus + carried;
            let extra = plus(
                times(whole, bonus).record(control),
                ticks(fractional / 1000, 0),
            )
            .record(control);
            let scaled = plus(output, extra).record(control);
            ledger.set_number(Field::ScaleRemainder, actor, fractional % 1000);
            scaled
        } else {
            output
        }
    }

    fn initialize(&self, economy: &mut EconomyView<'_>) {
        let actor = self.actor;
        let config = economy.ledger.config;
        if actor < config.actors {
            let current = economy.ledger.number(Field::PropertyRegion, actor);
            let mut selected = current;
            for region in 0..config.property_regions.len() / 6 {
                selected = select(
                    current == 0 && contains(&economy.actors, config, region as u32, actor),
                    region as u32 + 1,
                    selected,
                );
            }
            economy
                .ledger
                .set_number(Field::PropertyRegion, actor, selected);
            let usage = self.usage(&*economy);
            economy.ledger.set_number(Field::PropertyUse, actor, usage);
            let kind = economy.actors.kind(actor).bits();
            let status = economy.ledger.number(Field::EnterpriseStatus, actor);
            let abandoned = select(
                kind == 0,
                economy.ledger.number(Field::EstateSettled, actor) > 0,
                status >= comptime!(EnterpriseState::Liquidating as u32),
            );
            if abandoned && self.has_construction_project(&economy.ledger) {
                economy.ledger.set_number(Field::RequestedPlots, actor, 0);
                economy.ledger.set_number(Field::ReservedPlots, actor, 0);
                economy
                    .ledger
                    .set_number(Field::ConstructionRemaining, actor, 0);
                economy
                    .ledger
                    .set_number(Field::ConstructionDaysRemaining, actor, 0);
                economy.ledger.set_number(
                    Field::ProjectStatus,
                    actor,
                    comptime!(ProjectStatus::Cancelled as u32),
                );
            }
        }
    }

    fn usage(&self, economy: &EconomyView<'_>) -> u32 {
        let actor = self.actor;
        let kind = economy.actors.kind(actor).bits();
        let natural = comptime!(
            Good::ALL
                .into_iter()
                .filter(|good| good.kind() == crate::domain::GoodKind::Natural)
                .fold(0u32, |mask, good| mask | (1 << good as u32))
        );
        let usage = select(
            natural & (1u32 << kind) != 0,
            comptime!(PropertyUse::Farm as u32),
            comptime!(PropertyUse::Factory as u32),
        );
        let distributor = economy.ledger.number(Field::EnterpriseModel, actor)
            == comptime!(EnterpriseModel::Distributor as u32);
        let usage = select(distributor, comptime!(PropertyUse::Store as u32), usage);
        select(
            kind == 0 || kind == comptime!(Good::Housing as u32),
            comptime!(PropertyUse::Housing as u32),
            usage,
        )
    }

    pub fn has_construction_project(&self, ledger: &Ledger<'_>) -> bool {
        let status = ledger.number(Field::ProjectStatus, self.actor);
        status == comptime!(ProjectStatus::Requested as u32)
            || status == comptime!(ProjectStatus::Building as u32)
    }

    pub(in crate::simulation::economy) fn request_construction(
        &self,
        economy: &mut EconomyView<'_>,
    ) {
        let actor = self.actor;
        let usage = self.usage(&*economy);
        let days = select(
            usage == comptime!(PropertyUse::Factory as u32),
            comptime!(PropertyUse::Factory.construction_days()),
            comptime!(PropertyUse::Store.construction_days()),
        );
        let days = select(
            usage == comptime!(PropertyUse::Farm as u32),
            comptime!(PropertyUse::Farm.construction_days()),
            days,
        );
        let days = select(
            usage == comptime!(PropertyUse::Housing as u32),
            comptime!(PropertyUse::Housing.construction_days()),
            days,
        );
        let ledger = &mut economy.ledger;
        ledger.set_number(Field::ConstructionDaysRemaining, actor, days);
        ledger.set_number(
            Field::ConstructionRemaining,
            actor,
            comptime!(CONSTRUCTION_WORK_PER_PLOT),
        );
        let status = match comptime!(ledger.config.property_mode) {
            Feature::Enabled => {
                ledger.set_number(Field::RequestedPlots, actor, 1);
                comptime!(ProjectStatus::Requested as u32)
            }
            Feature::Disabled => comptime!(ProjectStatus::Building as u32),
        };
        ledger.set_number(Field::ProjectStatus, actor, status);
    }

    pub fn authorized(&self, ledger: &Ledger<'_>, owner: u32) -> bool {
        let config = ledger.config;
        let mut allowed = true;
        if comptime!(config.property_mode == Feature::Enabled) {
            let region = ledger.number(Field::PropertyRegion, self.actor);
            allowed = false;
            if region > 0 {
                allowed = config.property_regions[(region as usize - 1) * 6 + 5] == owner;
            }
        }
        allowed
    }

    fn reserve(&self, ledger: &mut Ledger<'_>, available: Ticks) -> Ticks {
        let actor = self.actor;
        let mut free = available;
        let status = ledger.number(Field::ProjectStatus, actor);
        let requested = ledger.value(Field::RequestedPlots, actor);
        if status == comptime!(ProjectStatus::Requested as u32) && !lt(free, requested) {
            free = minus(free, requested);
            ledger.set(Field::ReservedPlots, actor, requested);
            ledger.set_number(Field::RequestedPlots, actor, 0);
            ledger.set_number(
                Field::ProjectStatus,
                actor,
                comptime!(ProjectStatus::Building as u32),
            );
        }
        free
    }

    fn allocate(&self, ledger: &mut Ledger<'_>, control: &[Atomic<u32>], free: Ticks) -> Ticks {
        let actor = self.actor;
        let plots = ledger.value(Field::LandPlots, actor);
        let settled = ledger.number(Field::EstateSettled, actor);
        let retained = select(!lt(free, plots) && settled == 0, plots, zero());
        let lost = select(settled == 0, minus(plots, retained), zero());
        ledger.set(Field::LandPlots, actor, retained);
        ledger.increase(control, Field::DisplacedPlots, actor, lost);
        let work = ledger.value(Field::CropWork, actor);
        ledger.set(
            Field::CropWork,
            actor,
            select(equal(retained, zero()), zero(), work),
        );
        if !equal(lost, zero()) {
            self.discard_grain(ledger, control);
        }
        minus(free, retained)
    }

    pub(in crate::simulation::economy) fn construction_due(&self, ledger: &Ledger<'_>) -> Ticks {
        let actor = self.actor;
        select(
            self.building(ledger),
            ledger.value(Field::ConstructionRemaining, actor),
            zero(),
        )
    }

    fn building(&self, ledger: &Ledger<'_>) -> bool {
        let actor = self.actor;
        ledger.number(Field::ProjectStatus, actor) == comptime!(ProjectStatus::Building as u32)
            && ledger.number(Field::EnterpriseStatus, actor)
                < comptime!(EnterpriseState::Liquidating as u32)
    }

    fn construct(&self, economy: &mut EconomyView<'_>, control: &[Atomic<u32>]) {
        let actor = self.actor;
        let config = economy.ledger.config;
        if actor < config.actors {
            if economy.actors.kind(actor).output() == comptime!(Good::Housing as u32) {
                landlords::maintain(&mut economy.ledger, control, actor);
            }
            let stock = economy.ledger.value(Field::Construction, actor);
            let work = if self.building(&economy.ledger) {
                self.advance_construction(economy, control, stock)
            } else {
                zero()
            };
            let unused = minus(stock, work);
            if !equal(unused, zero()) {
                economy.ledger.set(Field::Construction, actor, zero());
                economy.ledger.increase_at(
                    control,
                    CONSUMPTION_FIELDS as u32 + comptime!(Good::Construction as u32) - 1,
                    actor,
                    unused,
                );
            }
        }
    }

    fn advance_construction(
        &self,
        economy: &mut EconomyView<'_>,
        control: &[Atomic<u32>],
        stock: Ticks,
    ) -> Ticks {
        let actor = self.actor;
        let ledger = &mut economy.ledger;
        let remaining = ledger.value(Field::ConstructionRemaining, actor);
        let work = minimum(stock, remaining);
        if !equal(work, zero()) {
            self.consume(ledger, control, comptime!(Good::Construction as u32), work);
            ledger.set(Field::ConstructionRemaining, actor, minus(remaining, work));
            ledger.increase(control, Field::InstalledCapital, actor, work);
        }
        let days = max(ledger.number(Field::ConstructionDaysRemaining, actor), 1) - 1;
        ledger.set_number(Field::ConstructionDaysRemaining, actor, days);
        if days == 0 && equal(work, remaining) {
            self.finish_construction(economy, control);
        }
        work
    }

    fn finish_construction(&self, economy: &mut EconomyView<'_>, control: &[Atomic<u32>]) {
        let actor = self.actor;
        let ledger = &mut economy.ledger;
        let plots = ledger.value(Field::OccupiedPlots, actor);
        let built = ledger.value(Field::ReservedPlots, actor);
        let firm = !economy
            .actors
            .kind(actor)
            .is(comptime!(ActorKind::Household));
        let workers = ledger.number(Field::WorkersPerPlot, actor);
        let developed = plus(plots, built).record(control);
        let target = times(developed, workers).record(control);
        let capacity = economy.actors.capacity(actor);
        let expansion = select(firm, positive(target, ticks(capacity, 0)), zero());
        ledger.increase(control, Field::PendingCapacity, actor, expansion);
        ledger.set(Field::OccupiedPlots, actor, developed);
        ledger.set_number(Field::ReservedPlots, actor, 0);
        ledger.set_number(
            Field::ProjectStatus,
            actor,
            comptime!(ProjectStatus::Completed as u32),
        );
        let enterprise = ledger.number(Field::EnterpriseStatus, actor);
        ledger.set_number(
            Field::EnterpriseStatus,
            actor,
            select(
                firm && enterprise == comptime!(EnterpriseState::Forming as u32),
                comptime!(EnterpriseState::Operating as u32),
                enterprise,
            ),
        );
    }
}

use super::*;
use crate::domain::{
    enterprise::EnterpriseState,
    property::{ProjectStatus, PropertyUse},
};
use crate::simulation::{Device, configuration::Configuration};
use burn_wgpu::CubeTensor;

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
    config: EconomyConfig,
    control: Box<[Atomic<u32>]>,
    #[cube(comptime)]
    stage: PropertyStage,
}

struct PropertySystem;
struct AllocateProperty;

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
        PropertySystem::launch(
            device,
            device.count(count),
            CubeDim::new_1d(device.workgroup),
            PropertyInputLaunch {
                state: state.clone().into_buffer_arg(),
                actors: actors.clone().into_buffer_arg(),
                config: config.launch(),
                control: control.clone().into_buffer_arg(),
                stage,
            },
        );
    }
}

#[cube]
fn consolidate(state: &mut [u32], actors: &[u32], config: &EconomyConfig, control: &[Atomic<u32>]) {
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
                config,
                control,
                farm,
                replacement.extract(LOW_WORD),
            );
        }
    }
}

#[cube]
fn consolidate_farm(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
    farm: u32,
    required: u32,
) {
    let mut remaining = required;
    while remaining > 0 {
        let selected = nearest_farmer(state, actors, config, farm);
        if selected < config.actors {
            let plots = value(state, config, comptime!(Field::LandPlots as u32), selected);
            sync_storage();
            if UNIT_POS == 0 {
                increase(
                    state,
                    config,
                    control,
                    comptime!(Field::OccupiedPlots as u32),
                    farm,
                    plots,
                );
                increase(
                    state,
                    config,
                    control,
                    comptime!(Field::DisplacedPlots as u32),
                    selected,
                    plots,
                );
                set32(
                    state,
                    config,
                    comptime!(Field::LandPlots as u32),
                    selected,
                    0,
                );
                set32(
                    state,
                    config,
                    comptime!(Field::CropWork as u32),
                    selected,
                    0,
                );
            }
            remaining = max(remaining, plots.extract(LOW_WORD)) - plots.extract(LOW_WORD);
        } else {
            remaining = 0;
        }
        sync_storage();
    }
}

#[cube]
fn nearest_farmer(state: &mut [u32], actors: &[u32], config: &EconomyConfig, farm: u32) -> u32 {
    let mut nearest = u64::new(comptime!(u64::MAX as i64));
    let mut household = UNIT_POS;
    while household < config.actors {
        let plots = number(state, config, comptime!(Field::LandPlots as u32), household);
        let trip = distance(actors, farm, household);
        let eligible = kind(actors, household).is(comptime!(ActorKind::Household))
            && plots > 0
            && trip <= config.search_radius;
        let rank = (u64::cast_from(trip) << 32) | u64::cast_from(household);
        nearest = min(nearest, select(eligible, rank, u64::MAX));
        household += CUBE_DIM;
    }
    let mut ranks = Shared::<[u64]>::new_slice(256usize);
    ranks[UNIT_POS as usize] = nearest;
    sync_cube();
    let mut stride = CUBE_DIM / 2;
    while stride > 0 {
        if UNIT_POS < stride {
            ranks[UNIT_POS as usize] = min(
                ranks[UNIT_POS as usize],
                ranks[(UNIT_POS + stride) as usize],
            );
        }
        sync_cube();
        stride /= 2;
    }
    u32::cast_from(ranks[0])
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

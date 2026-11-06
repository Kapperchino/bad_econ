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
        _ => &[],
    };
    for &stage in stages {
        let count = match stage {
            PropertyStage::Allocate => config.property_regions.meta.num_elements() / 6,
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
        usage = select(kind == 0, comptime!(PropertyUse::Housing as u32), usage);
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
                comptime!(Field::RequestedSpace as u32),
                actor,
                0,
            );
            set32(
                state,
                config,
                comptime!(Field::ReservedSpace as u32),
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
                    value(state, config, comptime!(Field::PropertySpace as u32), actor),
                );
                occupied = add_checked(
                    control,
                    occupied,
                    value(state, config, comptime!(Field::ReservedSpace as u32), actor),
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
        comptime!(Field::RequestedSpace as u32),
        actor,
    );
    if status == comptime!(ProjectStatus::Requested as u32) && !lt(free, requested) {
        free = minus(free, requested);
        set(
            state,
            config,
            comptime!(Field::ReservedSpace as u32),
            actor,
            requested,
        );
        set32(
            state,
            config,
            comptime!(Field::RequestedSpace as u32),
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
                let space = value(state, config, comptime!(Field::PropertySpace as u32), actor);
                let built = value(state, config, comptime!(Field::ReservedSpace as u32), actor);
                let firm = !kind(actors, actor).is(comptime!(ActorKind::Household));
                let expansion = select(firm && !equal(space, zero()), built, zero());
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
                    comptime!(Field::PropertySpace as u32),
                    actor,
                    add_checked(control, space, built),
                );
                set32(
                    state,
                    config,
                    comptime!(Field::ReservedSpace as u32),
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

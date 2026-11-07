use super::*;
use crate::domain::{
    enterprise::EnterpriseState,
    landlords::MAINTENANCE_DAYS,
    property::{CONSTRUCTION_WORK_PER_PLOT, ProjectStatus},
};
use crate::simulation::{Device, configuration::Configuration};
use burn_wgpu::CubeTensor;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum RentalStage {
    Reset,
    Supply,
    Accounts,
}

#[derive(CubeType, CubeLaunch)]
struct RentalInput {
    state: Box<[u32]>,
    actors: Box<[u32]>,
    config: EconomyConfig,
    control: Box<[Atomic<u32>]>,
    #[cube(comptime)]
    stage: RentalStage,
}

struct RentalBusinesses;
struct AdvanceRentals;

impl System for RentalBusinesses {
    type Input = RentalInput;
    type Passes = gpu_passes![AdvanceRentals];
}

#[cube]
impl GpuPass<RentalInput> for AdvanceRentals {
    type Output = ();

    fn execute(mut input: RentalInput) {
        let group = ABSOLUTE_POS as u32;
        let count = match comptime!(input.stage) {
            RentalStage::Reset => input.config.actors,
            _ => input.config.rental_businesses,
        };
        if group < count {
            match comptime!(input.stage) {
                RentalStage::Reset => {
                    set32(
                        &mut input.state,
                        &input.config,
                        comptime!(Field::HousingAvailable as u32),
                        group,
                        0,
                    );
                    set32(
                        &mut input.state,
                        &input.config,
                        comptime!(Field::Landlord as u32),
                        group,
                        0,
                    );
                }
                RentalStage::Supply => open(
                    &mut input.state,
                    &input.actors,
                    &input.config,
                    &input.control,
                    group,
                ),
                RentalStage::Accounts => settle(
                    &mut input.state,
                    &input.actors,
                    &input.config,
                    &input.control,
                    group,
                ),
            }
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
    let stages: &[RentalStage] = match (config.rental_businesses, phase) {
        (0, _) => &[],
        (_, Phase::Observation) => &[RentalStage::Reset, RentalStage::Supply],
        (_, Phase::AccountsDecisions) => &[RentalStage::Accounts],
        _ => &[],
    };
    for &stage in stages {
        let count = match stage {
            RentalStage::Reset => config.actors,
            _ => config.rental_businesses,
        };
        RentalBusinesses::launch(
            device,
            device.count(count as usize),
            CubeDim::new_1d(device.workgroup),
            RentalInputLaunch {
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
fn consume(
    state: &mut [u32],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
    actor: u32,
    good: u32,
    amount: Ticks,
) {
    let stock = value(state, config, good, actor);
    set(state, config, good, actor, minus(stock, amount));
    increase(
        state,
        config,
        control,
        USED_FIELDS as u32 + good - 1,
        actor,
        amount,
    );
    increase(
        state,
        config,
        control,
        comptime!(Field::Used as u32),
        actor,
        amount,
    );
}

#[cube]
pub(super) fn maintenance(state: &mut [u32], config: &EconomyConfig, firm: u32) -> Ticks {
    times32(
        number(state, config, comptime!(Field::OccupiedPlots as u32), firm),
        number(state, config, comptime!(Field::InputTicks as u32), firm),
    )
}

#[cube]
fn open(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
    group: u32,
) {
    let firm = config.rental_accounts[(group * 3) as usize];
    let start = config.rental_accounts[(group * 3 + 1) as usize];
    let end = config.rental_accounts[(group * 3 + 2) as usize];
    let expired = value(state, config, comptime!(Field::Housing as u32), firm);
    consume(
        state,
        config,
        control,
        firm,
        comptime!(Good::Housing as u32),
        expired,
    );
    let mut days = number(
        state,
        config,
        comptime!(Field::MaintenanceDays as u32),
        firm,
    );
    let upkeep = maintenance(state, config, firm);
    let operating = number(
        state,
        config,
        comptime!(Field::EnterpriseStatus as u32),
        firm,
    ) == comptime!(EnterpriseState::Operating as u32);
    if operating
        && days == 0
        && !lt(
            value(state, config, comptime!(Field::Construction as u32), firm),
            upkeep,
        )
    {
        consume(
            state,
            config,
            control,
            firm,
            comptime!(Good::Construction as u32),
            upkeep,
        );
        days = MAINTENANCE_DAYS;
    }
    let capacity = times32(
        number(state, config, comptime!(Field::OccupiedPlots as u32), firm),
        number(state, config, comptime!(Field::OutputTicks as u32), firm),
    );
    let supply = select(operating && days > 0, capacity, zero());
    set32(
        state,
        config,
        comptime!(Field::MaintenanceDays as u32),
        firm,
        select(operating && days > 0, max(days, 1) - 1, days),
    );
    set(
        state,
        config,
        comptime!(Field::Housing as u32),
        firm,
        supply,
    );
    increase(
        state,
        config,
        control,
        comptime!(Field::Produced as u32),
        firm,
        supply,
    );
    increase(
        state,
        config,
        control,
        comptime!(Field::UnscaledProduced as u32),
        firm,
        supply,
    );
    let mut available = supply;
    for entry in start..end {
        let tenant = config.rental_accounts[entry as usize];
        let household = kind(actors, tenant).is(comptime!(ActorKind::Household));
        let renter = household
            && number(state, config, comptime!(Field::LandPlots as u32), tenant) == 0
            && number(
                state,
                config,
                comptime!(Field::LivingMembers as u32),
                tenant,
            ) > 0;
        let housed = renter && !equal(available, zero());
        available = positive(available, ticks(u32::cast_from(housed), 0));
        set32(
            state,
            config,
            comptime!(Field::HousingAvailable as u32),
            tenant,
            u32::cast_from(housed),
        );
        set32(
            state,
            config,
            comptime!(Field::Landlord as u32),
            tenant,
            firm + 1,
        );
    }
    consume(
        state,
        config,
        control,
        firm,
        comptime!(Good::Housing as u32),
        minus(supply, available),
    );
}

#[cube]
fn settle(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
    group: u32,
) {
    let firm = config.rental_accounts[(group * 3) as usize];
    let start = config.rental_accounts[(group * 3 + 1) as usize];
    let end = config.rental_accounts[(group * 3 + 2) as usize];
    let mut receipts = zero();
    let mut households = 0u32;
    let mut served = 0u32;
    for entry in start..end {
        let tenant = config.rental_accounts[entry as usize];
        receipts = add_checked(
            control,
            receipts,
            value(state, config, comptime!(Field::RentPending as u32), tenant),
        );
        set32(
            state,
            config,
            comptime!(Field::RentPending as u32),
            tenant,
            0,
        );
        let household = kind(actors, tenant).is(comptime!(ActorKind::Household));
        households += u32::cast_from(
            household
                && number(state, config, comptime!(Field::LandPlots as u32), tenant) == 0
                && number(
                    state,
                    config,
                    comptime!(Field::LivingMembers as u32),
                    tenant,
                ) > 0,
        );
        served += number(
            state,
            config,
            comptime!(Field::HousingAvailable as u32),
            tenant,
        );
    }
    increase(
        state,
        config,
        control,
        comptime!(Field::Cash as u32),
        firm,
        receipts,
    );
    increase(
        state,
        config,
        control,
        comptime!(Field::RentalIncome as u32),
        firm,
        receipts,
    );
    set32(state, config, comptime!(Field::Sales as u32), firm, served);
    set32(
        state,
        config,
        comptime!(Field::Demand as u32),
        firm,
        households,
    );
    set32(
        state,
        config,
        comptime!(Field::Unfilled as u32),
        firm,
        households - min(households, served),
    );
    let upkeep = multiply(
        control,
        maintenance(state, config, firm),
        max(
            number(state, config, comptime!(Field::FirstInputCost as u32), firm),
            1,
        ),
    );
    develop(state, actors, config, firm, households, upkeep);
    let project = value(
        state,
        config,
        comptime!(Field::ConstructionRemaining as u32),
        firm,
    );
    let interested = enterprises::expands(actors, config, firm);
    let capacity = number(state, config, comptime!(Field::OccupiedPlots as u32), firm)
        * number(state, config, comptime!(Field::OutputTicks as u32), firm);
    let planned = select(
        interested && households >= capacity,
        maximum(project, ticks(CONSTRUCTION_WORK_PER_PLOT, 0)),
        project,
    );
    let reserve = add_checked(
        control,
        upkeep,
        multiply(
            control,
            planned,
            max(
                number(state, config, comptime!(Field::FirstInputCost as u32), firm),
                1,
            ),
        ),
    );
    set(
        state,
        config,
        comptime!(Field::OperatingReserve as u32),
        firm,
        reserve,
    );
}

#[cube]
fn develop(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    firm: u32,
    demand: u32,
    upkeep: Ticks,
) {
    let status = number(state, config, comptime!(Field::ProjectStatus as u32), firm);
    let ready = status == comptime!(ProjectStatus::None as u32)
        || status == comptime!(ProjectStatus::Completed as u32)
        || status == comptime!(ProjectStatus::Cancelled as u32);
    let cost = times32(
        CONSTRUCTION_WORK_PER_PLOT,
        max(
            number(state, config, comptime!(Field::FirstInputCost as u32), firm),
            1,
        ),
    );
    let available = positive(
        value(state, config, comptime!(Field::Cash as u32), firm),
        upkeep,
    );
    let owner = actors[ActorLayout::OWNER * (actors.len() / ActorLayout::WORDS) + firm as usize];
    let interested = enterprises::expands(actors, config, firm);
    let units = number(state, config, comptime!(Field::OccupiedPlots as u32), firm)
        * number(state, config, comptime!(Field::OutputTicks as u32), firm);
    let operating = number(
        state,
        config,
        comptime!(Field::EnterpriseStatus as u32),
        firm,
    ) == comptime!(EnterpriseState::Operating as u32);
    if ready
        && operating
        && interested
        && demand >= units
        && !lt(available, cost)
        && property::authorized(state, config, firm, owner)
    {
        set32(
            state,
            config,
            comptime!(Field::ConstructionRemaining as u32),
            firm,
            CONSTRUCTION_WORK_PER_PLOT,
        );
        match comptime!(config.property_mode) {
            Feature::Enabled => {
                set32(
                    state,
                    config,
                    comptime!(Field::RequestedPlots as u32),
                    firm,
                    1,
                );
                set32(
                    state,
                    config,
                    comptime!(Field::ProjectStatus as u32),
                    firm,
                    comptime!(ProjectStatus::Requested as u32),
                );
            }
            Feature::Disabled => {
                set32(
                    state,
                    config,
                    comptime!(Field::ReservedPlots as u32),
                    firm,
                    1,
                );
                set32(
                    state,
                    config,
                    comptime!(Field::ProjectStatus as u32),
                    firm,
                    comptime!(ProjectStatus::Building as u32),
                );
            }
        }
    }
}

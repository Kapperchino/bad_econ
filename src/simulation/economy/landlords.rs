use super::*;
use crate::domain::{
    enterprise::EnterpriseState,
    landlords::{EXPANSION_OCCUPANCY_PERCENT, MAINTENANCE_DAYS},
    property::{CONSTRUCTION_WORK_PER_PLOT, ProjectStatus},
};
use crate::simulation::{Device, configuration::Configuration};
use burn_wgpu::CubeTensor;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum RentalStage {
    Reset,
    Supply,
    Allocate,
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
    #[cube(comptime)]
    scan: parallel::ScanMode,
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
        let group = match comptime!(input.stage) {
            RentalStage::Allocate | RentalStage::Accounts => CUBE_POS as u32,
            _ => ABSOLUTE_POS as u32,
        };
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
                RentalStage::Supply => open(&mut input.state, &input.config, &input.control, group),
                RentalStage::Allocate => allocate(
                    &mut input.state,
                    &input.actors,
                    &input.config,
                    &input.control,
                    group,
                    input.scan,
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
        (_, Phase::Observation) => &[
            RentalStage::Reset,
            RentalStage::Supply,
            RentalStage::Allocate,
        ],
        (_, Phase::AccountsDecisions) => &[RentalStage::Accounts],
        _ => &[],
    };
    for &stage in stages {
        let count = match stage {
            RentalStage::Reset => config.actors as usize,
            RentalStage::Allocate | RentalStage::Accounts => {
                config.rental_businesses as usize * device.workgroup as usize
            }
            _ => config.rental_businesses as usize,
        };
        RentalBusinesses::launch(
            device,
            device.count(count),
            CubeDim::new_1d(device.workgroup),
            RentalInputLaunch {
                state: state.clone().into_buffer_arg(),
                actors: actors.clone().into_buffer_arg(),
                config: config.launch(),
                control: control.clone().into_buffer_arg(),
                stage,
                scan: parallel::ScanMode::new(device),
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
fn open(state: &mut [u32], config: &EconomyConfig, control: &[Atomic<u32>], group: u32) {
    let firm = config.rental_accounts[(group * 3) as usize];
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
}

#[cube]
fn renter(state: &mut [u32], actors: &[u32], config: &EconomyConfig, tenant: u32) -> bool {
    kind(actors, tenant).is(comptime!(ActorKind::Household))
        && number(state, config, comptime!(Field::LandPlots as u32), tenant) == 0
        && number(
            state,
            config,
            comptime!(Field::LivingMembers as u32),
            tenant,
        ) > 0
}

#[cube]
fn allocate(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
    group: u32,
    #[comptime] scan: parallel::ScanMode,
) {
    let firm = config.rental_accounts[(group * 3) as usize];
    let start = config.rental_accounts[(group * 3 + 1) as usize];
    let end = config.rental_accounts[(group * 3 + 2) as usize];
    let supply = value(state, config, comptime!(Field::Housing as u32), firm);
    let mut households = 0u32;
    let mut offset = start;
    while offset < end {
        let entry = offset + UNIT_POS;
        let mut tenant = 0u32;
        let mut eligible = false;
        if entry < end {
            tenant = config.rental_accounts[entry as usize];
            eligible = renter(state, actors, config, tenant);
        }
        let prefix = parallel::scan_prefix(u32::cast_from(eligible), scan);
        let housed = eligible && !lt(supply, ticks(households + prefix.inclusive, 0));
        if entry < end {
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
        households += prefix.total;
        offset += CUBE_DIM;
        sync_cube();
    }
    sync_cube();
    if UNIT_POS == 0 {
        consume(
            state,
            config,
            control,
            firm,
            comptime!(Good::Housing as u32),
            minimum(supply, ticks(households, 0)),
        );
    }
}

#[derive(CubeType)]
struct RentalReceipts {
    cash: Ticks,
    households: u32,
    served: u32,
}

#[cube]
fn receipts(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
    group: u32,
) -> RentalReceipts {
    let start = config.rental_accounts[(group * 3 + 1) as usize];
    let end = config.rental_accounts[(group * 3 + 2) as usize];
    let mut receipts = zero();
    let mut households = 0u32;
    let mut served = 0u32;
    let mut entry = start + UNIT_POS;
    while entry < end {
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
        households += u32::cast_from(renter(state, actors, config, tenant));
        served += number(
            state,
            config,
            comptime!(Field::HousingAvailable as u32),
            tenant,
        );
        entry += CUBE_DIM;
    }
    let mut cash = Shared::<[Ticks]>::new_slice(256usize);
    cash[UNIT_POS as usize] = receipts;
    sync_cube();
    let mut stride = CUBE_DIM / 2;
    while stride > 0 {
        if UNIT_POS < stride {
            cash[UNIT_POS as usize] = add_checked(
                control,
                cash[UNIT_POS as usize],
                cash[(UNIT_POS + stride) as usize],
            );
        }
        sync_cube();
        stride /= 2;
    }
    RentalReceipts {
        cash: cash[0],
        households: parallel::reduce(households, comptime!(Reduction::Sum)),
        served: parallel::reduce(served, comptime!(Reduction::Sum)),
    }
}

#[cube]
fn settle(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
    group: u32,
) {
    let collected = receipts(state, actors, config, control, group);
    if UNIT_POS == 0 {
        close(state, actors, config, control, group, collected);
    }
}

#[cube]
fn close(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
    group: u32,
    collected: RentalReceipts,
) {
    let firm = config.rental_accounts[(group * 3) as usize];
    increase(
        state,
        config,
        control,
        comptime!(Field::Cash as u32),
        firm,
        collected.cash,
    );
    increase(
        state,
        config,
        control,
        comptime!(Field::RentalIncome as u32),
        firm,
        collected.cash,
    );
    let cash = value(state, config, comptime!(Field::Cash as u32), firm);
    let revenue = minus(
        value(state, config, comptime!(Field::RentalIncome as u32), firm),
        value(
            state,
            config,
            comptime!(Field::BusinessOpeningRent as u32),
            firm,
        ),
    );
    set(
        state,
        config,
        comptime!(Field::BusinessClosingCash as u32),
        firm,
        cash,
    );
    set(
        state,
        config,
        comptime!(Field::BusinessRevenue as u32),
        firm,
        revenue,
    );
    set32(
        state,
        config,
        comptime!(Field::Sales as u32),
        firm,
        collected.served,
    );
    set32(
        state,
        config,
        comptime!(Field::Demand as u32),
        firm,
        collected.households,
    );
    set32(
        state,
        config,
        comptime!(Field::Unfilled as u32),
        firm,
        collected.households - min(collected.households, collected.served),
    );
    let upkeep = multiply(
        control,
        maintenance(state, config, firm),
        max(
            number(state, config, comptime!(Field::FirstInputCost as u32), firm),
            1,
        ),
    );
    develop(state, actors, config, firm, collected.households, upkeep);
    let project = value(
        state,
        config,
        comptime!(Field::ConstructionRemaining as u32),
        firm,
    );
    let interested = enterprises::expands(state, actors, config, firm);
    let planned = select(
        interested && expansion_demand(state, config, firm, collected.households),
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
fn expansion_demand(state: &mut [u32], config: &EconomyConfig, firm: u32, demand: u32) -> bool {
    let capacity = times32(
        number(state, config, comptime!(Field::OccupiedPlots as u32), firm),
        number(state, config, comptime!(Field::OutputTicks as u32), firm),
    );
    !lt(
        divide(times32(demand, 100), EXPANSION_OCCUPANCY_PERCENT),
        maximum(capacity, ticks(1, 0)),
    )
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
    let interested = enterprises::expands(state, actors, config, firm);
    let operating = number(
        state,
        config,
        comptime!(Field::EnterpriseStatus as u32),
        firm,
    ) == comptime!(EnterpriseState::Operating as u32);
    if ready
        && operating
        && interested
        && expansion_demand(state, config, firm, demand)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{
        Experiment, Scenario, enterprise::ExpansionPolicy, landlords::RentalRegistry,
    };
    use crate::simulation::{decode, encode, tests::device};
    use anyhow::Result;

    #[test]
    fn rental_batches_preserve_tenant_priority_wide_payments_and_overflow_on_gpu() -> Result<()> {
        let mut policy = Scenario::industrial(1031, 42, Experiment::Concentrated)?
            .config()
            .clone();
        policy.expansion = ExpansionPolicy {
            small_business_per_mille: 0,
            industrial_per_mille: 0,
        };
        let scenario = Scenario::new(policy)?;
        let mut initial = scenario.initial()?;
        let rentals = RentalRegistry::new(&scenario, &initial);
        assert_eq!(rentals.accounts.len(), 1);
        let account = &rentals.accounts[0];
        let firm = account.landlord;
        let tenants = account
            .tenants
            .iter()
            .copied()
            .filter(|&tenant| initial.actors.at(tenant).kind == 0)
            .collect::<Vec<_>>();
        for &tenant in &tenants {
            initial.set(Field::LandPlots, tenant, u64::from(tenant % 3 == 0));
            initial.set(Field::LivingMembers, tenant, u64::from(tenant % 5 != 0));
        }
        let eligible = tenants
            .into_iter()
            .filter(|&tenant| tenant % 3 != 0 && tenant % 5 != 0)
            .collect::<Vec<_>>();
        for &tenant in &account.tenants {
            initial.set(Field::RentPending, tenant, (1 << 32) + tenant as u64 + 1);
        }
        initial.set(Field::OccupiedPlots, firm, 1);
        initial.set(Field::MaintenanceDays, firm, 1);
        initial.set(Field::Housing, firm, 7);
        let payments = account
            .tenants
            .iter()
            .map(|&tenant| initial.get(Field::RentPending, tenant))
            .sum::<u64>();
        for workgroup in [32, 64, 128, 256] {
            let device = device(workgroup)?;
            let config = Configuration::new(&device, &scenario, &initial);
            let metadata = initial
                .actors
                .iter()
                .map(|actor| ActorLayout::new(&actor, &scenario))
                .collect::<ActorMetadataRows>();
            let actors = device.tensor(&metadata.words());
            for capacity in [0, 65, 257, u32::MAX] {
                initial.set(Field::OutputTicks, firm, u64::from(capacity));
                let state = device.tensor(bytemuck::cast_slice(&encode(&initial.values)));
                let control = device.tensor(&[0; CONTROL_WORDS]);
                for phase in [Phase::Observation, Phase::AccountsDecisions] {
                    launch(&device, &state, &actors, &config, &control, phase);
                }
                let mut actual = initial.clone();
                actual.values = decode(&device.read(&state)?);
                FaultFlags::read(&device.read(&control)?)?.into_result()?;
                let housed = eligible
                    .iter()
                    .copied()
                    .take(capacity as usize)
                    .collect::<Vec<_>>();
                for &tenant in &account.tenants {
                    assert_eq!(
                        actual.get(Field::HousingAvailable, tenant),
                        u64::from(housed.contains(&tenant))
                    );
                    assert_eq!(actual.get(Field::Landlord, tenant), firm as u64 + 1);
                    assert_eq!(actual.get(Field::RentPending, tenant), 0);
                }
                assert_eq!(
                    actual.get(Field::Housing, firm),
                    u64::from(capacity) - housed.len() as u64
                );
                assert_eq!(
                    actual.get(Field::Used, firm) - initial.get(Field::Used, firm),
                    7 + housed.len() as u64
                );
                assert_eq!(
                    actual.get(Field::Cash, firm),
                    initial.get(Field::Cash, firm) + payments
                );
                assert_eq!(
                    actual.get(Field::RentalIncome, firm),
                    initial.get(Field::RentalIncome, firm) + payments
                );
                assert_eq!(actual.get(Field::Demand, firm), eligible.len() as u64);
                assert_eq!(actual.get(Field::Sales, firm), housed.len() as u64);
                assert_eq!(
                    actual.get(Field::Unfilled, firm),
                    (eligible.len() - housed.len()) as u64
                );
            }
            let mut overflow = initial.clone();
            overflow.set(Field::RentPending, account.tenants[0], u64::MAX);
            let state = device.tensor(bytemuck::cast_slice(&encode(&overflow.values)));
            let control = device.tensor(&[0; CONTROL_WORDS]);
            launch(
                &device,
                &state,
                &actors,
                &config,
                &control,
                Phase::AccountsDecisions,
            );
            assert!(
                FaultFlags::read(&device.read(&control)?)?
                    .contains(FaultFlags::ARITHMETIC_OVERFLOW)
            );
        }
        Ok(())
    }
}

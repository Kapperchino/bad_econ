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
        let mut economy = EconomyView::new(&mut input.state, &input.actors, &input.config);
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
                    economy.ledger.set_number(Field::HousingAvailable, group, 0);
                    economy.ledger.set_number(Field::Landlord, group, 0);
                }
                RentalStage::Supply => open(&mut economy.ledger, &input.control, group),
                RentalStage::Allocate => allocate(&mut economy, &input.control, group, input.scan),
                RentalStage::Accounts => settle(&mut economy, &input.control, group),
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
        RentalBusinesses::launch_elements(
            device,
            count,
            RentalInputLaunch {
                state: state.buffer_arg(),
                actors: actors.buffer_arg(),
                config: config.launch(),
                control: control.buffer_arg(),
                stage,
                scan: parallel::ScanMode::new(device),
            },
        );
    }
}

#[cube]
fn consume(ledger: &mut Ledger<'_>, control: &[Atomic<u32>], actor: u32, good: u32, amount: Ticks) {
    let stock = ledger.value_at(good, actor);
    ledger.set_at(good, actor, minus(stock, amount));
    ledger.increase_at(control, USED_FIELDS as u32 + good - 1, actor, amount);
    ledger.increase(control, Field::Used, actor, amount);
}

#[cube]
pub(super) fn maintenance(ledger: &Ledger<'_>, firm: u32) -> Ticks {
    times32(
        ledger.number(Field::OccupiedPlots, firm),
        ledger.number(Field::InputTicks, firm),
    )
}

#[cube]
fn open(ledger: &mut Ledger<'_>, control: &[Atomic<u32>], group: u32) {
    let config = ledger.config;
    let firm = config.rental_accounts[(group * 3) as usize];
    let expired = ledger.value(Field::Housing, firm);
    consume(
        ledger,
        control,
        firm,
        comptime!(Good::Housing as u32),
        expired,
    );
    let mut days = ledger.number(Field::MaintenanceDays, firm);
    let upkeep = maintenance(&*ledger, firm);
    let operating = ledger.number(Field::EnterpriseStatus, firm)
        == comptime!(EnterpriseState::Operating as u32);
    if operating && days == 0 && !lt(ledger.value(Field::Construction, firm), upkeep) {
        consume(
            ledger,
            control,
            firm,
            comptime!(Good::Construction as u32),
            upkeep,
        );
        days = MAINTENANCE_DAYS;
    }
    let capacity = times32(
        ledger.number(Field::OccupiedPlots, firm),
        ledger.number(Field::OutputTicks, firm),
    );
    let supply = select(operating && days > 0, capacity, zero());
    ledger.set_number(
        Field::MaintenanceDays,
        firm,
        select(operating && days > 0, max(days, 1) - 1, days),
    );
    ledger.set(Field::Housing, firm, supply);
    ledger.increase(control, Field::Produced, firm, supply);
    ledger.increase(control, Field::UnscaledProduced, firm, supply);
}

#[cube]
fn rental_units(economy: &EconomyView<'_>, tenant: u32) -> u32 {
    let renter = economy
        .actors
        .kind(tenant)
        .is(comptime!(ActorKind::Household))
        && economy.ledger.number(Field::LandPlots, tenant) == 0
        && economy.ledger.number(Field::LivingMembers, tenant) > 0;
    select(renter, consumption::consumers(&economy.ledger, tenant), 0)
}

#[cube]
fn allocate(
    economy: &mut EconomyView<'_>,
    control: &[Atomic<u32>],
    group: u32,
    #[comptime] scan: parallel::ScanMode,
) {
    let config = economy.ledger.config;
    let firm = config.rental_accounts[(group * 3) as usize];
    let start = config.rental_accounts[(group * 3 + 1) as usize];
    let end = config.rental_accounts[(group * 3 + 2) as usize];
    let supply = economy.ledger.value(Field::Housing, firm);
    let mut households = 0u32;
    let mut offset = start;
    while offset < end {
        let entry = offset + UNIT_POS;
        let mut tenant = 0u32;
        let mut units = 0u32;
        if entry < end {
            tenant = config.rental_accounts[entry as usize];
            units = rental_units(&*economy, tenant);
        }
        let prefix = parallel::scan_prefix(units, scan);
        let preceding = households + prefix.inclusive - units;
        let housed =
            minimum(ticks(units, 0), positive(supply, ticks(preceding, 0))).extract(LOW_WORD);
        if entry < end {
            economy
                .ledger
                .set_number(Field::HousingAvailable, tenant, housed);
            economy.ledger.set_number(Field::Landlord, tenant, firm + 1);
        }
        households += prefix.total;
        offset += CUBE_DIM;
        sync_cube();
    }
    sync_cube();
    if UNIT_POS == 0 {
        consume(
            &mut economy.ledger,
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
fn receipts(economy: &mut EconomyView<'_>, control: &[Atomic<u32>], group: u32) -> RentalReceipts {
    let config = economy.ledger.config;
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
            economy.ledger.value(Field::RentPending, tenant),
        );
        economy.ledger.set_number(Field::RentPending, tenant, 0);
        let occupants = rental_units(&*economy, tenant);
        let allocated = economy.ledger.number(Field::HousingAvailable, tenant);
        households += occupants;
        served += allocated;
        economy
            .ledger
            .set_number(Field::HousingAvailable, tenant, min(occupants, allocated));
        entry += CUBE_DIM;
    }
    let mut cash = TickTree::new(receipts);
    WorkgroupTree::reduce::<TickTree>(&mut cash);
    cash.faults.record(control);
    RentalReceipts {
        cash: cash.values[0],
        households: parallel::reduce::<u32>(households, comptime!(Reduction::Sum)),
        served: parallel::reduce::<u32>(served, comptime!(Reduction::Sum)),
    }
}

#[cube]
fn settle(economy: &mut EconomyView<'_>, control: &[Atomic<u32>], group: u32) {
    let collected = receipts(economy, control, group);
    if UNIT_POS == 0 {
        close(economy, control, group, collected);
    }
}

#[cube]
fn close(
    economy: &mut EconomyView<'_>,
    control: &[Atomic<u32>],
    group: u32,
    collected: RentalReceipts,
) {
    let config = economy.ledger.config;
    let firm = config.rental_accounts[(group * 3) as usize];
    economy
        .ledger
        .increase(control, Field::Cash, firm, collected.cash);
    economy
        .ledger
        .increase(control, Field::RentalIncome, firm, collected.cash);
    let cash = economy.ledger.value(Field::Cash, firm);
    let revenue = minus(
        economy.ledger.value(Field::RentalIncome, firm),
        economy.ledger.value(Field::BusinessOpeningRent, firm),
    );
    economy.ledger.set(Field::BusinessClosingCash, firm, cash);
    economy.ledger.set(Field::BusinessRevenue, firm, revenue);
    economy
        .ledger
        .set_number(Field::Sales, firm, collected.served);
    economy
        .ledger
        .set_number(Field::Demand, firm, collected.households);
    economy.ledger.set_number(
        Field::Unfilled,
        firm,
        collected.households - min(collected.households, collected.served),
    );
    let upkeep = multiply(
        control,
        maintenance(&economy.ledger, firm),
        max(economy.ledger.number(Field::FirstInputCost, firm), 1),
    );
    develop(economy, firm, collected.households, upkeep);
    let project = economy.ledger.value(Field::ConstructionRemaining, firm);
    let interested = enterprises::expands(&*economy, firm);
    let planned = select(
        interested && expansion_demand(&economy.ledger, firm, collected.households),
        maximum(project, ticks(CONSTRUCTION_WORK_PER_PLOT, 0)),
        project,
    );
    let reserve = add_checked(
        control,
        upkeep,
        multiply(
            control,
            planned,
            max(economy.ledger.number(Field::FirstInputCost, firm), 1),
        ),
    );
    economy.ledger.set(Field::OperatingReserve, firm, reserve);
}

#[cube]
fn expansion_demand(ledger: &Ledger<'_>, firm: u32, demand: u32) -> bool {
    let capacity = times32(
        ledger.number(Field::OccupiedPlots, firm),
        ledger.number(Field::OutputTicks, firm),
    );
    !lt(
        divide(times32(demand, 100), EXPANSION_OCCUPANCY_PERCENT),
        maximum(capacity, ticks(1, 0)),
    )
}

#[cube]
fn develop(economy: &mut EconomyView<'_>, firm: u32, demand: u32, upkeep: Ticks) {
    let config = economy.ledger.config;
    let status = economy.ledger.number(Field::ProjectStatus, firm);
    let ready = status == comptime!(ProjectStatus::None as u32)
        || status == comptime!(ProjectStatus::Completed as u32)
        || status == comptime!(ProjectStatus::Cancelled as u32);
    let cost = times32(
        CONSTRUCTION_WORK_PER_PLOT,
        max(economy.ledger.number(Field::FirstInputCost, firm), 1),
    );
    let available = positive(economy.ledger.value(Field::Cash, firm), upkeep);
    let owner = economy.actors.owner(firm);
    let interested = enterprises::expands(&*economy, firm);
    let operating = economy.ledger.number(Field::EnterpriseStatus, firm)
        == comptime!(EnterpriseState::Operating as u32);
    if ready
        && operating
        && interested
        && expansion_demand(&economy.ledger, firm, demand)
        && !lt(available, cost)
        && property::authorized(&economy.ledger, firm, owner)
    {
        economy.ledger.set_number(
            Field::ConstructionRemaining,
            firm,
            CONSTRUCTION_WORK_PER_PLOT,
        );
        match comptime!(config.property_mode) {
            Feature::Enabled => {
                economy.ledger.set_number(Field::RequestedPlots, firm, 1);
                economy.ledger.set_number(
                    Field::ProjectStatus,
                    firm,
                    comptime!(ProjectStatus::Requested as u32),
                );
            }
            Feature::Disabled => {
                economy.ledger.set_number(Field::ReservedPlots, firm, 1);
                economy.ledger.set_number(
                    Field::ProjectStatus,
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
        assert!(!rentals.accounts.is_empty());
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
        let workgroup = 64;
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
            let mut actual = initial.clone();
            launch(
                &device,
                &state,
                &actors,
                &config,
                &control,
                Phase::Observation,
            );
            actual.values = decode(&device.read(&state)?);
            let departed = eligible[0];
            actual.set(Field::LivingMembers, departed, 0);
            device.write(&state, bytemuck::cast_slice(&encode(&actual.values)));
            launch(
                &device,
                &state,
                &actors,
                &config,
                &control,
                Phase::AccountsDecisions,
            );
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
                    u64::from(housed.contains(&tenant) && tenant != departed)
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
            assert_eq!(actual.get(Field::Demand, firm), eligible.len() as u64 - 1);
            assert_eq!(actual.get(Field::Sales, firm), housed.len() as u64);
            assert_eq!(
                actual.get(Field::Unfilled, firm),
                (eligible.len() - 1).saturating_sub(housed.len()) as u64
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
            FaultFlags::read(&device.read(&control)?)?.contains(FaultFlags::ARITHMETIC_OVERFLOW)
        );
        Ok(())
    }
}

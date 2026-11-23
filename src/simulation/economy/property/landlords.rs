use crate::domain::{
    enterprise::EnterpriseState,
    landlords::{EXPANSION_OCCUPANCY_PERCENT, MAINTENANCE_DAYS},
    property::{CONSTRUCTION_WORK_PER_PLOT, ProjectStatus},
};
use crate::simulation::DeviceEconomy;
use crate::simulation::economy::*;
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
                RentalStage::Supply => {
                    RentalGroup::new(&input.config, &input.control, group)
                        .open(&mut economy.ledger);
                }
                RentalStage::Allocate => {
                    RentalGroup::new(&input.config, &input.control, group)
                        .allocate(&mut economy, input.scan);
                }
                RentalStage::Accounts => {
                    RentalGroup::new(&input.config, &input.control, group).settle(&mut economy);
                }
            }
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
pub(in crate::simulation::economy) fn maintenance(ledger: &Ledger<'_>, firm: u32) -> Ticks {
    times32(
        ledger.number(Field::OccupiedPlots, firm),
        ledger.number(Field::InputTicks, firm),
    )
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

#[derive(CubeType)]
struct RentalReceipts {
    cash: Ticks,
    households: u32,
    served: u32,
}

#[derive(CubeType)]
struct RentalGroup<'a> {
    firm: u32,
    start: u32,
    end: u32,
    control: &'a [Atomic<u32>],
}

#[cube]
impl<'a> RentalGroup<'a> {
    fn new(config: &EconomyConfig, control: &'a [Atomic<u32>], group: u32) -> Self {
        let base = (group * 3) as usize;
        RentalGroup::<'a> {
            firm: config.rental_accounts[base],
            start: config.rental_accounts[base + 1],
            end: config.rental_accounts[base + 2],
            control,
        }
    }

    fn open(&self, ledger: &mut Ledger<'_>) {
        let control = self.control;
        let firm = self.firm;
        let expired = ledger.value(Field::Housing, firm);
        property::PropertySite::new(firm).consume(
            ledger,
            control,
            comptime!(Good::Housing as u32),
            expired,
        );
        let mut days = ledger.number(Field::MaintenanceDays, firm);
        let upkeep = maintenance(&*ledger, firm);
        let operating = ledger.number(Field::EnterpriseStatus, firm)
            == comptime!(EnterpriseState::Operating as u32);
        if operating && days == 0 && !lt(ledger.value(Field::Construction, firm), upkeep) {
            property::PropertySite::new(firm).consume(
                ledger,
                control,
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

    fn allocate(&self, economy: &mut EconomyView<'_>, #[comptime] scan: parallel::ScanMode) {
        let control = self.control;
        let firm = self.firm;
        let config = economy.ledger.config;
        let start = self.start;
        let end = self.end;
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
            property::PropertySite::new(firm).consume(
                &mut economy.ledger,
                control,
                comptime!(Good::Housing as u32),
                minimum(supply, ticks(households, 0)),
            );
        }
    }

    fn receipts(&self, economy: &mut EconomyView<'_>) -> RentalReceipts {
        let control = self.control;
        let config = economy.ledger.config;
        let start = self.start;
        let end = self.end;
        let mut receipts = zero();
        let mut households = 0u32;
        let mut served = 0u32;
        let mut entry = start + UNIT_POS;
        while entry < end {
            let tenant = config.rental_accounts[entry as usize];
            receipts =
                plus(receipts, economy.ledger.value(Field::RentPending, tenant)).record(control);
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

    fn settle(&self, economy: &mut EconomyView<'_>) {
        let collected = self.receipts(economy);
        if UNIT_POS == 0 {
            self.close(economy, collected);
        }
    }

    fn close(&self, economy: &mut EconomyView<'_>, collected: RentalReceipts) {
        let control = self.control;
        let firm = self.firm;
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
        let upkeep = times(
            maintenance(&economy.ledger, firm),
            max(economy.ledger.number(Field::FirstInputCost, firm), 1),
        )
        .record(control);
        self.develop(economy, collected.households, upkeep);
        let project = economy.ledger.value(Field::ConstructionRemaining, firm);
        let interested = enterprises::expands(&*economy, firm);
        let planned = select(
            interested && self.expansion_demand(&economy.ledger, collected.households),
            maximum(project, ticks(CONSTRUCTION_WORK_PER_PLOT, 0)),
            project,
        );
        let reserve = plus(
            upkeep,
            times(
                planned,
                max(economy.ledger.number(Field::FirstInputCost, firm), 1),
            )
            .record(control),
        )
        .record(control);
        economy.ledger.set(Field::OperatingReserve, firm, reserve);
    }

    fn expansion_demand(&self, ledger: &Ledger<'_>, demand: u32) -> bool {
        let firm = self.firm;
        let capacity = times32(
            ledger.number(Field::OccupiedPlots, firm),
            ledger.number(Field::OutputTicks, firm),
        );
        !lt(
            divide(times32(demand, 100), EXPANSION_OCCUPANCY_PERCENT),
            maximum(capacity, ticks(1, 0)),
        )
    }

    fn develop(&self, economy: &mut EconomyView<'_>, demand: u32, upkeep: Ticks) {
        let firm = self.firm;
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
            && self.expansion_demand(&economy.ledger, demand)
            && !lt(available, cost)
            && property::PropertySite::new(firm).authorized(&economy.ledger, owner)
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{
        Experiment, Scenario, enterprise::ExpansionPolicy, landlords::RentalRegistry,
    };
    use crate::simulation::economy::configuration::Configuration;
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
                DeviceEconomy {
                    device: &device,
                    state: &state,
                    actors: &actors,
                    config: &config,
                },
                &control,
                Phase::Observation,
            );
            actual.values = decode(&device.read(&state)?);
            let departed = eligible[0];
            actual.set(Field::LivingMembers, departed, 0);
            device.write(&state, bytemuck::cast_slice(&encode(&actual.values)));
            launch(
                DeviceEconomy {
                    device: &device,
                    state: &state,
                    actors: &actors,
                    config: &config,
                },
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
            DeviceEconomy {
                device: &device,
                state: &state,
                actors: &actors,
                config: &config,
            },
            &control,
            Phase::AccountsDecisions,
        );
        assert!(
            FaultFlags::read(&device.read(&control)?)?.contains(FaultFlags::ARITHMETIC_OVERFLOW)
        );
        Ok(())
    }
}

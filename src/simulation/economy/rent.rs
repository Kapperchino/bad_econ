use super::*;
use crate::domain::{FIRM_RENT_GRACE_DAYS, HousingState, enterprise::EnterpriseState};

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(super) enum Tenants {
    Households,
    Firms,
}

#[cube]
pub(super) fn unit_price(daily: u32, demand: Ticks, output: u32) -> u32 {
    let forecast = (u64::cast_from(demand.extract(HIGH_WORD)) << 32)
        | u64::cast_from(demand.extract(LOW_WORD));
    let units = max(forecast, u64::cast_from(max(output, 1)));
    u32::cast_from((u64::cast_from(daily) + units / 2) / units)
}

#[cube]
pub(super) fn obligation(
    state: &mut [u32],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
    firm: u32,
) -> Ticks {
    add_checked(
        control,
        value(state, config, comptime!(Field::RentArrears as u32), firm),
        ticks(
            select(
                number(state, config, comptime!(Field::LaborMinutes as u32), firm) == 0,
                0,
                config.firm_rent,
            ),
            0,
        ),
    )
}

#[cube]
pub(super) fn collect(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
    actor: u32,
    #[comptime] tenants: Tenants,
) {
    if actor < config.actors {
        let household = kind(actors, actor).is(comptime!(ActorKind::Household));
        let selected = match tenants {
            Tenants::Households => household,
            Tenants::Firms => !household,
        };
        if selected {
            let assigned = config.rental_businesses == 0
                || number(state, config, comptime!(Field::Landlord as u32), actor) > 0;
            let mut daily = match kind(actors, actor).bits() {
                ActorKind::HOUSEHOLD_BITS => select(
                    number(state, config, comptime!(Field::LivingMembers as u32), actor) > 0
                        && number(state, config, comptime!(Field::LandPlots as u32), actor) == 0,
                    config.household_rent,
                    0,
                ),
                _ => select(
                    number(
                        state,
                        config,
                        comptime!(Field::EnterpriseStatus as u32),
                        actor,
                    ) < comptime!(EnterpriseState::Liquidating as u32),
                    select(
                        kind(actors, actor).bits() == comptime!(Good::Housing as u32),
                        0,
                        config.firm_rent,
                    ),
                    0,
                ),
            };
            daily = select(assigned, daily, 0);
            let arrears = value(state, config, comptime!(Field::RentArrears as u32), actor);
            let obligation = add_checked(control, arrears, ticks(daily, 0));
            let cash = value(state, config, comptime!(Field::Cash as u32), actor);
            let mut available = cash;
            if comptime!(tenants == Tenants::Households) {
                let reserve = times32(
                    number(state, config, comptime!(Field::FoodNeed as u32), actor),
                    config.coverage + config.ring - 1,
                );
                let food = add_checked(
                    control,
                    value(state, config, comptime!(Field::Bread as u32), actor),
                    value(
                        state,
                        config,
                        TRANSIT_FIELDS as u32 + comptime!(Good::Bread as u32) - 1,
                        actor,
                    ),
                );
                available = select(lt(food, reserve), zero(), cash);
            }
            available = select(assigned, available, zero());
            let mut housing = 0u32;
            if comptime!(tenants == Tenants::Households) {
                let previous = number(state, config, comptime!(Field::HousingStatus as u32), actor);
                let supplied = config.rental_businesses == 0
                    || number(
                        state,
                        config,
                        comptime!(Field::HousingAvailable as u32),
                        actor,
                    ) > 0;
                let exempt = config.household_rent == 0
                    || number(state, config, comptime!(Field::LandPlots as u32), actor) > 0
                    || number(state, config, comptime!(Field::LivingMembers as u32), actor) == 0;
                let sheltered = exempt
                    || (assigned
                        && supplied
                        && (previous == comptime!(HousingState::Housed as u32)
                            || !lt(available, obligation)));
                daily = select(sheltered, daily, 0);
                housing = select(
                    sheltered,
                    comptime!(HousingState::Housed as u32),
                    comptime!(HousingState::Homeless as u32),
                );
            }
            let due = add_checked(control, arrears, ticks(daily, 0));
            let paid = minimum(available, due);
            let remaining = minus(due, paid);
            match tenants {
                Tenants::Households => {
                    let evicted = daily > 0 && !equal(remaining, zero());
                    set32(
                        state,
                        config,
                        comptime!(Field::HousingStatus as u32),
                        actor,
                        select(evicted, comptime!(HousingState::Homeless as u32), housing),
                    );
                }
                Tenants::Firms => {
                    let previous = number(
                        state,
                        config,
                        comptime!(Field::RentOverdueDays as u32),
                        actor,
                    );
                    let days = select(
                        equal(remaining, zero()),
                        0,
                        min(previous + 1, FIRM_RENT_GRACE_DAYS),
                    );
                    let status = number(
                        state,
                        config,
                        comptime!(Field::EnterpriseStatus as u32),
                        actor,
                    );
                    let next = select(
                        days == FIRM_RENT_GRACE_DAYS,
                        max(status, comptime!(EnterpriseState::Liquidating as u32)),
                        status,
                    );
                    set32(
                        state,
                        config,
                        comptime!(Field::RentOverdueDays as u32),
                        actor,
                        days,
                    );
                    set32(
                        state,
                        config,
                        comptime!(Field::EnterpriseStatus as u32),
                        actor,
                        next,
                    );
                }
            }
            let treasury = add_checked(
                control,
                value(state, config, comptime!(Field::RentTreasury as u32), actor),
                paid,
            );
            increase(
                state,
                config,
                control,
                comptime!(Field::RentPending as u32),
                actor,
                paid,
            );
            set(
                state,
                config,
                comptime!(Field::Cash as u32),
                actor,
                minus(cash, paid),
            );
            set(
                state,
                config,
                comptime!(Field::RentArrears as u32),
                actor,
                remaining,
            );
            set(
                state,
                config,
                comptime!(Field::RentTreasury as u32),
                actor,
                treasury,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{Experiment, RentPolicy, Scenario};
    use crate::simulation::{configuration::Configuration, decode, encode, tests::device};
    use anyhow::Result;

    #[cube(launch)]
    fn collect_rent(
        state: &mut [u32],
        actors: &[u32],
        config: &EconomyConfig,
        control: &[Atomic<u32>],
    ) {
        collect(
            state,
            actors,
            config,
            control,
            ABSOLUTE_POS as u32,
            Tenants::Firms,
        );
        collect(
            state,
            actors,
            config,
            control,
            ABSOLUTE_POS as u32,
            Tenants::Households,
        );
    }

    struct TenantCase {
        actor: usize,
        living: u64,
        land: u64,
        status: EnterpriseState,
        cash: u64,
        arrears: u64,
        paid: u64,
        remaining: u64,
    }

    #[test]
    fn rent_exemptions_partial_payments_and_arrears_settle_on_gpu() -> Result<()> {
        let cases = [
            TenantCase {
                actor: 0,
                living: 1,
                land: 1,
                status: EnterpriseState::Operating,
                cash: 100,
                arrears: 0,
                paid: 0,
                remaining: 0,
            },
            TenantCase {
                actor: 1,
                living: 1,
                land: 0,
                status: EnterpriseState::Operating,
                cash: 100,
                arrears: 0,
                paid: 0,
                remaining: 60,
            },
            TenantCase {
                actor: 2,
                living: 1,
                land: 0,
                status: EnterpriseState::Operating,
                cash: 25,
                arrears: 0,
                paid: 25,
                remaining: 35,
            },
            TenantCase {
                actor: 3,
                living: 0,
                land: 0,
                status: EnterpriseState::Operating,
                cash: 100,
                arrears: 0,
                paid: 0,
                remaining: 0,
            },
            TenantCase {
                actor: 4,
                living: 1,
                land: 0,
                status: EnterpriseState::Operating,
                cash: 100,
                arrears: 75,
                paid: 100,
                remaining: 35,
            },
            TenantCase {
                actor: 5,
                living: 1,
                land: 1,
                status: EnterpriseState::Operating,
                cash: 100,
                arrears: 40,
                paid: 40,
                remaining: 0,
            },
            TenantCase {
                actor: 6,
                living: 0,
                land: 0,
                status: EnterpriseState::Operating,
                cash: 100,
                arrears: 0,
                paid: 60,
                remaining: 0,
            },
            TenantCase {
                actor: 7,
                living: 0,
                land: 0,
                status: EnterpriseState::Dormant,
                cash: 25,
                arrears: 0,
                paid: 25,
                remaining: 35,
            },
            TenantCase {
                actor: 8,
                living: 0,
                land: 0,
                status: EnterpriseState::Closed,
                cash: 100,
                arrears: 0,
                paid: 0,
                remaining: 0,
            },
            TenantCase {
                actor: 9,
                living: 0,
                land: 0,
                status: EnterpriseState::Forming,
                cash: 100,
                arrears: 0,
                paid: 60,
                remaining: 0,
            },
        ];
        let mut config = Scenario::baseline(6, 42, Experiment::Concentrated)?
            .config()
            .clone();
        config.firms = 4;
        config.rent = Some(RentPolicy {
            household_daily: 60,
            firm_daily: 60,
        });
        let scenario = Scenario::new(config)?;
        let mut initial = scenario.initial()?;
        for case in &cases {
            initial.set(Field::HousingAvailable, case.actor, 1);
            initial.set(Field::LivingMembers, case.actor, case.living);
            initial.set(Field::FoodNeed, case.actor, case.living);
            initial.set(Field::LandPlots, case.actor, case.land);
            initial.set(Field::EnterpriseStatus, case.actor, case.status as u64);
            initial.set(Field::Cash, case.actor, case.cash);
            initial.set(Field::RentArrears, case.actor, case.arrears);
            initial.set(Field::RentTreasury, case.actor, 1 << 32);
            let food = initial.get(Field::FoodNeed, case.actor)
                * (u64::from(scenario.config().coverage) + scenario.ring() as u64 - 1);
            let stock = match case.actor {
                1 => food - 1,
                4 => 0,
                _ => food,
            };
            initial.set(Field::Bread, case.actor, stock);
            if case.actor == 4 {
                initial.values[(TRANSIT_FIELDS + Good::Bread as usize - 1)
                    * initial.actors.len()
                    + case.actor] = food;
            }
        }
        for workgroup in [32, 128] {
            let device = device(workgroup)?;
            let config = Configuration::new(&device, &scenario, &initial);
            let state = device.tensor(bytemuck::cast_slice(&encode(&initial.values)));
            let metadata = initial
                .actors
                .iter()
                .map(|actor| ActorLayout::new(&actor, &scenario))
                .collect::<ActorMetadataRows>();
            let actors = device.tensor(&metadata.words());
            let control = device.tensor(&[0; CONTROL_WORDS]);
            collect_rent::launch(
                &device.client,
                device.count(initial.actors.len()),
                CubeDim::new_1d(workgroup),
                state.clone().into_buffer_arg(),
                actors.into_buffer_arg(),
                config.launch(),
                control.clone().into_buffer_arg(),
            );
            let mut actual = initial.clone();
            actual.values = decode(&device.read(&state)?);
            for case in &cases {
                assert_eq!(actual.get(Field::Cash, case.actor), case.cash - case.paid);
                assert_eq!(
                    actual.get(Field::RentTreasury, case.actor),
                    (1 << 32) + case.paid
                );
                assert_eq!(actual.get(Field::RentArrears, case.actor), case.remaining);
            }
            for (actor, housing) in [
                HousingState::Housed,
                HousingState::Homeless,
                HousingState::Homeless,
                HousingState::Housed,
                HousingState::Homeless,
                HousingState::Housed,
            ]
            .into_iter()
            .enumerate()
            {
                assert_eq!(actual.get(Field::HousingStatus, actor), housing as u64);
            }
            assert_eq!(actual.get(Field::RentOverdueDays, 7), 1);
            FaultFlags::read(&device.read(&control)?)?.into_result()?;
        }
        Ok(())
    }

    #[test]
    fn housing_recovery_and_thirty_day_firm_eviction_run_on_gpu() -> Result<()> {
        struct RentCase {
            housing: HousingState,
            status: EnterpriseState,
            overdue: u64,
            cash: u64,
            arrears: u64,
            paid: u64,
            remaining: u64,
            next_housing: HousingState,
            next_status: EnterpriseState,
            next_overdue: u64,
        }
        let households = [
            RentCase {
                housing: HousingState::Housed,
                status: EnterpriseState::Operating,
                overdue: 0,
                cash: 59,
                arrears: 0,
                paid: 59,
                remaining: 1,
                next_housing: HousingState::Homeless,
                next_status: EnterpriseState::Operating,
                next_overdue: 0,
            },
            RentCase {
                housing: HousingState::Housed,
                status: EnterpriseState::Operating,
                overdue: 0,
                cash: 60,
                arrears: 0,
                paid: 60,
                remaining: 0,
                next_housing: HousingState::Housed,
                next_status: EnterpriseState::Operating,
                next_overdue: 0,
            },
            RentCase {
                housing: HousingState::Homeless,
                status: EnterpriseState::Operating,
                overdue: 0,
                cash: 0,
                arrears: 60,
                paid: 0,
                remaining: 60,
                next_housing: HousingState::Homeless,
                next_status: EnterpriseState::Operating,
                next_overdue: 0,
            },
            RentCase {
                housing: HousingState::Homeless,
                status: EnterpriseState::Operating,
                overdue: 0,
                cash: 60,
                arrears: 60,
                paid: 60,
                remaining: 0,
                next_housing: HousingState::Homeless,
                next_status: EnterpriseState::Operating,
                next_overdue: 0,
            },
            RentCase {
                housing: HousingState::Homeless,
                status: EnterpriseState::Operating,
                overdue: 0,
                cash: 120,
                arrears: 60,
                paid: 120,
                remaining: 0,
                next_housing: HousingState::Housed,
                next_status: EnterpriseState::Operating,
                next_overdue: 0,
            },
            RentCase {
                housing: HousingState::Homeless,
                status: EnterpriseState::Operating,
                overdue: 0,
                cash: 119,
                arrears: 60,
                paid: 60,
                remaining: 0,
                next_housing: HousingState::Homeless,
                next_status: EnterpriseState::Operating,
                next_overdue: 0,
            },
        ];
        let firms = [
            RentCase {
                housing: HousingState::Housed,
                status: EnterpriseState::Operating,
                overdue: 28,
                cash: 0,
                arrears: 1680,
                paid: 0,
                remaining: 1740,
                next_housing: HousingState::Housed,
                next_status: EnterpriseState::Operating,
                next_overdue: 29,
            },
            RentCase {
                housing: HousingState::Housed,
                status: EnterpriseState::Operating,
                overdue: 29,
                cash: 0,
                arrears: 1740,
                paid: 0,
                remaining: 1800,
                next_housing: HousingState::Housed,
                next_status: EnterpriseState::Liquidating,
                next_overdue: 30,
            },
            RentCase {
                housing: HousingState::Housed,
                status: EnterpriseState::Dormant,
                overdue: 29,
                cash: 1800,
                arrears: 1740,
                paid: 1800,
                remaining: 0,
                next_housing: HousingState::Housed,
                next_status: EnterpriseState::Dormant,
                next_overdue: 0,
            },
            RentCase {
                housing: HousingState::Housed,
                status: EnterpriseState::Forming,
                overdue: 29,
                cash: 0,
                arrears: 1740,
                paid: 0,
                remaining: 1800,
                next_housing: HousingState::Housed,
                next_status: EnterpriseState::Liquidating,
                next_overdue: 30,
            },
            RentCase {
                housing: HousingState::Housed,
                status: EnterpriseState::Closed,
                overdue: 30,
                cash: 0,
                arrears: 1800,
                paid: 0,
                remaining: 1800,
                next_housing: HousingState::Housed,
                next_status: EnterpriseState::Closed,
                next_overdue: 30,
            },
            RentCase {
                housing: HousingState::Housed,
                status: EnterpriseState::Liquidating,
                overdue: 30,
                cash: 100,
                arrears: 1800,
                paid: 100,
                remaining: 1700,
                next_housing: HousingState::Housed,
                next_status: EnterpriseState::Liquidating,
                next_overdue: 30,
            },
        ];
        let cases = households.into_iter().chain(firms).collect::<Vec<_>>();
        let mut policy = Scenario::baseline(6, 42, Experiment::Concentrated)?
            .config()
            .clone();
        policy.firms = 6;
        policy.rent = Some(RentPolicy {
            household_daily: 60,
            firm_daily: 60,
        });
        let scenario = Scenario::new(policy)?;
        let mut initial = scenario.initial()?;
        for (actor, case) in cases.iter().enumerate() {
            let living = u64::from(initial.actors.at(actor).kind == 0);
            initial.set(Field::HousingAvailable, actor, 1);
            initial.set(Field::LivingMembers, actor, living);
            initial.set(Field::FoodNeed, actor, living);
            initial.set(Field::LandPlots, actor, 0);
            initial.set(Field::HousingStatus, actor, case.housing as u64);
            initial.set(Field::EnterpriseStatus, actor, case.status as u64);
            initial.set(Field::RentOverdueDays, actor, case.overdue);
            initial.set(Field::Cash, actor, case.cash);
            initial.set(Field::RentArrears, actor, case.arrears);
            initial.set(Field::Bread, actor, 1000);
        }
        for workgroup in [32, 128] {
            let device = device(workgroup)?;
            let config = Configuration::new(&device, &scenario, &initial);
            let state = device.tensor(bytemuck::cast_slice(&encode(&initial.values)));
            let metadata = initial
                .actors
                .iter()
                .map(|actor| ActorLayout::new(&actor, &scenario))
                .collect::<ActorMetadataRows>();
            let actors = device.tensor(&metadata.words());
            let control = device.tensor(&[0; CONTROL_WORDS]);
            collect_rent::launch(
                &device.client,
                device.count(initial.actors.len()),
                CubeDim::new_1d(workgroup),
                state.clone().into_buffer_arg(),
                actors.into_buffer_arg(),
                config.launch(),
                control.clone().into_buffer_arg(),
            );
            let mut actual = initial.clone();
            actual.values = decode(&device.read(&state)?);
            for (actor, case) in cases.iter().enumerate() {
                assert_eq!(
                    actual.get(Field::Cash, actor),
                    case.cash - case.paid,
                    "actor {actor}"
                );
                assert_eq!(
                    actual.get(Field::RentTreasury, actor),
                    case.paid,
                    "actor {actor}"
                );
                assert_eq!(
                    actual.get(Field::RentArrears, actor),
                    case.remaining,
                    "actor {actor}"
                );
                assert_eq!(
                    actual.get(Field::HousingStatus, actor),
                    case.next_housing as u64,
                    "actor {actor}"
                );
                assert_eq!(
                    actual.get(Field::EnterpriseStatus, actor),
                    case.next_status as u64,
                    "actor {actor}"
                );
                assert_eq!(
                    actual.get(Field::RentOverdueDays, actor),
                    case.next_overdue,
                    "actor {actor}"
                );
            }
            FaultFlags::read(&device.read(&control)?)?.into_result()?;
        }
        Ok(())
    }
}

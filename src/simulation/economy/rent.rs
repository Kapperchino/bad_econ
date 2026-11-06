use super::*;
use crate::domain::enterprise::EnterpriseState;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(super) enum Tenants {
    Households,
    Firms,
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
        ticks(config.firm_rent, 0),
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
            let daily = match kind(actors, actor).bits() {
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
                    ) != comptime!(EnterpriseState::Closed as u32),
                    config.firm_rent,
                    0,
                ),
            };
            let due = add_checked(
                control,
                value(state, config, comptime!(Field::RentArrears as u32), actor),
                ticks(daily, 0),
            );
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
            let paid = minimum(available, due);
            let treasury = add_checked(
                control,
                value(state, config, comptime!(Field::RentTreasury as u32), actor),
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
                minus(due, paid),
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
            FaultFlags::read(&device.read(&control)?)?.into_result()?;
        }
        Ok(())
    }
}

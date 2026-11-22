use super::*;
use crate::domain::{Experiment, HouseholdLiquidity, Scenario};
use crate::simulation::{Simulation, decode, encode};

#[derive(Clone, Copy, Debug)]
enum EntryInputs {
    Stocked,
    Missing,
}

#[test]
fn cached_migration_needs_preserve_destinations_and_compacted_records_on_gpu() -> anyhow::Result<()>
{
    #[derive(Clone, Copy, Debug)]
    enum Supplies {
        Available,
        MissingFood,
        MissingHeat,
    }

    let device = crate::simulation::tests::device(64)?;
    let mut config = Scenario::industrial(257, 42, Experiment::Baseline)?
        .config()
        .clone();
    config.entrepreneurship.interested_per_mille = 0;
    config.commute_speed = 1;
    config.delivery_penalty = 10_000;
    config.household_liquidity = HouseholdLiquidity::Uniform;
    config.household_cash = 1_000_000;
    let scenario = Scenario::new(config)?;
    for day in [1, 20, 30, 120] {
        for supplies in [
            Supplies::Available,
            Supplies::MissingFood,
            Supplies::MissingHeat,
        ] {
            let expected = Simulation::new(scenario.clone(), device.clone())?;
            let actual = Simulation::new(scenario.clone(), device.clone())?;
            let mut initial = scenario.initial()?;
            let firms = initial.actors.firm_indices().collect::<Vec<_>>();
            let urban = firms
                .iter()
                .copied()
                .filter(|&firm| initial.actors.at(firm).kind == Good::Bread as u32)
                .collect::<Vec<_>>();
            for owner in 0..scenario.config().households as usize {
                let position = initial.actors.at(owner).position;
                let employer = urban
                    .iter()
                    .copied()
                    .max_by_key(|&firm| position.distance(initial.actors.at(firm).position))
                    .expect("urban employer");
                let rural = owner >= 64;
                let employed = !(128..192).contains(&owner);
                initial.set(
                    Field::Employer,
                    owner,
                    if employed { employer as u64 + 1 } else { 0 },
                );
                initial.set(Field::Employed, owner, u64::from(employed));
                initial.set(Field::LivingMembers, owner, 1);
                initial.set(Field::LandPlots, owner, u64::from(rural));
                initial.set(Field::IncomeForecast, owner, 4_800_000);
                initial.set(
                    Field::FoodPrice,
                    owner,
                    if owner >= 192 { 9_000 } else { 100 },
                );
                initial.set(Field::NutritionNeed, owner, 1);
                initial.set(Field::HeatingNeed, owner, 2);
                initial.set(Field::MoveDue, owner, 0);
                initial.set(Field::MoveTarget, owner, 0);
                initial.set(Field::Shortage, owner, 0);
                initial.set(Field::EquityInvested, owner, 1);
            }
            for firm in firms {
                initial.set(
                    Field::EnterpriseStatus,
                    firm,
                    EnterpriseState::Operating as u64,
                );
                initial.set(Field::Cash, firm, 1_000_000);
                initial.set(Field::Wage, firm, 10);
                initial.set(Field::LaborMinutes, firm, 480);
                initial.set(Field::Slots, firm, 100);
                initial.set(Field::Price, firm, 1);
                initial.set(
                    Field::Bread,
                    firm,
                    match supplies {
                        Supplies::MissingFood => 0,
                        _ => 1_000_000,
                    },
                );
                for field in [Field::Coal, Field::Wood] {
                    initial.set(
                        field,
                        firm,
                        match supplies {
                            Supplies::MissingHeat => 0,
                            _ => 1_000_000,
                        },
                    );
                }
            }
            for (simulation, mode) in [
                (&expected, EntryQuoteMode::Sampled),
                (&actual, EntryQuoteMode::Cached),
            ] {
                device.write(
                    &simulation.state,
                    bytemuck::cast_slice(&encode(&crate::simulation::tensor::state_values(
                        &initial.values,
                        initial.actors.len(),
                        simulation.config.state_stride as usize,
                    ))),
                );
                let records = RecordBuffers {
                    requests: &simulation.enterprise_requests,
                    births: simulation.demography.birth_buffer(),
                    scratch: &simulation.work.scratch,
                };
                propose_quotes(
                    DeviceEconomy {
                        device: &device,
                        ..simulation.gpu()
                    },
                    &simulation.spatial,
                    records,
                    day,
                    mode,
                );
            }
            let values = device.read(&actual.state)?;
            assert_eq!(
                values,
                device.read(&expected.state)?,
                "day {day}, {supplies:?}"
            );
            let values = decode(&values);
            let start = Field::MoveTarget as usize * actual.config.state_stride as usize;
            let targets = &values[start..start + scenario.config().households as usize];
            assert!(targets[128..].iter().all(|&target| target == 0));
            let farm_moves = (64..128)
                .filter(|&owner| targets[owner] > 0)
                .collect::<Vec<_>>();
            assert!(
                farm_moves
                    .iter()
                    .all(|&owner| (day as usize + owner).is_multiple_of(30))
            );
            match supplies {
                Supplies::Available => {
                    assert!(targets[..64].iter().any(|&target| target > 0));
                    assert!(!farm_moves.is_empty(), "day {day}");
                }
                _ => assert!(targets.iter().all(|&target| target == 0)),
            }
            let pending = |simulation: &Simulation| {
                crate::simulation::registry::records::PendingRecords::read(
                    &device,
                    RecordBuffers {
                        requests: &simulation.enterprise_requests,
                        births: simulation.demography.birth_buffer(),
                        scratch: &simulation.work.scratch,
                    },
                    &simulation.searches.requests,
                )
            };
            assert_eq!(pending(&actual)?.records, pending(&expected)?.records);
        }
    }
    Ok(())
}

#[test]
fn cached_entry_quotes_preserve_sampled_proposals_and_household_state_on_gpu() -> anyhow::Result<()>
{
    let device = crate::simulation::tests::device(64)?;
    let mut config = Scenario::industrial(3000, 42, Experiment::Baseline)?
        .config()
        .clone();
    config.entrepreneurship.interested_per_mille = 1000;
    config.household_liquidity = HouseholdLiquidity::Uniform;
    config.household_cash = 1_000_000;
    let scenario = Scenario::new(config)?;
    for (day, inputs) in [1, 20, 30, 120].into_iter().flat_map(|day| {
        [EntryInputs::Stocked, EntryInputs::Missing]
            .into_iter()
            .map(move |inputs| (day, inputs))
    }) {
        let expected = Simulation::new(scenario.clone(), device.clone())?;
        let actual = Simulation::new(scenario.clone(), device.clone())?;
        let mut initial = scenario.initial()?;
        for owner in 0..scenario.config().households as usize {
            initial.set(
                Field::LivingMembers,
                owner,
                u64::from(scenario.config().demography.initial_household_members),
            );
            initial.set(Field::InvestmentFirm, owner, 0);
            initial.set(Field::InvestmentGood, owner, 1);
            initial.set(Field::Tools, owner, 2);
            initial.set(Field::Wood, owner, 10);
        }
        for firm in initial.actors.firm_indices().collect::<Vec<_>>() {
            let good = Good::from_index(initial.actors.at(firm).kind)?;
            initial.set(
                Field::EnterpriseStatus,
                firm,
                match good {
                    Good::Bread | Good::Flour => EnterpriseState::Operating as u64,
                    _ => EnterpriseState::Dormant as u64,
                },
            );
            match good {
                Good::Bread => {
                    initial.set(
                        Field::Price,
                        firm,
                        u64::from(scenario.config().maximum_price),
                    );
                    initial.set(Field::Bread, firm, 0);
                }
                Good::Flour => {
                    initial.set(Field::Price, firm, 1);
                    initial.set(
                        Field::Flour,
                        firm,
                        match inputs {
                            EntryInputs::Stocked => 1_000_000,
                            EntryInputs::Missing => 0,
                        },
                    );
                }
                _ => (),
            }
        }
        for (simulation, quote_mode) in [
            (&expected, opportunities::EntryQuoteMode::Sampled),
            (&actual, opportunities::EntryQuoteMode::Cached),
        ] {
            device.write(
                &simulation.state,
                bytemuck::cast_slice(&encode(&initial.values)),
            );
            propose_quotes(
                DeviceEconomy {
                    device: &device,
                    ..simulation.gpu()
                },
                &simulation.spatial,
                RecordBuffers {
                    requests: &simulation.enterprise_requests,
                    births: simulation.demography.birth_buffer(),
                    scratch: &simulation.work.scratch,
                },
                day,
                quote_mode,
            );
        }
        let state = device.read(&actual.state)?;
        assert_eq!(
            state,
            device.read(&expected.state)?,
            "day {day}, {inputs:?}"
        );
        assert_eq!(
            device.read(&actual.enterprise_requests)?,
            device.read(&expected.enterprise_requests)?,
            "day {day}, {inputs:?}"
        );
        let values = decode(&state);
        let stride = actual.config.state_stride as usize;
        let producers = (0..scenario.config().households as usize)
            .filter(|&owner| values[Field::InvestmentGood as usize * stride + owner] >> 24 == 8)
            .collect::<Vec<_>>();
        match inputs {
            EntryInputs::Stocked => {
                assert!(!producers.is_empty(), "day {day}: no producer discoveries");
                for owner in producers {
                    assert_eq!(values[Field::ReplacementCost as usize * stride + owner], 1);
                    assert_eq!(values[Field::SecondInputCost as usize * stride + owner], 0);
                }
            }
            EntryInputs::Missing => assert!(
                producers.is_empty(),
                "day {day}: unavailable input was accepted"
            ),
        }
    }
    Ok(())
}

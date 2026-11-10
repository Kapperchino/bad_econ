use super::*;
use std::{
    collections::{HashMap, hash_map::Entry},
    sync::{Mutex, OnceLock},
};

pub(super) fn device(workgroup: u32) -> Result<Device> {
    static DEVICES: OnceLock<Mutex<HashMap<u32, Device>>> = OnceLock::new();
    let mut devices = DEVICES
        .get_or_init(Mutex::default)
        .lock()
        .map_err(|_| anyhow!("test GPU device cache poisoned"))?;
    match devices.entry(workgroup) {
        Entry::Occupied(entry) => Ok(entry.get().clone()),
        Entry::Vacant(entry) => Ok(entry.insert(Device::headless(workgroup)?).clone()),
    }
}

fn compare_market_execution(scenario: Scenario, days: usize) -> Result<()> {
    let initial = scenario.initial()?;
    compare_market_state(scenario, initial, days)
}

fn compare_market_state(scenario: Scenario, initial: State, days: usize) -> Result<()> {
    let mut grouped = Simulation::from_state(scenario.clone(), initial.clone(), device(128)?)?;
    let mut reference = Simulation::from_state(scenario, initial, device(64)?)?;
    reference.market_execution = MarketExecution::Reference;
    for day in 0..days {
        for phase in Phase::ALL {
            let expected = reference.phase(phase, Trace::Record)?;
            let actual = grouped.phase(phase, Trace::Record)?;
            assert_eq!(actual.events, expected.events, "day {day} {phase:?}");
            assert_eq!(actual.decisions, expected.decisions, "day {day} {phase:?}");
            assert_eq!(
                grouped.download()?,
                reference.download()?,
                "day {day} {phase:?}"
            );
        }
        assert_eq!(grouped.snapshot, reference.snapshot, "day {day}");
    }
    Ok(())
}

#[test]
fn price_blocks_preserve_rotated_search_and_unaffordable_food_diagnostics() -> Result<()> {
    let mut config = Scenario::baseline(769, 52, Experiment::Baseline)?
        .config()
        .clone();
    config.firms = 3;
    config.population = Some(Population {
        households: (1..=769)
            .map(|identity| HouseholdSeed {
                identity,
                position: match identity % 3 {
                    0 => Position { x: 999, y: 999 },
                    1 => Position { x: 0, y: 0 },
                    _ => Position { x: 500, y: 500 },
                },
                cash: match identity % 4 {
                    0 => 1,
                    1 => 35,
                    _ => 5000,
                },
                inventories: Good::food([0, 0, u64::from(identity > 256) * 100]),
                fatigue: 0,
            })
            .collect(),
        firms: (770..=772)
            .map(|identity| FirmSeed {
                identity,
                position: Position { x: 0, y: 0 },
                owner: 1,
                good: Good::Bread,
                cash: 0,
                inventories: [0; Good::COUNT],
                capacity: 1,
                technique: Technique {
                    labor_minutes: 60,
                    input_ticks: 1,
                    output_ticks: 1,
                },
                wage: 1,
                price: 10,
            })
            .collect(),
    });
    let scenario = Scenario::new(config)?;
    let mut initial = scenario.initial()?;
    for household in 0..769 {
        initial.set(Field::Price, household, 10 * (household as u64 / 128 + 1));
    }
    compare_market_state(scenario, initial, 4)
}

#[test]
fn grouped_markets_match_coordinate_scheduler_at_every_phase() -> Result<()> {
    for experiment in [
        Experiment::Baseline,
        Experiment::Shortage,
        Experiment::Transport,
        Experiment::Concentrated,
    ] {
        let mut config = Scenario::baseline(193, 42, experiment)?.config().clone();
        config.firms = 12;
        config.batch_size = 31;
        compare_market_execution(Scenario::new(config)?, 8)?;
    }
    Ok(())
}

#[test]
fn cooperative_consumer_markets_preserve_trades_and_accounting_at_every_phase() -> Result<()> {
    for experiment in [Experiment::Baseline, Experiment::Concentrated] {
        compare_market_execution(Scenario::industrial(97, 42, experiment)?, 12)?;
    }
    Ok(())
}

#[test]
fn mixed_rail_and_animal_sellers_match_serial_settlement_at_every_phase() -> Result<()> {
    for units in [1, 100] {
        let mut config: ScenarioConfig =
            serde_json::from_str(include_str!("../../scenarios/railway.json"))?;
        config.railway.units_per_level = units;
        config.firms += 1;
        let population = config.population.as_mut().expect("explicit population");
        population.households[1].position = Position { x: 900, y: 900 };
        population.households[1].cash = 3000;
        population.households[1].inventories = [0; Good::COUNT];
        let mut local = population.firms[0].clone();
        local.identity = 19;
        local.owner = 2;
        local.position = Position { x: 900, y: 900 };
        local.price += 1;
        population.firms.push(local);
        compare_market_execution(Scenario::new(config)?, 8)?;
    }
    Ok(())
}

#[test]
fn grouped_institutions_and_custom_packets_match_coordinate_scheduler() -> Result<()> {
    for scenario in [
        include_str!("../../scenarios/institutions.json"),
        include_str!("../../scenarios/custom-population.json"),
    ] {
        let mut config: ScenarioConfig = serde_json::from_str(scenario)?;
        config.batch_size = 1;
        compare_market_execution(Scenario::new(config)?, 8)?;
    }
    let mut config = Scenario::baseline(67, 93, Experiment::Shortage)?
        .config()
        .clone();
    config.household_cash = 1u64 << 40;
    config.firm_cash = 1u64 << 42;
    config.reserve = 1;
    config.batch_size = 13;
    compare_market_execution(Scenario::new(config)?, 8)
}

struct MarketFault {
    phase: Phase,
    field: Field,
    actor: usize,
}

#[test]
fn grouped_market_faults_match_reference_and_restore_the_audited_state() -> Result<()> {
    let cases = [
        MarketFault {
            phase: Phase::Commitment,
            field: Field::Due,
            actor: 0,
        },
        MarketFault {
            phase: Phase::Commitment,
            field: Field::Escrow,
            actor: 3,
        },
        MarketFault {
            phase: Phase::Commitment,
            field: Field::Pipeline,
            actor: 3,
        },
        MarketFault {
            phase: Phase::Commitment,
            field: Field::Used,
            actor: 4,
        },
        MarketFault {
            phase: Phase::ProvisionPurchase,
            field: Field::Sales,
            actor: 5,
        },
        MarketFault {
            phase: Phase::ProvisionPurchase,
            field: Field::Purchased,
            actor: 0,
        },
    ];
    for case in cases {
        let scenario = Scenario::baseline(3, 42, Experiment::Concentrated)?;
        let initial = scenario.initial()?;
        let mut messages = Vec::new();
        for (execution, trace) in [
            (MarketExecution::Grouped, Trace::Record),
            (MarketExecution::Grouped, Trace::Discard),
            (MarketExecution::Original, Trace::Discard),
            (MarketExecution::Reference, Trace::Record),
        ] {
            let mut simulation = Simulation::new(scenario.clone(), device(64)?)?;
            simulation.market_execution = execution;
            for phase in Phase::ALL
                .into_iter()
                .take_while(|phase| *phase != case.phase)
            {
                simulation.phase(phase, Trace::Record)?;
            }
            let mut state = simulation.download()?;
            match case.phase {
                Phase::Commitment => {
                    for firm in 3..6 {
                        state.set(Field::Slots, firm, 1);
                    }
                    if case.actor == 4 {
                        state.set(Field::Slots, 3, 0);
                    }
                }
                _ => {
                    for household in 0..3 {
                        state.set(Field::Bread, household, 0);
                    }
                }
            }
            state.set(case.field, case.actor, u64::MAX);
            simulation.device.write(
                &simulation.state,
                bytemuck::cast_slice(&encode(&state.values)),
            );
            messages.push(simulation.phase(case.phase, trace).unwrap_err().to_string());
            assert_eq!(simulation.download()?, initial);
        }
        assert!(messages.iter().all(|message| message == &messages[0]));
    }
    Ok(())
}

#[test]
fn gpu_registry_migration_births_and_estates_match_audited_rebuilds() -> Result<()> {
    let mut config = Scenario::baseline(300, 42, Experiment::Baseline)?
        .config()
        .clone();
    config.firms = 3;
    config.search_radius = 200;
    config.shipment_speed = 100;
    config.household_production = crate::domain::enterprise::HouseholdProduction::industrial();
    config.demography.annual_births_per_million_fertile_women = 1_000_000;
    config.demography.mortality.background_annual_ppm = [1_000_000; 5];
    config.demography.mortality.famine_daily_ppm_at_thirty_days = 0;
    config.population = Some(Population {
        households: (1..=300)
            .map(|identity| HouseholdSeed {
                identity,
                position: Position { x: 0, y: 0 },
                cash: 5000,
                inventories: [0; Good::COUNT],
                fatigue: 0,
            })
            .collect(),
        firms: [400, 700, 900]
            .into_iter()
            .enumerate()
            .map(|(index, x)| FirmSeed {
                identity: 1000 + index as u64,
                position: Position { x, y: 0 },
                owner: 1,
                good: Good::Bread,
                cash: 0,
                inventories: Good::food([0, 0, if index == 0 { 6000 } else { 0 }]),
                capacity: 1,
                technique: Technique::new(60, 1, 1).expect("basic bakery"),
                wage: 1,
                price: 10,
            })
            .collect(),
    });
    let scenario = Scenario::new(config)?;
    let mut gpu = Simulation::new(scenario.clone(), device(64)?)?;
    let mut reference = Simulation::new(scenario.clone(), device(128)?)?;
    reference.registry_execution = RegistryExecution::Reference;
    for day in 0..60 {
        if day == 12 {
            let mut state = gpu.download()?;
            let stock = state.get(Field::Bread, 300);
            state.set(Field::Bread, 300, 0);
            state.set(Field::Bread, 302, stock);
            let checkpoint = Checkpoint::new(&scenario, &state)?;
            gpu.command(SimulationCommand::Restore(Box::new(checkpoint.clone())))?;
            reference.command(SimulationCommand::Restore(Box::new(checkpoint)))?;
        }
        gpu.advance()?;
        reference.advance()?;
        assert_eq!(gpu.download()?, reference.download()?, "day {day}");
        assert_eq!(gpu.snapshot, reference.snapshot, "day {day}");
    }
    let state = gpu.download()?;
    assert!(
        state
            .actors
            .iter()
            .any(|actor| actor.kind == 0 && actor.position.x == 900)
    );
    assert!(gpu.snapshot.population.births > 0);
    assert!(gpu.snapshot.population.deaths[0] > 0);
    assert!((0..state.actors.len()).any(|actor| state.get(Field::EstateSettled, actor) == 2));
    gpu.checkpoint()?.restore()?;
    Ok(())
}

#[test]
fn gpu_parent_estates_preserve_inheritance_without_full_state_readback() -> Result<()> {
    use crate::domain::demography::{DeathCause, LivingState};
    use crate::domain::enterprise::OwnershipShare;

    let mut config = Scenario::industrial(3000, 42, Experiment::Baseline)?
        .config()
        .clone();
    config.research = None;
    config.firms = 6 * Good::COUNT as u32;
    let scenario = Scenario::new(config)?;
    let mut initial = scenario.initial()?;
    assert!(
        crate::domain::landlords::RentalRegistry::new(&scenario, &initial)
            .accounts
            .len()
            > 1
    );
    initial.day = 1700;
    for (child, parent) in [(1, 0), (2, 0), (3, 0), (32, 31), (41, 40)] {
        initial.set(
            Field::ParentHousehold,
            child,
            initial.actors.at(parent).identity,
        );
    }
    let deceased = [0, 1, 31, 40, 41].map(|index| initial.actors.at(index).identity);
    for person in &mut initial.people {
        if deceased.contains(&person.household) {
            person.state = LivingState::Deceased {
                day: initial.day,
                cause: DeathCause::Background,
            };
        }
    }
    for household in 0..3000 {
        let identity = initial.actors.at(household).identity;
        let living = initial
            .people
            .iter()
            .filter(|person| {
                person.household == identity && matches!(person.state, LivingState::Living)
            })
            .count() as u64;
        initial.set(Field::LivingMembers, household, living);
    }
    for (firm, parent) in [(3000, 0), (3001, 31), (3002, 10)] {
        let mut actor = initial.actors.at(firm);
        actor.owner = parent as u32;
        let identity = actor.identity;
        initial.actors.set(firm, actor);
        let household = initial.actors.at(parent).identity;
        let ownership = initial
            .ownership
            .iter_mut()
            .find(|ownership| ownership.firm == identity)
            .context("issued ownership")?;
        ownership.shares = vec![OwnershipShare {
            household,
            units: 1_000_000,
        }];
    }
    let heir = initial.actors.at(2).identity;
    initial
        .ownership
        .iter_mut()
        .find(|ownership| ownership.firm == initial.actors.at(3000).identity)
        .context("joint ownership")?
        .shares = vec![
        OwnershipShare {
            household: initial.actors.at(0).identity,
            units: 500_000,
        },
        OwnershipShare {
            household: heir,
            units: 500_000,
        },
    ];
    let slot = (initial.day % scenario.ring() as u64) as usize;
    let delivery = initial
        .shipment_coordinate(Good::Bread, slot, scenario.ring(), 0)
        .0 as usize;
    initial.set(Field::Bread, 0, initial.get(Field::Bread, 0) - 1);
    initial.values[delivery] += 1;
    initial.values[(TRANSIT_FIELDS + Good::Bread as usize - 1) * initial.actors.len()] += 1;
    initial.set(Field::Transit, 0, initial.get(Field::Transit, 0) + 1);
    let mut requests = vec![0; initial.actors.len()];
    for actor in [0, 1, 31, 40, 41] {
        requests[actor] = 5 << 24;
    }
    requests[5] = (2 << 24) + 3003;
    requests[3001] = (6 << 24) + 2;
    initial.set(Field::PendingCapacity, 3001, 2);
    initial.set(Field::InstalledCapital, 3001, 1);
    for formation in [None, Some(6)] {
        let mut initial = initial.clone();
        let mut requests = requests.clone();
        if let Some(parent) = formation {
            let identity = initial.actors.at(parent).identity;
            let person = initial
                .people
                .iter()
                .find(|person| person.household == identity)
                .context("forming household member")?
                .identity;
            initial.set(Field::SplitPerson, parent, person);
            requests[parent] = (1 << 24) + 1;
        }
        let pending =
            registry::records::PendingRecords::from_dense(&requests, &vec![0; requests.len()])?;
        let mut expected = initial.clone();
        enterprise::transitions(&mut expected, &scenario, &requests)?;
        for workgroup in [32, 128] {
            let mut gpu =
                Simulation::from_state(scenario.clone(), initial.clone(), device(workgroup)?)
                    .context("initial inheritance state")?;
            gpu.device.write(&gpu.enterprise_requests, &requests);
            let before = gpu.readback_bytes;
            match registry::entries::EntryGrowth::new(
                &scenario,
                &gpu.metadata,
                &gpu.migrations,
                &pending.records,
            )? {
                Some(growth) => {
                    gpu.commit_entries(&growth.requests, growth.plan, &pending.records)?
                }
                None => {
                    let plan = registry::RegistryPlan::new(
                        &scenario,
                        &gpu.metadata,
                        &gpu.migrations,
                        &pending.records,
                    )?
                    .context("parent estates must use sparse GPU settlement")?;
                    gpu.commit_registry(plan, &pending.records)?;
                }
            }
            assert!(gpu.readback_bytes - before < initial.values.len() as u64);
            let mut reference =
                Simulation::from_state(scenario.clone(), expected.clone(), device(64)?)?;
            assert!(gpu.download()? == reference.download()?);
            assert_eq!(gpu.actors().at(3000).owner, 2);
            assert_eq!(gpu.actors().at(3001).owner, 32);
            for phase in Phase::ALL {
                gpu.phase(phase, Trace::Discard)?;
                reference.phase(phase, Trace::Discard)?;
                let actual = gpu.download()?;
                let expected = reference.download()?;
                let differences = actual
                    .values
                    .iter()
                    .zip(&expected.values)
                    .enumerate()
                    .filter(|(_, (actual, expected))| actual != expected)
                    .take(12)
                    .collect::<Vec<_>>();
                assert!(
                    actual == expected,
                    "formation {formation:?}, workgroup {workgroup}, phase {phase:?}: {differences:?}"
                );
            }
            gpu.checkpoint()?.restore()?;
        }
    }
    Ok(())
}

#[test]
fn gpu_household_growth_matches_rebuilds_and_keeps_unrelated_estates_sparse() -> Result<()> {
    let mut config = Scenario::industrial(300, 42, Experiment::Baseline)?
        .config()
        .clone();
    config.household_cash = 1_000_000;
    config.household_liquidity = HouseholdLiquidity::Uniform;
    config.buffer_days = 30;
    config.demography.annual_births_per_million_fertile_women = 0;
    let scenario = Scenario::new(config)?;
    let mut initial = scenario.initial()?;
    for person in &mut initial.people {
        person.identity += 1 << 32;
    }
    initial.next_identity += 1 << 32;
    let parents = [0, 31, 299];
    for parent in parents {
        let identity = initial.actors.at(parent).identity;
        let person = initial
            .people
            .iter()
            .filter(|person| person.household == identity)
            .nth(parent % 3)
            .context("household member")?
            .identity;
        initial.set(Field::SplitPerson, parent, person);
        initial.set(
            Field::LivingMembers,
            parent,
            initial
                .people
                .iter()
                .filter(|person| person.household == identity)
                .count() as u64,
        );
    }
    for (good, amount) in [(Good::Tools, 2), (Good::Wood, 5)] {
        let field = Field::inventory(good);
        let donor = (300..initial.actors.len())
            .max_by_key(|&actor| initial.get(field, actor))
            .context("material supplier")?;
        initial.set(field, donor, initial.get(field, donor) - amount);
        initial.set(field, 1, initial.get(field, 1) + amount);
    }
    initial.set(Field::PendingCapacity, 302, 2);
    initial.set(Field::InstalledCapital, 302, 1);
    for workgroup in [32, 128] {
        let mut gpu =
            Simulation::from_state(scenario.clone(), initial.clone(), device(workgroup)?)?;
        let mut reference = Simulation::from_state(scenario.clone(), initial.clone(), device(64)?)?;
        let mut requests = vec![0; initial.actors.len()];
        for parent in parents {
            requests[parent] = 1 << 24 | 1;
        }
        requests[1] = Good::Bread as u32;
        requests[5] = (2 << 24) + 302;
        requests[300] = 4 << 24;
        requests[302] = (6 << 24) + 2;
        let mut births = vec![0; requests.len()];
        births[0] = 1;
        births[1] = 2;
        births[299] = 2;
        for simulation in [&gpu, &reference] {
            simulation
                .device
                .write(simulation.demography.birth_buffer(), &births);
        }
        let pending = registry::records::PendingRecords::from_dense(&requests, &births)?;
        let growth = registry::entries::EntryGrowth::new(
            &scenario,
            &gpu.metadata,
            &gpu.migrations,
            &pending.records,
        )?
        .context("household growth should use the GPU")?;
        gpu.commit_entries(&growth.requests, growth.plan, &pending.records)?;
        let mut expected = reference.download()?;
        expected.people = reference
            .demography
            .reference_births(&reference.device, &births, expected.next_identity)?
            .download(&reference.device, &expected, CopyDirection::Save)?;
        expected.next_identity += 3;
        enterprise::enter(&mut expected, &scenario, &requests)?;
        enterprise::transitions(&mut expected, &scenario, &requests)?;
        reference
            .replace(scenario.clone(), expected)
            .context("reference household rebuild")?;
        assert_eq!(gpu.download()?, reference.download()?);
        assert_eq!(gpu.snapshot, reference.snapshot);
        gpu.copy_audited(CopyDirection::Save)?;
        let checkpoint = gpu.checkpoint().context("grown household checkpoint")?;
        let restored = Simulation::from_checkpoint(checkpoint, device(workgroup)?)?;
        for simulation in [&gpu, &restored] {
            for actor in [0, 31, 299, 2] {
                let records = [PendingRecord {
                    actor,
                    request: 5 << 24,
                    birth: 0,
                }];
                let plan = registry::RegistryPlan::new(
                    &scenario,
                    &simulation.metadata,
                    &simulation.migrations,
                    &records,
                )?;
                assert!(plan.is_some());
                assert_eq!(
                    simulation
                        .migrations
                        .children(simulation.actors().at(actor as usize).identity)
                        .count(),
                    usize::from(actor != 2),
                );
            }
        }
        let before = gpu.download()?;
        let invalid = [enterprise::HouseholdFormation {
            parent: 0,
            household: gpu.actors().len(),
            person: u64::MAX,
        }];
        assert!(
            gpu.demography
                .form_households(&gpu.device, &invalid, gpu.actors().len() + 1)
                .is_err()
        );
        assert_eq!(gpu.download()?, before);
        for _ in 0..3 {
            gpu.advance()?;
            reference.advance()?;
            assert_eq!(gpu.download()?, reference.download()?);
            assert_eq!(gpu.snapshot, reference.snapshot);
        }
    }
    Ok(())
}

#[test]
fn gpu_firm_growth_matches_full_rebuild_with_births_and_migration() -> Result<()> {
    let mut config = Scenario::industrial(300, 42, Experiment::Baseline)?
        .config()
        .clone();
    config.demography.annual_births_per_million_fertile_women = 1_000_000;
    config.buffer_days = 30;
    config.household_cash = 1_000_000;
    let scenario = Scenario::new(config)?;
    let mut initial = scenario.initial()?;
    for actor in 0..3 {
        for (good, amount) in [(Good::Tools, 20), (Good::Wood, 50)] {
            let field = Field::inventory(good);
            let donor = (3..initial.actors.len())
                .max_by_key(|&index| initial.get(field, index))
                .expect("material supplier");
            initial.set(
                field,
                donor,
                initial
                    .get(field, donor)
                    .checked_sub(amount)
                    .context(format!(
                        "insufficient {good:?}: {}",
                        initial.get(field, donor)
                    ))?,
            );
            initial.set(field, actor, initial.get(field, actor) + amount);
        }
    }
    let mut gpu = Simulation::from_state(scenario.clone(), initial.clone(), device(64)?)?;
    let mut reference = Simulation::from_state(scenario.clone(), initial, device(128)?)?;
    for day in 0..4 {
        let source = scenario.config().households as usize + 2;
        let mut requests = vec![0; gpu.actors().len()];
        requests[0] = Good::Bread as u32;
        requests[1] = (8 << 24) + source as u32 + 1;
        requests[2] = (9 << 24) + source as u32 + 1;
        let closing = scenario.config().households as usize + day;
        requests[3] = (2 << 24) + closing as u32 + 1;
        requests[10] = (7 << 24) + closing as u32 + 1;
        requests[closing] = 4 << 24;
        let mut births = vec![0; requests.len()];
        births[0] = 1;
        births[299] = 2;
        for simulation in [&gpu, &reference] {
            simulation
                .device
                .write(simulation.demography.birth_buffer(), &births);
        }
        let pending = registry::records::PendingRecords::from_dense(&requests, &births)?;
        let entries =
            registry::entries::EntryRequests::new(&scenario, requests.len(), &pending.records)
                .expect("entry requests");
        if day == 0 {
            let before = gpu.download()?;
            let budget = gpu.device.memory_budget;
            gpu.device.memory_budget = MemoryBudget::Reported { available_bytes: 2 };
            let plan = registry::RegistryPlan::new(
                &scenario,
                &gpu.metadata,
                &gpu.migrations,
                &entries.transitions,
            )?
            .expect("supported transitions");
            assert!(
                gpu.commit_entries(&entries, plan, &pending.records)
                    .is_err()
            );
            gpu.device.memory_budget = budget;
            assert_eq!(gpu.download()?, before);
        }
        let plan = registry::RegistryPlan::new(
            &scenario,
            &gpu.metadata,
            &gpu.migrations,
            &entries.transitions,
        )?
        .expect("supported transitions");
        gpu.commit_entries(&entries, plan, &pending.records)
            .context("GPU growth")?;
        let mut expected = reference.download()?;
        expected.people = reference
            .demography
            .reference_births(&reference.device, &births, expected.next_identity)?
            .download(&reference.device, &expected, CopyDirection::Save)?;
        expected.next_identity += 2;
        enterprise::enter(&mut expected, &scenario, &requests)?;
        enterprise::transitions(&mut expected, &scenario, &requests)?;
        reference.replace(scenario.clone(), expected)?;
        assert_eq!(gpu.download()?, reference.download()?, "growth {day}");
        assert_eq!(gpu.snapshot, reference.snapshot, "growth {day}");
        gpu.copy_audited(CopyDirection::Save)?;
        gpu.advance()?;
        reference.advance()?;
        assert_eq!(gpu.download()?, reference.download()?, "day {day}");
        assert_eq!(gpu.snapshot, reference.snapshot, "day {day}");
    }
    gpu.checkpoint()?.restore()?;
    Ok(())
}

#[test]
fn sparse_settlements_match_rebuilds_with_shared_assets_migration_and_investment() -> Result<()> {
    use crate::domain::enterprise::{Ownership, OwnershipShare};
    let mut config = Scenario::industrial(1000, 42, Experiment::Baseline)?
        .config()
        .clone();
    config.research = None;
    config.household_cash = 5000;
    config.household_liquidity = crate::domain::HouseholdLiquidity::Uniform;
    let scenario = Scenario::new(config)?;
    for shareholders in [1, 2] {
        let mut initial = scenario.initial()?;
        initial.day = 31;
        let firm = scenario.config().households as usize;
        let identity = initial.actors.at(firm).identity;
        let ownership = initial
            .ownership
            .iter_mut()
            .find(|owned| owned.firm == identity)
            .unwrap();
        *ownership = Ownership::new(Ownership {
            firm: identity,
            shares: match shareholders {
                1 => vec![OwnershipShare {
                    household: 1,
                    units: 1_000_000,
                }],
                _ => vec![
                    OwnershipShare {
                        household: 1,
                        units: 333_333,
                    },
                    OwnershipShare {
                        household: 3,
                        units: 666_667,
                    },
                ],
            },
        })?;
        initial.set(Field::MoveDue, 5, 31);
        initial.set(Field::MoveX, 5, 123);
        initial.set(Field::MoveY, 5, 321);
        initial.set(Field::PendingCapacity, firm + 2, 3);
        initial.set(Field::InstalledCapital, firm + 2, 1);
        let shipment = initial
            .shipment_coordinate(
                Good::Grain,
                initial.day as usize % scenario.ring(),
                scenario.ring(),
                firm,
            )
            .0 as usize;
        initial.values[shipment] = 7;
        initial.set(Field::Grain, firm, initial.get(Field::Grain, firm) - 7);
        initial.values[TRANSIT_FIELDS * initial.actors.len() + firm] = 7;
        initial.set(Field::Transit, firm, 7);
        let mut gpu = Simulation::from_state(scenario.clone(), initial.clone(), device(64)?)?;
        let mut requests = vec![0; initial.actors.len()];
        requests[4] = (2 << 24) + 6;
        requests[5] = 3 << 24;
        requests[6] = (7 << 24) + firm as u32 + 1;
        requests[8] = (7 << 24) + firm as u32 + 2;
        requests[firm] = 4 << 24;
        requests[firm + 2] = (6 << 24) + 3;
        let pending =
            registry::records::PendingRecords::from_dense(&requests, &vec![0; requests.len()])?;
        gpu.device.write(&gpu.enterprise_requests, &requests);
        let plan = registry::RegistryPlan::new(
            &scenario,
            &gpu.metadata,
            &gpu.migrations,
            &pending.records,
        )?
        .context("sparse settlements")?;
        let budget = gpu.device.memory_budget;
        gpu.device.memory_budget = MemoryBudget::Reported { available_bytes: 2 };
        assert!(gpu.commit_registry(plan, &pending.records).is_err());
        gpu.device.memory_budget = budget;
        assert_eq!(gpu.download()?, initial);
        let plan = registry::RegistryPlan::new(
            &scenario,
            &gpu.metadata,
            &gpu.migrations,
            &pending.records,
        )?
        .context("sparse settlements")?;
        let readback = gpu.readback_bytes;
        gpu.commit_registry(plan, &pending.records)?;
        assert!(gpu.readback_bytes - readback < 20 * scenario.fields() as u64 * 8);
        enterprise::transitions(&mut initial, &scenario, &requests)?;
        let mut reference = Simulation::from_state(scenario.clone(), initial, device(128)?)?;
        assert_eq!(gpu.download()?, reference.download()?);
        assert_eq!(gpu.snapshot, reference.snapshot);
        gpu.copy_audited(CopyDirection::Save)?;
        for _ in 0..3 {
            gpu.advance()?;
            reference.advance()?;
            assert_eq!(gpu.download()?, reference.download()?);
            assert_eq!(gpu.snapshot, reference.snapshot);
        }
    }
    Ok(())
}

#[test]
fn company_investment_updates_every_site_and_closing_one_site_preserves_the_others() -> Result<()> {
    use crate::domain::{
        company::Company,
        enterprise::{EnterpriseState, Ownership, OwnershipShare},
    };
    let mut config = Scenario::baseline(8, 42, Experiment::Concentrated)?
        .config()
        .clone();
    config.entrepreneurship.interested_per_mille = 0;
    config.companies = vec![Company {
        identity: 9,
        sites: vec![9, 10],
    }];
    config.ownership = vec![Ownership {
        firm: 9,
        shares: vec![
            OwnershipShare {
                household: 1,
                units: 500_000,
            },
            OwnershipShare {
                household: 2,
                units: 500_000,
            },
        ],
    }];
    let scenario = Scenario::new(config)?;
    let mut expected = scenario.initial()?;
    let mut gpu = Simulation::from_state(scenario.clone(), expected.clone(), device(64)?)?;
    let mut requests = vec![0; expected.actors.len()];
    requests[2] = (7 << 24) + 10;
    requests[9] = 4 << 24;
    let pending =
        registry::records::PendingRecords::from_dense(&requests, &vec![0; requests.len()])?;
    gpu.device.write(&gpu.enterprise_requests, &requests);
    let plan =
        registry::RegistryPlan::new(&scenario, &gpu.metadata, &gpu.migrations, &pending.records)?
            .context("company settlement")?;
    gpu.commit_registry(plan, &pending.records)?;
    enterprise::transitions(&mut expected, &scenario, &requests)?;
    let actual = gpu.download()?;
    assert_eq!(actual, expected);
    let primary = actual
        .ownership
        .iter()
        .find(|entry| entry.firm == 9)
        .unwrap();
    let branch = actual
        .ownership
        .iter()
        .find(|entry| entry.firm == 10)
        .unwrap();
    assert_eq!(primary.shares, branch.shares);
    assert!(
        primary
            .shares
            .iter()
            .any(|share| share.household == 3 && share.units > 0)
    );
    assert_eq!(
        actual.get(Field::EnterpriseStatus, 8),
        EnterpriseState::Operating as u64
    );
    assert_eq!(
        actual.get(Field::EnterpriseStatus, 9),
        EnterpriseState::Closed as u64
    );
    gpu.copy_audited(CopyDirection::Save)?;
    gpu.checkpoint()?.restore()?;
    Ok(())
}

#[test]
fn device_markets_match_original_gpu_phases() -> Result<()> {
    for workgroup in [32, 64, 128, 256] {
        let scenario = Scenario::industrial(193, 42, Experiment::Concentrated)?;
        let mut actual = Simulation::new(scenario.clone(), Device::headless(workgroup)?)?;
        let mut expected = Simulation::new(scenario, Device::headless(64)?)?;
        expected.market_execution = MarketExecution::Original;
        expected.audit_execution = AuditExecution::Original;
        for day in 0..4 {
            for phase in Phase::ALL {
                let reference = expected.phase(phase, Trace::Record)?;
                let result = actual.phase(phase, Trace::Record)?;
                assert_eq!(
                    result.decisions, reference.decisions,
                    "{workgroup} {day} {phase:?}"
                );
                assert_eq!(
                    result.frontier_depth, reference.frontier_depth,
                    "{workgroup} {day} {phase:?}"
                );
                assert_eq!(
                    result.events, reference.events,
                    "{workgroup} {day} {phase:?}"
                );
                assert_eq!(
                    actual.download()?,
                    expected.download()?,
                    "{workgroup} {day} {phase:?}"
                );
                assert_eq!(actual.snapshot, expected.snapshot);
            }
        }
    }
    Ok(())
}

#[test]
fn optimized_experiments_continue_original_checkpoints_through_day_fifty() -> Result<()> {
    for experiment in [Experiment::Baseline, Experiment::Concentrated] {
        let mut config = Scenario::industrial(300, 42, experiment)?.config().clone();
        config.demography.annual_births_per_million_fertile_women = 1_000_000;
        let scenario = Scenario::new(config)?;
        let mut expected = Simulation::new(scenario.clone(), Device::headless(64)?)?;
        expected.market_execution = MarketExecution::Original;
        expected.audit_execution = AuditExecution::Original;
        expected.registry_execution = RegistryExecution::Reference;
        let mut actual = Simulation::new(scenario, Device::headless(128)?)?;
        for day in 0..50 {
            if matches!(day, 29 | 39) {
                actual =
                    Simulation::from_checkpoint(expected.checkpoint()?, Device::headless(256)?)?;
            }
            for phase in Phase::ALL {
                let reference = expected.phase(phase, Trace::Record)?;
                let result = actual.phase(phase, Trace::Record)?;
                assert_eq!(
                    result.events, reference.events,
                    "{experiment:?} {day} {phase:?}"
                );
                assert_eq!(
                    result.decisions, reference.decisions,
                    "{experiment:?} {day} {phase:?}"
                );
                assert_eq!(
                    actual.download()?,
                    expected.download()?,
                    "{experiment:?} {day} {phase:?}"
                );
                assert_eq!(actual.snapshot, expected.snapshot);
            }
        }
        assert_eq!(actual.checkpoint()?, expected.checkpoint()?);
    }
    Ok(())
}

#[test]
fn capacity_changes_preserve_spatial_membership_and_rollback_preparation() -> Result<()> {
    let scenario = Scenario::industrial(97, 42, Experiment::Baseline)?;
    let mut simulation = Simulation::new(scenario.clone(), Device::headless(64)?)?;
    let mut initial = simulation.download()?;
    initial.set(Field::PendingCapacity, 97, 5);
    simulation.device.write(
        &simulation.state,
        bytemuck::cast_slice(&encode(&initial.values)),
    );
    let spatial = simulation.device.read(&simulation.spatial)?;
    let mut requests = vec![0; simulation.actors().len()];
    requests[97] = (6 << 24) | 5;
    let pending =
        registry::records::PendingRecords::from_dense(&requests, &vec![0; requests.len()])?;
    simulation
        .device
        .write(&simulation.enterprise_requests, &requests);
    let plan = registry::RegistryPlan::new(
        &scenario,
        &simulation.metadata,
        &simulation.migrations,
        &pending.records,
    )?
    .context("capacity transition")?;
    let budget = simulation.device.memory_budget;
    simulation.device.memory_budget = MemoryBudget::Reported { available_bytes: 2 };
    assert!(simulation.commit_registry(plan, &[]).is_err());
    simulation.device.memory_budget = budget;
    assert_eq!(simulation.download()?, initial);
    let plan = registry::RegistryPlan::new(
        &scenario,
        &simulation.metadata,
        &simulation.migrations,
        &pending.records,
    )?
    .context("capacity transition")?;
    simulation.commit_registry(plan, &[])?;
    let mut expected = initial;
    enterprise::transitions(&mut expected, &scenario, &requests)?;
    assert_eq!(simulation.download()?, expected);
    assert_eq!(simulation.device.read(&simulation.spatial)?, spatial);
    Ok(())
}

#[test]
fn restored_zero_capacity_growth_matches_complete_gpu_metadata() -> Result<()> {
    let scenario = Scenario::industrial(97, 42, Experiment::Baseline)?;
    let mut initial = scenario.initial()?;
    let mut firm = initial.actors.at(97);
    firm.capacity = 0;
    initial.actors.set(97, firm);
    initial.set(Field::InstalledCapital, 97, 1);
    initial.set(Field::PendingCapacity, 97, 5);
    let checkpoint = Checkpoint::new(&scenario, &initial)?;
    let mut requests = vec![0; initial.actors.len()];
    requests[97] = (6 << 24) | 5;
    let pending =
        registry::records::PendingRecords::from_dense(&requests, &vec![0; requests.len()])?;
    for workgroup in [32, 64, 128, 256] {
        let mut simulation =
            Simulation::from_checkpoint(checkpoint.clone(), Device::headless(workgroup)?)?;
        let mut expected = simulation.download()?;
        enterprise::transitions(&mut expected, &scenario, &requests)?;
        let spatial = simulation.device.read(&simulation.spatial)?;
        let rows = expected
            .actors
            .iter()
            .map(|actor| ActorLayout::new(&actor, &scenario))
            .collect::<ActorMetadataRows>();
        simulation
            .device
            .write(&simulation.enterprise_requests, &requests);
        let plan = registry::RegistryPlan::new(
            &scenario,
            &simulation.metadata,
            &simulation.migrations,
            &pending.records,
        )?
        .context("restored capacity transition")?;
        simulation.commit_registry(plan, &[])?;
        assert_eq!(simulation.download()?.digest(), expected.digest());
        assert_eq!(simulation.device.read(&simulation.spatial)?, spatial);
        for (index, (actual, expected)) in simulation
            .device
            .read(&simulation.actors)?
            .into_iter()
            .zip(rows.words())
            .enumerate()
        {
            assert_eq!(
                actual, expected,
                "actor metadata word {index}, workgroup {workgroup}"
            );
        }
    }
    Ok(())
}

#[test]
fn speculative_candidates_only_commit_faults_inside_the_visited_prefix() -> Result<()> {
    let mut config = Scenario::baseline(1, 42, Experiment::Concentrated)?
        .config()
        .clone();
    config.firms = 9;
    config.bargaining = Some(crate::domain::bargaining::BargainingPolicy {
        legal_minimum: 1,
        quantum: 1,
        daily_cash_cost: 1_000_000,
        funded_other: 0,
        participating_adults: 1,
        normal_day_minutes: 1440,
        work_cost_per_minute: 0,
        desired_wage: 1000,
        mandate: 500,
        withdrawal: 0,
        alternatives: 0,
        concentration: 0,
        replacements: 0,
    });
    config.household_production = crate::domain::enterprise::HouseholdProduction::industrial();
    config.population = Some(Population {
        households: vec![HouseholdSeed {
            identity: 1,
            position: Position { x: 500, y: 500 },
            cash: 5000,
            inventories: Good::food([0, 0, 100]),
            fatigue: 0,
        }],
        firms: (2..=10)
            .map(|identity| FirmSeed {
                identity,
                position: Position { x: 500, y: 500 },
                owner: 1,
                good: Good::Bread,
                cash: 5000,
                inventories: [0; Good::COUNT],
                capacity: 1,
                technique: Technique::new(1, 1, 1).expect("one-minute packet"),
                wage: 1,
                price: 1,
            })
            .collect(),
    });
    let scenario = Scenario::new(config)?;
    for available in [8, 7] {
        let mut outcomes = Vec::new();
        for execution in [MarketExecution::Grouped, MarketExecution::Original] {
            let mut simulation = Simulation::new(scenario.clone(), Device::headless(64)?)?;
            simulation.market_execution = execution;
            let mut state = simulation.download()?;
            state.set(Field::Time, 0, u64::from(u32::MAX));
            for firm in 1..=9 {
                state.set(
                    Field::Slots,
                    firm,
                    u64::from(firm <= available || firm == 9),
                );
                state.set(Field::LaborMinutes, firm, 1);
                state.set(Field::InputTicks, firm, 0);
                state.set(Field::EquipmentCondition, firm, 1000);
            }
            state.set(Field::LaborMinutes, 9, u64::from(u32::MAX));
            state.set(Field::EquipmentCondition, 9, u64::from(u32::MAX));
            simulation.device.write(
                &simulation.state,
                bytemuck::cast_slice(&encode(&state.values)),
            );
            let mut evidence = PhaseEvidence::default();
            let result = simulation.market(
                Market::Labor,
                Phase::Commitment,
                Trace::Record,
                &mut evidence,
            );
            outcomes.push(result.as_ref().err().map(ToString::to_string));
            match available {
                8 => {
                    result?;
                    assert_eq!(
                        simulation.device.read(&simulation.proposals)?[ProposalLayout::ACTOR],
                        1
                    );
                }
                _ => assert!(
                    outcomes
                        .last()
                        .unwrap()
                        .as_ref()
                        .context("visited fault")?
                        .contains("ARITHMETIC_OVERFLOW")
                ),
            }
        }
        assert_eq!(outcomes[0], outcomes[1]);
    }
    Ok(())
}

#[test]
fn vectorized_opening_matches_serial_deliveries_resets_and_faults_on_gpu() -> Result<()> {
    let scenario = Scenario::baseline(97, 42, Experiment::Transport)?;
    let initial = scenario.initial()?;
    let count = initial.actors.len();
    for workgroup in [32, 64, 128, 256] {
        let simulation = Simulation::new(scenario.clone(), device(workgroup)?)?;
        for day in [0, 1, simulation.config.ring - 1, simulation.config.ring + 1] {
            let mut values = initial.values.clone();
            for actor in 0..count {
                let mut transit = 0;
                for good in 0..Good::COUNT {
                    let delivery = (1u64 << 32) + (actor * Good::COUNT + good) as u64;
                    let slot =
                        BASE_FIELDS + good * scenario.ring() + day as usize % scenario.ring();
                    values[slot * count + actor] = delivery;
                    values[(TRANSIT_FIELDS + good) * count + actor] = delivery;
                    transit += delivery;
                }
                values[Field::Transit as usize * count + actor] = transit;
                values[Field::ResearchMinutes as usize * count + actor] = 123;
            }
            values[Field::Grain as usize * count] = u64::MAX;
            values[(TRANSIT_FIELDS + Good::COUNT - 1) * count + count - 1] = 0;
            let words = encode(&values);
            let actual = simulation.device.tensor(bytemuck::cast_slice(&words));
            let expected = simulation.device.tensor(bytemuck::cast_slice(&words));
            let actual_control = simulation.device.tensor(&[0; CONTROL_WORDS]);
            let expected_control = simulation.device.tensor(&[0; CONTROL_WORDS]);
            let mut parameters = simulation.parameters(Phase::Opening, Market::Labor, 0, count)?;
            parameters.day = day;
            economy::EconomySystem::launch(
                &simulation.device,
                simulation.device.count(count),
                cubecl::CubeDim::new_1d(workgroup),
                economy::EconomyInputLaunch {
                    state: expected.clone().into_buffer_arg(),
                    actors: simulation.actors.clone().into_buffer_arg(),
                    config: simulation.config.launch(),
                    spatial: simulation.spatial.clone().into_buffer_arg(),
                    proposals: simulation.proposals.clone().into_buffer_arg(),
                    bundles: simulation.events.clone().into_buffer_arg(),
                    control: expected_control.clone().into_buffer_arg(),
                    parameters: parameters.launch(),
                    stage: EconomyStage::Local,
                },
            );
            economy::opening::launch(
                &simulation.device,
                &actual,
                &simulation.config,
                &actual_control,
                day,
            );
            assert_eq!(
                simulation.device.read(&actual)?,
                simulation.device.read(&expected)?
            );
            let faults = FaultFlags::read(&simulation.device.read(&actual_control)?)?;
            assert_eq!(
                faults,
                FaultFlags::read(&simulation.device.read(&expected_control)?)?
            );
            assert_eq!(
                faults,
                FaultFlags::ARITHMETIC_OVERFLOW | FaultFlags::TRANSIT_UNDERFLOW
            );
        }
    }
    Ok(())
}

#[test]
fn partitioned_audits_match_original_tiles_extrema_and_overflow_on_gpu() -> Result<()> {
    #[derive(Clone, Copy)]
    enum Population {
        Households,
        SparseFirms,
        MultipleFirmTiles,
    }
    #[derive(Clone, Copy)]
    enum Amounts {
        Wide,
        Overflow,
    }
    for workgroup in [32, 64, 128, 256] {
        for population in [
            Population::Households,
            Population::SparseFirms,
            Population::MultipleFirmTiles,
        ] {
            let mut config = Scenario::baseline(4097, 42, Experiment::Baseline)?
                .config()
                .clone();
            config.firms = match population {
                Population::Households => 3,
                Population::SparseFirms => 7,
                Population::MultipleFirmTiles => 4097,
            };
            let scenario = Scenario::new(config)?;
            let mut simulation = Simulation::new(scenario.clone(), device(workgroup)?)?;
            let mut initial = scenario.initial()?;
            for index in 0..initial.actors.len() {
                let mut actor = initial.actors.at(index);
                if let Population::Households = population {
                    actor.kind = 0;
                    initial.actors.set(index, actor.clone());
                }
                let wage = match actor.kind {
                    0 => u64::MAX,
                    _ => (1u64 << 40) + index as u64,
                };
                initial.set(Field::Wage, index, wage);
                initial.set(Field::Price, index, (1u64 << 41) + index as u64);
                initial.set(Field::EnterpriseStatus, index, 4 * (index % 2) as u64);
                initial.set(Field::EnterpriseModel, index, (index % 2) as u64);
                initial.set(Field::HousingStatus, index, (index % 2) as u64);
                initial.set(Field::LivingMembers, index, (index % 3) as u64);
            }
            simulation.actors = simulation.device.tensor(
                &initial
                    .actors
                    .iter()
                    .map(|actor| ActorLayout::new(&actor, &scenario))
                    .collect::<ActorMetadataRows>()
                    .words(),
            );
            simulation.spatial = simulation
                .device
                .tensor(&SpatialIndex::new(&scenario, &initial).words);
            simulation.audit =
                economy::audit::AuditBuffers::new(&simulation.device, &initial.actors);
            simulation.metadata.actors = initial.actors.clone();
            for amounts in [Amounts::Wide, Amounts::Overflow] {
                if let Amounts::Overflow = amounts {
                    for index in 0..initial.actors.len() {
                        initial.set(Field::Wage, index, u64::from(u32::MAX));
                    }
                }
                simulation.device.write(
                    &simulation.state,
                    bytemuck::cast_slice(&encode(&initial.values)),
                );
                let poison = match amounts {
                    Amounts::Wide => u32::MAX,
                    Amounts::Overflow => 0,
                };
                simulation.device.write(
                    &simulation.events,
                    &vec![poison; simulation.events.meta.num_elements()],
                );
                simulation.audit_execution = AuditExecution::Tiled;
                let actual = simulation.observe();
                let totals =
                    simulation
                        .device
                        .read_range(&simulation.summary, 0, SummaryLayout::WORDS)?;
                let control = simulation.device.read(&simulation.control)?;
                simulation.audit_execution = AuditExecution::Original;
                let expected = simulation.observe();
                assert_eq!(actual.is_ok(), expected.is_ok());
                assert_eq!(
                    totals,
                    simulation
                        .device
                        .read_range(&simulation.summary, 0, SummaryLayout::WORDS)?
                );
                assert_eq!(control, simulation.device.read(&simulation.control)?);
                let faults = match (amounts, population) {
                    (
                        Amounts::Overflow,
                        Population::SparseFirms | Population::MultipleFirmTiles,
                    ) => FaultFlags::ARITHMETIC_OVERFLOW,
                    _ => FaultFlags::empty(),
                };
                assert_eq!(FaultFlags::read(&control)?, faults);
                if let (Ok(actual), Ok(expected)) = (actual, expected) {
                    assert_eq!(actual, expected);
                }
            }
        }
    }
    Ok(())
}

#[test]
fn parallel_transit_audits_match_serial_mismatch_and_overflow_faults_on_gpu() -> Result<()> {
    let scenario = Scenario::baseline(97, 42, Experiment::Transport)?;
    for workgroup in [32, 64, 128, 256] {
        let mut simulation = Simulation::new(scenario.clone(), device(workgroup)?)?;
        let mut state = scenario.initial()?;
        let count = state.actors.len();
        state.values[(BASE_FIELDS + scenario.ring() * (Good::COUNT - 1)) * count + count - 1] =
            u64::MAX;
        state.values[(BASE_FIELDS + scenario.ring() * (Good::COUNT - 1) + 1) * count + count - 1] =
            1;
        state.values[(TRANSIT_FIELDS + Good::COUNT - 1) * count + count - 1] = 7;
        simulation.device.write(
            &simulation.state,
            bytemuck::cast_slice(&encode(&state.values)),
        );
        assert!(simulation.observe().is_err());
        let actual = FaultFlags::read(&simulation.device.read(&simulation.control)?)?;
        simulation.audit_execution = AuditExecution::Original;
        assert!(simulation.observe().is_err());
        let expected = FaultFlags::read(&simulation.device.read(&simulation.control)?)?;
        assert_eq!(actual, expected);
        assert_eq!(
            actual,
            FaultFlags::ARITHMETIC_OVERFLOW | FaultFlags::TRANSIT_MISMATCH
        );
    }
    Ok(())
}

#[test]
fn property_allocation_fault_restores_the_audited_gpu_state() -> Result<()> {
    let config = serde_json::from_str(include_str!("../../scenarios/property.json"))?;
    let scenario = Scenario::new(config)?;
    let mut initial = scenario.initial()?;
    initial.set(Field::DisplacedPlots, 1, u64::MAX);
    let mut simulation = Simulation::from_state(scenario, initial, device(64)?)?;
    let before = simulation.download()?;
    assert!(simulation.phase(Phase::Opening, Trace::Discard).is_err());
    assert!(matches!(simulation.status, RuntimeState::Fault { .. }));
    assert_eq!(simulation.download()?, before);
    Ok(())
}

use super::*;
use std::{
    collections::{HashMap, hash_map::Entry},
    sync::{Mutex, OnceLock},
};

fn device(workgroup: u32) -> Result<Device> {
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

#[test]
fn industrial_recipes_match_coordinate_scheduler() -> Result<()> {
    let mut config = Scenario::industrial(64, 42, Experiment::Concentrated)?
        .config()
        .clone();
    config.batch_size = 13;
    for lab in &mut config.research.as_mut().expect("industrial research").labs {
        lab.discovered = 31;
        lab.mastered = 31;
        lab.adopted = 31;
    }
    compare_market_execution(Scenario::new(config)?, 4)
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
        requests[3] = (2 << 24) + 5;
        let entries =
            registry::entries::EntryRequests::new(&scenario, &requests).expect("entry requests");
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
                gpu.commit_entries(&requests, &entries, plan, &vec![0; requests.len()])
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
        let births = vec![0; requests.len()];
        gpu.commit_entries(&requests, &entries, plan, &births)
            .context("GPU growth")?;
        let mut expected = reference.download()?;
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
fn optimized_experiments_continue_original_checkpoints_across_day_thirty() -> Result<()> {
    for experiment in [
        Experiment::Baseline,
        Experiment::Shortage,
        Experiment::Transport,
        Experiment::Concentrated,
    ] {
        let mut config = Scenario::industrial(300, 42, experiment)?.config().clone();
        config.demography.annual_births_per_million_fertile_women = 1_000_000;
        let scenario = Scenario::new(config)?;
        let mut expected = Simulation::new(scenario.clone(), Device::headless(64)?)?;
        expected.market_execution = MarketExecution::Original;
        expected.audit_execution = AuditExecution::Original;
        expected.registry_execution = RegistryExecution::Reference;
        let mut actual = Simulation::new(scenario, Device::headless(128)?)?;
        for day in 0..33 {
            if day == 29 {
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
    simulation
        .device
        .write(&simulation.enterprise_requests, &requests);
    let plan = registry::RegistryPlan::new(
        &scenario,
        &simulation.metadata,
        &simulation.migrations,
        &requests,
    )?
    .context("capacity transition")?;
    let budget = simulation.device.memory_budget;
    simulation.device.memory_budget = MemoryBudget::Reported { available_bytes: 2 };
    assert!(
        simulation
            .commit_registry(plan, &vec![0; requests.len()])
            .is_err()
    );
    simulation.device.memory_budget = budget;
    assert_eq!(simulation.download()?, initial);
    let plan = registry::RegistryPlan::new(
        &scenario,
        &simulation.metadata,
        &simulation.migrations,
        &requests,
    )?
    .context("capacity transition")?;
    simulation.commit_registry(plan, &vec![0; requests.len()])?;
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
            &requests,
        )?
        .context("restored capacity transition")?;
        simulation.commit_registry(plan, &vec![0; requests.len()])?;
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

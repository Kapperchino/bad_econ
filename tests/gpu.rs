#![cfg(feature = "compute")]
use anyhow::{Context, Result};
use bad_econ::{
    domain::*,
    ledger::*,
    simulation::{self, Device, MemoryBudget, Simulation, Trace},
};
use std::collections::{BTreeMap, btree_map::Entry};

#[test]
fn gpu_memory_budget_rejects_scenarios_before_allocation() -> Result<()> {
    let scenario = Scenario::baseline(1_000_000, 42, Experiment::Transport)?;
    assert!(scenario.memory_estimate() > 4 * 1024 * 1024 * 1024);
    let mut device = Device::headless(64)?;
    eprintln!("{} memory budget: {:?}", device.name, device.memory_budget);
    assert!(matches!(
        device.memory_budget,
        MemoryBudget::Reported { .. }
    ));
    assert!(device.memory_budget.bytes() > 0);
    device.memory_budget = MemoryBudget::Reported {
        available_bytes: (scenario.memory_estimate() - 1) * 2,
    };
    let client = device.client.clone();
    let reserved = client
        .memory_report(cubecl::MemoryScope::Device)
        .usage()
        .bytes_reserved;
    let error = Simulation::new(scenario, device)
        .err()
        .context("scenario should exceed the selected device budget")?;
    assert!(
        error
            .to_string()
            .contains("exceeding the device memory budget")
    );
    assert_eq!(
        client
            .memory_report(cubecl::MemoryScope::Device)
            .usage()
            .bytes_reserved,
        reserved
    );
    Ok(())
}

#[test]
fn gpu_memory_budget_survives_restore_and_faults_on_excess_usage() -> Result<()> {
    use bad_econ::runtime::{RuntimeState, SimulationCommand};

    let scenario = Scenario::baseline(48, 42, Experiment::Baseline)?;
    let device = Device::headless(64)?;
    let budget = device.memory_budget.bytes();
    let mut simulation = Simulation::new(scenario, device)?;
    let checkpoint = simulation.checkpoint()?;
    simulation.advance()?;
    simulation.command(SimulationCommand::Restore(checkpoint.into()))?;
    assert_eq!(simulation.device.memory_budget.bytes(), budget);
    simulation.device.memory_budget = MemoryBudget::Reported { available_bytes: 2 };
    assert!(simulation.advance().is_err());
    assert!(matches!(simulation.status, RuntimeState::Fault { .. }));
    Ok(())
}

#[test]
fn released_large_gpu_buffers_do_not_cause_a_false_capacity_fault() -> Result<()> {
    let scenario = Scenario::industrial(48, 42, Experiment::Baseline)?;
    let mut constrained = Simulation::new(scenario, Device::headless(64)?)?;
    constrained.advance()?;
    let mut reference =
        Simulation::from_checkpoint(constrained.checkpoint()?, Device::headless(128)?)?;
    let before = constrained.memory_usage().bytes_reserved;
    let spare = constrained
        .device
        .empty((before as usize + 64 * 1024 * 1024) / size_of::<u32>());
    constrained.device.sync()?;
    let inflated = constrained.memory_usage().bytes_reserved;
    drop(spare);
    let budget = before + 32 * 1024 * 1024;
    assert!(inflated > budget);
    constrained.device.memory_budget = MemoryBudget::Reported {
        available_bytes: budget * 2,
    };
    for _ in 0..5 {
        constrained.advance()?;
        reference.advance()?;
        assert!(constrained.memory_usage().bytes_reserved <= budget);
        assert_eq!(constrained.download()?, reference.download()?);
    }
    Ok(())
}

#[test]
fn growing_gpu_readbacks_preserve_ranges_and_bound_cached_staging() -> Result<()> {
    let device = Device::headless(64)?;
    let before = device
        .client
        .memory_report(cubecl::MemoryScope::Device)
        .usage();
    for generation in 0..24usize {
        let count = 4 * 1024 * 1024 + generation * 101 + 17;
        let expected = (0..count)
            .map(|index| index as u32 ^ 0x923a5ce1)
            .collect::<Vec<_>>();
        let buffer = device.tensor(&expected);
        assert_eq!(device.read(&buffer)?, expected);
        for (start, length) in [(13, count - 37), (count - 1027, 1027), (count, 0)] {
            assert_eq!(
                device.read_range(&buffer, start, length)?,
                expected[start..start + length]
            );
        }
    }
    device.sync()?;
    device.client.memory_cleanup()?;
    let after = device
        .client
        .memory_report(cubecl::MemoryScope::Device)
        .usage();
    assert!(
        after.bytes_reserved <= before.bytes_reserved + 96 * 1024 * 1024,
        "{after:?}"
    );
    Ok(())
}

#[test]
fn scarce_work_is_accessible_beyond_the_first_identity_block() -> Result<()> {
    let mut config = Scenario::baseline(1000, 42, Experiment::Concentrated)?
        .config()
        .clone();
    config.firms = 3;
    config.distribution_rate = 0;
    config.forecast_weight = 0;
    config.demography.annual_births_per_million_fertile_women = 0;
    config.demography.mortality.background_annual_ppm = [0; 5];
    config.demography.mortality.occupational_annual_ppm = [0; Good::COUNT];
    config.demography.mortality.famine_daily_ppm_at_thirty_days = 0;
    config.resources = ResourceMap::Explicit {
        regions: vec![ResourceRegion::new(
            Good::Grain,
            Position { x: 0, y: 0 },
            Position { x: 1000, y: 1000 },
            1000,
        )?],
    };
    config.population = Some(Population {
        households: (1..=1000)
            .map(|identity| HouseholdSeed {
                identity,
                position: Position { x: 500, y: 500 },
                cash: 0,
                inventories: [0; Good::COUNT],
                fatigue: 0,
            })
            .collect(),
        firms: (1001..=1003)
            .map(|identity| FirmSeed {
                identity,
                position: Position { x: 500, y: 500 },
                owner: 1,
                good: Good::Grain,
                cash: if identity == 1001 { 1_000_000 } else { 0 },
                inventories: [0; Good::COUNT],
                capacity: 25,
                technique: Technique {
                    labor_minutes: 60,
                    input_ticks: 0,
                    output_ticks: 1,
                },
                wage: 1,
                price: 1,
            })
            .collect(),
    });
    let scenario = Scenario::new(config)?;
    let mut state = scenario.initial()?;
    state.set(Field::Demand, 1000, 1000);
    let checkpoint = Checkpoint::new(&scenario, &state)?;
    let mut simulation = Simulation::from_checkpoint(checkpoint.clone(), Device::headless(64)?)?;
    let mut replay = Simulation::from_checkpoint(checkpoint, Device::headless(128)?)?;
    let mut worked = vec![0; 1000];
    for _ in 0..24 {
        simulation.advance()?;
        replay.advance()?;
        let state = simulation.download()?;
        assert_eq!(state, replay.download()?);
        assert_eq!(simulation.snapshot.employed, 25);
        for (index, days) in worked.iter_mut().enumerate() {
            *days += state.get(Field::Employed, index);
        }
    }
    assert!(worked.iter().filter(|&&days| days > 0).count() > 200);
    assert!(worked[750..].iter().any(|&days| days > 0));
    assert_eq!(worked.iter().sum::<u64>(), 24 * 25);
    simulation.checkpoint()?.restore()?;
    Ok(())
}

fn event(identity: u64, legs: &[Leg]) -> Bundle {
    Bundle::new(
        EventKey {
            day: 0,
            phase: 0,
            subphase: 0,
            priority: 0,
            identity,
        },
        legs.iter().copied(),
    )
    .unwrap()
}

#[test]
fn gpu_integer_ledger() -> Result<()> {
    let device = Device::headless(64)?;
    let events = [
        event(
            1,
            &[
                Leg::transfer(Coordinate(0), 3, 0),
                Leg::transfer(Coordinate(1), 0, 3),
            ],
        ),
        event(
            2,
            &[
                Leg::read(Coordinate(0), 2),
                Leg::transfer(Coordinate(1), 0, 1),
            ],
        ),
        event(
            3,
            &[
                Leg::transfer(Coordinate(0), 2, 0),
                Leg::transfer(Coordinate(1), 0, 2),
            ],
        ),
    ];
    let result = simulation::ledger(&device, &[2, u32::MAX as u64], &events)?;
    assert_eq!(result.state, [0, u32::MAX as u64 + 3]);
    assert_eq!(
        result.decisions,
        [Decision::Rejected, Decision::Accepted, Decision::Accepted]
    );
    assert_eq!(result.frontier_depth, 3);
    let overflow = [event(1, &[Leg::transfer(Coordinate(0), 0, 1)])];
    assert!(simulation::ledger(&device, &[u64::MAX], &overflow).is_err());
    assert_eq!(simulation::ledger(&device, &[1], &[])?.state, [1]);
    assert_eq!(
        simulation::ledger(&device, &[1], &[event(1, &[])])?.decisions,
        [Decision::Accepted]
    );
    Ok(())
}

#[derive(Debug, PartialEq, Eq)]
struct PhaseReplay {
    decisions: Vec<Decision>,
    state_hash: String,
}

#[test]
fn gpu_phases_match_across_workgroups_and_experiments() -> Result<()> {
    let mut canonical = BTreeMap::new();
    for workgroup in [32, 64, 128, 256] {
        for experiment in [
            Experiment::Baseline,
            Experiment::Shortage,
            Experiment::Transport,
            Experiment::Concentrated,
        ] {
            let mut config = Scenario::baseline(48, 42, experiment)?.config().clone();
            config.firms = 12;
            config.batch_size = 7;
            let scenario = Scenario::new(config)?;
            let mut gpu = Simulation::new(scenario, Device::headless(workgroup)?)?;
            let mut replay = Vec::new();
            for _ in 0..8 {
                for phase in Phase::ALL {
                    let actual = gpu.phase(phase, Trace::Record)?;
                    replay.push(PhaseReplay {
                        decisions: actual.decisions,
                        state_hash: gpu.download()?.digest(),
                    });
                }
                let state = gpu.download()?;
                assert_eq!(
                    gpu.snapshot,
                    ObservationSnapshot::from_state(&gpu.scenario, &state)?
                );
            }
            match canonical.entry(format!("{experiment:?}")) {
                Entry::Vacant(entry) => {
                    entry.insert(replay);
                }
                Entry::Occupied(entry) => {
                    assert_eq!(
                        &replay,
                        entry.get(),
                        "{workgroup} {experiment:?} phase replay"
                    );
                }
            }
        }
    }
    Ok(())
}

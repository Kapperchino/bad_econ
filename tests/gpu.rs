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

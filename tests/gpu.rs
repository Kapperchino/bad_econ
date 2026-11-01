#![cfg(feature = "compute")]
use anyhow::Result;
use bad_econ::{
    domain::*,
    ledger::*,
    simulation::{self, Device, Simulation, Trace},
};
use std::collections::{BTreeMap, btree_map::Entry};

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

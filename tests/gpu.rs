#![cfg(feature = "compute")]
use anyhow::Result;
use bad_econ::{
    domain::*,
    ledger::*,
    simulation::{self, Device, Simulation, Trace},
};

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

#[test]
fn every_phase_matches_cpu_across_workgroups_and_experiments() -> Result<()> {
    let mut canonical = std::collections::BTreeMap::new();
    for workgroup in [32, 64, 128, 256, 64] {
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
            let mut cpu = Simulation::new(scenario.clone(), Device::cpu(workgroup)?)?;
            let mut gpu = Simulation::new(scenario, Device::headless(workgroup)?)?;
            for day in 0..8 {
                for phase in Phase::ALL {
                    let expected = cpu.phase(phase, Trace::Record)?;
                    let actual = gpu.phase(phase, Trace::Record)?;
                    assert_eq!(
                        actual.decisions, expected.decisions,
                        "{workgroup} {experiment:?} day {day} phase {phase:?} decisions"
                    );
                    let actual = gpu.download()?;
                    assert_eq!(
                        actual.values,
                        cpu.download()?.values,
                        "{workgroup} {experiment:?} day {day} phase {phase:?} values"
                    );
                    assert_eq!(actual.day, cpu.download()?.day);
                }
                let state = cpu.download()?;
                assert_eq!(
                    gpu.snapshot,
                    ObservationSnapshot::from_state(&cpu.scenario, &state)?
                );
            }
            let state = gpu.download()?;
            if let Some(expected) = canonical.insert(format!("{experiment:?}"), state.clone()) {
                assert_eq!(state, expected, "repeated run and workgroup invariance");
            }
        }
    }
    Ok(())
}

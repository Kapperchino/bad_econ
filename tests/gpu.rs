#![cfg(feature = "compute")]
mod support;

use anyhow::{Context, Result};
use bad_econ::{
    domain::*,
    simulation::{MemoryBudget, Simulation},
};
use cubecl::prelude::*;

#[cube(launch)]
fn write_buffer_probes(buffer: &mut [u32], positions: &[u32], seed: u32) {
    if ABSOLUTE_POS < positions.len() {
        let position = positions[ABSOLUTE_POS];
        buffer[position as usize] = position ^ seed;
    }
}

#[test]
#[ignore = "allocates the reported 3.46 GB buffer; requires sufficient GPU memory"]
fn gpu_buffer_larger_than_two_gib_preserves_distant_writes() -> Result<()> {
    let device = support::device(64)?;
    let words = 3_458_240_000 / size_of::<u32>();
    let boundary = (1usize << 31) / size_of::<u32>();
    let positions =
        [0, boundary - 1, boundary, boundary + 1, words - 1].map(|position| position as u32);
    eprintln!("{}: {} bytes", device.name, words * size_of::<u32>());
    assert!(device.memory_budget.bytes() >= (words * size_of::<u32>()) as u64);
    let buffer = device.empty(words);
    let probes = device.tensor(&positions);
    for seed in [0x923a5ce1, 0x6dc5a31e] {
        write_buffer_probes::launch(
            &device.client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(device.workgroup),
            buffer.clone().into_buffer_arg(),
            probes.clone().into_buffer_arg(),
            seed,
        );
        for position in positions {
            assert_eq!(
                device.read_range(&buffer, position as usize, 1)?,
                vec![position ^ seed],
                "word {position}"
            );
        }
    }
    Ok(())
}

#[test]
fn gpu_memory_budget_rejects_scenarios_before_allocation() -> Result<()> {
    let scenario = Scenario::baseline(1_000_000, 42, Experiment::Transport)?;
    assert!(scenario.memory_estimate() > 4 * 1024 * 1024 * 1024);
    let mut device = support::device(64)?;
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
    let device = support::device(64)?;
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
    let scenario = Scenario::baseline(48, 42, Experiment::Baseline)?;
    let mut constrained = Simulation::new(scenario, support::device(64)?)?;
    constrained.advance()?;
    let mut reference =
        Simulation::from_checkpoint(constrained.checkpoint()?, support::device(64)?)?;
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
    for _ in 0..2 {
        constrained.advance()?;
        reference.advance()?;
        assert!(constrained.memory_usage().bytes_reserved <= budget);
        assert_eq!(constrained.download()?, reference.download()?);
    }
    Ok(())
}

#[test]
fn growing_gpu_readbacks_preserve_ranges_and_bound_cached_staging() -> Result<()> {
    let device = support::device(64)?;
    let before = device
        .client
        .memory_report(cubecl::MemoryScope::Device)
        .usage();
    for generation in 0..8usize {
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
    let mut simulation = Simulation::from_checkpoint(checkpoint.clone(), support::device(64)?)?;
    let mut replay = Simulation::from_checkpoint(checkpoint, support::device(64)?)?;
    let mut worked = vec![0; 1000];
    for _ in 0..12 {
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
    assert_eq!(worked.iter().sum::<u64>(), 12 * 25);
    simulation.checkpoint()?.restore()?;
    Ok(())
}

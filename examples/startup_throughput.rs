use anyhow::Result;
use bad_econ::{
    domain::{Experiment, Scenario},
    simulation::{Device, Simulation},
};
use clap::Parser;
use std::{path::PathBuf, time::Instant};

#[derive(Parser)]
struct Arguments {
    #[arg(long, value_delimiter = ',', default_value = "1000000")]
    sizes: Vec<u32>,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, default_value_t = 64)]
    workgroup: u32,
    #[arg(long, default_value_t = 3)]
    days: u32,
    #[arg(long)]
    output: PathBuf,
}

fn main() -> Result<()> {
    let args = Arguments::parse();
    let mut reports = Vec::new();
    for households in args.sizes {
        let start = Instant::now();
        let scenario = Scenario::industrial(households, args.seed, Experiment::Baseline)?;
        let scenario_seconds = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let device = Device::headless(args.workgroup)?;
        let device_seconds = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let mut simulation = Simulation::new(scenario, device)?;
        simulation.device.sync()?;
        let simulation_seconds = start.elapsed().as_secs_f64();
        let startup_seconds = scenario_seconds + device_seconds + simulation_seconds;
        eprintln!(
            "{households} households: {startup_seconds:.3}s (scenario {scenario_seconds:.3}s, device {device_seconds:.3}s, simulation {simulation_seconds:.3}s)"
        );
        let mut daily = Vec::new();
        for _ in 0..args.days {
            let start = Instant::now();
            simulation.advance()?;
            simulation.device.sync()?;
            let seconds = start.elapsed().as_secs_f64();
            eprintln!(
                "{households} households day {}: {seconds:.3}s",
                simulation.snapshot.day
            );
            daily.push(serde_json::json!({
                "day": simulation.snapshot.day,
                "seconds": seconds,
                "phases": simulation.timings,
            }));
        }
        reports.push(serde_json::json!({
            "households": households,
            "device": simulation.device.name,
            "scenario_seconds": scenario_seconds,
            "device_seconds": device_seconds,
            "simulation_seconds": simulation_seconds,
            "startup_seconds": startup_seconds,
            "scenario_hash": simulation.scenario.fingerprint()?,
            "final_state_hash": simulation.download()?.digest(),
            "snapshot": simulation.snapshot,
            "daily": daily,
        }));
        std::fs::write(&args.output, serde_json::to_vec_pretty(&reports)?)?;
    }
    Ok(())
}

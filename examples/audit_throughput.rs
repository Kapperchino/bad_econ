use anyhow::{Result, anyhow};
use bad_econ::{
    domain::{Experiment, Scenario},
    simulation::{Device, Simulation},
};
use clap::Parser;
use std::{num::NonZeroU32, path::PathBuf, time::Instant};

#[derive(Parser)]
struct Arguments {
    #[arg(long, value_delimiter = ',', default_value = "10000,100000,1000000")]
    sizes: Vec<u32>,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, default_value_t = 64)]
    workgroup: u32,
    #[arg(long, default_value_t = 2)]
    warmup_days: u32,
    #[arg(long, default_value = "10")]
    repeats: NonZeroU32,
    #[arg(long, default_value = "3")]
    trials: NonZeroU32,
    #[arg(long)]
    output: PathBuf,
}

fn main() -> Result<()> {
    let args = Arguments::parse();
    let mut reports = Vec::new();
    for households in args.sizes {
        let device = Device::headless(args.workgroup)?;
        let scenario = Scenario::industrial(households, args.seed, Experiment::Baseline)?;
        let mut simulation = Simulation::new(scenario, device)?;
        (0..args.warmup_days).try_for_each(|_| simulation.advance())?;
        let expected = simulation.observe()?;
        let before = simulation.download()?.digest();
        simulation.observe()?;
        let mut milliseconds = Vec::new();
        for _ in 0..args.trials.get() {
            let start = Instant::now();
            for _ in 0..args.repeats.get() {
                match simulation.observe()? == expected {
                    true => (),
                    false => Err(anyhow!("audit snapshot changed across observations"))?,
                }
            }
            simulation.device.sync()?;
            milliseconds
                .push(start.elapsed().as_secs_f64() * 1000.0 / f64::from(args.repeats.get()));
        }
        let after = simulation.download()?.digest();
        match before == after {
            true => (),
            false => Err(anyhow!("audit observations changed simulation state"))?,
        }
        eprintln!("{households} households: {milliseconds:?} ms/observation");
        reports.push(serde_json::json!({
            "households": households,
            "firms": expected.firms,
            "device": simulation.device.name,
            "workgroup": args.workgroup,
            "warmup_days": args.warmup_days,
            "repeats_per_trial": args.repeats.get(),
            "milliseconds_per_observation": milliseconds,
            "scenario_hash": simulation.scenario.fingerprint()?,
            "state_hash": after,
            "snapshot": expected,
        }));
        std::fs::write(&args.output, serde_json::to_vec_pretty(&reports)?)?;
    }
    Ok(())
}

use anyhow::{Context, Result, anyhow};
use bad_econ::{
    domain::{Experiment, STATE_HASH_VERSION, Scenario},
    simulation::{Device, Simulation},
};
use clap::Parser;
use std::{
    num::NonZeroU32,
    path::PathBuf,
    time::{Duration, Instant},
};

#[derive(Parser)]
struct Arguments {
    #[arg(long, value_delimiter = ',', default_value = "10000,100000,1000000")]
    sizes: Vec<u32>,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, default_value_t = 64)]
    workgroup: u32,
    #[arg(long, default_value_t = 2)]
    warmup: u32,
    #[arg(long, default_value = "60")]
    days: NonZeroU32,
    #[arg(long)]
    max_day_ms: Option<NonZeroU32>,
    #[arg(long)]
    output: PathBuf,
}

fn main() -> Result<()> {
    let args = Arguments::parse();
    let days = args.days.get();
    let preflight_days = args
        .warmup
        .checked_add(days)
        .context("benchmark day count exhausted")?;
    let mut reports = Vec::new();
    for households in args.sizes {
        let scenario = Scenario::industrial(households, args.seed, Experiment::Baseline)?;
        let device = Device::headless(args.workgroup)?;
        let preflight_start = Instant::now();
        let mut preflight = Simulation::new(scenario.clone(), device.clone())?;
        (0..preflight_days).try_for_each(|_| preflight.advance())?;
        let expected_hash = preflight.download()?.digest();
        drop(preflight);
        device.sync()?;
        device.client.memory_cleanup()?;
        device.sync()?;
        let preflight_seconds = preflight_start.elapsed().as_secs_f64();
        let startup = Instant::now();
        let mut simulation = Simulation::new(scenario, device)?;
        let startup_seconds = startup.elapsed().as_secs_f64();
        (0..args.warmup).try_for_each(|_| simulation.advance())?;
        let starting_day = simulation.snapshot.day;
        let readback_before = simulation.readback_bytes;
        let mut daily = Vec::new();
        let mut slowest = Duration::ZERO;
        let start = Instant::now();
        for _ in 0..days {
            let day_start = Instant::now();
            simulation.advance()?;
            let elapsed = day_start.elapsed();
            slowest = slowest.max(elapsed);
            let memory = simulation.memory_usage();
            eprintln!(
                "{households} households day {}: {:.3}s, reserved={} bytes",
                simulation.snapshot.day,
                elapsed.as_secs_f64(),
                memory.bytes_reserved
            );
            daily.push(serde_json::json!({
                "day": simulation.snapshot.day,
                "seconds": elapsed.as_secs_f64(),
                "phases": simulation.timings,
                "memory": memory,
            }));
        }
        let seconds = start.elapsed().as_secs_f64();
        let readback_bytes = simulation.readback_bytes - readback_before;
        let final_state_hash = simulation.download()?.digest();
        match final_state_hash == expected_hash {
            true => (),
            false => Err(anyhow!("preflight and measured state hashes differ"))?,
        }
        reports.push(serde_json::json!({
            "households": households,
            "device": simulation.device.name,
            "workgroup": args.workgroup,
            "days": days,
            "starting_day": starting_day,
            "warmup_days": args.warmup,
            "preflight_days": preflight_days,
            "preflight_seconds": preflight_seconds,
            "preflight_state_hash": expected_hash,
            "startup_seconds": startup_seconds,
            "seconds": seconds,
            "max_day_milliseconds": slowest.as_secs_f64() * 1000.0,
            "target_day_milliseconds": args.max_day_ms,
            "readback_bytes": readback_bytes,
            "final_state_hash": final_state_hash,
            "state_hash_version": STATE_HASH_VERSION,
            "scenario_hash": simulation.scenario.fingerprint()?,
            "estimated_bytes": simulation.scenario.memory_estimate(),
            "memory_budget_bytes": simulation.device.memory_budget.bytes(),
            "daily": daily,
        }));
        std::fs::write(&args.output, serde_json::to_vec_pretty(&reports)?)?;
        match args.max_day_ms {
            Some(limit) if slowest >= Duration::from_millis(u64::from(limit.get())) => {
                Err(anyhow!(
                    "{households} households: slowest day was {:.3} ms, exceeding the strict {} ms budget; report saved to {}",
                    slowest.as_secs_f64() * 1000.0,
                    limit,
                    args.output.display(),
                ))?
            }
            _ => (),
        }
    }
    Ok(())
}

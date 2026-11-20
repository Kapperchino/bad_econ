use anyhow::{Context, Result, anyhow};
use bad_econ::{
    domain::{Checkpoint, CohortSizes, Economy, Experiment, STATE_HASH_VERSION, Scenario},
    simulation::{Device, Simulation},
};
use clap::Parser;
use mimalloc::MiMalloc;
use std::{
    num::NonZeroU32,
    path::PathBuf,
    time::{Duration, Instant},
};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[derive(Parser)]
struct Arguments {
    #[arg(long, value_delimiter = ',', default_value = "10000,100000,1000000")]
    sizes: Vec<u32>,
    #[arg(long)]
    worker_cohort_size: Option<NonZeroU32>,
    #[arg(long)]
    farmer_cohort_size: Option<NonZeroU32>,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, default_value_t = 64)]
    workgroup: u32,
    #[arg(long, default_value_t = 2)]
    warmup: u32,
    #[arg(long, default_value_t = 0)]
    start_day: u64,
    #[arg(long, default_value = "60")]
    days: NonZeroU32,
    #[arg(long)]
    max_day_ms: Option<NonZeroU32>,
    #[arg(long)]
    max_median_ms: Option<NonZeroU32>,
    #[arg(long)]
    output: PathBuf,
}

fn starting_simulation(scenario: Scenario, device: Device, day: u64) -> Result<Simulation> {
    let mut simulation = Simulation::new(scenario, device)?;
    match day {
        0 => Ok(simulation),
        _ => {
            let mut state = simulation.download()?;
            state.day = day;
            Simulation::from_checkpoint(
                Checkpoint::new(&simulation.scenario, &state)?,
                simulation.device.clone(),
            )
        }
    }
}

fn main() -> Result<()> {
    let args = Arguments::parse();
    let days = args.days.get();
    let preflight_days = args
        .warmup
        .checked_add(days)
        .context("benchmark day count exhausted")?;
    let mut reports = Vec::new();
    for people in args.sizes {
        let defaults = CohortSizes::default();
        let sizes = CohortSizes {
            workers: args.worker_cohort_size.unwrap_or(defaults.workers),
            farmers: args.farmer_cohort_size.unwrap_or(defaults.farmers),
        };
        let scenario = Scenario::for_people(
            people,
            sizes,
            Economy::Industrial,
            args.seed,
            Experiment::Baseline,
        )?;
        let device = Device::headless(args.workgroup)?;
        let preflight_start = Instant::now();
        let mut preflight = starting_simulation(scenario.clone(), device.clone(), args.start_day)?;
        (0..preflight_days).try_for_each(|_| preflight.advance())?;
        let expected_hash = preflight.download()?.digest();
        drop(preflight);
        device.sync()?;
        device.client.memory_cleanup()?;
        device.sync()?;
        let preflight_seconds = preflight_start.elapsed().as_secs_f64();
        let startup = Instant::now();
        let mut simulation = starting_simulation(scenario, device, args.start_day)?;
        simulation.device.sync()?;
        let startup_seconds = startup.elapsed().as_secs_f64();
        (0..args.warmup).try_for_each(|_| simulation.advance())?;
        simulation.device.sync()?;
        let starting_day = simulation.snapshot.day;
        let readback_before = simulation.readback_bytes;
        let mut daily = Vec::new();
        let mut durations = Vec::new();
        let start = Instant::now();
        for _ in 0..days {
            let day_start = Instant::now();
            simulation.advance()?;
            simulation.device.sync()?;
            let elapsed = day_start.elapsed();
            durations.push(elapsed);
            let memory = simulation.memory_usage();
            eprintln!(
                "{people} people day {}: {:.3}s, reserved={} bytes",
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
        durations.sort_unstable();
        let slowest = durations[durations.len() - 1];
        let median = (durations[(durations.len() - 1) / 2] + durations[durations.len() / 2]) / 2;
        let readback_bytes = simulation.readback_bytes - readback_before;
        let final_state_hash = simulation.download()?.digest();
        match final_state_hash == expected_hash {
            true => (),
            false => Err(anyhow!("preflight and measured state hashes differ"))?,
        }
        reports.push(serde_json::json!({
            "people": people,
            "seed": args.seed,
            "population_rows": simulation.scenario.population_rows(),
            "cohort_sizes": sizes,
            "device": simulation.device.name,
            "workgroup": args.workgroup,
            "days": days,
            "starting_day": starting_day,
            "initial_day": args.start_day,
            "warmup_days": args.warmup,
            "preflight_days": preflight_days,
            "preflight_seconds": preflight_seconds,
            "preflight_state_hash": expected_hash,
            "startup_seconds": startup_seconds,
            "seconds": seconds,
            "max_day_milliseconds": slowest.as_secs_f64() * 1000.0,
            "median_day_milliseconds": median.as_secs_f64() * 1000.0,
            "target_day_milliseconds": args.max_day_ms,
            "target_median_milliseconds": args.max_median_ms,
            "readback_bytes": readback_bytes,
            "final_state_hash": final_state_hash,
            "state_hash_version": STATE_HASH_VERSION,
            "scenario_hash": simulation.scenario.fingerprint()?,
            "estimated_bytes": simulation.scenario.memory_estimate(),
            "memory_budget_bytes": simulation.device.memory_budget.bytes(),
            "daily": daily,
        }));
        std::fs::write(&args.output, serde_json::to_vec_pretty(&reports)?)?;
        for (metric, duration, limit) in [
            ("slowest", slowest, args.max_day_ms),
            ("median", median, args.max_median_ms),
        ] {
            if let Some(limit) =
                limit.filter(|limit| duration >= Duration::from_millis(u64::from(limit.get())))
            {
                Err(anyhow!(
                    "{people} people: {metric} day was {:.3} ms, exceeding the strict {} ms budget; report saved to {}",
                    duration.as_secs_f64() * 1000.0,
                    limit,
                    args.output.display(),
                ))?
            }
        }
    }
    Ok(())
}

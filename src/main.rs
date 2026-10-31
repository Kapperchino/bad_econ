use anyhow::{Result, anyhow};
use bad_econ::domain::*;
#[cfg(feature = "compute")]
use bad_econ::runtime::SimulationCommand;
use clap::{Parser, ValueEnum};
use std::path::PathBuf;
#[cfg(feature = "compute")]
use std::time::Instant;

#[derive(Parser)]
#[command(about = "An exact spatial grain–flour–bread economy", version)]
struct Cli {
    #[arg(value_enum, default_value_t = Mode::Run)]
    mode: Mode,
    #[arg(long, default_value_t = 10_000)]
    households: u32,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, value_enum, default_value_t = ExperimentArg::Baseline)]
    experiment: ExperimentArg,
    #[arg(long)]
    scenario: Option<PathBuf>,
    #[arg(long, default_value_t = 10)]
    days: u32,
    #[arg(long, default_value_t = 2)]
    warmup: u32,
    #[arg(long, default_value_t = 64)]
    workgroup: u32,
    #[arg(long, value_enum, default_value_t = Backend::Gpu)]
    backend: Backend,
    #[arg(long)]
    restore: Option<PathBuf>,
    #[arg(long)]
    checkpoint: Option<PathBuf>,
    #[arg(long)]
    output: Option<PathBuf>,
    #[arg(long, value_delimiter = ',', default_value = "10000,100000,1000000")]
    sizes: Vec<u32>,
}
#[derive(Clone, Copy, ValueEnum)]
enum Mode {
    Run,
    Scenario,
    Replay,
    Benchmark,
    View,
    ViewCheck,
}
#[derive(Clone, Copy, ValueEnum)]
enum Backend {
    Cpu,
    Gpu,
}
#[derive(Clone, Copy, ValueEnum)]
enum ExperimentArg {
    Baseline,
    Shortage,
    Transport,
    Concentrated,
}
impl From<ExperimentArg> for Experiment {
    fn from(value: ExperimentArg) -> Self {
        match value {
            ExperimentArg::Baseline => Self::Baseline,
            ExperimentArg::Shortage => Self::Shortage,
            ExperimentArg::Transport => Self::Transport,
            ExperimentArg::Concentrated => Self::Concentrated,
        }
    }
}
fn main() -> Result<()> {
    let args = Cli::parse();
    let scenario = match &args.scenario {
        Some(path) => Scenario::new(serde_json::from_slice(&std::fs::read(path)?)?)?,
        None => Scenario::baseline(args.households, args.seed, args.experiment.into())?,
    };
    match args.mode {
        Mode::Scenario => output(&args.output, &serde_json::to_value(scenario.config())?),
        Mode::Run => run(scenario, &args),
        Mode::Replay => replay(scenario, &args),
        Mode::Benchmark => benchmark(scenario, &args),
        Mode::View | Mode::ViewCheck => view(scenario, args.workgroup, args.mode),
    }
}
#[cfg(feature = "compute")]
fn run(scenario: Scenario, args: &Cli) -> Result<()> {
    use bad_econ::simulation::{Device, Simulation};
    let device = match args.backend {
        Backend::Cpu => Device::cpu(args.workgroup)?,
        Backend::Gpu => Device::headless(args.workgroup)?,
    };
    let mut simulation = Simulation::new(scenario, device)?;
    if let Some(path) = &args.restore {
        simulation.command(SimulationCommand::Restore(serde_json::from_slice(
            &std::fs::read(path)?,
        )?))?;
    }
    let start = Instant::now();
    (0..args.days).try_for_each(|_| -> Result<()> {
        simulation.advance()?;
        eprintln!(
            "day {}: employed={} shortage={} elapsed={:.3}s",
            simulation.snapshot.day,
            simulation.snapshot.employed,
            simulation.snapshot.shortages,
            start.elapsed().as_secs_f64()
        );
        Ok(())
    })?;
    output(
        &args.output,
        &serde_json::json!({"backend":simulation.device.name,"seconds":start.elapsed().as_secs_f64(),"snapshot":simulation.snapshot,"last_day_phases":simulation.timings,"readback_bytes":simulation.readback_bytes,"memory":simulation.memory_usage()}),
    )?;
    if let Some(path) = &args.checkpoint {
        std::fs::write(path, serde_json::to_vec(&simulation.checkpoint()?)?)?;
    }
    Ok(())
}
#[cfg(not(feature = "compute"))]
fn run(_: Scenario, _: &Cli) -> Result<()> {
    Err(anyhow!("build with the compute feature"))
}
#[cfg(feature = "compute")]
fn replay(scenario: Scenario, args: &Cli) -> Result<()> {
    use bad_econ::simulation::{Device, Simulation, Trace};
    let mut cpu = Simulation::new(scenario.clone(), Device::cpu(args.workgroup)?)?;
    let mut gpu = Simulation::new(scenario, Device::headless(args.workgroup)?)?;
    for day in 0..args.days {
        for phase in Phase::ALL {
            let expected = cpu.phase(phase, Trace::Record)?;
            let actual = gpu.phase(phase, Trace::Record)?;
            let state = gpu.download()?;
            match expected.decisions == actual.decisions && state == cpu.download()? {
                true => Ok(()),
                false => Err(anyhow!("replay divergence: day {day}, {phase:?}")),
            }?;
        }
        eprintln!("day {day}: identical decisions and state after all nine phases");
    }
    output(
        &args.output,
        &serde_json::json!({"days":args.days,"workgroup":args.workgroup,"state_hash":cpu.download()?.digest(),"state_hash_version":STATE_HASH_VERSION,"scenario_hash":cpu.scenario.fingerprint()?,"profile":bad_econ::arithmetic::PROFILE,"policy_table_hash":bad_econ::arithmetic::table_hash(),"device":gpu.device.name}),
    )
}
#[cfg(not(feature = "compute"))]
fn replay(_: Scenario, _: &Cli) -> Result<()> {
    Err(anyhow!("build with the compute feature"))
}
#[cfg(feature = "compute")]
fn benchmark(scenario: Scenario, args: &Cli) -> Result<()> {
    use bad_econ::simulation::{Device, Simulation};
    match args.days {
        0 => Err(anyhow!("benchmark days must be positive")),
        _ => Ok(()),
    }?;
    let scenarios = match args.scenario {
        Some(_) => vec![scenario],
        None => args
            .sizes
            .iter()
            .map(|&households| Scenario::baseline(households, args.seed, args.experiment.into()))
            .collect::<Result<Vec<_>>>()?,
    };
    let mut reports = Vec::new();
    for scenario in scenarios {
        let households = scenario.config().households;
        let estimate = scenario.memory_estimate();
        let fingerprint = scenario.fingerprint()?;
        let device = match args.backend {
            Backend::Cpu => Device::cpu(args.workgroup)?,
            Backend::Gpu => Device::headless(args.workgroup)?,
        };
        let mut simulation = Simulation::new(scenario, device)?;
        (0..args.warmup).try_for_each(|_| simulation.advance())?;
        let mut daily = Vec::new();
        let start = Instant::now();
        let readback_before = simulation.readback_bytes;
        for _ in 0..args.days {
            let day_start = Instant::now();
            simulation.advance()?;
            let memory = simulation.memory_usage();
            eprintln!(
                "{households} households day {}: {:.3}s, reserved={} bytes",
                simulation.snapshot.day,
                day_start.elapsed().as_secs_f64(),
                memory.bytes_reserved
            );
            match memory.bytes_reserved <= MEMORY_BUDGET {
                true => Ok(()),
                false => Err(anyhow!("GPU memory exceeded 4 GiB")),
            }?;
            daily.push(serde_json::json!({"day":simulation.snapshot.day,"seconds":day_start.elapsed().as_secs_f64(),"phases":simulation.timings,"memory":memory}));
        }
        let seconds = start.elapsed().as_secs_f64();
        let readback_bytes = simulation.readback_bytes - readback_before;
        let state_hash = simulation.download()?.digest();
        reports.push(serde_json::json!({"households":households,"firms":simulation.scenario.config().firms,"scenario":simulation.scenario.config(),"scenario_hash":fingerprint,"device":simulation.device.name,"workgroup":args.workgroup,"days":args.days,"warmup_days":args.warmup,"seconds":seconds,"days_per_second":f64::from(args.days)/seconds,"readback_bytes":readback_bytes,"final_state_hash":state_hash,"state_hash_version":STATE_HASH_VERSION,"profile":bad_econ::arithmetic::PROFILE,"policy_table_hash":bad_econ::arithmetic::table_hash(),"estimated_bytes":estimate,"memory_budget_bytes":MEMORY_BUDGET,"daily":daily,"snapshot":simulation.snapshot}));
        output(&args.output, &serde_json::to_value(&reports)?)?;
    }
    Ok(())
}
#[cfg(not(feature = "compute"))]
fn benchmark(_: Scenario, _: &Cli) -> Result<()> {
    Err(anyhow!("build with the compute feature"))
}
#[cfg(feature = "viewer")]
fn view(scenario: Scenario, workgroup: u32, mode: Mode) -> Result<()> {
    use bad_econ::viewer::ViewMode;
    bad_econ::viewer::run(
        scenario,
        workgroup,
        match mode {
            Mode::ViewCheck => ViewMode::Verify,
            _ => ViewMode::Interactive,
        },
    )
}
#[cfg(not(feature = "viewer"))]
fn view(_: Scenario, _: u32, _: Mode) -> Result<()> {
    Err(anyhow!("build with --features viewer"))
}
fn output(path: &Option<PathBuf>, value: &serde_json::Value) -> Result<()> {
    let json = serde_json::to_string_pretty(value)?;
    match path {
        Some(path) => {
            if let Some(parent) = path
                .parent()
                .filter(|parent| !parent.as_os_str().is_empty())
            {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(path, json)?;
        }
        None => println!("{json}"),
    };
    Ok(())
}

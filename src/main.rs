use anyhow::{Result, anyhow};
use bad_econ::domain::*;
use clap::{Parser, ValueEnum};
use mimalloc::MiMalloc;
use std::path::PathBuf;
#[cfg(feature = "compute")]
use std::time::Instant;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[derive(Parser)]
#[command(about = "An exact spatial industrial economy", version)]
struct Cli {
    #[arg(value_enum, default_value_t = Mode::Run)]
    mode: Mode,
    #[arg(long, default_value_t = 10_000)]
    households: u32,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, value_enum, default_value_t = ExperimentArg::Baseline)]
    experiment: ExperimentArg,
    #[arg(long, value_enum, default_value_t = EconomyArg::Industrial)]
    economy: EconomyArg,
    #[arg(long)]
    scenario: Option<PathBuf>,
    #[arg(long)]
    analysis: Option<PathBuf>,
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
    #[arg(long)]
    trajectory: Option<PathBuf>,
    #[arg(long, default_value_t = 0)]
    checkpoint_every: u32,
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
    Analyze,
}
#[derive(Clone, Copy, ValueEnum)]
enum Backend {
    Cpu,
    Gpu,
}
#[derive(Clone, Copy, ValueEnum)]
enum EconomyArg {
    Industrial,
    Agrarian,
}
impl EconomyArg {
    fn scenario(self, households: u32, seed: u64, experiment: Experiment) -> Result<Scenario> {
        match self {
            Self::Industrial => Scenario::industrial(households, seed, experiment),
            Self::Agrarian => Scenario::baseline(households, seed, experiment),
        }
    }
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
    match args.mode {
        Mode::Scenario => output(
            &args.output,
            &serde_json::to_value(scenario(&args)?.config())?,
        ),
        Mode::Run => run(scenario(&args)?, &args),
        Mode::Replay => replay(scenario(&args)?, &args),
        Mode::Benchmark => benchmark(scenario(&args)?, &args),
        Mode::View | Mode::ViewCheck => view(scenario(&args)?, args.workgroup, args.mode),
        Mode::Analyze => analyze(&args),
    }
}
fn scenario(args: &Cli) -> Result<Scenario> {
    match &args.scenario {
        Some(path) => Scenario::new(serde_json::from_slice(&std::fs::read(path)?)?),
        None => args
            .economy
            .scenario(args.households, args.seed, args.experiment.into()),
    }
}
#[cfg(feature = "compute")]
fn analyze(args: &Cli) -> Result<()> {
    use anyhow::Context;
    use bad_econ::simulation::{Device, analytics::AnalysisRequest};
    let path = args
        .analysis
        .as_ref()
        .context("analyze requires --analysis PATH")?;
    let request: AnalysisRequest = serde_json::from_slice(&std::fs::read(path)?)?;
    let device = match args.backend {
        Backend::Cpu => Device::cpu(args.workgroup)?,
        Backend::Gpu => Device::headless(args.workgroup)?,
    };
    let report = request.run(&device)?;
    output(&args.output, &serde_json::to_value(report)?)
}
#[cfg(not(feature = "compute"))]
fn analyze(_: &Cli) -> Result<()> {
    Err(anyhow!("build with the compute feature"))
}
#[cfg(feature = "compute")]
fn run(scenario: Scenario, args: &Cli) -> Result<()> {
    use bad_econ::simulation::{Device, Simulation};
    let startup = Instant::now();
    let device = match args.backend {
        Backend::Cpu => Device::cpu(args.workgroup)?,
        Backend::Gpu => Device::headless(args.workgroup)?,
    };
    let mut simulation = match &args.restore {
        Some(path) => {
            Simulation::from_checkpoint(serde_json::from_slice(&std::fs::read(path)?)?, device)?
        }
        None => Simulation::new(scenario, device)?,
    };
    let startup_seconds = startup.elapsed().as_secs_f64();
    use std::io::Write;
    let mut trajectory = args
        .trajectory
        .as_ref()
        .map(std::fs::File::create)
        .transpose()?
        .map(std::io::BufWriter::new);
    let start = Instant::now();
    (0..args.days).try_for_each(|_| -> Result<()> {
        simulation.advance()?;
        eprintln!(
            "day {}: living={} births={} deaths={:?} employed={} shortage={} elapsed={:.3}s",
            simulation.snapshot.day,
            simulation.snapshot.population.living,
            simulation.snapshot.population.births,
            simulation.snapshot.population.deaths,
            simulation.snapshot.employed,
            simulation.snapshot.shortages,
            start.elapsed().as_secs_f64()
        );
        if let Some(trajectory) = &mut trajectory {
            serde_json::to_writer(&mut *trajectory, &simulation.snapshot)?;
            trajectory.write_all(b"\n")?;
            trajectory.flush()?;
        }
        if let (Some(path), interval) = (&args.checkpoint, args.checkpoint_every)
            && interval > 0
            && simulation.snapshot.day % u64::from(interval) == 0
        {
            std::fs::write(path, serde_json::to_vec(&simulation.checkpoint()?)?)?;
        }
        Ok(())
    })?;
    let property = simulation.property_snapshot()?;
    output(
        &args.output,
        &serde_json::json!({"property":property,"backend":simulation.device.name,"startup_seconds":startup_seconds,"seconds":start.elapsed().as_secs_f64(),"snapshot":simulation.snapshot,"opening":simulation.scenario.opening_totals(),"last_day_phases":simulation.timings,"readback_bytes":simulation.readback_bytes,"memory":simulation.memory_usage(),"memory_budget_bytes":simulation.device.memory_budget.bytes(),"memory_budget":simulation.device.memory_budget}),
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
    let scenarios = match (&args.scenario, &args.restore) {
        (Some(_), _) | (_, Some(_)) => vec![scenario],
        (None, None) => args
            .sizes
            .iter()
            .map(|&households| {
                args.economy
                    .scenario(households, args.seed, args.experiment.into())
            })
            .collect::<Result<Vec<_>>>()?,
    };
    let mut reports = Vec::new();
    for scenario in scenarios {
        let startup = Instant::now();
        let device = match args.backend {
            Backend::Cpu => Device::cpu(args.workgroup)?,
            Backend::Gpu => Device::headless(args.workgroup)?,
        };
        let mut simulation = match &args.restore {
            Some(path) => {
                Simulation::from_checkpoint(serde_json::from_slice(&std::fs::read(path)?)?, device)?
            }
            None => Simulation::new(scenario, device)?,
        };
        let startup_seconds = startup.elapsed().as_secs_f64();
        let households = simulation.scenario.config().households;
        let estimate = simulation.scenario.memory_estimate();
        let fingerprint = simulation.scenario.fingerprint()?;
        (0..args.warmup).try_for_each(|_| simulation.advance())?;
        let starting_day = simulation.snapshot.day;
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
            daily.push(serde_json::json!({"day":simulation.snapshot.day,"seconds":day_start.elapsed().as_secs_f64(),"phases":simulation.timings,"memory":memory}));
        }
        let seconds = start.elapsed().as_secs_f64();
        let readback_bytes = simulation.readback_bytes - readback_before;
        let state_hash = simulation.download()?.digest();
        reports.push(serde_json::json!({"households":households,"firms":simulation.scenario.config().firms,"scenario":simulation.scenario.config(),"scenario_hash":fingerprint,"device":simulation.device.name,"workgroup":args.workgroup,"days":args.days,"starting_day":starting_day,"warmup_days":args.warmup,"startup_seconds":startup_seconds,"seconds":seconds,"days_per_second":f64::from(args.days)/seconds,"readback_bytes":readback_bytes,"final_state_hash":state_hash,"state_hash_version":STATE_HASH_VERSION,"profile":bad_econ::arithmetic::PROFILE,"policy_table_hash":bad_econ::arithmetic::table_hash(),"estimated_bytes":estimate,"memory_budget_bytes":simulation.device.memory_budget.bytes(),"memory_budget":simulation.device.memory_budget,"daily":daily,"snapshot":simulation.snapshot}));
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

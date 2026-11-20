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
    #[arg(long, default_value_t = 10_000, conflicts_with = "households")]
    people: u32,
    #[arg(long, help = "Generate a legacy household-based population", conflicts_with_all = ["worker_cohort_size", "farmer_cohort_size"])]
    households: Option<u32>,
    #[arg(long, help = "Maximum workers per starting cohort (default 10)")]
    worker_cohort_size: Option<std::num::NonZeroU32>,
    #[arg(long, help = "Maximum farmers per starting cohort (default 10)")]
    farmer_cohort_size: Option<std::num::NonZeroU32>,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, value_enum, default_value_t = Experiment::Baseline)]
    experiment: Experiment,
    #[arg(long, value_enum, default_value_t = Economy::Industrial)]
    economy: Economy,
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
    #[cfg(feature = "compute")]
    #[arg(
        long,
        help = "Write CubeCL GPU kernel timings and dispatch details to PATH"
    )]
    kernel_profile: Option<PathBuf>,
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
#[cfg(feature = "compute")]
impl Backend {
    fn device(self, workgroup: u32) -> Result<bad_econ::simulation::Device> {
        use bad_econ::simulation::Device;
        match self {
            Self::Cpu => Device::cpu(workgroup),
            Self::Gpu => Device::headless(workgroup),
        }
    }
}
#[cfg(feature = "compute")]
impl Cli {
    fn simulation(&self, scenario: Scenario) -> Result<bad_econ::simulation::Simulation> {
        use bad_econ::simulation::Simulation;
        let device = self.backend.device(self.workgroup)?;
        match &self.restore {
            Some(path) => {
                Simulation::from_checkpoint(serde_json::from_slice(&std::fs::read(path)?)?, device)
            }
            None => Simulation::new(scenario, device),
        }
    }
}
impl Cli {
    fn cohort_sizes(&self) -> CohortSizes {
        let defaults = CohortSizes::default();
        CohortSizes {
            workers: self.worker_cohort_size.unwrap_or(defaults.workers),
            farmers: self.farmer_cohort_size.unwrap_or(defaults.farmers),
        }
    }

    fn generated_scenario(&self, size: u32) -> Result<Scenario> {
        match (self.households, self.economy) {
            (None, economy) => Scenario::for_people(
                size,
                self.cohort_sizes(),
                economy,
                self.seed,
                self.experiment,
            ),
            (Some(_), Economy::Industrial) => {
                Scenario::industrial(size, self.seed, self.experiment)
            }
            (Some(_), Economy::Agrarian) => Scenario::baseline(size, self.seed, self.experiment),
        }
    }
}
fn main() -> Result<()> {
    let args = Cli::parse();
    #[cfg(feature = "compute")]
    if let Some(path) = &args.kernel_profile {
        use cubecl::config::{CubeClRuntimeConfig, RuntimeConfig, profiling::ProfilingLogLevel};
        let mut config = CubeClRuntimeConfig::from_current_dir().override_from_env();
        config.profiling.logger.level = ProfilingLogLevel::Full;
        config.profiling.logger.file = Some(path.clone());
        config.profiling.logger.append = false;
        CubeClRuntimeConfig::set(config);
    }
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
        Some(path) => {
            let mut config: ScenarioConfig = serde_json::from_slice(&std::fs::read(path)?)?;
            if config.people.is_none()
                && (args.worker_cohort_size.is_some() || args.farmer_cohort_size.is_some())
            {
                config.people = Some(config.households);
            }
            config.cohort_sizes.workers = args
                .worker_cohort_size
                .unwrap_or(config.cohort_sizes.workers);
            config.cohort_sizes.farmers = args
                .farmer_cohort_size
                .unwrap_or(config.cohort_sizes.farmers);
            Scenario::new(config)
        }
        None => args.generated_scenario(args.households.unwrap_or(args.people)),
    }
}
#[cfg(feature = "compute")]
fn analyze(args: &Cli) -> Result<()> {
    use anyhow::Context;
    use bad_econ::simulation::analytics::AnalysisRequest;
    let path = args
        .analysis
        .as_ref()
        .context("analyze requires --analysis PATH")?;
    let request: AnalysisRequest = serde_json::from_slice(&std::fs::read(path)?)?;
    let device = args.backend.device(args.workgroup)?;
    let report = request.run(&device)?;
    output(&args.output, &serde_json::to_value(report)?)
}
#[cfg(not(feature = "compute"))]
fn analyze(_: &Cli) -> Result<()> {
    Err(anyhow!("build with the compute feature"))
}
#[cfg(feature = "compute")]
fn run(scenario: Scenario, args: &Cli) -> Result<()> {
    let startup = Instant::now();
    let mut simulation = args.simulation(scenario)?;
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
    match args.days {
        0 => Err(anyhow!("benchmark days must be positive")),
        _ => Ok(()),
    }?;
    let scenarios = match (&args.scenario, &args.restore) {
        (Some(_), _) | (_, Some(_)) => vec![scenario],
        (None, None) => args
            .sizes
            .iter()
            .map(|&size| args.generated_scenario(size))
            .collect::<Result<Vec<_>>>()?,
    };
    let mut reports = Vec::new();
    for scenario in scenarios {
        let startup = Instant::now();
        let mut simulation = args.simulation(scenario)?;
        simulation.device.sync()?;
        let startup_seconds = startup.elapsed().as_secs_f64();
        let population_rows = simulation.scenario.population_rows();
        let people = simulation.snapshot.population.living;
        let estimate = simulation.scenario.memory_estimate();
        let fingerprint = simulation.scenario.fingerprint()?;
        let mut first_day_seconds = None;
        for day in 0..args.warmup {
            let start = Instant::now();
            simulation.advance()?;
            if day == 0 || day + 1 == args.warmup {
                simulation.device.sync()?;
            }
            first_day_seconds.get_or_insert_with(|| start.elapsed().as_secs_f64());
        }
        let starting_day = simulation.snapshot.day;
        let mut daily = Vec::new();
        let start = Instant::now();
        let readback_before = simulation.readback_bytes;
        for _ in 0..args.days {
            let day_start = Instant::now();
            simulation.advance()?;
            simulation.device.sync()?;
            let day_seconds = day_start.elapsed().as_secs_f64();
            first_day_seconds.get_or_insert(day_seconds);
            let memory = simulation.memory_usage();
            eprintln!(
                "{people} people in {population_rows} population accounts day {}: {:.3}s, reserved={} bytes",
                simulation.snapshot.day, day_seconds, memory.bytes_reserved
            );
            daily.push(serde_json::json!({"day":simulation.snapshot.day,"seconds":day_seconds,"phases":simulation.timings,"memory":memory}));
        }
        let seconds = start.elapsed().as_secs_f64();
        let readback_bytes = simulation.readback_bytes - readback_before;
        let state_hash = simulation.download()?.digest();
        reports.push(serde_json::json!({"population_rows":population_rows,"people":people,"firms":simulation.scenario.config().firms,"scenario":simulation.scenario.config(),"scenario_hash":fingerprint,"device":simulation.device.name,"workgroup":args.workgroup,"days":args.days,"starting_day":starting_day,"warmup_days":args.warmup,"startup_seconds":startup_seconds,"first_day_seconds":first_day_seconds,"seconds":seconds,"days_per_second":f64::from(args.days)/seconds,"readback_bytes":readback_bytes,"final_state_hash":state_hash,"state_hash_version":STATE_HASH_VERSION,"profile":bad_econ::arithmetic::PROFILE,"policy_table_hash":bad_econ::arithmetic::table_hash(),"estimated_bytes":estimate,"memory_budget_bytes":simulation.device.memory_budget.bytes(),"memory_budget":simulation.device.memory_budget,"daily":daily,"snapshot":simulation.snapshot}));
        output(&args.output, &reports)?;
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
fn output(path: &Option<PathBuf>, value: &impl serde::Serialize) -> Result<()> {
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

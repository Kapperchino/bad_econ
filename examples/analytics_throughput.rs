use anyhow::{Result, anyhow};
use bad_econ::simulation::{
    Device,
    analytics::{
        Coefficient, Contraction, DeviceOperator, DeviceVector, Diagnostic, SolveBudget,
        SparseMatrix,
    },
};
use clap::Parser;
use std::{num::NonZeroU32, path::PathBuf, time::Instant};

#[derive(Parser)]
struct Arguments {
    #[arg(long, value_delimiter = ',', default_value = "16,128,512")]
    sizes: Vec<usize>,
    #[arg(long, default_value = "128")]
    iterations: NonZeroU32,
    #[arg(long, default_value = "10")]
    repeats: NonZeroU32,
    #[arg(long, default_value = "3")]
    trials: NonZeroU32,
    #[arg(long, default_value_t = 64)]
    workgroup: u32,
    #[arg(long)]
    output: PathBuf,
}

fn main() -> Result<()> {
    let args = Arguments::parse();
    let device = Device::headless(args.workgroup)?;
    let mut reports = Vec::new();
    for rows in args.sizes {
        match rows >= 2 {
            true => (),
            false => Err(anyhow!("benchmark dimensions must be at least two"))?,
        }
        let entries = (0..rows)
            .flat_map(|row| {
                [row, (row + 1) % rows].map(|column| Coefficient {
                    row,
                    column,
                    numerator: 1,
                    denominator: 4,
                })
            })
            .collect();
        let operator = DeviceOperator::new(&device, &SparseMatrix::new(rows, rows, entries)?);
        let rhs = DeviceVector::from_ticks(&device, &vec![1000; rows])?;
        let certificate = Contraction::new(1, 2, vec![1; rows])?;
        let budget = SolveBudget {
            iterations: args.iterations.get(),
            weighted_tolerance: 2,
        };
        let solve = || operator.solve(&device, &rhs, &certificate, budget);
        let reference = solve()?.read(&device)?;
        match &reference {
            Diagnostic::Available(values) => {
                let expected = 2000.0 * (1.0 - 0.5f64.powf(f64::from(budget.iterations)));
                match values.values.iter().all(|value| {
                    (value.lower as f64 - expected).abs() <= 2.0
                        && (value.upper as f64 - expected).abs() <= 2.0
                }) {
                    true => (),
                    false => Err(anyhow!(
                        "solver result differs from the independent reference"
                    ))?,
                }
            }
            _ => Err(anyhow!("benchmark solve was not available"))?,
        }
        let reference = serde_json::to_value(reference)?;
        solve()?;
        let mut milliseconds = Vec::new();
        for _ in 0..args.trials.get() {
            let start = Instant::now();
            for _ in 0..args.repeats.get() {
                solve()?;
            }
            device.sync()?;
            milliseconds
                .push(start.elapsed().as_secs_f64() * 1000.0 / f64::from(args.repeats.get()));
            match serde_json::to_value(solve()?.read(&device)?)? == reference {
                true => (),
                false => Err(anyhow!("benchmark solve changed across trials"))?,
            }
        }
        eprintln!("{rows} rows: {milliseconds:?} ms/solve");
        reports.push(serde_json::json!({
            "rows": rows,
            "device": device.name,
            "workgroup": args.workgroup,
            "iterations": budget.iterations,
            "repeats_per_trial": args.repeats.get(),
            "warmup_solves": 2,
            "milliseconds_per_solve": milliseconds,
            "result": reference,
        }));
        std::fs::write(&args.output, serde_json::to_vec_pretty(&reports)?)?;
    }
    Ok(())
}

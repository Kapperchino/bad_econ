# Spatial economy

A Rust simulation of workers, farms and firms trading 22 goods and services across a map. The economy advances one day at a time on the GPU using Burn/CubeCL. A Bevy viewer lets you follow prices, employment and migration.

Use Metal 3 on macOS or Vulkan with `SHADER_INT64` on Linux and Windows. The Linux viewer also requires ALSA development libraries. [Pinned Rust toolchain](rust-toolchain.toml)

## Run

The default industrial economy starts with 10,000 people and seed 42. These commands run a year headless, open the viewer, and load a custom scenario, respectively.

```sh
cargo run --release -- run --days 365 --output year.json
cargo run --release --features viewer -- view
cargo run --release -- run --scenario scenarios/institutions.json --days 12
```

| Option | Purpose |
| --- | --- |
| `--people N`, `--seed N` | Set the starting population and generation seed. |
| `--economy agrarian` | Run the simpler food economy. |
| `--trajectory PATH` | Write daily observations as JSONL. |
| `--checkpoint PATH`, `--restore PATH` | Save or resume a run. Add `--checkpoint-every N` for periodic saves. |

Workers and farmers start in cohorts of up to 10. Each cohort shares a budget, inventory, location and employer. Firms can hire part of a cohort. In output, `population.living` counts people and `households` counts accounts. For a billion-person run, use `--people 1000000000 --worker-cohort-size 10000 --farmer-cohort-size 10000` on a GPU with sufficient memory.

The viewer starts paused. **Space** runs or pauses, **N** advances one day, and **1/2/3** change speed. Click a firm on the map to inspect its accounts, hover over a chart for daily values, or use **Compare seed** to overlay another run.

## Model

- **Work and migration.** Generated industrial populations start 90% rural. Farm households review paid industrial work monthly and give up their plots when earnings and savings cover food, fuel, rent and moving costs. Landless farm workers have annual contracts and can take temporary work on idle days. Plot holders take farm work by the day, miners sign for 30 days and other regular hires for 14. [Historical farm hiring](https://doi.org/10.1093/ehr/cep178) · [Industrial contracts](https://www.nber.org/system/files/working_papers/w17051/w17051.pdf)
- **Food and fuel.** Households buy bread and heating fuel before discretionary goods. Crop farms sell stored harvests year-round. Farming households keep grain for food and seed and adjust planting to stored grain, recent sales and paid work. They sell surplus wheat to grain distributors, who hire workers to handle it and resell it to mills. Commercial wheat farms can sell directly to mills. The default harvest runs from 28 September to 27 October. Industrial wheat farms use the Norfolk four-course rotation and reserve enough payroll to reap what they plant. [Historical wheat yields](https://www.gutenberg.org/ebooks/16594)
- **Firms and trade.** Producers hire workers and buy local inputs. Manufacturers adjust quotes to stock and costs and can restart with a trial batch. Goods travel by animal transport or by rail when a faster route has capacity and the sender can pay. Daily audits check cash and goods conservation and restore the last audited state on failure.
- **Ownership.** Households can start firms, receive dividends and collect rent through housing firms. If an owner dies without a recorded heir, the estate keeps its assets and the firm retains its share of future dividends.

Scenario files configure populations, production, research, credit, property and railways. [Example scenarios](scenarios/)

## Test and benchmark

```sh
cargo gpu-fast
cargo gpu-full
cargo run --release -- benchmark --sizes 100000,1000000 --days 10
```

`gpu-fast` runs the smoke tests. `gpu-full` runs the regression suite, including viewer tests. Add `--kernel-profile PATH` to a benchmark for per-kernel timings. [Startup and throughput benchmarks](examples/)

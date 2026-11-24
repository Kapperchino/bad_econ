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

The viewer starts paused. **Space** runs or pauses, **N** advances one day, and **1/2/3** change speed. Click a firm on the map to inspect its accounts, hover over a chart for daily values, or use **Compare seed** to overlay another run. Open **Companies** to rank businesses by worker capacity or daily cash profit. Each row includes all of the company’s sites. Select **Inspect site** to see a facility’s accounts and its owner’s stubbornness.

## Model

- **Work and migration.** Generated industrial populations start 90% rural. Farm households review paid industrial work monthly and give up their plots only when paid hours displace cultivation and earnings and savings cover food, fuel, rent and moving costs. Short shifts can fit around farm work. Worker cohorts accrue fatigue from minutes worked per eligible worker. Landless farm workers have annual contracts and can take temporary work on idle days. Plot holders take farm work by the day, miners sign for 30 days and other regular hires for 14. [Historical farm hiring](https://doi.org/10.1093/ehr/cep178) · [Industrial contracts](https://www.nber.org/system/files/working_papers/w17051/w17051.pdf)
- **Food and fuel.** Households buy bread and heating fuel before discretionary goods. Only mills turn grain into flour. Farmers buy flour to bake at home. Landless households buy finished bread. Crop farms sell stored harvests year-round. Farming households keep seed grain and adjust planting to their food budget, stored grain, recent sales and paid work. In the generated industrial economy, farmers live at village grain markets served by mills and bakeries. Grain dealers start with working cash for a year of grain purchases and enough handling capacity to buy during harvest. They hire workers to handle wheat bought from households and resell it to mills. Farm households and commercial wheat farms offer surplus wheat to the nearest distributor with an open order and enough cash to cover the delivered price. Farmers spread sales over the days until the next harvest and can offer more wheat to pay for food. When several farms approach the same distributor, households short of food money get the first chance to sell grain. The seed determines the order within each group. Mills can also buy directly from commercial wheat farms with at least 100 hectares, counting occupied farmland. Each distributor can order and resell multiple goods in response to customer orders, using one cash balance and one staffed handling limit. The default harvest runs from 28 September to 27 October. Industrial wheat farms use the Norfolk four-course rotation and reserve enough payroll to reap what they plant. [Historical wheat yields](https://www.gutenberg.org/ebooks/16594)
- **Firms and trade.** Producers hire workers and buy local inputs. Manufacturers adjust quotes to stock and costs and can restart with a trial batch. Construction firms sell same-day work for buildings, business expansions and landlord upkeep. Unused construction capacity expires at the end of the day. Railways hire and pay crews to build and operate track. Each crew can operate four completed segments, and building the next segment takes another crew. Faster rail routes carry goods when capacity is available and the sender can pay a tariff below the carting cost. Animal freight takes two worker-minutes per unit per travel day. Buyers pay transport companies or independent carters three minutes at the scenario wage, including an allowance for pickup travel and overhead. After commercial carriers pick up loads, households with a worker and time left can earn the remaining fares. Loads wait for a funded pickup and enough worker time. Buyers and sellers at the same location exchange goods without a freight fee. Daily audits check cash and goods conservation and restore the last audited state on failure.
- **Ownership.** Households can start firms, receive dividends and collect rent through housing firms. If an owner dies without a recorded heir, the estate keeps its assets and the firm retains its share of future dividends. Every 30 days, operating companies can bid for another company using cash above their reserves. They look first for the same product and business model. A buyer can enter a new line of business once its open sites together have capacity for 250 workers or it has 25 open sites. Smaller buyers require every open site in the target company to fit one of their existing lines. Each shareholder’s stubbornness is fixed by the seed. Profitable businesses command a higher asking price, and a sale requires every shareholder to agree. The sellers split the payment by their shares, and the buyer takes ownership of all sites.

Expansion preferences are fixed for each owner by the scenario seed. Commercial farms, mills, mines, railways and other industrial producers use the industrial expansion rate (80% by default) at any size. Distributors, bakeries, tailors and furniture workshops use the small-business rate (8% by default) below 20 worker slots. Landlords use the same cutoff measured in homes. At 20 or above, these businesses use the industrial rate.

Scenario files configure populations, production, research, credit, property and railways. [Example scenarios](scenarios/)

## Test and benchmark

```sh
cargo gpu-fast
cargo gpu-full
cargo run --release -- benchmark --sizes 100000,1000000 --days 10
```

`gpu-fast` runs the smoke tests. `gpu-full` runs the regression suite, including viewer tests. Add `--kernel-profile PATH` to a benchmark for per-kernel timings. [Startup and throughput benchmarks](examples/)

To time day advancement in the viewer worker with one million people, run the release benchmark below. It measures 60 days after two warm-up days and includes publishing each daily snapshot. Rendering and playback delays are outside the timer.

```sh
cargo test --locked --release --features viewer --lib benchmark_million_people_viewer_days_on_gpu -- --ignored --nocapture
```

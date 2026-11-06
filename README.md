# Spatial economy

A Rust simulation of households, farms, factories, trade, and research across 16 goods. Runs on the GPU with Burn/CubeCL and includes a Bevy viewer.

Requires the [pinned Rust toolchain](rust-toolchain.toml) and a GPU supporting `SHADER_INT64`. The Linux viewer also needs ALSA development libraries.

## Run

```sh
cargo run --release --features viewer -- view --households 1000000 --seed 42
cargo run --release -- run --households 1000000 --days 30
cargo run --release -- run --scenario scenarios/institutions.json --days 12
```

GPU and the industrial economy are the defaults. Use `--economy agrarian` for the simpler food economy, or load a [scenario](scenarios/) with its own population and settings.

- `--seed N`: generation seed; defaults to 42.
- `--output PATH`: save results as JSON.
- `--trajectory PATH`: record daily accounts as JSONL.
- `--checkpoint PATH` / `--restore PATH`: save or resume a simulation. Add `--checkpoint-every N` for periodic saves.

See `cargo run -- --help` for all options.

The viewer starts paused. **Space** runs/pauses, **N** steps one day, **R** resets, **1/2/3** change speed, **G** changes the display, and **C** compares the next seed.

## Model

- Subsistence farmers grow their own food. Wage workers start employed by capitalist-owned farms, factories, or distributors and buy food.
- Firms produce goods using labor and inputs. Natural resources constrain production sites; deliveries take at least one day.
- Births, deaths, migration, business entry, and closures change the economy over time.
- Generated industrial economies charge 5 cash ticks daily per living landless household and per firm until closure. Subsistence households with land are exempt. Firms pay after wages. Households pay after buying bread, before discretionary purchases, and keep their cash when their food buffer (including deliveries) is short. Unpaid balances carry forward. A separate landlord account holds payments and is included in cash conservation; it does not yet spend or redistribute them.
- Scenario files can set `"rent": { "household_daily": 5, "firm_daily": 5 }`; omitting it disables rent. Existing scenario files and checkpoints retain their explicit rates. The defaults were checked over 365 GPU days with one million households; long-term stability still depends on cash circulation. The viewer shows total collected rent and arrears. Checkpoints use format 9 and require regeneration from older formats.
- Transactions use exact integer accounting. Daily audits check cash and goods conservation; faults restore the last audited state.

## Test and benchmark

```sh
cargo gpu-fast
cargo gpu-full
cargo run --release -- benchmark --sizes 100000,1000000 --seed 42 --warmup 0 --days 1000 --output /tmp/benchmark.json
```

Both test aliases use the GPU. Benchmarks report daily phase timings, memory use, and final state hashes. Add integration modules to `tests/integration.rs`; automatic test discovery is disabled.

Code: [domain](src/domain.rs), [simulation](src/simulation/), [viewer](src/viewer.rs), [tests](tests/).

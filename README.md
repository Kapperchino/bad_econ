# Spatial economy

A Rust simulation of households, farms, factories, trade, and research across 17 goods and services. Runs on the GPU with Burn/CubeCL and includes a Bevy viewer.

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
- Entrepreneurial interest is a persistent seeded trait: 8% of households are interested by default, including small businesses. Interested households review local business opportunities monthly, then need savings, construction materials, and operating funds to open a firm. They estimate margins from nearby prices and available inputs; they do not know other firms' order books or prospective customers' bank balances, and sales are not guaranteed. Set `"entrepreneurship": { "interested_per_mille": 80 }` to change this modeling assumption (0 disables entry, 1000 makes every household interested).
- Generated industrial households start with unequal liquid savings: about 60% have 1.75–7 days of starting food and rent expenses, 30% have 7–28 days, and 10% have 28–112 days. This is a modeling distribution, separate from initial food inventories and ownership of land or firms. `"household_liquidity": "PaycheckToPaycheck"` uses `household_cash` as the one-week reference; omitting the setting preserves uniform balances, and explicit population cash balances are used exactly.
- Generated industrial producers start with two days of finished goods so replenishment provides paid work early. `"producer_buffer_days": 2` controls these stocks separately from household food and production inputs; omitting it retains the scenario's existing `buffer_days` behavior.
- Generated industrial economies charge 5 cash ticks daily per living landless household and per firm until closure. Subsistence households with land are exempt. Firms pay after wages. Households pay after buying bread, before discretionary purchases, and keep their cash when their food buffer (including deliveries) is short. Unpaid balances carry forward. A separate landlord account holds payments and is included in cash conservation; it does not yet spend or redistribute them.
- Scenario files can set `"rent": { "household_daily": 5, "firm_daily": 5 }`; omitting it disables rent. Existing scenario files and checkpoints retain their explicit rates. The defaults were checked over 365 GPU days with one million households; long-term stability still depends on cash circulation. The viewer shows total collected rent and arrears. Checkpoints use format 10 and require regeneration from older formats.
- The optional `property` configuration partitions the map into nonoverlapping rectangular regions with fixed space and a landowner household. Housing, commercial farms, factories, and reserved construction sites share that space. Existing actors start with one unit of building space; `buildings` can override each site's footprint. Natural-good firms use farms; manufactured-good firms, including builders, use factories. Buildings remain on their original land after household migration or business closure.
- Landowners commission `orders` for their own household or firms using actor identities (`landowner`, `site`), `space`, `work`, and `development` (`NewBuilding` or `Extension`). New firms cannot produce until completion; extensions add one production slot per unit of new space. Projects move through Requested → Building → Completed, reserving space in actor order. Unavailable space leaves requests pending; absent materials, workers, funds, or deliveries stall construction. Abandoned projects release reservations at the next opening. With property enabled, entrepreneurial entry and automatic expansion require the regional landowner and purchase 20 construction work units per added space.
- `Construction` firms hire labor and consume wood and tools using ordinary production recipes. `MechanizedConstruction` is available through mechanization research. Projects buy their output through the existing paid market and delivery system, consuming it as construction work. Subsistence households keep their plots only when their entire grant fits in the space left after buildings and reservations; otherwise they lose their plots and unharvested crop work and can seek wage employment. Allocation is deterministic, regional, and preserves stored food. Migration relinquishes subsistence plots. Run `cargo run --release -- run --scenario scenarios/property.json --days 14` for a two-region example with housing, farm, and factory projects. Run output includes regional property totals; actor inspection shows land and project status. Omitting `property` retains the existing owner-built construction behavior.
- Transactions use exact integer accounting. Daily audits check cash and goods conservation; faults restore the last audited state.

## Test and benchmark

```sh
cargo gpu-fast
cargo gpu-full
cargo run --release -- benchmark --sizes 100000,1000000 --seed 42 --warmup 0 --days 1000 --output /tmp/benchmark.json
```

Both test aliases use the GPU. Benchmarks report daily phase timings, memory use, and final state hashes. Add integration modules to `tests/integration.rs`; automatic test discovery is disabled.

Code: [domain](src/domain.rs), [simulation](src/simulation/), [viewer](src/viewer.rs), [tests](tests/).

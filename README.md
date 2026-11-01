# Spatial economy

A grain–flour–bread economy for up to one million households and 10,000 firms. Burn/CubeCL runs the same Rust kernels on WGPU and CPU; Bevy provides an optional viewer. The Rust toolchain and dependencies are pinned.

## Run

```sh
cargo run -- run --households 1000000 --days 30
cargo run --features viewer -- view --households 1000000
cargo run -- run --scenario scenarios/institutions.json --days 12
cargo run -- analyze --analysis scenarios/analysis.json --output /tmp/accounts.json
```

GPU is the default; use `--backend cpu` for CPU execution. `--seed` controls generated populations, `--output PATH` saves JSON, and `--checkpoint PATH` / `--restore PATH` save and resume audited state. See `cargo run -- --help` for all options.

The viewer starts paused: Space toggles running, N advances one day, R resets, G changes the display, and C compares the next seed. Simulation runs on a worker; the viewer renders aggregate snapshots rather than household entities.

## Model

Each day follows opening → observation → proposal → commitment → work/payment → provision/purchase → physical close → accounts/decisions → audit.

- Households supply labor and consume bread. Farms, mills, and bakeries produce discrete packets; each firm has a household owner.
- Labor and goods markets run four matching rounds, considering at most eight eligible nearby candidates per actor. Matching follows deterministic cell and identity order.
- Deliveries belong to the buyer and take at least one day. Pending goods cannot be consumed or used in production; new output reaches the next day's market.
- Resources use exact unsigned 64-bit ticks. Transactions check all debits and authorization before posting. Daily audits enforce cash and goods identities; faults restore the last audited state.
- Optional institutions include payroll taxes, wage bargaining, banks, support, research, equipment, natural stocks, and strikes. Populations and actor identities are fixed.

[Scenario examples](scenarios/) cover shortages, transport, concentrated markets, institutions, and custom populations. Export a starting configuration with:

```sh
cargo run -- scenario --households 1000000 --output /tmp/scenario.json
```

A loaded scenario supplies its own seed and population settings.

## Code

- [Domain](src/domain.rs): scenario validation, populations, state, checkpoints, and observations.
- [Simulation](src/simulation/): GPU/CPU kernels and phase orchestration. Markets group offers by counterparty and identity; fixed spatial membership is cached while eligibility refreshes each round.
- [Ledger](src/ledger.rs) and [arithmetic](src/arithmetic.rs): canonical transactions and exact arithmetic contracts.
- [Analytics](src/simulation/analytics.rs): exact interval accounts from explicitly supplied analytical inputs.
- [Viewer](src/viewer.rs): Bevy controls, inspection, and visualization.

## Verify and benchmark

```sh
cargo test --features viewer -- --include-ignored
cargo clippy --all-targets --features viewer -- -D warnings
cargo run -- replay --households 100 --days 8
cargo run -- benchmark --sizes 1000000 --warmup 2 --days 60 --output /tmp/benchmark.json
```

Tests require a GPU and compare CPU/GPU phase states, the original transaction scheduler, and independent integer oracles. Benchmarks record daily timings, event counts, memory, readback volume, and state hashes. The final state download is outside the timed interval; simulation buffers have a 4 GiB budget.

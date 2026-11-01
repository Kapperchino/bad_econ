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

## Lean correspondence

Compared against `../marx-lean-proof` at `1ad0e02`. The executable contracts correspond to the following proof modules; the Rust tests check finite arithmetic and execution, rather than extending the Lean proofs to the runtime.

| Lean contracts | Implementation and regression coverage |
| --- | --- |
| `Basic`, `Productivity`, `Spectral`, `Conversion`, `Theorems`; simulator `Analytics`, `Solver` | Certified goods and occupational equations, external and normalized weights, research plans, labeled comparators, residual bounds, and context caches in [analytics](src/simulation/analytics.rs), checked in [analytics tests](tests/analytics.rs). Positive-vector certificates establish productivity without computing floating-point eigenvalues. Empty systems have empty certificates and exact empty solutions. |
| `Model.valuation`, `ValuationRecord` | [Valuation records](src/simulation/analytics/valuation.rs) attach context versions, raw occupational hours, and supporting capabilities to selected activities. [Valuation tests](tests/valuation.rs) cover orientation, declared graphs, independent diagnostics, and request serialization on GPU. |
| `Capabilities`, `Support`, `BooleanBoundary`; simulator `Execution.Readiness` | [Capability closure](src/simulation/research/capabilities.rs), [boundary elimination](src/simulation/research/capabilities/boundary.rs), and separate discovery/mastery/permission/adoption gates; [research tests](tests/research.rs). |
| `Accounting`, `Blocks`, `Units` | Signed inventory reconciliation and horizons, research input accounts, block inverses, boundary demand, lifetime conventions, unit conversion and rebasing; [retained account tests](tests/retained.rs). |
| `Sensitivity`, `Truncation` | Signed derivatives, finite changes, coefficient responses, residual errors and weighted Neumann bounds in [sensitivity](src/simulation/analytics/sensitivity.rs); [sensitivity tests](tests/sensitivity.rs). |
| Simulator `Ledger`, `Packets`, `Execution` | Grouped gross debits, authorization, atomic posting, packets, fractional claims, keyed 128-bit research draws, dependency scheduling and audited checkpoints; [contracts](tests/contracts.rs), [execution](tests/execution.rs), and [GPU replay](tests/gpu.rs). Deserialized bundles and probabilities pass through their constructors. |
| Simulator `Policies`, `Finance` | Household health/time, quotes, forecasts, production starts, equipment, stocks, research hazards, payroll, bargaining, lending, dated funding, support and bank settlement; [household](tests/households.rs), [policy](tests/policies.rs), [payroll](tests/payroll.rs), [bargaining](tests/bargaining.rs), [finance](tests/finance.rs), and [institution](tests/institutions.rs) tests. |

The optional `valuation` object in [the analysis example](scenarios/analysis.json) selects zero-based goods indices. Each record reports hours per unit of output using the request's scale; historical research spending and scalar labor weights do not change those hours. Capability labels are 32-bit masks. `"InputSupport"` follows strictly positive goods inputs; `{"Declared": [{"prerequisite": 0, "dependent": 1}]}` uses the supplied dependency graph. Omitting capability inputs reports `MissingData`; an explicitly empty label set reports `Available(0)`. Boolean support remains available when the numerical account is inadmissible or unresolved.

The runtime uses bounded integers, directed rational intervals, and versioned nonlinear tables. It does not execute arbitrary real-valued inputs, unbounded catalogues, or every institutional scenario from the inherited PDFs. The fixed spatial population and grain–flour–bread simulation remain the concrete economy; the analytical APIs accept separate finite catalogues. Lean's conditional scheduling theorem, empirical calibration, and arbitrary scenario completeness are not runtime certification claims.

## Verify and benchmark

```sh
cargo test --features viewer
cargo clippy --all-targets --features viewer -- -D warnings
cargo run -- benchmark --sizes 1000000 --warmup 2 --days 60 --output /tmp/benchmark.json
```

Compute tests run exclusively on GPU and compare phase states across workgroups, the original transaction scheduler, and independent integer oracles. The test helper always selects the GPU. Benchmarks record daily timings, event counts, memory, readback volume, and state hashes. The final state download is outside the timed interval; simulation buffers have a 4 GiB budget.

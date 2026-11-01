# Spatial economy

A spatial industrial economy with 16 goods, supply chains, and research-driven production methods, for up to one million households and 10,000 generated firms. Burn/CubeCL runs the same Rust kernels on WGPU and CPU; Bevy provides an optional viewer. The Rust toolchain and dependencies are pinned.

## Run

```sh
cargo run -- run --households 1000000 --days 30
cargo run --features viewer -- view --households 1000000
cargo run -- run --economy agrarian --households 10000 --days 30
cargo run -- run --scenario scenarios/institutions.json --days 12
cargo run -- analyze --analysis scenarios/analysis.json --output /tmp/accounts.json
```

GPU and the industrial economy are the defaults. `--economy agrarian` selects the original food economy; loaded scenarios retain their own economy setting. `--seed` controls generated populations, `--output PATH` saves JSON, and `--checkpoint PATH` / `--restore PATH` save and resume audited state. See `cargo run -- --help` for all options.

State stores actor metadata, active recipes, per-good deliveries and consumption, bank and loan balances, and research records in field rows on the host and GPU. Checkpoints use format 4 with the `ResearchFixedV3` research layout; checkpoints from earlier formats must be regenerated. Existing three-good scenario files still load, with additional inventories and research inputs initialized to zero.

The viewer starts paused: Space toggles running, N advances one day, R resets, G changes the display, and C compares the next seed. Simulation runs on a worker; the viewer renders aggregate snapshots rather than household entities.

## Model

Each day follows opening → observation → proposal → commitment → work/payment → provision/purchase → physical close → accounts/decisions → audit.

- Households supply labor and consume bread daily. Industrial households also buy linen and cloth every 20 days, coats every 40 days, and furniture and glass every 60 days, on staggered schedules. Firms produce discrete packets; each firm has a household owner.
- Labor and goods markets run four matching rounds, considering at most eight eligible nearby candidates per actor. Matching follows deterministic cell and identity order.
- Deliveries belong to the buyer and take at least one day. Pending goods cannot be consumed or used in production; new output reaches the next day's market.
- Resources use exact unsigned 64-bit ticks. Transactions check all debits and authorization before posting. Daily audits enforce cash and goods identities; faults restore the last audited state.
- Optional institutions include payroll taxes, wage bargaining, banks, support, research, equipment, natural stocks, and strikes. Populations and actor identities are fixed.

[Scenario examples](scenarios/) cover shortages, transport, concentrated markets, institutions, and custom populations. Export a starting configuration with:

```sh
cargo run -- scenario --households 1000000 --output /tmp/scenario.json
```

A loaded scenario supplies its own seed and population settings.

## Goods and production

Inventory, research-input, consumption, and market arrays follow this order: grain, flour, bread, iron, coal, steel, flax, linen, cloth, coats, wood, tools, furniture, cotton, dye, glass. `FirmSeed.good` uses the corresponding Rust enum name, such as `"Steel"` or `"Coats"`. A firm's `Technique.input_ticks` is consumed for **each** ingredient per packet; both ingredients must be available before labor can be committed.

| Industry | Basic inputs | Research method | Required technology |
| --- | --- | --- | --- |
| Grain, flax, cotton, dye | None | MechanicalHarvesting; adds tools | Mechanization |
| Flour | Grain | SteamMilling; adds coal | SteamPower |
| Bread | Flour | MechanicalBaking | Mechanization |
| Iron, coal | None | SteamMining; adds tools and coal | SteamPower |
| Steel | Iron + coal | BessemerProcess | Steelmaking |
| Linen | Flax | PowerLooms | PowerLooms |
| Cloth | Cotton + dye | PowerLooms | PowerLooms |
| Coats | Cloth + linen | SewingMachines | SewingMachines |
| Wood | None | Sawmills; adds tools | Mechanization |
| Tools | Iron + wood | MachineTools; replaces iron with steel | Steelmaking |
| Furniture | Wood + tools | FurnitureWorkshops | Mechanization |
| Glass | Coal | Glassworks | SteamPower |

These initial upgrades double packet output (capped at 100), reduce labor to two-thirds (rounded down, at least one minute), and preserve the base input quantity, with a minimum of one tick for newly introduced inputs. An upgrade can therefore stall a firm until its new ingredients arrive. Existing shipments keep their original good when a firm changes methods. The viewer's firm inspection shows its current method, output, and labor requirement.

Industrial presets create firm-owned research labs. Mechanization unlocks SteamPower and PowerLooms; SteamPower unlocks Steelmaking; PowerLooms and Steelmaking together unlock SewingMachines. Their capability bits are 1, 2, 4, 8, and 16 respectively. Projects spend available household research time, use deterministic probabilistic discovery, and stop spending resources once their outcomes are discovered. Preset projects require labor; custom projects can also consume goods through `inputs`.

To enable an upgrade in a custom scenario, add its enum name to the owning research lab's `methods` array and supply projects with the required outcome bits. Discovery, mastery, permission, and adoption must all be present. `"MasterAndAdoptPermitted"` automatically masters and adopts permitted discoveries; `"Manual"` uses the explicitly supplied access masks. Methods are considered in array order and the last eligible method becomes active at the next opening. The optional `production_requirements` mask still gates the firm's entire production separately. The compact [industrial scenario](scenarios/industrial.json) includes all goods and research labs for steel and coats.

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

The runtime uses bounded integers, directed rational intervals, and versioned nonlinear tables. It does not execute arbitrary real-valued inputs, unbounded catalogues, or every institutional scenario from the inherited PDFs. The fixed spatial population and 16-good catalogue define the concrete economy; the analytical APIs accept separate finite catalogues. Lean's conditional scheduling theorem, empirical calibration, and arbitrary scenario completeness are not runtime certification claims.

## Verify and benchmark

```sh
cargo test --features viewer
cargo clippy --all-targets --features viewer -- -D warnings
cargo run -- benchmark --sizes 1000000 --warmup 2 --days 60 --output /tmp/benchmark.json
```

Compute tests run exclusively on GPU and compare phase states across workgroups, the original transaction scheduler, and independent integer oracles. The test helper always selects the GPU. Benchmarks record daily timings, event counts, memory, readback volume, and state hashes. The final state download is outside the timed interval; simulation buffers have a 4 GiB budget.

# Spatial economy

A spatial industrial economy with people, household enterprises, factories, grocery distributors, 16 goods, supply chains, and research-driven production methods. Generated scenarios support up to one million household accounts; births, deaths, migration, and business entry change the runtime population. Burn/CubeCL runs the Rust kernels on WGPU and CPU, with CubeK providing matrix multiplication, reductions, and random sampling; Bevy provides an optional viewer. The Rust toolchain and dependencies are pinned.

## Run

```sh
cargo run -- run --households 1000000 --days 30
cargo run --features viewer -- view --households 1000000
cargo run -- run --economy agrarian --households 10000 --days 30
cargo run -- run --scenario scenarios/institutions.json --days 12
cargo run -- analyze --analysis scenarios/analysis.json --output /tmp/accounts.json
```

GPU and the industrial economy are the defaults. The GPU device must support `SHADER_INT64` for exact integer reductions. `--economy agrarian` selects the original food economy; loaded scenarios retain their own economy setting. `--seed` controls generated populations and resource geography, `--output PATH` saves JSON, and `--checkpoint PATH` / `--restore PATH` save and resume audited state. See `cargo run -- --help` for all options.

On Windows, Smart App Control can block locally compiled build scripts with error 4551. The development profile leaves `icu_normalizer_data` and `icu_segmenter_data` unoptimized, which resolved the blocked ICU build scripts on the audited Windows setup. Other dependencies retain their optimization settings. This workaround leaves Windows security settings unchanged; future unsigned outputs can still be blocked. See [Microsoft's Smart App Control guidance](https://support.microsoft.com/en-us/windows/security/threat-malware-protection/smart-app-control-frequently-asked-questions).

State stores actor metadata, active recipes, per-good deliveries and consumption, bank and loan balances, and research records in field rows on the host and GPU. Checkpoints use format 7 with the `ResearchCubekV4` research layout and `bad-econ/state/CubekV2/v8` state hashes; earlier checkpoint formats are rejected. Scenario format 2 explicitly migrates version 1 and discards its obsolete health settings. Existing three-good scenario files still load, with additional inventories and research inputs initialized to zero, and omitted resource maps default to generated geography.

The viewer starts paused: Space toggles running, N advances one day, R resets, G changes the display, and C compares the next seed. Simulation runs on a worker; the viewer renders aggregate snapshots rather than household entities. Selected inspections refresh each day; firm inspection includes lifecycle state, construction, equipment, and safety.

A status banner confirms completed controls and reports failures, with short sound cues and a subtle 240 ms color pulse confined to the banner. Automatic days stay quiet. M toggles sound; V reduces flashes to a static tint. Feedback preferences survive reset. The generated WAV cues are embedded in the executable, so launching from another directory needs no asset setup. The Linux viewer additionally needs ALSA development libraries for audio.

Presentation geometry and statistics refresh when the published display or selected chart changes. Worker publications wake the window immediately; between changes, it updates at most every 250 ms when focused or 500 ms when unfocused. Banner pulses temporarily use 60 Hz updates. The simulation worker and audit cadence are independent of rendering.

## Model

Each day follows opening → observation → proposal → commitment → work/payment → provision/purchase → physical close → accounts/decisions → audit.

- Households supply labor and consume bread daily. Industrial households also buy linen and cloth every 20 days, coats every 40 days, and furniture and glass every 60 days, on staggered schedules. Firms produce discrete packets; each firm has a household owner.
- Labor and goods markets run four matching rounds, considering at most eight eligible nearby candidates per actor. Discovery rotates deterministically; settlement uses seeded offsets across the whole runtime population for each day, phase, market, and round.
- Deliveries belong to the buyer and take at least one day. Pending goods cannot be consumed or used in production; new output reaches the next day's market.
- Resources use exact unsigned 64-bit ticks. Transactions check all debits and authorization before posting. Daily audits enforce cash and goods identities; faults restore the last audited state.
- Optional institutions include payroll taxes, wage bargaining, banks, support, research, equipment, natural stocks, and strikes. People age, are born, die, and form households; identities remain stable and are never reused.

[Scenario examples](scenarios/) cover shortages, transport, concentrated markets, institutions, and custom populations. Export a starting configuration with:

```sh
cargo run -- scenario --households 1000000 --output /tmp/scenario.json
```

A loaded scenario supplies its own seed and population settings.

The [industrial model and mortality profile](docs/industrial-economy.md) describe household production, population transitions, firm formation, ownership, and grocery operation, including assumptions and limits. The [validation report](docs/industrial-validation.md) records GPU checks, long-run trajectories, and remaining model limitations. `--trajectory /tmp/daily.jsonl` records daily population and food accounts; `--checkpoint-every 30` writes periodic checkpoints when `--checkpoint` is supplied.

## Goods and production

Inventory, research-input, consumption, and market arrays follow this order: grain, flour, bread, iron, coal, steel, flax, linen, cloth, coats, wood, tools, furniture, cotton, dye, glass. `FirmSeed.good` uses the corresponding Rust enum name, such as `"Steel"` or `"Coats"`. A firm's `Technique.input_ticks` is consumed for **each** ingredient per packet; both ingredients must be available before labor can be committed.

Grain, iron, coal, flax, wood, cotton, and dye are **natural** goods. Their producers require the matching local resource as well as labor and any recipe inputs. All other goods are **manufactured** and can be produced anywhere with sufficient labor and inputs. Tools and fuel used by extraction upgrades do not change a good's category.

The optional `resources` setting defaults to `"Generated"`. Generation creates four seeded square patches per natural good, each with side length `max(1, map_size / 4)`, using random draws independent of population generation. Generated natural producers are placed within matching patches; household and manufacturer placement follows the existing population rules. The concentrated preset puts one patch of each natural good around the shared center.

An explicit resource map replaces all generated patches. For example:

```json
"resources": {
  "Explicit": {
    "regions": [
      { "good": "Grain", "lower": { "x": 0, "y": 0 }, "upper": { "x": 250, "y": 250 } },
      { "good": "Iron", "lower": { "x": 100, "y": 100 }, "upper": { "x": 500, "y": 500 } }
    ]
  }
}
```

Regions must name natural goods and be nonempty rectangles within the map. Lower bounds are inclusive, upper bounds exclusive; overlaps make all listed resources available. `{"Explicit":{"regions":[]}}` describes a map with no natural resources. Automatic population generation rejects a missing region for any natural good it needs to produce. Explicit populations keep their supplied positions: a natural producer at an unsuitable site remains valid and may sell existing inventory, but plans no production, hires no production labor, and creates no output. Its nominal capacity is retained. Resource capacity is cached separately in GPU actor metadata at initialization and restoration; research, equipment, and optional natural-stock constraints can further limit production. Geography itself does not deplete or renew, and firms do not relocate.

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

These initial upgrades double packet output (capped at 100), reduce labor to two-thirds (rounded down, at least one minute), and preserve the base input quantity, with a minimum of one tick for newly introduced inputs. Industrial firms retain basic methods and can revert when upgrade equipment or inputs are unavailable. Existing shipments keep their original good when a firm changes methods. The viewer's firm inspection shows its current method, output, and labor requirement.

Industrial presets create firm-owned research labs. Mechanization unlocks SteamPower and PowerLooms; SteamPower unlocks Steelmaking; PowerLooms and Steelmaking together unlock SewingMachines. Their capability bits are 1, 2, 4, 8, and 16 respectively. Projects spend available household research time, use deterministic probabilistic discovery, and stop spending resources once their outcomes are discovered. Preset projects require labor; custom projects can also consume goods through `inputs`.

To enable an upgrade in a custom scenario, add its enum name to the owning research lab's `methods` array and supply projects with the required outcome bits. Discovery, mastery, permission, and adoption must all be present. `"MasterAndAdoptPermitted"` automatically masters and adopts permitted discoveries; `"Manual"` uses the explicitly supplied access masks. Methods are considered in array order and the last eligible method becomes active at the next opening. The optional `production_requirements` mask still gates the firm's entire production separately. The compact [industrial scenario](scenarios/industrial.json) includes all goods and research labs for steel and coats.

## Code

- [Domain](src/domain.rs): scenario validation, populations, state, checkpoints, and observations.
- [Simulation](src/simulation/): GPU/CPU kernels and phase orchestration. Markets keep phase counters on the GPU, stably sort seeded offers by counterparty, and search candidates in cooperative tiles. Capacity changes preserve spatial membership; migration and entry rebuild it. Tiled integer audits retain overflow checks and audited rollback copies.
- [Ledger](src/ledger.rs) and [arithmetic](src/arithmetic.rs): canonical transactions and exact arithmetic contracts.
- [Analytics](src/simulation/analytics.rs): approximate f32 accounts from explicitly supplied analytical inputs, using CubeK batched matrix multiplication and reductions.
- [Viewer](src/viewer.rs): Bevy controls, inspection, and visualization.

Kernel interfaces use CubeCL `CubeType` and `CubeLaunch` records. Stages, markets, discovery strategies, economy modes, and optional policy modes use comptime enums; days, populations, rates, balances, and per-actor decisions remain runtime data. Tax brackets use a typed `Sequence` with an unrolled structural loop. Research and demography pass named launch parameters and table buffers. Burn tensor arguments and CubeCL dispatch sizing handle buffer metadata and device limits; CubeK supplies matrix multiplication, reductions, and random sampling.

## Lean correspondence

Compared against `../marx-lean-proof` at `1ad0e02`. The executable contracts correspond to the following proof modules; the Rust tests check finite arithmetic and execution, rather than extending the Lean proofs to the runtime.

| Lean contracts | Implementation and regression coverage |
| --- | --- |
| `Basic`, `Productivity`, `Spectral`, `Conversion`, `Theorems`; simulator `Analytics`, `Solver` | Approximate goods and occupational equations, external and normalized weights, research plans, labeled comparators, residual estimates, and context caches in [analytics](src/simulation/analytics.rs), checked in [analytics tests](tests/analytics.rs). Positive-vector contraction witnesses are checked numerically without computing eigenvalues. Empty systems have vacuous witnesses and empty solutions. These floating-point checks do not certify exact real-valued bounds. |
| `Model.valuation`, `ValuationRecord` | [Valuation records](src/simulation/analytics/valuation.rs) attach context versions, raw occupational hours, and supporting capabilities to selected activities. [Valuation tests](tests/valuation.rs) cover orientation, declared graphs, independent diagnostics, and request serialization on GPU. |
| `Capabilities`, `Support`, `BooleanBoundary`; simulator `Execution.Readiness` | [Capability closure](src/simulation/research/capabilities.rs), [boundary elimination](src/simulation/research/capabilities/boundary.rs), and separate discovery/mastery/permission/adoption gates; [research tests](tests/research.rs). |
| `Accounting`, `Blocks`, `Units` | Signed inventory reconciliation and horizons, research input accounts, block inverses, boundary demand, lifetime conventions, unit conversion and rebasing; [retained account tests](tests/retained.rs). |
| `Sensitivity`, `Truncation` | Signed derivatives, finite changes, coefficient responses, residual errors and weighted Neumann bounds in [sensitivity](src/simulation/analytics/sensitivity.rs); [sensitivity tests](tests/sensitivity.rs). |
| Simulator `Ledger`, `Packets`, `Execution` | Grouped gross debits, authorization, atomic posting, packets, fractional claims, seeded CubeK research draws, dependency scheduling and audited checkpoints; [contracts](tests/contracts.rs), [execution](tests/execution.rs), and [GPU replay](tests/gpu.rs). Deserialized bundles and probabilities pass through their constructors. |
| Simulator `Policies`, `Finance` | Household time, fatigue, quotes, forecasts, production starts, equipment, stocks, research hazards, payroll, bargaining, lending, dated funding, support and bank settlement; [household](tests/households.rs), [policy](tests/policies.rs), [payroll](tests/payroll.rs), [bargaining](tests/bargaining.rs), [finance](tests/finance.rs), and [institution](tests/institutions.rs) tests. |

The optional `valuation` object in [the analysis example](scenarios/analysis.json) selects zero-based goods indices. Each record reports hours per unit of output using the request's scale; historical research spending and scalar labor weights do not change those hours. Capability labels are 32-bit masks. `"InputSupport"` follows strictly positive goods inputs; `{"Declared": [{"prerequisite": 0, "dependent": 1}]}` uses the supplied dependency graph. Omitting capability inputs reports `MissingData`; an explicitly empty label set reports `Available(0)`. Boolean support remains available when the numerical account is inadmissible or unresolved.

The simulation ledger uses bounded integers and versioned nonlinear tables. Analytics uses the `CubekF32V2` profile: `DeviceVector::from_ticks` quantizes inputs to f32, and solvers return `ApproximateValues`. The existing `Interval` and signed tick report fields round the floating estimates down and up at readback; they are approximate ranges, not certified enclosures. Residual and truncation estimates omit floating-point roundoff bounds. Values beyond f32 precision lose integer detail, and estimates outside the finite nonnegative tick range are rejected. Positive fractional denominators remain usable even when their reported lower tick rounds to zero. It does not execute arbitrary real-valued inputs, unbounded catalogues, or every institutional scenario from the inherited PDFs. Runtime registries and the fixed 16-good catalogue define the concrete economy; the analytical APIs accept separate finite catalogues. Lean's conditional scheduling theorem, empirical calibration, and arbitrary scenario completeness are not runtime certification claims.

CubeK random streams are keyed by scenario seed, day, and purpose (research, strikes, or demography). Demography combines two uniform f32 draws into 48 random bits for mortality and birth thresholds. Replay is checked on the same GPU across workgroup sizes and checkpoint restoration; bitwise random outcomes are not promised across different devices or CubeK versions. The new streams intentionally change outcomes from older profiles.

## Verify and benchmark

```sh
cargo test --features viewer
cargo clippy --all-targets --features viewer -- -D warnings
cargo run --release -- benchmark --sizes 1000000 --warmup 2 --days 60 --output /tmp/benchmark.json
cargo run --release -- benchmark --restore /tmp/economy.checkpoint.json --warmup 1 --days 30 --output /tmp/continuation-benchmark.json
```

Compute tests run exclusively on GPU and compare phase states across workgroups, the original transaction scheduler, and independent integer oracles. Analytics checks use independent f64 references and f32 tolerances, including one tick for report rounding. The test helper always selects the GPU. Benchmarks record daily timings, event counts, memory, readback volume, and state hashes. Startup and checkpoint restoration are reported separately. A restored benchmark measures that checkpoint's economy; its starting day follows the requested warmup. The final state download is outside the timed interval.

The [M5 Pro optimization measurements](benchmarks/gpu-speedup.json) compare three release runs per revision using seed 42, workgroup 64, two warmup days, and 60 measured days. Median time at one million households improved from 785 to 687 ms/day (1.14×); 10k and 100k improved 1.58× and 1.63×. Final hashes match and all runs fit their memory budgets. The 10× target was not reached. Goods matching and audit/registry work remain the largest costs. Single-run, ten-day ablations favored the original merge/sparse settlement over packed radix settlement; the report includes those results separately from acceptance measurements.

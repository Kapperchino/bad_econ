# Spatial economy

An observable grain–flour–bread economy for up to one million households and 10,000 firms in the baseline. Bevy presents the map and controls; custom Burn operations run CubeCL Rust kernels on either WGPU or CubeCL's CPU backend. There is no separate CPU economy implementation and no handwritten WGSL.

Rust 1.97.1, Bevy 0.20.1, Burn 0.22.1, and CubeCL 0.11 are pinned by the toolchain and lockfile. Both rendering and simulation use WGPU 30. Autodiff and automatic fusion are disabled. Headless commands never initialize Bevy's renderer.

## Run

```sh
cargo run -- run --households 1000000 --days 30
cargo run -- run --backend cpu --households 100 --days 10
cargo run --features viewer -- view --households 1000000
cargo run -- run --scenario scenarios/transport.json --days 30
cargo run -- run --scenario scenarios/custom-population.json --days 4
cargo run -- run --scenario scenarios/institutions.json --days 12
cargo run -- run --households 10000 --days 10 --checkpoint /tmp/economy.json
cargo run -- run --restore /tmp/economy.json --days 10
```

`--seed`, `--workgroup` (32/64/128/256), and `--experiment baseline|shortage|transport|concentrated` select reproducible runs. `--output PATH` saves the result as JSON. Export a configuration with `cargo run -- scenario --households 1000000 --output /tmp/scenario.json`; edit its validated parameters and load it with `--scenario`. A scenario file supplies its own seed and population settings; generation flags such as `--households`, `--seed`, and `--experiment` apply when no file is supplied.

The viewer starts paused. Space pauses/resumes, N advances one audited day, R resets, 1/2/3 select speed limits, H/F select a household/firm, and Tab advances the inspection selection. G cycles employment/shortages, prices, wages, inventories, and the household cash histogram. The map shows regional shortages and firm positions; price and wage statistics include unweighted means and standard deviations of firm offers for each industry.

C retains the current seed's recent history, resets to the next seed, and runs to the same day before pausing. Comparison values and faint chart reference traces always refer to matching days; history is limited to the most recent 160 snapshots. Explicit populations supply their own positions and resources, so changing their seed alone leaves the economy unchanged.

Simulation runs on a worker using Bevy's imported device and queue. Rendering uses the last published snapshot and never creates a household entity. Inspection reads only the requested actor's fields and labels the observation day; firm records include owner identity, wage, price, capacity, planned packets, and open slots.

## Economic rules

Each household has cash, goods, needs, health, fatigue, and daily labor time. Each firm has cash, stocks, capacity, a technique, wage and price offers, and one explicit household owner. Actor identities are permanent throughout this fixed-population model. Logical IDs can differ from buffer positions. Households and firms each occupy a contiguous range sorted by logical ID; entry, deletion, and compaction are not implemented.

Resources use unsigned 64-bit ticks. Time is in minutes; distance is integer Manhattan distance. Commute speed is distance per minute, shipment speed is distance per day, wage is cash per working minute, and prices are cash per goods tick. Health, fatigue, forecast weight, and distribution rate use a scale of 1,000. Generated populations use a production packet with `packet_minutes` of labor and `packet_output` ticks of output; mills and bakeries also consume that many ticks of grain or flour. Explicit firm techniques can set different labor durations, input quantities, and output quantities. Farms require labor alone.

A day has nine ordered phases:

1. **Opening:** mature buyer-owned deliveries and reset daily counters.
2. **Observation:** compute labor capacity from health, fatigue, and reserved time.
3. **Proposal:** forecast production requirements and cap starts by inputs, capacity, and wage funding.
4. **Commitment:** run four labor matching rounds; reserve cash in escrow and debit time and production inputs atomically.
5. **Work/payment:** transfer committed wages to workers.
6. **Provision/purchase:** run four refreshed matching rounds for each input market and household bread; transfer payment at dispatch and add dated, buyer-owned transit stock.
7. **Physical close:** consume available bread, update health/fatigue, and finish production. New output first reaches the next day's market.
8. **Accounts/decisions:** update demand, prices, and wages, then distribute a configured fraction of cash above the operating reserve.
9. **Audit:** verify cash conservation, goods source/use identities, delivery-ring ownership, and cleared commitments; publish an aggregate and retain a device copy for rollback.

A GPU-resident spatial index filters eligible suppliers/employers on each round. An actor discovers at most eight eligible candidates in canonical cell/identity traversal order within its radius, then ranks those by wage after commuting time or price plus delivery penalty. Ties use identity. Proposals refresh between the four sequential rounds. Discovery is intentionally bounded: it does not find a global optimum or guarantee a match. Firms without affordable inputs or wage funding stay idle.

Deliveries take at least one day, including at zero distance. Pending stock cannot be consumed or used in production. A bounded dated ring is sufficient because positions and maximum discovery distance are fixed. Each buyer purchases only one goods type, so ring slots aggregate deliveries of that type while preserving due day and ownership.

Demand is the integer convex forecast of sales plus rejected, initially funded purchase offers. Repeated attempts in different matching rounds are distinct observations. This observation convention, matching, sole ownership, and reserve distributions are scenario rules, not Lean theorems. The initial buffers, quantities, speeds, radius, funding, coverage, policy coefficients, and quote bounds are configuration inputs. Health/fatigue increments and price-response strength are configurable; the nonlinear price lookup is versioned.

## Configure a population

Omitting `population` generates households, firms, ownership, and positions from the scenario defaults and seed. The optional `population` object replaces these initial actor records; its household and firm counts must match the top-level counts. [The custom population example](scenarios/custom-population.json) supplies three households and three firms with sparse IDs, distinct owners, and different production packets.

Household records specify `identity`, `position`, `cash`, `inventories`, `health`, and `fatigue`. Firm records specify `identity`, `position`, household `owner` identity, output `good` (`Grain`, `Flour`, or `Bread`), cash, inventories, capacity, wage, price, and `technique`. All inventory arrays are ordered grain, flour, bread. A technique specifies `labor_minutes`, `input_ticks`, and `output_ticks`; mills consume grain and bakeries consume flour. Capacity counts production packets per day. Farm input must be zero; mill and bakery inputs must be positive.

Admission rejects duplicate/zero IDs, unknown owners, invalid positions or quotes, invalid packet bounds, and overflowing opening totals. Input order is normalized by logical ID. The population and technique definitions are retained in checkpoints. Top-level need, time, transport, and policy parameters still apply to all actors.

The optional `behavior` object has these defaults:

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `health_recovery` | 5 | Health gained after meeting bread need |
| `hunger_damage` | 20 | Health lost after any bread shortage |
| `work_fatigue` | 10 | Fatigue gained after employment |
| `rest_recovery` | 20 | Fatigue recovered when unemployed |
| `wage_increment` | 1 | Wage increase considered when funded slots remain open |
| `price_response` | 1000 | Price adjustment strength, from 0 (unchanged) to 1000 (full response) |

Health and fatigue clamp to 0–1000. Wage increases remain constrained by funding and the wage bounds. The four health/fatigue coefficients and price response accept 0–1000; the wage increment accepts 0–10000. Omitted behavior preserves the original defaults.

## Institutions

[The institutions scenario](scenarios/institutions.json) enables payroll taxes, wage bargaining, a funded bank loan, household support, research, production permissions, equipment, a renewable stock, and strikes. Their calculations and resource updates use the same CubeCL kernels on CPU and GPU. No big-integer arithmetic or separate CPU institution implementation is used.

The optional `payroll` object specifies `base_rate`, `employee_charge`, and up to eight ordered `brackets` with `threshold` and `marginal_rate`. Rates use a scale of 1,000; every marginal tax rate plus the employee charge must be below 1,000. Thresholds are cumulative gross income ticks. Income accumulates across employers and days. Payments partition gross pay into net pay, tax, and employee charges. Each cumulative entitlement settles whole ticks; up to two cash ticks remain in a payroll reserve to fund fractional entitlements. A rejected payment changes none of these accounts. Tax and charge treasury balances are stored per household to avoid a shared write bottleneck and reduced for observation.

The optional `household_policy` selects the v6.2 health and fatigue formulas in place of the original constant increments. Its `recovery`, `deficit_damage`, `fatigue_damage`, and `fatigue_response` use scale 1,000. `rest_minutes` must fit the reserved daily time. The current household service is delivered bread; its shortage fraction is rounded down to scale 1,000 before applying the health response. Fatigue uses executed labor and research minutes. Results round down and clamp to 0–1,000.

The optional `bargaining` policy computes a household reservation payment from daily cash cost, other funded income, participating adults, normal working minutes, cash cover, and work cost. It inverts cumulative payroll taxes across the configured brackets, rounds the legal wage floor upward and the funded ceiling downward, then applies the bargaining-weight second offer. The negotiated wage funds the actual labor commitment and payroll transfer. An empty wage interval prevents the match. Reservation payment fractions are retained until the final wage grid; cash-cover days round down to scale 1,000. Bargaining factors use the same scale. A `desired_wage` of zero uses the firm's current offer.

`finance` configures banks, dated loan contracts, and an optional support treasury. These banks lend existing reserve cash. Initial bank reserves equal equity, and lending transfers reserves to the borrower's cash account; it does not create money. Loans are capped by the request, dated cash target, debt service, collateral, capital, and available reserves. Consent is explicit. Opening accrues interest and attempts due repayments; an unfunded repayment remains overdue. Fractional interest remains recorded after whole-tick settlement. Independent banks run concurrently, with each bank processing its contracts in canonical order. Support considers each consenting household once and distributes the available treasury proportionally to its cash shortfall, using canonical cumulative rounding to conserve whole ticks.

`research` configures labs, equipment, natural stocks, and strikes. Each lab has separate discovered, mastered, permitted, and adopted capability masks, explicit consent, a household researcher, and a firm owning its inputs. Production requirements must satisfy every access gate. A funded attempt consumes its inputs and remaining daily time whether it succeeds or fails. Automatic mastery/adoption is an explicit policy; grants remain reusable. Research and maintenance execute before physical close, so they cannot use the day's unfinished output. Equipment condition can recover through maintenance, while lifetime service only decreases with executed production. Renewable stocks are depleted by executed packets and then replenished by the configured bounded growth rule. Strike decisions remove labor availability before matching.

Research uses the versioned `ResearchFixedV1` profile. Capability masks carry 32 labels per lab. The hazard table stores `floor(1,000,000 × (1 − exp(−i/16)))` for indices 0–256; milliscale arguments interpolate between entries and clamp at 16. Research effort scales and exposure round down. Success and strike draws use SHA-256 logical keys and exact 128-bit floored thresholds for the table probability. The table approximates the real policy curve; the checkpoint records its hash and profile.

Configured projects remain eligible for repeated attempts. The summary's `discoveries` counter counts successful attempts; capability grants are idempotent, so it is not a count of distinct capabilities.

Checkpoints include bank and loan accounts, payroll accruals and reserves, research access states, attempt counters, equipment, natural stocks, and strike outcomes. Mid-day checkpoints retain the last audited state for all institutions. The state layout now uses checkpoint version 2 and state digest version 3; version 1 checkpoints are rejected. JSON run results and the viewer include institution balances and research summaries, and household inspection includes payroll details.

## Exact execution

`DeterministicV1` is a local arithmetic profile, not a claim of full GPU-D1 conformance. Kernels represent every 64-bit value as two `u32` words, using carry/borrow checks, limb multiplication, long division, and quotient/remainder decomposition for scaled products. Host audits and independent arithmetic fixtures use `u128` intermediates. Floating point appears only in presentation and timing.

Policies specify floor rounding for forecasts, funded capacities, and reserve distributions; price quotes round up and obey the configured integer grid bounds. The price lookup contains `ceil(1_000_000 × exp((i−32)/640))` for indices 0 through 64. Its SHA-256 hash and the arithmetic profile are checkpointed. The integer imbalance index is clipped to ±32, then its magnitude is multiplied by `price_response / 1000` and rounded down before selecting the table entry. Logical random draws hash a versioned prefix and fixed-width little-endian seed/day/phase/actor/purpose/index fields. Probability tests retain the floored threshold, including exact zero and one cases. Randomness seeds fixed actor positions and controls enabled research and strike outcomes; daily matching has no random draws.

Transactions contain grouped gross debits, credits, and authorization reads. All legs must be affordable and representable before any leg posts. Canonical order is day, phase, subphase, priority, then actor identity. The scheduler sorts coordinate/event incidences and links consecutive events sharing each coordinate, including authorization reads. Independent frontiers execute on device; acceptance and rejection both release successors. Disjoint linear chains execute sequentially inside separate invocations. Branched dependencies use successive kernel dispatches with small control readbacks, never cross-workgroup spin waits.

Canonical batches bound scratch allocation. Economic state, proposals, event construction, indexing, dependency construction, execution, summaries, and delivery scheduling remain on device. Base observation reads 1,696 bytes plus scheduler counters; enabled institutions add their summary and control reads. Diagnostic replay and checkpoints explicitly download full state. Per-industry quote counts, minima, maxima, sums, and squared sums use exact integer reduction. Floating-point means and standard deviations are display values. The cash histogram counts household liquid cash, excluding inventory valuation and firm ownership value. The worker makes those reads while rendering continues independently.

Authorization predicates and fault reports use typed `bitflags` sets; phases, markets, actor kinds, kernel stages, scheduler dependency markers, and event statuses use `enumflags2`. Kernel parameters and buffer bindings use named structs; field positions, buffer layouts, arithmetic widths, and fixed policy limits use shared names. Each event has exactly one status flag, with conflicting or empty status sets rejected on readback. CubeCL keeps typed authorization, event status, and fault wrappers around the `u32` buffer encoding, and concurrent reports combine with atomic OR. Host readback rejects unknown flag bits and event statuses. Arithmetic or capacity faults restore the last audited device state and enter `Fault`. Reset or checkpoint restoration is required before resuming. Checkpoints include the scenario, full audited state, transit ring, commitment fields, logical identity counter, profile/table versions, and a content checksum. Mid-day requests return the previous audited boundary. The checksum detects corruption; it is not an authentication mechanism. The versioned state digest covers day and identity counters, actor metadata, opening cash/goods totals, and all resource values; the scenario fingerprint separately identifies techniques and policy inputs.

## Analytical accounts

```sh
cargo run -- analyze --analysis scenarios/analysis.json --output /tmp/accounts.json
cargo run -- analyze --analysis scenarios/analysis.json --backend cpu
```

[The analysis request](scenarios/analysis.json) supplies the Lean grain/bread teaching model, selected techniques, occupational weights, research plan, reproduction inputs, context versions, and contraction witnesses. These are explicitly supplied analytical inputs, separate from a simulation's heterogeneous firm offers. The report includes goods values, gross requirements, occupational hours, research costs, plan totals, optional occupational closure, and a labeled surplus comparator. Values are reference requirements, not prices or money profits.

The `DirectedRationalTicksV1` solver runs sparse products, simultaneous coupled updates, certificate checks, residual bounds, and normalization on CubeCL. Coefficients are bounded integer ratios; vectors and interval endpoints use two-word 64-bit ticks. The request's `scale` defines presentation units. Products and division round outwards to retain certified intervals. A contraction witness must apply to the actual operator, so goods and transposed value equations require separate certificates. Reports distinguish available results, missing data, inapplicability, inadmissibility, and unresolved accuracy. Failure of optional occupational closure does not remove a valid goods account. Context-version equality controls cached account reuse.

The library also exposes device APIs for capability closure, dated funding, exact quote directions, dense interval accounts, boundary elimination, unit transformations, inventory reconciliation, signed sensitivity transforms, finite changes, and certified weighted Neumann truncation. Independent Lean fixtures check these contracts on both supported backends. See [the comparison checklist](IMPLEMENTATION.md) for implementation and verification coverage.

## Verify and measure

```sh
cargo test --features viewer
cargo clippy --all-targets --features viewer -- -D warnings
cargo run -- replay --households 100 --days 8 --workgroup 64
cargo run --features viewer -- view-check --households 10000
cargo run -- benchmark --sizes 10000,100000,1000000 --warmup 2 --days 10 --output reports/baseline.json
cargo run -- benchmark --experiment concentrated --warmup 2 --days 10 --output reports/concentrated.json
cargo run -- benchmark --sizes 1000000 --warmup 2 --days 60 --output reports/sustained.json
cargo run -- benchmark --sizes 10000 --warmup 2 --days 30 --output reports/experiment-baseline.json
cargo run -- benchmark --sizes 10000 --experiment shortage --warmup 2 --days 30 --output reports/shortage.json
cargo run -- benchmark --sizes 10000 --experiment transport --warmup 2 --days 30 --output reports/transport.json
```

Tests use the GPU by default and require an actual adapter. Standalone CPU backend checks are ignored; parity tests run a paired CubeCL CPU reference alongside the GPU. Replay compares decisions and state after every phase, including repeated runs and four workgroup sizes. Independent serial bundle fixtures, wide-integer arithmetic oracles, and conservation audits provide separate implementation checks; shared CPU/GPU kernels alone are not an independent economic reference.

Regression fixtures cover the neighboring Lean library's simulator examples and retained analytical contracts: event ordering, gross affordability, packets and fractional claims, policy grids and draws, payroll and bargaining, finance and support, sparse and coupled accounts, research plans, residual bounds, inventory, units, capabilities, derivatives, and truncation. Lean supplies contracts and examples; it does not prove this Rust/CubeCL implementation.

Benchmark mode generates scenarios from `--sizes`, `--seed`, and `--experiment`. Supplying `--scenario` benchmarks that fixed population once instead of the size list. GPU is the default; `--backend cpu` selects CubeCL CPU. Benchmark JSON records daily throughput, phase time, maximum per-batch frontier depth, event counts, readback bytes, device allocator memory, and scenario/profile/table/state hashes. The final state download and hash are outside the measured interval. The 4 GiB limit covers live and reserved simulation buffers, including scratch and the audited backup. Driver/pipeline caches, host memory, and Bevy rendering allocations are outside this headless simulation budget. See [verification results](reports/README.md) for the measured hardware and backend coverage. Economic outcomes are observations rather than pass/fail targets.

Roads, congestion, carriers, firm entry, and bankruptcy resolution remain outside this slice. The Lean library does not specify their institutional handlers. See [the comparison checklist](IMPLEMENTATION.md) for contract coverage and verification.

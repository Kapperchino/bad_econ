# Lean comparison and implementation tracking

Target: the contracts in `../marx-lean-proof`, including the simulator v6.2
modules and the research-accounting modules they retain. The user also requested
new institutions in the running simulation. Formula APIs alone do not complete
that requirement.

## Existing implementation

The initial repository implements gross-debit atomic bundles, authorization-read
dependencies, canonical scheduling, production packets, fractional claims,
integer quote grids, keyed draws, and a fixed-population spatial grain/flour/bread
economy. Its health/fatigue rules are simplified scenario rules. The initial
`cargo test --no-default-features` passes eight tests.

## Required work and completion evidence

| Contract | Implementation and verification required | Status |
| --- | --- | --- |
| Exact arithmetic and units | Rational finance/payroll; validated quanta; 128-bit draw thresholds; overflow and invalid-domain tests | Implemented with bounded integer arithmetic; GPU arithmetic, payroll, finance, bargaining and draw fixtures pass |
| Ledger and payments | Cross-bank deposit/reserve events, fractional settlement tied to accepted transfers, tests for rejected transactions | Implemented; device payment construction, complete-bundle rejection, retained claims and rollback fixtures |
| Households and work | Weighted delivered-service deficits, participation/shared clock, v6.2 health and fatigue policies, phase integration | Implemented; four GPU household fixtures cover weighted/zero needs, participation, missing observations, wide ticks and admission; shared helpers drive phase integration |
| Firms and equipment | Demand/starts, lifetime service, condition/maintenance, renewable stocks, dispatch/input accounting and checkpoints | Implemented in packet production; GPU fixtures verify service/stock limits, combined maintenance/wear clipping and checkpoint persistence |
| Research | Effort scale, success/exposure, attempts consume inputs regardless of outcome, grants and adoption gates, deterministic replay | Implemented with ResearchFixedV1; 13 GPU integration/graph fixtures plus nonlinear-table and exact SHA-256/draw boundary tests |
| Payroll and bargaining | Hinge taxes, cumulative withholding, bracket inversion, directional grids, anchors/offers, actual payroll transfers | Implemented; 12 payroll integration tests pass on GPU, plus bargaining and 31,824 inversion fixtures |
| Finance and support | Lending caps, dated cash need, earmarks/rents/discounting, proportional grants, divisible pools, bank/loan/support events | Implemented in shared CubeCL kernels; 14 finance fixtures pass on GPU, including support event counts |
| Collective institutions | Strike scores/decisions, authorization and participation effects, funded support without duplicate recipients | Implemented; GPU strike/shared-clock and funded-support fixtures verify actual participation and transfers |
| Analytical accounts | Goods and occupational accounts, coupled solve and normalization, research/plan totals, labeled comparators | Implemented with device-resident sparse/dense APIs and user-supplied analysis CLI; 14 GPU analytical fixtures verify the source equations and complete normalized pair |
| Certified solver | Actual-operator contraction certificates, residual and normalization bounds, diagnostics and context cache invalidation | Implemented with directed rational tick intervals; GPU fixtures cover actual-operator witnesses, weighted residuals, zero lower bounds, invalid optional closure and context-version invalidation |
| Capabilities | Forward closure, reversed masked missing-project closure, readiness gates, project access and reusable grants | Implemented on device, including positive matrix support and Boolean boundary elimination; cyclic/empty graph, gate and local-grant fixtures |
| Retained research accounting | Inventory/horizon identities, boundary reduction, unit conversions, sensitivity and truncation interfaces and fixtures | Implemented; 7 retained-accounting and 11 sensitivity fixtures pass on GPU, with source mapping and profile limits below |
| Runtime integration | Validated scenarios, all nine phases, device execution, rollback, complete checkpoints/digests, observations and CLI/viewer | Implemented; full phase replay, institutional checkpoints/rollback, CLI examples and viewer verification pass |
| Verification and documentation | Port every simulator worked example and review regression; institution scenarios, CPU/GPU replay, feature builds and lint | Verified; 116 tests pass with the GPU backend, five optional CPU-only checks ignored; all-feature lint and no-default-feature build pass |

The mathematical library proves contracts, not an implementation. New code must
retain that distinction, keep analytical values separate from prices and cash,
and expose unavailable or inadmissible results without fabricating zero values.
Existing scenarios must continue to run when optional institutions are omitted.

User constraints: no big integers; use the existing shared CubeCL kernels for
both CPU and GPU execution. No separate host production implementation of the
new policies or institutions. Host work is configuration validation, orchestration,
observation, and test fixtures. All new economic and analytical computations must
have GPU execution paths.

Tests now select the GPU by default. Standalone CPU backend checks are optional
ignored fixtures; backend replay retains a paired CPU reference and checks the
GPU after every phase. Earlier CPU runs below are historical verification,
performed before the user requested GPU testing exclusively.

## Execution, payroll and integration evidence

The existing ledger and scheduler implement `Simulator/Ledger.lean` and the
complete-footprint ordering contract in `Simulator/Execution.lean`. New cross-bank
payment construction uses that same device scheduler. Complete gross debits and
authorization reads are checked before posting. The indivisible `[6, 4]` job
counterexample with capacity 5 is exercised independently of the divisible pool
API. `tests/contracts.rs`, `tests/execution.rs` and `tests/gpu.rs` cover rejection,
wide guards, branched dependencies, overflow and audited rollback.

`src/domain/payroll.rs`, `src/simulation/payroll.rs`, `src/simulation/bargaining.rs`
and `src/simulation/negotiation.rs` implement the tax, net-payment, inversion,
household-anchor and wage-grid contracts of `Simulator/Policies.lean`. Cumulative
withholding settles only whole ticks and retains cash backing for fractional
entitlements. Tax thresholds do not restart when the employer changes. Bracket
inversion traverses at most eight segments, preserving the reservation payment's
fraction until the wage grid is selected. Labor planning uses the legal grid
floor and remaining slots share remaining funding. Regression fixtures check
affordable offers below a posted wage and funding in later matching rounds.
The GPU teaching fixture also checks `proposedStarts 4 10 20 5 = 15`, net
incomes 68 and 85, and the fractional net-payment example 24.02 in hundredth
cash units.

Optional institution configurations are validated and checkpointed by
`src/domain.rs`; `src/simulation/mod.rs` dispatches them at their dated phase
boundaries and restores all state on a fault. Finance accounts share the main
device state buffer. Research has a complete audited device backup and snapshot.
The digest includes institution state, research configuration and policy-table
identity. Aggregate conservation includes bank/support cash, payroll treasuries,
reserves and research inputs. Labor, commute and research share the daily clock.

`scenarios/institutions.json` enables the new institutions together, including
negotiated wages. Its 12-day CLI replay compared all nine phases' decisions and
complete CPU/GPU state successfully. `view-check` passed with this scenario,
including the shared rendering/compute device, inspection, same-day comparison
and all chart captures; the rendered institution summary was inspected. The
analytical CLI example is `scenarios/analysis.json`. A short one-million-household
GPU run and its buffer-memory evidence are recorded in `reports/`.
The final institutions CLI checkpoint restored at day six and advanced to day
twelve matches the uninterrupted GPU run's complete checkpoint byte-for-byte
and its observation snapshot exactly. The final analytical CLI report encloses
all eight teaching accounts, including the normalized residual scale 1/4.

## Finance implementation evidence

`src/domain/finance.rs` validates and lays out funded lenders, one-shot loan
contracts, and support recipients. `src/simulation/finance.rs` and its submodules
run lending, dated simple interest, maturity/overdue repayment, support allocation,
bank audits, financial policies, and cross-bank payment construction on CubeCL.
The parent simulation connects lending before production planning and support
before purchases; the complete financial tail is included in state copies,
checkpoints, digests, and cash audits.

Banks lend existing reserve cash and do not create deposits. Bank reserves plus
outstanding principal equal bank equity; realized interest increases equity.
Interest fractions remain payable claims after whole-tick settlement. Consent is
an explicit immutable contract gate. Independent banks execute concurrently,
with canonical sequential loan dependencies within each bank.

Rates use scale 1,000. Loan bounds round down; weighted risk assets round up.
Support uses parallel prefix allocation with cumulative floor rounding, preserving
the exact available-treasury total and every individual need bound. Discounted
flows return conservative lower/upper tick bounds using outward-rounded discount
weights at scale 1,000,000. No big integers or floating-point economic arithmetic
are used. The standalone cross-bank API keeps deposits and reserves as separate
instruments and constructs complete bundles on device for the existing scheduler.

Verified commands: `cargo test --test finance -- --nocapture` and
`BAD_ECON_TEST_BACKEND=gpu cargo test --test finance -- --nocapture`, each passing
14 tests. Fixtures cover the Lean funding/support/pool/cross-bank examples,
u64-width allocation, every lending constraint, fractional maturity claims,
consent and empty cases, malformed configuration, forged risk balances, support
overflow without posting, and full simulation rollback after a loan fault.

The final finance audit maps every computational definition in
`Simulator/Finance.lean` to an executable interface:

| Lean definitions | Device implementation and evidence |
| --- | --- |
| `lendingLimit`, `lendingLimit_liquidity` | `FinancePolicies::lending` and actual loan issuance share `lending_limit`; fixtures isolate each cap and verify invalid domains |
| `openingNeed`, `financingGap`, `openingNeed_iff` | `FinancePolicies::funding` includes prefix zero, nondecreasing execution dates, signed outflow/receipt pairs and every dated cushion |
| `replacementEarmark`, `rentBid` | `FinancePolicies::budgets` clips before fraction multiplication |
| `discountWeight`, `discountedFlows` | `FinancePolicies::discounted_flows` returns signed interval bounds and starts the first flow at day zero |
| `supportScale`, `supportGrant` and bounds/total | `FinanceRuntime::support` constructs cumulative proportional grants, checks all destination additions before posting, and counts nonzero transfers on device |
| `prefixFills`, residual and total identities | `FinancePolicies::pools` implements the divisible serial residual allocation separately from indivisible scheduler events |
| `Ledger.crossBankDelta` and claims | Device construction of complete deposit/reserve bundles; repayment retains fractional interest and rejected transfers leave balances unchanged |

The financial profile admits unsigned 64-bit cash/quantities, rates and fractions
in thousandths, daily rates up to one, positive capital ratios up to one, risk
weights up to ten, and terms from 1 to 36,500 days. Lending uses conservative
nested integer floors, so the returned amount can be below the floor of the
unbounded real formula. Exact-real support shares are represented by cumulative
floor differences: each share differs by less than one tick, every individual
need bound is preserved, and total support equals the available funded amount.
Discount weights use scale 1,000,000 and outward rounding. All intermediate
accumulations must fit bounded words; a small net result does not waive a gross
intermediate overflow. Loan lifecycle and funded reserve accounting are explicit
scenario policies, not a claim that Lean prescribes fractional-reserve banking.
`FinanceConfig::memory_estimate` counts runtime metadata and scratch buffers
separately from the state tail already covered by checkpoint memory accounting.

## Retained accounting source audit

`src/simulation/analytics/retained.rs` closes the horizon and boundary-demand gaps
found during the final comparison. `InventoryHorizon` reports chronological
hours, final/project hours, accumulated and terminal signed stock adjustments,
per-stage physical residuals, and the final reconciliation residual. It rejects
context changes before applying the constant-context telescoping identity and
admits at most 4096 stages. Changing project input/hour vectors is allowed.
`BoundaryDemandAccount` retains both effective retained demand and the outside
direct-labor contribution; dropping that contribution is no longer necessary
when using the reduced system. `AugmentedInverse` constructs all four inverse
blocks and applies both source blocks on device.

| Retained source obligations | Executable coverage |
| --- | --- |
| `Accounting.projectLabor`, fresh/scalar project identities | `DeviceAccounts` research and plan APIs; device matrix product/addition; existing analytical fixtures |
| `Accounting.inventory_reconciliation`, `inventory_telescope`, `inventory_horizon` | `InventoryAccount` and `InventoryHorizon`; fixtures include multiple projects/stages, stock rundown, zero horizons, changed contexts and broken physical balances |
| `Accounting.boundary_labor`, solution, conversion and demand | `BoundaryReduction`, matrix products and `BoundaryDemandAccount`; fixtures compare retained labor balance, full/reduced reproduction matrices and external final-demand totals |
| `Blocks.augmented_inverse`, append and terminal labor | `AugmentedInverse` and matrix product/addition; all four inverse blocks, both source equations and multiple terminal activities are exercised |
| `Units` physical, occupational, residual, reference and lifetime covariance | Matrix/vector `scale_units`, positive `UnitFactor`, transposes, normalization and per-hour constructors; fixtures transform A, N, H, G, F, T, demand, weights, residuals, subsistence and lifetime training together |
| `Conversion.beta`, `alpha`, `kappa`, residual and ratio bounds | Existing certified solve/normalization plus `DeviceVector::reciprocal_reference`; the normalized weights and residual scale are checked against the residual equation |
| `Sensitivity` inverse/labor/coefficient/reproduction/conversion/normalized/value formulas and residual bound | `Sensitivity`, signed matrix/vector APIs and 11 sensitivity fixtures; finite labor changes use the new inverse |
| `Truncation` ordinary and weighted Neumann bounds | `LaborTruncation` and `SignedLaborTruncation`; actual-operator witness checks, retained-term count, signed direct matrices, zero contraction and overflow fixtures |

These APIs implement the finite accounting equations under supplied valid
inverse/productivity assumptions. They do not reconstruct Lean proof objects,
prove differentiability of arbitrary state transitions, or compute infinite
limits. The retained arithmetic profile uses nonnegative rational coefficients,
positive rational diagonal unit factors, outward-rounded unsigned 64-bit
interval endpoints, and positive/negative parts for signed results. Dense
matrices admit at most 1,048,576 cells and 65,535 rows/columns. Generic arbitrary
real changes of basis are outside this profile. Different operation orders can
produce different enclosing intervals even when Lean proves the real results
equal. Horizon telescoping is exposed with residuals rather than asserted for
unbalanced physical histories. All arithmetic, including sums across stages,
unit changes and support event counts, runs through shared CubeCL kernels.

Final targeted GPU verification passed: 14 finance tests, 7 retained-accounting
tests, and 11 sensitivity tests. The retained command was
`BAD_ECON_TEST_BACKEND=gpu cargo test --test retained -- --nocapture`.

## Sensitivity and truncation implementation

`src/simulation/analytics/sensitivity.rs` provides signed matrix directions and
the retained formulas for inverse, occupational labor, individual coefficient,
reproduction, conversion, reference normalization, and scalar-value derivatives.
Finite labor changes use the new inverse, as required by the finite-change
identity, and residual error propagation uses the absolute signed residual.
These are conditional algebraic transforms of the supplied account matrices;
they do not assert differentiability of an arbitrary scenario transition.

`LaborTruncation` checks the actual sparse operator against a supplied positive
weighted contraction witness on device. It computes the first `terms` Neumann
terms, an outward-rounded weighted tail bound, and an entrywise enclosure of
the full labor requirements. Zero terms and zero contraction are supported.
The tail bounds the exact mathematical truncation; rounding uncertainty is
also retained in the partial-sum intervals and final enclosure. Invalid or
insufficient witnesses yield analytical diagnostics. Intermediate overflow is
reported rather than wrapped. The finite implementation admits at most 4096
terms and uses the existing bounded rational-tick profile. `SignedLaborTruncation`
extends the enclosure to signed direct matrices by propagating positive and
negative parts and adding their conservative weighted tail bounds.

Every derivative, signed operation, certificate check, matrix product, norm,
and tail calculation executes through shared CubeCL kernels on CPU and GPU.
Host work validates shapes, uploads configuration, schedules kernels, and
decodes reports. `tests/sensitivity.rs` contains the Lean formula fixtures,
signed and asymmetric cases, wide ticks, term counting, weighted witnesses,
zero and invalid cases, and bounded-overflow checks.

Verified commands: `cargo test --test sensitivity -- --nocapture` and
`BAD_ECON_TEST_BACKEND=gpu cargo test --test sensitivity -- --nocapture`, each
passing 11 tests; `cargo clippy --lib -- -D warnings` also passes.


## Research, physical institutions and capability evidence

`Simulator/Policies.lean` functions `machineServiceCap`, `nextMachine`,
`renewableStock`, `researchScale`, `researchSuccess`, `nextExposure`, `strikeScore`
and `sigmoid` map to `src/simulation/research/kernels.rs` and `nonlinear.rs`.
`Simulator/Execution.lean` supplies the separate readiness gates and floored
128-bit draw contract. `Model.lean` supplies local, reusable result grants and
physical attempts whose inputs and direct time remain consumed after failure.

The live phase hooks cap both `Plan` and `Slots` after Proposal by installed
readiness, condition/lifetime service and available natural stock. Strike draws
run after Observation and remove the affected household's available time before
matching. Maintenance and research run before PhysicalClose: committed production
inputs are already consumed, unfinished output remains unavailable, and research
uses the household's remaining shared clock. Research minutes contribute to the
optional health/fatigue policy. PhysicalClose charges actual completed packets to
machine lifetime and natural stocks; maintenance restores condition but never
lifetime. Source/use audits include institutional input consumption.

Each lab has one distinct firm and one distinct household researcher. Each firm
has at most one equipment record and one natural-stock record. These admission
rules make writes independent within each parallel phase; a lab executes its
projects in canonical identity order. Research consent and automatic adoption
are explicit configuration choices. Four access masks gate installed production;
supporting graph labels do not silently become local eligibility requirements.

`ResearchFixedV1` admits 32 capabilities per lab, positive effort scales,
integer shape exponents 1–16, and fractions on a scale of 1,000. Intermediate
research powers, effort scales, exposure updates, service capacity and renewable
growth round down. Hazard/sigmoid use the committed 257-entry exponential table
and probability scale 1,000,000; they approximate the real Lean curves. Table
fixtures check every milliscale argument, monotonicity, hash and stated error
bounds. GPU SHA-256 serializes fixed little-endian logical keys and compares four
fixed-width words against the exact floored 128-bit threshold, including zero and
one. These bounded integers are not arbitrary-precision arithmetic.

`Capabilities.lean` maps to the sparse forward and reversed masked closures in
`research/capabilities.rs`. They compute the least closure from supplied labels,
stop missing-project traversal at satisfied nodes, and use at most n−1 rounds.
`Support.lean` maps to device extraction of positive sparse-matrix entries in
`research/capabilities/boundary.rs`; inverse-support interpretation remains
conditional on the matrix being productive. Extra declared edges may enlarge
annotations. `BooleanBoundary.lean` maps to device contraction of external nodes,
producing both the effective graph and effective labels. Cycles, empty blocks,
and retained reachable-label equality have GPU fixtures. The sparse API admits
one million nodes/ten million edges; packed support and boundary elimination
admit 4,096 nodes. Each label word carries 32 independent annotations.

Research snapshots retain all access masks, attempt outcomes and counters,
exposure, equipment, stock and strike state. Restore checks profile, table hash,
canonical configuration hash, dimensions, state invariants and the device-reduced
summary. The configuration hash covers canonical institution inputs, initial
institution state, actor mappings, technique bounds and seed. Audited device
backups preserve whole-day rollback and mid-day checkpoint semantics. Typed
observations expose each lab, project, machine, stock and strike without rerunning
its economic policy on the host.

`BAD_ECON_TEST_BACKEND=gpu cargo test --test research -- --nocapture` passes
13 tests covering the source mappings above, independent readiness gates,
local/reusable grants, zero-effort failed consumption, consent/resource/time
rejections, next-day production activation, actual strike participation,
service exhaustion, renewable growth, restoration-before-clipping, deterministic
checkpoint replay and changed-configuration rejection.

## Negotiation audit evidence

`Simulator/Policies.lean` functions `householdAnchor`, `refusalFactor`,
`requiredNetRate`, `inverseBracket`, `wageFloor`, `wageCap`, `bargainingWeight`
and `secondOffer` map to `simulation/bargaining.rs`, its kernels and
`simulation/negotiation.rs`. The live helper preserves fractional reservation
payments through marginal-tax inversion and rounds only the final wage floor.
It traverses at most eight finite tax intervals and the final open interval;
there is no candidate-wise binary search. A native fixed-width test oracle checks
31,824 threshold/fraction cases on GPU, including decreasing marginal rates,
large prior incomes and denominators near the u32 bound.

Cash cover floors to millidays and bargaining weight floors to thousandths.
Those policy-grid projections are finite approximations of the real formulas;
the subsequent reservation and tax calculation is exact on the admitted grid.
Live employer surcharge is zero, consistent with the implemented payroll.
Employer funding uses the available cash reserved for remaining planned slots;
initial planning uses the legal grid rate when bargaining is enabled. The
negotiated result funds actual escrow/payroll transfers, while empty legal/funded
intervals prevent commitment. Payroll integration fixtures cover the Lean wage
example, fractional reservation precision, lower affordable negotiated offers and
remaining-slot funding, in addition to whole-phase GPU replay.

## Analytical and solver implementation evidence

`src/simulation/analytics.rs` and its account, matrix and request modules implement
`Simulator/Analytics.lean`. Sparse operators retain exact nonnegative u32/u32
coefficients. `DeviceAccounts` computes goods requirements, external occupational
weights, optional coupled reproduction, research inputs/direct hours and plan
totals. The coupled operator contains the prescribed four transposes and both
Jacobi updates read the previous pair. Reference normalization divides both
blocks by the same positive interval; `AnalysisReport::residual_scale` includes
the reciprocal reference required by the normalized residual equation.

`AnalysisRequest::run` accepts the actual supplied matrices, context, witness,
scale and solve budget. The `analyze` CLI serializes its report, with the named
example in `scenarios/analysis.json`. It does not infer one aggregate technique
from heterogeneous firms. Comparators preserve supplied population,
classification, period and convention labels and signed tick bounds; a zero
requirement makes the rate inapplicable, while a denominator interval crossing
zero makes the rate unresolved. Optional closure failure leaves valid goods
accounts available. Missing/inapplicable/inadmissible/unresolved diagnostics
carry no fabricated numeric value.

`Simulator/Solver.lean` maps to device checks of the actual nonnegative operator
against a positive weighted contraction witness. A successful witness bounds
the weighted residual by `1/(1-c)` and scales each coordinate enclosure by its
weight. A failed sufficient witness is unresolved unless a separate positive
vector test certifies nonproductivity. Fixed iteration budgets do not imply
convergence: the returned bounds must satisfy the requested tolerance.
Normalization permits zero goods-value lower bounds but requires a strictly
positive reference lower bound. `CachedAccount` checks all ten context tokens;
callers must advance the relevant token when its underlying data changes.

`DirectedRationalTicksV1` uses unsigned 64-bit interval endpoints and bounded
32-bit-word arithmetic, including fixed 128-bit product intermediates. All
rounding is outward and all gross intermediate overflows are errors. Supplied
scales convert tick bounds into physical units; there is no floating tolerance
called a certificate. Sparse dimensions are capped at 65,535 and the solver
budget at 100,000 Jacobi steps. Certificate weights are positive u32 values;
the witness check uses a fixed 32-bit fractional working scale and may report
unresolved when its enclosure cannot certify a near-boundary inequality.

`BAD_ECON_TEST_BACKEND=gpu cargo test --test analytics --test bargaining --test households -- --nocapture`
passes 14 analytical, three bargaining and four household tests. Analytical
fixtures cover the grain/research account, asymmetric four-block transposes,
Jacobi old-block behavior, the supplied residual 0.1 example and physical error
bounds, optional inadmissible closure, reference normalization including zero
goods values, the reciprocal residual scale, cache invalidation, lifetime
conventions, boundary/inventory/unit formulas, serialization and independent
fixed-u128 wide-product oracles.

## Household policy implementation evidence

`src/simulation/households.rs` and its kernels implement `needDeficit`,
`availableHours`, `nextHealth`, `nextFatigue` and `observedBlend` from
`Simulator/Policies.lean`. The batch API admits any finite service list whose
positive-need weights sum to at most 1,000. Zero needs contribute zero,
over-delivery contributes no deficit, and full-u64 needs/deliveries use bounded
96-bit products with a ten-step quotient search. Each weighted service returns
floor/ceiling bounds; the sum encloses the real weighted deficit. Health uses
the declared lower-grid deficit. Missing observations retain the previous
value, and observed forecasts return the exact floor of the convex blend on
the admitted thousandth-weight grid, including u64 endpoints.

Participation is explicit. Available time is the minimum of health/fatigue
capacity and a shared 1,440-minute clock after reserved activities. Health and
fatigue use thousandth fractions, and the fatigue denominator is the specified
eight hours, represented by 480 minutes. Capacity and exertion are bounded by
one day; rest must fit reserved time. Results clip to the admitted health and
fatigue interval. These are the finite-grid implementation of the real formulas,
not claims that rounding preserves every real equality.

The running simulation uses the same capacity, health, fatigue and observation
helpers. The optional household policy includes work and research exertion;
its single delivered bread service retains an equivalent bounded-u32 deficit
calculation. Strike decisions remove participation before matching. General
multiservice deficits and missing observation inputs are exposed by the GPU
batch API; the current scenario has one consumption service and records a
forecast observation each day. `tests/households.rs` checks the Lean deficit
example, nonparticipation, exhausted shared clocks, zero-need branches, wide
quantities, convex endpoints and invalid domains on GPU. The parent phase
fixtures additionally verify the configured household transition in the live
simulation.

## Completion verification

The final `BAD_ECON_TEST_BACKEND=gpu cargo test --all-features` run passes
116 tests on Apple M5 Pro / Metal, including all GPU and paired replay tests.
Only five explicitly optional CPU-only checks are ignored. The final
`cargo clippy --all-targets --all-features -- -D warnings`,
`cargo check --no-default-features`, `cargo fmt --all -- --check`, and
`git diff --check` pass. Cargo dependencies and the lockfile are unchanged.

Every worked example and review regression in `Simulator/Examples.lean` has
an executable fixture:

| Example group | Evidence |
| --- | --- |
| Wage then purchase, received-wage affordability, rejected first buyer, gross inputs | `tests/contracts.rs`, with device ledger ordering and atomicity in `tests/execution.rs` and `tests/gpu.rs` |
| Cross-bank deposits and reserves | `tests/finance.rs::lean_cross_bank_payments_conserve_deposits_and_reserves_separately` |
| Packet size/ticks/capacity, empty resources and zero leg; thirds and fractional settlement; non-grid and empty quotes | `tests/contracts.rs::lean_packets_claims_quotes_and_draws`, plus device payroll/bargaining fixtures |
| Floored keyed draw threshold | Contract fixture and GPU SHA-256/128-bit threshold boundary fixtures in `research/random.rs` |
| Smoothing, missing observation and weighted household deficit | `tests/households.rs::lean_weighted_need_missing_observation_and_shared_clock` |
| Proposed starts and fractional net-income/payment examples | GPU teaching fixture in `simulation/payroll.rs`; wage grids and exact bracket inverse in `tests/bargaining.rs` |
| Grain values, coupled fixed point, research attempt cost, gross requirements, plan value and hours | `tests/analytics.rs` grain/research fixtures and `scenarios/analysis.json` |
| Contraction, residual and scaled physical bound | `tests/analytics.rs::lean_residual_certificate_is_for_the_actual_operator` |
| Dated funding, cushions, household support and divisible prefix | Corresponding named fixtures in `tests/finance.rs` |
| Indivisible failed-job fallback | `tests/contracts.rs::lean_indivisible_jobs_release_capacity_after_rejection` |
| Old-block coupled update, unequal occupational fixed point and normalization | `tests/analytics.rs::lean_coupled_step_uses_old_blocks_and_normalizes_both_blocks` |
| Certificate does not transfer to transpose; optional closure may fail | Actual-operator certificate and optional-closure fixtures in `tests/analytics.rs` |

The remaining distinction is numerical, not an unimplemented institution:
the executable profiles use bounded machine integers, declared grids and
certified intervals where Lean uses unbounded integers, rationals or reals.
Nonlinear lookup approximations and domain limits are stated above. These
tests establish implementation evidence; they do not make the Rust program
a Lean-verified implementation.

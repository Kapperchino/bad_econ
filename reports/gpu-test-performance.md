# GPU test iteration

This change reduces repeated GPU initialization and test executable linking, and adds a small developer feedback command. The original revision preserved all 182 tests from its baseline; no scenario, assertion, workgroup, simulation horizon or test case was removed or ignored. Later upstream tests remain registered in the combined targets; merge validation is recorded separately below.

The historical baseline is `7ded21494010e218e54cd8503473d1fca176590f`, compared with PR revision `bd142fb1c2cae7e541bb04740f49a20ad82d89e3`. Measurements below were collected on November 2, 2026, on Windows with an RTX 3090, Vulkan, driver 616.56 and Rust 1.97.1. They predate upstream's CubeK refactor and do not measure the merged implementation. The development profile uses application optimization level 1 and dependency level 2, with the existing ICU build overrides unchanged.

## Cause and change

The integration helper previously called `Device::headless` for every fixture. Each call requested and registered a GPU device, preventing reuse of that device's compiled kernels across fixtures. The installed CubeCL WGPU runtime documents that imported setups receive unique device IDs and recommends cloning an initialized device to share its runtime.

Test helpers now retain one device per workgroup and clone the handle for each simulation. Simulation state and buffers are still constructed separately. The library's grouped/reference comparisons and the GPU workgroup matrix use the same approach. Cases that inspect device-wide allocation totals keep fresh devices, including the demographic growth-budget rollback test.

Nineteen integration files are modules of `tests/integration.rs`, sharing one executable and its device cache. The dedicated `gpu` target stays separate. This reduces integration executables from 20 to 2. The original comparison contained 19 library tests, 156 integration cases and 7 dedicated GPU cases, totaling 182. A comparison of those runtime `--list` inventories, normalizing the new module prefixes, found no missing or added test names.

## Developer commands and coverage

| Command | Purpose | Coverage boundary |
| --- | --- | --- |
| `cargo gpu-fast` | 11 checks, four test threads, two build jobs. | Ledger authorization and wide arithmetic, dependency ordering, phase ordering, transport/checkpoint continuation, rollback, births and actor storage, industrial conservation. |
| `cargo gpu-full` | All unit/integration tests, one test thread, two build jobs. | Includes the long annual crop scenario, broad workgroup/experiment matrices, reference schedulers, all research/finance cases and allocation boundaries. |
| `cargo test --locked --features viewer --test integration demography:: --jobs 2 -- --test-threads=1` | Run the subsystem affected by a change. | Replace the module filter as appropriate. |

The fast subset is an iteration aid; it does not replace the full suite before merging changes. In particular, it does not cover the complete research/finance APIs, all workgroups, or long-run population and crop behavior. Tests use GPU simulation and independent integer oracles; none invokes the CPU simulation backend.

Add new integration modules to `tests/integration.rs`: `autotests = false` makes target registration explicit. The former `--test demography` form becomes `--test integration demography::`. GPU allocation-budget tests must continue to use fresh devices.

## Original revision: validation and observed timing

**These are diagnostic timings, not a controlled speedup benchmark.** Another workload shared the GPU, its utilization varied substantially, and the candidate full run overlapped host-only build checks. The baseline also had three failures. Test target ordering and driver cache warmth were not held constant. Do not derive a reliable speedup ratio from the full-run totals.

| Run | Result | Observed elapsed time |
| --- | --- | ---: |
| Baseline, all 182 cases attempted | 179 passed; 3 capacity failures in geography cases | 1,302.69 s across test executables |
| Candidate, full suite | 182 passed; 0 failed | 350.57 s across test executables |
| Initial candidate fast subset, serial | 11 passed | 98.73 s test time; 105.05 s including Cargo/startup |
| Candidate parallel smoke, four threads | 11 passed | 17.69 s process time |
| Warm serial fast subset, three fresh processes | 11 passed in every run | 20.05, 13.71, 13.32 s |
| Warm four-thread fast subset, three fresh processes | 11 passed in every run | 9.41, 10.99, 10.39 s |
| Final `cargo gpu-fast` alias, four threads | 11 passed | 17.23 s test time; 19.12 s including Cargo/startup |

The six final concurrency runs alternated thread counts in the order 1, 4, 4, 1, 1, 4. Each launched the same 11 exact test names in a fresh process. The four-thread setting was consistently faster in those local samples, motivating the fast alias. The initial serial result shows why warm observations should not be advertised as cold-start guarantees. The proposed 30-second feedback budget was met in the warm samples, but is not established across arbitrary machines, cold shader caches or competing workloads.

The baseline geography failures reported admission budgets of roughly 51 MB and then zero for scenarios estimated at about 134 MB, after repeated device creation. The candidate completed all of those unchanged cases. Another baseline executable required a normal process outside the tool sandbox to launch; it then passed all nine cases. No operating-system security settings were changed.

The initial unrestricted parallel build exhausted Windows commit memory. A two-job retry completed. Baseline and candidate preparation took 103.47 s and 101.79 s respectively, but their invalidated artifacts differed and the tool sandbox emitted incremental-cache hard-link warnings; these are not comparable edit/rebuild benchmarks. No compilation speedup percentage is claimed.

Additional checks passed:

- `cargo fmt --all -- --check`.
- `cargo clippy --locked --all-targets --features viewer --jobs 2 -- -D warnings`.
- `cargo check --locked --no-default-features --lib --tests --jobs 2` (compile only).
- Full test-name inventory equality and validation of every name in the fast alias.
- The four-thread smoke runs exercise shared devices concurrently with the isolated growth-budget rollback case. This addresses the cache-isolation issue found in the requested GPT-6.1 Sol source review.

Final GPT-6.1 Sol review found no blocking issues after the isolation fix and reviewed the runtime inventory and validation logs. That review was read-only; the reviewer did not independently rerun the GPU checks. Other hardware/backend combinations remain unverified.

## Validation after merging CubeK upstream main

Upstream `ba6421a45665a1b90e42b4be4e6b5c471f2322a8` adds CubeK analytics, reductions and random streams, updated checkpoint semantics, and additional tests. The merge preserves those production changes and assertions. The analytics, retained-account and sensitivity modules import one shared `estimates` helper registered in `tests/integration.rs`, alongside the cached GPU support module. Allocation-budget cases retain fresh devices.

The final runtime inventory matches all **186** upstream test names after normalizing integration-module prefixes: **21 library**, **158 integration** and **7 GPU-specific** checks. Every upstream integration module remains registered, and all 11 fast-alias names resolve.

Validation on RTX 3090 / Vulkan:

- All 186 checks passed, with no failures or ignored cases. The final integration and GPU executables were rerun after consolidating the estimates helper; the unchanged, cached library executable's 21 passing results were retained.
- `cargo gpu-fast` passed all 11 cases on the final merged build.
- `cargo fmt --all -- --check` passed.
- `cargo clippy --locked --all-targets --features viewer --jobs 2 -- -D warnings` passed.
- `cargo check --locked --no-default-features --lib --tests --jobs 2` passed (compile only). Windows emitted the existing incremental-cache hard-link fallback warnings.
- GPT-6.1 Sol reviewed the conflict resolution and final shared-helper placement with no blocking source findings. The review was read-only; the reviewer did not independently execute tests.

GPU contention remains uncontrolled. This validation establishes correctness coverage; no measured speedup is claimed. The earlier timing table applies only to the original pre-CubeK revisions.

Local evidence is in `target/pr6-merge-main/`: `runtime-inventory-final.json`, `validated-summary.json`, the target logs in `full/` and `full-final/`, `fast-final.log`, `clippy-final.log` and `no-default.log`. `validated-summary.json` contains exactly the passing 186-case inventory from the unchanged library target and final integration/GPU targets.

## Validation after merging settlement and registry updates

Upstream `1c2854f023e2aa935b2f5b0281a009cf12f81007` adds settlement batches, GPU market counters, tiled audits, registry changes, and more regression coverage. This merge preserves those production changes, scenarios and assertions. The market-fault test retains all four upstream execution/trace combinations while using the cached device. The workgroup matrix obtains one cached device per workgroup and clones it for each experiment. Allocation-budget cases retain their fresh devices, and the README keeps upstream's benchmark documentation alongside the fast/full test commands.

On November 3, 2026, the runtime inventory matched all **196** upstream test names after normalizing integration-module prefixes: **30 library**, **159 integration** and **7 GPU-specific** checks. Every upstream integration module remains registered, including the new payroll specialization case, and all 11 fast-alias names resolve.

Validation on RTX 3090 / Vulkan:

- All 196 checks passed, with no failures, ignored cases, missing cases or extra cases.
- `cargo gpu-fast` passed all 11 cases on the final merged build.
- `cargo fmt --all -- --check` passed.
- `cargo clippy --locked --all-targets --features viewer --jobs 2 -- -D warnings` passed.
- `cargo check --locked --no-default-features --lib --tests --jobs 2` passed (compile only). The existing Windows incremental-cache hard-link fallback warnings remain.
- GPT-6.1 Sol reviewed the conflict resolution and retained coverage with no blocking source findings. The review was read-only; the reviewer did not independently run tests.

These checks establish correctness coverage, not a controlled speedup measurement. Local evidence is in `target/pr6-merge-main-1c2854f/`: `source-inventory.json`, `runtime-inventory.json`, `validated-summary.json`, `full/summary.json`, the full-suite target logs, `fast.log`, `clippy.log` and `no-default.log`.

## Reproduce the timing inventory

On an otherwise idle GPU, build once and then time the emitted test executables:

```sh
cargo test --locked --features viewer --lib --tests --jobs 2 --no-run --message-format=json > target/gpu-test-build.jsonl
python scripts/measure-gpu-tests.py target/gpu-test-build.jsonl target/gpu-test-timings
```

The script uses stable libtest output, runs cases serially within each executable, and saves target logs plus `summary.json`. `--target integration` restricts it to that target. Per-case durations approximate the interval between completed output lines; total executable elapsed time is measured directly. Compilation is excluded. A failed executable or incomplete parsed inventory produces a nonzero exit status. A process-launch error also fails the command rather than reporting success.

Local raw evidence for this change is in ignored `target/test-performance/`: the baseline build manifest, `baseline-summary.json`, candidate manifest and logs, `coverage-inventory.json`, the fast/concurrency logs, and lint/build logs. These are machine-local artifacts, not files bundled with this report. GPU PR profiling in GitHub Actions remains a separate deferred change.

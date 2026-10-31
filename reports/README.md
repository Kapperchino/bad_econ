# Verification evidence

The contract expansion uses CubeCL's CPU backend and WGPU on Apple M5 Pro / Metal.
The default suite checks actual GPU execution;
phase replay compares complete CPU/GPU state and ledger decisions. Independent
integer and Lean worked-example fixtures check the underlying formulas.

`contract-expansion-benchmark.json` records a short GPU smoke measurement with
the expanded state layout:

```sh
cargo run -- benchmark --sizes 10000,1000000 --warmup 1 --days 2 \
  --output /tmp/bad-econ-final-benchmark.json
```

| Households | Firms | Measured days/second | Peak reserved simulation bytes |
| ---: | ---: | ---: | ---: |
| 10,000 | 100 | 14.94 | 27,946,496 |
| 1,000,000 | 10,000 | 2.04 | 848,507,392 |

These are two measured days after one warmup day, in the optimized development
profile. The baseline starts with stocked inventories and no optional
institutions; these figures check execution and memory capacity, not sustained
throughput or the cost of a million research labs. Daily phase timings,
event counts, readback volume, allocator usage and hashes are retained in the
JSON. The final snapshot and repeated scenario JSON were omitted from this
compact report. Simulation buffers remained below the 4 GiB budget.

The institutions scenario and dedicated integration tests exercise enabled
banks, support, payroll, bargaining, research, equipment, stocks and strikes.
See [the contract comparison](../IMPLEMENTATION.md) for their source mappings
and verification commands.

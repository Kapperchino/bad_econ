# Spatial economy

A Rust simulation of households, farms, factories, trade, and research across 18 goods and services. Runs on the GPU with Burn/CubeCL and includes a Bevy viewer.

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

The viewer starts paused and uses native Bevy UI with a Victoria 3–inspired resource bar, navigation rail, economic ledger, regional map, and actor inspector. Click **Overview**, **Market**, **Industry**, **Population**, or **Finance**, then a graph tab to view employment, wealth, prices, stocks, wages, business activity, research, demography, housing, or credit. The 28 charts include unemployment, daily production and consumption, input use, deliveries, business cash profit, receipts and outlays, business types and industries, population occupations, loan balances, rent and taxes. Graph colors match the visible legend; seed comparisons overlay the saved run with faint lines.

Use **30 days**, **1 year**, **5 years**, or **All history** to change the period across charts. The viewer retains every daily observation in the current run; reset starts a new history. **Focus next** isolates a series and rescales its axis; **Show all** restores the full chart. Hover a chart to read a day's exact values and, when comparing seeds, same-day reference values and differences. Cash brackets are also shown over time. Flow charts show changes per day, not cumulative totals; data before the first observation is unavailable.

**Population → Occupations** shows farmers, wage workers, owners, unemployed people, dependents and other adults as percentages of all living people, with person counts in the readout. Categories never overlap: age-based dependents take priority, followed by the selected adult in each managing-owner household, subsistence farmers, paid workers, unemployed labor representatives, and other adults. Farm employees count as wage workers. The simulation selects one labor representative per household; these categories describe that model. Unemployment divides unemployed representatives by wage workers plus unemployed representatives and excludes the other categories. Labor and occupation charts become available after the first simulated day.

**Industry → Profits / Accounts** shows cash-basis profit before owner payouts: actual sales receipts and collected rent minus cash outlays, including wages, inputs, rent and construction purchases. It excludes loan funding and debt service, and does not apply depreciation or inventory valuation. Negative profits remain visible below zero. Individual firm inspection includes daily receipts, outlays and cash profit. These observations are included in JSON output and checkpoints. Scroll over the ledger or inspector to read longer sections.

The top bar has **Resume/Pause**, **+1 day**, and three speed buttons (0.5 days/s, 2 days/s, maximum). **Compare seed** saves the current history and runs the next seed to the same day; **Reset run** restores the launch scenario. Click **Household**, **Firm**, **Next actor**, or a firm marker on the map to inspect its accounts. Sound and flash controls are in the footer.

Keyboard shortcuts remain available: **Space** runs/pauses, **N** steps one day, **R** resets, **1/2/3** change speed, **G** cycles graphs across menus, **C** compares the next seed, **H/F/Tab** inspect actors, and **M/V** toggle sound/flashes. Run `cargo run --features viewer -- view-check --households 100` for the GPU UI check: it clicks the native controls, firm markers, and graph menus, checks that buttons fit a 1280×800 window, and writes screenshots to the system temporary directory. On macOS, keep the session unlocked for window captures; capture failures are reported separately and still make the check fail.

## Model

- Subsistence farmers grow their own food. Wage workers start employed by capitalist-owned farms, factories, or distributors and buy food.
- Generated industrial producers make 10 units in 20 labor minutes. Grain farming therefore needs one fifth of the subsistence cultivation labor per unit, and commercial milling and baking each need one fifteenth of the household labor. The industrial wage is 12 cash units per minute, preserving 240 cash units per production batch; distributors use five-minute shifts at the same rate. Explicit scenario recipes and wages remain configurable.
- In industrial economies without explicit property regions, completed grain-farm expansions consolidate nearby subsistence plots when the farm uses less cultivation labor per unit and its annual output per worker exceeds the household yield per plot. Farms absorb whole local grants covering the household harvest their added annual output can replace, ordered by distance and then actor order. Displaced households lose unharvested crop work, retain stored food and cash, and can seek wage employment. Consolidated plots remain with the commercial farm and receive the same scale bonuses as developed property. Expansion still requires demand, funding, materials, construction labor, and an owner who wants to grow.
- Firms produce goods using labor and inputs. Natural resources constrain production sites; deliveries take at least one day.
- Births, deaths, migration, business entry, and closures change the economy over time.
- Entrepreneurial interest is a persistent seeded trait: 8% of households are interested by default, including small businesses. Interested households review local business opportunities monthly, then need savings, construction materials, and operating funds to open a firm. They estimate margins from nearby prices and available inputs; they do not know other firms' order books or prospective customers' bank balances, and sales are not guaranteed. Set `"entrepreneurship": { "interested_per_mille": 80 }` to change this modeling assumption (0 disables entry, 1000 makes every household interested).
- Generated industrial households start with unequal liquid savings: about 60% have 1.75–7 days of starting food and rent expenses, 30% have 7–28 days, and 10% have 28–112 days. This is a modeling distribution, separate from initial food inventories and ownership of land or firms. `"household_liquidity": "PaycheckToPaycheck"` uses `household_cash` as the one-week reference; omitting the setting preserves uniform balances, and explicit population cash balances are used exactly.
- Generated industrial producers start with two days of finished goods so replenishment provides paid work early. `"producer_buffer_days": 2` controls these stocks separately from household food and production inputs; omitting it retains the scenario's existing `buffer_days` behavior.
- Rent goes to household-owned housing firms. A housing firm has zero workers and produces `output_ticks` housing units per developed plot each day. It buys construction services for maintenance: `input_ticks` per plot covers 30 days. Existing buildings start with 30 days of maintenance coverage; without maintenance, housing supply stops. Housing services expire daily, and households without an available unit become homeless. Subsistence households with land remain exempt from rent.
- Housing firms receive household and business rent in spendable cash. Households pay after buying bread and keep their cash when their food buffer, including deliveries, is short. Ordinary firms pay after wages and retain the existing 30-day arrears grace period. Unpaid balances carry forward. Housing firms do not charge themselves rent. Generated rental buildings group existing homes into eight units per plot when no property regions are configured, with ownership shares and no extra starting money; explicit populations can supply `Housing` firms with `capacity: 0` and a technique with `labor_minutes: 0`.
- Owners can pay themselves through the existing ownership distributions (`distribution_rate`). Housing firms first reserve maintenance funds and the cost of any construction project. Owners interested in expansion also save toward the next project when their housing is fully occupied. New housing uses paid construction services and deliveries; property regions additionally require land and the regional owner's authorization. One completed plot adds the building's configured number of housing units.
- Expansion preferences are persistent and separate from interest in starting a business. By default, 8% of small-business owners and 60% of industrial owners seek expansion. Firms with fewer than 20 worker slots, including zero-worker landlords, count as small businesses; larger employers count as industrial. Set `"expansion": { "small_business_per_mille": 80, "industrial_per_mille": 600 }` to change these modeling assumptions. Owners who prefer to maintain their business can still receive payouts; cash alone does not trigger expansion.
- Scenario files can set `"rent": { "household_daily": 1, "firm_daily": 10 }`; these are also the generated industrial defaults. Omitting rent disables collection. The viewer shows rent paid, arrears, and housing firms' units and rental income. Checkpoints use format 14 and require regeneration from older formats. Legacy inventory arrays with 3, 16, or 17 goods remain readable in scenarios.
- The optional `property` configuration partitions the map into nonoverlapping rectangular regions with fixed space and a landowner household. Housing, commercial farms, factories, and reserved construction sites share that space. Existing sites start with one plot; landless households served by configured housing firms share those buildings instead. `buildings` can override each site's footprint. Natural-good firms use farms; manufactured-good firms, including builders, use factories; housing firms use housing plots. Buildings remain on their original land after household migration or business closure.
- Landowners commission `orders` for their own household or firms using actor identities (`landowner`, `site`), `space`, `work`, and `development` (`NewBuilding` or `Extension`). New firms cannot produce until completion; extensions add the site’s existing workers per plot, or housing units per plot for rental buildings. Projects move through Requested → Building → Completed, reserving space in actor order. Unavailable space leaves requests pending; absent materials, workers, funds, or deliveries stall construction. Abandoned projects release reservations at the next opening. With property enabled, entrepreneurial entry and automatic expansion require the regional landowner and purchase 20 construction work units per added space.
- `Construction` firms hire labor and consume wood and tools using ordinary production recipes. `MechanizedConstruction` is available through mechanization research. Projects buy their output through the existing paid market and delivery system, consuming it as construction work. Subsistence households keep their plots only when their entire grant fits in the space left after buildings and reservations; otherwise they lose their plots and unharvested crop work and can seek wage employment. Allocation is deterministic, regional, and preserves stored food. Migration relinquishes subsistence plots. Run `cargo run --release -- run --scenario scenarios/property.json --days 14` for a two-region example with housing, farm, and factory projects. Run output includes regional property totals; actor inspection shows land and project status. Omitting `property` retains owner-built construction for ordinary firms; housing developments still purchase construction services.
- Transactions use exact integer accounting. Daily audits check cash and goods conservation; faults restore the last audited state.

## Test and benchmark

```sh
cargo gpu-fast
cargo gpu-full
cargo run --release -- benchmark --sizes 100000,1000000 --seed 42 --warmup 0 --days 1000 --output /tmp/benchmark.json
```

Both test aliases use the GPU. Benchmarks report daily phase timings, memory use, and final state hashes. Add integration modules to `tests/integration.rs`; automatic test discovery is disabled.

Code: [domain](src/domain.rs), [simulation](src/simulation/), [viewer](src/viewer.rs), [tests](tests/).

use super::*;
use crate::domain::railway::SETTLEMENT_PLAN_WORDS;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(in crate::simulation) enum PlanningStage {
    Prepare,
    Finish,
}

#[derive(Clone, Copy)]
#[repr(u32)]
enum PlanMode {
    Fallback,
    Ready,
}

#[derive(Clone, Copy)]
#[repr(u32)]
enum BuyerDecision {
    Rejected,
    Accepted,
}

#[derive(Clone, Copy)]
#[repr(usize)]
enum BuyerWord {
    GrossBefore = 0,
    ArrivalField = 3,
    Arrival = 4,
    Decision = 6,
    RailCandidate = 7,
    RoutePrice = 8,
    RouteStations = 10,
    RouteCarrier = 12,
    Quantity = 14,
    Eligible = 15,
    CapacityEpoch = 16,
    SelectionCache = 17,
    CapacityCache = 18,
}

#[derive(Clone, Copy)]
#[repr(usize)]
enum SellerWord {
    InitialCash = 0,
    Gross = 2,
    Stock = 4,
    Sales = 6,
    Queued = 8,
    Mode = 9,
    Freight = 10,
    Cursor = 12,
}

#[derive(CubeType, CubeLaunch)]
struct PlanningInput {
    state: Box<[u32]>,
    actors: Box<[u32]>,
    config: EconomyConfig,
    scratch: Box<[u32]>,
    control: Box<[Atomic<u32>]>,
    parameters: EconomyParameters,
    #[cube(comptime)]
    stage: PlanningStage,
}

struct PlanningSystem;
struct PlanTrades;

impl System for PlanningSystem {
    type Input = PlanningInput;
    type Passes = gpu_passes![PlanTrades];
}

pub(in crate::simulation) fn launch(
    device: &Device,
    state: &CubeTensor,
    actors: &CubeTensor,
    config: &Configuration,
    _proposals: &CubeTensor,
    scratch: &CubeTensor,
    control: &CubeTensor,
    dispatches: &CubeTensor,
    parameters: &EconomyParameters,
    stage: PlanningStage,
) {
    PlanningSystem::launch(
        device,
        markets::dispatch(dispatches, markets::QUEUE_DISPATCH),
        CubeDim::new_1d(device.workgroup),
        PlanningInputLaunch {
            state: state.clone().into_buffer_arg(),
            actors: actors.clone().into_buffer_arg(),
            config: config.launch(),
            scratch: scratch.clone().into_buffer_arg(),
            control: control.clone().into_buffer_arg(),
            parameters: parameters.launch(),
            stage,
        },
    );
}

#[cfg(test)]
pub(super) fn planned_sellers(device: &Device, config: &Configuration) -> anyhow::Result<usize> {
    let words = device.read(&config.railway.settlements)?;
    let actors = config.actors as usize;
    let plans = actors * (REJECTION_WORDS + 1 + BUYER_CACHE_WORDS) + 1;
    Ok((0..actors)
        .filter(|&actor| {
            words[actor] == Settlement::Serial as u32
                && words[plans + actor * SETTLEMENT_PLAN_WORDS + SellerWord::Mode as usize]
                    == PlanMode::Ready as u32
        })
        .count())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ProfileMode {
    Parallel,
    Planned,
    SerialFallback,
    Dependent,
    Unbatched,
    Other,
}

struct ProfileGroup {
    seller: usize,
    mode: ProfileMode,
    buyers: usize,
    accepted: usize,
    queued: usize,
    stable_pruned: usize,
    animal: usize,
    rejected_tail: usize,
}

#[derive(Default)]
struct ProfileTotals {
    groups: usize,
    buyers: usize,
    accepted: usize,
    queued: usize,
    stable_pruned: usize,
    animal: usize,
    rejected_tail: usize,
    largest_group: usize,
    largest_queue: usize,
}

pub(in crate::simulation) fn profile_plan(
    device: &Device,
    config: &Configuration,
    scratch: &CubeTensor,
    control: &CubeTensor,
    parameters: &EconomyParameters,
) -> anyhow::Result<()> {
    if std::env::var_os("BAD_ECON_PROFILE_PLAN").is_some()
        && u64::from(parameters.day) + 1 == crate::simulation::profiling_day()
        && parameters.market == MarketFlag::Household
    {
        let active = match parameters.discovery {
            Discovery::Indexed => {
                device.read_range(control, CONSTRUCTED_EVENTS_WORD, 1)?[0] as usize
            }
            #[cfg(test)]
            _ => parameters.sort_count as usize,
        };
        let words = device.read(&config.railway.settlements)?;
        let sellers = device.read_range(scratch, parameters.sort_source as usize, active)?;
        let buyers = device.read_range(
            scratch,
            (parameters.sort_source + parameters.count) as usize,
            active,
        )?;
        let plans = config.actors as usize * (REJECTION_WORDS + 1 + BUYER_CACHE_WORDS) + 1;
        let mut groups = Vec::new();
        let mut start = 0;
        for sellers in sellers.chunk_by(|left, right| left == right) {
            let seller = sellers[0] as usize;
            let seller_plan = plans + seller * SETTLEMENT_PLAN_WORDS;
            let mode = match words[seller] {
                class if class == Settlement::Parallel as u32 => ProfileMode::Parallel,
                class if class == Settlement::Serial as u32 => {
                    match words[seller_plan + SellerWord::Mode as usize] {
                        ready if ready == PlanMode::Ready as u32 => ProfileMode::Planned,
                        _ => ProfileMode::SerialFallback,
                    }
                }
                class if class == Settlement::Dependent as u32 => ProfileMode::Dependent,
                class if class == Settlement::Unbatched as u32 => ProfileMode::Unbatched,
                _ => ProfileMode::Other,
            };
            let end = start + sellers.len();
            let mut group = ProfileGroup {
                seller,
                mode,
                buyers: sellers.len(),
                accepted: 0,
                queued: 0,
                stable_pruned: 0,
                animal: 0,
                rejected_tail: 0,
            };
            if mode == ProfileMode::Planned {
                let cursor = (words[seller_plan + SellerWord::Cursor as usize] as usize).min(end);
                group.queued = words[seller_plan + SellerWord::Queued as usize] as usize;
                group.rejected_tail = end - cursor;
                for buyer in &buyers[start..cursor] {
                    let buyer = (*buyer + parameters.offset) as usize;
                    let base = plans + buyer * SETTLEMENT_PLAN_WORDS;
                    if words[base + BuyerWord::Decision as usize] == BuyerDecision::Accepted as u32
                    {
                        group.accepted += 1;
                        match words[base + BuyerWord::RouteCarrier as usize]
                            < config.railway.company_count
                        {
                            true => {
                                group.stable_pruned += usize::from(
                                    words[base + BuyerWord::RailCandidate as usize] == 0,
                                );
                            }
                            false => group.animal += 1,
                        }
                    }
                }
            }
            groups.push(group);
            start = end;
        }
        eprintln!("PLAN_PROFILE day={} active={active}", parameters.day + 1);
        for mode in [
            ProfileMode::Parallel,
            ProfileMode::Planned,
            ProfileMode::SerialFallback,
            ProfileMode::Dependent,
            ProfileMode::Unbatched,
            ProfileMode::Other,
        ] {
            let totals = groups.iter().filter(|group| group.mode == mode).fold(
                ProfileTotals::default(),
                |mut totals, group| {
                    totals.groups += 1;
                    totals.buyers += group.buyers;
                    totals.accepted += group.accepted;
                    totals.queued += group.queued;
                    totals.stable_pruned += group.stable_pruned;
                    totals.animal += group.animal;
                    totals.rejected_tail += group.rejected_tail;
                    totals.largest_group = totals.largest_group.max(group.buyers);
                    totals.largest_queue = totals.largest_queue.max(group.queued);
                    totals
                },
            );
            eprintln!(
                "PLAN_PROFILE mode={mode:?} groups={} buyers={} accepted={} queued={} stable_pruned={} animal={} rejected_tail={} largest_group={} largest_queue={}",
                totals.groups,
                totals.buyers,
                totals.accepted,
                totals.queued,
                totals.stable_pruned,
                totals.animal,
                totals.rejected_tail,
                totals.largest_group,
                totals.largest_queue,
            );
        }
        groups.sort_by_key(|group| std::cmp::Reverse(group.buyers));
        for group in groups.iter().take(12) {
            eprintln!(
                "PLAN_GROUP seller={} mode={:?} buyers={} accepted={} queued={} stable_pruned={} animal={} rejected_tail={}",
                group.seller,
                group.mode,
                group.buyers,
                group.accepted,
                group.queued,
                group.stable_pruned,
                group.animal,
                group.rejected_tail,
            );
        }
    }
    Ok(())
}

#[cube]
impl GpuPass<PlanningInput> for PlanTrades {
    type Output = ();

    fn execute(mut input: PlanningInput) {
        let job = CUBE_POS as u32;
        if job < input.control[markets::RANGE_SLOTS_WORD].load() {
            let start = input.scratch[(5 * input.parameters.count + job) as usize];
            let length = input.scratch[(6 * input.parameters.count + job) as usize];
            if start < markets::active(&input.control, &input.parameters) && length > 0 {
                let end = start + length;
                match comptime!(input.stage) {
                    PlanningStage::Prepare => prepare(
                        &mut input.state,
                        &input.actors,
                        &input.config,
                        &mut input.scratch,
                        &input.control,
                        &input.parameters,
                        start,
                        end,
                    ),
                    PlanningStage::Finish => finish(
                        &mut input.state,
                        &input.config,
                        &mut input.scratch,
                        &input.control,
                        &input.parameters,
                        start,
                        end,
                    ),
                }
            }
        }
    }
}

#[cube]
fn plan_base(config: &EconomyConfig, actor: u32) -> usize {
    config.actors as usize * (REJECTION_WORDS + 1 + BUYER_CACHE_WORDS)
        + 1
        + actor as usize * SETTLEMENT_PLAN_WORDS
}

#[cube]
fn read_word(config: &EconomyConfig, actor: u32, #[comptime] word: usize) -> u32 {
    config.railway.settlements[plan_base(config, actor) + word].load()
}

#[cube]
fn write_word(config: &EconomyConfig, actor: u32, #[comptime] word: usize, value: u32) {
    config.railway.settlements[plan_base(config, actor) + word].store(value);
}

#[cube]
fn read_value(config: &EconomyConfig, actor: u32, #[comptime] word: usize) -> Ticks {
    ticks(
        config.railway.settlements[plan_base(config, actor) + word].load(),
        config.railway.settlements[plan_base(config, actor) + word + 1].load(),
    )
}

#[cube]
fn write_value(config: &EconomyConfig, actor: u32, #[comptime] word: usize, value: Ticks) {
    let base = plan_base(config, actor) + word;
    config.railway.settlements[base].store(value.extract(LOW_WORD));
    config.railway.settlements[base + 1].store(value.extract(HIGH_WORD));
}

#[cube]
pub(super) fn cache_candidate(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    parameters: &EconomyParameters,
    buyer: u32,
    seller: u32,
    quantity: Ticks,
    path: &FreightRoute,
) -> u32 {
    let good = comptime!(parameters.market.good().expect("goods market") as u32);
    let mut arrival_overflow = 0u32;
    let mut rail_candidate = u32::cast_from(path.carrier < config.railway.company_count);
    write_value(
        config,
        buyer,
        comptime!(BuyerWord::RoutePrice as usize),
        ticks(path.days, path.fee),
    );
    write_value(
        config,
        buyer,
        comptime!(BuyerWord::RouteStations as usize),
        ticks(path.origin, path.destination),
    );
    write_value(
        config,
        buyer,
        comptime!(BuyerWord::RouteCarrier as usize),
        ticks(path.carrier, path.foreign_distance),
    );
    let base =
        config.actors as usize * (REJECTION_WORDS + 1) + 1 + buyer as usize * BUYER_CACHE_WORDS;
    let mut status_word = Array::<u32>::new(1usize);
    status_word[0usize] = config.railway.settlements[base + 15].load();
    let status = KernelEventStatus::load(&mut status_word[..], 0);
    write_word(
        config,
        buyer,
        comptime!(BuyerWord::Quantity as usize),
        quantity.extract(LOW_WORD),
    );
    write_word(
        config,
        buyer,
        comptime!(BuyerWord::Eligible as usize),
        u32::cast_from(status.contains(comptime!(EventStatus::Accepted))),
    );
    if path.carrier < config.railway.company_count {
        let animal = animal_route(actors, config, buyer, seller);
        let access = ceildiv(
            config.railway.access[seller as usize * 2 + 1]
                + config.railway.access[buyer as usize * 2 + 1],
            config.shipment_speed,
        );
        let limit = min(max(animal.days, access + 1) - access - 1, config.ring - 1);
        let quote = (path.origin * config.railway.nodes + path.destination) * config.ring + limit;
        let capacity = (2 * config.railway.region_count + 2) as usize
            + ((path.carrier * config.railway.nodes + path.origin) * config.railway.nodes
                + path.destination) as usize
                * CAPACITY_WORDS;
        let selection = selection_cache(config, quote);
        let prior_quantity = ticks(
            config.railway.requests[selection + 1].load(),
            config.railway.requests[selection + 2].load(),
        );
        let exhausted = config.railway.requests[selection].load() > 0
            && config.railway.requests[selection + 3].load() >= config.railway.company_count
            && !lt(quantity, prior_quantity);
        rail_candidate = u32::cast_from(
            !exhausted || !capacity_stable_demand_recorded(config, capacity, quantity),
        );
        write_word(
            config,
            buyer,
            comptime!(BuyerWord::SelectionCache as usize),
            selection as u32,
        );
        write_word(
            config,
            buyer,
            comptime!(BuyerWord::CapacityCache as usize),
            capacity as u32,
        );
    }
    if rail_candidate > 0 {
        let limit = u32::MAX - quantity.extract(LOW_WORD);
        for slot in 0..config.ring {
            let field = BASE_FIELDS as u32 + (good - 1) * config.ring + slot;
            let coordinate = at(config, field, buyer) as usize * TICK_WORDS;
            if state[coordinate + HIGH_WORD] == u32::MAX {
                arrival_overflow |= u32::cast_from(state[coordinate + LOW_WORD] > limit);
            }
        }
    }
    write_word(
        config,
        buyer,
        comptime!(BuyerWord::RailCandidate as usize),
        rail_candidate,
    );
    write_word(
        config,
        buyer,
        comptime!(BuyerWord::CapacityEpoch as usize),
        settlement::capacity_epoch(actors, config, buyer, seller, quantity, path),
    );
    arrival_overflow | u32::cast_from(status.contains(comptime!(EventStatus::Faulted)))
}

#[cube]
fn prepare(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    scratch: &mut [u32],
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
    start: u32,
    end: u32,
) {
    let seller = scratch[(parameters.sort_source + start) as usize];
    if UNIT_POS == 0 {
        write_word(
            config,
            seller,
            comptime!(SellerWord::Mode as usize),
            comptime!(PlanMode::Fallback as u32),
        );
    }
    let mode = config.railway.settlements[seller as usize].load();
    let reference = config.railway.settlements[config.actors as usize].load();
    if mode == comptime!(Settlement::Serial as u32)
        && reference != comptime!(Settlement::Serial as u32)
        && control[markets::LATCH_WORD].load() == 0
    {
        if UNIT_POS == 0 {
            let balances = GoodsSeller::new(state, actors, config, parameters, seller);
            let available = positive(balances.stock, balances.reserved());
            let quantity = plus(available, ticks(u32::MAX, 0));
            let gross = times(quantity.value, balances.quote);
            let cash = plus(balances.cash, gross.value);
            let sales = plus(balances.sales, quantity.value);
            let reserve = plus(balances.reserved(), ticks(u32::MAX, 0));
            let faults = quantity
                .faults
                .union(gross.faults)
                .union(cash.faults)
                .union(sales.faults)
                .union(reserve.faults);
            if faults.is_empty() {
                plan_group(config, scratch, parameters, balances, seller, start, end);
            }
        }
    }
}

#[cube]
fn plan_group(
    config: &EconomyConfig,
    scratch: &mut [u32],
    parameters: &EconomyParameters,
    mut balances: GoodsSeller,
    seller: u32,
    start: u32,
    end: u32,
) {
    let initial_cash = balances.cash;
    let mut gross = zero();
    let mut queued = 0u32;
    let queue = 2 * parameters.count - parameters.sort_source + start;
    let mut index = start;
    while index < end && lt(balances.reserved(), balances.stock) {
        let buyer = scratch[(parameters.sort_source + parameters.count + index) as usize]
            + parameters.offset;
        let quantity = ticks(
            read_word(config, buyer, comptime!(BuyerWord::Quantity as usize)),
            0,
        );
        let required = plus(balances.reserved(), quantity).value;
        let accepted = read_word(config, buyer, comptime!(BuyerWord::Eligible as usize)) > 0
            && !lt(balances.stock, required);
        write_word(
            config,
            buyer,
            comptime!(BuyerWord::Decision as usize),
            select(
                accepted,
                comptime!(BuyerDecision::Accepted as u32),
                comptime!(BuyerDecision::Rejected as u32),
            ),
        );
        if accepted {
            let base = config.actors as usize * (REJECTION_WORDS + 1)
                + 1
                + buyer as usize * BUYER_CACHE_WORDS;
            let arrival_field = config.railway.settlements[base + 14].load();
            let arrival = ticks(
                config.railway.settlements[base + 6].load(),
                config.railway.settlements[base + 7].load(),
            );
            write_value(
                config,
                buyer,
                comptime!(BuyerWord::GrossBefore as usize),
                gross,
            );
            write_word(
                config,
                buyer,
                comptime!(BuyerWord::ArrivalField as usize),
                arrival_field,
            );
            write_value(
                config,
                buyer,
                comptime!(BuyerWord::Arrival as usize),
                arrival,
            );
            gross = plus(gross, times32(quantity.extract(LOW_WORD), balances.quote)).value;
            balances.stock = minus(balances.stock, quantity);
            balances.sales = plus(balances.sales, quantity).value;
            if read_word(config, buyer, comptime!(BuyerWord::RailCandidate as usize)) > 0 {
                scratch[(queue + queued) as usize] = buyer;
                queued += 1;
            }
        }
        index += 1;
    }
    write_value(
        config,
        seller,
        comptime!(SellerWord::InitialCash as usize),
        initial_cash,
    );
    write_value(config, seller, comptime!(SellerWord::Gross as usize), gross);
    write_value(
        config,
        seller,
        comptime!(SellerWord::Stock as usize),
        balances.stock,
    );
    write_value(
        config,
        seller,
        comptime!(SellerWord::Sales as usize),
        balances.sales,
    );
    write_value(
        config,
        seller,
        comptime!(SellerWord::Freight as usize),
        zero(),
    );
    write_word(
        config,
        seller,
        comptime!(SellerWord::Queued as usize),
        queued,
    );
    write_word(
        config,
        seller,
        comptime!(SellerWord::Cursor as usize),
        index,
    );
    write_word(
        config,
        seller,
        comptime!(SellerWord::Mode as usize),
        comptime!(PlanMode::Ready as u32),
    );
}

#[cube]
pub(super) fn prepared(config: &EconomyConfig, seller: u32) -> bool {
    read_word(config, seller, comptime!(SellerWord::Mode as usize))
        == comptime!(PlanMode::Ready as u32)
}

const PREFETCH_LANES: u32 = 64;
const PREFETCH_WORDS: usize = 13;
pub(super) const SHARED_WORDS: usize = PREFETCH_LANES as usize * PREFETCH_WORDS + 1;

#[derive(Clone, Copy)]
#[repr(usize)]
enum PrefetchWord {
    Buyer,
    Quantity,
    Epoch,
    GrossLow,
    GrossHigh,
    Days,
    Fee,
    Origin,
    Destination,
    Carrier,
    ForeignDistance,
    SelectionCache,
    CapacityCache,
}

#[derive(CubeType)]
struct RailLedger {
    initial_cash: Ticks,
    freight: Ticks,
    epoch: u32,
}

#[cube]
fn prefetch(config: &EconomyConfig, shared: &mut [u32], buyer: u32, lane: u32) -> u32 {
    let base = lane as usize * PREFETCH_WORDS;
    let quantity = read_word(config, buyer, comptime!(BuyerWord::Quantity as usize));
    let gross = read_value(config, buyer, comptime!(BuyerWord::GrossBefore as usize));
    let price = read_value(config, buyer, comptime!(BuyerWord::RoutePrice as usize));
    let stations = read_value(config, buyer, comptime!(BuyerWord::RouteStations as usize));
    let carrier = read_value(config, buyer, comptime!(BuyerWord::RouteCarrier as usize));
    let selection = read_word(config, buyer, comptime!(BuyerWord::SelectionCache as usize));
    let capacity = read_word(config, buyer, comptime!(BuyerWord::CapacityCache as usize));
    let epoch = config.railway.requests[(2 * config.railway.region_count + 1) as usize].load();
    let cached = read_word(config, buyer, comptime!(BuyerWord::CapacityEpoch as usize));
    let mut proof = select(cached == epoch, cached, 0);
    if proof == 0 {
        let selection = selection as usize;
        let capacity = capacity as usize;
        let prior_quantity = ticks(
            config.railway.requests[selection + 1].load(),
            config.railway.requests[selection + 2].load(),
        );
        let exhausted = config.railway.requests[selection].load() > 0
            && config.railway.requests[selection + 3].load() >= config.railway.company_count
            && !lt(ticks(quantity, 0), prior_quantity);
        if exhausted && capacity_demand_recorded(config, capacity, ticks(quantity, 0), epoch) {
            proof = epoch;
        }
    }
    shared[base + comptime!(PrefetchWord::Buyer as usize)] = buyer;
    shared[base + comptime!(PrefetchWord::Quantity as usize)] = quantity;
    shared[base + comptime!(PrefetchWord::Epoch as usize)] = proof;
    shared[base + comptime!(PrefetchWord::GrossLow as usize)] = gross.extract(LOW_WORD);
    shared[base + comptime!(PrefetchWord::GrossHigh as usize)] = gross.extract(HIGH_WORD);
    shared[base + comptime!(PrefetchWord::Days as usize)] = price.extract(LOW_WORD);
    shared[base + comptime!(PrefetchWord::Fee as usize)] = price.extract(HIGH_WORD);
    shared[base + comptime!(PrefetchWord::Origin as usize)] = stations.extract(LOW_WORD);
    shared[base + comptime!(PrefetchWord::Destination as usize)] = stations.extract(HIGH_WORD);
    shared[base + comptime!(PrefetchWord::Carrier as usize)] = carrier.extract(LOW_WORD);
    shared[base + comptime!(PrefetchWord::ForeignDistance as usize)] = carrier.extract(HIGH_WORD);
    shared[base + comptime!(PrefetchWord::SelectionCache as usize)] = selection;
    shared[base + comptime!(PrefetchWord::CapacityCache as usize)] = capacity;
    proof
}

#[cube]
fn settle_prefetched(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
    shared: &[u32],
    ledger: &mut RailLedger,
    seller: u32,
    lane: u32,
    requests: &mut RequestLists,
) {
    let base = lane as usize * PREFETCH_WORDS;
    let cached = shared[base + comptime!(PrefetchWord::Epoch as usize)];
    if cached == 0 || cached != ledger.epoch {
        let buyer = shared[base + comptime!(PrefetchWord::Buyer as usize)];
        let quantity = ticks(shared[base + comptime!(PrefetchWord::Quantity as usize)], 0);
        let selection = shared[base + comptime!(PrefetchWord::SelectionCache as usize)] as usize;
        let prior_quantity = ticks(
            config.railway.requests[selection + 1].load(),
            config.railway.requests[selection + 2].load(),
        );
        let exhausted = config.railway.requests[selection].load() > 0
            && config.railway.requests[selection + 3].load() >= config.railway.company_count
            && !lt(quantity, prior_quantity);
        let mut required = 1u32;
        if exhausted {
            let capacity = shared[base + comptime!(PrefetchWord::CapacityCache as usize)] as usize;
            required = u32::cast_from(!capacity_demand_recorded(
                config,
                capacity,
                quantity,
                ledger.epoch,
            ));
        }
        if required > 0 {
            let preferred = FreightRoute {
                days: shared[base + comptime!(PrefetchWord::Days as usize)],
                fee: shared[base + comptime!(PrefetchWord::Fee as usize)],
                origin: shared[base + comptime!(PrefetchWord::Origin as usize)],
                destination: shared[base + comptime!(PrefetchWord::Destination as usize)],
                carrier: shared[base + comptime!(PrefetchWord::Carrier as usize)],
                foreign_distance: shared[base + comptime!(PrefetchWord::ForeignDistance as usize)],
            };
            record_route_demand_with_requests(
                state, config, control, &preferred, quantity, requests,
            );
            if !exhausted {
                let gross_before = ticks(
                    shared[base + comptime!(PrefetchWord::GrossLow as usize)],
                    shared[base + comptime!(PrefetchWord::GrossHigh as usize)],
                );
                let cash_before = minus(
                    plus(ledger.initial_cash, gross_before).value,
                    ledger.freight,
                );
                set(
                    state,
                    config,
                    comptime!(Field::Cash as u32),
                    seller,
                    cash_before,
                );
                let selected = available_route_with_requests(
                    state, actors, config, control, buyer, seller, quantity, &preferred, requests,
                );
                if selected.carrier < config.railway.company_count {
                    record_route_demand_with_requests(
                        state, config, control, &selected, quantity, requests,
                    );
                    prepare_request_with_lists(config, &selected, requests);
                    let fee = multiply(control, quantity, selected.fee);
                    ledger.freight = add_checked(control, ledger.freight, fee);
                    let good = comptime!(parameters.market.good().expect("goods market") as u32);
                    let field = BASE_FIELDS as u32
                        + (good - 1) * config.ring
                        + (parameters.day + selected.days) % config.ring;
                    let arrival =
                        add_checked(control, value(state, config, field, buyer), quantity);
                    write_word(
                        config,
                        buyer,
                        comptime!(BuyerWord::ArrivalField as usize),
                        field,
                    );
                    write_value(
                        config,
                        buyer,
                        comptime!(BuyerWord::Arrival as usize),
                        arrival,
                    );
                    purchase_with_requests(
                        state, config, control, seller, &selected, quantity, requests,
                    );
                    ledger.epoch = config.railway.requests
                        [(2 * config.railway.region_count + 1) as usize]
                        .load();
                }
            }
        }
    }
}

#[cube]
pub(super) fn execute_plan(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    _proposals: &mut [u32],
    scratch: &mut [u32],
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
    start: u32,
    _end: u32,
    requests: &mut RequestLists,
) {
    let seller = scratch[(parameters.sort_source + start) as usize];
    let queued = read_word(config, seller, comptime!(SellerWord::Queued as usize));
    let mut ledger = RailLedger {
        initial_cash: read_value(config, seller, comptime!(SellerWord::InitialCash as usize)),
        freight: zero(),
        epoch: config.railway.requests[(2 * config.railway.region_count + 1) as usize].load(),
    };
    let width = min(CUBE_DIM, PREFETCH_LANES);
    let queue = 2 * parameters.count - parameters.sort_source + start;
    let mut shared = Shared::<[u32]>::new_slice(PREFETCH_LANES as usize * PREFETCH_WORDS);
    let required = Shared::<[Atomic<u32>]>::new_slice(1usize);
    if UNIT_POS == 0 {
        required[0usize].store(0);
    }
    sync_cube();
    let mut cursor = 0u32;
    while cursor < queued {
        let slot = cursor + UNIT_POS;
        if UNIT_POS < width && slot < queued {
            let proof = prefetch(
                config,
                &mut shared[..],
                scratch[(queue + slot) as usize],
                UNIT_POS,
            );
            if proof == 0 {
                required[0usize].fetch_or(1);
            }
        }
        sync_cube();
        if UNIT_POS == 0 {
            if required[0usize].load() > 0 {
                for lane in 0..min(width, queued - cursor) {
                    settle_prefetched(
                        state,
                        actors,
                        config,
                        control,
                        parameters,
                        &shared[..],
                        &mut ledger,
                        seller,
                        lane,
                        requests,
                    );
                }
            }
            required[0usize].store(0);
        }
        sync_cube();
        cursor += width;
    }
    if UNIT_POS == 0 {
        write_value(
            config,
            seller,
            comptime!(SellerWord::Freight as usize),
            ledger.freight,
        );
        let gross = read_value(config, seller, comptime!(SellerWord::Gross as usize));
        set(
            state,
            config,
            comptime!(Field::Cash as u32),
            seller,
            minus(plus(ledger.initial_cash, gross).value, ledger.freight),
        );
    }
}

#[cube]
fn finish(
    state: &mut [u32],
    config: &EconomyConfig,
    scratch: &mut [u32],
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
    start: u32,
    end: u32,
) {
    let seller = scratch[(parameters.sort_source + start) as usize];
    let reference = config.railway.settlements[config.actors as usize].load();
    if prepared(config, seller) && reference != comptime!(Settlement::Serial as u32) {
        let cursor = read_word(config, seller, comptime!(SellerWord::Cursor as usize));
        let mut index = start + UNIT_POS;
        while index < end {
            let id = scratch[(parameters.sort_source + parameters.count + index) as usize];
            let buyer = id + parameters.offset;
            let mut decision = u32::new(comptime!(BuyerDecision::Rejected as u32 as i64));
            if index < cursor {
                decision = read_word(config, buyer, comptime!(BuyerWord::Decision as usize));
            }
            let mut status = KernelEventStatus::new(comptime!(EventStatus::Rejected));
            if decision == comptime!(BuyerDecision::Accepted as u32) {
                let prepared = settlement::GoodsBuyer::cached_unchecked(config, buyer);
                let mut posting = prepared.posting();
                posting.arrival_field =
                    read_word(config, buyer, comptime!(BuyerWord::ArrivalField as usize));
                posting.arrival = read_value(config, buyer, comptime!(BuyerWord::Arrival as usize));
                posting.apply_buyer(state, config, parameters, buyer, seller);
                status.set(comptime!(EventStatus::Accepted));
            }
            status.store(scratch, 4 * parameters.count + id);
            index += CUBE_DIM;
        }
        if UNIT_POS == 0 {
            let good = comptime!(parameters.market.good().expect("goods market") as u32);
            set(
                state,
                config,
                good,
                seller,
                read_value(config, seller, comptime!(SellerWord::Stock as usize)),
            );
            set(
                state,
                config,
                comptime!(Field::Sales as u32),
                seller,
                read_value(config, seller, comptime!(SellerWord::Sales as usize)),
            );
            control[COMPLETED_WORD].fetch_add(end - start);
        }
    }
}

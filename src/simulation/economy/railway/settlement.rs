use super::*;

const BUYER_FIELDS: u32 = 12;
const SETTLEMENT_LANES: u32 = 64;
pub(super) const SHARED_WORDS: usize = SETTLEMENT_LANES as usize * 2 * BUYER_FIELDS as usize + 1;

#[derive(Clone, Copy)]
#[repr(u32)]
enum BuyerField {
    Quantity,
    Cost,
    Cash,
    Arrival,
    TransitGood,
    Transit,
    Purchased,
    Routing,
    RoutePrice,
    RouteStations,
    RouteCarrier,
    CapacityEpoch,
}

#[derive(CubeType)]
pub(super) struct GoodsBuyer {
    pub quantity: Ticks,
    pub cost: Ticks,
    pub cash: Ticks,
    pub arrival: Ticks,
    pub transit_good: Ticks,
    pub transit: Ticks,
    pub purchased: Ticks,
    pub arrival_field: u32,
    pub status: KernelEventStatus,
}

#[derive(CubeType)]
struct BuyerPreflight {
    buyer: GoodsBuyer,
    faults: KernelFaults,
}

#[cube]
impl GoodsBuyer {
    pub fn new(
        state: &mut [u32],
        actors: &[u32],
        config: &EconomyConfig,
        control: &[Atomic<u32>],
        parameters: &EconomyParameters,
        buyer: u32,
        seller: u32,
        quantity: Ticks,
        quote: u32,
    ) -> Self {
        let prepared = Self::preflight(
            state, actors, config, parameters, buyer, seller, quantity, quote,
        );
        prepared.faults.record(control);
        prepared.buyer
    }

    fn preflight(
        state: &mut [u32],
        actors: &[u32],
        config: &EconomyConfig,
        parameters: &EconomyParameters,
        buyer: u32,
        seller: u32,
        quantity: Ticks,
        quote: u32,
    ) -> BuyerPreflight {
        let good = comptime!(parameters.market.good().expect("goods market") as u32);
        let cost = times(quantity, quote);
        let cash = value(state, config, comptime!(Field::Cash as u32), buyer);
        let path = animal_route(actors, config, buyer, seller);
        let arrival_field = BASE_FIELDS as u32
            + (good - 1) * config.ring
            + (parameters.day + path.days) % config.ring;
        let arrival = plus(value(state, config, arrival_field, buyer), quantity);
        let transit_good = plus(
            value(state, config, TRANSIT_FIELDS as u32 + good - 1, buyer),
            quantity,
        );
        let transit = plus(
            value(state, config, comptime!(Field::Transit as u32), buyer),
            quantity,
        );
        let purchased = plus(
            value(state, config, comptime!(Field::Purchased as u32), buyer),
            quantity,
        );
        let faults = arrival
            .faults
            .union(transit_good.faults)
            .union(transit.faults)
            .union(purchased.faults);
        let mut status = KernelEventStatus::new(comptime!(EventStatus::Rejected));
        if !lt(cash, cost.value) {
            status.set(comptime!(EventStatus::Accepted));
        }
        if !faults.is_empty() {
            status.set(comptime!(EventStatus::Faulted));
        }
        BuyerPreflight {
            buyer: GoodsBuyer {
                quantity,
                cost: cost.value,
                cash,
                arrival: arrival.value,
                transit_good: transit_good.value,
                transit: transit.value,
                purchased: purchased.value,
                arrival_field,
                status,
            },
            faults: faults.union(cost.faults),
        }
    }

    fn write(&self, shared: &mut [u32], lane: u32) {
        let base = lane * BUYER_FIELDS;
        store(
            shared,
            base + comptime!(BuyerField::Quantity as u32),
            self.quantity,
        );
        store(shared, base + comptime!(BuyerField::Cost as u32), self.cost);
        store(shared, base + comptime!(BuyerField::Cash as u32), self.cash);
        store(
            shared,
            base + comptime!(BuyerField::Arrival as u32),
            self.arrival,
        );
        store(
            shared,
            base + comptime!(BuyerField::TransitGood as u32),
            self.transit_good,
        );
        store(
            shared,
            base + comptime!(BuyerField::Transit as u32),
            self.transit,
        );
        store(
            shared,
            base + comptime!(BuyerField::Purchased as u32),
            self.purchased,
        );
        store(
            shared,
            base + comptime!(BuyerField::Routing as u32),
            ticks(self.arrival_field, 0),
        );
        self.status.store(
            shared,
            2 * (base + comptime!(BuyerField::Routing as u32)) + 1,
        );
    }

    fn read(shared: &mut [u32], lane: u32) -> Self {
        let base = lane * BUYER_FIELDS;
        GoodsBuyer {
            quantity: load(shared, base + comptime!(BuyerField::Quantity as u32)),
            cost: load(shared, base + comptime!(BuyerField::Cost as u32)),
            cash: load(shared, base + comptime!(BuyerField::Cash as u32)),
            arrival: load(shared, base + comptime!(BuyerField::Arrival as u32)),
            transit_good: load(shared, base + comptime!(BuyerField::TransitGood as u32)),
            transit: load(shared, base + comptime!(BuyerField::Transit as u32)),
            purchased: load(shared, base + comptime!(BuyerField::Purchased as u32)),
            arrival_field: shared[2 * (base + comptime!(BuyerField::Routing as u32)) as usize],
            status: KernelEventStatus::load(
                shared,
                2 * (base + comptime!(BuyerField::Routing as u32)) + 1,
            ),
        }
    }

    fn cache(&self, config: &EconomyConfig, buyer: u32) {
        let mut words = Array::<u32>::new(BUYER_CACHE_WORDS);
        self.write(&mut words[..], 0);
        let base =
            config.actors as usize * (REJECTION_WORDS + 1) + 1 + buyer as usize * BUYER_CACHE_WORDS;
        #[unroll]
        for word in 0..BUYER_CACHE_WORDS {
            config.railway.settlements[base + word].store(words[word]);
        }
    }

    fn cached(config: &EconomyConfig, control: &[Atomic<u32>], buyer: u32) -> Self {
        let prepared = Self::cached_unchecked(config, buyer);
        if prepared.status.contains(comptime!(EventStatus::Faulted)) {
            KernelFaults::new(comptime!(FaultFlags::ARITHMETIC_OVERFLOW)).record(control);
        }
        prepared
    }

    pub(super) fn cached_unchecked(config: &EconomyConfig, buyer: u32) -> Self {
        let mut words = Array::<u32>::new(BUYER_CACHE_WORDS);
        let base =
            config.actors as usize * (REJECTION_WORDS + 1) + 1 + buyer as usize * BUYER_CACHE_WORDS;
        #[unroll]
        for word in 0..BUYER_CACHE_WORDS {
            words[word] = config.railway.settlements[base + word].load();
        }
        Self::read(&mut words[..], 0)
    }

    pub(super) fn posting(self) -> GoodsSettlement {
        GoodsSettlement {
            buyer_cash: minus(self.cash, self.cost),
            seller_cash: zero(),
            stock: zero(),
            arrival: self.arrival,
            transit_good: self.transit_good,
            transit: self.transit,
            sales: zero(),
            purchased: self.purchased,
            arrival_field: self.arrival_field,
            status: self.status,
        }
    }
}

#[cube]
pub(super) fn prepare_buyer(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    parameters: &EconomyParameters,
    buyer: u32,
    seller: u32,
    quantity: Ticks,
) {
    let quote = number(state, config, comptime!(Field::Price as u32), seller);
    let prepared = GoodsBuyer::preflight(
        state, actors, config, parameters, buyer, seller, quantity, quote,
    );
    prepared.buyer.cache(config, buyer);
}

#[cube]
fn prepare_shared(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
    shared: &mut [u32],
    buyer: u32,
    seller: u32,
    quantity: Ticks,
    quote: u32,
    lane: u32,
) {
    let isolated =
        config.railway.settlements[buyer as usize].load() == comptime!(Settlement::Absent as u32);
    if isolated {
        let prepared = GoodsBuyer::cached(config, control, buyer);
        prepared.write(shared, lane);
    } else {
        let prepared = GoodsBuyer::new(
            state, actors, config, control, parameters, buyer, seller, quantity, quote,
        );
        prepared.write(shared, lane);
    }
}

#[cube]
fn prepare_route(
    shared: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    buyer: u32,
    seller: u32,
    lane: u32,
) {
    let path = route(actors, config, buyer, seller);
    let base = lane * BUYER_FIELDS;
    store(
        shared,
        base + comptime!(BuyerField::RoutePrice as u32),
        ticks(path.days, path.fee),
    );
    store(
        shared,
        base + comptime!(BuyerField::RouteStations as u32),
        ticks(path.origin, path.destination),
    );
    store(
        shared,
        base + comptime!(BuyerField::RouteCarrier as u32),
        ticks(path.carrier, path.foreign_distance),
    );
    let quantity = load(shared, base + comptime!(BuyerField::Quantity as u32));
    let proven = capacity_epoch(actors, config, buyer, seller, quantity, &path);
    store(
        shared,
        base + comptime!(BuyerField::CapacityEpoch as u32),
        ticks(proven, 0),
    );
}

#[cube]
pub(super) fn capacity_epoch(
    actors: &[u32],
    config: &EconomyConfig,
    buyer: u32,
    seller: u32,
    quantity: Ticks,
    path: &FreightRoute,
) -> u32 {
    let epoch = config.railway.requests[(2 * config.railway.region_count + 1) as usize].load();
    let mut proven = 0u32;
    if path.carrier >= config.railway.company_count {
        proven = epoch;
    } else {
        let animal = animal_route(actors, config, buyer, seller);
        let access = ceildiv(
            config.railway.access[seller as usize * 2 + 1]
                + config.railway.access[buyer as usize * 2 + 1],
            config.shipment_speed,
        );
        let limit = min(max(animal.days, access + 1) - access - 1, config.ring - 1);
        let quote = (path.origin * config.railway.nodes + path.destination) * config.ring + limit;
        let selection = selection_cache(config, quote);
        let prior_quantity = ticks(
            config.railway.requests[selection + 1].load(),
            config.railway.requests[selection + 2].load(),
        );
        let exhausted = config.railway.requests[selection].load() > 0
            && config.railway.requests[selection + 3].load() >= config.railway.company_count
            && !lt(quantity, prior_quantity);
        let capacity = (2 * config.railway.region_count + 2) as usize
            + ((path.carrier * config.railway.nodes + path.origin) * config.railway.nodes
                + path.destination) as usize
                * CAPACITY_WORDS;
        if exhausted && capacity_demand_recorded(config, capacity, quantity, epoch) {
            proven = epoch;
        }
    }
    proven
}

#[cube]
fn preferred_route(shared: &mut [u32], lane: u32) -> FreightRoute {
    let base = lane * BUYER_FIELDS;
    let price = load(shared, base + comptime!(BuyerField::RoutePrice as u32));
    let stations = load(shared, base + comptime!(BuyerField::RouteStations as u32));
    let carrier = load(shared, base + comptime!(BuyerField::RouteCarrier as u32));
    FreightRoute {
        days: price.extract(LOW_WORD),
        fee: price.extract(HIGH_WORD),
        origin: stations.extract(LOW_WORD),
        destination: stations.extract(HIGH_WORD),
        carrier: carrier.extract(LOW_WORD),
        foreign_distance: carrier.extract(HIGH_WORD),
    }
}

#[cube]
fn decide(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
    buyer: u32,
    seller: u32,
    prepared: &GoodsBuyer,
    preferred: &FreightRoute,
    capacity_epoch: u32,
    balances: &mut GoodsSeller,
) -> GoodsSettlement {
    let settlement = GoodsSettlement::from_buyer(control, prepared, &*balances);
    let settlement = allocate(
        state,
        actors,
        config,
        control,
        parameters,
        buyer,
        seller,
        prepared.quantity,
        settlement,
        preferred,
        capacity_epoch,
    );
    settlement.apply_seller(state, config, parameters, seller);
    balances.apply(&settlement);
    settlement
}

#[cube]
fn allocate(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
    buyer: u32,
    seller: u32,
    quantity: Ticks,
    mut settlement: GoodsSettlement,
    preferred: &FreightRoute,
    capacity_epoch: u32,
) -> GoodsSettlement {
    let current = config.railway.requests[(2 * config.railway.region_count + 1) as usize].load();
    if capacity_epoch == 0 || capacity_epoch != current {
        let mut selected = animal_route(actors, config, buyer, seller);
        if settlement.status.contains(comptime!(EventStatus::Accepted))
            && preferred.carrier < config.railway.company_count
        {
            record_route_demand(state, config, control, preferred, quantity);
            let path = available_route(
                state, actors, config, control, buyer, seller, quantity, preferred,
            );
            selected.replace(&path);
        }
        if selected.carrier < config.railway.company_count {
            record_route_demand(state, config, control, &selected, quantity);
            prepare_request(config, &selected);
            settlement.reroute(
                state, config, control, parameters, buyer, quantity, &selected,
            );
        }
        if selected.carrier < config.railway.company_count
            && settlement.status.contains(comptime!(EventStatus::Accepted))
        {
            purchase(state, config, control, seller, &selected, quantity);
        }
    }
    settlement
}

#[cube]
fn ordered_group(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    proposals: &mut [u32],
    scratch: &mut [u32],
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
    shared: &mut [u32],
    start: u32,
    end: u32,
) {
    let count = parameters.count;
    let seller = scratch[(parameters.sort_source + start) as usize];
    let width = min(CUBE_DIM, SETTLEMENT_LANES);
    let mut next = Shared::<u32>::new();
    let mut balances = GoodsSeller::new(state, actors, config, parameters, seller);
    let mut cursor = start;
    while cursor < end {
        if UNIT_POS == 0 {
            let reserve = balances.reserved();
            *next = cursor;
            if !lt(reserve, balances.stock) && reserve.extract(HIGH_WORD) == 0 {
                reject_remainder(state, actors, config, parameters, seller, cursor);
                *next = end;
            }
        }
        sync_cube();
        cursor = *next;
        if cursor < end {
            let slot = cursor + UNIT_POS;
            if UNIT_POS < width && slot < end {
                let buyer =
                    scratch[(parameters.sort_source + count + slot) as usize] + parameters.offset;
                let quantity = ticks(
                    proposals
                        [buyer as usize * ProposalLayout::WORDS + ProposalLayout::QUANTITY_LOW],
                    proposals
                        [buyer as usize * ProposalLayout::WORDS + ProposalLayout::QUANTITY_HIGH],
                );
                prepare_shared(
                    state,
                    actors,
                    config,
                    control,
                    parameters,
                    shared,
                    buyer,
                    seller,
                    quantity,
                    balances.quote,
                    UNIT_POS,
                );
                prepare_route(shared, actors, config, buyer, seller, UNIT_POS);
            }
            sync_cube();
            if UNIT_POS == 0 {
                for lane in 0..min(width, end - cursor) {
                    let buyer = scratch[(parameters.sort_source + count + cursor + lane) as usize]
                        + parameters.offset;
                    let prepared = GoodsBuyer::read(shared, lane);
                    let preferred = preferred_route(shared, lane);
                    let capacity_epoch = load(
                        shared,
                        lane * BUYER_FIELDS + comptime!(BuyerField::CapacityEpoch as u32),
                    )
                    .extract(LOW_WORD);
                    let decision = decide(
                        state,
                        actors,
                        config,
                        control,
                        parameters,
                        buyer,
                        seller,
                        &prepared,
                        &preferred,
                        capacity_epoch,
                        &mut balances,
                    );
                    let base = lane * BUYER_FIELDS;
                    store(
                        shared,
                        base + comptime!(BuyerField::Arrival as u32),
                        decision.arrival,
                    );
                    shared[2 * (base + comptime!(BuyerField::Routing as u32)) as usize] =
                        decision.arrival_field;
                    decision.status.store(
                        shared,
                        2 * (base + comptime!(BuyerField::Routing as u32)) + 1,
                    );
                }
            }
            sync_cube();
            if UNIT_POS < width && slot < end {
                let id = scratch[(parameters.sort_source + count + slot) as usize];
                let prepared = GoodsBuyer::read(shared, UNIT_POS);
                let posting = prepared.posting();
                posting.apply_buyer(state, config, parameters, id + parameters.offset, seller);
                posting.status.store(scratch, 4 * count + id);
            }
            sync_cube();
            cursor += width;
        }
    }
    if UNIT_POS == 0 {
        control[COMPLETED_WORD].fetch_add(end - start);
    }
}

#[cube]
pub(super) fn dispatch(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    proposals: &mut [u32],
    scratch: &mut [u32],
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
) {
    let count = parameters.count;
    let active = markets::active(control, parameters);
    let serial = config.railway.settlements[config.actors as usize].load()
        == comptime!(Settlement::Serial as u32);
    let mut shared =
        Shared::<[u32]>::new_slice(SETTLEMENT_LANES as usize * 2 * BUYER_FIELDS as usize);
    let mut requests = RequestLists::new(comptime!(config.railway.request_storage));
    requests.initialize(config);
    requests.refresh(state, config);
    let mut index = 0u32;
    if active > 0 {
        let rotated = parameters.settlement_rotation % active;
        let seller = scratch[(parameters.sort_source + rotated) as usize];
        index = scratch[(7 * count + seller) as usize] % active;
    }
    let mut remaining = active;
    while remaining > 0 {
        let seller = scratch[(parameters.sort_source + index) as usize];
        let end = scratch[(7 * count + seller) as usize];
        if !parallel_settlement(config, parameters, seller) {
            let mode = config.railway.settlements[seller as usize].load();
            if serial || mode == comptime!(Settlement::Unbatched as u32) {
                serial_dispatch(
                    state, actors, config, proposals, scratch, control, parameters, UNIT_POS,
                    index, end,
                );
                requests.refresh(state, config);
            } else if planning::prepared(config, seller) {
                planning::execute_plan(
                    state,
                    actors,
                    config,
                    proposals,
                    scratch,
                    control,
                    parameters,
                    index,
                    end,
                    &mut requests,
                );
            } else {
                ordered_group(
                    state,
                    actors,
                    config,
                    proposals,
                    scratch,
                    control,
                    parameters,
                    &mut shared[..],
                    index,
                    end,
                );
                requests.refresh(state, config);
            }
            sync_cube();
        }
        remaining -= end - index;
        index = end % max(active, 1);
    }
    if UNIT_POS == 0 {
        control[PROGRESS_WORD].fetch_max(active);
    }
}

use super::super::routing::FreightRouteExpand;
use super::*;

#[derive(CubeType)]
pub(super) struct AtomicBalance<'a> {
    words: &'a [Atomic<u32>],
    offset: usize,
}

#[cube]
impl<'a> AtomicBalance<'a> {
    pub(super) fn at(words: &'a [Atomic<u32>], offset: usize) -> Self {
        AtomicBalance::<'a> { words, offset }
    }

    pub(super) fn load(&self) -> Ticks {
        ticks(
            self.words[self.offset].load(),
            self.words[self.offset + 1].load(),
        )
    }

    pub(super) fn store(&self, amount: Ticks) {
        self.words[self.offset].store(amount.extract(LOW_WORD));
        self.words[self.offset + 1].store(amount.extract(HIGH_WORD));
    }

    pub(super) fn overflowing_add(&self, amount: Ticks) -> bool {
        let low = amount.extract(LOW_WORD);
        let carry = if low == 0 {
            0u32
        } else {
            u32::cast_from(self.words[self.offset].fetch_add(low) > u32::MAX - low)
        };
        let high = amount.extract(HIGH_WORD);
        let increment = high + carry;
        if increment == 0 {
            high != 0
        } else {
            self.words[self.offset + 1].fetch_add(increment) > u32::MAX - increment
        }
    }

    pub(super) fn add(&self, amount: Ticks, control: &[Atomic<u32>]) {
        if self.overflowing_add(amount) {
            KernelFaults::new(comptime!(FaultFlags::ARITHMETIC_OVERFLOW)).record(control);
        }
    }
}

#[repr(u32)]
enum FreightFit {
    Tracking,
    Allowed,
    Denied,
    Exhaustive,
}

#[derive(CubeType)]
pub(super) struct RegionCursor {
    node: u32,
    origin: u32,
    carrier: u32,
    edge: u32,
    entry: u32,
    length: u32,
    steps: u32,
    previous: u32,
    pending: u32,
    first: u32,
    emitted: u32,
    finished: u32,
    reverse: bool,
}

#[cube]
impl RegionCursor {
    pub(super) fn new(path: &FreightRoute) -> Self {
        RegionCursor {
            node: path.destination,
            origin: path.origin,
            carrier: path.carrier,
            edge: 0,
            entry: 0,
            length: 0,
            steps: 0,
            previous: u32::MAX,
            pending: u32::MAX,
            first: u32::MAX,
            emitted: 0,
            finished: 0,
            reverse: false,
        }
    }

    pub(super) fn next(&mut self, config: &EconomyConfig) -> u32 {
        let mut result = u32::new(comptime!(i64::from(u32::MAX)));
        while result == u32::MAX && self.finished == 0 {
            let phase = select(
                self.entry < self.length,
                0,
                select(
                    self.node != self.origin && self.steps < config.railway.nodes,
                    1u32,
                    2u32,
                ),
            );
            match phase {
                0 => {
                    let start = config.railway.segments[self.edge as usize * SEGMENT_WORDS + 5];
                    let offset = select(self.reverse, self.length - self.entry - 1, self.entry);
                    let record = config.railway.segment_regions[(start + offset) as usize];
                    let region = config.railway.regions[record as usize * REGION_WORDS + 1];
                    self.first = select(self.first == u32::MAX, record, self.first);
                    if region != self.previous && self.pending < config.railway.region_count {
                        result = self.pending;
                        self.emitted += 1;
                    }
                    self.pending = record;
                    self.previous = region;
                    self.entry += 1;
                }
                1 => {
                    self.edge = config.railway.routes[config.railway.route_cell(
                        self.carrier,
                        self.origin,
                        self.node,
                    ) + 1];
                    self.steps += 1;
                    if self.edge < config.railway.segment_count {
                        let base = self.edge as usize * SEGMENT_WORDS;
                        self.reverse = config.railway.segments[base + 1] == self.node;
                        self.node = select(
                            self.reverse,
                            config.railway.segments[base],
                            config.railway.segments[base + 1],
                        );
                        self.length = config.railway.segments[base + 6];
                        self.entry = 0;
                    } else {
                        self.finished = 1;
                    }
                }
                _ => {
                    self.finished = 1;
                    result = select(self.emitted == 0, self.first, u32::MAX);
                }
            }
        }
        result
    }
}

#[derive(CubeType)]
pub(super) struct CapacityRequest<'a> {
    config: &'a EconomyConfig,
    resources: &'a [Atomic<u32>],
    control: &'a [Atomic<u32>],
    path: &'a FreightRoute,
    quantity: Ticks,
    sender: u32,
}

#[cube]
impl<'a> CapacityRequest<'a> {
    pub(super) fn new(
        input: &'a FreightSettlementInput,
        path: &'a FreightRoute,
        quantity: Ticks,
        sender: u32,
    ) -> Self {
        CapacityRequest::<'a> {
            config: &input.config,
            resources: &input.resources,
            control: &input.control,
            path,
            quantity,
            sender,
        }
    }

    fn available(&self, resource: u32) -> Ticks {
        AtomicBalance::at(self.resources, resource as usize * RESOURCE_WORDS).load()
    }

    fn funded(&self) -> bool {
        !lt(
            self.available(self.sender),
            times(self.quantity, self.path.fee).record(self.control),
        )
    }

    pub(super) fn fits_tracked(&self) -> bool {
        let funded = self.funded();
        let upper = times(self.quantity, self.config.railway.nodes).record(self.control);
        let mut state = select(
            funded,
            select(
                self.quantity.extract(HIGH_WORD) == 0,
                comptime!(FreightFit::Tracking as u32),
                comptime!(FreightFit::Exhaustive as u32),
            ),
            comptime!(FreightFit::Denied as u32),
        );
        let mut records = Array::<u32>::new(16usize);
        let mut counts = Array::<u32>::new(16usize);
        let mut tracked = 0u32;
        let mut cursor = RegionCursor::new(self.path);
        let mut record = u32::new(comptime!(i64::from(u32::MAX)));
        if state == comptime!(FreightFit::Tracking as u32) {
            record = cursor.next(self.config);
        }
        while record < self.config.railway.region_count
            && state == comptime!(FreightFit::Tracking as u32)
        {
            let mut slot = tracked;
            for index in 0..tracked {
                if records[index as usize] == record {
                    slot = index;
                }
            }
            state = select(
                slot < 16,
                comptime!(FreightFit::Tracking as u32),
                comptime!(FreightFit::Exhaustive as u32),
            );
            if state == comptime!(FreightFit::Tracking as u32) {
                if slot == tracked {
                    records[slot as usize] = record;
                    counts[slot as usize] = 0;
                    tracked += 1;
                }
                counts[slot as usize] += 1;
                let required = times(self.quantity, counts[slot as usize]).record(self.control);
                let stock = self.available(self.config.actors + record);
                state = select(
                    lt(stock, upper) && lt(stock, required),
                    comptime!(FreightFit::Denied as u32),
                    comptime!(FreightFit::Tracking as u32),
                );
                record = cursor.next(self.config);
            }
        }
        state = select(
            state == comptime!(FreightFit::Tracking as u32),
            comptime!(FreightFit::Allowed as u32),
            state,
        );
        let mut allowed = state == comptime!(FreightFit::Allowed as u32);
        if state == comptime!(FreightFit::Exhaustive as u32) {
            allowed = self.fits_exhaustive();
        }
        allowed
    }

    pub(super) fn fits(&self) -> bool {
        let nodes = self.config.railway.nodes;
        let mut state = u32::new(comptime!(FreightFit::Exhaustive as i64));
        if self.quantity.extract(HIGH_WORD) == 0
            && self.path.carrier < self.config.railway.company_count
            && self.path.origin < nodes
            && self.path.destination < nodes
        {
            let funded = self.funded();
            let original_upper = times(self.quantity, nodes).record(self.control);
            let base = self.config.railway.route_cell(
                self.path.carrier,
                self.path.origin,
                self.path.destination,
            );
            let visits = self.config.railway.routes[base + 4];
            let upper = minimum(
                original_upper,
                times(self.quantity, visits).record(self.control),
            );
            state = select(
                funded,
                select(
                    visits > 0,
                    comptime!(FreightFit::Tracking as u32),
                    comptime!(FreightFit::Exhaustive as u32),
                ),
                comptime!(FreightFit::Denied as u32),
            );
            let mut cursor = RegionCursor::new(self.path);
            let mut record = u32::new(comptime!(i64::from(u32::MAX)));
            if state == comptime!(FreightFit::Tracking as u32) {
                record = cursor.next(self.config);
            }
            while record < self.config.railway.region_count
                && state == comptime!(FreightFit::Tracking as u32)
            {
                let stock = self.available(self.config.actors + record);
                state = select(
                    lt(stock, self.quantity),
                    comptime!(FreightFit::Denied as u32),
                    select(
                        lt(stock, upper),
                        comptime!(FreightFit::Exhaustive as u32),
                        comptime!(FreightFit::Tracking as u32),
                    ),
                );
                record = cursor.next(self.config);
            }
        }
        let mut allowed = state == comptime!(FreightFit::Tracking as u32);
        if state == comptime!(FreightFit::Exhaustive as u32) {
            allowed = self.fits_tracked();
        }
        allowed
    }

    #[cfg(test)]
    pub(super) fn fits_unconstrained(&self) -> bool {
        let funded = self.funded();
        let upper = times(self.quantity, self.config.railway.nodes).record(self.control);
        let mut state = select(
            funded,
            select(
                self.quantity.extract(HIGH_WORD) == 0,
                comptime!(FreightFit::Tracking as u32),
                comptime!(FreightFit::Exhaustive as u32),
            ),
            comptime!(FreightFit::Denied as u32),
        );
        let mut cursor = RegionCursor::new(self.path);
        let mut record = u32::new(comptime!(i64::from(u32::MAX)));
        if state == comptime!(FreightFit::Tracking as u32) {
            record = cursor.next(self.config);
        }
        while record < self.config.railway.region_count
            && state == comptime!(FreightFit::Tracking as u32)
        {
            let stock = self.available(self.config.actors + record);
            state = select(
                lt(stock, upper),
                select(
                    lt(stock, self.quantity),
                    comptime!(FreightFit::Denied as u32),
                    comptime!(FreightFit::Exhaustive as u32),
                ),
                comptime!(FreightFit::Tracking as u32),
            );
            record = cursor.next(self.config);
        }
        let mut allowed = state == comptime!(FreightFit::Tracking as u32);
        if state == comptime!(FreightFit::Exhaustive as u32) {
            allowed = self.fits_exhaustive();
        }
        allowed
    }

    pub(super) fn fits_exhaustive(&self) -> bool {
        let mut allowed = self.funded();
        let mut cursor = RegionCursor::new(self.path);
        let mut record = cursor.next(self.config);
        let upper = times(self.quantity, self.config.railway.nodes).record(self.control);
        while record < self.config.railway.region_count && allowed {
            let stock = self.available(self.config.actors + record);
            if lt(stock, upper) {
                let required = times(self.quantity, self.path.visits(self.config, record))
                    .record(self.control);
                allowed = !lt(stock, required);
            }
            record = cursor.next(self.config);
        }
        allowed
    }
}
#[cube]
impl FreightRoute {
    pub(super) fn visits(&self, config: &EconomyConfig, target: u32) -> u32 {
        let mut cursor = RegionCursor::new(self);
        let mut record = cursor.next(config);
        let mut total = 0u32;
        while record < config.railway.region_count {
            total += u32::cast_from(record == target);
            record = cursor.next(config);
        }
        total
    }
}

#[cube]
impl FreightSettlementInput {
    pub(super) fn budget(&self, resource: u32) -> Ticks {
        AtomicBalance::at(&self.resources, resource as usize * RESOURCE_WORDS).load()
    }

    pub(super) fn claim(&self, resource: u32, amount: Ticks) {
        let base = resource as usize * RESOURCE_WORDS;
        if !equal(amount, zero()) {
            self.resources[base + RESOURCE_CLAIMS].fetch_add(1);
            claims::add_demand(
                &self.resources,
                base + REQUESTED,
                base + REQUEST_OVERFLOW,
                amount,
            );
        }
    }

    pub(super) fn consume_resource(&self, resource: u32, amount: Ticks) {
        AtomicBalance::at(
            &self.resources,
            resource as usize * RESOURCE_WORDS + CONSUMED,
        )
        .add(amount, &self.control);
    }

    pub(super) fn reset_resource(&self, id: u32) {
        if id < self.config.actors + self.config.railway.region_count {
            for offset in 2..RESOURCE_WORDS {
                self.resources[id as usize * RESOURCE_WORDS + offset].store(0);
            }
        }
    }

    pub(super) fn release_resource(&mut self, id: u32) {
        let actors = self.config.actors;
        let resources = actors + self.config.railway.region_count;
        let base = id as usize * RESOURCE_WORDS;
        if id < resources {
            let available = self.budget(id);
            let used = AtomicBalance::at(&self.resources, base + CONSUMED).load();
            AtomicBalance::at(&self.resources, base).store(minus(available, used));
        }
        if id >= actors && id < resources {
            let region = id - actors;
            let capacity = RegionalCapacity::new(&self.config.railway, region);
            let available = self.budget(id);
            let mut ledger = Ledger::new(&mut self.state, &self.config);
            capacity.set(
                &mut ledger,
                comptime!(RailCapacityField::Available),
                available,
            );
            let level = capacity.level(&ledger);
            let before = plus(
                available,
                AtomicBalance::at(&self.resources, base + CONSUMED).load(),
            )
            .value;
            let used = minus(times32(level, self.config.railway.units_per_level), before);
            let wanted = claims::saturated_sum(
                used,
                maximum(
                    claims::demand(&self.resources, base + REQUESTED, base + REQUEST_OVERFLOW),
                    claims::demand(&self.resources, base + PREFERRED, base + PREFERRED_OVERFLOW),
                ),
            );
            let previous = capacity.value(&ledger, comptime!(RailCapacityField::Demand));
            capacity.set(
                &mut ledger,
                comptime!(RailCapacityField::Demand),
                maximum(previous, wanted),
            );
        }
    }
}

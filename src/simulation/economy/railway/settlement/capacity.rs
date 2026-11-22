use super::*;

#[repr(u32)]
enum FreightFit {
    Tracking,
    Allowed,
    Denied,
    Exhaustive,
}

#[cube]
pub(super) fn atomic_ticks(words: &[Atomic<u32>], index: usize) -> Ticks {
    ticks(words[index].load(), words[index + 1].load())
}

#[cube]
pub(super) fn atomic_store(words: &[Atomic<u32>], index: usize, amount: Ticks) {
    words[index].store(amount.extract(LOW_WORD));
    words[index + 1].store(amount.extract(HIGH_WORD));
}

#[cube]
pub(super) fn atomic_add_overflows(words: &[Atomic<u32>], index: usize, amount: Ticks) -> bool {
    let low = amount.extract(LOW_WORD);
    let carry = if low == 0 {
        0u32
    } else {
        u32::cast_from(words[index].fetch_add(low) > u32::MAX - low)
    };
    let high = amount.extract(HIGH_WORD);
    let increment = high + carry;
    if increment == 0 {
        high != 0
    } else {
        words[index + 1].fetch_add(increment) > u32::MAX - increment
    }
}

#[cube]
pub(super) fn atomic_add(
    words: &[Atomic<u32>],
    index: usize,
    amount: Ticks,
    control: &[Atomic<u32>],
) {
    if atomic_add_overflows(words, index, amount) {
        KernelFaults::new(comptime!(FaultFlags::ARITHMETIC_OVERFLOW)).record(control);
    }
}

#[cube]
pub(super) fn budget(input: &Input, resource: u32) -> Ticks {
    atomic_ticks(&input.resources, resource as usize * RESOURCE_WORDS)
}

#[cube]
pub(super) fn claim(input: &Input, resource: u32, amount: Ticks) {
    let base = resource as usize * RESOURCE_WORDS;
    if !equal(amount, zero()) {
        input.resources[base + RESOURCE_CLAIMS].fetch_add(1);
        claims::add_demand(
            &input.resources,
            base + REQUESTED,
            base + REQUEST_OVERFLOW,
            amount,
        );
    }
}

#[cube]
pub(super) fn consume_resource(input: &Input, resource: u32, amount: Ticks) {
    atomic_add(
        &input.resources,
        resource as usize * RESOURCE_WORDS + CONSUMED,
        amount,
        &input.control,
    );
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
                    self.edge = config.railway.routes
                        [route_cell(config, self.carrier, self.origin, self.node) + 1];
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

#[cube]
pub(super) fn visits(config: &EconomyConfig, path: &FreightRoute, target: u32) -> u32 {
    let mut cursor = RegionCursor::new(path);
    let mut record = cursor.next(config);
    let mut total = 0u32;
    while record < config.railway.region_count {
        total += u32::cast_from(record == target);
        record = cursor.next(config);
    }
    total
}

#[cube]
pub(super) fn fits_tracked(
    input: &Input,
    path: &FreightRoute,
    quantity: Ticks,
    sender: u32,
) -> bool {
    let funded = !lt(
        budget(input, sender),
        multiply(&input.control, quantity, path.fee),
    );
    let upper = multiply(&input.control, quantity, input.config.railway.nodes);
    let mut state = select(
        funded,
        select(
            quantity.extract(HIGH_WORD) == 0,
            comptime!(FreightFit::Tracking as u32),
            comptime!(FreightFit::Exhaustive as u32),
        ),
        comptime!(FreightFit::Denied as u32),
    );
    let mut records = Array::<u32>::new(16usize);
    let mut counts = Array::<u32>::new(16usize);
    let mut tracked = 0u32;
    let mut cursor = RegionCursor::new(path);
    let mut record = u32::new(comptime!(i64::from(u32::MAX)));
    if state == comptime!(FreightFit::Tracking as u32) {
        record = cursor.next(&input.config);
    }
    while record < input.config.railway.region_count
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
            let required = multiply(&input.control, quantity, counts[slot as usize]);
            let stock = budget(input, input.config.actors + record);
            state = select(
                lt(stock, upper) && lt(stock, required),
                comptime!(FreightFit::Denied as u32),
                comptime!(FreightFit::Tracking as u32),
            );
            record = cursor.next(&input.config);
        }
    }
    state = select(
        state == comptime!(FreightFit::Tracking as u32),
        comptime!(FreightFit::Allowed as u32),
        state,
    );
    let mut allowed = state == comptime!(FreightFit::Allowed as u32);
    if state == comptime!(FreightFit::Exhaustive as u32) {
        allowed = fits_exhaustive(input, path, quantity, sender);
    }
    allowed
}

#[cube]
pub(super) fn fits(input: &Input, path: &FreightRoute, quantity: Ticks, sender: u32) -> bool {
    let nodes = input.config.railway.nodes;
    let mut state = u32::new(comptime!(FreightFit::Exhaustive as i64));
    if quantity.extract(HIGH_WORD) == 0
        && path.carrier < input.config.railway.company_count
        && path.origin < nodes
        && path.destination < nodes
    {
        let funded = !lt(
            budget(input, sender),
            multiply(&input.control, quantity, path.fee),
        );
        let original_upper = multiply(&input.control, quantity, nodes);
        let base = route_cell(&input.config, path.carrier, path.origin, path.destination);
        let visits = input.config.railway.routes[base + 4];
        let upper = minimum(original_upper, multiply(&input.control, quantity, visits));
        state = select(
            funded,
            select(
                visits > 0,
                comptime!(FreightFit::Tracking as u32),
                comptime!(FreightFit::Exhaustive as u32),
            ),
            comptime!(FreightFit::Denied as u32),
        );
        let mut cursor = RegionCursor::new(path);
        let mut record = u32::new(comptime!(i64::from(u32::MAX)));
        if state == comptime!(FreightFit::Tracking as u32) {
            record = cursor.next(&input.config);
        }
        while record < input.config.railway.region_count
            && state == comptime!(FreightFit::Tracking as u32)
        {
            let stock = budget(input, input.config.actors + record);
            state = select(
                lt(stock, quantity),
                comptime!(FreightFit::Denied as u32),
                select(
                    lt(stock, upper),
                    comptime!(FreightFit::Exhaustive as u32),
                    comptime!(FreightFit::Tracking as u32),
                ),
            );
            record = cursor.next(&input.config);
        }
    }
    let mut allowed = state == comptime!(FreightFit::Tracking as u32);
    if state == comptime!(FreightFit::Exhaustive as u32) {
        allowed = fits_tracked(input, path, quantity, sender);
    }
    allowed
}

#[cfg(test)]
#[cube]
pub(super) fn fits_unconstrained(
    input: &Input,
    path: &FreightRoute,
    quantity: Ticks,
    sender: u32,
) -> bool {
    let funded = !lt(
        budget(input, sender),
        multiply(&input.control, quantity, path.fee),
    );
    let upper = multiply(&input.control, quantity, input.config.railway.nodes);
    let mut state = select(
        funded,
        select(
            quantity.extract(HIGH_WORD) == 0,
            comptime!(FreightFit::Tracking as u32),
            comptime!(FreightFit::Exhaustive as u32),
        ),
        comptime!(FreightFit::Denied as u32),
    );
    let mut cursor = RegionCursor::new(path);
    let mut record = u32::new(comptime!(i64::from(u32::MAX)));
    if state == comptime!(FreightFit::Tracking as u32) {
        record = cursor.next(&input.config);
    }
    while record < input.config.railway.region_count
        && state == comptime!(FreightFit::Tracking as u32)
    {
        let stock = budget(input, input.config.actors + record);
        state = select(
            lt(stock, upper),
            select(
                lt(stock, quantity),
                comptime!(FreightFit::Denied as u32),
                comptime!(FreightFit::Exhaustive as u32),
            ),
            comptime!(FreightFit::Tracking as u32),
        );
        record = cursor.next(&input.config);
    }
    let mut allowed = state == comptime!(FreightFit::Tracking as u32);
    if state == comptime!(FreightFit::Exhaustive as u32) {
        allowed = fits_exhaustive(input, path, quantity, sender);
    }
    allowed
}

#[cube]
pub(super) fn fits_exhaustive(
    input: &Input,
    path: &FreightRoute,
    quantity: Ticks,
    sender: u32,
) -> bool {
    let mut allowed = !lt(
        budget(input, sender),
        multiply(&input.control, quantity, path.fee),
    );
    let mut cursor = RegionCursor::new(path);
    let mut record = cursor.next(&input.config);
    let upper = multiply(&input.control, quantity, input.config.railway.nodes);
    while record < input.config.railway.region_count && allowed {
        let stock = budget(input, input.config.actors + record);
        if lt(stock, upper) {
            let required = multiply(
                &input.control,
                quantity,
                visits(&input.config, path, record),
            );
            allowed = !lt(stock, required);
        }
        record = cursor.next(&input.config);
    }
    allowed
}

#[cube]
pub(super) fn reset_resource(input: &Input, id: u32) {
    if id < input.config.actors + input.config.railway.region_count {
        for offset in 2..RESOURCE_WORDS {
            input.resources[id as usize * RESOURCE_WORDS + offset].store(0);
        }
    }
}

#[cube]
pub(super) fn release_resource(input: &mut Input, id: u32) {
    let actors = input.config.actors;
    let resources = actors + input.config.railway.region_count;
    let base = id as usize * RESOURCE_WORDS;
    if id < resources {
        let available = budget(&*input, id);
        let used = atomic_ticks(&input.resources, base + CONSUMED);
        atomic_store(&input.resources, base, minus(available, used));
    }
    if id >= actors && id < resources {
        let region = id - actors;
        let available = budget(&*input, id);
        let mut ledger = Ledger::new(&mut input.state, &input.config);
        regional_set(
            &mut ledger,
            region,
            available,
            comptime!(RailCapacityField::Available),
        );
        let level = regional_number(&ledger, region, comptime!(RailCapacityField::Level));
        let before = plus(available, atomic_ticks(&input.resources, base + CONSUMED)).value;
        let used = minus(times32(level, input.config.railway.units_per_level), before);
        let wanted = claims::saturated_sum(
            used,
            maximum(
                claims::demand(&input.resources, base + REQUESTED, base + REQUEST_OVERFLOW),
                claims::demand(
                    &input.resources,
                    base + PREFERRED,
                    base + PREFERRED_OVERFLOW,
                ),
            ),
        );
        let previous = regional_value(&ledger, region, comptime!(RailCapacityField::Demand));
        regional_set(
            &mut ledger,
            region,
            maximum(previous, wanted),
            comptime!(RailCapacityField::Demand),
        );
    }
}

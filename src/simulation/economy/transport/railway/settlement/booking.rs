use super::*;

#[derive(Clone, Copy)]
#[repr(usize)]
enum Account {
    Cost = 0,
    Revenue = 2,
    Used = 4,
    Sold = 6,
}

#[cube]
impl FreightSettlementInput {
    fn account(&self, actor: u32, amount: Ticks, #[comptime] field: Account) {
        AtomicBalance::at(
            &self.accounts,
            actor as usize * ACCOUNT_WORDS + comptime!(field as usize),
        )
        .add(amount, &self.control);
    }

    pub(super) fn quantity(&self, buyer: u32) -> Ticks {
        ticks(
            self.proposals[buyer as usize * ProposalLayout::WORDS + ProposalLayout::QUANTITY_LOW],
            self.proposals[buyer as usize * ProposalLayout::WORDS + ProposalLayout::QUANTITY_HIGH],
        )
    }

    pub(super) fn active_bookings(&self) -> u32 {
        self.progress[comptime!(self.execution.wave() as usize)].load()
    }

    pub(super) fn booking_priority(&self, slot: u32) -> u32 {
        let offset = match comptime!(self.execution.wave()) {
            Wave::Initial => 0u32,
            Wave::Retry => self.config.actors,
        };
        self.queue[(offset + slot) as usize]
    }

    pub(super) fn booking_buyer(&self, priority: u32) -> u32 {
        self.scratch[(self.parameters.sort_source + self.parameters.count + priority) as usize]
            + self.parameters.offset
    }

    pub(super) fn propose(&mut self, invocation: u32) {
        if invocation < self.active_bookings() {
            let index = self.booking_priority(invocation);
            let buyer = self.booking_buyer(index);
            let booking = self.bookings[buyer as usize * FREIGHT_BOOKING_WORDS];
            if booking != comptime!(BookingState::Complete as u32) {
                self.accounts[buyer as usize * ACCOUNT_WORDS + DENIED].store(0);
                let seller =
                    self.proposals[buyer as usize * ProposalLayout::WORDS + ProposalLayout::ACTOR];
                let amount = self.quantity(buyer);
                let request = FreightRequest::new(
                    &ActorTable::new(&self.actors),
                    &self.config,
                    buyer,
                    seller,
                );
                let mut selected = request.animal();
                let quoted = request.quoted_carrier(&self.config);
                let preferred = request.candidate(&self.config, quoted);
                if preferred.carrier < self.config.railway.company_count {
                    let mut cursor = RegionCursor::new(&preferred);
                    let mut record = cursor.next(&self.config);
                    while record < self.config.railway.region_count {
                        claims::add_demand(
                            &self.resources,
                            (self.config.actors + record) as usize * RESOURCE_WORDS + PREFERRED,
                            (self.config.actors + record) as usize * RESOURCE_WORDS
                                + PREFERRED_OVERFLOW,
                            amount,
                        );
                        record = cursor.next(&self.config);
                    }
                }
                let origin = self.config.railway.access[seller as usize * 2];
                let mut start = 0u32;
                let mut end = 0u32;
                if (quoted < self.config.railway.company_count
                    || request.animal_days > self.config.ring)
                    && origin < self.config.railway.nodes
                {
                    start = self.config.railway.station_companies[origin as usize];
                    end = self.config.railway.station_companies[origin as usize + 1];
                }
                if booking == comptime!(BookingState::Retrying as u32) {
                    let previous = request.candidate(
                        &self.config,
                        self.bookings[buyer as usize * FREIGHT_BOOKING_WORDS + 1],
                    );
                    if CapacityRequest::new(&*self, &previous, amount, seller).fits() {
                        selected.replace(&previous);
                        start = end;
                    }
                }
                let tested_carrier = select(
                    booking == comptime!(BookingState::Pending as u32)
                        && amount.extract(HIGH_WORD) == 0
                        && request.animal_days <= self.config.ring
                        && preferred.carrier < self.config.railway.company_count,
                    preferred.carrier,
                    u32::MAX,
                );
                if tested_carrier < self.config.railway.company_count
                    && CapacityRequest::new(&*self, &preferred, amount, seller).fits()
                {
                    selected.replace(&preferred);
                    start = end;
                }
                for entry in start..end {
                    let company = self.config.railway.station_companies[entry as usize];
                    let path = request.candidate(&self.config, company);
                    if company != tested_carrier
                        && path.carrier < self.config.railway.company_count
                        && path.preferred_to(&selected)
                        && CapacityRequest::new(&*self, &path, amount, seller).fits()
                    {
                        selected.replace(&path);
                    }
                }
                self.bookings[buyer as usize * FREIGHT_BOOKING_WORDS] =
                    comptime!(BookingState::Complete as u32);
                if selected.carrier < self.config.railway.company_count {
                    self.bookings[buyer as usize * FREIGHT_BOOKING_WORDS] =
                        comptime!(BookingState::Ready as u32);
                    self.bookings[buyer as usize * FREIGHT_BOOKING_WORDS + 1] = selected.carrier;
                    let cost = times(amount, selected.fee).record(&self.control);
                    self.claim(seller, cost);
                    let mut cursor = RegionCursor::new(&selected);
                    let mut record = cursor.next(&self.config);
                    while record < self.config.railway.region_count {
                        self.claim(self.config.actors + record, amount);
                        record = cursor.next(&self.config);
                    }
                }
            }
        }
    }

    fn toll(&self, carrier: u32, owner: u32, units: u32, quantity: Ticks) {
        if owner < self.config.railway.company_count {
            let companies = &self.config.railway.companies;
            if companies[owner as usize * COMPANY_WORDS + 3]
                != companies[carrier as usize * COMPANY_WORDS + 3]
            {
                let fee = max(units, 1) * companies[owner as usize * COMPANY_WORDS + 2];
                self.account(
                    companies[owner as usize * COMPANY_WORDS],
                    times(quantity, fee).record(&self.control),
                    comptime!(Account::Revenue),
                );
            }
        }
    }

    fn credit_tolls(&self, path: &FreightRoute, quantity: Ticks) {
        let config = &self.config;
        let mut node = path.destination;
        let mut owner = u32::new(comptime!(i64::from(u32::MAX)));
        let mut region = u32::new(comptime!(i64::from(u32::MAX)));
        let mut units = 0u32;
        let mut steps = 0u32;
        while node != path.origin && steps < config.railway.nodes {
            let edge = config.railway.routes
                [config.railway.route_cell(path.carrier, path.origin, node) + 1];
            if edge < config.railway.segment_count {
                let base = edge as usize * SEGMENT_WORDS;
                let next_owner = config.railway.segments[base + 2];
                if next_owner != owner {
                    self.toll(path.carrier, owner, units, quantity);
                    owner = next_owner;
                    region = u32::MAX;
                    units = 0;
                }
                let reverse = config.railway.segments[base + 1] == node;
                let start = config.railway.segments[base + 5];
                let length = config.railway.segments[base + 6];
                for entry in 0..length {
                    let offset = select(reverse, length - entry - 1, entry);
                    let record = config.railway.segment_regions[(start + offset) as usize];
                    let next_region = config.railway.regions[record as usize * REGION_WORDS + 1];
                    units += u32::cast_from(region != u32::MAX && region != next_region);
                    region = next_region;
                }
                node = select(
                    reverse,
                    config.railway.segments[base],
                    config.railway.segments[base + 1],
                );
            }
            steps += 1;
        }
        self.toll(path.carrier, owner, units, quantity);
    }

    pub(super) fn commit(&mut self, invocation: u32) {
        if invocation < self.active_bookings() {
            let index = self.booking_priority(invocation);
            let buyer = self.booking_buyer(index);
            if self.bookings[buyer as usize * FREIGHT_BOOKING_WORDS]
                == comptime!(BookingState::Ready as u32)
            {
                let seller =
                    self.proposals[buyer as usize * ProposalLayout::WORDS + ProposalLayout::ACTOR];
                let path = FreightRequest::new(
                    &ActorTable::new(&self.actors),
                    &self.config,
                    buyer,
                    seller,
                )
                .candidate(
                    &self.config,
                    self.bookings[buyer as usize * FREIGHT_BOOKING_WORDS + 1],
                );
                let amount = self.quantity(buyer);
                let allowed = self.accounts[buyer as usize * ACCOUNT_WORDS + DENIED].load() == 0;
                self.bookings[buyer as usize * FREIGHT_BOOKING_WORDS] =
                    comptime!(BookingState::Retrying as u32);
                if allowed {
                    let good = self.parameters.traded_good();
                    let animal = FreightRoute::by_road(
                        &ActorTable::new(&self.actors),
                        &self.config,
                        buyer,
                        seller,
                    );
                    let base = BASE_FIELDS as u32 + (good - 1) * self.config.ring;
                    let old_field = cartage::waiting_field(&self.config, good, animal.days);
                    let mut ledger = Ledger::new(&mut self.state, &self.config);
                    let payable = ledger.value(Field::CartagePayable, buyer);
                    let waived = times(amount, cartage::rate(&self.config, animal.days))
                        .record(&self.control);
                    ledger.set(Field::CartagePayable, buyer, minus(payable, waived));
                    let waiting = ledger.value(Field::RoadFreight, buyer);
                    ledger.set(Field::RoadFreight, buyer, minus(waiting, amount));
                    let new_field = base + (self.parameters.day + path.days) % self.config.ring;
                    let old = ledger.value_at(old_field, buyer);
                    let new = plus(ledger.value_at(new_field, buyer), amount).record(&self.control);
                    ledger.set_at(old_field, buyer, minus(old, amount));
                    ledger.set_at(new_field, buyer, new);
                    let cost = times(amount, path.fee).record(&self.control);
                    self.consume_resource(seller, cost);
                    self.account(seller, cost, comptime!(Account::Cost));
                    let mut cursor = RegionCursor::new(&path);
                    let mut record = cursor.next(&self.config);
                    let mut units = 0u32;
                    while record < self.config.railway.region_count {
                        self.consume_resource(self.config.actors + record, amount);
                        let owner = self.config.railway.regions[record as usize * REGION_WORDS];
                        let firm = self.config.railway.companies[owner as usize * COMPANY_WORDS];
                        self.account(firm, amount, comptime!(Account::Sold));
                        units += 1;
                        record = cursor.next(&self.config);
                    }
                    let total = times(amount, units).record(&self.control);
                    let carrier =
                        self.config.railway.companies[path.carrier as usize * COMPANY_WORDS];
                    let rate =
                        self.config.railway.companies[path.carrier as usize * COMPANY_WORDS + 1];
                    self.account(seller, total, comptime!(Account::Used));
                    self.account(
                        carrier,
                        times(total, rate).record(&self.control),
                        comptime!(Account::Revenue),
                    );
                    self.credit_tolls(&path, amount);
                    self.bookings[buyer as usize * FREIGHT_BOOKING_WORDS] =
                        comptime!(BookingState::Complete as u32);
                } else {
                    match comptime!(self.execution.wave()) {
                        Wave::Initial => {
                            let next = self.progress[comptime!(Wave::Retry as usize)].fetch_add(1);
                            self.queue[(self.config.actors + next) as usize] = index;
                        }
                        _ => {
                            self.bookings[buyer as usize * FREIGHT_BOOKING_WORDS] =
                                comptime!(BookingState::Complete as u32)
                        }
                    }
                }
            }
        }
    }

    pub(super) fn initialize_booking(&mut self, invocation: u32) {
        if invocation < orders::active(&self.control, &self.parameters) {
            let buyer = self.booking_buyer(invocation);
            let status = KernelEventStatus::load(
                &mut self.scratch,
                4 * self.parameters.count + buyer - self.parameters.offset,
            );
            self.bookings[buyer as usize * FREIGHT_BOOKING_WORDS] = select(
                status.contains(comptime!(EventStatus::Accepted)),
                comptime!(BookingState::Pending as u32),
                comptime!(BookingState::Complete as u32),
            );
            if status.contains(comptime!(EventStatus::Accepted)) {
                let slot = self.progress[comptime!(Wave::Initial as usize)].fetch_add(1);
                self.queue[slot as usize] = invocation;
            }
        }
    }

    pub(super) fn settle_account(&mut self, id: u32) {
        let mut ledger = Ledger::new(&mut self.state, &self.config);
        let actors = self.config.actors;
        if id < actors {
            let offset = id as usize * ACCOUNT_WORDS;
            let cost =
                AtomicBalance::at(&self.accounts, offset + comptime!(Account::Cost as usize))
                    .load();
            let revenue = AtomicBalance::at(
                &self.accounts,
                offset + comptime!(Account::Revenue as usize),
            )
            .load();
            let used =
                AtomicBalance::at(&self.accounts, offset + comptime!(Account::Used as usize))
                    .load();
            let sold =
                AtomicBalance::at(&self.accounts, offset + comptime!(Account::Sold as usize))
                    .load();
            let cash = ledger.value(Field::Cash, id);
            let cash = plus(minus(cash, cost), revenue).record(&self.control);
            ledger.set(Field::Cash, id, cash);
            ledger.increase(&self.control, Field::RailRevenue, id, revenue);
            record_use(&mut ledger, &self.control, id, used);
            let stock = ledger.value(Field::Railway, id);
            ledger.set(Field::Railway, id, minus(stock, sold));
            ledger.increase(&self.control, Field::Sales, id, sold);
        }
    }
}

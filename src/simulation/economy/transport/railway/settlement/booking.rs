use super::super::routing::{RouteCursor, TrackAccess};
use super::*;

#[derive(Clone, Copy)]
#[repr(usize)]
enum Account {
    Cost = 0,
    Revenue = 2,
    Used = 4,
    Sold = 6,
}

#[derive(CubeType)]
pub(super) struct ReadyBooking {
    pub(super) priority: u32,
    pub(super) buyer: u32,
    pub(super) seller: u32,
    pub(super) quantity: Ticks,
    pub(super) path: FreightRoute,
    road_days: u32,
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

    fn quantity(&self, buyer: u32) -> Ticks {
        ticks(
            self.proposals[buyer as usize * ProposalLayout::WORDS + ProposalLayout::QUANTITY_LOW],
            self.proposals[buyer as usize * ProposalLayout::WORDS + ProposalLayout::QUANTITY_HIGH],
        )
    }

    fn seller(&self, buyer: u32) -> u32 {
        self.proposals[buyer as usize * ProposalLayout::WORDS + ProposalLayout::ACTOR]
    }

    fn active_bookings(&self) -> u32 {
        self.progress[comptime!(self.execution.wave() as usize)].load()
    }

    fn booking_priority(&self, slot: u32) -> u32 {
        let offset = match comptime!(self.execution.wave()) {
            Wave::Initial => 0u32,
            Wave::Retry => self.config.actors,
        };
        self.queue[(offset + slot) as usize]
    }

    fn booking_buyer(&self, priority: u32) -> u32 {
        self.scratch[(self.parameters.sort_source + self.parameters.count + priority) as usize]
            + self.parameters.offset
    }

    pub(super) fn process_booking(&mut self, invocation: u32) {
        if invocation < self.active_bookings() {
            let priority = self.booking_priority(invocation);
            let buyer = self.booking_buyer(priority);
            let state = self.bookings[buyer as usize * FREIGHT_BOOKING_WORDS];
            if comptime!(self.stage == Stage::Propose) {
                self.propose(buyer, state);
            } else if state == comptime!(BookingState::Ready as u32) {
                let seller = self.seller(buyer);
                let request = FreightRequest::new(
                    &ActorTable::new(&self.actors),
                    &self.config,
                    buyer,
                    seller,
                );
                let booking = ReadyBooking {
                    priority,
                    buyer,
                    seller,
                    quantity: self.quantity(buyer),
                    path: request.candidate(
                        &self.config,
                        self.bookings[buyer as usize * FREIGHT_BOOKING_WORDS + 1],
                    ),
                    road_days: request.animal_days,
                };
                match comptime!(self.stage) {
                    Stage::EmitClaims => self.emit_claims(&booking),
                    Stage::Commit => self.commit(&booking),
                    _ => (),
                }
            }
        }
    }

    fn propose(&mut self, buyer: u32, booking: u32) {
        if booking != comptime!(BookingState::Complete as u32) {
            self.accounts[buyer as usize * ACCOUNT_WORDS + DENIED].store(0);
            let seller = self.seller(buyer);
            let amount = self.quantity(buyer);
            let request =
                FreightRequest::new(&ActorTable::new(&self.actors), &self.config, buyer, seller);
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

    fn toll(&self, carrier: u32, access: &TrackAccess, quantity: Ticks) {
        let railway = &self.config.railway;
        let fee = railway.access_fare(access.owner, carrier, access.units);
        if fee > 0 {
            self.account(
                railway.companies[access.owner as usize * COMPANY_WORDS],
                times(quantity, fee).record(&self.control),
                comptime!(Account::Revenue),
            );
        }
    }

    fn credit_tolls(&self, path: &FreightRoute, quantity: Ticks) {
        let railway = &self.config.railway;
        let mut cursor = RouteCursor::new(path);
        let mut access = TrackAccess::new();
        while cursor.has_next(railway) {
            if cursor.advance(railway) {
                if cursor.owner != access.owner {
                    self.toll(path.carrier, &access, quantity);
                    access.start_on(cursor.owner);
                }
                for entry in 0..cursor.length {
                    let record = cursor.record(railway, entry);
                    access.visit(railway.regions[record as usize * REGION_WORDS + 1]);
                }
            }
        }
        self.toll(path.carrier, &access, quantity);
    }

    fn commit(&mut self, booking: &ReadyBooking) {
        let buyer = booking.buyer;
        let seller = booking.seller;
        let path = &booking.path;
        let amount = booking.quantity;
        let allowed = self.accounts[buyer as usize * ACCOUNT_WORDS + DENIED].load() == 0;
        self.bookings[buyer as usize * FREIGHT_BOOKING_WORDS] =
            comptime!(BookingState::Retrying as u32);
        if allowed {
            let good = self.parameters.good;
            let base = BASE_FIELDS as u32 + (good - 1) * self.config.ring;
            let old_field = cartage::waiting_field(&self.config, good, booking.road_days);
            let mut ledger = Ledger::new(&mut self.state, &self.config);
            let actors = ActorTable::new(&self.actors);
            let terms = household_production::TradeFields::new(&actors, &self.config, seller, good);
            let price = ledger.value_at(terms.price, seller);
            if good == comptime!(Good::Bread as u32)
                && actors.kind(buyer).is(comptime!(ActorKind::Household))
            {
                ledger.set(Field::FoodPrice, buyer, price);
            }
            for input in 0u32..2 {
                if ledger.number_at(comptime!(Field::FirstInput as u32) + input, buyer) == good {
                    ledger.set_at(
                        comptime!(Field::FirstInputCost as u32) + input,
                        buyer,
                        price,
                    );
                }
            }
            if pricing::distributor(&ledger, buyer) {
                let field = pricing::cost_field(&self.config, actors.kind(buyer).output(), good);
                ledger.set_at(field, buyer, price);
            }
            let payable = ledger.value(Field::CartagePayable, buyer);
            let waived =
                times(amount, cartage::rate(&self.config, booking.road_days)).record(&self.control);
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
            let mut cursor = RegionCursor::new(path);
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
            let carrier = self.config.railway.companies[path.carrier as usize * COMPANY_WORDS];
            let rate = self.config.railway.companies[path.carrier as usize * COMPANY_WORDS + 1];
            self.account(seller, total, comptime!(Account::Used));
            self.account(
                carrier,
                times(total, rate).record(&self.control),
                comptime!(Account::Revenue),
            );
            self.credit_tolls(path, amount);
            self.bookings[buyer as usize * FREIGHT_BOOKING_WORDS] =
                comptime!(BookingState::Complete as u32);
        } else {
            match comptime!(self.execution.wave()) {
                Wave::Initial => {
                    let next = self.progress[comptime!(Wave::Retry as usize)].fetch_add(1);
                    self.queue[(self.config.actors + next) as usize] = booking.priority;
                }
                _ => {
                    self.bookings[buyer as usize * FREIGHT_BOOKING_WORDS] =
                        comptime!(BookingState::Complete as u32)
                }
            }
        }
    }

    pub(super) fn initialize_booking(&mut self, invocation: u32) {
        if invocation < self.parameters.active(&self.control) {
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

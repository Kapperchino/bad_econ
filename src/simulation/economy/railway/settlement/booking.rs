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
fn account(input: &Input, actor: u32, amount: Ticks, #[comptime] field: Account) {
    atomic_add(
        &input.accounts,
        actor as usize * ACCOUNT_WORDS + comptime!(field as usize),
        amount,
        &input.control,
    );
}

#[cube]
pub(super) fn quantity(input: &Input, buyer: u32) -> Ticks {
    ticks(
        input.proposals[buyer as usize * ProposalLayout::WORDS + ProposalLayout::QUANTITY_LOW],
        input.proposals[buyer as usize * ProposalLayout::WORDS + ProposalLayout::QUANTITY_HIGH],
    )
}

#[cube]
pub(super) fn active_bookings(input: &Input) -> u32 {
    input.progress[comptime!(input.execution.wave() as usize)].load()
}

#[cube]
pub(super) fn booking_priority(input: &Input, slot: u32) -> u32 {
    let offset = match comptime!(input.execution.wave()) {
        Wave::Initial => 0u32,
        Wave::Retry => input.config.actors,
    };
    input.queue[(offset + slot) as usize]
}

#[cube]
pub(super) fn booking_buyer(input: &Input, priority: u32) -> u32 {
    input.scratch[(input.parameters.sort_source + input.parameters.count + priority) as usize]
        + input.parameters.offset
}

#[cube]
pub(super) fn propose(input: &mut Input, invocation: u32) {
    if invocation < active_bookings(&*input) {
        let index = booking_priority(&*input, invocation);
        let buyer = booking_buyer(&*input, index);
        let booking = input.bookings[buyer as usize * FREIGHT_BOOKING_WORDS];
        if booking != comptime!(BookingState::Complete as u32) {
            input.accounts[buyer as usize * ACCOUNT_WORDS + DENIED].store(0);
            let seller =
                input.proposals[buyer as usize * ProposalLayout::WORDS + ProposalLayout::ACTOR];
            let amount = quantity(&*input, buyer);
            let request = FreightRequest::new(
                &ActorTable::new(&input.actors),
                &input.config,
                buyer,
                seller,
            );
            let mut selected = request.animal();
            let quoted = request.quoted_carrier(&input.config);
            let preferred = request.candidate(&input.config, quoted);
            if preferred.carrier < input.config.railway.company_count {
                let mut cursor = RegionCursor::new(&preferred);
                let mut record = cursor.next(&input.config);
                while record < input.config.railway.region_count {
                    claims::add_demand(
                        &input.resources,
                        (input.config.actors + record) as usize * RESOURCE_WORDS + PREFERRED,
                        (input.config.actors + record) as usize * RESOURCE_WORDS
                            + PREFERRED_OVERFLOW,
                        amount,
                    );
                    record = cursor.next(&input.config);
                }
            }
            let origin = input.config.railway.access[seller as usize * 2];
            let mut start = 0u32;
            let mut end = 0u32;
            if (quoted < input.config.railway.company_count
                || request.animal_days > input.config.ring)
                && origin < input.config.railway.nodes
            {
                start = input.config.railway.station_companies[origin as usize];
                end = input.config.railway.station_companies[origin as usize + 1];
            }
            if booking == comptime!(BookingState::Retrying as u32) {
                let previous = request.candidate(
                    &input.config,
                    input.bookings[buyer as usize * FREIGHT_BOOKING_WORDS + 1],
                );
                if fits(&*input, &previous, amount, seller) {
                    selected.replace(&previous);
                    start = end;
                }
            }
            let tested_carrier = select(
                booking == comptime!(BookingState::Pending as u32)
                    && amount.extract(HIGH_WORD) == 0
                    && request.animal_days <= input.config.ring
                    && preferred.carrier < input.config.railway.company_count,
                preferred.carrier,
                u32::MAX,
            );
            if tested_carrier < input.config.railway.company_count
                && fits(&*input, &preferred, amount, seller)
            {
                selected.replace(&preferred);
                start = end;
            }
            for entry in start..end {
                let company = input.config.railway.station_companies[entry as usize];
                let path = request.candidate(&input.config, company);
                if company != tested_carrier
                    && path.carrier < input.config.railway.company_count
                    && path.preferred_to(&selected)
                    && fits(&*input, &path, amount, seller)
                {
                    selected.replace(&path);
                }
            }
            input.bookings[buyer as usize * FREIGHT_BOOKING_WORDS] =
                comptime!(BookingState::Complete as u32);
            if selected.carrier < input.config.railway.company_count {
                input.bookings[buyer as usize * FREIGHT_BOOKING_WORDS] =
                    comptime!(BookingState::Ready as u32);
                input.bookings[buyer as usize * FREIGHT_BOOKING_WORDS + 1] = selected.carrier;
                let cost = multiply(&input.control, amount, selected.fee);
                claim(&*input, seller, cost);
                let mut cursor = RegionCursor::new(&selected);
                let mut record = cursor.next(&input.config);
                while record < input.config.railway.region_count {
                    claim(&*input, input.config.actors + record, amount);
                    record = cursor.next(&input.config);
                }
            }
        }
    }
}

#[cube]
fn toll(input: &Input, carrier: u32, owner: u32, units: u32, quantity: Ticks) {
    if owner < input.config.railway.company_count {
        let companies = &input.config.railway.companies;
        if companies[owner as usize * COMPANY_WORDS + 3]
            != companies[carrier as usize * COMPANY_WORDS + 3]
        {
            let fee = max(units, 1) * companies[owner as usize * COMPANY_WORDS + 2];
            account(
                input,
                companies[owner as usize * COMPANY_WORDS],
                multiply(&input.control, quantity, fee),
                comptime!(Account::Revenue),
            );
        }
    }
}

#[cube]
fn credit_tolls(input: &Input, path: &FreightRoute, quantity: Ticks) {
    let config = &input.config;
    let mut node = path.destination;
    let mut owner = u32::new(comptime!(i64::from(u32::MAX)));
    let mut region = u32::new(comptime!(i64::from(u32::MAX)));
    let mut units = 0u32;
    let mut steps = 0u32;
    while node != path.origin && steps < config.railway.nodes {
        let edge = config.railway.routes[route_cell(config, path.carrier, path.origin, node) + 1];
        if edge < config.railway.segment_count {
            let base = edge as usize * SEGMENT_WORDS;
            let next_owner = config.railway.segments[base + 2];
            if next_owner != owner {
                toll(input, path.carrier, owner, units, quantity);
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
    toll(input, path.carrier, owner, units, quantity);
}

#[cube]
pub(super) fn commit(input: &mut Input, invocation: u32) {
    if invocation < active_bookings(&*input) {
        let index = booking_priority(&*input, invocation);
        let buyer = booking_buyer(&*input, index);
        if input.bookings[buyer as usize * FREIGHT_BOOKING_WORDS]
            == comptime!(BookingState::Ready as u32)
        {
            let seller =
                input.proposals[buyer as usize * ProposalLayout::WORDS + ProposalLayout::ACTOR];
            let path = candidate(
                &ActorTable::new(&input.actors),
                &input.config,
                buyer,
                seller,
                input.bookings[buyer as usize * FREIGHT_BOOKING_WORDS + 1],
            );
            let amount = quantity(&*input, buyer);
            let allowed = input.accounts[buyer as usize * ACCOUNT_WORDS + DENIED].load() == 0;
            input.bookings[buyer as usize * FREIGHT_BOOKING_WORDS] =
                comptime!(BookingState::Retrying as u32);
            if allowed {
                let good = input.parameters.traded_good();
                let animal = animal_route(
                    &ActorTable::new(&input.actors),
                    &input.config,
                    buyer,
                    seller,
                );
                let base = BASE_FIELDS as u32 + (good - 1) * input.config.ring;
                let old_field = cartage::waiting_field(&input.config, good, animal.days);
                let mut ledger = Ledger::new(&mut input.state, &input.config);
                let payable = ledger.value(Field::CartagePayable, buyer);
                let waived = multiply(
                    &input.control,
                    amount,
                    cartage::rate(&input.config, animal.days),
                );
                ledger.set(Field::CartagePayable, buyer, minus(payable, waived));
                let waiting = ledger.value(Field::RoadFreight, buyer);
                ledger.set(Field::RoadFreight, buyer, minus(waiting, amount));
                let new_field = base + (input.parameters.day + path.days) % input.config.ring;
                let old = ledger.value_at(old_field, buyer);
                let new = add_checked(&input.control, ledger.value_at(new_field, buyer), amount);
                ledger.set_at(old_field, buyer, minus(old, amount));
                ledger.set_at(new_field, buyer, new);
                let cost = multiply(&input.control, amount, path.fee);
                consume_resource(&*input, seller, cost);
                account(&*input, seller, cost, comptime!(Account::Cost));
                let mut cursor = RegionCursor::new(&path);
                let mut record = cursor.next(&input.config);
                let mut units = 0u32;
                while record < input.config.railway.region_count {
                    consume_resource(&*input, input.config.actors + record, amount);
                    let owner = input.config.railway.regions[record as usize * REGION_WORDS];
                    let firm = input.config.railway.companies[owner as usize * COMPANY_WORDS];
                    account(&*input, firm, amount, comptime!(Account::Sold));
                    units += 1;
                    record = cursor.next(&input.config);
                }
                let total = multiply(&input.control, amount, units);
                let carrier = input.config.railway.companies[path.carrier as usize * COMPANY_WORDS];
                let rate =
                    input.config.railway.companies[path.carrier as usize * COMPANY_WORDS + 1];
                account(&*input, seller, total, comptime!(Account::Used));
                account(
                    &*input,
                    carrier,
                    multiply(&input.control, total, rate),
                    comptime!(Account::Revenue),
                );
                credit_tolls(&*input, &path, amount);
                input.bookings[buyer as usize * FREIGHT_BOOKING_WORDS] =
                    comptime!(BookingState::Complete as u32);
            } else {
                match comptime!(input.execution.wave()) {
                    Wave::Initial => {
                        let next = input.progress[comptime!(Wave::Retry as usize)].fetch_add(1);
                        input.queue[(input.config.actors + next) as usize] = index;
                    }
                    _ => {
                        input.bookings[buyer as usize * FREIGHT_BOOKING_WORDS] =
                            comptime!(BookingState::Complete as u32)
                    }
                }
            }
        }
    }
}

#[cube]
pub(super) fn initialize_booking(input: &mut Input, invocation: u32) {
    if invocation < markets::active(&input.control, &input.parameters) {
        let buyer = booking_buyer(&*input, invocation);
        let status = KernelEventStatus::load(
            &mut input.scratch,
            4 * input.parameters.count + buyer - input.parameters.offset,
        );
        input.bookings[buyer as usize * FREIGHT_BOOKING_WORDS] = select(
            status.contains(comptime!(EventStatus::Accepted)),
            comptime!(BookingState::Pending as u32),
            comptime!(BookingState::Complete as u32),
        );
        if status.contains(comptime!(EventStatus::Accepted)) {
            let slot = input.progress[comptime!(Wave::Initial as usize)].fetch_add(1);
            input.queue[slot as usize] = invocation;
        }
    }
}

#[cube]
pub(super) fn settle_account(input: &mut Input, id: u32) {
    let mut ledger = Ledger::new(&mut input.state, &input.config);
    let actors = input.config.actors;
    if id < actors {
        let offset = id as usize * ACCOUNT_WORDS;
        let cost = atomic_ticks(&input.accounts, offset + comptime!(Account::Cost as usize));
        let revenue = atomic_ticks(
            &input.accounts,
            offset + comptime!(Account::Revenue as usize),
        );
        let used = atomic_ticks(&input.accounts, offset + comptime!(Account::Used as usize));
        let sold = atomic_ticks(&input.accounts, offset + comptime!(Account::Sold as usize));
        let cash = ledger.value(Field::Cash, id);
        let cash = add_checked(&input.control, minus(cash, cost), revenue);
        ledger.set(Field::Cash, id, cash);
        ledger.increase(&input.control, Field::RailRevenue, id, revenue);
        record_use(&mut ledger, &input.control, id, used);
        let stock = ledger.value(Field::Railway, id);
        ledger.set(Field::Railway, id, minus(stock, sold));
        ledger.increase(&input.control, Field::Sales, id, sold);
    }
}

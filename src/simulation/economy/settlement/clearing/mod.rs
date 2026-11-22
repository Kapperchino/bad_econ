use crate::simulation::configuration::{EconomyConfig, Feature};
use crate::simulation::economy::*;
use crate::simulation::scheduler::settle;

#[cfg(test)]
mod tests;

#[cube]
pub(in crate::simulation::economy) fn actor_settlement(
    economy: &mut EconomyView<'_>,
    proposals: &mut [u32],
    scratch: &mut [u32],
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
    id: u32,
    #[comptime] posting: Posting,
) {
    let mut invocation = id;
    if comptime!(parameters.discovery == Discovery::Indexed) {
        invocation = parameters.count;
        if id < control[CONSTRUCTED_EVENTS_WORD].load() {
            invocation = scratch[(parameters.sort_source + parameters.count + id) as usize];
        }
    }
    let id = invocation;
    if id < parameters.count {
        let actor = id + parameters.offset;
        let other = proposals[actor as usize * ProposalLayout::WORDS + ProposalLayout::ACTOR];
        let status = KernelEventStatus::load(scratch, 4 * parameters.count + id);
        let eligible = match comptime!(posting) {
            Posting::CheckActor => other != NO_ACTOR,
            _ => status.contains(comptime!(EventStatus::Accepted)),
        };
        if eligible {
            let status = match comptime!(parameters.market) {
                MarketFlag::Input(_) | MarketFlag::Consumer(_) | MarketFlag::Household => {
                    buyer_settlement(
                        economy, proposals, control, parameters, actor, other, posting,
                    )
                }
                _ => event_settlement(economy, proposals, control, parameters, id, posting),
            };
            status.store(scratch, 4 * parameters.count + id);
        }
    }
}

#[cube]
fn event_settlement(
    economy: &mut EconomyView<'_>,
    proposals: &mut [u32],
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
    id: u32,
    #[comptime] posting: Posting,
) -> KernelEventStatus {
    let config = economy.ledger.config;
    let mut event = Array::<u32>::new(EVENT_WORDS);
    construct(
        economy,
        proposals,
        &mut event[..],
        control,
        parameters,
        id,
        comptime!(EventStorage::Local),
    );
    settle(
        economy.ledger.words,
        &event[..],
        control,
        0,
        id + parameters.offset,
        config.state_stride,
        posting,
    )
}

#[cube]
pub(in crate::simulation::economy) fn buyer_settlement(
    economy: &mut EconomyView<'_>,
    proposals: &mut [u32],
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
    actor: u32,
    other: u32,
    #[comptime] posting: Posting,
) -> KernelEventStatus {
    let config = economy.ledger.config;
    let good = parameters.traded_good();
    let quantity = ticks(
        proposals[actor as usize * ProposalLayout::WORDS + ProposalLayout::QUANTITY_LOW],
        proposals[actor as usize * ProposalLayout::WORDS + ProposalLayout::QUANTITY_HIGH],
    );
    let terms = household_production::TradeFields::new(&economy.actors, other, good);
    let price = economy.ledger.value_at(terms.price, other);
    let cost = times(quantity, price.extract(LOW_WORD)).record(control);
    let retained = household_production::reserve(&*economy, other, good);
    let _ = plus(retained, quantity).record(control);
    let route = railway::FreightRoute::by_road(&economy.actors, config, actor, other);
    let trip = economy.actors.distance(actor, other);
    let slot = cartage::booking_field(config, good, parameters.day, route.days, trip);
    let cash = economy.ledger.value(Field::Cash, actor);
    let incoming = plus(economy.ledger.value_at(slot, actor), quantity);
    let transit_good = plus(
        economy
            .ledger
            .value_at(TRANSIT_FIELDS as u32 + good - 1, actor),
        quantity,
    );
    let transit = plus(economy.ledger.value(Field::Transit, actor), quantity);
    let purchased = plus(economy.ledger.value(Field::Purchased, actor), quantity);
    let mut faults = incoming
        .faults
        .union(transit_good.faults)
        .union(transit.faults)
        .union(purchased.faults);
    let mut waiting = zero();
    let mut payable = zero();
    if trip > 0 {
        let freight = times(quantity, cartage::rate(config, route.days)).record(control);
        let orders = plus(economy.ledger.value(Field::RoadFreight, actor), quantity);
        let fees = plus(economy.ledger.value(Field::CartagePayable, actor), freight);
        faults.insert(orders.faults.union(fees.faults));
        waiting = orders.value;
        payable = fees.value;
    }
    let mut status = KernelEventStatus::new(comptime!(EventStatus::Rejected));
    if !lt(cash, cost) {
        status.set(comptime!(EventStatus::Accepted));
    }
    if !faults.is_empty() {
        faults.record(control);
        status.set(comptime!(EventStatus::Faulted));
    }
    if comptime!(posting == Posting::ApplyActor)
        && status.contains(comptime!(EventStatus::Accepted))
    {
        economy.ledger.set(Field::Cash, actor, minus(cash, cost));
        economy.ledger.set_at(slot, actor, incoming.value);
        if trip > 0 {
            economy.ledger.set(Field::CartagePayable, actor, payable);
            economy.ledger.set(Field::RoadFreight, actor, waiting);
        }
        if good == comptime!(Good::Bread as u32)
            && economy
                .actors
                .kind(actor)
                .is(comptime!(ActorKind::Household))
        {
            economy.ledger.set(Field::FoodPrice, actor, price);
        }
        for input in 0u32..2 {
            if economy
                .ledger
                .number_at(comptime!(Field::FirstInput as u32) + input, actor)
                == good
            {
                economy.ledger.set_at(
                    comptime!(Field::FirstInputCost as u32) + input,
                    actor,
                    price,
                );
            }
        }
        if economy.ledger.number(Field::EnterpriseModel, actor) == 1 {
            economy.ledger.set(Field::ReplacementCost, actor, price);
        }
        economy
            .ledger
            .set_at(TRANSIT_FIELDS as u32 + good - 1, actor, transit_good.value);
        economy.ledger.set(Field::Transit, actor, transit.value);
        economy.ledger.set(Field::Purchased, actor, purchased.value);
    }
    status
}

#[cube]
fn empty_seller_bulk(config: &EconomyConfig, parameters: &EconomyParameters) -> bool {
    let good = parameters.traded_good();
    comptime!(config.cohorts == Feature::Enabled)
        || good == comptime!(Good::Coal as u32)
        || good == comptime!(Good::Wood as u32)
        || good == comptime!(Good::LuxuryClothes as u32)
        || good == comptime!(Good::LuxuryFurniture as u32)
}

#[cube]
fn empty_seller_quantity(
    config: &EconomyConfig,
    proposals: &[u32],
    parameters: &EconomyParameters,
    actor: u32,
) -> Ticks {
    let base = actor as usize * ProposalLayout::WORDS;
    select(
        empty_seller_bulk(config, parameters),
        ticks(
            proposals[base + ProposalLayout::QUANTITY_LOW],
            proposals[base + ProposalLayout::QUANTITY_HIGH],
        ),
        ticks(1, 0),
    )
}

#[cube]
pub(in crate::simulation::economy) fn empty_seller_safe(
    ledger: &Ledger<'_>,
    parameters: &EconomyParameters,
    peer: u32,
) -> bool {
    let config = ledger.config;
    let quantity = select(
        empty_seller_bulk(config, parameters),
        ticks(u32::MAX, 0),
        ticks(1, 0),
    );
    let quote = ledger.number(Field::Price, peer);
    let cost = times(quantity, quote);
    let cash = plus(ledger.value(Field::Cash, peer), cost.value);
    let sales = plus(ledger.value(Field::Sales, peer), quantity);
    cost.faults
        .union(cash.faults)
        .union(sales.faults)
        .is_empty()
}

#[cube]
pub(in crate::simulation::economy) fn empty_seller_settlement(
    economy: &mut EconomyView<'_>,
    proposals: &[u32],
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
    actor: u32,
    peer: u32,
    prior: &KernelEventStatus,
) -> KernelEventStatus {
    let config = economy.ledger.config;
    let quantity = empty_seller_quantity(config, proposals, parameters, actor);
    let terms =
        household_production::TradeFields::new(&economy.actors, peer, parameters.traded_good());
    let quote = economy.ledger.number_at(terms.price, peer);
    let cost = times(quantity, quote).record(control);
    let cash = plus(economy.ledger.value(Field::Cash, peer), cost);
    let sales = plus(economy.ledger.value_at(terms.sales, peer), quantity);
    let faults = cash.faults.union(sales.faults);
    let mut status = KernelEventStatus::new(comptime!(EventStatus::Rejected));
    if !faults.is_empty() || prior.contains(comptime!(EventStatus::Faulted)) {
        faults.record(control);
        status.set(comptime!(EventStatus::Faulted));
    }
    status
}

#[derive(CubeType, Clone, Copy)]
pub(in crate::simulation::economy) struct PeerOffer {
    pub actor: u32,
    pub peer: u32,
    pub status: KernelEventStatus,
}

#[derive(CubeType, Clone, Copy)]
pub(in crate::simulation::economy) struct PeerBalances {
    cash: Ticks,
    escrow: Ticks,
    slots: Ticks,
    second_stock: Ticks,
    first_used: Ticks,
    second_used: Ticks,
    stock: Ticks,
    pipeline: Ticks,
    used: Ticks,
    sales: Ticks,
    quote: u32,
    reserve: Ticks,
}

#[cube]
impl PeerBalances {
    pub fn has_supply(&self) -> bool {
        lt(self.reserve, self.stock)
    }

    pub fn new(economy: &mut EconomyView<'_>, parameters: &EconomyParameters, peer: u32) -> Self {
        let mut balances = PeerBalances {
            cash: zero(),
            escrow: zero(),
            slots: zero(),
            second_stock: zero(),
            first_used: zero(),
            second_used: zero(),
            stock: zero(),
            pipeline: zero(),
            used: zero(),
            sales: zero(),
            quote: 0,
            reserve: zero(),
        };
        match comptime!(parameters.market) {
            MarketFlag::Labor => {
                balances.cash = economy.ledger.value(Field::Cash, peer);
                balances.escrow = economy.ledger.value(Field::Escrow, peer);
                balances.slots = economy.ledger.value(Field::Slots, peer);
                balances.pipeline = economy.ledger.value(Field::Pipeline, peer);
                balances.quote = economy.ledger.number(Field::Wage, peer);
                balances.used = economy.ledger.value(Field::Used, peer);
                let first = economy.ledger.number(Field::FirstInput, peer);
                let second = economy.ledger.number(Field::SecondInput, peer);
                if first > 0 {
                    balances.stock = economy.ledger.value_at(first, peer);
                    balances.first_used = economy
                        .ledger
                        .value_at(USED_FIELDS as u32 + first - 1, peer);
                }
                if second > 0 {
                    balances.second_stock = economy.ledger.value_at(second, peer);
                    balances.second_used = economy
                        .ledger
                        .value_at(USED_FIELDS as u32 + second - 1, peer);
                }
            }
            MarketFlag::Payroll | MarketFlag::Distribution => (),
            _ => {
                let good = parameters.traded_good();
                balances.cash = economy.ledger.value(Field::Cash, peer);
                balances.stock = economy.ledger.value_at(good, peer);
                balances.reserve = household_production::reserve(&*economy, peer, good);
                let terms = household_production::TradeFields::new(&economy.actors, peer, good);
                balances.sales = economy.ledger.value_at(terms.sales, peer);
                balances.quote = economy.ledger.number_at(terms.price, peer);
            }
        }
        balances
    }

    pub fn offer(
        &mut self,
        ledger: &mut Ledger<'_>,
        proposals: &mut [u32],
        control: &[Atomic<u32>],
        parameters: &EconomyParameters,
        offer: PeerOffer,
    ) -> KernelEventStatus {
        let config = ledger.config;
        let actor = offer.actor;
        let peer = offer.peer;
        let prior = offer.status;
        let quantity = ticks(
            proposals[actor as usize * ProposalLayout::WORDS + ProposalLayout::QUANTITY_LOW],
            proposals[actor as usize * ProposalLayout::WORDS + ProposalLayout::QUANTITY_HIGH],
        );
        let mut valid = prior.contains(comptime!(EventStatus::Accepted));
        let mut faults = KernelFaults::none();
        let mut next_cash = self.cash;
        let mut next_escrow = self.escrow;
        let mut next_slots = self.slots;
        let mut next_stock = self.stock;
        let mut next_second_stock = self.second_stock;
        let mut next_first_used = self.first_used;
        let mut next_second_used = self.second_used;
        let mut next_pipeline = self.pipeline;
        let mut next_used = self.used;
        let mut next_sales = self.sales;
        match comptime!(parameters.market) {
            MarketFlag::Labor => {
                let mut shifts = employment::shifts(config, quantity);
                let mut rate = self.quote;
                if comptime!(config.bargaining.enabled == Feature::Enabled) {
                    rate = quantity.extract(LOW_WORD);
                }
                let labor = ledger.number(Field::LaborMinutes, peer);
                let first = ledger.number(Field::FirstInput, peer);
                let second = ledger.number(Field::SecondInput, peer);
                let unit_input = ledger.value(Field::InputTicks, peer);
                if comptime!(config.cohorts == Feature::Enabled) {
                    shifts = minimum(ticks(shifts, 0), self.slots).extract(LOW_WORD);
                    shifts = batch::bounded_capacity(self.cash, times32(rate, labor), shifts);
                    shifts = batch::bounded_capacity(
                        self.stock,
                        select(first > 0, unit_input, zero()),
                        shifts,
                    );
                    shifts = batch::bounded_capacity(
                        self.second_stock,
                        select(second > 0, unit_input, zero()),
                        shifts,
                    );
                    proposals
                        [actor as usize * ProposalLayout::WORDS + ProposalLayout::QUANTITY_HIGH] =
                        shifts;
                    valid = valid && shifts > 0;
                }
                let wage = times32(rate, labor * shifts);
                let output = times32(ledger.number(Field::OutputTicks, peer), shifts);
                let escrow = plus(self.escrow, wage);
                let pipeline = plus(self.pipeline, output);
                faults.insert(escrow.faults.union(pipeline.faults));
                valid = valid && !lt(self.cash, wage) && !lt(self.slots, ticks(shifts, 0));
                next_cash = minus(self.cash, wage);
                next_escrow = escrow.value;
                next_slots = minus(self.slots, ticks(shifts, 0));
                next_pipeline = pipeline.value;
                let scaled_input = times(unit_input, shifts);
                faults.insert(scaled_input.faults);
                let input = scaled_input.value;
                let mut total_input = zero();
                if first > 0 {
                    let used = plus(self.first_used, input);
                    faults.insert(used.faults);
                    valid = valid && !lt(self.stock, input);
                    next_stock = minus(self.stock, input);
                    next_first_used = used.value;
                    total_input = input;
                }
                if second > 0 {
                    let used = plus(self.second_used, input);
                    faults.insert(used.faults);
                    valid = valid && !lt(self.second_stock, input);
                    next_second_stock = minus(self.second_stock, input);
                    next_second_used = used.value;
                    total_input = plus(total_input, input).record(control);
                }
                let used = plus(self.used, total_input);
                faults.insert(used.faults);
                next_used = used.value;
            }
            MarketFlag::Payroll | MarketFlag::Distribution => (),
            _ => {
                let cost = times(quantity, self.quote).record(control);
                let cash = plus(self.cash, cost);
                let sales = plus(self.sales, quantity);
                faults.insert(cash.faults.union(sales.faults));
                valid = valid && !lt(positive(self.stock, self.reserve), quantity);
                next_cash = cash.value;
                next_stock = minus(self.stock, quantity);
                next_sales = sales.value;
            }
        }
        let mut status = KernelEventStatus::new(comptime!(EventStatus::Rejected));
        if valid {
            status.set(comptime!(EventStatus::Accepted));
        }
        if !faults.is_empty() || prior.contains(comptime!(EventStatus::Faulted)) {
            faults.record(control);
            status.set(comptime!(EventStatus::Faulted));
        }
        if status.contains(comptime!(EventStatus::Accepted)) {
            self.cash = next_cash;
            self.escrow = next_escrow;
            self.slots = next_slots;
            self.stock = next_stock;
            self.second_stock = next_second_stock;
            self.first_used = next_first_used;
            self.second_used = next_second_used;
            self.pipeline = next_pipeline;
            self.used = next_used;
            self.sales = next_sales;
        }
        status
    }

    pub fn store(&self, economy: &mut EconomyView<'_>, parameters: &EconomyParameters, peer: u32) {
        match comptime!(parameters.market) {
            MarketFlag::Labor => {
                economy.ledger.set(Field::Cash, peer, self.cash);
                economy.ledger.set(Field::Escrow, peer, self.escrow);
                economy.ledger.set(Field::Slots, peer, self.slots);
                economy.ledger.set(Field::Pipeline, peer, self.pipeline);
                let first = economy.ledger.number(Field::FirstInput, peer);
                let second = economy.ledger.number(Field::SecondInput, peer);
                economy.ledger.set(Field::Used, peer, self.used);
                if first > 0 {
                    economy.ledger.set_at(first, peer, self.stock);
                    economy
                        .ledger
                        .set_at(USED_FIELDS as u32 + first - 1, peer, self.first_used);
                }
                if second > 0 {
                    economy.ledger.set_at(second, peer, self.second_stock);
                    economy
                        .ledger
                        .set_at(USED_FIELDS as u32 + second - 1, peer, self.second_used);
                }
            }
            MarketFlag::Payroll | MarketFlag::Distribution => (),
            _ => {
                let good = parameters.traded_good();
                economy.ledger.set(Field::Cash, peer, self.cash);
                economy.ledger.set_at(good, peer, self.stock);
                economy.ledger.set_at(
                    household_production::TradeFields::new(&economy.actors, peer, good).sales,
                    peer,
                    self.sales,
                );
            }
        }
    }
}

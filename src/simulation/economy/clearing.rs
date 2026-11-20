use super::*;
use crate::simulation::configuration::{EconomyConfig, Feature};
use crate::simulation::scheduler::settle;

#[cfg(test)]
mod tests;

#[cube]
pub(super) fn actor_settlement(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
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
                        state, actors, config, proposals, control, parameters, actor, other,
                        posting,
                    )
                }
                _ => event_settlement(
                    state, actors, config, proposals, control, parameters, id, posting,
                ),
            };
            status.store(scratch, 4 * parameters.count + id);
        }
    }
}

#[cube]
fn event_settlement(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    proposals: &mut [u32],
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
    id: u32,
    #[comptime] posting: Posting,
) -> KernelEventStatus {
    let mut event = Array::<u32>::new(EVENT_WORDS);
    construct(
        state,
        actors,
        config,
        proposals,
        &mut event[..],
        control,
        parameters,
        id,
        comptime!(EventStorage::Local),
    );
    settle(
        state,
        &event[..],
        control,
        0,
        id + parameters.offset,
        config.state_stride,
        posting,
    )
}

#[cube]
pub(super) fn buyer_settlement(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    proposals: &mut [u32],
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
    actor: u32,
    other: u32,
    #[comptime] posting: Posting,
) -> KernelEventStatus {
    let good = parameters.traded_good();
    let quantity = ticks(
        proposals[actor as usize * ProposalLayout::WORDS + ProposalLayout::QUANTITY_LOW],
        proposals[actor as usize * ProposalLayout::WORDS + ProposalLayout::QUANTITY_HIGH],
    );
    let price = value(state, config, comptime!(Field::Price as u32), other);
    let cost = multiply(control, quantity, price.extract(LOW_WORD));
    let retained = household_production::reserve(state, actors, config, other, good);
    let _ = add_checked(control, retained, quantity);
    let route = railway::animal_route(actors, config, actor, other);
    let slot =
        BASE_FIELDS as u32 + (good - 1) * config.ring + (parameters.day + route.days) % config.ring;
    let cash = value(state, config, comptime!(Field::Cash as u32), actor);
    let incoming = plus(value(state, config, slot, actor), quantity);
    let transit_good = plus(
        value(state, config, TRANSIT_FIELDS as u32 + good - 1, actor),
        quantity,
    );
    let transit = plus(
        value(state, config, comptime!(Field::Transit as u32), actor),
        quantity,
    );
    let purchased = plus(
        value(state, config, comptime!(Field::Purchased as u32), actor),
        quantity,
    );
    let faults = incoming
        .faults
        .union(transit_good.faults)
        .union(transit.faults)
        .union(purchased.faults);
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
        set(
            state,
            config,
            comptime!(Field::Cash as u32),
            actor,
            minus(cash, cost),
        );
        set(state, config, slot, actor, incoming.value);
        if good == comptime!(Good::Bread as u32)
            && kind(actors, actor).is(comptime!(ActorKind::Household))
        {
            set(
                state,
                config,
                comptime!(Field::FoodPrice as u32),
                actor,
                price,
            );
        }
        for input in 0u32..2 {
            if number(
                state,
                config,
                comptime!(Field::FirstInput as u32) + input,
                actor,
            ) == good
            {
                set(
                    state,
                    config,
                    comptime!(Field::FirstInputCost as u32) + input,
                    actor,
                    price,
                );
            }
        }
        if number(
            state,
            config,
            comptime!(Field::EnterpriseModel as u32),
            actor,
        ) == 1
        {
            set(
                state,
                config,
                comptime!(Field::ReplacementCost as u32),
                actor,
                price,
            );
        }
        set(
            state,
            config,
            TRANSIT_FIELDS as u32 + good - 1,
            actor,
            transit_good.value,
        );
        set(
            state,
            config,
            comptime!(Field::Transit as u32),
            actor,
            transit.value,
        );
        set(
            state,
            config,
            comptime!(Field::Purchased as u32),
            actor,
            purchased.value,
        );
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
pub(super) fn empty_seller_safe(
    state: &mut [u32],
    config: &EconomyConfig,
    parameters: &EconomyParameters,
    peer: u32,
) -> bool {
    let quantity = select(
        empty_seller_bulk(config, parameters),
        ticks(u32::MAX, 0),
        ticks(1, 0),
    );
    let quote = number(state, config, comptime!(Field::Price as u32), peer);
    let cost = times(quantity, quote);
    let cash = plus(
        value(state, config, comptime!(Field::Cash as u32), peer),
        cost.value,
    );
    let sales = plus(
        value(state, config, comptime!(Field::Sales as u32), peer),
        quantity,
    );
    cost.faults
        .union(cash.faults)
        .union(sales.faults)
        .is_empty()
}

#[cube]
pub(super) fn empty_seller_settlement(
    state: &mut [u32],
    _actors: &[u32],
    config: &EconomyConfig,
    proposals: &[u32],
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
    actor: u32,
    peer: u32,
    prior: &KernelEventStatus,
) -> KernelEventStatus {
    let quantity = empty_seller_quantity(config, proposals, parameters, actor);
    let quote = number(state, config, comptime!(Field::Price as u32), peer);
    let cost = multiply(control, quantity, quote);
    let cash = plus(
        value(state, config, comptime!(Field::Cash as u32), peer),
        cost,
    );
    let sales = plus(
        value(state, config, comptime!(Field::Sales as u32), peer),
        quantity,
    );
    let faults = cash.faults.union(sales.faults);
    let mut status = KernelEventStatus::new(comptime!(EventStatus::Rejected));
    if !faults.is_empty() || prior.contains(comptime!(EventStatus::Faulted)) {
        faults.record(control);
        status.set(comptime!(EventStatus::Faulted));
    }
    status
}

#[derive(CubeType, Clone, Copy)]
pub(super) struct PeerOffer {
    pub actor: u32,
    pub peer: u32,
    pub status: KernelEventStatus,
}

#[derive(CubeType, Clone, Copy)]
pub(super) struct PeerBalances {
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

    pub fn new(
        state: &mut [u32],
        actors: &[u32],
        config: &EconomyConfig,
        parameters: &EconomyParameters,
        peer: u32,
    ) -> Self {
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
                balances.cash = value(state, config, comptime!(Field::Cash as u32), peer);
                balances.escrow = value(state, config, comptime!(Field::Escrow as u32), peer);
                balances.slots = value(state, config, comptime!(Field::Slots as u32), peer);
                balances.pipeline = value(state, config, comptime!(Field::Pipeline as u32), peer);
                balances.quote = number(state, config, comptime!(Field::Wage as u32), peer);
                balances.used = value(state, config, comptime!(Field::Used as u32), peer);
                let first = number(state, config, comptime!(Field::FirstInput as u32), peer);
                let second = number(state, config, comptime!(Field::SecondInput as u32), peer);
                if first > 0 {
                    balances.stock = value(state, config, first, peer);
                    balances.first_used =
                        value(state, config, USED_FIELDS as u32 + first - 1, peer);
                }
                if second > 0 {
                    balances.second_stock = value(state, config, second, peer);
                    balances.second_used =
                        value(state, config, USED_FIELDS as u32 + second - 1, peer);
                }
            }
            MarketFlag::Payroll | MarketFlag::Distribution => (),
            _ => {
                let good = parameters.traded_good();
                balances.cash = value(state, config, comptime!(Field::Cash as u32), peer);
                balances.stock = value(state, config, good, peer);
                balances.reserve = household_production::reserve(state, actors, config, peer, good);
                balances.sales = value(state, config, comptime!(Field::Sales as u32), peer);
                balances.quote = number(state, config, comptime!(Field::Price as u32), peer);
            }
        }
        balances
    }

    pub fn offer(
        &mut self,
        state: &mut [u32],
        config: &EconomyConfig,
        proposals: &mut [u32],
        control: &[Atomic<u32>],
        parameters: &EconomyParameters,
        offer: PeerOffer,
    ) -> KernelEventStatus {
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
                let labor = number(state, config, comptime!(Field::LaborMinutes as u32), peer);
                let first = number(state, config, comptime!(Field::FirstInput as u32), peer);
                let second = number(state, config, comptime!(Field::SecondInput as u32), peer);
                let unit_input = value(state, config, comptime!(Field::InputTicks as u32), peer);
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
                let output = times32(
                    number(state, config, comptime!(Field::OutputTicks as u32), peer),
                    shifts,
                );
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
                    total_input = add_checked(control, total_input, input);
                }
                let used = plus(self.used, total_input);
                faults.insert(used.faults);
                next_used = used.value;
            }
            MarketFlag::Payroll | MarketFlag::Distribution => (),
            _ => {
                let cost = multiply(control, quantity, self.quote);
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

    pub fn store(
        &self,
        state: &mut [u32],
        _actors: &[u32],
        config: &EconomyConfig,
        parameters: &EconomyParameters,
        peer: u32,
    ) {
        match comptime!(parameters.market) {
            MarketFlag::Labor => {
                set(
                    state,
                    config,
                    comptime!(Field::Cash as u32),
                    peer,
                    self.cash,
                );
                set(
                    state,
                    config,
                    comptime!(Field::Escrow as u32),
                    peer,
                    self.escrow,
                );
                set(
                    state,
                    config,
                    comptime!(Field::Slots as u32),
                    peer,
                    self.slots,
                );
                set(
                    state,
                    config,
                    comptime!(Field::Pipeline as u32),
                    peer,
                    self.pipeline,
                );
                let first = number(state, config, comptime!(Field::FirstInput as u32), peer);
                let second = number(state, config, comptime!(Field::SecondInput as u32), peer);
                set(
                    state,
                    config,
                    comptime!(Field::Used as u32),
                    peer,
                    self.used,
                );
                if first > 0 {
                    set(state, config, first, peer, self.stock);
                    set(
                        state,
                        config,
                        USED_FIELDS as u32 + first - 1,
                        peer,
                        self.first_used,
                    );
                }
                if second > 0 {
                    set(state, config, second, peer, self.second_stock);
                    set(
                        state,
                        config,
                        USED_FIELDS as u32 + second - 1,
                        peer,
                        self.second_used,
                    );
                }
            }
            MarketFlag::Payroll | MarketFlag::Distribution => (),
            _ => {
                let good = parameters.traded_good();
                set(
                    state,
                    config,
                    comptime!(Field::Cash as u32),
                    peer,
                    self.cash,
                );
                set(state, config, good, peer, self.stock);
                set(
                    state,
                    config,
                    comptime!(Field::Sales as u32),
                    peer,
                    self.sales,
                );
            }
        }
    }
}

use super::*;
use crate::simulation::configuration::{EconomyConfig, Feature};
use crate::simulation::scheduler::settle;

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
    if id < parameters.count {
        let actor = id + parameters.offset;
        let other = proposals[actor as usize * ProposalLayout::WORDS + ProposalLayout::ACTOR];
        let status = KernelEventStatus::load(scratch, 4 * parameters.count + id);
        let eligible = match comptime!(posting) {
            Posting::CheckActor => other != NO_ACTOR,
            _ => status.contains(comptime!(EventStatus::Accepted)),
        };
        if eligible {
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
            settle(state, &event[..], control, 0, actor, config.actors, posting)
                .store(scratch, 4 * parameters.count + id);
        }
    }
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
                let good = comptime!(parameters.market.good().expect("goods market") as u32);
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
                let mut rate = self.quote;
                if comptime!(config.bargaining.enabled == Feature::Enabled) {
                    rate = quantity.extract(LOW_WORD);
                }
                let wage = times32(
                    rate,
                    number(state, config, comptime!(Field::LaborMinutes as u32), peer),
                );
                let output = ticks(
                    number(state, config, comptime!(Field::OutputTicks as u32), peer),
                    0,
                );
                let escrow = plus(self.escrow, wage);
                let pipeline = plus(self.pipeline, output);
                faults.insert(escrow.faults.union(pipeline.faults));
                valid = valid && !lt(self.cash, wage) && !lt(self.slots, ticks(1, 0));
                next_cash = minus(self.cash, wage);
                next_escrow = escrow.value;
                next_slots = minus(self.slots, ticks(1, 0));
                next_pipeline = pipeline.value;
                let input = value(state, config, comptime!(Field::InputTicks as u32), peer);
                let first = number(state, config, comptime!(Field::FirstInput as u32), peer);
                let second = number(state, config, comptime!(Field::SecondInput as u32), peer);
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
                let good = comptime!(parameters.market.good().expect("goods market") as u32);
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

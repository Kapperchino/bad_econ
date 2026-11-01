use super::*;
use crate::simulation::scheduler::settle;

#[cube]
pub(super) fn actor_settlement(
    state: &mut [u32],
    actors: &[u32],
    config: &[u32],
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
            settle(
                state,
                &event[..],
                control,
                0,
                actor,
                config[ConfigLayout::ACTORS],
                posting,
            )
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
    stock: Ticks,
    pipeline: Ticks,
    used: Ticks,
    sales: Ticks,
    quote: u32,
}

#[cube]
impl PeerBalances {
    pub fn new(
        state: &mut [u32],
        actors: &[u32],
        config: &[u32],
        parameters: &EconomyParameters,
        peer: u32,
    ) -> Self {
        let mut balances = PeerBalances {
            cash: zero(),
            escrow: zero(),
            slots: zero(),
            stock: zero(),
            pipeline: zero(),
            used: zero(),
            sales: zero(),
            quote: 0,
        };
        match comptime!(parameters.market) {
            MarketFlag::Labor => {
                balances.cash = value(state, config, comptime!(Field::Cash as u32), peer);
                balances.escrow = value(state, config, comptime!(Field::Escrow as u32), peer);
                balances.slots = value(state, config, comptime!(Field::Slots as u32), peer);
                balances.pipeline = value(state, config, comptime!(Field::Pipeline as u32), peer);
                balances.quote = number(state, config, comptime!(Field::Wage as u32), peer);
                if kind(actors, peer).processes_input() {
                    balances.stock = value(state, config, kind(actors, peer).input(), peer);
                    balances.used = value(state, config, comptime!(Field::Used as u32), peer);
                }
            }
            MarketFlag::Payroll | MarketFlag::Distribution => (),
            _ => {
                let good = comptime!(parameters.market.good().expect("goods market") as u32);
                balances.cash = value(state, config, comptime!(Field::Cash as u32), peer);
                balances.stock = value(state, config, good, peer);
                balances.sales = value(state, config, comptime!(Field::Sales as u32), peer);
                balances.quote = number(state, config, comptime!(Field::Price as u32), peer);
            }
        }
        balances
    }

    pub fn offer(
        &mut self,
        actors: &[u32],
        config: &[u32],
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
        let mut next_pipeline = self.pipeline;
        let mut next_used = self.used;
        let mut next_sales = self.sales;
        match comptime!(parameters.market) {
            MarketFlag::Labor => {
                let mut rate = self.quote;
                if config[ConfigLayout::BARGAINING + PolicyLayout::ENABLED] != 0 {
                    rate = quantity.extract(LOW_WORD);
                }
                let wage = times32(
                    rate,
                    actors[ActorLayout::LABOR_MINUTES * (actors.len() / ActorLayout::WORDS)
                        + peer as usize],
                );
                let output = ticks(
                    actors[ActorLayout::OUTPUT_TICKS * (actors.len() / ActorLayout::WORDS)
                        + peer as usize],
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
                if kind(actors, peer).processes_input() {
                    let input = ticks(
                        actors[ActorLayout::INPUT_TICKS * (actors.len() / ActorLayout::WORDS)
                            + peer as usize],
                        0,
                    );
                    let used = plus(self.used, input);
                    faults.insert(used.faults);
                    valid = valid && !lt(self.stock, input);
                    next_stock = minus(self.stock, input);
                    next_used = used.value;
                }
            }
            MarketFlag::Payroll | MarketFlag::Distribution => (),
            _ => {
                let cost = multiply(control, quantity, self.quote);
                let cash = plus(self.cash, cost);
                let sales = plus(self.sales, quantity);
                faults.insert(cash.faults.union(sales.faults));
                valid = valid && !lt(self.stock, quantity);
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
            self.pipeline = next_pipeline;
            self.used = next_used;
            self.sales = next_sales;
        }
        status
    }

    pub fn store(
        &self,
        state: &mut [u32],
        actors: &[u32],
        config: &[u32],
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
                if kind(actors, peer).processes_input() {
                    set(state, config, kind(actors, peer).input(), peer, self.stock);
                    set(
                        state,
                        config,
                        comptime!(Field::Used as u32),
                        peer,
                        self.used,
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

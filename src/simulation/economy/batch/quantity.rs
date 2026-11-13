use super::*;

#[cfg(test)]
mod tests;

#[repr(u32)]
enum QuantityMode {
    Aggregate,
    Unavailable,
    Ordered,
}

#[derive(CubeType)]
struct QuantitySummary {
    total: Ticks,
    minimum: Ticks,
    ordered: u32,
}

#[cube]
fn summarize(quantity: Ticks, prior: &KernelEventStatus) -> QuantitySummary {
    let mut quantities = Shared::<[u64]>::new_slice(256usize);
    let mut minima = Shared::<[u64]>::new_slice(256usize);
    let mut flags = Shared::<[u32]>::new_slice(256usize);
    let lane = UNIT_POS as usize;
    let amount = u64::cast_from(quantity.extract(LOW_WORD))
        | (u64::cast_from(quantity.extract(HIGH_WORD)) << 32);
    let accepted = prior.contains(comptime!(EventStatus::Accepted));
    quantities[lane] = amount;
    minima[lane] = select(accepted && amount > 0, amount, u64::MAX);
    flags[lane] = u32::cast_from(!accepted || prior.contains(comptime!(EventStatus::Faulted)));
    sync_cube();
    let mut stride = CUBE_DIM / 2;
    while stride > 0 {
        if UNIT_POS < stride {
            let other = lane + stride as usize;
            let sum = quantities[lane] + quantities[other];
            flags[lane] |= flags[other] | u32::cast_from(sum < quantities[lane]);
            quantities[lane] = sum;
            minima[lane] = min(minima[lane], minima[other]);
        }
        sync_cube();
        stride /= 2;
    }
    let total = quantities[0usize];
    let minimum = minima[0usize];
    let result = QuantitySummary {
        total: ticks(total as u32, (total >> 32) as u32),
        minimum: ticks(minimum as u32, (minimum >> 32) as u32),
        ordered: flags[0usize],
    };
    sync_cube();
    result
}

#[derive(CubeType)]
struct QuantityBalances {
    cash: Ticks,
    stock: Ticks,
    sales: Ticks,
    reserve: Ticks,
    quote: u32,
}

#[cube]
impl QuantityBalances {
    fn new(
        state: &mut [u32],
        actors: &[u32],
        config: &EconomyConfig,
        peer: u32,
        good: u32,
    ) -> Self {
        QuantityBalances {
            cash: value(state, config, comptime!(Field::Cash as u32), peer),
            stock: value(state, config, good, peer),
            sales: value(state, config, comptime!(Field::Sales as u32), peer),
            reserve: household_production::reserve(state, actors, config, peer, good),
            quote: number(state, config, comptime!(Field::Price as u32), peer),
        }
    }

    fn mode(&self, summary: &QuantitySummary) -> u32 {
        let cost = times(summary.total, self.quote);
        let cash = plus(self.cash, cost.value);
        let sales = plus(self.sales, summary.total);
        let available = positive(self.stock, self.reserve);
        let funded = summary.ordered == 0
            && !lt(available, summary.total)
            && cost.faults.is_empty()
            && cash.faults.is_empty()
            && sales.faults.is_empty();
        select(
            funded,
            comptime!(QuantityMode::Aggregate as u32),
            select(
                lt(available, summary.minimum),
                comptime!(QuantityMode::Unavailable as u32),
                comptime!(QuantityMode::Ordered as u32),
            ),
        )
    }

    fn aggregate(&mut self, quantity: Ticks) {
        self.cash = plus(self.cash, times(quantity, self.quote).value).value;
        self.stock = minus(self.stock, quantity);
        self.sales = plus(self.sales, quantity).value;
    }

    fn offer(
        &mut self,
        quantity: Ticks,
        prior: KernelEventStatus,
        control: &[Atomic<u32>],
    ) -> KernelEventStatus {
        let cost = multiply(control, quantity, self.quote);
        let cash = plus(self.cash, cost);
        let sales = plus(self.sales, quantity);
        let faults = cash.faults.union(sales.faults);
        let mut status = KernelEventStatus::new(comptime!(EventStatus::Rejected));
        if prior.contains(comptime!(EventStatus::Accepted))
            && !lt(positive(self.stock, self.reserve), quantity)
        {
            status.set(comptime!(EventStatus::Accepted));
        }
        if !faults.is_empty() || prior.contains(comptime!(EventStatus::Faulted)) {
            faults.record(control);
            status.set(comptime!(EventStatus::Faulted));
        }
        if status.contains(comptime!(EventStatus::Accepted)) {
            self.cash = cash.value;
            self.stock = minus(self.stock, quantity);
            self.sales = sales.value;
        }
        status
    }

    fn store(&self, state: &mut [u32], config: &EconomyConfig, peer: u32, good: u32) {
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

#[derive(CubeType, CubeLaunch)]
pub(in crate::simulation) struct QuantityInput {
    pub state: Box<[u32]>,
    pub actors: Box<[u32]>,
    pub config: EconomyConfig,
    pub proposals: Box<[u32]>,
    pub scratch: Box<[u32]>,
    pub control: Box<[Atomic<u32>]>,
    pub parameters: EconomyParameters,
}

pub(in crate::simulation) struct QuantitySettlement;
pub(in crate::simulation) struct SettleQuantities;

impl System for QuantitySettlement {
    type Input = QuantityInput;
    type Passes = gpu_passes![SettleQuantities];
}

#[cube]
impl GpuPass<QuantityInput> for SettleQuantities {
    type Output = ();

    fn execute(mut input: QuantityInput) {
        let job = CUBE_POS as u32;
        if job < input.control[RANGE_SLOTS_WORD].load() && input.control[LATCH_WORD].load() == 0 {
            let count = input.parameters.count;
            let source = input.parameters.sort_source;
            let start = input.scratch[(5 * count + job) as usize];
            let length = input.scratch[(6 * count + job) as usize];
            let end = start + length;
            let peer = input.scratch[(source + start) as usize];
            let good = input.parameters.traded_good();
            let mut balances =
                QuantityBalances::new(&mut input.state, &input.actors, &input.config, peer, good);
            let mut shared = Shared::<[Ticks]>::new_slice(3usize);
            let mut cursor = start;
            while cursor < end {
                let index = cursor + UNIT_POS;
                let mut actor = 0u32;
                let mut quantity = zero();
                let mut prior = KernelEventStatus::new(comptime!(EventStatus::Accepted));
                if index < end {
                    actor = input.scratch[(source + count + index) as usize];
                    let base = (actor + input.parameters.offset) as usize * ProposalLayout::WORDS;
                    quantity = ticks(
                        input.proposals[base + ProposalLayout::QUANTITY_LOW],
                        input.proposals[base + ProposalLayout::QUANTITY_HIGH],
                    );
                    prior.replace(KernelEventStatus::load(
                        &mut input.scratch,
                        4 * count + actor,
                    ));
                }
                let mut total = zero();
                let mut mode = u32::new(comptime!(QuantityMode::Unavailable as i64));
                if !equal(positive(balances.stock, balances.reserve), zero()) {
                    let summary = summarize(quantity, &prior);
                    total = summary.total;
                    mode = balances.mode(&summary);
                }
                match mode {
                    0 => {
                        balances.aggregate(total);
                        if index < end {
                            KernelEventStatus::new(comptime!(EventStatus::Accepted))
                                .store(&mut input.scratch, 4 * count + actor);
                        }
                    }
                    1 => {
                        if index < end {
                            balances
                                .offer(quantity, prior, &input.control)
                                .store(&mut input.scratch, 4 * count + actor);
                        }
                    }
                    _ => {
                        if UNIT_POS == 0 {
                            for offset in 0..min(CUBE_DIM, end - cursor) {
                                let actor =
                                    input.scratch[(source + count + cursor + offset) as usize];
                                let base = (actor + input.parameters.offset) as usize
                                    * ProposalLayout::WORDS;
                                let quantity = ticks(
                                    input.proposals[base + ProposalLayout::QUANTITY_LOW],
                                    input.proposals[base + ProposalLayout::QUANTITY_HIGH],
                                );
                                let prior =
                                    KernelEventStatus::load(&mut input.scratch, 4 * count + actor);
                                balances
                                    .offer(quantity, prior, &input.control)
                                    .store(&mut input.scratch, 4 * count + actor);
                            }
                            shared[0usize] = balances.cash;
                            shared[1usize] = balances.stock;
                            shared[2usize] = balances.sales;
                        }
                        sync_cube();
                        balances.cash = shared[0usize];
                        balances.stock = shared[1usize];
                        balances.sales = shared[2usize];
                    }
                }
                sync_cube();
                cursor += CUBE_DIM;
            }
            if UNIT_POS == 0 {
                balances.store(&mut input.state, &input.config, peer, good);
                input.control[PROGRESS_WORD].fetch_max(length);
                input.control[COMPLETED_WORD].fetch_add(length);
            }
        }
    }
}

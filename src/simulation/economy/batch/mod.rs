use super::*;

pub(in crate::simulation) mod labor;

#[derive(CubeType)]
struct UniformCapacity {
    available: u32,
    overflow: u32,
}

#[cube]
fn bounded_capacity(balance: Ticks, amount: Ticks, limit: u32) -> u32 {
    let product = times(amount, limit);
    let mut low = limit;
    if !product.faults.is_empty() || lt(balance, product.value) {
        low = 0;
        let mut high = limit;
        while low < high {
            let middle = low + ceildiv(high - low, 2);
            let product = times(amount, middle);
            let fits = product.faults.is_empty() && !lt(balance, product.value);
            low = select(fits, middle, low);
            high = select(fits, high, middle - 1);
        }
    }
    low
}

#[cube]
impl UniformCapacity {
    fn limit(&self) -> u32 {
        min(self.available, self.overflow)
    }

    fn decision(&self, prior: KernelEventStatus, preceding: u32) -> UnitDecision {
        let before = min(preceding, self.limit());
        let overflow = before >= self.overflow;
        let mut status = KernelEventStatus::new(comptime!(EventStatus::Rejected));
        if prior.contains(comptime!(EventStatus::Accepted)) && preceding < self.limit() {
            status.set(comptime!(EventStatus::Accepted));
        }
        if overflow || prior.contains(comptime!(EventStatus::Faulted)) {
            status.set(comptime!(EventStatus::Faulted));
        }
        UnitDecision { status, overflow }
    }
}

#[derive(CubeType)]
struct UnitBalances {
    cash: Ticks,
    stock: Ticks,
    sales: Ticks,
    quote: u32,
    available: Ticks,
    cash_capacity: Ticks,
    sales_capacity: Ticks,
}

#[derive(CubeType)]
struct UnitDecision {
    status: KernelEventStatus,
    overflow: bool,
}

#[cube]
impl UnitBalances {
    fn new(
        state: &mut [u32],
        actors: &[u32],
        config: &EconomyConfig,
        peer: u32,
        good: u32,
    ) -> Self {
        let cash = value(state, config, comptime!(Field::Cash as u32), peer);
        let stock = value(state, config, good, peer);
        let sales = value(state, config, comptime!(Field::Sales as u32), peer);
        let quote = number(state, config, comptime!(Field::Price as u32), peer);
        let mut cash_capacity = ticks(u32::MAX, u32::MAX);
        if quote > 0 {
            cash_capacity = divide(
                ticks(!cash.extract(LOW_WORD), !cash.extract(HIGH_WORD)),
                quote,
            );
        }
        UnitBalances {
            cash,
            stock,
            sales,
            quote,
            available: positive(
                stock,
                household_production::reserve(state, actors, config, peer, good),
            ),
            cash_capacity,
            sales_capacity: ticks(!sales.extract(LOW_WORD), !sales.extract(HIGH_WORD)),
        }
    }

    fn capacity(&self) -> Ticks {
        minimum(
            self.available,
            minimum(self.cash_capacity, self.sales_capacity),
        )
    }

    fn decision(&self, prior: KernelEventStatus, preceding: u32) -> UnitDecision {
        let capacity = self.capacity();
        let before = minimum(ticks(preceding, 0), capacity);
        let overflow =
            (self.quote > 0 && !lt(before, self.cash_capacity)) || !lt(before, self.sales_capacity);
        let mut status = KernelEventStatus::new(comptime!(EventStatus::Rejected));
        if prior.contains(comptime!(EventStatus::Accepted)) && lt(ticks(preceding, 0), capacity) {
            status.set(comptime!(EventStatus::Accepted));
        }
        if overflow || prior.contains(comptime!(EventStatus::Faulted)) {
            status.set(comptime!(EventStatus::Faulted));
        }
        UnitDecision { status, overflow }
    }

    fn apply(&mut self, eligible: u32) {
        let applied = minimum(ticks(eligible, 0), self.capacity());
        self.cash = plus(self.cash, times(applied, self.quote).value).value;
        self.stock = minus(self.stock, applied);
        self.sales = plus(self.sales, applied).value;
        self.available = minus(self.available, applied);
        self.cash_capacity = minus(self.cash_capacity, applied);
        self.sales_capacity = minus(self.sales_capacity, applied);
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
pub(in crate::simulation) struct SettlementInput {
    pub state: Box<[u32]>,
    pub actors: Box<[u32]>,
    pub config: EconomyConfig,
    pub scratch: Box<[u32]>,
    pub control: Box<[Atomic<u32>]>,
    pub parameters: EconomyParameters,
}

pub(in crate::simulation) struct ConsumerSettlement;
pub(in crate::simulation) struct SettleUnits;

impl System for ConsumerSettlement {
    type Input = SettlementInput;
    type Passes = gpu_passes![SettleUnits];
}

#[cube]
impl GpuPass<SettlementInput> for SettleUnits {
    type Output = ();

    fn execute(mut input: SettlementInput) {
        settle_units(
            &mut input.state,
            &input.actors,
            &input.config,
            &mut input.scratch,
            &input.control,
            &input.parameters,
        );
    }
}

#[cube]
fn settle_units(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    scratch: &mut [u32],
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
) {
    let job = CUBE_POS as u32;
    if job < control[RANGE_SLOTS_WORD].load() && control[LATCH_WORD].load() == 0 {
        let count = parameters.count;
        let start = scratch[(5 * count + job) as usize];
        let length = scratch[(6 * count + job) as usize];
        let end = start + length;
        let peer = scratch[(parameters.sort_source + start) as usize];
        let good = comptime!(parameters.market.good().expect("consumer good") as u32);
        let mut balances = UnitBalances::new(state, actors, config, peer, good);
        let mut cursor = start;
        let mut faulted = 0u32;
        while cursor < end {
            let index = cursor + UNIT_POS;
            let mut actor = 0u32;
            let mut prior = KernelEventStatus::new(comptime!(EventStatus::Inactive));
            if index < end {
                actor = scratch[(parameters.sort_source + count + index) as usize];
                prior.replace(KernelEventStatus::load(scratch, 4 * count + actor));
            }
            let eligible = u32::cast_from(prior.contains(comptime!(EventStatus::Accepted)));
            let prefix = parallel::scan_prefix(eligible, comptime!(parameters.scan));
            if index < end {
                let decision = balances.decision(prior, prefix.inclusive - eligible);
                faulted |= u32::cast_from(decision.overflow);
                decision.status.store(scratch, 4 * count + actor);
            }
            balances.apply(prefix.total);
            sync_cube();
            cursor += CUBE_DIM;
        }
        let faults = parallel::scan_prefix(faulted, comptime!(parameters.scan));
        if UNIT_POS == 0 && faults.total > 0 {
            KernelFaults::new(comptime!(FaultFlags::ARITHMETIC_OVERFLOW)).record(control);
        }
        if UNIT_POS == 0 {
            balances.store(state, config, peer, good);
            control[PROGRESS_WORD].fetch_max(length);
            control[COMPLETED_WORD].fetch_add(length);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{Experiment, Market, Scenario};
    use crate::simulation::{Device, MarketExecution, Simulation, encode};
    use anyhow::Result;

    #[derive(Clone, Copy)]
    enum PriorPattern {
        Accepted,
        Rejected,
        Faulted,
        Mixed,
    }

    impl PriorPattern {
        fn status(self, slot: usize) -> EventStatus {
            match self {
                Self::Accepted => EventStatus::Accepted,
                Self::Rejected => EventStatus::Rejected,
                Self::Faulted => EventStatus::Faulted,
                Self::Mixed => match slot % 7 {
                    0 => EventStatus::Rejected,
                    _ => EventStatus::Accepted,
                },
            }
        }
    }

    struct SettlementCase {
        cash: u64,
        stock: u64,
        sales: u64,
        quote: u32,
        length: usize,
        prior: PriorPattern,
    }

    #[test]
    fn unit_batches_match_sequential_settlement_and_faults_on_gpu() -> Result<()> {
        let scenario = Scenario::industrial(513, 42, Experiment::Concentrated)?;
        let initial = scenario.initial()?;
        let peer = 513;
        let count = initial.actors.len();
        let cases = [
            SettlementCase {
                cash: 0,
                stock: 200,
                sales: 0,
                quote: u32::MAX,
                length: 257,
                prior: PriorPattern::Mixed,
            },
            SettlementCase {
                cash: u64::MAX - 2 * u32::MAX as u64,
                stock: 200,
                sales: 0,
                quote: u32::MAX,
                length: 64,
                prior: PriorPattern::Mixed,
            },
            SettlementCase {
                cash: u32::MAX as u64 - 1,
                stock: 200,
                sales: u32::MAX as u64,
                quote: 3,
                length: 257,
                prior: PriorPattern::Mixed,
            },
            SettlementCase {
                cash: u64::MAX - 6,
                stock: 1,
                sales: 0,
                quote: 3,
                length: 65,
                prior: PriorPattern::Accepted,
            },
            SettlementCase {
                cash: u64::MAX,
                stock: 0,
                sales: 0,
                quote: 3,
                length: 33,
                prior: PriorPattern::Rejected,
            },
            SettlementCase {
                cash: u64::MAX - 201,
                stock: 200,
                sales: 0,
                quote: 3,
                length: 257,
                prior: PriorPattern::Mixed,
            },
            SettlementCase {
                cash: 0,
                stock: 0,
                sales: u64::MAX,
                quote: 0,
                length: 31,
                prior: PriorPattern::Accepted,
            },
            SettlementCase {
                cash: 0,
                stock: 200,
                sales: u64::MAX - 1,
                quote: 1,
                length: 64,
                prior: PriorPattern::Mixed,
            },
            SettlementCase {
                cash: u64::MAX,
                stock: 200,
                sales: 0,
                quote: 0,
                length: 257,
                prior: PriorPattern::Mixed,
            },
            SettlementCase {
                cash: u64::MAX - 4,
                stock: 200,
                sales: 0,
                quote: 3,
                length: 32,
                prior: PriorPattern::Rejected,
            },
            SettlementCase {
                cash: 0,
                stock: 200,
                sales: 0,
                quote: 3,
                length: 63,
                prior: PriorPattern::Faulted,
            },
            SettlementCase {
                cash: u64::MAX - 192,
                stock: 64,
                sales: 0,
                quote: 3,
                length: 64,
                prior: PriorPattern::Accepted,
            },
            SettlementCase {
                cash: u64::MAX - 192,
                stock: 64,
                sales: 0,
                quote: 3,
                length: 65,
                prior: PriorPattern::Accepted,
            },
        ];
        for workgroup in [32, 64, 128, 256] {
            let device = Device::headless(workgroup)?;
            let actual = Simulation::new(scenario.clone(), device.clone())?;
            let mut expected = Simulation::new(scenario.clone(), device.clone())?;
            expected.market_execution = MarketExecution::Original;
            for mode in [parallel::ScanMode::Shared, parallel::ScanMode::new(&device)] {
                for (case_index, case) in cases.iter().enumerate() {
                    let mut state = initial.clone();
                    state.set(Field::Cash, peer, case.cash);
                    state.set(Field::Tools, peer, case.stock);
                    state.set(Field::Sales, peer, case.sales);
                    state.set(Field::Price, peer, case.quote.into());
                    let words = encode(&state.values);
                    let mut proposals = vec![0; count * ProposalLayout::WORDS];
                    let mut scratch = vec![0; actual.scratch.meta.num_elements()];
                    for slot in 0..case.length {
                        let actor = (slot + 17) % 513;
                        proposals[actor * ProposalLayout::WORDS + ProposalLayout::ACTOR] =
                            peer as u32;
                        proposals[actor * ProposalLayout::WORDS + ProposalLayout::QUANTITY_LOW] = 1;
                        scratch[slot] = peer as u32;
                        scratch[count + slot] = actor as u32;
                        scratch[4 * count + actor] =
                            enumflags2::BitFlags::from_flag(case.prior.status(slot)).bits();
                    }
                    scratch[5 * count] = 0;
                    scratch[6 * count] = case.length as u32;
                    let mut control = vec![0; actual.control.meta.num_elements()];
                    control[CONSTRUCTED_EVENTS_WORD] = case.length as u32;
                    control[RANGE_SLOTS_WORD] = 1;
                    for simulation in [&actual, &expected] {
                        device.write(&simulation.state, bytemuck::cast_slice(&words));
                        device.write(&simulation.proposals, bytemuck::cast_slice(&proposals));
                        device.write(&simulation.scratch, bytemuck::cast_slice(&scratch));
                        device.write(&simulation.control, bytemuck::cast_slice(&control));
                    }
                    let market = Market::Consumer(Good::Tools);
                    let mut original =
                        expected.parameters(Phase::ProvisionPurchase, market, 0, count)?;
                    original.sort_count = case.length as u32;
                    EconomySystem::launch(
                        &device,
                        device.count(case.length),
                        CubeDim::new_1d(workgroup),
                        EconomyInputLaunch {
                            state: expected.state.clone().into_buffer_arg(),
                            actors: expected.actors.clone().into_buffer_arg(),
                            config: expected.config.launch(),
                            spatial: expected.spatial.clone().into_buffer_arg(),
                            proposals: expected.proposals.clone().into_buffer_arg(),
                            bundles: expected.scratch.clone().into_buffer_arg(),
                            control: expected.control.clone().into_buffer_arg(),
                            parameters: original.launch(),
                            stage: EconomyStage::GroupExecute,
                        },
                    );
                    let mut parameters =
                        actual.parameters(Phase::ProvisionPurchase, market, 0, count)?;
                    parameters.scan = mode;
                    ConsumerSettlement::launch(
                        &device,
                        CubeCount::Static(1, 1, 1),
                        CubeDim::new_1d(workgroup),
                        SettlementInputLaunch {
                            state: actual.state.clone().into_buffer_arg(),
                            actors: actual.actors.clone().into_buffer_arg(),
                            config: actual.config.launch(),
                            scratch: actual.scratch.clone().into_buffer_arg(),
                            control: actual.control.clone().into_buffer_arg(),
                            parameters: parameters.launch(),
                        },
                    );
                    for (name, actual, expected) in [
                        ("state", &actual.state, &expected.state),
                        ("statuses", &actual.scratch, &expected.scratch),
                        ("faults", &actual.control, &expected.control),
                    ] {
                        let actual = device.read(actual)?;
                        let expected = device.read(expected)?;
                        assert_eq!(actual.len(), expected.len());
                        for (word, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                            assert_eq!(
                                actual, expected,
                                "{name} word {word}: {workgroup} {mode:?} case {case_index}"
                            );
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

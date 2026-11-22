use super::*;
use crate::simulation::DeviceEconomy;

pub(in crate::simulation) mod labor;
pub(in crate::simulation) mod payroll;
pub(in crate::simulation) mod quantity;

#[derive(CubeType)]
struct UniformCapacity {
    available: u32,
    overflow: u32,
}

#[cube]
pub(super) fn bounded_capacity(balance: Ticks, amount: Ticks, limit: u32) -> u32 {
    let product = times(amount, limit);
    if !product.faults.is_empty() || lt(balance, product.value) {
        let mut low = 0;
        let mut high = limit;
        while low < high {
            let middle = low + ceildiv(high - low, 2);
            let product = times(amount, middle);
            let fits = product.faults.is_empty() && !lt(balance, product.value);
            low = select(fits, middle, low);
            high = select(fits, high, middle - 1);
        }
        low
    } else {
        limit
    }
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
    sales_field: u32,
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

#[repr(u32)]
enum UnitSettlementState {
    Funded,
    Exhausted,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum UnitSettlementExecution {
    #[cfg_attr(not(test), expect(dead_code))]
    Complete,
    Prefix,
}

#[cube]
impl UnitBalances {
    fn new(economy: &mut EconomyView<'_>, peer: u32, good: u32, length: u32) -> Self {
        let cash = economy.ledger.value(Field::Cash, peer);
        let stock = economy.ledger.value_at(good, peer);
        let terms = household_production::TradeFields::new(&economy.actors, peer, good);
        let sales = economy.ledger.value_at(terms.sales, peer);
        let quote = economy.ledger.number_at(terms.price, peer);
        let cash_capacity = bounded_capacity(
            ticks(!cash.extract(LOW_WORD), !cash.extract(HIGH_WORD)),
            ticks(quote, 0),
            length,
        );
        UnitBalances {
            cash,
            stock,
            sales,
            sales_field: terms.sales,
            quote,
            available: positive(stock, household_production::reserve(&*economy, peer, good)),
            cash_capacity: ticks(cash_capacity, 0),
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

    fn store(&self, ledger: &mut Ledger<'_>, peer: u32, good: u32) {
        ledger.set(Field::Cash, peer, self.cash);
        ledger.set_at(good, peer, self.stock);
        ledger.set_at(self.sales_field, peer, self.sales);
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

#[cfg(test)]
pub(in crate::simulation) struct ConsumerSettlement;
#[cfg(test)]
pub(in crate::simulation) struct SettleUnits;

#[cfg(test)]
impl System for ConsumerSettlement {
    type Input = SettlementInput;
    type Passes = gpu_passes![SettleUnits];
}

#[cfg(test)]
#[cube]
impl GpuPass<SettlementInput> for SettleUnits {
    type Output = ();

    fn execute(mut input: SettlementInput) {
        input.parameters.refresh(&mut input.config);
        let mut economy = EconomyView::new(&mut input.state, &input.actors, &input.config);
        settle_units(
            &mut economy,
            &mut input.scratch,
            &input.control,
            &input.parameters,
            comptime!(UnitSettlementExecution::Complete),
        );
    }
}

struct ConsumerPrefix;
struct SettlePrefix;
struct ConsumerTail;
struct SettleTail;

impl System for ConsumerPrefix {
    type Input = SettlementInput;
    type Passes = gpu_passes![SettlePrefix];
}

impl System for ConsumerTail {
    type Input = SettlementInput;
    type Passes = gpu_passes![SettleTail];
}

#[cube]
impl GpuPass<SettlementInput> for SettlePrefix {
    type Output = ();

    fn execute(mut input: SettlementInput) {
        input.parameters.refresh(&mut input.config);
        let mut economy = EconomyView::new(&mut input.state, &input.actors, &input.config);
        settle_units(
            &mut economy,
            &mut input.scratch,
            &input.control,
            &input.parameters,
            comptime!(UnitSettlementExecution::Prefix),
        );
    }
}

#[cube]
impl GpuPass<SettlementInput> for SettleTail {
    type Output = ();

    fn execute(mut input: SettlementInput) {
        input.parameters.refresh(&mut input.config);
        let mut economy = EconomyView::new(&mut input.state, &input.actors, &input.config);
        settle_tail(
            &mut economy,
            &mut input.scratch,
            &input.control,
            &input.parameters,
        );
    }
}

pub(in crate::simulation) fn settle_consumers(
    DeviceEconomy {
        device,
        state,
        actors,
        config,
    }: DeviceEconomy<'_>,
    scratch: &burn_wgpu::CubeTensor,
    control: &burn_wgpu::CubeTensor,
    dispatches: &burn_wgpu::CubeTensor,
    parameters: &crate::simulation::parameters::EconomyParameters,
) {
    let input = || SettlementInputLaunch {
        state: state.buffer_arg(),
        actors: actors.buffer_arg(),
        config: config.launch(),
        scratch: scratch.buffer_arg(),
        control: control.buffer_arg(),
        parameters: parameters.launch(),
    };
    ConsumerPrefix::launch(
        device,
        markets::job_dispatch(
            dispatches,
            markets::QUEUE_DISPATCH,
            parameters.count as usize,
        ),
        CubeDim::new_1d(device.workgroup),
        input(),
    );
    ConsumerTail::launch(
        device,
        markets::dispatch(
            dispatches,
            markets::SORT_DISPATCH,
            parameters.count.div_ceil(device.workgroup) as usize,
        ),
        CubeDim::new_1d(device.workgroup),
        input(),
    );
}

#[cube]
fn settle_units(
    economy: &mut EconomyView<'_>,
    scratch: &mut [u32],
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
    #[comptime] execution: UnitSettlementExecution,
) {
    let jobs = markets::job_count(control, RANGE_SLOTS_WORD);
    let mut job = CUBE_POS as u32;
    while job < jobs {
        let count = parameters.count;
        let start = scratch[(5 * count + job) as usize];
        let length = scratch[(6 * count + job) as usize];
        let end = start + length;
        let peer = scratch[(parameters.sort_source + start) as usize];
        let good = parameters.traded_good();
        let mut balances = UnitBalances::new(economy, peer, good, length);
        let mut cursor = start;
        let mut tail_cursor = end;
        let mut faulted = 0u32;
        while cursor < end {
            let settlement = select(
                equal(balances.capacity(), zero()),
                comptime!(UnitSettlementState::Exhausted as u32),
                comptime!(UnitSettlementState::Funded as u32),
            );
            match settlement {
                0 => {
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
                _ => {
                    match comptime!(execution) {
                        UnitSettlementExecution::Complete => {
                            let mut index = cursor + UNIT_POS;
                            while index < end {
                                let actor =
                                    scratch[(parameters.sort_source + count + index) as usize];
                                let prior = KernelEventStatus::load(scratch, 4 * count + actor);
                                let decision = balances.decision(prior, 0);
                                faulted |= u32::cast_from(decision.overflow);
                                decision.status.store(scratch, 4 * count + actor);
                                index += CUBE_DIM;
                            }
                        }
                        UnitSettlementExecution::Prefix => tail_cursor = cursor,
                    }
                    cursor = end;
                }
            }
        }
        sync_cube();
        let faults = parallel::scan_prefix(faulted, comptime!(parameters.scan));
        if UNIT_POS == 0 && faults.total > 0 {
            KernelFaults::new(comptime!(FaultFlags::ARITHMETIC_OVERFLOW)).record(control);
        }
        if UNIT_POS == 0 {
            if comptime!(execution == UnitSettlementExecution::Prefix) {
                scratch[(8 * count + peer) as usize] = tail_cursor;
            }
            balances.store(&mut economy.ledger, peer, good);
            control[PROGRESS_WORD].fetch_max(length);
            control[COMPLETED_WORD].fetch_add(length);
        }
        sync_cube();
        job += CUBE_COUNT as u32;
    }
}

#[cube]
fn settle_tail(
    economy: &mut EconomyView<'_>,
    scratch: &mut [u32],
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
) {
    let index = ABSOLUTE_POS as u32;
    if index < markets::active(control, parameters) && control[LATCH_WORD].load() == 0 {
        let count = parameters.count;
        let peer = scratch[(parameters.sort_source + index) as usize];
        let cursor = scratch[(8 * count + peer) as usize];
        if index >= cursor {
            let actor = scratch[(parameters.sort_source + count + index) as usize];
            let prior = KernelEventStatus::load(scratch, 4 * count + actor);
            let terms = household_production::TradeFields::new(
                &economy.actors,
                peer,
                parameters.traded_good(),
            );
            let cash = plus(
                economy.ledger.value(Field::Cash, peer),
                ticks(economy.ledger.number_at(terms.price, peer), 0),
            );
            let sales = plus(economy.ledger.value_at(terms.sales, peer), ticks(1, 0));
            let overflow = !cash.faults.is_empty() || !sales.faults.is_empty();
            let mut status = KernelEventStatus::new(comptime!(EventStatus::Rejected));
            if overflow || prior.contains(comptime!(EventStatus::Faulted)) {
                status.set(comptime!(EventStatus::Faulted));
            }
            status.store(scratch, 4 * count + actor);
            if overflow {
                KernelFaults::new(comptime!(FaultFlags::ARITHMETIC_OVERFLOW)).record(control);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{Experiment, Market, Scenario};
    use crate::simulation::{MarketExecution, Simulation, encode};
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
            match (self, slot % 7) {
                (Self::Accepted, _) | (Self::Mixed, 1..) => EventStatus::Accepted,
                (Self::Rejected, _) | (Self::Mixed, 0) => EventStatus::Rejected,
                (Self::Faulted, _) => EventStatus::Faulted,
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
    fn parallel_large_consumer_tails_preserve_retail_reserves_and_overflow_statuses_on_gpu()
    -> Result<()> {
        struct TailCase {
            cash: u64,
            stock: u64,
            sales: u64,
            quote: u32,
            retail_open: u64,
        }
        let households = 65_537;
        let scenario = Scenario::industrial(households, 42, Experiment::Concentrated)?;
        let initial = scenario.initial()?;
        let count = initial.actors.len();
        let peer = households as usize;
        let length = households as usize;
        let device = crate::simulation::tests::device(64)?;
        let actual = Simulation::new(scenario.clone(), device.clone())?;
        let expected = Simulation::new(scenario.clone(), device.clone())?;
        let mut scratch = vec![0; actual.work.scratch.meta.num_elements()];
        for actor in 0..length {
            scratch[actor] = peer as u32;
            scratch[count + actor] = actor as u32;
            let status = match actor % 7 {
                0 => EventStatus::Rejected,
                1 => EventStatus::Faulted,
                _ => EventStatus::Accepted,
            };
            scratch[4 * count + actor] = enumflags2::BitFlags::from_flag(status).bits();
        }
        scratch[5 * count] = 0;
        scratch[6 * count] = length as u32;
        let mut control = vec![0; actual.control.meta.num_elements()];
        control[CONSTRUCTED_EVENTS_WORD] = length as u32;
        control[RANGE_SLOTS_WORD] = 1;
        for case in [
            TailCase {
                cash: 0,
                stock: 400,
                sales: 1,
                quote: 3,
                retail_open: 3,
            },
            TailCase {
                cash: u64::MAX - 3,
                stock: 2,
                sales: 0,
                quote: 3,
                retail_open: length as u64,
            },
        ] {
            let mut state = initial.clone();
            for (field, amount) in [
                (Field::Cash, case.cash),
                (Field::Tools, case.stock),
                (Field::Sales, case.sales),
                (Field::Price, u64::from(case.quote)),
                (Field::EnterpriseModel, 1),
                (Field::RetailOpen, case.retail_open),
            ] {
                state.set(field, peer, amount);
            }
            let padded = crate::simulation::tensor::state_values(
                &state.values,
                count,
                actual.config.state_stride as usize,
            );
            let words = encode(&padded);
            for simulation in [&actual, &expected] {
                device.write(&simulation.state, bytemuck::cast_slice(&words));
                device.write(&simulation.work.scratch, &scratch);
                device.write(&simulation.control, &control);
            }
            let parameters = actual.parameters(
                Phase::ProvisionPurchase,
                Market::Consumer(Good::Tools),
                0,
                count,
            )?;
            ConsumerSettlement::launch(
                &device,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(64),
                SettlementInputLaunch {
                    state: expected.state.buffer_arg(),
                    actors: expected.actors.buffer_arg(),
                    config: expected.config.launch(),
                    scratch: expected.work.scratch.buffer_arg(),
                    control: expected.control.buffer_arg(),
                    parameters: parameters.launch(),
                },
            );
            markets::prepare_dispatch(&device, &actual.control, &actual.searches.dispatches, count);
            settle_consumers(
                DeviceEconomy {
                    device: &device,
                    ..actual.gpu()
                },
                &actual.work.scratch,
                &actual.control,
                &actual.searches.dispatches,
                &parameters,
            );
            assert_eq!(device.read(&actual.state)?, device.read(&expected.state)?);
            assert_eq!(
                device.read(&actual.control)?,
                device.read(&expected.control)?
            );
            assert_eq!(
                device.read_range(&actual.work.scratch, 4 * count, count)?,
                device.read_range(&expected.work.scratch, 4 * count, count)?
            );
        }
        Ok(())
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
            SettlementCase {
                cash: 0,
                stock: 1,
                sales: 0,
                quote: 3,
                length: 513,
                prior: PriorPattern::Mixed,
            },
            SettlementCase {
                cash: u64::MAX,
                stock: 513,
                sales: 0,
                quote: 3,
                length: 513,
                prior: PriorPattern::Mixed,
            },
            SettlementCase {
                cash: 0,
                stock: 513,
                sales: u64::MAX,
                quote: 0,
                length: 513,
                prior: PriorPattern::Mixed,
            },
        ];
        let workgroup = 64;
        let device = crate::simulation::tests::device(workgroup)?;
        let parallel = Simulation::new(scenario.clone(), device.clone())?;
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
                let mut scratch = vec![0; parallel.work.scratch.meta.num_elements()];
                for slot in 0..case.length {
                    let actor = (slot + 17) % 513;
                    proposals[actor * ProposalLayout::WORDS + ProposalLayout::ACTOR] = peer as u32;
                    proposals[actor * ProposalLayout::WORDS + ProposalLayout::QUANTITY_LOW] = 1;
                    scratch[slot] = peer as u32;
                    scratch[count + slot] = actor as u32;
                    scratch[4 * count + actor] =
                        enumflags2::BitFlags::from_flag(case.prior.status(slot)).bits();
                }
                scratch[5 * count] = 0;
                scratch[6 * count] = case.length as u32;
                let mut control = vec![0; parallel.control.meta.num_elements()];
                control[CONSTRUCTED_EVENTS_WORD] = case.length as u32;
                control[RANGE_SLOTS_WORD] = 1;
                for simulation in [&expected, &parallel] {
                    device.write(&simulation.state, bytemuck::cast_slice(&words));
                    device.write(&simulation.work.proposals, bytemuck::cast_slice(&proposals));
                    device.write(&simulation.work.scratch, bytemuck::cast_slice(&scratch));
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
                        state: expected.state.buffer_arg(),
                        actors: expected.actors.buffer_arg(),
                        config: expected.config.launch(),
                        spatial: expected.spatial.buffer_arg(),
                        proposals: expected.work.proposals.buffer_arg(),
                        bundles: expected.work.scratch.buffer_arg(),
                        control: expected.control.buffer_arg(),
                        parameters: original.launch(),
                        stage: EconomyStage::GroupExecute,
                    },
                );
                let mut parameters =
                    parallel.parameters(Phase::ProvisionPurchase, market, 0, count)?;
                parameters.scan = mode;
                markets::prepare_dispatch(
                    &device,
                    &parallel.control,
                    &parallel.searches.dispatches,
                    count,
                );
                settle_consumers(
                    DeviceEconomy {
                        device: &device,
                        ..parallel.gpu()
                    },
                    &parallel.work.scratch,
                    &parallel.control,
                    &parallel.searches.dispatches,
                    &parameters,
                );
                assert_eq!(
                    device.read(&parallel.state)?,
                    device.read(&expected.state)?,
                    "parallel state case {case_index}"
                );
                assert_eq!(
                    device.read(&parallel.control)?,
                    device.read(&expected.control)?,
                    "parallel counters case {case_index}"
                );
                let parallel_statuses =
                    device.read_range(&parallel.work.scratch, 4 * count, count)?;
                let expected_statuses =
                    device.read_range(&expected.work.scratch, 4 * count, count)?;
                assert_eq!(
                    parallel_statuses, expected_statuses,
                    "parallel statuses case {case_index}"
                );
            }
        }
        Ok(())
    }
}

use super::*;

#[derive(CubeType)]
struct UniformBalance {
    initial: Ticks,
    amount: Ticks,
}

#[cube]
impl UniformBalance {
    fn new(ledger: &Ledger<'_>, field: u32, peer: u32, amount: Ticks) -> Self {
        UniformBalance {
            initial: ledger.value_at(field, peer),
            amount,
        }
    }

    fn debit_capacity(&self, length: u32) -> u32 {
        bounded_capacity(self.initial, self.amount, length)
    }

    fn credit_capacity(&self, length: u32) -> u32 {
        bounded_capacity(
            ticks(
                !self.initial.extract(LOW_WORD),
                !self.initial.extract(HIGH_WORD),
            ),
            self.amount,
            length,
        )
    }

    fn debited(&self, accepted: u32) -> Ticks {
        minus(self.initial, times(self.amount, accepted).value)
    }

    fn credited(&self, accepted: u32) -> Ticks {
        plus(self.initial, times(self.amount, accepted).value).value
    }
}

#[derive(CubeType)]
struct LaborBalances {
    cash: UniformBalance,
    escrow: UniformBalance,
    slots: UniformBalance,
    pipeline: UniformBalance,
    stock: UniformBalance,
    second_stock: UniformBalance,
    first_used: UniformBalance,
    second_used: UniformBalance,
    used: UniformBalance,
    first: u32,
    second: u32,
    faults: KernelFaults,
}

#[cube]
impl LaborBalances {
    fn new(ledger: &Ledger<'_>, peer: u32) -> Self {
        let wage = times32(
            ledger.number(Field::Wage, peer),
            ledger.number(Field::LaborMinutes, peer),
        );
        let output = ticks(ledger.number(Field::OutputTicks, peer), 0);
        let first = ledger.number(Field::FirstInput, peer);
        let second = ledger.number(Field::SecondInput, peer);
        let input = ledger.value(Field::InputTicks, peer);
        let first_input = select(first > 0, input, zero());
        let second_input = select(second > 0, input, zero());
        let total_input = plus(first_input, second_input);
        LaborBalances {
            cash: UniformBalance::new(ledger, comptime!(Field::Cash as u32), peer, wage),
            escrow: UniformBalance::new(ledger, comptime!(Field::Escrow as u32), peer, wage),
            slots: UniformBalance::new(ledger, comptime!(Field::Slots as u32), peer, ticks(1, 0)),
            pipeline: UniformBalance::new(ledger, comptime!(Field::Pipeline as u32), peer, output),
            stock: UniformBalance::new(ledger, first, peer, first_input),
            second_stock: UniformBalance::new(ledger, second, peer, second_input),
            first_used: UniformBalance::new(
                ledger,
                USED_FIELDS as u32 + max(first, 1) - 1,
                peer,
                first_input,
            ),
            second_used: UniformBalance::new(
                ledger,
                USED_FIELDS as u32 + max(second, 1) - 1,
                peer,
                second_input,
            ),
            used: UniformBalance::new(
                ledger,
                comptime!(Field::Used as u32),
                peer,
                total_input.value,
            ),
            first,
            second,
            faults: total_input.faults,
        }
    }

    fn capacity(&self, length: u32) -> UniformCapacity {
        UniformCapacity {
            available: min(
                min(
                    self.cash.debit_capacity(length),
                    self.slots.debit_capacity(length),
                ),
                min(
                    self.stock.debit_capacity(length),
                    self.second_stock.debit_capacity(length),
                ),
            ),
            overflow: min(
                min(
                    self.escrow.credit_capacity(length),
                    self.pipeline.credit_capacity(length),
                ),
                min(
                    self.used.credit_capacity(length),
                    min(
                        self.first_used.credit_capacity(length),
                        self.second_used.credit_capacity(length),
                    ),
                ),
            ),
        }
    }

    fn store(&self, ledger: &mut Ledger<'_>, peer: u32, accepted: u32) {
        ledger.set(Field::Cash, peer, self.cash.debited(accepted));
        ledger.set(Field::Escrow, peer, self.escrow.credited(accepted));
        ledger.set(Field::Slots, peer, self.slots.debited(accepted));
        ledger.set(Field::Pipeline, peer, self.pipeline.credited(accepted));
        ledger.set(Field::Used, peer, self.used.credited(accepted));
        if self.first > 0 {
            ledger.set_at(self.first, peer, self.stock.debited(accepted));
            ledger.set_at(
                USED_FIELDS as u32 + self.first - 1,
                peer,
                self.first_used.credited(accepted),
            );
        }
        if self.second > 0 {
            ledger.set_at(self.second, peer, self.second_stock.debited(accepted));
            ledger.set_at(
                USED_FIELDS as u32 + self.second - 1,
                peer,
                self.second_used.credited(accepted),
            );
        }
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(in crate::simulation) enum LaborExecution {
    Complete,
    Deferred,
}

#[derive(CubeType, CubeLaunch)]
pub(in crate::simulation) struct LaborInput {
    pub state: Box<[u32]>,
    pub config: EconomyConfig,
    pub proposals: Box<[u32]>,
    pub scratch: Box<[u32]>,
    pub control: Box<[Atomic<u32>]>,
    pub parameters: EconomyParameters,
    #[cube(comptime)]
    pub execution: LaborExecution,
}

pub(in crate::simulation) struct LaborSettlement;
pub(in crate::simulation) struct SettleLabor;

impl System for LaborSettlement {
    type Input = LaborInput;
    type Passes = gpu_passes![SettleLabor];
}

#[cube]
impl GpuPass<LaborInput> for SettleLabor {
    type Output = ();

    fn execute(mut input: LaborInput) {
        input.parameters.refresh(&mut input.config);
        let mut ledger = Ledger::new(&mut input.state, &input.config);
        settle_labor(
            &mut ledger,
            &mut input.proposals,
            &mut input.scratch,
            &input.control,
            &input.parameters,
            comptime!(input.execution),
        );
    }
}

#[cube]
fn settle_labor(
    ledger: &mut Ledger<'_>,
    proposals: &mut [u32],
    scratch: &mut [u32],
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
    #[comptime] execution: LaborExecution,
) {
    let config = ledger.config;
    let jobs = orders::job_count(control, RANGE_SLOTS_WORD);
    let mut job = CUBE_POS as u32;
    while job < jobs {
        let count = parameters.count;
        let start = scratch[(5 * count + job) as usize];
        let length = scratch[(6 * count + job) as usize];
        let end = start + length;
        let peer = scratch[(parameters.sort_source + start) as usize];
        let balances = LaborBalances::new(&*ledger, peer);
        let limit = match comptime!(config.cohorts) {
            Feature::Disabled => length,
            Feature::Enabled => ledger.number(Field::Slots, peer) + 1,
        };
        let capacity = balances.capacity(limit);
        let mut accepted = 0u32;
        let mut cursor = start;
        let mut faulted = 0u32;
        while cursor < end
            && (comptime!(execution == LaborExecution::Complete)
                || capacity.overflow <= capacity.available
                || accepted < capacity.available)
        {
            let index = cursor + UNIT_POS;
            let mut actor = 0u32;
            let mut prior = KernelEventStatus::new(comptime!(EventStatus::Inactive));
            if index < end {
                actor = scratch[(parameters.sort_source + count + index) as usize];
                prior.replace(KernelEventStatus::load(scratch, 4 * count + actor));
            }
            let word = (actor + parameters.offset) as usize * ProposalLayout::WORDS
                + ProposalLayout::QUANTITY_HIGH;
            let mut shifts = 1u32;
            if comptime!(config.cohorts == Feature::Enabled) && index < end {
                shifts = max(proposals[word], 1);
            }
            let eligible = select(prior.contains(comptime!(EventStatus::Accepted)), shifts, 0);
            let prefix = parallel::scan_prefix(eligible, comptime!(parameters.scan));
            if index < end {
                let before = accepted + prefix.inclusive - eligible;
                let decision = capacity.decision(prior, before);
                if comptime!(config.cohorts == Feature::Enabled) {
                    proposals[word] = min(eligible, max(capacity.limit(), before) - before);
                }
                faulted |= u32::cast_from(decision.overflow);
                decision.status.store(scratch, 4 * count + actor);
            }
            accepted = min(accepted + prefix.total, capacity.limit());
            sync_cube();
            cursor += CUBE_DIM;
        }
        let faults = parallel::scan_prefix(faulted, comptime!(parameters.scan));
        if UNIT_POS == 0 && faults.total > 0 {
            KernelFaults::new(comptime!(FaultFlags::ARITHMETIC_OVERFLOW)).record(control);
        }
        if UNIT_POS == 0 {
            if comptime!(execution == LaborExecution::Deferred) {
                scratch[(7 * count + peer) as usize] = min(cursor, end);
            }
            balances.faults.record(control);
            balances.store(ledger, peer, accepted);
            control[PROGRESS_WORD].fetch_max(length);
            control[COMPLETED_WORD].fetch_add(length);
        }
        sync_cube();
        job += CUBE_COUNT as u32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{Experiment, Market, Scenario};
    use crate::simulation::{MarketExecution, Simulation, encode};
    use anyhow::Result;

    struct LaborCase {
        field: Field,
        value: u64,
        length: usize,
        prior: EventStatus,
        first_used: u64,
        second_used: u64,
    }

    impl LaborCase {
        fn balance(field: Field, value: u64) -> Self {
            Self {
                field,
                value,
                length: 257,
                prior: EventStatus::Accepted,
                first_used: 0,
                second_used: 0,
            }
        }
    }

    #[test]
    fn fixed_wage_batches_match_sequential_settlement_on_gpu() -> Result<()> {
        let scenario = Scenario::industrial(513, 42, Experiment::Concentrated)?;
        let initial = scenario.initial()?;
        let peer = 513;
        let count = initial.actors.len();
        let cases = [
            LaborCase::balance(Field::Cash, 1_000_000),
            LaborCase::balance(Field::Cash, 30 * 63),
            LaborCase::balance(Field::Cash, 0),
            LaborCase::balance(Field::Slots, 0),
            LaborCase::balance(Field::Slots, 64),
            LaborCase::balance(Field::Grain, 129),
            LaborCase::balance(Field::Flour, 1),
            LaborCase::balance(Field::Escrow, u64::MAX - 30 * 64),
            LaborCase::balance(Field::Pipeline, u64::MAX - 5 * 64),
            LaborCase::balance(Field::Used, u64::MAX - 4 * 64),
            LaborCase::balance(Field::Wage, u32::MAX.into()),
            LaborCase::balance(Field::LaborMinutes, u32::MAX.into()),
            LaborCase::balance(Field::InputTicks, u64::MAX),
            LaborCase::balance(Field::InputTicks, 1 << 63),
            LaborCase::balance(Field::InputTicks, 0),
            LaborCase::balance(Field::FirstInput, 0),
            LaborCase::balance(Field::SecondInput, 0),
            LaborCase::balance(Field::Wage, 0),
            LaborCase::balance(Field::Pipeline, u32::MAX.into()),
            LaborCase {
                first_used: u64::MAX - 128,
                ..LaborCase::balance(Field::Cash, 1_000_000)
            },
            LaborCase {
                second_used: u64::MAX - 128,
                ..LaborCase::balance(Field::Cash, 1_000_000)
            },
            LaborCase {
                length: 65,
                prior: EventStatus::Rejected,
                ..LaborCase::balance(Field::Escrow, u64::MAX)
            },
            LaborCase {
                length: 63,
                prior: EventStatus::Faulted,
                ..LaborCase::balance(Field::Escrow, 0)
            },
            LaborCase {
                length: 64,
                ..LaborCase::balance(Field::Escrow, u64::MAX - 30 * 64)
            },
            LaborCase {
                length: 65,
                ..LaborCase::balance(Field::Escrow, u64::MAX - 30 * 64)
            },
        ];
        let workgroup = 64;
        let device = crate::simulation::tests::device(workgroup)?;
        let actual = Simulation::new(scenario.clone(), device.clone())?;
        let mut expected = Simulation::new(scenario.clone(), device.clone())?;
        expected.market_execution = MarketExecution::Original;
        for mode in [parallel::ScanMode::Shared, parallel::ScanMode::new(&device)] {
            for (case_index, case) in cases.iter().enumerate() {
                let mut state = initial.clone();
                state.set(Field::Cash, peer, 1_000_000);
                state.set(Field::Escrow, peer, 0);
                state.set(Field::Slots, peer, 200);
                state.set(Field::Pipeline, peer, 0);
                state.set(Field::Used, peer, 0);
                state.set(Field::Wage, peer, 3);
                state.set(Field::LaborMinutes, peer, 10);
                state.set(Field::OutputTicks, peer, 5);
                state.set(Field::FirstInput, peer, Good::Grain as u64);
                state.set(Field::SecondInput, peer, Good::Flour as u64);
                state.set(Field::InputTicks, peer, 2);
                state.set(Field::Grain, peer, 1000);
                state.set(Field::Flour, peer, 1000);
                match case.field {
                    Field::Wage | Field::LaborMinutes => state.set(Field::Cash, peer, u64::MAX),
                    Field::InputTicks => {
                        state.set(Field::Grain, peer, u64::MAX);
                        state.set(Field::Flour, peer, u64::MAX);
                    }
                    _ => (),
                }
                state.set(case.field, peer, case.value);
                let mut words = encode(&state.values);
                let first_used = (USED_FIELDS + Good::Grain as usize - 1) * count + peer;
                let second_used = (USED_FIELDS + Good::Flour as usize - 1) * count + peer;
                words[first_used] = case.first_used.into();
                words[second_used] = case.second_used.into();
                let mut proposals = vec![0; count * ProposalLayout::WORDS];
                let mut scratch = vec![0; actual.work.scratch.meta.num_elements()];
                let first_length = match case_index {
                    0 => case.length / 2,
                    _ => case.length,
                };
                for slot in 0..case.length {
                    let actor = (slot + 17) % 513;
                    let peer = peer + usize::from(slot >= first_length);
                    proposals[actor * ProposalLayout::WORDS + ProposalLayout::ACTOR] = peer as u32;
                    scratch[slot] = peer as u32;
                    scratch[count + slot] = actor as u32;
                    let status = match (case_index % 2, slot % 7, case.length) {
                        (0, 0, 257) => EventStatus::Rejected,
                        _ => case.prior,
                    };
                    scratch[4 * count + actor] = enumflags2::BitFlags::from_flag(status).bits();
                }
                scratch[5 * count] = 0;
                scratch[6 * count] = first_length as u32;
                let mut control = vec![0; actual.control.meta.num_elements()];
                control[CONSTRUCTED_EVENTS_WORD] = case.length as u32;
                control[RANGE_SLOTS_WORD] = 1;
                if first_length < case.length {
                    scratch[5 * count + 1] = first_length as u32;
                    scratch[6 * count + 1] = (case.length - first_length) as u32;
                    control[RANGE_SLOTS_WORD] = 2;
                }
                for simulation in [&actual, &expected] {
                    device.write(&simulation.state, bytemuck::cast_slice(&words));
                    device.write(&simulation.work.proposals, bytemuck::cast_slice(&proposals));
                    device.write(&simulation.work.scratch, bytemuck::cast_slice(&scratch));
                    device.write(&simulation.control, bytemuck::cast_slice(&control));
                }
                let mut original =
                    expected.parameters(Phase::Commitment, Market::Labor, 0, count)?;
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
                    actual.parameters(Phase::Commitment, Market::Labor, 0, count)?;
                parameters.scan = mode;
                LaborSettlement::launch(
                    &device,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new_1d(workgroup),
                    LaborInputLaunch {
                        proposals: actual.work.proposals.buffer_arg(),
                        state: actual.state.buffer_arg(),
                        config: actual.config.launch(),
                        scratch: actual.work.scratch.buffer_arg(),
                        control: actual.control.buffer_arg(),
                        parameters: parameters.launch(),
                        execution: LaborExecution::Complete,
                    },
                );
                for (name, actual, expected) in [
                    ("state", &actual.state, &expected.state),
                    ("statuses", &actual.work.scratch, &expected.work.scratch),
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
        Ok(())
    }
}

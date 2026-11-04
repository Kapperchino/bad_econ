use super::*;

#[derive(CubeType)]
struct UniformBalance {
    initial: Ticks,
    amount: Ticks,
}

#[cube]
impl UniformBalance {
    fn new(
        state: &mut [u32],
        config: &EconomyConfig,
        field: u32,
        peer: u32,
        amount: Ticks,
    ) -> Self {
        UniformBalance {
            initial: value(state, config, field, peer),
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
    fn new(state: &mut [u32], config: &EconomyConfig, peer: u32) -> Self {
        let wage = times32(
            number(state, config, comptime!(Field::Wage as u32), peer),
            number(state, config, comptime!(Field::LaborMinutes as u32), peer),
        );
        let output = ticks(
            number(state, config, comptime!(Field::OutputTicks as u32), peer),
            0,
        );
        let first = number(state, config, comptime!(Field::FirstInput as u32), peer);
        let second = number(state, config, comptime!(Field::SecondInput as u32), peer);
        let input = value(state, config, comptime!(Field::InputTicks as u32), peer);
        let first_input = select(first > 0, input, zero());
        let second_input = select(second > 0, input, zero());
        let total_input = plus(first_input, second_input);
        LaborBalances {
            cash: UniformBalance::new(state, config, comptime!(Field::Cash as u32), peer, wage),
            escrow: UniformBalance::new(state, config, comptime!(Field::Escrow as u32), peer, wage),
            slots: UniformBalance::new(
                state,
                config,
                comptime!(Field::Slots as u32),
                peer,
                ticks(1, 0),
            ),
            pipeline: UniformBalance::new(
                state,
                config,
                comptime!(Field::Pipeline as u32),
                peer,
                output,
            ),
            stock: UniformBalance::new(state, config, first, peer, first_input),
            second_stock: UniformBalance::new(state, config, second, peer, second_input),
            first_used: UniformBalance::new(
                state,
                config,
                USED_FIELDS as u32 + max(first, 1) - 1,
                peer,
                first_input,
            ),
            second_used: UniformBalance::new(
                state,
                config,
                USED_FIELDS as u32 + max(second, 1) - 1,
                peer,
                second_input,
            ),
            used: UniformBalance::new(
                state,
                config,
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

    fn store(&self, state: &mut [u32], config: &EconomyConfig, peer: u32, accepted: u32) {
        set(
            state,
            config,
            comptime!(Field::Cash as u32),
            peer,
            self.cash.debited(accepted),
        );
        set(
            state,
            config,
            comptime!(Field::Escrow as u32),
            peer,
            self.escrow.credited(accepted),
        );
        set(
            state,
            config,
            comptime!(Field::Slots as u32),
            peer,
            self.slots.debited(accepted),
        );
        set(
            state,
            config,
            comptime!(Field::Pipeline as u32),
            peer,
            self.pipeline.credited(accepted),
        );
        set(
            state,
            config,
            comptime!(Field::Used as u32),
            peer,
            self.used.credited(accepted),
        );
        if self.first > 0 {
            set(
                state,
                config,
                self.first,
                peer,
                self.stock.debited(accepted),
            );
            set(
                state,
                config,
                USED_FIELDS as u32 + self.first - 1,
                peer,
                self.first_used.credited(accepted),
            );
        }
        if self.second > 0 {
            set(
                state,
                config,
                self.second,
                peer,
                self.second_stock.debited(accepted),
            );
            set(
                state,
                config,
                USED_FIELDS as u32 + self.second - 1,
                peer,
                self.second_used.credited(accepted),
            );
        }
    }
}

#[cube(launch)]
pub(in crate::simulation) fn settle_labor(
    state: &mut [u32],
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
        let balances = LaborBalances::new(state, config, peer);
        let capacity = balances.capacity(length);
        let mut accepted = 0u32;
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
                let decision = capacity.decision(prior, accepted + prefix.inclusive - eligible);
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
            balances.faults.record(control);
            balances.store(state, config, peer, accepted);
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
        for workgroup in [32, 64, 128, 256] {
            let device = Device::headless(workgroup)?;
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
                    let mut scratch = vec![0; actual.scratch.meta.num_elements()];
                    for slot in 0..case.length {
                        let actor = (slot + 17) % 513;
                        proposals[actor * ProposalLayout::WORDS + ProposalLayout::ACTOR] =
                            peer as u32;
                        scratch[slot] = peer as u32;
                        scratch[count + slot] = actor as u32;
                        let status = match (case_index % 2, slot % 7, case.length) {
                            (0, 0, 257) => EventStatus::Rejected,
                            _ => case.prior,
                        };
                        scratch[4 * count + actor] = enumflags2::BitFlags::from_flag(status).bits();
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
                    let mut original =
                        expected.parameters(Phase::Commitment, Market::Labor, 0, count)?;
                    original.sort_count = case.length as u32;
                    crate::simulation::economy::economy::launch(
                        &device.client,
                        device.count(case.length),
                        CubeDim::new_1d(workgroup),
                        expected.state.clone().into_buffer_arg(),
                        expected.actors.clone().into_buffer_arg(),
                        expected.config.launch(),
                        expected.spatial.clone().into_buffer_arg(),
                        expected.proposals.clone().into_buffer_arg(),
                        expected.scratch.clone().into_buffer_arg(),
                        expected.control.clone().into_buffer_arg(),
                        original.launch(),
                        EconomyStage::GroupExecute,
                    );
                    let mut parameters =
                        actual.parameters(Phase::Commitment, Market::Labor, 0, count)?;
                    parameters.scan = mode;
                    settle_labor::launch(
                        &device.client,
                        CubeCount::Static(1, 1, 1),
                        CubeDim::new_1d(workgroup),
                        actual.state.clone().into_buffer_arg(),
                        actual.config.launch(),
                        actual.scratch.clone().into_buffer_arg(),
                        actual.control.clone().into_buffer_arg(),
                        parameters.launch(),
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

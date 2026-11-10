use super::*;
use crate::simulation::scheduler::settle;

#[derive(CubeType, CubeLaunch)]
pub(in crate::simulation) struct PayrollInput {
    pub state: Box<[u32]>,
    pub actors: Box<[u32]>,
    pub config: EconomyConfig,
    pub proposals: Box<[u32]>,
    pub scratch: Box<[u32]>,
    pub control: Box<[Atomic<u32>]>,
    pub parameters: EconomyParameters,
}

pub(in crate::simulation) struct PayrollSettlement;
pub(in crate::simulation) struct SettlePayroll;

impl System for PayrollSettlement {
    type Input = PayrollInput;
    type Passes = gpu_passes![SettlePayroll];
}

#[cube]
impl GpuPass<PayrollInput> for SettlePayroll {
    type Output = ();

    fn execute(mut input: PayrollInput) {
        let job = CUBE_POS as u32;
        if job < input.control[RANGE_SLOTS_WORD].load() && input.control[LATCH_WORD].load() == 0 {
            let count = input.parameters.count;
            let source = input.parameters.sort_source;
            let start = input.scratch[(5 * count + job) as usize];
            let length = input.scratch[(6 * count + job) as usize];
            let peer = input.scratch[(source + start) as usize];
            let balance = value(
                &mut input.state,
                &input.config,
                comptime!(Field::Escrow as u32),
                peer,
            );
            let mut escrow = Shared::<[u64]>::new_slice(1usize);
            if UNIT_POS == 0 {
                escrow[0usize] = u64::cast_from(balance.extract(LOW_WORD))
                    | (u64::cast_from(balance.extract(HIGH_WORD)) << 32);
            }
            sync_cube();
            let mut cursor = start;
            while cursor < start + length {
                let index = cursor + UNIT_POS;
                let mut actor = 0u32;
                let mut quantity = 0u64;
                let mut event = Array::<u32>::new(EVENT_WORDS);
                let mut status = KernelEventStatus::new(comptime!(EventStatus::Inactive));
                if index < start + length {
                    actor = input.scratch[(source + count + index) as usize];
                    construct(
                        &mut input.state,
                        &input.actors,
                        &input.config,
                        &mut input.proposals,
                        &mut event[..],
                        &input.control,
                        &input.parameters,
                        actor,
                        comptime!(EventStorage::Local),
                    );
                    status.replace(settle(
                        &mut input.state,
                        &event[..],
                        &input.control,
                        0,
                        actor + input.parameters.offset,
                        input.config.actors,
                        comptime!(Posting::CheckActor),
                    ));
                    let base = (actor + input.parameters.offset) as usize * ProposalLayout::WORDS;
                    quantity = u64::cast_from(input.proposals[base + ProposalLayout::QUANTITY_LOW])
                        | (u64::cast_from(input.proposals[base + ProposalLayout::QUANTITY_HIGH])
                            << 32);
                }
                let eligible = status.contains(comptime!(EventStatus::Accepted));
                let total = parallel::sum_wide(select(eligible, quantity, 0u64));
                let funded = total.overflow == 0 && total.value <= escrow[0usize];
                sync_cube();
                if funded {
                    if UNIT_POS == 0 {
                        escrow[0usize] -= total.value;
                    }
                } else {
                    for lane in 0..CUBE_DIM {
                        if UNIT_POS == lane && eligible {
                            let affordable = quantity <= escrow[0usize];
                            escrow[0usize] -= select(affordable, quantity, 0u64);
                            if !affordable {
                                status.set(comptime!(EventStatus::Rejected));
                            }
                        }
                        sync_cube();
                    }
                }
                if status.contains(comptime!(EventStatus::Accepted)) {
                    status.replace(settle(
                        &mut input.state,
                        &event[..],
                        &input.control,
                        0,
                        actor + input.parameters.offset,
                        input.config.actors,
                        comptime!(Posting::ApplyActor),
                    ));
                }
                if index < start + length {
                    status.store(&mut input.scratch, 4 * count + actor);
                }
                sync_storage();
                cursor += CUBE_DIM;
            }
            if UNIT_POS == 0 {
                set(
                    &mut input.state,
                    &input.config,
                    comptime!(Field::Escrow as u32),
                    peer,
                    ticks(
                        u32::cast_from(escrow[0usize]),
                        u32::cast_from(escrow[0usize] >> 32),
                    ),
                );
                input.control[PROGRESS_WORD].fetch_max(length);
                input.control[COMPLETED_WORD].fetch_add(length);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{Experiment, Market, Scenario};
    use crate::simulation::{Device, Simulation, encode};
    use anyhow::Result;

    struct PayrollCase {
        escrow: u64,
        amounts: Vec<u64>,
        cash: u64,
        gross: u64,
    }

    #[test]
    fn cooperative_payroll_preserves_ordered_funding_wide_balances_and_faults_on_gpu() -> Result<()>
    {
        let scenario = Scenario::baseline(257, 42, Experiment::Baseline)?;
        let initial = scenario.initial()?;
        let count = initial.actors.len();
        let peer = 257;
        let cases = [
            PayrollCase {
                escrow: 1_000_000,
                amounts: (0..257).map(|index| (index % 13 + 1) * 17).collect(),
                cash: 0,
                gross: 0,
            },
            PayrollCase {
                escrow: 23,
                amounts: vec![29, 11, 17, 5, 9, 3, 4, 0, 1],
                cash: 0,
                gross: 0,
            },
            PayrollCase {
                escrow: u64::MAX,
                amounts: vec![u64::MAX - 31, 32, 17, 14, 1],
                cash: 0,
                gross: 0,
            },
            PayrollCase {
                escrow: u64::from(u32::MAX) * 65,
                amounts: vec![u64::from(u32::MAX); 65],
                cash: 100,
                gross: 0,
            },
            PayrollCase {
                escrow: 17,
                amounts: vec![29, 11, 7, 5, 3, 1],
                cash: u64::MAX - 5,
                gross: u64::MAX - 7,
            },
        ];
        for workgroup in [32, 64, 128, 256] {
            let device = Device::headless(workgroup)?;
            let mut actual = Simulation::new(scenario.clone(), device.clone())?;
            let mut expected = Simulation::new(scenario.clone(), device.clone())?;
            for payroll in [Feature::Disabled, Feature::Enabled] {
                for simulation in [&mut actual, &mut expected] {
                    simulation.config.payroll = payroll;
                    simulation.config.tax_base_rate = 100;
                    simulation.config.employee_charge = 50;
                }
                for (case_index, case) in cases.iter().enumerate() {
                    let mut state = initial.clone();
                    state.set(Field::Escrow, peer, case.escrow);
                    let mut proposals = vec![0; count * ProposalLayout::WORDS];
                    let mut scratch = vec![0; actual.scratch.meta.num_elements()];
                    for (actor, &amount) in case.amounts.iter().enumerate() {
                        state.set(Field::Due, actor, amount);
                        state.set(Field::Employer, actor, peer as u64 + 1);
                        state.set(Field::Cash, actor, case.cash);
                        state.set(Field::GrossIncome, actor, case.gross);
                        state.set(Field::PayrollReserve, actor, 0);
                        let base = actor * ProposalLayout::WORDS;
                        proposals[base + ProposalLayout::ACTOR] = peer as u32;
                        proposals[base + ProposalLayout::QUANTITY_LOW] = amount as u32;
                        proposals[base + ProposalLayout::QUANTITY_HIGH] = (amount >> 32) as u32;
                        scratch[actor] = peer as u32;
                        scratch[count + actor] = actor as u32;
                    }
                    scratch[6 * count] = case.amounts.len() as u32;
                    let mut control = vec![0; actual.control.meta.num_elements()];
                    control[CONSTRUCTED_EVENTS_WORD] = case.amounts.len() as u32;
                    control[RANGE_SLOTS_WORD] = 1;
                    for simulation in [&actual, &expected] {
                        device.write(
                            &simulation.state,
                            bytemuck::cast_slice(&encode(&state.values)),
                        );
                        device.write(&simulation.proposals, &proposals);
                        device.write(&simulation.scratch, &scratch);
                        device.write(&simulation.control, &control);
                    }
                    let parameters =
                        actual.parameters(Phase::WorkPayment, Market::Payroll, 0, count)?;
                    EconomySystem::launch(
                        &device,
                        device.count(1),
                        CubeDim::new_1d(workgroup),
                        EconomyInputLaunch {
                            state: expected.state.clone().into_buffer_arg(),
                            actors: expected.actors.clone().into_buffer_arg(),
                            config: expected.config.launch(),
                            spatial: expected.spatial.clone().into_buffer_arg(),
                            proposals: expected.proposals.clone().into_buffer_arg(),
                            bundles: expected.scratch.clone().into_buffer_arg(),
                            control: expected.control.clone().into_buffer_arg(),
                            parameters: parameters.launch(),
                            stage: EconomyStage::GroupExecute,
                        },
                    );
                    PayrollSettlement::launch(
                        &device,
                        CubeCount::Static(1, 1, 1),
                        CubeDim::new_1d(workgroup),
                        PayrollInputLaunch {
                            state: actual.state.clone().into_buffer_arg(),
                            actors: actual.actors.clone().into_buffer_arg(),
                            config: actual.config.launch(),
                            proposals: actual.proposals.clone().into_buffer_arg(),
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
                        for (word, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                            assert_eq!(
                                actual, expected,
                                "{name} word {word}: workgroup {workgroup}, case {case_index}"
                            );
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

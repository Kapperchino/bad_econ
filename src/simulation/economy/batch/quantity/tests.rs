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
        match self {
            Self::Accepted => EventStatus::Accepted,
            Self::Rejected => EventStatus::Rejected,
            Self::Faulted => EventStatus::Faulted,
            Self::Mixed => match slot % 7 {
                0 => EventStatus::Rejected,
                1 => EventStatus::Faulted,
                _ => EventStatus::Accepted,
            },
        }
    }
}

struct QuantityCase {
    cash: u64,
    stock: u64,
    sales: u64,
    quote: u32,
    length: usize,
    quantities: [u64; 4],
    prior: PriorPattern,
}

#[test]
fn quantity_batches_match_ordered_settlement_with_shortages_and_wide_faults_on_gpu() -> Result<()> {
    let scenario = Scenario::industrial(513, 42, Experiment::Concentrated)?;
    let initial = scenario.initial()?;
    let peer = 513;
    let count = initial.actors.len();
    let cases = [
        QuantityCase {
            cash: 0,
            stock: 1_000_000,
            sales: 0,
            quote: 100,
            length: 513,
            quantities: [1, 7, 13, 3],
            prior: PriorPattern::Accepted,
        },
        QuantityCase {
            cash: 0,
            stock: 4,
            sales: 0,
            quote: 3,
            length: 257,
            quantities: [12, 3, 2, 1],
            prior: PriorPattern::Accepted,
        },
        QuantityCase {
            cash: 0,
            stock: 0,
            sales: 0,
            quote: 3,
            length: 513,
            quantities: [1, 7, 13, 3],
            prior: PriorPattern::Accepted,
        },
        QuantityCase {
            cash: 100,
            stock: 0,
            sales: 0,
            quote: 0,
            length: 65,
            quantities: [0, 0, 0, 0],
            prior: PriorPattern::Accepted,
        },
        QuantityCase {
            cash: 100,
            stock: 0,
            sales: 0,
            quote: 3,
            length: 65,
            quantities: [0, 8, 0, 3],
            prior: PriorPattern::Mixed,
        },
        QuantityCase {
            cash: 0,
            stock: 1_000_000,
            sales: 0,
            quote: 3,
            length: 257,
            quantities: [1, 7, 13, 3],
            prior: PriorPattern::Mixed,
        },
        QuantityCase {
            cash: u32::MAX as u64 - 1,
            stock: u32::MAX as u64 * 200,
            sales: u32::MAX as u64,
            quote: 3,
            length: 129,
            quantities: [u32::MAX as u64, u32::MAX as u64 + 1, 1, 13],
            prior: PriorPattern::Accepted,
        },
        QuantityCase {
            cash: u64::MAX - 6,
            stock: 200,
            sales: 0,
            quote: 3,
            length: 65,
            quantities: [12, 1, 2, 0],
            prior: PriorPattern::Accepted,
        },
        QuantityCase {
            cash: 0,
            stock: u64::MAX,
            sales: u64::MAX - 4,
            quote: 0,
            length: 65,
            quantities: [12, 1, 2, 0],
            prior: PriorPattern::Accepted,
        },
        QuantityCase {
            cash: 0,
            stock: u64::MAX,
            sales: 0,
            quote: 2,
            length: 31,
            quantities: [u64::MAX, 1, 3, 0],
            prior: PriorPattern::Accepted,
        },
        QuantityCase {
            cash: 0,
            stock: u64::MAX,
            sales: 0,
            quote: 0,
            length: 129,
            quantities: [1 << 63, 1 << 63, 2, 1],
            prior: PriorPattern::Accepted,
        },
        QuantityCase {
            cash: u64::MAX,
            stock: 0,
            sales: u64::MAX,
            quote: u32::MAX,
            length: 65,
            quantities: [u64::MAX, 1, 0, 3],
            prior: PriorPattern::Mixed,
        },
        QuantityCase {
            cash: u64::MAX,
            stock: u64::MAX,
            sales: 0,
            quote: 0,
            length: 65,
            quantities: [u64::MAX, 1, 0, 3],
            prior: PriorPattern::Rejected,
        },
        QuantityCase {
            cash: 0,
            stock: 1_000_000,
            sales: 0,
            quote: 100,
            length: 257,
            quantities: [1, 7, 13, 3],
            prior: PriorPattern::Faulted,
        },
    ];
    for workgroup in [32, 64, 128, 256] {
        let device = crate::simulation::tests::device(workgroup)?;
        let actual = Simulation::new(scenario.clone(), device.clone())?;
        let mut expected = Simulation::new(scenario.clone(), device.clone())?;
        expected.market_execution = MarketExecution::Original;
        for (case_index, case) in cases.iter().enumerate() {
            let mut state = initial.clone();
            state.set(Field::Cash, peer, case.cash);
            state.set(Field::Coal, peer, case.stock);
            state.set(Field::Sales, peer, case.sales);
            state.set(Field::Price, peer, case.quote.into());
            let words = encode(&state.values);
            let mut proposals = vec![0; count * ProposalLayout::WORDS];
            let mut scratch = vec![0; actual.scratch.meta.num_elements()];
            for slot in 0..case.length {
                let actor = (slot + 17) % 513;
                let quantity = case.quantities[slot % case.quantities.len()];
                proposals[actor * ProposalLayout::WORDS + ProposalLayout::ACTOR] = peer as u32;
                proposals[actor * ProposalLayout::WORDS + ProposalLayout::QUANTITY_LOW] =
                    quantity as u32;
                proposals[actor * ProposalLayout::WORDS + ProposalLayout::QUANTITY_HIGH] =
                    (quantity >> 32) as u32;
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
            let market = Market::Consumer(Good::Coal);
            let mut original = expected.parameters(Phase::ProvisionPurchase, market, 0, count)?;
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
            let parameters = actual.parameters(Phase::ProvisionPurchase, market, 0, count)?;
            QuantitySettlement::launch(
                &device,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(workgroup),
                QuantityInputLaunch {
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
    Ok(())
}

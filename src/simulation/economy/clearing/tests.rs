use super::*;
use crate::domain::{Experiment, Market, Scenario};
use crate::simulation::{MarketExecution, Simulation, encode};
use anyhow::Result;

#[derive(Clone, Copy, Debug)]
enum Seller {
    Producer,
    Household,
    Retailer,
}

#[derive(Debug)]
struct QueueCase {
    stock: u64,
    cash: u64,
    sales: u64,
    quote: u32,
}

#[test]
fn exhausted_purchase_queues_match_serial_settlement_and_faults_on_gpu() -> Result<()> {
    let scenario = Scenario::industrial(513, 42, Experiment::Concentrated)?;
    let initial = scenario.initial()?;
    let count = initial.actors.len();
    let cases = [
        QueueCase {
            stock: 0,
            cash: 0,
            sales: 0,
            quote: 3,
        },
        QueueCase {
            stock: 1,
            cash: 0,
            sales: 0,
            quote: 3,
        },
        QueueCase {
            stock: 19,
            cash: 0,
            sales: 0,
            quote: 3,
        },
        QueueCase {
            stock: 300,
            cash: 0,
            sales: 0,
            quote: 3,
        },
        QueueCase {
            stock: 10_000,
            cash: 0,
            sales: 0,
            quote: 3,
        },
        QueueCase {
            stock: 0,
            cash: u64::MAX,
            sales: u64::MAX,
            quote: u32::MAX,
        },
        QueueCase {
            stock: 19,
            cash: u64::MAX - 57,
            sales: u64::MAX - 19,
            quote: 3,
        },
        QueueCase {
            stock: 19,
            cash: u64::MAX,
            sales: 0,
            quote: 0,
        },
    ];
    let workgroup = 64;
    let device = crate::simulation::tests::device(workgroup)?;
    let actual = Simulation::new(scenario.clone(), device.clone())?;
    let mut expected = Simulation::new(scenario.clone(), device.clone())?;
    expected.market_execution = MarketExecution::Original;
    for seller in [Seller::Producer, Seller::Household, Seller::Retailer] {
        for case in &cases {
            let peer = match seller {
                Seller::Household => 512,
                _ => 513,
            };
            let mut state = initial.clone();
            let reserve = match seller {
                Seller::Household => {
                    state.get(Field::FoodNeed, peer)
                        * u64::from(actual.config.coverage + actual.config.ring)
                }
                _ => 0,
            };
            state.set(Field::Bread, peer, case.stock + reserve);
            state.set(Field::Cash, peer, case.cash);
            state.set(Field::Sales, peer, case.sales);
            state.set(Field::Price, peer, u64::from(case.quote));
            if let Seller::Retailer = seller {
                state.set(Field::EnterpriseModel, peer, 1);
                state.set(Field::Bread, peer, 10_000);
                state.set(Field::Sales, peer, 0);
                state.set(Field::RetailOpen, peer, case.stock);
            }
            let mut proposals = vec![0; count * ProposalLayout::WORDS];
            let mut scratch = vec![NO_ACTOR; actual.scratch.meta.num_elements()];
            for actor in 0..512 {
                let quantity = match actor % 17 {
                    0 => 0,
                    1 => u64::MAX,
                    _ => (actor % 7 + 1) as u64,
                };
                let status = match actor % 13 {
                    0 => EventStatus::Rejected,
                    1 => EventStatus::Faulted,
                    _ => EventStatus::Accepted,
                };
                let base = actor * ProposalLayout::WORDS;
                proposals[base + ProposalLayout::ACTOR] = peer as u32;
                proposals[base + ProposalLayout::QUANTITY_LOW] = quantity as u32;
                proposals[base + ProposalLayout::QUANTITY_HIGH] = (quantity >> 32) as u32;
                scratch[actor] = peer as u32;
                scratch[count + actor] = actor as u32;
                scratch[4 * count + actor] = enumflags2::BitFlags::from_flag(status).bits();
            }
            scratch[5 * count] = 0;
            scratch[6 * count] = 512;
            let mut control = vec![0; actual.control.meta.num_elements()];
            control[CONSTRUCTED_EVENTS_WORD] = 512;
            control[RANGE_SLOTS_WORD] = 1;
            for simulation in [&actual, &expected] {
                device.write(
                    &simulation.state,
                    bytemuck::cast_slice(&encode(&state.values)),
                );
                device.write(&simulation.proposals, &proposals);
                device.write(&simulation.scratch, &scratch);
                device.write(&simulation.control, &control);
                let mut parameters =
                    simulation.parameters(Phase::ProvisionPurchase, Market::Household, 0, count)?;
                parameters.sort_count = 512;
                let stages: &[EconomyStage] = match simulation.market_execution {
                    MarketExecution::Grouped => {
                        &[EconomyStage::GroupExecute, EconomyStage::GroupRemainder]
                    }
                    _ => &[EconomyStage::GroupExecute],
                };
                for &stage in stages {
                    EconomySystem::launch(
                        &device,
                        device.count(512),
                        CubeDim::new_1d(workgroup),
                        EconomyInputLaunch {
                            state: simulation.state.clone().into_buffer_arg(),
                            actors: simulation.actors.clone().into_buffer_arg(),
                            config: simulation.config.launch(),
                            spatial: simulation.spatial.clone().into_buffer_arg(),
                            proposals: simulation.proposals.clone().into_buffer_arg(),
                            bundles: simulation.scratch.clone().into_buffer_arg(),
                            control: simulation.control.clone().into_buffer_arg(),
                            parameters: parameters.launch(),
                            stage,
                        },
                    );
                }
            }
            assert_eq!(
                device.read(&actual.state)?,
                device.read(&expected.state)?,
                "state: {workgroup} {seller:?} {case:?}"
            );
            assert_eq!(
                device.read_range(&actual.scratch, 4 * count, count)?,
                device.read_range(&expected.scratch, 4 * count, count)?,
                "statuses: {workgroup} {seller:?} {case:?}"
            );
            assert_eq!(
                device.read(&actual.control)?,
                device.read(&expected.control)?,
                "faults: {workgroup} {seller:?} {case:?}"
            );
        }
    }
    Ok(())
}

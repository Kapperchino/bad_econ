use super::*;
use crate::domain::{Experiment, Market, Scenario};
use crate::simulation::{MarketExecution, Simulation, encode};
use anyhow::Result;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum BuyerExecution {
    Direct,
    Event,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum BuyerPosting {
    Check,
    Apply,
}

impl BuyerPosting {
    fn posting(self) -> Posting {
        match self {
            Self::Check => Posting::CheckActor,
            Self::Apply => Posting::ApplyActor,
        }
    }
}

#[derive(CubeType, CubeLaunch)]
struct BuyerProbeInput {
    state: Box<[u32]>,
    actors: Box<[u32]>,
    config: EconomyConfig,
    proposals: Box<[u32]>,
    control: Box<[Atomic<u32>]>,
    statuses: Box<[u32]>,
    parameters: EconomyParameters,
    #[cube(comptime)]
    execution: BuyerExecution,
    #[cube(comptime)]
    posting: BuyerPosting,
}

struct BuyerProbe;
struct CompareBuyer;

#[derive(Debug, PartialEq, Eq)]
struct BuyerOutcome {
    state: Vec<u32>,
    control: Vec<u32>,
    statuses: Vec<u32>,
}

impl System for BuyerProbe {
    type Input = BuyerProbeInput;
    type Passes = gpu_passes![CompareBuyer];
}

#[cube]
impl GpuPass<BuyerProbeInput> for CompareBuyer {
    type Output = ();

    fn execute(mut input: BuyerProbeInput) {
        let id = ABSOLUTE_POS as u32;
        if id < input.parameters.count {
            let actor = id + input.parameters.offset;
            let other =
                input.proposals[actor as usize * ProposalLayout::WORDS + ProposalLayout::ACTOR];
            let mut status = KernelEventStatus::new(comptime!(EventStatus::Rejected));
            match comptime!(input.execution) {
                BuyerExecution::Direct => status.replace(buyer_settlement(
                    &mut input.state,
                    &input.actors,
                    &input.config,
                    &mut input.proposals,
                    &input.control,
                    &input.parameters,
                    actor,
                    other,
                    comptime!(input.posting.posting()),
                )),
                BuyerExecution::Event => status.replace(event_settlement(
                    &mut input.state,
                    &input.actors,
                    &input.config,
                    &mut input.proposals,
                    &input.control,
                    &input.parameters,
                    id,
                    comptime!(input.posting.posting()),
                )),
            }
            status.store(&mut input.statuses, id);
        }
    }
}

#[test]
fn direct_goods_buyers_match_event_fields_rejections_and_faults_on_gpu() -> Result<()> {
    let device = crate::simulation::tests::device(64)?;
    let scenario = Scenario::baseline(135, 42, Experiment::Baseline)?;
    let simulation = Simulation::new(scenario.clone(), device.clone())?;
    let count = scenario.actors();
    for market in [
        Market::Household,
        Market::Input(Good::Bread),
        Market::Consumer(Good::Coal),
    ] {
        let good = match market {
            Market::Household => Good::Bread,
            Market::Input(good) | Market::Consumer(good) => good,
            _ => unreachable!(),
        };
        let mut state = scenario.initial()?;
        let mut proposals = vec![0; count * ProposalLayout::WORDS];
        let mut parameters = simulation.parameters(Phase::ProvisionPurchase, market, 7, 64)?;
        parameters.day = 23;
        for actor in 7..71 {
            let peer = actor + 64;
            let case = actor % 12;
            let quantity = match case {
                2 | 10 => u64::MAX,
                9 => 0,
                _ => 3,
            };
            let cash = match case {
                1 | 7 => 0,
                _ => u64::MAX,
            };
            state.set(Field::Cash, actor, cash);
            state.set(Field::FirstInput, actor, good as u64);
            state.set(Field::SecondInput, actor, good as u64);
            state.set(Field::FirstInputCost, actor, u64::MAX);
            state.set(Field::SecondInputCost, actor, u64::MAX);
            state.set(Field::ReplacementCost, actor, u64::MAX);
            state.set(Field::EnterpriseModel, actor, 1);
            state.set(Field::Price, peer, (1u64 << 40) + 17);
            state.set(Field::inventory(good), peer, u64::MAX);
            if case == 11 {
                state.set(Field::EnterpriseModel, peer, 1);
                state.set(Field::RetailOpen, peer, 0);
            }
            let overflow_field = match case {
                3 => {
                    let buyer = state.actors.at(actor).position;
                    let seller = state.actors.at(peer).position;
                    let distance = buyer.x.abs_diff(seller.x) + buyer.y.abs_diff(seller.y);
                    let days = distance.div_ceil(simulation.config.shipment_speed).max(1);
                    BASE_FIELDS
                        + (good as usize - 1) * scenario.ring()
                        + (parameters.day + days) as usize % scenario.ring()
                }
                4 => TRANSIT_FIELDS + good as usize - 1,
                5 => Field::Transit as usize,
                _ => Field::Purchased as usize,
            };
            if matches!(case, 3..=7) {
                state.values[overflow_field * count + actor] = u64::MAX;
            }
            let base = actor * ProposalLayout::WORDS;
            proposals[base + ProposalLayout::ACTOR] = peer as u32;
            proposals[base + ProposalLayout::QUANTITY_LOW] = quantity as u32;
            proposals[base + ProposalLayout::QUANTITY_HIGH] = (quantity >> 32) as u32;
        }
        let proposals = device.tensor(&proposals);
        for posting in [BuyerPosting::Check, BuyerPosting::Apply] {
            let mut outcomes = Vec::new();
            for execution in [BuyerExecution::Direct, BuyerExecution::Event] {
                let state = device.tensor(bytemuck::cast_slice(&encode(&state.values)));
                let control = device.tensor(&vec![0; CONTROL_WORDS]);
                let statuses = device.empty(64);
                BuyerProbe::launch(
                    &device,
                    device.count(64),
                    CubeDim::new_1d(device.workgroup),
                    BuyerProbeInputLaunch {
                        state: state.clone().into_buffer_arg(),
                        actors: simulation.actors.clone().into_buffer_arg(),
                        config: simulation.config.launch(),
                        proposals: proposals.clone().into_buffer_arg(),
                        control: control.clone().into_buffer_arg(),
                        statuses: statuses.clone().into_buffer_arg(),
                        parameters: parameters.launch(),
                        execution,
                        posting,
                    },
                );
                outcomes.push(BuyerOutcome {
                    state: device.read(&state)?,
                    control: device.read(&control)?,
                    statuses: device.read(&statuses)?,
                });
            }
            assert_eq!(outcomes[0], outcomes[1], "{market:?}");
        }
    }
    Ok(())
}

struct CompactedBuyerCase {
    cash: u64,
    purchased: u64,
}

#[test]
fn compacted_goods_buyers_match_event_checks_rejections_and_faults_on_gpu() -> Result<()> {
    use crate::simulation::economy::{markets, search};

    let device = crate::simulation::tests::device(64)?;
    let scenario = Scenario::industrial(97, 42, Experiment::Baseline)?;
    for market in [Market::Household, Market::Consumer(Good::Coal)] {
        let good = match market {
            Market::Consumer(good) => good,
            _ => Good::Bread,
        };
        for case in [
            CompactedBuyerCase {
                cash: 5000,
                purchased: 0,
            },
            CompactedBuyerCase {
                cash: 0,
                purchased: 0,
            },
            CompactedBuyerCase {
                cash: 5000,
                purchased: u64::MAX,
            },
        ] {
            let simulation = Simulation::new(scenario.clone(), device.clone())?;
            let mut state = scenario.initial()?;
            let count = state.actors.len();
            let peer = state
                .actors
                .firm_indices()
                .find(|&actor| state.actors.at(actor).kind == good as u32)
                .expect("supplier");
            state.set(Field::Cash, 0, case.cash);
            state.set(Field::Purchased, 0, case.purchased);
            state.set(Field::Price, peer, 17);
            state.set(Field::inventory(good), peer, 100);
            let encoded = bytemuck::cast_slice(&encode(&state.values)).to_vec();
            device.write(&simulation.state, &encoded);
            let expected_state = device.tensor(&encoded);
            let mut parameters =
                simulation.parameters(Phase::ProvisionPurchase, market, 0, count)?;
            parameters.day = 23;
            parameters.settlement_rotation = 5;
            let mut proposals = vec![NO_ACTOR; count * ProposalLayout::WORDS];
            proposals[ProposalLayout::ACTOR] = peer as u32;
            proposals[ProposalLayout::QUANTITY_LOW] = 3;
            proposals[ProposalLayout::QUANTITY_HIGH] = 0;
            proposals[ProposalLayout::DISTANCE] = 0;
            device.write(&simulation.proposals, &proposals);
            let mut control = vec![0; simulation.control.meta.num_elements()];
            control[markets::SEARCHES_WORD] = 1;
            device.write(&simulation.control, &control);
            device.write(&simulation.searches.requests, &[0]);
            let masks = count.div_ceil(32);
            let blocks = markets::search_blocks(count);
            let position = count - parameters.settlement_rotation as usize;
            let mut successes = vec![0; simulation.searches.successes.meta.num_elements()];
            successes[position / 32] = 1 << (position % 32);
            for block in 0..blocks {
                successes[masks + block] =
                    u32::from(block >= position / markets::COMPACTION_BLOCK as usize);
            }
            device.write(&simulation.searches.successes, &successes);
            search::MarketSearch::launch(
                &device,
                device.count(1),
                CubeDim::new_1d(device.workgroup),
                search::SearchInputLaunch {
                    state: simulation.state.clone().into_buffer_arg(),
                    actors: simulation.actors.clone().into_buffer_arg(),
                    config: simulation.config.launch(),
                    spatial: simulation.spatial.clone().into_buffer_arg(),
                    proposals: simulation.proposals.clone().into_buffer_arg(),
                    searches: simulation.searches.requests.clone().into_buffer_arg(),
                    successes: simulation.searches.successes.clone().into_buffer_arg(),
                    scratch: simulation.scratch.clone().into_buffer_arg(),
                    control: simulation.control.clone().into_buffer_arg(),
                    parameters: parameters.launch(),
                    status_reset: search::StatusReset::Active,
                    stage: search::SearchStage::Compact,
                },
            );
            let expected_control = device.tensor(&vec![0; CONTROL_WORDS]);
            let expected_status = device.empty(1);
            BuyerProbe::launch(
                &device,
                device.count(1),
                CubeDim::new_1d(device.workgroup),
                BuyerProbeInputLaunch {
                    state: expected_state.clone().into_buffer_arg(),
                    actors: simulation.actors.clone().into_buffer_arg(),
                    config: simulation.config.launch(),
                    proposals: simulation.proposals.clone().into_buffer_arg(),
                    control: expected_control.clone().into_buffer_arg(),
                    statuses: expected_status.clone().into_buffer_arg(),
                    parameters: EconomyParameters {
                        count: 1,
                        ..parameters
                    }
                    .launch(),
                    execution: BuyerExecution::Event,
                    posting: BuyerPosting::Check,
                },
            );
            assert_eq!(
                device.read(&simulation.state)?,
                device.read(&expected_state)?
            );
            assert_eq!(
                device.read_range(&simulation.scratch, 4 * count, 1)?,
                device.read(&expected_status)?,
            );
            let control = device.read(&simulation.control)?;
            assert_eq!(
                control[crate::simulation::flags::FAULT_WORD],
                device.read(&expected_control)?[crate::simulation::flags::FAULT_WORD]
            );
            assert_eq!(control[CONSTRUCTED_EVENTS_WORD], 1);
            assert_eq!(device.read_range(&simulation.scratch, 0, 1)?, [peer as u32]);
            assert_eq!(device.read_range(&simulation.scratch, count, 1)?, [0]);
        }
    }
    Ok(())
}

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

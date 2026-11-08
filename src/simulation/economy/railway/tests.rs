use super::*;
use crate::domain::{Experiment, Scenario};
use crate::simulation::Simulation;
use anyhow::Result;

#[derive(CubeType, CubeLaunch)]
struct QuoteCheckInput {
    actors: Box<[u32]>,
    config: EconomyConfig,
    mismatches: Box<[Atomic<u32>]>,
}

struct QuoteCheck;
struct CompareQuotes;

impl System for QuoteCheck {
    type Input = QuoteCheckInput;
    type Passes = gpu_passes![CompareQuotes];
}

#[cube]
impl GpuPass<QuoteCheckInput> for CompareQuotes {
    type Output = ();

    fn execute(input: QuoteCheckInput) {
        let index = ABSOLUTE_POS as u32;
        let count = input.config.actors;
        if index < count * count {
            let buyer = index / count;
            let seller = index % count;
            if distance(&input.actors, buyer, seller) <= input.config.search_radius {
                let mut expected = animal_route(&input.actors, &input.config, buyer, seller);
                for company in 0..input.config.railway.company_count {
                    let path = candidate(&input.actors, &input.config, buyer, seller, company);
                    if path.preferred_to(&expected) {
                        expected.replace(&path);
                    }
                }
                let actual = route(&input.actors, &input.config, buyer, seller);
                let node = station(&input.actors, &input.config, buyer);
                let access = access_distance(&input.actors, &input.config, buyer, node);
                let equal = actual.days == expected.days
                    && actual.fee == expected.fee
                    && actual.carrier == expected.carrier
                    && actual.foreign_distance == expected.foreign_distance
                    && input.config.railway.access[buyer as usize * 2] == node
                    && input.config.railway.access[buyer as usize * 2 + 1] == access;
                if !equal {
                    input.mismatches[0usize].fetch_add(1);
                }
            }
        }
    }
}

#[test]
fn cached_quotes_match_exhaustive_carriers_on_gpu() -> Result<()> {
    let config = serde_json::from_str(include_str!("../../../../scenarios/railway.json"))?;
    let railway = Scenario::new(config)?;
    let mut slow = railway.config().clone();
    slow.shipment_speed = 16;
    slow.railway.speed_multiplier = 2;
    let scenarios = [
        railway,
        Scenario::new(slow)?,
        Scenario::industrial(193, 42, Experiment::Baseline)?,
        Scenario::industrial(193, 42, Experiment::Transport)?,
    ];
    for scenario in scenarios {
        let device = crate::simulation::tests::device(64)?;
        let mut simulation = Simulation::new(scenario, device.clone())?;
        for _ in 0..3 {
            simulation.advance()?;
            for stage in [RailStage::Access, RailStage::Routes, RailStage::Quotes] {
                simulation.railways(stage);
            }
            let mismatches = device.tensor(&[0]);
            let count = simulation.metadata.actors.len();
            QuoteCheck::launch(
                &device,
                device.count(count * count),
                CubeDim::new_1d(device.workgroup),
                QuoteCheckInputLaunch {
                    actors: simulation.actors.clone().into_buffer_arg(),
                    config: simulation.config.launch(),
                    mismatches: mismatches.clone().into_buffer_arg(),
                },
            );
            assert_eq!(device.read(&mismatches)?, [0]);
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum TradeExecution {
    Ledger,
    Direct,
    DeferredRejection,
}

#[derive(CubeType, CubeLaunch)]
struct TradeCheckInput {
    state: Box<[u32]>,
    actors: Box<[u32]>,
    config: EconomyConfig,
    proposals: Box<[u32]>,
    control: Box<[Atomic<u32>]>,
    status: Box<[u32]>,
    parameters: EconomyParameters,
    buyer: u32,
    seller: u32,
    #[cube(comptime)]
    execution: TradeExecution,
}

struct TradeCheck;
struct CompareTrades;

impl System for TradeCheck {
    type Input = TradeCheckInput;
    type Passes = gpu_passes![CompareTrades];
}

#[cube]
impl GpuPass<TradeCheckInput> for CompareTrades {
    type Output = ();

    fn execute(mut input: TradeCheckInput) {
        let buyer = input.buyer;
        let seller = input.seller;
        let quantity = ticks(
            input.proposals[buyer as usize * ProposalLayout::WORDS + ProposalLayout::QUANTITY_LOW],
            input.proposals[buyer as usize * ProposalLayout::WORDS + ProposalLayout::QUANTITY_HIGH],
        );
        let path = FreightRoute {
            days: 2,
            fee: 17,
            origin: 0,
            destination: 0,
            carrier: 0,
            foreign_distance: 0,
        };
        let funded = !lt(
            value(
                &mut input.state,
                &input.config,
                comptime!(Field::Cash as u32),
                seller,
            ),
            multiply(&input.control, quantity, path.fee),
        );
        match comptime!(input.execution) {
            TradeExecution::DeferredRejection => {
                let cash = value(
                    &mut input.state,
                    &input.config,
                    comptime!(Field::Cash as u32),
                    seller,
                );
                let sales = value(
                    &mut input.state,
                    &input.config,
                    comptime!(Field::Sales as u32),
                    seller,
                );
                reject_remainder(
                    &mut input.state,
                    &input.actors,
                    &input.config,
                    &input.parameters,
                    seller,
                    0,
                );
                set32(
                    &mut input.state,
                    &input.config,
                    comptime!(Field::Cash as u32),
                    seller,
                    0,
                );
                set32(
                    &mut input.state,
                    &input.config,
                    comptime!(Field::Sales as u32),
                    seller,
                    0,
                );
                finalize_rejection(
                    &mut input.state,
                    &input.actors,
                    &input.config,
                    &input.control,
                    &input.parameters,
                    &input.proposals,
                    &mut input.status,
                    buyer,
                    seller,
                    0,
                );
                let status =
                    KernelEventStatus::load(&mut input.status, 4 * input.parameters.count + buyer);
                status.store(&mut input.status, 0);
                set(
                    &mut input.state,
                    &input.config,
                    comptime!(Field::Cash as u32),
                    seller,
                    cash,
                );
                set(
                    &mut input.state,
                    &input.config,
                    comptime!(Field::Sales as u32),
                    seller,
                    sales,
                );
            }
            TradeExecution::Ledger => {
                let animal = animal_route(&input.actors, &input.config, buyer, seller);
                let mut event = Array::<u32>::new(EVENT_WORDS);
                construct_route(
                    &mut input.state,
                    &input.actors,
                    &input.config,
                    &mut input.proposals,
                    &mut event[..],
                    &input.control,
                    &input.parameters,
                    &animal,
                    buyer,
                    comptime!(EventStorage::Local),
                );
                let eligible = scheduler::settle(
                    &mut input.state,
                    &event[..],
                    &input.control,
                    0,
                    0,
                    1,
                    comptime!(Posting::CheckAll),
                );
                if eligible.contains(comptime!(EventStatus::Accepted)) && funded {
                    construct_route(
                        &mut input.state,
                        &input.actors,
                        &input.config,
                        &mut input.proposals,
                        &mut event[..],
                        &input.control,
                        &input.parameters,
                        &path,
                        buyer,
                        comptime!(EventStorage::Local),
                    );
                }
                apply(&mut input.state, &event[..], &input.control, 0).store(&mut input.status, 0);
            }
            TradeExecution::Direct => {
                let mut settlement = GoodsSettlement::new(
                    &mut input.state,
                    &input.actors,
                    &input.config,
                    &input.control,
                    &input.parameters,
                    buyer,
                    seller,
                    quantity,
                );
                if settlement.status.contains(comptime!(EventStatus::Accepted)) && funded {
                    settlement.reroute(
                        &mut input.state,
                        &input.config,
                        &input.control,
                        &input.parameters,
                        buyer,
                        quantity,
                        &path,
                    );
                }
                settlement.apply(
                    &mut input.state,
                    &input.config,
                    &input.parameters,
                    buyer,
                    seller,
                );
                settlement.status.store(&mut input.status, 0);
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum TradeCase {
    Ordinary,
    Wide,
    Unaffordable,
    Unstocked,
    UnstockedSellerOverflow,
    UnstockedSalesOverflow,
    UnstockedTransitOverflow,
    UnstockedPurchaseOverflow,
    UnfundedFreight,
    SellerOverflow,
    TransitOverflow,
    PurchaseOverflow,
    AnimalOverflow,
    RailOverflow,
    RetailCapacity,
    ExhaustedRetailer,
}

#[test]
fn direct_goods_settlement_preserves_ledger_balances_and_faults_on_gpu() -> Result<()> {
    use crate::simulation::{Market, encode};
    let scenario = Scenario::baseline(3, 42, Experiment::Concentrated)?;
    let device = crate::simulation::tests::device(64)?;
    let simulation = Simulation::new(scenario.clone(), device.clone())?;
    for market in [
        Market::Household,
        Market::Consumer(Good::Bread),
        Market::Input(Good::Bread),
    ] {
        let buyer = match market {
            Market::Input(_) => 4,
            _ => 0,
        };
        let seller = 5;
        for case in [
            TradeCase::Ordinary,
            TradeCase::Wide,
            TradeCase::Unaffordable,
            TradeCase::Unstocked,
            TradeCase::UnstockedSellerOverflow,
            TradeCase::UnstockedSalesOverflow,
            TradeCase::UnstockedTransitOverflow,
            TradeCase::UnstockedPurchaseOverflow,
            TradeCase::UnfundedFreight,
            TradeCase::SellerOverflow,
            TradeCase::TransitOverflow,
            TradeCase::PurchaseOverflow,
            TradeCase::AnimalOverflow,
            TradeCase::RailOverflow,
            TradeCase::RetailCapacity,
            TradeCase::ExhaustedRetailer,
        ] {
            let mut initial = scenario.initial()?;
            initial.set(Field::Cash, buyer, 1 << 48);
            initial.set(Field::Cash, seller, 1 << 48);
            initial.set(Field::Bread, seller, 1 << 48);
            initial.set(Field::Price, seller, 10);
            initial.set(Field::FirstInput, buyer, Good::Bread as u64);
            initial.set(Field::SecondInput, buyer, Good::Bread as u64);
            initial.set(Field::EnterpriseModel, buyer, 1);
            let quantity = match case {
                TradeCase::Wide => u64::from(u32::MAX) + 5,
                _ => 3,
            };
            match case {
                TradeCase::Unaffordable => initial.set(Field::Cash, buyer, 0),
                TradeCase::Unstocked => initial.set(Field::Bread, seller, 0),
                TradeCase::UnfundedFreight => initial.set(Field::Cash, seller, 0),
                TradeCase::SellerOverflow | TradeCase::UnstockedSellerOverflow => {
                    initial.set(Field::Cash, seller, u64::MAX)
                }
                TradeCase::UnstockedSalesOverflow => initial.set(Field::Sales, seller, u64::MAX),
                TradeCase::TransitOverflow | TradeCase::UnstockedTransitOverflow => {
                    initial.set(Field::Transit, buyer, u64::MAX)
                }
                TradeCase::PurchaseOverflow | TradeCase::UnstockedPurchaseOverflow => {
                    initial.set(Field::Purchased, buyer, u64::MAX)
                }
                TradeCase::AnimalOverflow | TradeCase::RailOverflow => {
                    let day = match case {
                        TradeCase::AnimalOverflow => 1,
                        _ => 0,
                    };
                    let cell =
                        initial.shipment_coordinate(Good::Bread, day, scenario.ring(), buyer);
                    initial.values[cell.0 as usize] = u64::MAX;
                }
                TradeCase::RetailCapacity | TradeCase::ExhaustedRetailer => {
                    initial.set(Field::EnterpriseModel, seller, 1);
                    initial.set(Field::RetailOpen, seller, 5);
                    initial.set(Field::Sales, seller, 4);
                }
                _ => (),
            }
            if let TradeCase::ExhaustedRetailer = case {
                initial.set(Field::Bread, seller, 200);
                initial.set(Field::Sales, seller, 5);
            }
            match case {
                TradeCase::UnstockedSellerOverflow
                | TradeCase::UnstockedSalesOverflow
                | TradeCase::UnstockedTransitOverflow
                | TradeCase::UnstockedPurchaseOverflow => initial.set(Field::Bread, seller, 0),
                _ => (),
            }
            let count = initial.actors.len();
            let mut proposals = vec![NO_ACTOR; count * ProposalLayout::WORDS];
            proposals[buyer * ProposalLayout::WORDS + ProposalLayout::ACTOR] = seller as u32;
            proposals[buyer * ProposalLayout::WORDS + ProposalLayout::QUANTITY_LOW] =
                quantity as u32;
            proposals[buyer * ProposalLayout::WORDS + ProposalLayout::QUANTITY_HIGH] =
                (quantity >> 32) as u32;
            let proposals = device.tensor(&proposals);
            let parameters = simulation.parameters(Phase::ProvisionPurchase, market, 0, count)?;
            let mut results = Vec::new();
            let executions: &[TradeExecution] = match (initial.get(Field::Bread, seller), case) {
                (0, _) | (_, TradeCase::ExhaustedRetailer) => &[
                    TradeExecution::Ledger,
                    TradeExecution::Direct,
                    TradeExecution::DeferredRejection,
                ],
                _ => &[TradeExecution::Ledger, TradeExecution::Direct],
            };
            for &execution in executions {
                let state = device.tensor(bytemuck::cast_slice(&encode(&initial.values)));
                let control = device.tensor(&[0; CONTROL_WORDS]);
                let status = device.tensor(&vec![0; count * 5 + 1]);
                TradeCheck::launch(
                    &device,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new_1d(1),
                    TradeCheckInputLaunch {
                        state: state.clone().into_buffer_arg(),
                        actors: simulation.actors.clone().into_buffer_arg(),
                        config: simulation.config.launch(),
                        proposals: proposals.clone().into_buffer_arg(),
                        control: control.clone().into_buffer_arg(),
                        status: status.clone().into_buffer_arg(),
                        parameters: parameters.launch(),
                        buyer: buyer as u32,
                        seller: seller as u32,
                        execution,
                    },
                );
                let mut result = device.read(&state)?;
                result.extend(device.read(&control)?);
                result.extend(device.read_range(&status, 0, 1)?);
                results.push(result);
            }
            for result in results.iter().skip(1) {
                assert_eq!(results[0], *result, "{market:?} {case:?}");
            }
        }
    }
    Ok(())
}

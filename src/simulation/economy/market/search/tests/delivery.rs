use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum DeliveryScoreCase {
    Tied,
    QuoteWrap,
    PenaltyWrap,
    AccessWrap,
    ScaledSpeedWrap,
}

#[derive(CubeType, CubeLaunch)]
struct DeliveryScoreFixtureInput {
    config: EconomyConfig,
    buyer: u32,
    seller: u32,
    #[cube(comptime)]
    case: DeliveryScoreCase,
}

struct DeliveryScoreFixture;
struct ConfigureDeliveryScore;

impl System for DeliveryScoreFixture {
    type Input = DeliveryScoreFixtureInput;
    type Passes = gpu_passes![ConfigureDeliveryScore];
}

#[cube]
impl GpuPass<DeliveryScoreFixtureInput> for ConfigureDeliveryScore {
    type Output = ();

    fn execute(mut input: DeliveryScoreFixtureInput) {
        if ABSOLUTE_POS == 0 {
            match comptime!(input.case) {
                DeliveryScoreCase::QuoteWrap | DeliveryScoreCase::AccessWrap => {
                    for index in 0..input.config.railway.quotes.len() {
                        input.config.railway.quotes[index] = u32::MAX;
                    }
                }
                _ => (),
            }
            if comptime!(input.case == DeliveryScoreCase::AccessWrap) {
                for actor in 0..input.config.actors {
                    input.config.railway.access[actor as usize * 2] = 2;
                    input.config.railway.access[actor as usize * 2 + 1] = 0;
                }
                input.config.railway.access[input.buyer as usize * 2] = 0;
                input.config.railway.access[input.seller as usize * 2] = 1;
                input.config.railway.access[input.seller as usize * 2 + 1] = u32::MAX - 3;
                let nodes = input.config.railway.nodes;
                let base = nodes as usize * crate::domain::railway::RAIL_ROUTE_WORDS;
                input.config.railway.routes[base] = 50;
                input.config.railway.routes[base + 2] = 0;
                input.config.railway.routes[base + 3] = 0;
                input.config.railway.quotes[(nodes * input.config.ring) as usize] = 0;
            }
        }
    }
}

fn delivery_score_probe(good: Good, rotation: u32, case: DeliveryScoreCase) -> Result<SearchProbe> {
    let mut config: crate::domain::ScenarioConfig =
        serde_json::from_str(include_str!("../../../../../../scenarios/railway.json"))?;
    let population = config.population.as_mut().expect("railway population");
    let first = population.households.len() + population.firms.len();
    let template = population.firms[3].clone();
    for index in 0..9 {
        let mut inventories = [0; Good::COUNT];
        inventories[good as usize - 1] = u64::MAX / 1000;
        population.firms.push(FirmSeed {
            identity: 100 + index,
            position: match index % 4 {
                0 => Position { x: 900, y: 200 },
                1 => Position { x: 0, y: 0 },
                2 => Position { x: 400, y: 50 },
                _ => Position { x: 900, y: 0 },
            },
            good,
            inventories,
            price: 10,
            ..template.clone()
        });
    }
    config.firms = population.firms.len() as u32;
    let scenario = Scenario::new(config)?;
    let mut state = scenario.initial()?;
    state.set(Field::Cash, 0, u64::MAX);
    state.set(Field::HeatingNeed, 0, u64::from(u32::MAX));
    for actor in [
        scenario.config().households as usize + 1,
        scenario.config().households as usize + 2,
    ] {
        state.set(Field::RailBuilt, actor, 1);
        state.set(Field::RailOperating, actor, 1);
    }
    match case {
        DeliveryScoreCase::QuoteWrap | DeliveryScoreCase::AccessWrap => {
            for actor in first..first + 9 {
                state.set(Field::Price, actor, 1000);
            }
            state.set(Field::Price, first, 1);
            state.set(
                Field::Price,
                first + 1,
                match case {
                    DeliveryScoreCase::QuoteWrap => u64::from(u32::MAX - 67),
                    _ => 5,
                },
            );
        }
        DeliveryScoreCase::PenaltyWrap => state.set(Field::Price, first, 1),
        _ => (),
    }
    let mut simulation = Simulation::new(scenario, device(64)?)?;
    simulation.device.write(
        &simulation.state,
        bytemuck::cast_slice(&encode(&state.values)),
    );
    for stage in [
        railway::RailStage::Access,
        railway::RailStage::Routes,
        railway::RailStage::Quotes,
    ] {
        simulation.railways(stage);
    }
    match case {
        DeliveryScoreCase::PenaltyWrap => simulation.config.delivery_penalty = u32::MAX,
        DeliveryScoreCase::AccessWrap => simulation.config.shipment_speed = 1,
        DeliveryScoreCase::ScaledSpeedWrap => simulation.config.shipment_speed = u32::MAX,
        _ => (),
    }
    DeliveryScoreFixture::launch_single(
        &simulation.device,
        DeliveryScoreFixtureInputLaunch {
            config: simulation.config.launch(),
            buyer: 0,
            seller: first as u32 + 1,
            case,
        },
    );
    let count = state.actors.len();
    let cells = (simulation.config.spatial_side * simulation.config.spatial_side) as usize;
    let segment = cells + 1 + count;
    let mut spatial = simulation.device.read(&simulation.spatial)?;
    spatial.fill(0);
    spatial[1] = 9;
    spatial[cells + 1..cells + 10]
        .copy_from_slice(&(first as u32..first as u32 + 9).collect::<Vec<_>>());
    let query = 3 * segment + count.div_ceil(DISCOVERY_BLOCK as usize) + 1;
    spatial[query..query + count].copy_from_slice(&(0..count as u32).collect::<Vec<_>>());
    simulation.device.write(&simulation.spatial, &spatial);
    let mut parameters =
        simulation.parameters(Phase::ProvisionPurchase, Market::Consumer(good), 0, count)?;
    parameters.settlement_rotation = rotation;
    Ok(SearchProbe {
        simulation,
        state,
        parameters,
    })
}

#[test]
fn consumer_score_bounds_preserve_eight_offer_quota_ties_wrapping_and_rail_access_on_gpu()
-> Result<()> {
    for good in [Good::Coal, Good::Wood] {
        let mut quota =
            SearchProbe::with_suppliers(SupplyCase::Capped, 0, Market::Consumer(good), 9)?;
        quota.state.set(Field::HeatingNeed, 0, u64::from(u32::MAX));
        quota.state.set(Field::Cash, 0, u64::MAX);
        for actor in 1..quota.state.actors.len() {
            quota
                .state
                .set(Field::inventory(good), actor, u64::MAX / 1000);
        }
        let expected = quota.outcome(SearchExecution::Plane)?;
        assert_eq!(expected.proposal[ProposalLayout::ACTOR], 1);
        assert!(expected.proposal[ProposalLayout::QUANTITY_HIGH] > 0);
        for workgroup in [32, 256] {
            assert_eq!(
                quota.outcome_with_workgroup(SearchExecution::SpatialFuel, workgroup)?,
                expected
            );
        }
        for case in [
            DeliveryScoreCase::Tied,
            DeliveryScoreCase::QuoteWrap,
            DeliveryScoreCase::PenaltyWrap,
            DeliveryScoreCase::AccessWrap,
            DeliveryScoreCase::ScaledSpeedWrap,
        ] {
            for rotation in [0, 5] {
                let probe = delivery_score_probe(good, rotation, case)?;
                let expected = probe.outcome(SearchExecution::Plane)?;
                if rotation == 0
                    && matches!(
                        case,
                        DeliveryScoreCase::QuoteWrap | DeliveryScoreCase::AccessWrap
                    )
                {
                    assert_eq!(
                        expected.proposal[ProposalLayout::ACTOR],
                        probe.state.actors.len() as u32 - 8
                    );
                }
                for workgroup in [32, 256] {
                    assert_eq!(
                        probe.outcome_with_workgroup(SearchExecution::SpatialFuel, workgroup)?,
                        expected,
                        "{good:?}, {case:?}, rotation {rotation}, workgroup {workgroup}",
                    );
                }
            }
        }
    }
    Ok(())
}

#[derive(CubeType, CubeLaunch)]
struct DeliveryFloorInput {
    actors: Box<[u32]>,
    config: EconomyConfig,
    findings: Box<[Atomic<u32>]>,
}

struct DeliveryFloorCheck;
struct CompareDeliveryFloor;

impl System for DeliveryFloorCheck {
    type Input = DeliveryFloorInput;
    type Passes = gpu_passes![CompareDeliveryFloor];
}

#[cube]
impl GpuPass<DeliveryFloorInput> for CompareDeliveryFloor {
    type Output = ();

    fn execute(input: DeliveryFloorInput) {
        let actors = ActorTable::new(&input.actors);
        let index = ABSOLUTE_POS as u32;
        let count = input.config.actors;
        if index < count * count {
            let buyer = index / count;
            let seller = index % count;
            let trip = actors.distance(buyer, seller);
            let mut access = 0u32;
            if input.config.railway.has_routes() {
                access = input.config.railway.access[buyer as usize * 2 + 1];
            }
            let floor = minimum_delivery_days(&input.config, access, seller, trip);
            let route = railway::FreightRequest::new(&actors, &input.config, buyer, seller)
                .preferred(&input.config);
            let days = railway::DeliveryRequest {
                buyer,
                seller,
                trip,
                buyer_access: access,
            }
            .days(&input.config);
            if route.days < max(ceildiv(trip, input.config.shipment_speed), 1) {
                input.findings[3usize].fetch_add(1);
            }
            if days != route.days {
                input.findings[2usize].fetch_add(1);
            }
            if floor > route.days {
                input.findings[0usize].fetch_add(1);
            }
            let previous = max(
                ceildiv(
                    trip,
                    input.config.shipment_speed * input.config.railway.speed_multiplier,
                ),
                1,
            );
            if floor > previous {
                input.findings[1usize].fetch_add(1);
            }
        }
    }
}

#[test]
fn station_access_floor_never_exceeds_selected_route_days_on_gpu() -> Result<()> {
    let mut rail: crate::domain::ScenarioConfig =
        serde_json::from_str(include_str!("../../../../../../scenarios/railway.json"))?;
    let build = rail.railway.construction.quantities();
    let population = rail
        .population
        .as_mut()
        .expect("explicit railway population");
    for firm in population
        .firms
        .iter_mut()
        .filter(|firm| firm.good == Good::Railway)
    {
        for (good, amount) in [Good::Coal, Good::Steel, Good::Locomotives]
            .into_iter()
            .zip(build)
        {
            firm.inventories[good as usize - 1] = u64::from(amount);
        }
    }
    let device = device(64)?;
    let mut improved = 0;
    let mut faster_routes = 0;
    for scenario in [
        Scenario::industrial(193, 42, Experiment::Baseline)?,
        Scenario::industrial(193, 42, Experiment::Transport)?,
        Scenario::new(rail)?,
    ] {
        let mut simulation = Simulation::new(scenario, device.clone())?;
        simulation.advance()?;
        for stage in [
            railway::RailStage::Access,
            railway::RailStage::Routes,
            railway::RailStage::Quotes,
        ] {
            simulation.railways(stage);
        }
        let findings = device.tensor(&[0; 4]);
        let count = simulation.metadata.actors.len();
        DeliveryFloorCheck::launch_elements(
            &device,
            count * count,
            DeliveryFloorInputLaunch {
                actors: simulation.actors.buffer_arg(),
                config: simulation.config.launch(),
                findings: findings.buffer_arg(),
            },
        );
        let findings = device.read(&findings)?;
        assert_eq!(findings[0], 0);
        assert_eq!(findings[2], 0);
        improved += findings[1];
        faster_routes += findings[3];
    }
    assert!(improved > 0);
    assert!(faster_routes > 0);
    Ok(())
}

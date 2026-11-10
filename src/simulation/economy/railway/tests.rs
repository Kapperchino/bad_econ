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

#[test]
fn shared_freight_settles_multiple_workgroups_in_one_gpu_wave() -> Result<()> {
    use crate::domain::{HouseholdSeed, Position, ScenarioConfig};
    use crate::simulation::{Market, decode, encode};
    let mut config: ScenarioConfig =
        serde_json::from_str(include_str!("../../../../scenarios/railway.json"))?;
    config.railway.units_per_level = 1000;
    let population = config.population.as_mut().unwrap();
    let template = population.households[0].clone();
    population
        .households
        .extend((0..121).map(|index| HouseholdSeed {
            identity: 100 + index,
            ..template.clone()
        }));
    for household in &mut population.households {
        household.position = Position { x: 900, y: 0 };
    }
    config.households = population.households.len() as u32;
    let buyers = config.households as usize;
    let scenario = Scenario::new(config)?;
    for workgroup in [32, 64, 128, 256] {
        let device = crate::simulation::tests::device(workgroup)?;
        let simulation = Simulation::new(scenario.clone(), device.clone())?;
        let mut state = scenario.initial()?;
        let count = state.actors.len();
        let seller = buyers;
        for firm in [seller + 1, seller + 2] {
            state.set(Field::RailBuilt, firm, 1);
            state.set(Field::RailOperating, firm, 1);
            state.set(Field::Railway, firm, 1000);
        }
        for region in 0..scenario.rail_geography().regions.len() {
            for (field, amount) in [
                (RailCapacityField::Level, 1),
                (RailCapacityField::Available, 1000),
            ] {
                let coordinate = scenario
                    .rail_geography()
                    .coordinate(&scenario, &state, field, region);
                state.values[coordinate] = amount;
            }
        }
        let mut parameters =
            simulation.parameters(Phase::ProvisionPurchase, Market::Household, 0, count)?;
        parameters.discovery = Discovery::Indexed;
        parameters.sort_count = buyers as u32;
        let mut proposals = vec![NO_ACTOR; count * ProposalLayout::WORDS];
        let mut scratch = vec![0; markets::scratch_words(count)];
        let mut control = vec![0; CONTROL_WORDS + 4];
        control[CONSTRUCTED_EVENTS_WORD] = buyers as u32;
        for buyer in 0..buyers {
            let delivery = state.shipment_coordinate(
                Good::Bread,
                (parameters.day as usize + 9) % scenario.ring(),
                scenario.ring(),
                buyer,
            );
            state.values[delivery.0 as usize] = 1;
            proposals[buyer * ProposalLayout::WORDS + ProposalLayout::ACTOR] = seller as u32;
            proposals[buyer * ProposalLayout::WORDS + ProposalLayout::QUANTITY_LOW] = 1;
            proposals[buyer * ProposalLayout::WORDS + ProposalLayout::QUANTITY_HIGH] = 0;
            scratch[buyer] = seller as u32;
            scratch[count + buyer] = buyer as u32;
            scratch[4 * count + buyer] = EventStatus::Accepted as u32;
        }
        device.write(
            &simulation.state,
            bytemuck::cast_slice(&encode(&state.values)),
        );
        device.write(&simulation.proposals, &proposals);
        device.write(&simulation.scratch, &scratch);
        device.write(&simulation.control, &control);
        for stage in [RailStage::Access, RailStage::Routes, RailStage::Quotes] {
            simulation.railways(stage);
        }
        for stage in [
            settlement::Stage::Snapshot,
            settlement::Stage::Initialize,
            settlement::Stage::Reset,
            settlement::Stage::Propose,
            settlement::Stage::Commit,
            settlement::Stage::Release,
            settlement::Stage::Accounts,
        ] {
            settlement::launch(
                &device,
                &simulation.state,
                &simulation.actors,
                &simulation.config,
                &simulation.proposals,
                &simulation.scratch,
                &simulation.control,
                &parameters,
                stage,
            );
        }
        assert_eq!(settlement::pending(&device, &simulation.config)?, 0);
        assert_eq!(
            device.read(&simulation.control)?[crate::simulation::flags::FAULT_WORD],
            0
        );
        let mut actual = state.clone();
        actual.values = decode(&device.read(&simulation.state)?);
        assert_eq!(
            actual.get(Field::Cash, seller),
            state.get(Field::Cash, seller) - buyers as u64 * 7
        );
        assert_eq!(
            actual.get(Field::RailRevenue, seller + 1),
            buyers as u64 * 2
        );
        assert_eq!(
            actual.get(Field::RailRevenue, seller + 2),
            buyers as u64 * 5
        );
        assert_eq!(actual.get(Field::Railway, seller + 2), 1000 - buyers as u64);
        for buyer in 0..buyers {
            let delivery = actual.shipment_coordinate(
                Good::Bread,
                (parameters.day as usize + 1) % scenario.ring(),
                scenario.ring(),
                buyer,
            );
            assert_eq!(actual.values[delivery.0 as usize], 1);
        }
    }
    Ok(())
}

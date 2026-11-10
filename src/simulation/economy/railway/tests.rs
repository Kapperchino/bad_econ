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
        Scenario::industrial(17, 42, Experiment::Baseline)?,
        Scenario::industrial(17, 42, Experiment::Transport)?,
    ];
    for scenario in scenarios {
        let device = crate::simulation::tests::device(64)?;
        let mut simulation = Simulation::new(scenario, device.clone())?;
        for _ in 0..2 {
            for stage in [
                RailStage::Infrastructure,
                RailStage::Access,
                RailStage::Routes,
                RailStage::Quotes,
            ] {
                simulation.railways(stage);
            }
            simulation.fault_check()?;
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
fn repeated_input_initialization_resets_prior_suppliers_and_preserves_faults_on_gpu() -> Result<()>
{
    use crate::simulation::{Market, encode, flags::FAULT_WORD};

    #[derive(Clone, Copy, Debug)]
    enum ResetCase {
        Suppliers,
        Empty,
        Fault,
    }

    let workgroup = 64;
    let device = crate::simulation::tests::device(workgroup)?;
    for case in [ResetCase::Suppliers, ResetCase::Empty, ResetCase::Fault] {
        let scenario = Scenario::industrial(67, 42, Experiment::Transport)?;
        let mut initial = scenario.initial()?;
        initial.set(Field::Bread, 0, 0);
        let simulation = Simulation::new(scenario, device.clone())?;
        let count = initial.actors.len();
        let parameters = simulation.parameters(
            Phase::ProvisionPurchase,
            Market::Input(Good::Bread),
            0,
            count,
        )?;
        let initialize = |stage| {
            EconomySystem::launch(
                &device,
                device.count(count),
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
        };
        device.write(
            &simulation.state,
            bytemuck::cast_slice(&encode(&initial.values)),
        );
        let mut control = vec![0; simulation.control.meta.num_elements()];
        device.write(&simulation.control, &control);
        initialize(EconomyStage::InputMarketStart);
        let clean_statuses = device.read_range(&simulation.scratch, 4 * count, count)?;
        let cells = (simulation.config.spatial_side * simulation.config.spatial_side) as usize;
        let mut spatial = vec![0; simulation.spatial.meta.num_elements()];
        match case {
            ResetCase::Suppliers | ResetCase::Fault => {
                let suppliers = [0, 67, 68];
                spatial[cells] = suppliers.len() as u32;
                for (index, seller) in suppliers.into_iter().enumerate() {
                    spatial[cells + 1 + index] = seller as u32;
                }
            }
            ResetCase::Empty => (),
        }
        control[FAULT_WORD] = match case {
            ResetCase::Fault => FaultFlags::ARITHMETIC_OVERFLOW.bits(),
            _ => 0,
        };
        let sentinel = 0x1234_5678;
        device.write(&simulation.control, &control);
        device.write(&simulation.spatial, &spatial);
        device.write(
            &simulation.scratch,
            &vec![sentinel; simulation.scratch.meta.num_elements()],
        );
        initialize(EconomyStage::InputMarketRepeat);
        let expected_statuses = match case {
            ResetCase::Fault => vec![sentinel; count],
            _ => clean_statuses,
        };
        assert_eq!(
            device.read_range(&simulation.scratch, 4 * count, count)?,
            expected_statuses,
            "{case:?}, workgroup {workgroup}, statuses"
        );
        let actual_control = device.read(&simulation.control)?;
        assert_eq!(actual_control[FAULT_WORD], control[FAULT_WORD]);
        assert_eq!(actual_control[LATCH_WORD], control[FAULT_WORD]);
    }
    Ok(())
}

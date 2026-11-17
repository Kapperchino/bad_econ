#![cfg(feature = "compute")]
use super::support;

use anyhow::Result;
use bad_econ::{
    domain::{research::*, *},
    runtime::SimulationCommand,
    simulation::Simulation,
};

fn steelworks(coal: u64) -> Result<ScenarioConfig> {
    let mut config = Scenario::baseline(6, 42, Experiment::Concentrated)?
        .config()
        .clone();
    config.coverage = 30;
    config.resources = ResourceMap::Explicit { regions: vec![] };
    config.forecast_weight = 0;
    config.distribution_rate = 0;
    config.behavior = BehaviorPolicy {
        work_fatigue: 0,
        rest_recovery: 0,
        wage_increment: 0,
        price_response: 0,
    };
    let position = Position { x: 0, y: 0 };
    let mut inventory = [0; Good::COUNT];
    inventory[Good::Iron as usize - 1] = 20;
    inventory[Good::Coal as usize - 1] = coal;
    config.population = Some(Population {
        households: (1..=6)
            .map(|identity| HouseholdSeed {
                identity,
                position,
                cash: 0,
                inventories: Good::food([0, 0, 30]),
                fatigue: 0,
            })
            .collect(),
        firms: [Good::Steel, Good::Iron, Good::Coal]
            .into_iter()
            .enumerate()
            .map(|(index, good)| FirmSeed {
                identity: 100 + index as u64,
                position,
                owner: index as u64 + 1,
                good,
                cash: match index {
                    0 => 100_000,
                    _ => 0,
                },
                inventories: match index {
                    0 => inventory,
                    _ => [0; Good::COUNT],
                },
                capacity: 2,
                technique: Technique::new(60, u32::from(index == 0) * 2, 3).unwrap(),
                wage: 1,
                price: 350,
            })
            .collect(),
    });
    Ok(config)
}

fn research_lab() -> ResearchLab {
    ResearchLab {
        owner: 100,
        researcher: 1,
        discovered: 0,
        mastered: 0,
        permitted: Technology::Steelmaking as u32,
        adopted: 0,
        consent: ResearchConsent::Granted,
        production_requirements: 0,
        adoption: AdoptionPolicy::MasterAndAdoptPermitted,
        methods: vec![ProductionMethod::BessemerProcess],
        projects: vec![ResearchProject {
            identity: 900,
            prerequisites: 0,
            outcomes: Technology::Steelmaking as u32,
            inputs: [0; Good::COUNT],
            labor_minutes: 30,
            minimum_effort: 1,
            pioneer_effort: 1,
            capacity: 1000,
            relatedness: 1000,
            shape: 1,
            exposure: 0,
            exposure_signal: 1000,
        }],
    }
}

#[test]
fn steel_requires_both_inputs_and_conserves_each_good_on_gpu() -> Result<()> {
    for coal in [0, 2, 20] {
        let scenario = Scenario::new(steelworks(coal)?)?;
        let mut simulation = Simulation::new(scenario.clone(), support::device(64)?)?;
        simulation.advance()?;
        let packets = (coal / 2).min(2);
        assert_eq!(
            simulation.snapshot.produced[Good::Steel as usize - 1],
            packets * 3
        );
        assert_eq!(
            simulation.snapshot.used[Good::Iron as usize - 1],
            packets * 2
        );
        assert_eq!(
            simulation.snapshot.used[Good::Coal as usize - 1],
            packets * 2
        );
        let state = simulation.download()?;
        assert_eq!(
            simulation.snapshot,
            ObservationSnapshot::from_state(&scenario, &state)?
        );
    }
    Ok(())
}

#[test]
fn method_changes_preserve_old_shipments_and_count_equipment_packets_on_gpu() -> Result<()> {
    let mut config = steelworks(0)?;
    let firms = &mut config.population.as_mut().unwrap().firms;
    firms[0].good = Good::Tools;
    firms[0].inventories = [0; Good::COUNT];
    firms[0].inventories[Good::Wood as usize - 1] = 20;
    firms[1].inventories[Good::Iron as usize - 1] = 20;
    firms[2].good = Good::Steel;
    firms[2].technique = Technique::new(60, 2, 3)?;
    firms[2].inventories[Good::Steel as usize - 1] = 20;
    let mut lab = research_lab();
    lab.methods = vec![ProductionMethod::MachineTools];
    config.research = Some(ResearchConfig {
        labs: vec![lab],
        equipment: vec![Equipment {
            firm: 100,
            remaining_service: 10,
            condition: 1000,
            nameplate: 2,
            downtime: 0,
            wear_per_packet: 0,
            maintenance_good: Good::Tools,
            maintenance_ticks: 0,
            restoration_per_tick: 0,
        }],
        ..Default::default()
    });
    let scenario = Scenario::new(config)?;
    let mut simulation = Simulation::new(scenario.clone(), support::device(64)?)?;
    simulation.advance()?;
    assert_eq!(simulation.snapshot.transit[Good::Iron as usize - 1], 20);
    let checkpoint = simulation.checkpoint()?;
    simulation.advance()?;
    let state = simulation.download()?;
    assert_eq!(state.get(Field::Iron, 6), 20);
    assert_eq!(state.get(Field::Steel, 6), 0);
    assert_eq!(simulation.snapshot.transit[Good::Steel as usize - 1], 20);
    simulation.advance()?;
    let expected = simulation.download()?;
    assert_eq!(expected.get(Field::Produced, 6), 12);
    assert_eq!(expected.get(Field::Iron, 6), 20);
    assert_eq!(simulation.snapshot.used[Good::Iron as usize - 1], 0);
    assert_eq!(simulation.snapshot.used[Good::Steel as usize - 1], 4);
    assert_eq!(simulation.snapshot.research.remaining_service, 8);
    let mut restored = Simulation::new(scenario, support::device(64)?)?;
    restored.command(SimulationCommand::Restore(Box::new(checkpoint)))?;
    restored.advance()?;
    restored.advance()?;
    assert_eq!(restored.download()?, expected);
    Ok(())
}

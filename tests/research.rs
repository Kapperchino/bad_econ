#![cfg(feature = "compute")]
use super::support;

use anyhow::Result;
use bad_econ::{
    domain::{research::*, *},
    runtime::SimulationCommand,
    simulation::{Simulation, Trace, research::capabilities::*},
};

fn base_config() -> Result<ScenarioConfig> {
    let mut config = Scenario::baseline(6, 42, Experiment::Concentrated)?
        .config()
        .clone();
    config.firms = 3;
    config.coverage = 30;
    config.forecast_weight = 0;
    config.distribution_rate = 0;
    config.behavior = BehaviorPolicy {
        work_fatigue: 0,
        rest_recovery: 0,
        wage_increment: 0,
        price_response: 0,
    };
    let position = Position { x: 0, y: 0 };
    config.resources = ResourceMap::Explicit {
        regions: vec![ResourceRegion::new(
            Good::Wood,
            Position { x: 0, y: 0 },
            Position {
                x: config.map_size,
                y: config.map_size,
            },
            config.map_size,
        )?],
    };
    config.population = Some(Population {
        households: (1..=6)
            .map(|identity| HouseholdSeed {
                identity,
                position,
                cash: 0,
                inventories: Good::food([0, 0, 10]),
                fatigue: 0,
            })
            .collect(),
        firms: (0..3)
            .map(|index| FirmSeed {
                identity: 100 + index,
                position,
                owner: index + 1,
                good: Good::Wood,
                cash: match index {
                    0 => 100_000,
                    _ => 0,
                },
                inventories: Good::food([5, 0, 5]),
                capacity: 5,
                wage: 1,
                price: 100,
                technique: Technique::new(60, 0, 1).unwrap(),
            })
            .collect(),
    });
    Ok(config)
}

fn lab() -> ResearchLab {
    ResearchLab {
        methods: Vec::new(),
        owner: 100,
        researcher: 1,
        discovered: 0,
        mastered: 0,
        permitted: 1,
        adopted: 0,
        consent: ResearchConsent::Granted,
        production_requirements: 1,
        adoption: AdoptionPolicy::MasterAndAdoptPermitted,
        projects: vec![ResearchProject {
            identity: 900,
            prerequisites: 0,
            outcomes: 1,
            inputs: Good::food([1, 0, 0]),
            labor_minutes: 100,
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
fn research_checkpoint_keeps_midday_boundary_and_restores_keyed_outcomes() -> Result<()> {
    let mut config = base_config()?;
    config.research = Some(ResearchConfig {
        labs: vec![lab()],
        strikes: vec![StrikePolicy {
            household: 6,
            grievance: 1000,
            solidarity: 1000,
            trust: 1000,
            efficacy: 1000,
            peers: 1000,
            cost: 0,
            risk: 0,
        }],
        ..ResearchConfig::default()
    });
    let mut simulation = Simulation::new(Scenario::new(config)?, support::device(64)?)?;
    simulation.advance()?;
    let audited = simulation.download()?;
    for phase in [
        Phase::Opening,
        Phase::Observation,
        Phase::Proposal,
        Phase::Commitment,
        Phase::WorkPayment,
        Phase::ProvisionPurchase,
        Phase::PhysicalClose,
    ] {
        simulation.phase(phase, Trace::Record)?;
    }
    let checkpoint = simulation.checkpoint()?;
    assert_eq!(checkpoint.restore()?.state, audited);
    for phase in [Phase::AccountsDecisions, Phase::Audit] {
        simulation.phase(phase, Trace::Record)?;
    }
    let expected = simulation.download()?;
    simulation.command(SimulationCommand::Restore(checkpoint.into()))?;
    simulation.advance()?;
    assert_eq!(simulation.download()?, expected);
    assert!(simulation.snapshot.research.striking <= 1);
    Ok(())
}

#[test]
fn boundary_elimination_preserves_reachable_labels_across_cycles_and_empty_blocks() -> Result<()> {
    let device = support::device(64)?;
    let mut edges = vec![CapabilityEdge {
        prerequisite: 0,
        dependent: 2,
    }];
    edges.extend((2..39).map(|node| CapabilityEdge {
        prerequisite: node,
        dependent: node + 1,
    }));
    edges.extend([
        CapabilityEdge {
            prerequisite: 39,
            dependent: 1,
        },
        CapabilityEdge {
            prerequisite: 20,
            dependent: 10,
        },
    ]);
    let graph = CapabilityGraph::new(40, edges)?;
    let mut labels = vec![0; 40];
    labels[0] = 1;
    labels[1] = 4;
    labels[5] = 2;
    let packed = graph.upload(&device, &labels)?;
    let reduced = packed.eliminate(&device, 2)?;
    let observation = reduced.download(&device)?;
    assert_eq!(observation.labels, [1, 6]);
    assert_eq!(observation.adjacency, [2, 0]);
    let expected = graph.closure(&device, &labels)?.download(&device)?;
    assert_eq!(reduced.closure(&device)?.download(&device)?, expected[..2]);
    assert_eq!(
        packed
            .eliminate(&device, 40)?
            .closure(&device)?
            .download(&device)?,
        expected
    );
    assert!(
        packed
            .eliminate(&device, 0)?
            .closure(&device)?
            .download(&device)?
            .is_empty()
    );
    assert!(
        CapabilityGraph::new(0, vec![])?
            .upload(&device, &[])?
            .eliminate(&device, 0)?
            .download(&device)?
            .labels
            .is_empty()
    );
    assert!(packed.eliminate(&device, 41).is_err());
    Ok(())
}

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
fn capability_closure_and_masked_prerequisites_run_on_device() -> Result<()> {
    let device = support::device(64)?;
    let graph = CapabilityGraph::new(
        4,
        vec![
            CapabilityEdge {
                prerequisite: 0,
                dependent: 1,
            },
            CapabilityEdge {
                prerequisite: 1,
                dependent: 2,
            },
            CapabilityEdge {
                prerequisite: 2,
                dependent: 3,
            },
            CapabilityEdge {
                prerequisite: 1,
                dependent: 3,
            },
            CapabilityEdge {
                prerequisite: 3,
                dependent: 2,
            },
        ],
    )?;
    assert_eq!(
        graph.closure(&device, &[1, 2, 0, 0])?.download(&device)?,
        [1, 3, 3, 3]
    );
    assert_eq!(
        graph.closure(&device, &[0, 0, 0, 0])?.download(&device)?,
        [0, 0, 0, 0]
    );
    assert_eq!(
        graph
            .missing(&device, &[0, 0, 0, 1], &[0, 1, 0, 0])?
            .download(&device)?,
        [0, 0, 1, 1]
    );
    assert!(
        CapabilityGraph::new(0, vec![])?
            .closure(&device, &[])?
            .download(&device)?
            .is_empty()
    );
    assert!(
        CapabilityGraph::new(
            1,
            vec![CapabilityEdge {
                prerequisite: 0,
                dependent: 1
            }]
        )
        .is_err()
    );
    Ok(())
}

#[test]
fn research_consumes_inputs_and_time_before_granting_production_readiness() -> Result<()> {
    let mut config = base_config()?;
    config.research = Some(ResearchConfig {
        labs: vec![lab()],
        ..ResearchConfig::default()
    });
    let mut simulation = Simulation::new(Scenario::new(config)?, support::device(64)?)?;
    for phase in [Phase::Opening, Phase::Observation, Phase::Proposal] {
        simulation.phase(phase, Trace::Record)?;
    }
    assert_eq!(simulation.download()?.get(Field::Plan, 6), 0);
    for phase in [
        Phase::Commitment,
        Phase::WorkPayment,
        Phase::ProvisionPurchase,
        Phase::PhysicalClose,
        Phase::AccountsDecisions,
        Phase::Audit,
    ] {
        simulation.phase(phase, Trace::Record)?;
    }
    let state = simulation.download()?;
    assert_eq!(simulation.snapshot.research.attempts, 1);
    assert_eq!(simulation.snapshot.research.discoveries, 1);
    assert_eq!(simulation.snapshot.research.consumed, Good::food([1, 0, 0]));
    assert_eq!(state.get(Field::ResearchMinutes, 0), 100);
    assert_eq!(state.get(Field::Time, 0), 380);
    assert_eq!(state.get(Field::Grain, 6), 4);
    assert_eq!(state.get(Field::Produced, 6), 0);
    for phase in [Phase::Opening, Phase::Observation, Phase::Proposal] {
        simulation.phase(phase, Trace::Record)?;
    }
    assert_eq!(simulation.download()?.get(Field::Plan, 6), 5);
    Ok(())
}

#[test]
#[ignore = "extended scenario coverage; run cargo gpu-extended"]
fn failed_research_consumes_inputs_but_missing_consent_and_time_do_not() -> Result<()> {
    for mode in 0..5 {
        let mut config = base_config()?;
        let mut lab = lab();
        match mode {
            0 => lab.projects[0].labor_minutes = 0,
            1 => lab.consent = ResearchConsent::Withheld,
            2 => lab.projects[0].labor_minutes = 1440,
            3 => lab.projects[0].prerequisites = 2,
            _ => lab.projects[0].inputs[0] = 6,
        }
        config.research = Some(ResearchConfig {
            labs: vec![lab],
            ..ResearchConfig::default()
        });
        let mut simulation = Simulation::new(Scenario::new(config)?, support::device(64)?)?;
        simulation.advance()?;
        let expected_attempts = u64::from(mode == 0);
        assert_eq!(simulation.snapshot.research.attempts, expected_attempts);
        assert_eq!(simulation.snapshot.research.discoveries, 0);
        assert_eq!(
            simulation.snapshot.research.consumed,
            Good::food([expected_attempts, 0, 0])
        );
        assert_eq!(simulation.download()?.get(Field::Produced, 6), 0);
    }
    Ok(())
}

#[test]
fn equipment_lifetime_and_renewable_stock_constrain_executed_production() -> Result<()> {
    let mut config = base_config()?;
    config.research = Some(ResearchConfig {
        equipment: vec![Equipment {
            firm: 100,
            remaining_service: 2,
            condition: 500,
            nameplate: 10,
            downtime: 0,
            wear_per_packet: 100,
            maintenance_good: Good::Bread,
            maintenance_ticks: 1,
            restoration_per_tick: 100,
        }],
        natural_stocks: vec![NaturalStock {
            firm: 100,
            stock: 8,
            capacity: 10,
            renewal_rate: 1000,
            extraction_per_packet: 2,
        }],
        ..ResearchConfig::default()
    });
    let mut simulation = Simulation::new(Scenario::new(config)?, support::device(64)?)?;
    simulation.advance()?;
    let state = simulation.download()?;
    assert_eq!(state.get(Field::Produced, 6), 2);
    assert_eq!(simulation.snapshot.research.remaining_service, 0);
    assert_eq!(simulation.snapshot.research.natural_stock, 6);
    assert_eq!(simulation.snapshot.research.consumed, Good::food([0, 0, 1]));
    simulation.advance()?;
    assert_eq!(simulation.download()?.get(Field::Produced, 6), 2);
    assert_eq!(simulation.snapshot.research.remaining_service, 0);
    assert_eq!(simulation.snapshot.research.natural_stock, 8);
    Ok(())
}

#[test]
#[ignore = "extended scenario coverage; run cargo gpu-extended"]
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
#[ignore = "extended scenario coverage; run cargo gpu-extended"]
fn maintenance_and_wear_are_combined_before_condition_clipping() -> Result<()> {
    let mut config = base_config()?;
    config.population.as_mut().unwrap().firms[0].capacity = 10;
    config.research = Some(ResearchConfig {
        equipment: vec![Equipment {
            firm: 100,
            remaining_service: 100,
            condition: 900,
            nameplate: 10,
            downtime: 0,
            wear_per_packet: 100,
            maintenance_good: Good::Bread,
            maintenance_ticks: 1,
            restoration_per_tick: 500,
        }],
        ..ResearchConfig::default()
    });
    let scenario = Scenario::new(config)?;
    let mut simulation = Simulation::new(scenario.clone(), support::device(64)?)?;
    simulation.advance()?;
    let state = simulation.download()?;
    let observation = state
        .research
        .as_ref()
        .unwrap()
        .equipment(scenario.config().research.as_ref().unwrap(), 0)?;
    assert_eq!(state.get(Field::Produced, 6), 6);
    assert_eq!(observation.condition, 800);
    assert_eq!(observation.remaining_service, 94);
    assert_eq!(observation.maintenance_consumed, 1);
    Ok(())
}

#[test]
#[ignore = "extended scenario coverage; run cargo gpu-extended"]
fn research_checkpoint_rejects_stale_profiles_and_unbacked_summaries() -> Result<()> {
    use bad_econ::simulation::research::ResearchRuntime;
    let mut config = base_config()?;
    config.research = Some(ResearchConfig {
        labs: vec![lab()],
        ..ResearchConfig::default()
    });
    let scenario = Scenario::new(config)?;
    let initial = scenario.initial()?;
    let config = scenario.config().research.as_ref().unwrap();
    let device = support::device(64)?;
    let runtime = ResearchRuntime::new(&device, config, &scenario, &initial)?;
    let snapshot = runtime.snapshot(&device)?;
    let mut changed = snapshot.clone();
    changed.profile = "unknown".into();
    assert!(ResearchRuntime::restore(&device, config, &scenario, &initial, &changed).is_err());
    changed = snapshot.clone();
    changed.table_hash = "stale".into();
    assert!(ResearchRuntime::restore(&device, config, &scenario, &initial, &changed).is_err());
    changed = snapshot.clone();
    changed.summary.consumed[0] = 1;
    assert!(ResearchRuntime::restore(&device, config, &scenario, &initial, &changed).is_err());
    changed = snapshot.clone();
    changed.configuration_hash = "other".into();
    assert!(ResearchRuntime::restore(&device, config, &scenario, &initial, &changed).is_err());
    let mut other_config = scenario.config().clone();
    other_config.research.as_mut().unwrap().labs[0].projects[0].capacity = 0;
    let other = Scenario::new(other_config)?;
    assert!(
        ResearchRuntime::restore(
            &device,
            other.config().research.as_ref().unwrap(),
            &other,
            &other.initial()?,
            &snapshot
        )
        .is_err()
    );
    changed = snapshot;
    changed.words.clear();
    assert!(ResearchRuntime::restore(&device, config, &scenario, &initial, &changed).is_err());
    Ok(())
}

#[test]
#[ignore = "extended scenario coverage; run cargo gpu-extended"]
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

#[test]
#[ignore = "extended scenario coverage; run cargo gpu-extended"]
fn positive_matrix_support_and_declared_edges_have_distinct_closures() -> Result<()> {
    use bad_econ::simulation::{
        analytics::{Coefficient, SparseMatrix},
        research::capabilities::boundary::DeviceCapabilityGraph,
    };
    let device = support::device(64)?;
    let matrix = SparseMatrix::new(
        4,
        4,
        vec![
            Coefficient {
                row: 0,
                column: 1,
                numerator: 1,
                denominator: 2,
            },
            Coefficient {
                row: 1,
                column: 2,
                numerator: 1,
                denominator: 3,
            },
            Coefficient {
                row: 2,
                column: 3,
                numerator: 0,
                denominator: 1,
            },
        ],
    )?;
    let support = DeviceCapabilityGraph::support(&device, &matrix, &[1, 0, 0, 0])?;
    assert_eq!(support.closure(&device)?.download(&device)?, [1, 1, 1, 0]);
    let declared = CapabilityGraph::new(
        4,
        (0..3)
            .map(|node| CapabilityEdge {
                prerequisite: node,
                dependent: node + 1,
            })
            .collect(),
    )?;
    assert_eq!(
        declared
            .closure(&device, &[1, 0, 0, 0])?
            .download(&device)?,
        [1, 1, 1, 1]
    );
    assert!(
        DeviceCapabilityGraph::support(&device, &SparseMatrix::new(1, 2, vec![])?, &[0]).is_err()
    );
    Ok(())
}

#[test]
#[ignore = "extended scenario coverage; run cargo gpu-extended"]
fn production_requires_every_installed_readiness_gate() -> Result<()> {
    for gate in 0..5 {
        let mut config = base_config()?;
        let mut installed = lab();
        installed.projects.clear();
        installed.adoption = AdoptionPolicy::Manual;
        installed.discovered = u32::from(gate > 0);
        installed.mastered = u32::from(gate > 1);
        installed.permitted = u32::from(gate != 2);
        installed.adopted = u32::from(gate == 4);
        config.research = Some(ResearchConfig {
            labs: vec![installed],
            ..ResearchConfig::default()
        });
        let mut simulation = Simulation::new(Scenario::new(config)?, support::device(64)?)?;
        simulation.advance()?;
        assert_eq!(
            simulation.download()?.get(Field::Produced, 6),
            if gate == 4 { 5 } else { 0 }
        );
    }
    Ok(())
}

#[test]
#[ignore = "extended scenario coverage; run cargo gpu-extended"]
fn successful_grants_are_reusable_and_local_to_the_implementer() -> Result<()> {
    let mut config = base_config()?;
    config.population.as_mut().unwrap().firms[1].cash = 100_000;
    let mut other = lab();
    other.owner = 101;
    other.researcher = 2;
    other.projects.clear();
    config.research = Some(ResearchConfig {
        labs: vec![lab(), other],
        ..ResearchConfig::default()
    });
    let scenario = Scenario::new(config)?;
    let mut simulation = Simulation::new(scenario.clone(), support::device(64)?)?;
    simulation.advance()?;
    let state = simulation.download()?;
    let research = state.research.as_ref().unwrap();
    let config = scenario.config().research.as_ref().unwrap();
    assert_eq!(research.lab(config, 0)?.adopted, 1);
    assert_eq!(research.lab(config, 1)?.discovered, 0);
    for phase in [Phase::Opening, Phase::Observation, Phase::Proposal] {
        simulation.phase(phase, Trace::Record)?;
    }
    let state = simulation.download()?;
    assert_eq!(state.get(Field::Plan, 6), 5);
    assert_eq!(state.get(Field::Plan, 7), 0);
    assert_eq!(state.research.as_ref().unwrap().lab(config, 0)?.adopted, 1);
    Ok(())
}

#[test]
#[ignore = "extended scenario coverage; run cargo gpu-extended"]
fn research_effort_and_exposure_follow_the_versioned_policy_grid() -> Result<()> {
    let mut config = base_config()?;
    let mut configured = lab();
    let project = &mut configured.projects[0];
    project.minimum_effort = 100;
    project.pioneer_effort = 300;
    project.exposure = 500;
    project.capacity = 500;
    project.relatedness = 1000;
    project.shape = 2;
    config.research = Some(ResearchConfig {
        labs: vec![configured],
        ..ResearchConfig::default()
    });
    let scenario = Scenario::new(config)?;
    let mut simulation = Simulation::new(scenario.clone(), support::device(64)?)?;
    simulation.advance()?;
    let state = simulation.download()?;
    let project = state.research.as_ref().unwrap().project(
        scenario.config().research.as_ref().unwrap(),
        0,
        0,
    )?;
    assert_eq!(project.probability, 375318);
    assert_eq!(project.exposure, 816);
    assert_eq!(project.attempts, 1);
    assert_eq!(state.get(Field::ResearchMinutes, 0), 100);
    Ok(())
}

#[test]
#[ignore = "extended scenario coverage; run cargo gpu-extended"]
fn strike_participation_and_research_share_the_actual_household_clock() -> Result<()> {
    use bad_econ::simulation::research::{AttemptOutcome, Participation};
    let mut config = base_config()?;
    let mut configured = lab();
    configured.production_requirements = 0;
    let policy = |household, grievance, cost| StrikePolicy {
        household,
        grievance,
        solidarity: grievance,
        trust: grievance,
        efficacy: grievance,
        peers: grievance,
        cost,
        risk: cost,
    };
    config.research = Some(ResearchConfig {
        labs: vec![configured],
        strikes: vec![policy(1, 1000, 0), policy(2, 0, 1000)],
        ..ResearchConfig::default()
    });
    let scenario = Scenario::new(config)?;
    let mut simulation = Simulation::new(scenario.clone(), support::device(64)?)?;
    simulation.advance()?;
    let state = simulation.download()?;
    let saved = state.research.as_ref().unwrap();
    let config = scenario.config().research.as_ref().unwrap();
    let striking = saved.strike(config, 0)?;
    let working = saved.strike(config, 1)?;
    assert_eq!(striking.score, 4500);
    assert_eq!(working.score, -4500);
    assert_eq!(striking.participation, Participation::Striking);
    assert_eq!(working.participation, Participation::Working);
    assert_eq!(state.get(Field::Hours, 0), 0);
    assert_eq!(state.get(Field::Time, 0), 0);
    assert_eq!(state.get(Field::Hours, 1), 60);
    assert_eq!(saved.project(config, 0, 0)?.outcome, AttemptOutcome::NoTime);
    assert_eq!(saved.summary.attempts, 0);
    assert_eq!(saved.summary.consumed, [0; Good::COUNT]);
    Ok(())
}

#[test]
#[ignore = "extended scenario coverage; run cargo gpu-extended"]
fn research_rows_preserve_multiple_records_and_checkpoint_replay() -> Result<()> {
    let mut config = base_config()?;
    for firm in &mut config.population.as_mut().unwrap().firms {
        firm.cash = 100_000;
    }
    let mut first = lab();
    first.permitted = 3;
    first.projects.push(ResearchProject {
        identity: 901,
        outcomes: 2,
        exposure: 300,
        inputs: Good::food([0, 0, 1]),
        ..first.projects[0].clone()
    });
    let mut second = lab();
    second.owner = 101;
    second.researcher = 2;
    second.permitted = 4;
    second.production_requirements = 4;
    second.projects[0].identity = 902;
    second.projects[0].outcomes = 4;
    second.projects[0].exposure = 700;
    config.research = Some(ResearchConfig {
        labs: vec![first, second],
        equipment: [100, 102]
            .into_iter()
            .enumerate()
            .map(|(index, firm)| Equipment {
                firm,
                remaining_service: (1 << 33) + index as u64 * 100,
                condition: 800 + index as u32 * 100,
                nameplate: 5,
                downtime: index as u32,
                wear_per_packet: 10 + index as u32 * 10,
                maintenance_good: Good::Bread,
                maintenance_ticks: 0,
                restoration_per_tick: 0,
            })
            .collect(),
        natural_stocks: [100, 102]
            .into_iter()
            .enumerate()
            .map(|(index, firm)| NaturalStock {
                firm,
                stock: 20 + index as u32 * 10,
                capacity: 40 + index as u32 * 10,
                renewal_rate: 0,
                extraction_per_packet: index as u32 + 1,
            })
            .collect(),
        strikes: [5, 6]
            .into_iter()
            .enumerate()
            .map(|(index, household)| StrikePolicy {
                household,
                grievance: index as u32 * 1000,
                solidarity: 0,
                trust: 0,
                efficacy: 0,
                peers: 0,
                cost: 1000,
                risk: 1000,
            })
            .collect(),
    });
    let scenario = Scenario::new(config)?;
    let research = scenario.config().research.as_ref().unwrap();
    let mut simulation = Simulation::new(scenario.clone(), support::device(64)?)?;
    let initial = simulation.download()?;
    let initial_research = initial.research.as_ref().unwrap();
    assert_eq!(initial_research.project(research, 0, 1)?.exposure, 300);
    assert_eq!(initial_research.project(research, 1, 0)?.exposure, 700);
    for _ in 0..2 {
        simulation.advance()?;
        let state = simulation.download()?;
        let snapshot = state.research.as_ref().unwrap();
        for (index, machine) in research.equipment.iter().enumerate() {
            let actor = scenario.index_of(machine.firm).unwrap();
            let produced = state.get(Field::Produced, actor);
            let actual = snapshot.equipment(research, index)?;
            assert_eq!(
                actual.remaining_service,
                machine.remaining_service - produced
            );
            assert_eq!(
                actual.condition,
                machine.condition - produced as u32 * machine.wear_per_packet
            );
            let stock = &research.natural_stocks[index];
            let extracted = produced * u64::from(stock.extraction_per_packet);
            assert_eq!(snapshot.stock(research, index)?.extracted, extracted);
            assert_eq!(
                snapshot.stock(research, index)?.stock,
                stock.stock - extracted as u32
            );
        }
        assert_eq!(snapshot.lab(research, 0)?.permitted, 3);
        assert_eq!(snapshot.lab(research, 1)?.permitted, 4);
        assert!(snapshot.project(research, 0, 1)?.attempts > 0);
        assert!(snapshot.project(research, 1, 0)?.attempts > 0);
        assert_ne!(
            snapshot.strike(research, 0)?.score,
            snapshot.strike(research, 1)?.score
        );
    }
    let checkpoint = simulation.checkpoint()?;
    let encoded = serde_json::to_string(&checkpoint)?;
    simulation.advance()?;
    let expected = simulation.download()?;
    let mut restored = Simulation::new(scenario, support::device(64)?)?;
    restored.command(SimulationCommand::Restore(
        serde_json::from_str::<Checkpoint>(&encoded)?.into(),
    ))?;
    restored.advance()?;
    assert_eq!(restored.download()?, expected);
    Ok(())
}

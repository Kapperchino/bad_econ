use super::support;
use anyhow::Result;
use bad_econ::{
    arithmetic::DrawKey,
    domain::*,
    ledger::*,
    runtime::*,
    simulation::{self, Simulation, Trace},
};

fn event(identity: u64, legs: Vec<Leg>) -> Result<Bundle> {
    Bundle::new(
        EventKey {
            day: 0,
            phase: 0,
            subphase: 0,
            priority: 0,
            identity,
        },
        legs,
    )
}

#[test]
fn authorization_flags_preserve_wide_guards_and_empty_events() -> Result<()> {
    let wide = (1u64 << 32) + 9;
    let balances = [0, 9, wide, u64::MAX, 0];
    let mut events = vec![event(0, Vec::new())?];
    for authorization in [
        Authorization::Any,
        Authorization::Equal(9),
        Authorization::Equal(wide),
        Authorization::AtLeast(9),
        Authorization::AtLeast(wide),
        Authorization::AtLeast(u64::MAX),
    ] {
        for coordinate in 0..4 {
            events.push(event(
                events.len() as u64,
                vec![
                    Leg {
                        coordinate: Coordinate(coordinate),
                        debit: 0,
                        credit: 0,
                        authorization,
                    },
                    Leg::transfer(Coordinate(4), 0, 1),
                ],
            )?);
        }
    }
    events.push(event(events.len() as u64, Vec::new())?);
    let mut expected = balances;
    let decisions = serial(&mut expected, &events)?;
    let actual = simulation::ledger(&support::device(64)?, &balances, &events)?;
    assert_eq!(actual.decisions, decisions);
    assert_eq!(actual.state, expected);
    assert_eq!(actual.state[4], 12);
    Ok(())
}

#[test]
fn branched_graph_and_grouped_debits_match_serial_ledger() -> Result<()> {
    let mut expected = vec![1000; 12];
    let events = (0..256u64)
        .map(|id| {
            let draw = DrawKey {
                seed: 91,
                day: 0,
                phase: 0,
                actor: id,
                purpose: 0,
                index: 0,
            }
            .draw();
            let from = (draw % 12) as u32;
            let to = ((draw >> 8) % 12) as u32;
            let guard = ((draw >> 16) % 12) as u32;
            let amount = (draw >> 24) % 1200;
            event(
                id,
                vec![
                    Leg::transfer(Coordinate(from), amount, 0),
                    Leg::transfer(Coordinate(to), 0, amount),
                    Leg {
                        coordinate: Coordinate(guard),
                        debit: 0,
                        credit: 0,
                        authorization: Authorization::AtLeast(10),
                    },
                ],
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let decisions = serial(&mut expected, &events)?;
    let actual = simulation::ledger(&support::device(64)?, &[1000; 12], &events)?;
    assert_eq!(actual.state, expected);
    assert_eq!(actual.decisions, decisions);
    assert_eq!(actual.state.iter().sum::<u64>(), 12_000);
    assert!(actual.frontier_depth > 1);
    Ok(())
}

#[test]
fn checkpoint_rejects_tampering_and_keeps_last_audited_boundary() -> Result<()> {
    let scenario = Scenario::baseline(30, 8, Experiment::Shortage)?;
    let mut simulation = Simulation::new(scenario.clone(), support::device(64)?)?;
    simulation.command(SimulationCommand::AdvanceDay)?;
    assert_eq!(simulation.status, RuntimeState::Paused);
    let checkpoint = simulation.checkpoint()?;
    assert!(
        simulation
            .phase(Phase::PhysicalClose, Trace::Record)
            .is_err()
    );
    simulation.phase(Phase::Opening, Trace::Discard)?;
    assert!(simulation.phase(Phase::Opening, Trace::Discard).is_err());
    assert!(simulation.advance().is_err());
    simulation.phase(Phase::Observation, Trace::Discard)?;
    assert_eq!(simulation.checkpoint()?, checkpoint);
    let mut json = serde_json::to_value(&checkpoint)?;
    json["state"]["values"][0] = serde_json::json!(0);
    assert!(
        serde_json::from_value::<Checkpoint>(json)?
            .restore()
            .is_err()
    );
    let mut invalid = scenario.initial()?;
    let mut actor = invalid.actors.at(0);
    actor.kind = 99;
    invalid.actors.set(0, actor);
    assert!(Checkpoint::new(&scenario, &invalid).is_err());
    invalid = scenario.initial()?;
    invalid.values.clear();
    assert!(Checkpoint::new(&scenario, &invalid).is_err());
    for phase in Phase::ALL.into_iter().skip(2) {
        simulation.phase(phase, Trace::Record)?;
    }
    assert_eq!(simulation.snapshot.day, 2);
    assert_eq!(
        simulation.checkpoint()?.restore()?.state,
        simulation.download()?
    );
    simulation.command(SimulationCommand::Restore(checkpoint.into()))?;
    let state = simulation.download()?;
    for identity in [1, 31] {
        let request = InspectionRequest::new(identity, &scenario)?;
        let expected = Inspection::from_state(&scenario, &state, &request)?;
        assert_eq!(simulation.inspect(request)?, expected);
    }
    simulation.command(SimulationCommand::Resume)?;
    assert_eq!(simulation.status, RuntimeState::Running);
    simulation.command(SimulationCommand::Pause)?;
    assert_eq!(simulation.status, RuntimeState::Paused);
    Ok(())
}

#[test]
fn arithmetic_fault_rolls_back_and_reset_recovers() -> Result<()> {
    let scenario = Scenario::baseline(1, 42, Experiment::Concentrated)?;
    let mut state = scenario.initial()?;
    state.set(Field::CropWork, 1, 1);
    state.day = 270;
    let grain = state.get(Field::Grain, 1);
    state.set(Field::Grain, 0, grain);
    state.set(Field::Grain, 1, 0);
    state.set(Field::Produced, 1, u64::MAX);
    state.set(Field::Used, 2, u64::MAX);
    state.values[USED_FIELDS * state.actors.len() + 2] = u64::MAX;
    let checkpoint = Checkpoint::new(&scenario, &state)?;
    let mut simulation = Simulation::new(scenario.clone(), support::device(64)?)?;
    simulation.command(SimulationCommand::Restore(checkpoint.clone().into()))?;
    let previous = simulation.snapshot.clone();
    assert!(simulation.advance().is_err());
    assert!(matches!(
        simulation.status,
        RuntimeState::Fault {
            last_audited_day: 270,
            ..
        }
    ));
    assert_eq!(simulation.snapshot, previous);
    assert_eq!(simulation.checkpoint()?, checkpoint);
    assert_eq!(simulation.download()?, state);
    assert!(simulation.command(SimulationCommand::Resume).is_err());
    simulation.command(SimulationCommand::Reset(scenario.into()))?;
    simulation.command(SimulationCommand::AdvanceDay)?;
    assert_eq!(simulation.snapshot.day, 1);
    assert_eq!(simulation.status, RuntimeState::Paused);
    Ok(())
}

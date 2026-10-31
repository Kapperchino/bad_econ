#![cfg(feature = "compute")]
mod support;
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
fn dispatch_is_buyer_owned_and_output_matures_at_close() -> Result<()> {
    let scenario = Scenario::baseline(1, 42, Experiment::Concentrated)?;
    let mut simulation = Simulation::new(scenario.clone(), support::device(32)?)?;
    simulation.command(SimulationCommand::AdvanceDay)?;
    simulation.command(SimulationCommand::AdvanceDay)?;
    for phase in [
        Phase::Opening,
        Phase::Observation,
        Phase::Proposal,
        Phase::Commitment,
        Phase::WorkPayment,
    ] {
        simulation.phase(phase, Trace::Record)?;
    }
    let before = simulation.download()?;
    simulation.phase(Phase::ProvisionPurchase, Trace::Record)?;
    let dispatched = simulation.download()?;
    assert_eq!(dispatched.get(Field::Bread, 0), before.get(Field::Bread, 0));
    assert_eq!(dispatched.get(Field::Transit, 0), 1);
    assert!(dispatched.get(Field::Cash, 0) < before.get(Field::Cash, 0));
    simulation.phase(Phase::PhysicalClose, Trace::Record)?;
    assert_eq!(simulation.download()?.get(Field::Bread, 0), 1);
    simulation.phase(Phase::AccountsDecisions, Trace::Record)?;
    simulation.phase(Phase::Audit, Trace::Record)?;
    simulation.phase(Phase::Opening, Trace::Record)?;
    let delivered = simulation.download()?;
    assert_eq!(delivered.get(Field::Bread, 0), 2);
    assert_eq!(delivered.get(Field::Transit, 0), 0);
    let shortage = Scenario::baseline(6, 17, Experiment::Shortage)?;
    let mut simulation = Simulation::new(shortage.clone(), support::device(64)?)?;
    for phase in [
        Phase::Opening,
        Phase::Observation,
        Phase::Proposal,
        Phase::Commitment,
        Phase::WorkPayment,
        Phase::ProvisionPurchase,
    ] {
        simulation.phase(phase, Trace::Record)?;
    }
    let pending = simulation.download()?;
    let firms = shortage.config().households as usize..shortage.actors();
    assert!(
        firms
            .clone()
            .map(|i| pending.get(Field::Pipeline, i))
            .sum::<u64>()
            > 0
    );
    assert_eq!(
        firms
            .clone()
            .map(|i| pending.get(Field::Grain, i))
            .sum::<u64>(),
        0
    );
    simulation.phase(Phase::PhysicalClose, Trace::Record)?;
    let closed = simulation.download()?;
    assert_eq!(
        firms
            .clone()
            .map(|i| closed.get(Field::Pipeline, i))
            .sum::<u64>(),
        0
    );
    assert!(firms.map(|i| closed.get(Field::Grain, i)).sum::<u64>() > 0);
    Ok(())
}

#[test]
fn checkpoint_rejects_tampering_and_keeps_last_audited_boundary() -> Result<()> {
    let scenario = Scenario::baseline(30, 8, Experiment::Shortage)?;
    let mut simulation = Simulation::new(scenario.clone(), support::device(64)?)?;
    simulation.command(SimulationCommand::AdvanceDay)?;
    assert_eq!(simulation.status, RuntimeState::Paused);
    let checkpoint = simulation.checkpoint()?;
    simulation.phase(Phase::Opening, Trace::Discard)?;
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
    invalid.actors[0].kind = 99;
    assert!(Checkpoint::new(&scenario, &invalid).is_err());
    invalid = scenario.initial()?;
    invalid.values.clear();
    assert!(Checkpoint::new(&scenario, &invalid).is_err());
    simulation.command(SimulationCommand::Restore(checkpoint.into()))?;
    let state = simulation.download()?;
    for identity in [1, 31] {
        let request = InspectionRequest::new(identity, &scenario)?;
        let expected = Inspection::from_state(&state, &request)?;
        assert_eq!(simulation.inspect(request)?, expected);
    }
    simulation.command(SimulationCommand::Resume)?;
    assert_eq!(simulation.status, RuntimeState::Running);
    simulation.command(SimulationCommand::Pause)?;
    assert_eq!(simulation.status, RuntimeState::Paused);
    Ok(())
}

#[test]
fn high_word_cash_and_reserve_distributions_remain_exact() -> Result<()> {
    let mut config = Scenario::baseline(10, 9, Experiment::Baseline)?
        .config()
        .clone();
    config.firm_cash = u64::MAX / 8;
    config.household_cash = 10;
    config.reserve = 1;
    config.distribution_rate = 1000;
    let scenario = Scenario::new(config)?;
    let mut simulation = Simulation::new(scenario, support::device(64)?)?;
    let initial_cash = simulation.snapshot.cash;
    simulation.advance()?;
    assert_eq!(simulation.snapshot.cash, initial_cash);
    let state = simulation.download()?;
    assert!((10..13).all(|firm| state.get(Field::Cash, firm) == 1));
    Ok(())
}

#[test]
fn arithmetic_fault_rolls_back_and_reset_recovers() -> Result<()> {
    let scenario = Scenario::baseline(1, 42, Experiment::Concentrated)?;
    let mut state = scenario.initial()?;
    let grain = state.get(Field::Grain, 1);
    state.set(Field::Grain, 0, grain);
    state.set(Field::Grain, 1, 0);
    state.set(Field::Produced, 1, u64::MAX);
    state.set(Field::Used, 2, u64::MAX);
    let checkpoint = Checkpoint::new(&scenario, &state)?;
    let mut simulation = Simulation::new(scenario.clone(), support::device(64)?)?;
    simulation.command(SimulationCommand::Restore(checkpoint.clone().into()))?;
    let previous = simulation.snapshot.clone();
    assert!(simulation.advance().is_err());
    assert!(matches!(
        simulation.status,
        RuntimeState::Fault {
            last_audited_day: 0,
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

#[test]
fn phase_machine_rejects_out_of_order_and_checkpoints_completed_days() -> Result<()> {
    let scenario = Scenario::baseline(12, 42, Experiment::Baseline)?;
    let mut simulation = Simulation::new(scenario, support::device(64)?)?;
    assert!(
        simulation
            .phase(Phase::PhysicalClose, Trace::Record)
            .is_err()
    );
    simulation.phase(Phase::Opening, Trace::Discard)?;
    assert!(simulation.phase(Phase::Opening, Trace::Discard).is_err());
    assert!(simulation.advance().is_err());
    for phase in Phase::ALL.into_iter().skip(1) {
        simulation.phase(phase, Trace::Record)?;
    }
    assert_eq!(
        simulation.checkpoint()?.restore()?.state,
        simulation.download()?
    );
    simulation.advance()?;
    assert_eq!(simulation.snapshot.day, 2);
    Ok(())
}

#[test]
fn delivery_date_capacity_failure_retains_audited_checkpoint() -> Result<()> {
    let scenario = Scenario::baseline(1, 42, Experiment::Concentrated)?;
    let mut state = scenario.initial()?;
    state.day = u64::from(u32::MAX) - scenario.ring() as u64;
    let checkpoint = Checkpoint::new(&scenario, &state)?;
    let mut simulation = Simulation::new(scenario, support::device(64)?)?;
    simulation.command(SimulationCommand::Restore(checkpoint.clone().into()))?;
    assert!(simulation.advance().is_err());
    assert!(matches!(simulation.status, RuntimeState::Fault { .. }));
    assert_eq!(simulation.checkpoint()?, checkpoint);
    assert_eq!(simulation.download()?, state);
    Ok(())
}

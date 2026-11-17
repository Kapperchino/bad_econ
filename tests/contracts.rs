#![cfg(feature = "compute")]
use super::support;
use anyhow::Result;
use bad_econ::{domain::*, ledger::*, runtime::*, simulation};

fn event(id: u64, legs: &[Leg]) -> Bundle {
    Bundle::new(
        EventKey {
            day: 0,
            phase: 0,
            subphase: 0,
            priority: 0,
            identity: id,
        },
        legs.iter().copied(),
    )
    .unwrap()
}
fn transfer(cell: u32, debit: u64, credit: u64) -> Leg {
    Leg::transfer(Coordinate(cell), debit, credit)
}

#[test]
fn gross_affordability_repeated_legs_and_atomic_overflow() -> Result<()> {
    let device = support::device(64)?;
    let gross = simulation::ledger(&device, &[1], &[event(1, &[transfer(0, 3, 2)])])?;
    assert_eq!(gross.decisions, [Decision::Rejected]);
    assert_eq!(gross.state, [1]);
    let grouped = simulation::ledger(
        &device,
        &[5],
        &[event(1, &[transfer(0, 3, 0), transfer(0, 3, 0)])],
    )?;
    assert_eq!(grouped.decisions, [Decision::Rejected]);
    assert_eq!(grouped.state, [5]);
    assert!(
        simulation::ledger(
            &device,
            &[100, u64::MAX],
            &[event(1, &[transfer(0, 1, 0), transfer(1, 0, 1)])],
        )
        .is_err()
    );
    Ok(())
}

#[test]
fn serialized_bundles_preserve_grouped_gross_debits_and_authorization() -> Result<()> {
    let key = event(1, &[]).key();
    let definition = |legs: Vec<Leg>| serde_json::json!({ "key": key, "legs": legs });
    let bundle: Bundle = serde_json::from_value(definition(vec![
        transfer(0, 3, 0),
        transfer(0, 3, 0),
        transfer(1, 0, 6),
    ]))?;
    let mut state = [5, 0];
    assert_eq!(bundle.apply(&mut state)?, Decision::Rejected);
    assert_eq!(state, [5, 0]);
    let actual = simulation::ledger(
        &support::device(64)?,
        &[5, 0],
        std::slice::from_ref(&bundle),
    )?;
    assert_eq!(actual.state, state);
    assert_eq!(actual.decisions, [Decision::Rejected]);
    assert_eq!(bundle.legs().len(), 2);
    assert_eq!(
        serde_json::from_str::<Bundle>(&serde_json::to_string(&bundle)?)?,
        bundle
    );
    assert!(
        serde_json::from_value::<Bundle>(definition(vec![
            transfer(0, u64::MAX, 0),
            transfer(0, 1, 0),
        ]))
        .is_err()
    );
    assert!(
        serde_json::from_value::<Bundle>(definition(vec![
            Leg::read(Coordinate(0), 1),
            Leg::read(Coordinate(0), 2),
        ]))
        .is_err()
    );
    assert!(
        serde_json::from_value::<Bundle>(definition(
            (0..=MAX_LEGS)
                .map(|index| transfer(index as u32, 1, 0))
                .collect(),
        ))
        .is_err()
    );
    Ok(())
}

#[test]
fn authorization_reads_create_dependencies() -> Result<()> {
    let mut state = vec![1, 0, 0];
    let events = [
        event(1, &[transfer(0, 1, 0)]),
        event(2, &[Leg::read(Coordinate(0), 1), transfer(1, 0, 1)]),
        event(3, &[Leg::read(Coordinate(0), 0), transfer(2, 0, 1)]),
    ];
    assert_eq!(
        Schedule::new(&events)?.frontiers,
        [vec![0], vec![1], vec![2]]
    );
    assert_eq!(
        serial(&mut state, &events)?,
        [Decision::Accepted, Decision::Rejected, Decision::Accepted]
    );
    assert_eq!(state, [0, 0, 1]);
    let actual = simulation::ledger(&support::device(64)?, &[1, 0, 0], &events)?;
    assert_eq!(actual.state, state);
    assert_eq!(
        actual.decisions,
        [Decision::Accepted, Decision::Rejected, Decision::Accepted]
    );
    assert!(serial(&mut state, &[])?.is_empty());
    assert!(Schedule::new(&[events[1].clone(), events[0].clone()]).is_err());
    Ok(())
}

#[cfg(feature = "compute")]
#[test]
fn transport_ownership_time_budgets_and_checkpoint_replay() -> Result<()> {
    let scenario = Scenario::baseline(120, 42, Experiment::Transport)?;
    let mut simulation =
        bad_econ::simulation::Simulation::new(scenario.clone(), support::device(64)?)?;
    (0..8).try_for_each(|_| simulation.advance())?;
    let initial_cash = simulation.download()?.initial_cash;
    assert_eq!(simulation.snapshot.cash, initial_cash);
    let checkpoint = simulation.checkpoint()?;
    let serialized = serde_json::to_string(&checkpoint)?;
    let checkpoint: Checkpoint = serde_json::from_str(&serialized)?;
    (0..4).try_for_each(|_| simulation.advance())?;
    let expected = simulation.download()?;
    simulation.command(SimulationCommand::Restore(checkpoint.into()))?;
    (0..4).try_for_each(|_| simulation.advance())?;
    assert_eq!(simulation.download()?, expected);
    let state = simulation.download()?;
    (0..scenario.config().households as usize).for_each(|i| {
        assert!(
            state.get(Field::Hours, i) + state.get(Field::Time, i)
                <= u64::from(scenario.config().work_cap)
        )
    });
    Ok(())
}

#[cfg(feature = "compute")]
use crate::support;
use anyhow::Result;
use bad_econ::{arithmetic::*, ledger::*};
#[cfg(feature = "compute")]
use bad_econ::{domain::*, runtime::*};

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
    let mut state = vec![1];
    assert_eq!(
        event(1, &[transfer(0, 3, 2)]).apply(&mut state)?,
        Decision::Rejected
    );
    let mut state = vec![5];
    assert_eq!(
        event(1, &[transfer(0, 3, 0), transfer(0, 3, 0)]).apply(&mut state)?,
        Decision::Rejected
    );
    let mut state = vec![100, u64::MAX];
    assert!(
        event(1, &[transfer(0, 1, 0), transfer(1, 0, 1)])
            .apply(&mut state)
            .is_err()
    );
    assert_eq!(state, [100, u64::MAX]);
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
    assert!(serial(&mut state, &[])?.is_empty());
    assert!(Schedule::new(&[events[1].clone(), events[0].clone()]).is_err());
    Ok(())
}

#[cfg(feature = "compute")]
#[test]
fn lean_indivisible_jobs_release_capacity_after_rejection() -> Result<()> {
    let events = [
        event(1, &[transfer(0, 6, 0)]),
        event(2, &[transfer(0, 4, 0)]),
    ];
    let result = bad_econ::simulation::ledger(&support::device(64)?, &[5], &events)?;
    assert_eq!(result.decisions, [Decision::Rejected, Decision::Accepted]);
    assert_eq!(result.state, [1]);
    Ok(())
}

#[test]
fn lean_packets_claims_quotes_and_draws() -> Result<()> {
    let packet = Packet::new(&[Ratio::new(1, 2)?, Ratio::new(1, 3)?, Ratio::new(1, 1)?])?;
    assert_eq!(packet.activities, 6);
    assert_eq!(packet.ticks, [3, 2, 6]);
    assert_eq!(packet.capacity(4, &[8, 5, 100])?, 2);
    assert_eq!(Packet::new(&[])?.capacity(4, &[])?, 4);
    assert_eq!(Packet::new(&[Ratio::new(0, 1)?])?.capacity(4, &[0])?, 4);
    let thirds = Claim::new(3, 0)?.accrue(1)?.accrue(1)?;
    assert_eq!(thirds.remainder(), 2);
    assert_eq!(thirds.accrue(1)?.whole(), 1);
    assert_eq!(Claim::new(3, 5)?.settle(1)?.remainder(), 2);
    assert_eq!(quote(2001, 0, 2009, 10)?, 2000);
    assert!(quote(2000, 2005, 2009, 10).is_err());
    assert!(!Probability::new(1, 3)?.succeeds(2, 1)?);
    assert!(!Probability::new(0, 3)?.succeeds(64, 0)?);
    assert!(Probability::new(3, 3)?.succeeds(64, u64::MAX)?);
    assert!(Probability::new(1, 1)?.succeeds(2, 4).is_err());
    assert!(Probability::new(1, 1)?.succeeds(65, 0).is_err());
    assert_eq!(forecast(10, 14, 0, 250)?, 11);
    assert_eq!(
        DrawKey {
            seed: 42,
            day: 7,
            phase: 3,
            actor: 12,
            purpose: 9,
            index: 0
        }
        .draw(),
        9_825_991_519_559_520_269
    );
    assert_eq!(floor_ratio(u64::MAX, u64::MAX, u64::MAX)?, u64::MAX);
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

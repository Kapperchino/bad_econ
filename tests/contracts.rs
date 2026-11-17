use super::support;
use anyhow::Result;
use bad_econ::{ledger::*, simulation};

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
fn serialized_bundles_preserve_grouped_gross_debits_and_authorization() -> Result<()> {
    let key = event(1, &[]).key();
    let definition = |legs: Vec<Leg>| serde_json::json!({ "key": key, "legs": legs });
    let bundle: Bundle = serde_json::from_value(definition(vec![
        transfer(0, 3, 3),
        transfer(0, 3, 0),
        transfer(1, 0, 3),
    ]))?;
    let mut state = [5, 0];
    assert_eq!(bundle.apply(&mut state)?, Decision::Rejected);
    assert_eq!(state, [5, 0]);
    let device = support::device(64)?;
    let actual = simulation::ledger(&device, &[5, 0], std::slice::from_ref(&bundle))?;
    assert!(
        simulation::ledger(
            &device,
            &[100, u64::MAX],
            &[event(1, &[transfer(0, 1, 0), transfer(1, 0, 1)])],
        )
        .is_err()
    );
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

use super::*;
use crate::domain::{
    Checkpoint, Experiment, Field,
    demography::{LivingState, Occupation, Person, PersonSeed, Sex},
};
use crate::simulation::{CopyDirection, Simulation, decode, encode};

fn fixture() -> Result<Simulation> {
    let mut config = Scenario::industrial(4, 42, Experiment::Baseline)?
        .config()
        .clone();
    config.demography.annual_births_per_million_fertile_women = 0;
    config.demography.mortality.background_annual_ppm = [0; 5];
    config.demography.mortality.occupational_annual_ppm = [0; crate::domain::Good::COUNT];
    config.demography.mortality.famine_daily_ppm_at_thirty_days = 0;
    config.demography.mortality.cold_daily_ppm_at_seven_days = 0;
    config.demography.people = (0..4u64)
        .flat_map(|household| {
            [30, 25, 20]
                .into_iter()
                .enumerate()
                .map(move |(offset, age)| PersonSeed {
                    identity: 10_000 + household * 10 + offset as u64,
                    household: household + 1,
                    age_days: age * DAYS_PER_YEAR,
                    sex: Sex::Female,
                })
        })
        .collect();
    let mut simulation = Simulation::new(
        Scenario::new(config)?,
        crate::simulation::tests::device(64)?,
    )?;
    simulation.advance()?;
    Ok(simulation)
}

fn worker_identity(state: &State, household: usize) -> u64 {
    let person = &state.people[state.get(Field::Worker, household) as usize];
    assert_eq!(person.household, state.actors.at(household).identity);
    person.identity
}

#[test]
fn births_preserve_worker_identities_occupations_and_audited_rollback_on_gpu() -> Result<()> {
    let mut simulation = fixture()?;
    let previous = simulation.download()?;
    let checkpoint = simulation.checkpoint()?;
    let before = simulation.demography.observation(
        &simulation.device,
        &simulation.state,
        &simulation.config,
    )?;
    let mut births = vec![0; previous.actors.len()];
    births[0] = 1;
    births[3] = 2;
    simulation
        .device
        .write(simulation.demography.birth_buffer(), &births);
    let records = crate::simulation::registry::records::PendingRecords::from_dense(
        &vec![0; births.len()],
        &births,
    )?;
    let grown = simulation.demography.grow_births(
        &simulation.device,
        &records.records,
        previous.next_identity,
        &simulation.work.scratch,
    )?;
    grown.remap_workers_from(
        &simulation.device,
        &simulation.state,
        simulation.config.state_stride,
        &simulation.demography,
    );
    let mut actual = previous.clone();
    actual.people = grown.download(&simulation.device, &actual, CopyDirection::Save)?;
    let workers = decode(&simulation.device.read_range(
        &simulation.state,
        Field::Worker as usize * simulation.config.state_stride as usize * 2,
        previous.actors.len() * 2,
    )?);
    for household in 0..4 {
        actual.set(Field::Worker, household, workers[household]);
        assert_eq!(
            worker_identity(&actual, household),
            worker_identity(&previous, household)
        );
    }
    let after = grown.observation(&simulation.device, &simulation.state, &simulation.config)?;
    for occupation in Occupation::ALL {
        let additional = match occupation {
            Occupation::Dependent => 2,
            _ => 0,
        };
        assert_eq!(
            after.occupations[occupation as usize],
            before.occupations[occupation as usize] + additional
        );
    }
    assert_ne!(workers[1], previous.get(Field::Worker, 1));
    simulation.copy_audited(CopyDirection::Restore)?;
    assert_eq!(simulation.download()?, previous);
    assert_eq!(simulation.checkpoint()?, checkpoint);
    simulation.device.memory_budget =
        crate::simulation::MemoryBudget::Reported { available_bytes: 2 };
    assert!(simulation.advance().is_err());
    assert_eq!(simulation.download()?, previous);
    assert_eq!(simulation.checkpoint()?, checkpoint);
    Ok(())
}

#[test]
fn formation_and_combined_births_preserve_workers_and_match_constructor_reference_on_gpu()
-> Result<()> {
    #[derive(Clone, Copy)]
    enum Growth {
        Formation,
        BirthsAndFormation,
    }
    for growth in [Growth::Formation, Growth::BirthsAndFormation] {
        let mut simulation = fixture()?;
        let mut original = simulation.download()?;
        let parent = 3;
        let selected = original
            .people
            .iter()
            .rposition(|person| person.household == original.actors.at(parent).identity)
            .expect("selected departing adult");
        original.set(Field::Worker, parent, selected as u64);
        simulation.device.write(
            &simulation.state,
            bytemuck::cast_slice(&encode(&original.values)),
        );
        let identity = original.people[selected].identity;
        let mut expected = original.clone();
        let mut births = vec![0; original.actors.len()];
        let grown = match growth {
            Growth::Formation => simulation.demography.clone(),
            Growth::BirthsAndFormation => {
                births[0] = 1;
                simulation
                    .device
                    .write(simulation.demography.birth_buffer(), &births);
                let records = crate::simulation::registry::records::PendingRecords::from_dense(
                    &vec![0; births.len()],
                    &births,
                )?;
                expected.people.push(Person {
                    identity: expected.next_identity,
                    household: expected.actors.at(0).identity,
                    age_days: 0,
                    sex: Sex::Female,
                    state: LivingState::Living,
                    employer: None,
                    work_minutes: 0,
                    fatigue: 0,
                    famine_exposure: 0,
                    cold_exposure: 0,
                    birth_cooldown: 0,
                });
                expected.next_identity += 1;
                simulation.demography.grow_births(
                    &simulation.device,
                    &records.records,
                    original.next_identity,
                    &simulation.work.scratch,
                )?
            }
        };
        expected.set(Field::SplitPerson, parent, identity);
        let mut requests = vec![0; original.actors.len()];
        requests[parent] = (1 << 24) + 1;
        crate::simulation::enterprise::transitions(&mut expected, &simulation.scenario, &requests)?;
        let household = original.actors.len();
        let formed = grown.form_households(
            &simulation.device,
            &[crate::simulation::enterprise::HouseholdFormation {
                parent,
                household,
                person: identity,
            }],
            expected.actors.len(),
        )?;
        let target = simulation
            .device
            .tensor(bytemuck::cast_slice(&encode(&expected.values)));
        formed.remap_workers_from(
            &simulation.device,
            &target,
            expected.actors.len() as u32,
            &simulation.demography,
        );
        let mut actual = expected.clone();
        actual.values = decode(&simulation.device.read(&target)?);
        actual.people = formed.download(&simulation.device, &actual, CopyDirection::Save)?;
        assert_eq!(worker_identity(&actual, household), identity);
        assert_ne!(worker_identity(&actual, parent), identity);
        for existing in 0..parent {
            assert_eq!(
                worker_identity(&actual, existing),
                worker_identity(&original, existing)
            );
        }
        let mut reference = Simulation::from_checkpoint(
            Checkpoint::new(&simulation.scenario, &expected)?,
            simulation.device.clone(),
        )?;
        assert_eq!(actual, reference.download()?);
        assert_eq!(
            formed.observation(&simulation.device, &target, &reference.config)?,
            reference.snapshot.population
        );
        let checkpoint = Checkpoint::new(&simulation.scenario, &actual)?;
        assert_eq!(checkpoint.restore()?.state, actual);
    }
    Ok(())
}

#[test]
fn reordered_checkpoint_people_preserve_selected_worker_identities_on_gpu() -> Result<()> {
    let mut simulation = fixture()?;
    let previous = simulation.download()?;
    let mut reordered = previous.clone();
    reordered.people.reverse();
    for household in 0..4 {
        let identity = worker_identity(&previous, household);
        let selected = reordered
            .people
            .iter()
            .position(|person| person.identity == identity)
            .expect("selected worker");
        reordered.set(Field::Worker, household, selected as u64);
    }
    let mut restored = Simulation::from_checkpoint(
        Checkpoint::new(&simulation.scenario, &reordered)?,
        simulation.device.clone(),
    )?;
    let actual = restored.download()?;
    for household in 0..4 {
        assert_eq!(
            worker_identity(&actual, household),
            worker_identity(&previous, household)
        );
    }
    assert_eq!(actual, previous);
    Ok(())
}

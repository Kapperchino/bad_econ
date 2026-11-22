use super::*;
use crate::{
    domain::{Experiment, Scenario},
    simulation::{Simulation, encode, registry, tests::device},
};

#[test]
fn plane_ring_masks_match_individual_marks_with_partial_actor_groups_on_gpu() -> Result<()> {
    let scenario = Scenario::baseline(4097, 42, Experiment::Baseline)?;
    for workgroup in [32, 256] {
        let simulation = Simulation::new(scenario.clone(), device(workgroup)?)?;
        let mut state = scenario.initial()?;
        let count = state.actors.len();
        let good = Good::COUNT - 1;
        let field = BASE_FIELDS + good * scenario.ring() + scenario.ring() - 1;
        for actor in [0, 31, 32, count - 1] {
            let amount = actor as u64 + 1;
            let transit_field = match actor % 2 {
                0 => field,
                _ => field + Good::COUNT * scenario.ring(),
            };
            state.values[transit_field * count + actor] = amount;
            state.values[(TRANSIT_FIELDS + good) * count + actor] = amount;
            state.set(Field::Transit, actor, amount);
        }
        simulation.device.write(
            &simulation.state,
            bytemuck::cast_slice(&encode(&crate::simulation::tensor::state_values(
                &state.values,
                count,
                simulation.config.state_stride as usize,
            ))),
        );
        let mask = simulation.snapshot_copy.ring_mask().expect("ring mask");
        let hardware = &simulation.device.client.properties().hardware;
        let tracking = match (
            parallel::ScanMode::new(&simulation.device),
            hardware.plane_size_min,
            hardware.plane_size_max,
        ) {
            (parallel::ScanMode::Plane, 32, 32) => RingTracking::ObservedPlane,
            _ => RingTracking::Observed,
        };
        crate::simulation::tensor::reset_ring_mask(
            &simulation.device,
            mask,
            crate::simulation::tensor::RingMaskState::Empty,
        );
        ActorValidation::launch(
            &simulation.device,
            dispatch(
                &simulation.device,
                count.div_ceil(workgroup as usize),
                Good::COUNT + 1,
            ),
            CubeDim::new_1d(workgroup),
            ValidationInputLaunch {
                state: simulation.state.buffer_arg(),
                actors: simulation.actors.buffer_arg(),
                config: simulation.config.launch(),
                control: simulation.control.buffer_arg(),
                ring_mask: mask.buffer_arg(),
                ring_tracking: tracking,
            },
        );
        let coalesced = simulation.device.read(mask)?;
        crate::simulation::tensor::reset_ring_mask(
            &simulation.device,
            mask,
            crate::simulation::tensor::RingMaskState::Empty,
        );
        let indices = simulation
            .device
            .tensor(&(0..count as u32).collect::<Vec<_>>());
        SelectedValidation::launch(
            &simulation.device,
            dispatch(
                &simulation.device,
                count.div_ceil(workgroup as usize),
                Good::COUNT,
            ),
            CubeDim::new_1d(workgroup),
            SelectedValidationInputLaunch {
                state: simulation.state.buffer_arg(),
                indices: indices.into_buffer_arg(),
                config: simulation.config.launch(),
                control: simulation.control.buffer_arg(),
                ring_mask: mask.buffer_arg(),
                ring_tracking: RingTracking::Observed,
            },
        );
        assert_eq!(coalesced, simulation.device.read(mask)?);
        for actor in [0, 31] {
            let field = field + actor % 2 * Good::COUNT * scenario.ring();
            let word = (field * simulation.config.state_stride as usize + actor) * TICK_WORDS;
            assert_eq!(
                coalesced[word / crate::simulation::tensor::SNAPSHOT_BLOCK_WORDS],
                1
            );
        }
        assert_eq!(
            FaultFlags::read(&simulation.device.read(&simulation.control)?)?,
            FaultFlags::empty()
        );
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum TransitCase {
    Unchanged,
    Mismatch,
    RingOverflow,
    TotalOverflow,
}

#[test]
fn sparse_ring_validation_matches_full_snapshots_and_faults_on_gpu() -> Result<()> {
    let scenario = Scenario::baseline(97, 42, Experiment::Transport)?;
    let workgroup = 64;
    let mut simulation = Simulation::new(scenario.clone(), device(workgroup)?)?;
    let count = scenario.actors();
    let indices = simulation
        .device
        .tensor(&(count as u32 - 65..count as u32).collect::<Vec<_>>());
    let entries = simulation.device.tensor(&[0]);
    for case in [
        TransitCase::Unchanged,
        TransitCase::Mismatch,
        TransitCase::RingOverflow,
        TransitCase::TotalOverflow,
    ] {
        let mut state = scenario.initial()?;
        let ring = BASE_FIELDS + scenario.ring() * (Good::COUNT - 1);
        let expected = match case {
            TransitCase::Unchanged => FaultFlags::empty(),
            TransitCase::Mismatch => {
                state.values[ring * count + count - 1] = 7;
                FaultFlags::TRANSIT_MISMATCH
            }
            TransitCase::RingOverflow => {
                state.values[ring * count + count - 1] = u64::MAX;
                state.values[(ring + 1) * count + count - 1] = 1;
                state.values[(TRANSIT_FIELDS + Good::COUNT - 1) * count + count - 1] = 7;
                state.set(Field::Transit, count - 1, 7);
                FaultFlags::ARITHMETIC_OVERFLOW | FaultFlags::TRANSIT_MISMATCH
            }
            TransitCase::TotalOverflow => {
                state.values[BASE_FIELDS * count] = u64::MAX;
                state.values[(BASE_FIELDS + scenario.ring()) * count] = 1;
                state.values[TRANSIT_FIELDS * count] = u64::MAX;
                state.values[(TRANSIT_FIELDS + 1) * count] = 1;
                state.set(Field::Transit, 0, 0);
                FaultFlags::ARITHMETIC_OVERFLOW
            }
        };
        simulation.device.write(
            &simulation.state,
            bytemuck::cast_slice(&encode(&state.values)),
        );
        let sparse = simulation.observe_rings(RingValidation::Selected(&indices));
        let summary = simulation
            .device
            .read_range(&simulation.summary, 0, SummaryLayout::WORDS)?;
        let control = simulation.device.read(&simulation.control)?;
        let full = simulation.observe();
        assert_eq!(sparse.is_ok(), full.is_ok());
        assert_eq!(
            summary,
            simulation
                .device
                .read_range(&simulation.summary, 0, SummaryLayout::WORDS,)?
        );
        assert_eq!(control, simulation.device.read(&simulation.control)?);
        assert_eq!(FaultFlags::read(&control)?, expected);
        let combined = simulation.observe_rings(RingValidation::Changed {
            entries: &entries,
            settlements: &indices,
        });
        assert_eq!(combined.is_ok(), full.is_ok());
        assert_eq!(control, simulation.device.read(&simulation.control)?);
        assert_eq!(
            summary,
            simulation
                .device
                .read_range(&simulation.summary, 0, SummaryLayout::WORDS)?,
        );
        if let (Ok(sparse), Ok(full), Ok(combined)) = (sparse, full, combined) {
            assert_eq!(sparse, full);
            assert_eq!(combined, full);
            assert_eq!(simulation.observe_rings(RingValidation::Unchanged)?, full);
        }
    }
    Ok(())
}

#[test]
fn direct_registry_commit_revalidates_rings_after_a_previous_audit_on_gpu() -> Result<()> {
    let scenario = Scenario::baseline(97, 42, Experiment::Transport)?;
    let mut simulation = Simulation::new(scenario.clone(), device(64)?)?;
    let _ = simulation.observe()?;
    let mut state = scenario.initial()?;
    let firm = scenario.config().households as usize;
    state.set(Field::PendingCapacity, firm, 1);
    state.values[BASE_FIELDS * state.actors.len()] = 7;
    simulation.device.write(
        &simulation.state,
        bytemuck::cast_slice(&encode(&state.values)),
    );
    let mut requests = vec![0; state.actors.len()];
    requests[firm] = (6 << 24) | 1;
    simulation
        .device
        .write(&simulation.enterprise_requests, &requests);
    let pending =
        registry::records::PendingRecords::from_dense(&requests, &vec![0; requests.len()])?;
    let plan = registry::RegistryPlan::new(
        &scenario,
        &simulation.metadata,
        &simulation.migrations,
        &pending.records,
    )?
    .expect("capacity registry plan");
    assert!(simulation.commit_registry(plan, &pending.records).is_err());
    assert_eq!(
        FaultFlags::read(&simulation.device.read(&simulation.control)?)?,
        FaultFlags::TRANSIT_MISMATCH,
    );
    Ok(())
}

#[derive(Clone, Copy)]
enum RegistryCase {
    Capacity,
    Estate,
    Migration,
}

struct RegistryResult {
    state: crate::domain::State,
    snapshot: crate::domain::ObservationSnapshot,
}

#[test]
fn capacity_snapshot_reuse_matches_full_birth_and_migration_audits_on_gpu() -> Result<()> {
    let scenario = Scenario::baseline(129, 42, Experiment::Baseline)?;
    let count = scenario.actors();
    let firm = scenario.config().households as usize;
    for case in [
        RegistryCase::Capacity,
        RegistryCase::Estate,
        RegistryCase::Migration,
    ] {
        let mut initial = scenario.initial()?;
        let mut requests = vec![0; count];
        let mut births = vec![0; count];
        match case {
            RegistryCase::Capacity => {
                initial.set(Field::PendingCapacity, firm, 2);
                requests[firm] = (6 << 24) | 2;
                births[0] = 1;
            }
            RegistryCase::Estate => {
                initial.set(Field::EstateSettled, 0, 1);
                requests[0] = 5 << 24;
            }
            RegistryCase::Migration => requests[0] = (2 << 24) | (firm as u32 + 1),
        }
        let records = registry::records::PendingRecords::from_dense(&requests, &births)?;
        let mut outcomes = Vec::new();
        for audit in [
            crate::simulation::RegistryAudit::Validated,
            crate::simulation::RegistryAudit::Full,
        ] {
            let mut simulation = Simulation::new(scenario.clone(), device(64)?)?;
            simulation.device.write(
                &simulation.state,
                bytemuck::cast_slice(&encode(&initial.values)),
            );
            simulation
                .device
                .write(&simulation.enterprise_requests, &requests);
            simulation
                .device
                .write(simulation.demography.birth_buffer(), &births);
            simulation.snapshot = simulation.observe()?;
            let before = simulation.snapshot.economy.household_cash;
            let plan = registry::RegistryPlan::new(
                &scenario,
                &simulation.metadata,
                &simulation.migrations,
                &records.records,
            )?
            .expect("supported registry changes");
            simulation.commit_registry_audited(plan, &records.records, audit)?;
            match case {
                RegistryCase::Capacity | RegistryCase::Estate => {
                    assert_eq!(simulation.snapshot.economy.household_cash, before);
                }
                RegistryCase::Migration => {
                    assert_eq!(simulation.snapshot.economy.household_cash, before - 100);
                }
            }
            outcomes.push(RegistryResult {
                state: simulation.download()?,
                snapshot: simulation.snapshot,
            });
        }
        assert_eq!(outcomes[0].state, outcomes[1].state);
        assert_eq!(outcomes[0].snapshot, outcomes[1].snapshot);
    }
    Ok(())
}

#[test]
fn reused_capacity_snapshot_preserves_existing_gpu_faults_on_gpu() -> Result<()> {
    let scenario = Scenario::baseline(97, 42, Experiment::Baseline)?;
    let mut simulation = Simulation::new(scenario.clone(), device(64)?)?;
    let mut requests = vec![0; scenario.actors()];
    let firm = scenario.config().households as usize;
    requests[firm] = (6 << 24) | 1;
    simulation
        .device
        .write(&simulation.enterprise_requests, &requests);
    let records =
        registry::records::PendingRecords::from_dense(&requests, &vec![0; requests.len()])?;
    let plan = registry::RegistryPlan::new(
        &scenario,
        &simulation.metadata,
        &simulation.migrations,
        &records.records,
    )?
    .expect("capacity registry plan");
    let mut control = simulation.device.read(&simulation.control)?;
    control[crate::simulation::flags::FAULT_WORD] = FaultFlags::ARITHMETIC_OVERFLOW.bits();
    simulation.device.write(&simulation.control, &control);
    let actors = simulation.metadata.actors.clone();
    let error = simulation
        .commit_registry_audited(
            plan,
            &records.records,
            crate::simulation::RegistryAudit::Validated,
        )
        .expect_err("the existing GPU fault must propagate");
    assert!(error.to_string().contains("ARITHMETIC_OVERFLOW"));
    assert_eq!(simulation.metadata.actors, actors);
    Ok(())
}

use super::*;

const FUEL_ROTATION_WIDTH: usize = HOUSEHOLD_SEARCH_SAMPLES as usize;

#[derive(CubeType, CubeLaunch)]
struct FuelRotationInput {
    cases: Box<[u32]>,
    members: Box<[u32]>,
}

struct FuelRotationCheck;
struct CompareFuelRotation;

impl System for FuelRotationCheck {
    type Input = FuelRotationInput;
    type Passes = gpu_passes![CompareFuelRotation];
}

#[cube]
impl GpuPass<FuelRotationInput> for CompareFuelRotation {
    type Output = ();

    fn execute(mut input: FuelRotationInput) {
        let index = ABSOLUTE_POS;
        if index < input.cases.len() / 3 {
            let length = input.cases[index * 3];
            let day = input.cases[index * 3 + 1];
            let household = input.cases[index * 3 + 2];
            let rotation = consumption::FuelSearchRotation::new(length, day, household);
            let samples = min(length, HOUSEHOLD_SEARCH_SAMPLES);
            for sample in 0..samples {
                let offset = sample * (length / samples) + sample * (length % samples) / samples;
                let original = (offset + day + household) % length;
                input.members[index * (2 * FUEL_ROTATION_WIDTH) + sample as usize] = original;
                input.members
                    [index * (2 * FUEL_ROTATION_WIDTH) + FUEL_ROTATION_WIDTH + sample as usize] =
                    rotation.member(sample);
            }
        }
    }
}

#[test]
fn fuel_sample_rotation_preserves_non_power_supplier_counts_and_wrapping_on_gpu() -> Result<()> {
    let mut cases = Vec::new();
    for length in [
        0,
        1,
        2,
        3,
        7,
        8,
        9,
        15,
        17,
        31,
        193,
        257,
        65_537,
        u32::MAX / 2,
        u32::MAX / 2 + 1,
        u32::MAX - 31,
        u32::MAX,
    ] {
        for day in [
            0,
            1,
            u32::MAX / 2,
            u32::MAX - 255,
            u32::MAX - 15,
            u32::MAX - 7,
            u32::MAX - 1,
            u32::MAX,
        ] {
            for household in [0, 1, 7, 31, 193, u32::MAX / 2, u32::MAX - 1, u32::MAX] {
                cases.extend([length, day, household]);
            }
        }
    }
    for workgroup in [32, 256] {
        let device = device(workgroup)?;
        let count = cases.len() / 3;
        let cases_buffer = device.tensor(&cases);
        let members = device.tensor(&vec![u32::MAX; count * 2 * FUEL_ROTATION_WIDTH]);
        FuelRotationCheck::launch_elements(
            &device,
            count,
            FuelRotationInputLaunch {
                cases: cases_buffer.into_buffer_arg(),
                members: members.buffer_arg(),
            },
        );
        let members = device.read(&members)?;
        for (case, members) in cases
            .chunks_exact(3)
            .zip(members.chunks_exact(2 * FUEL_ROTATION_WIDTH))
        {
            assert_eq!(
                members[..FUEL_ROTATION_WIDTH],
                members[FUEL_ROTATION_WIDTH..],
                "{case:?}, WG{workgroup}"
            );
        }
    }
    Ok(())
}

#[test]
fn bounded_household_search_rotates_samples_and_repeats_on_gpu() -> Result<()> {
    for day in [0, 1] {
        let mut probe = SearchProbe::new(SupplyCase::Empty, day, Market::Household)?;
        for firm in 1..=SUPPLIERS as usize {
            probe.state.set(Field::Price, firm, 0);
        }
        probe.state.set(Field::Bread, 10, 100);
        probe.state.set(Field::Price, 10, 1);
        let expected = match day {
            0 => NO_ACTOR,
            _ => 10,
        };
        for execution in [
            SearchExecution::Serial,
            SearchExecution::Shared,
            SearchExecution::Plane,
            SearchExecution::ScalarHousehold,
        ] {
            let first = probe.outcome(execution)?;
            assert_eq!(first.proposal[ProposalLayout::ACTOR], expected);
            assert_eq!(first, probe.outcome(execution)?);
        }
    }
    Ok(())
}

#[test]
fn deferred_consumer_quantity_preserves_wide_budgets_and_empty_requests_on_gpu() -> Result<()> {
    for good in [Good::Coal, Good::Wood] {
        for case in [SupplyCase::HighCash, SupplyCase::Empty] {
            let mut probe = SearchProbe::new(case, 128, Market::Consumer(good))?;
            probe.state.set(Field::Cash, 0, u64::MAX / 1000);
            probe.state.set(Field::HeatingNeed, 0, u64::from(u32::MAX));
            for firm in 1..probe.state.actors.len() {
                let stock = match case {
                    SupplyCase::Empty => 0,
                    _ => u64::MAX / 1000,
                };
                probe.state.set(Field::inventory(good), firm, stock);
            }
            let expected = probe.outcome(SearchExecution::Serial)?;
            assert_ne!(expected.proposal[ProposalLayout::ACTOR], NO_ACTOR);
            assert!(expected.proposal[ProposalLayout::QUANTITY_HIGH] > 0);
            for execution in [
                SearchExecution::Shared,
                SearchExecution::Plane,
                SearchExecution::ScalarFuel,
                SearchExecution::SpatialFuel,
            ] {
                assert_eq!(
                    probe.outcome(execution)?,
                    expected,
                    "{good:?}, {case:?}, {execution:?}",
                );
            }
        }
    }
    Ok(())
}

#[test]
fn scalar_fuel_search_preserves_partial_dispatch_buffers_and_rotated_suppliers_on_gpu() -> Result<()>
{
    for good in [Good::Coal, Good::Wood] {
        for case in [
            SupplyCase::Capped,
            SupplyCase::Sparse,
            SupplyCase::Empty,
            SupplyCase::Unaffordable,
        ] {
            let mut probe = SearchProbe::new(case, 128, Market::Consumer(good))?;
            let expected = probe.outcome(SearchExecution::Shared)?;
            let expected_proposals = probe
                .simulation
                .device
                .read(&probe.simulation.work.proposals)?;
            let expected_successes = probe
                .simulation
                .device
                .read(&probe.simulation.searches.successes)?;
            for workgroup in [32, 64, 128, 256] {
                probe.simulation.device.workgroup = workgroup;
                assert_eq!(
                    probe.outcome_with_workgroup(SearchExecution::ScalarFuel, workgroup)?,
                    expected,
                    "{good:?}, {case:?}, workgroup {workgroup}",
                );
                assert_eq!(
                    probe
                        .simulation
                        .device
                        .read(&probe.simulation.work.proposals)?,
                    expected_proposals,
                    "{good:?}, {case:?}, workgroup {workgroup}, proposals",
                );
                assert_eq!(
                    probe
                        .simulation
                        .device
                        .read(&probe.simulation.searches.successes)?,
                    expected_successes,
                    "{good:?}, {case:?}, workgroup {workgroup}, successes",
                );
            }
        }
        for case in [SupplyCase::Sparse, SupplyCase::Empty] {
            for day in [31, 1024, 2176] {
                let probe = SearchProbe::with_suppliers(case, day, Market::Consumer(good), 2177)?;
                let expected = probe.outcome(SearchExecution::Serial)?;
                for execution in [
                    SearchExecution::Shared,
                    SearchExecution::Plane,
                    SearchExecution::ScalarFuel,
                    SearchExecution::SpatialFuel,
                ] {
                    assert_eq!(
                        probe.outcome(execution)?,
                        expected,
                        "{good:?}, {case:?}, day {day}, {execution:?}",
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
fn filtered_empty_consumer_orders_preserve_state_feedback_and_diagnostics_on_gpu() -> Result<()> {
    #[derive(Clone, Copy, Debug)]
    enum Orders {
        Empty,
        Mixed,
        Wide,
        BuyerOverflow,
        SellerOverflow,
    }

    let households = 137usize;
    let scenario = Scenario::industrial(households as u32, 42, Experiment::Baseline)?;
    let device = device(64)?;
    for orders in [
        Orders::Empty,
        Orders::Mixed,
        Orders::Wide,
        Orders::BuyerOverflow,
        Orders::SellerOverflow,
    ] {
        let good = match orders {
            Orders::Wide => Good::Coal,
            _ => Good::Linen,
        };
        let mut state = scenario.initial()?;
        for actor in 0..households {
            state.set(Field::LivingMembers, actor, 3);
            state.set(Field::FoodNeed, actor, 3);
            state.set(Field::NutritionNeed, actor, 3);
            state.set(Field::FoodPrice, actor, 1);
            state.set(Field::Bread, actor, 100_000);
            state.set(Field::inventory(good), actor, 0);
            state.set(Field::Wood, actor, 0);
            state.set(
                Field::Cash,
                actor,
                match orders {
                    Orders::Wide => u64::MAX,
                    _ => 1_000_000,
                },
            );
            state.set(
                Field::HeatingNeed,
                actor,
                match orders {
                    Orders::Wide => u64::from(u32::MAX),
                    _ => 9,
                },
            );
            state.set(
                Field::Purchased,
                actor,
                match orders {
                    Orders::BuyerOverflow => u64::MAX,
                    _ => 0,
                },
            );
        }
        let peers = state
            .actors
            .firm_indices()
            .filter(|&peer| state.actors.at(peer).kind == good as u32)
            .collect::<Vec<_>>();
        assert!(!peers.is_empty());
        for peer in households..state.actors.len() {
            state.set(Field::inventory(good), peer, 0);
            state.set(Field::Wood, peer, 0);
            state.set(Field::Price, peer, 1);
            state.set(Field::Sales, peer, 0);
            state.set(
                Field::Cash,
                peer,
                match orders {
                    Orders::SellerOverflow => u64::MAX,
                    _ => 1000,
                },
            );
        }
        if let Orders::Mixed = orders {
            state.set(Field::inventory(good), peers[0], 3);
        }
        let initialize = || -> Result<Simulation> {
            let simulation = Simulation::new(scenario.clone(), device.clone())?;
            simulation.device.write(
                &simulation.state,
                bytemuck::cast_slice(&encode(&state.values)),
            );
            Ok(simulation)
        };
        let mut full = initialize()?;
        let mut filtered = initialize()?;
        let mut removed = 0u32;
        for round in 0..4 {
            full.matching_round = round;
            filtered.matching_round = round;
            let mut full_evidence = crate::domain::PhaseEvidence::default();
            let mut filtered_evidence = crate::domain::PhaseEvidence::default();
            let reference = full
                .device_market(
                    Market::Consumer(good),
                    Phase::ProvisionPurchase,
                    Trace::Record,
                    &mut full_evidence,
                )
                .and_then(|_| full.fault_check());
            let actual = filtered
                .device_market(
                    Market::Consumer(good),
                    Phase::ProvisionPurchase,
                    Trace::Discard,
                    &mut filtered_evidence,
                )
                .and_then(|_| filtered.fault_check());
            assert_eq!(
                actual.is_err(),
                reference.is_err(),
                "{orders:?}, round {round}"
            );
            let expected = device.read_range(&full.control, 0, 4)?;
            let observed = device.read_range(&filtered.control, 0, 4)?;
            let faults = match orders {
                Orders::BuyerOverflow | Orders::SellerOverflow => {
                    FaultFlags::ARITHMETIC_OVERFLOW.bits()
                }
                _ => 0,
            };
            assert_eq!(
                expected[crate::simulation::flags::FAULT_WORD],
                faults,
                "{orders:?}, round {round}"
            );
            assert_eq!(reference.is_err(), faults != 0, "{orders:?}, round {round}");
            assert_eq!(&observed[..3], &expected[..3], "{orders:?}, round {round}");
            assert_eq!(
                device.read(&filtered.state)?,
                device.read(&full.state)?,
                "{orders:?}, round {round}"
            );
            let expected_feedback = device.read(&full.work.feedback)?;
            let actual_feedback = device.read(&filtered.work.feedback)?;
            assert_eq!(
                actual_feedback[crate::simulation::flags::FAULT_WORD],
                expected_feedback[crate::simulation::flags::FAULT_WORD]
            );
            assert_eq!(
                &actual_feedback[CONTROL_WORDS..],
                &expected_feedback[CONTROL_WORDS..],
                "{orders:?}, round {round}"
            );
            removed += expected[CONSTRUCTED_EVENTS_WORD] - observed[CONSTRUCTED_EVENTS_WORD];
            if let (Orders::Wide, 3) = (orders, round) {
                assert_eq!(
                    device.read_range(&filtered.control, orders::WIDE_ORDERS_WORD, 1)?,
                    [1]
                );
                assert_eq!(
                    observed[CONSTRUCTED_EVENTS_WORD],
                    expected[CONSTRUCTED_EVENTS_WORD]
                );
            }
        }
        if let Orders::Empty | Orders::Mixed = orders {
            assert!(removed > 0, "{orders:?}");
        }
    }
    Ok(())
}

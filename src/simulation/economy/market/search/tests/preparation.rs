use super::*;

#[derive(Debug, PartialEq, Eq)]
struct PreparedRequests {
    actors: Vec<u32>,
    demand: Vec<u32>,
    candidates: Vec<u32>,
    statuses: Vec<u32>,
    faults: u32,
}

#[test]
fn inactive_household_input_preparation_preserves_inventory_faults_on_gpu() -> Result<()> {
    for good in Good::ALL {
        let mut probe = SearchProbe::new(SupplyCase::Capped, 0, Market::Input(Good::Linen))?;
        probe.parameters.good = good as u32;
        probe.state.set(Field::LivingMembers, 0, 0);
        for inventory in [0, u64::MAX] {
            probe.state.set(Field::inventory(good), 0, inventory);
            let transit = TRANSIT_FIELDS + good as usize - 1;
            probe.state.values[transit * probe.state.actors.len()] = 1;
            let expected = probe.outcome(SearchExecution::Serial)?;
            assert_eq!(expected.faults == 0, inventory == 0, "{good:?}");
            assert_eq!(
                probe.outcome(SearchExecution::Prepare)?,
                expected,
                "{good:?}, inventory {inventory}"
            );
        }
    }
    Ok(())
}

#[test]
fn consumer_preparation_clears_inactive_requests_and_preserves_captured_retries_on_gpu()
-> Result<()> {
    let scenario = Scenario::industrial(129, 42, Experiment::Baseline)?;
    let device = device(64)?;
    let simulation = Simulation::new(scenario.clone(), device.clone())?;
    let count = scenario.actors();
    for good in Good::ALL
        .into_iter()
        .filter(|good| good.consumption_interval() > 1 || matches!(good, Good::Coal | Good::Wood))
    {
        let mut state = scenario.initial()?;
        let substitute = match good {
            Good::Coal => Good::Wood,
            _ => Good::Coal,
        };
        let firms = state.actors.firm_indices().collect::<Vec<_>>();
        for actor in firms {
            state.set(Field::inventory(substitute), actor, 0);
        }
        let households = (0..count)
            .filter(|&actor| state.actors.at(actor).kind == 0)
            .collect::<Vec<_>>();
        for &actor in &households {
            state.set(Field::LivingMembers, actor, 3);
            state.set(Field::HeatingNeed, actor, 9);
            state.set(Field::Bread, actor, 100_000);
            state.set(Field::Cash, actor, 1_000_000);
            state.set(Field::Coal, actor, 0);
            state.set(Field::Wood, actor, 0);
            state.set(Field::Clothes, actor, 1);
            state.set(Field::Furniture, actor, 1);
            state.set(Field::inventory(good), actor, 0);
        }
        let mut remaining = state.clone();
        for actor in households.into_iter().filter(|actor| actor % 3 == 0) {
            remaining.set(Field::inventory(good), actor, 1_000_000);
        }
        let mut expected = Vec::new();
        for (discovery, preparation) in [
            (Discovery::Linear, orders::PreparationWorkgroup::Native),
            (Discovery::Indexed, orders::PreparationWorkgroup::Native),
            (Discovery::Indexed, orders::PreparationWorkgroup::Wide),
        ] {
            let mut parameters = simulation.parameters(
                Phase::ProvisionPurchase,
                Market::Consumer(good),
                0,
                count,
            )?;
            parameters.discovery = discovery;
            device.write(
                &simulation.state,
                bytemuck::cast_slice(&encode(&state.values)),
            );
            device.write(&simulation.control, &vec![0; CONTROL_WORDS + 4]);
            device.write(&simulation.searches.requests, &vec![37; 6 * count]);
            device.write(
                &simulation.work.scratch,
                &vec![u32::MAX; orders::scratch_words(count)],
            );
            let mut observations = Vec::new();
            for stage in [SearchStage::PrepareCandidates, SearchStage::PrepareSelected] {
                let workgroup = preparation.width(&device);
                let dispatch = match stage {
                    SearchStage::PrepareSelected => {
                        orders::prepare_dispatch_for(
                            &device,
                            &simulation.control,
                            &simulation.searches.dispatches,
                            count,
                            preparation,
                        );
                        orders::dispatch(
                            &simulation.searches.dispatches,
                            orders::PREPARATION_DISPATCH,
                            count.div_ceil(workgroup as usize),
                        )
                    }
                    _ => cubecl::calculate_cube_count_elemwise(
                        &device.client,
                        count,
                        CubeDim::new_1d(workgroup),
                    )
                    .into(),
                };
                let resets = orders::search_counts_offset(count) + orders::search_blocks(count);
                device.write(&simulation.searches.successes, &vec![37; resets]);
                MarketSearch::launch(
                    &device,
                    dispatch,
                    CubeDim::new_1d(workgroup),
                    SearchInputLaunch {
                        state: simulation.state.buffer_arg(),
                        actors: simulation.actors.buffer_arg(),
                        config: simulation.config.launch(),
                        spatial: simulation.spatial.buffer_arg(),
                        proposals: simulation.work.proposals.buffer_arg(),
                        searches: simulation.searches.requests.buffer_arg(),
                        successes: simulation.searches.successes.buffer_arg(),
                        scratch: simulation.work.scratch.buffer_arg(),
                        control: simulation.control.buffer_arg(),
                        parameters: parameters.launch(),
                        status_reset: StatusReset::Full,
                        program: stage.program(&simulation.config),
                    },
                );
                assert_eq!(
                    device.read_range(&simulation.searches.successes, 0, resets)?,
                    vec![0; resets],
                );
                let mut control = device.read(&simulation.control)?;
                let requests = device.read(&simulation.searches.requests)?;
                let mut actors = requests[..control[SEARCHES_WORD] as usize].to_vec();
                let mut candidates = requests
                    [5 * count..5 * count + control[orders::PREPARATION_ACTORS_WORD] as usize]
                    .to_vec();
                actors.sort_unstable();
                candidates.sort_unstable();
                for actor in 0..count {
                    match actors.binary_search(&(actor as u32)) {
                        Ok(_) => assert!(requests[count + actor] > 0),
                        Err(_) => {
                            assert_eq!(requests[count + actor], 0);
                            assert_eq!(requests[2 * count + actor], 0);
                        }
                    }
                }
                observations.push(PreparedRequests {
                    actors,
                    demand: requests[count..5 * count].to_vec(),
                    candidates,
                    statuses: device.read_range(&simulation.work.scratch, 4 * count, count)?,
                    faults: control[crate::simulation::flags::FAULT_WORD],
                });
                control[SEARCHES_WORD] = 0;
                device.write(&simulation.control, &control);
                device.write(
                    &simulation.state,
                    bytemuck::cast_slice(&encode(&remaining.values)),
                );
            }
            match discovery {
                Discovery::Linear => expected = observations,
                _ => assert_eq!(observations, expected, "{good:?}"),
            }
        }
        assert!(!expected[0].actors.is_empty());
        assert!(expected[1].actors.len() < expected[0].actors.len());
        assert_eq!(expected[1].candidates, expected[0].candidates);
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
enum PreparationEligibility {
    InactiveSeller,
    GrainCashless,
    GrainStocked,
    Investment,
    Building,
    Distributor,
}

#[test]
fn sparse_input_preparation_preserves_stable_eligibility_and_all_round_requests_on_gpu()
-> Result<()> {
    let workgroup = 64;
    for case in [
        PreparationEligibility::InactiveSeller,
        PreparationEligibility::GrainCashless,
        PreparationEligibility::GrainStocked,
        PreparationEligibility::Investment,
        PreparationEligibility::Building,
        PreparationEligibility::Distributor,
    ] {
        let mut dense =
            SearchProbe::with_suppliers(SupplyCase::Capped, 0, Market::Input(Good::Linen), 65)?;
        let mut sparse =
            SearchProbe::with_suppliers(SupplyCase::Capped, 0, Market::Input(Good::Linen), 65)?;
        let good = match case {
            PreparationEligibility::InactiveSeller => Good::Bread,
            PreparationEligibility::GrainCashless | PreparationEligibility::GrainStocked => {
                Good::Grain
            }
            PreparationEligibility::Investment => Good::Tools,
            PreparationEligibility::Building => Good::Construction,
            PreparationEligibility::Distributor => Good::Linen,
        };
        for probe in [&mut dense, &mut sparse] {
            probe.simulation.device.workgroup = workgroup;
            probe.parameters.good = good as u32;
            probe.state.set(Field::Cash, 0, 0);
            match case {
                PreparationEligibility::InactiveSeller => {
                    probe.state.set(Field::LivingMembers, 0, 0);
                    probe.state.set(Field::Bread, 0, 100);
                }
                PreparationEligibility::GrainCashless | PreparationEligibility::GrainStocked => {
                    probe.state.set(Field::LandPlots, 0, 1)
                }
                PreparationEligibility::Investment => probe.state.set(Field::InvestmentGood, 0, 1),
                PreparationEligibility::Building => {
                    probe.state.set(
                        Field::ProjectStatus,
                        0,
                        crate::domain::property::ProjectStatus::Building as u64,
                    );
                    probe.state.set(Field::ConstructionRemaining, 0, 7);
                }
                PreparationEligibility::Distributor => {
                    probe.state.set(Field::EnterpriseModel, 0, 1)
                }
            }
            if let PreparationEligibility::GrainStocked = case {
                probe.state.set(Field::Grain, 0, 10_000);
                probe.state.set(Field::Cash, 0, 5_000);
            }
            probe.clear_preparation();
        }
        let expected =
            dense.prepared_inputs(EconomyStage::InputMarketStart, SearchStage::Prepare)?;
        assert_eq!(
            sparse.prepared_inputs(
                EconomyStage::InputMarketStart,
                SearchStage::PrepareCandidates
            )?,
            expected,
            "{case:?}, workgroup {workgroup}, first"
        );
        let count = sparse.state.actors.len();
        let control = sparse.simulation.device.read(&sparse.simulation.control)?;
        let requests = sparse
            .simulation
            .device
            .read(&sparse.simulation.searches.requests)?;
        let mut eligible = requests
            [5 * count..5 * count + control[orders::PREPARATION_ACTORS_WORD] as usize]
            .to_vec();
        eligible.sort_unstable();
        let first = match case {
            PreparationEligibility::InactiveSeller => 1,
            _ => 0,
        };
        assert_eq!(eligible, (first..count as u32).collect::<Vec<_>>());
        for round in 1..4 {
            for probe in [&mut dense, &mut sparse] {
                probe.state.set(Field::Cash, 0, 5_000);
                probe.state.set(Field::inventory(good), 0, 0);
            }
            let expected =
                dense.prepared_inputs(EconomyStage::InputMarketRepeat, SearchStage::Prepare)?;
            assert_eq!(
                sparse.prepared_inputs(
                    EconomyStage::InputMarketRepeat,
                    SearchStage::PrepareSelected
                )?,
                expected,
                "{case:?}, workgroup {workgroup}, round {round}"
            );
        }
        if !matches!(case, PreparationEligibility::InactiveSeller) {
            for probe in [&mut dense, &mut sparse] {
                probe.state.set(Field::inventory(good), 0, u64::MAX);
                probe.state.values[(TRANSIT_FIELDS + good as usize - 1) * count] = 1;
            }
            let expected =
                dense.prepared_inputs(EconomyStage::InputMarketRepeat, SearchStage::Prepare)?;
            assert_ne!(expected.faults, 0);
            assert_eq!(
                sparse.prepared_inputs(
                    EconomyStage::InputMarketRepeat,
                    SearchStage::PrepareSelected
                )?,
                expected,
                "{case:?}, workgroup {workgroup}, repeated demand overflow"
            );
        }
    }
    Ok(())
}

#[test]
fn grain_capture_skips_stable_household_nonbuyers_and_keeps_firm_retries_on_gpu() -> Result<()> {
    #[derive(Clone, Copy, Debug)]
    enum HouseholdDemand {
        FullStock,
        Cashless,
        Searching,
    }

    for workgroup in [64, 256] {
        for demand in [
            HouseholdDemand::FullStock,
            HouseholdDemand::Cashless,
            HouseholdDemand::Searching,
        ] {
            let mut dense =
                SearchProbe::with_suppliers(SupplyCase::Capped, 0, Market::Input(Good::Grain), 65)?;
            let mut sparse =
                SearchProbe::with_suppliers(SupplyCase::Capped, 0, Market::Input(Good::Grain), 65)?;
            for probe in [&mut dense, &mut sparse] {
                probe.simulation.device.workgroup = workgroup;
                probe.simulation.config.firm_rent = 0;
                probe.state.set(Field::LandPlots, 0, 1);
                probe.state.set(
                    Field::Grain,
                    0,
                    match demand {
                        HouseholdDemand::FullStock => 10_000,
                        _ => 0,
                    },
                );
                probe.state.set(
                    Field::Cash,
                    0,
                    match demand {
                        HouseholdDemand::Cashless => 0,
                        _ => 5000,
                    },
                );
                for (field, value) in [
                    (Field::EnterpriseModel, 1),
                    (Field::Demand, 10),
                    (Field::StockCoverage, 3),
                    (Field::Cash, 0),
                    (Field::Grain, 0),
                    (Field::Wage, 0),
                    (Field::LaborMinutes, 0),
                ] {
                    probe.state.set(field, 1, value);
                }
                probe.clear_preparation();
            }
            let expected =
                dense.prepared_inputs(EconomyStage::InputMarketStart, SearchStage::Prepare)?;
            assert_eq!(
                sparse.prepared_inputs(
                    EconomyStage::InputMarketStart,
                    SearchStage::PrepareCandidates,
                )?,
                expected,
            );
            let count = sparse.state.actors.len();
            let control = sparse.simulation.device.read(&sparse.simulation.control)?;
            let requests = sparse
                .simulation
                .device
                .read(&sparse.simulation.searches.requests)?;
            let mut captured = requests
                [5 * count..5 * count + control[orders::PREPARATION_ACTORS_WORD] as usize]
                .to_vec();
            captured.sort_unstable();
            let first = match demand {
                HouseholdDemand::Searching => 0,
                _ => 1,
            };
            assert_eq!(captured, (first..count as u32).collect::<Vec<_>>());
            for round in 1..4 {
                for probe in [&mut dense, &mut sparse] {
                    probe.state.set(Field::Cash, 1, 5000);
                    if let HouseholdDemand::Searching = demand {
                        let transit =
                            3 * u64::from(
                                probe.simulation.config.coverage + probe.simulation.config.ring,
                            ) * round
                                / 3;
                        probe.state.values[(TRANSIT_FIELDS + Good::Grain as usize - 1) * count] =
                            transit;
                        probe.state.set(Field::Transit, 0, transit);
                        probe.state.set(Field::Cash, 0, 5000 - 10 * transit);
                    }
                }
                let expected =
                    dense.prepared_inputs(EconomyStage::InputMarketRepeat, SearchStage::Prepare)?;
                assert!(expected.queue.contains(&1));
                assert_eq!(
                    sparse.prepared_inputs(
                        EconomyStage::InputMarketRepeat,
                        SearchStage::PrepareSelected,
                    )?,
                    expected,
                    "{demand:?}, WG{workgroup}, round {round}",
                );
            }
        }
    }
    Ok(())
}

#[test]
fn sparse_input_preparation_preserves_first_round_inventory_faults_on_gpu() -> Result<()> {
    let workgroup = 64;
    let mut probe =
        SearchProbe::with_suppliers(SupplyCase::Capped, 0, Market::Input(Good::Linen), 65)?;
    probe.simulation.device.workgroup = workgroup;
    probe.state.set(Field::LivingMembers, 0, 0);
    probe.state.set(Field::Linen, 0, u64::MAX);
    let count = probe.state.actors.len();
    probe.state.values[(TRANSIT_FIELDS + Good::Linen as usize - 1) * count] = 1;
    probe.clear_preparation();
    let first = probe.prepared_inputs(
        EconomyStage::InputMarketStart,
        SearchStage::PrepareCandidates,
    )?;
    assert_ne!(first.faults, 0);
    let sentinel = 0x1234_5678;
    probe.simulation.device.write(
        &probe.simulation.work.scratch,
        &vec![sentinel; probe.simulation.work.scratch.meta.num_elements()],
    );
    let repeated = probe.prepared_inputs(
        EconomyStage::InputMarketRepeat,
        SearchStage::PrepareSelected,
    )?;
    assert_eq!(repeated.faults, first.faults);
    assert_eq!(repeated.statuses, vec![sentinel; count]);
    assert!(repeated.queue.is_empty());
    Ok(())
}

#[test]
fn selected_fuel_retries_include_buyers_with_only_freezing_reserves_on_gpu() -> Result<()> {
    for good in [Good::Coal, Good::Wood] {
        let mut dense =
            SearchProbe::with_suppliers(SupplyCase::Capped, 0, Market::Consumer(good), 65)?;
        let mut selected =
            SearchProbe::with_suppliers(SupplyCase::Capped, 0, Market::Consumer(good), 65)?;
        let units = match good {
            Good::Coal => 2,
            _ => 1,
        };
        let horizon =
            u64::from(dense.simulation.config.coverage + dense.simulation.config.ring - 1);
        let freezing = 2 * horizon / units;
        let comfort = 4 * horizon / units;
        for probe in [&mut dense, &mut selected] {
            probe.simulation.config.food_income = crate::domain::food::FoodIncome::PurchasingPower;
            probe.simulation.config.heating_per_person = 4;
            probe.state.set(Field::HeatingNeed, 0, 4);
            probe.state.set(Field::LivingMembers, 0, 1);
            probe.state.set(Field::FoodNeed, 0, 1);
            probe.state.set(Field::Bread, 0, 100);
            probe.state.set(Field::Cash, 0, 5_000);
            probe.state.set(Field::Coal, 0, 0);
            probe.state.set(Field::Wood, 0, 0);
            probe.state.set(Field::inventory(good), 0, freezing / 2);
            let count = probe.state.actors.len();
            let delivery = freezing - freezing / 2;
            probe.state.values[(TRANSIT_FIELDS + good as usize - 1) * count] = delivery;
            probe.state.set(Field::Transit, 0, delivery);
            probe.clear_preparation();
        }
        let expected =
            dense.prepared_inputs(EconomyStage::InputMarketStart, SearchStage::Prepare)?;
        assert!(expected.queue.is_empty());
        assert_eq!(expected.faults, 0);
        assert_eq!(
            selected.prepared_inputs(
                EconomyStage::InputMarketStart,
                SearchStage::PrepareCandidates,
            )?,
            expected,
            "{good:?} initial freezing demand",
        );
        let count = selected.state.actors.len();
        let control = selected
            .simulation
            .device
            .read(&selected.simulation.control)?;
        assert_eq!(control[orders::PREPARATION_ACTORS_WORD], 1);
        assert_eq!(
            selected.simulation.device.read_range(
                &selected.simulation.searches.requests,
                5 * count,
                1,
            )?,
            [0],
        );
        for round in 1..MATCHING_ROUNDS {
            for probe in [&mut dense, &mut selected] {
                probe.parameters.matching_round = round as u32;
            }
            let expected =
                dense.prepared_inputs(EconomyStage::InputMarketRepeat, SearchStage::Prepare)?;
            let actual = selected.prepared_inputs(
                EconomyStage::InputMarketRepeat,
                SearchStage::PrepareSelected,
            )?;
            assert_eq!(actual, expected, "{good:?} round {round}");
            match round + 1 == MATCHING_ROUNDS {
                true => {
                    assert_eq!(actual.queue, [0]);
                    let wanted =
                        u64::from(actual.payload[0]) | (u64::from(actual.payload[count]) << 32);
                    assert_eq!(wanted, comfort - freezing);
                }
                false => assert!(actual.queue.is_empty()),
            }
        }
    }
    Ok(())
}

#[test]
fn selected_fuel_buyers_match_dense_requests_and_reset_trace_statuses_on_gpu() -> Result<()> {
    let households = 97usize;
    let scenario = Scenario::baseline(households as u32, 42, Experiment::Concentrated)?;
    for good in [Good::Coal, Good::Wood] {
        let units = match good {
            Good::Coal => 2,
            _ => 1,
        };
        let target = 3 * (u64::from(scenario.config().coverage) + scenario.ring() as u64 - 1);
        let reserve = target.div_ceil(units);
        let probe = || -> Result<SearchProbe> {
            let simulation = Simulation::new(scenario.clone(), device(64)?)?;
            let mut state = scenario.initial()?;
            for household in 0..households {
                state.set(Field::HeatingNeed, household, 3);
                state.set(Field::FoodNeed, household, 3);
                state.set(Field::LivingMembers, household, 3);
                state.set(
                    Field::Cash,
                    household,
                    match household % 4 {
                        1 => 0,
                        _ => 5_000,
                    },
                );
                state.set(
                    Field::Bread,
                    household,
                    match household % 4 {
                        3 => 0,
                        _ => 100,
                    },
                );
                state.set(
                    Field::inventory(good),
                    household,
                    match household % 4 {
                        2 => reserve,
                        _ => 0,
                    },
                );
            }
            for firm in households..state.actors.len() {
                state.set(Field::inventory(good), firm, 1_000);
            }
            let parameters = simulation.parameters(
                Phase::ProvisionPurchase,
                Market::Consumer(good),
                0,
                state.actors.len(),
            )?;
            let probe = SearchProbe {
                simulation,
                state,
                parameters,
            };
            probe.clear_preparation();
            Ok(probe)
        };
        let mut dense = probe()?;
        let mut selected = probe()?;
        let expected =
            dense.prepared_inputs(EconomyStage::InputMarketStart, SearchStage::Prepare)?;
        assert_eq!(
            selected.prepared_inputs(
                EconomyStage::InputMarketStart,
                SearchStage::PrepareCandidates,
            )?,
            expected,
        );
        let count = selected.state.actors.len();
        let control = selected
            .simulation
            .device
            .read(&selected.simulation.control)?;
        let requests = selected
            .simulation
            .device
            .read(&selected.simulation.searches.requests)?;
        let mut captured = requests
            [5 * count..5 * count + control[orders::PREPARATION_ACTORS_WORD] as usize]
            .to_vec();
        captured.sort_unstable();
        assert_eq!(
            captured,
            (0..households as u32).step_by(4).collect::<Vec<_>>()
        );
        for round in 1..4u64 {
            for probe in [&mut dense, &mut selected] {
                for household in (0..households).step_by(4) {
                    probe.state.set(
                        Field::Cash,
                        household,
                        match round {
                            3 => 0,
                            _ => 5_000 - round * 1_000,
                        },
                    );
                    let delivered = match round {
                        1 => reserve / 2,
                        _ => reserve,
                    };
                    probe.state.values[(TRANSIT_FIELDS + good as usize - 1) * count + household] =
                        delivered;
                    probe.state.set(Field::Transit, household, delivered);
                }
                for firm in households..count {
                    probe.state.set(Field::Cash, firm, 100_000 + round * 100);
                    probe.state.set(Field::Sales, firm, round * 10);
                    probe
                        .state
                        .set(Field::inventory(good), firm, 1_000 - round * 10);
                }
                let simulation = &probe.simulation;
                simulation.device.write(
                    &simulation.work.scratch,
                    &vec![u32::MAX; simulation.work.scratch.meta.num_elements()],
                );
                simulation.device.write(
                    &simulation.searches.successes,
                    &vec![u32::MAX; simulation.searches.successes.meta.num_elements()],
                );
            }
            let expected =
                dense.prepared_inputs(EconomyStage::InputMarketRepeat, SearchStage::Prepare)?;
            assert_eq!(
                selected.prepared_inputs(
                    EconomyStage::InputMarketRepeat,
                    SearchStage::PrepareSelected,
                )?,
                expected,
                "{good:?} round {round}",
            );
            let resets = count.div_ceil(32) + orders::search_blocks(count);
            assert_eq!(
                selected.simulation.device.read_range(
                    &selected.simulation.searches.successes,
                    0,
                    resets,
                )?,
                vec![0; resets],
            );
        }
    }
    Ok(())
}

use super::*;

#[cube(launch)]
fn select_supplier_stride(lengths: &[u32], strides: &mut [u32]) {
    let length = lengths[CUBE_POS];
    let offset = UNIT_POS % max(length / 2, 1);
    strides[ABSOLUTE_POS] = coprime_stride(length, offset);
}

#[test]
fn supplier_strides_match_the_first_coprime_candidate_on_gpu() -> Result<()> {
    let workgroup = 128;
    let device = device(workgroup)?;
    let lengths = (2..=256)
        .chain([1024, 104_729, 1_000_000, 16_777_216, 1_000_000_000])
        .collect::<Vec<u32>>();
    let strides = device.empty(lengths.len() * workgroup as usize);
    select_supplier_stride::launch(
        &device.client,
        CubeCount::Static(lengths.len() as u32, 1, 1),
        CubeDim::new_1d(workgroup),
        device.tensor(&lengths).into_buffer_arg(),
        strides.buffer_arg(),
    );
    let actual = device.read(&strides)?;
    for (length, strides) in lengths
        .into_iter()
        .zip(actual.chunks_exact(workgroup as usize))
    {
        for (lane, &stride) in strides.iter().enumerate() {
            let offset = lane as u32 % (length / 2);
            let expected = (length / 2 + offset..=length).find(|&candidate| {
                let mut divisor = candidate;
                let mut remainder = length;
                while remainder != 0 {
                    (divisor, remainder) = (remainder, divisor % remainder);
                }
                divisor == 1
            });
            assert_eq!(Some(stride), expected, "length {length}, offset {offset}");
        }
    }
    Ok(())
}

#[test]
fn grain_firm_index_preserves_rotated_full_partial_and_empty_offers_on_gpu() -> Result<()> {
    for case in [
        SupplyCase::Capped,
        SupplyCase::Sparse,
        SupplyCase::Empty,
        SupplyCase::Unaffordable,
        SupplyCase::RetailSold,
    ] {
        let mut probe = SearchProbe::with_suppliers(case, 0, Market::Input(Good::Grain), 257)?;
        probe.state.set(Field::FoodNeed, 1, 10);
        probe.state.set(Field::InputTicks, 1, 1000);
        for rotation in [0, 13, 128, 256] {
            probe.parameters.settlement_rotation = rotation;
            for buyer in [0, 1] {
                let result = |mode| probe.actor_outcome(mode, 32, CacheRefresh::Reset, buyer);
                assert_eq!(
                    result(SearchExecution::Grain)?,
                    result(SearchExecution::Plane)?,
                    "{case:?}, buyer {buyer}, rotation {rotation}"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn sparse_grain_dealers_preserve_rotated_quota_partial_and_empty_offers_on_gpu() -> Result<()> {
    let mut probe =
        SearchProbe::with_suppliers(SupplyCase::Capped, 0, Market::Input(Good::Grain), 8193)?;
    let simulation = &probe.simulation;
    let mut spatial = simulation.device.read(&simulation.spatial)?;
    spatial[2] = 0;
    simulation.device.write(&simulation.spatial, &spatial);
    probe.state.set(Field::GrainQuote, 0, 4);
    probe.state.set(Field::EnterpriseModel, 1, 1);
    probe.state.set(Field::Cash, 1, 16);
    probe.state.set(Field::FoodNeed, 1, 20);
    for (stock, target) in [(20, 1), (2, 1000), (0, 1000)] {
        probe.state.set(Field::InputTicks, 1, target);
        for seller in 2..probe.state.actors.len() {
            probe.state.set(Field::Grain, seller, stock);
        }
        for rotation in [0, 510] {
            probe.parameters.settlement_rotation = rotation;
            let result = |mode| probe.actor_outcome(mode, 32, CacheRefresh::Reset, 1);
            assert_eq!(
                result(SearchExecution::Grain)?,
                result(SearchExecution::Plane)?,
                "stock {stock}, input {target}, rotation {rotation}"
            );
        }
    }
    Ok(())
}

#[test]
fn funded_stock_search_preserves_wide_price_division_and_overflow_fallback_on_gpu() -> Result<()> {
    for price in [5_001, u32::MAX - 1] {
        let mut probe = SearchProbe::new(SupplyCase::HighCash, 0, Market::Input(Good::Bread))?;
        probe.state.set(Field::Cash, 0, u64::MAX);
        probe.state.set(Field::FoodNeed, 0, u64::from(u32::MAX));
        for supplier in 1..=SUPPLIERS as usize {
            probe.state.set(Field::Bread, supplier, u64::MAX);
            probe.state.set(Field::Price, supplier, u64::from(price));
        }
        let coverage = probe.simulation.config.coverage + probe.simulation.config.ring - 1;
        let expected = (u64::from(u32::MAX) * u64::from(coverage)).min(u64::MAX / u64::from(price));
        for execution in [
            SearchExecution::Plane,
            SearchExecution::Shared,
            SearchExecution::Cached,
        ] {
            let outcome = probe.outcome(execution)?;
            assert_ne!(outcome.proposal[ProposalLayout::ACTOR], NO_ACTOR);
            let quantity = u64::from(outcome.proposal[ProposalLayout::QUANTITY_LOW])
                | (u64::from(outcome.proposal[ProposalLayout::QUANTITY_HIGH]) << 32);
            assert_eq!(quantity, expected, "price {price}, execution {execution:?}");
            assert_eq!(outcome.faults, 0);
        }
    }
    Ok(())
}

#[test]
fn indexed_bread_minimum_quotes_preserve_zero_targets_empty_offers_and_cash_boundaries_on_gpu()
-> Result<()> {
    #[derive(Clone, Copy, Debug)]
    enum SupplyStock {
        Stocked,
        Empty,
        Mixed,
    }

    struct CashBoundary {
        quote: u32,
        cash: u64,
    }

    for stock in [SupplyStock::Stocked, SupplyStock::Empty, SupplyStock::Mixed] {
        for boundary in [
            CashBoundary { quote: 8, cash: 0 },
            CashBoundary { quote: 8, cash: 7 },
            CashBoundary { quote: 8, cash: 8 },
            CashBoundary {
                quote: 8,
                cash: u64::from(u32::MAX) + 8,
            },
            CashBoundary {
                quote: u32::MAX,
                cash: u64::from(u32::MAX) - 1,
            },
            CashBoundary {
                quote: u32::MAX,
                cash: u64::from(u32::MAX),
            },
            CashBoundary {
                quote: u32::MAX,
                cash: u64::MAX,
            },
        ] {
            for wanted in [3, u32::MAX] {
                let mut probe =
                    SearchProbe::new(SupplyCase::Capped, 127, Market::Input(Good::Bread))?;
                probe.state.set(Field::InputTicks, 1, 0);
                probe.state.set(Field::FoodNeed, 1, u64::from(wanted));
                probe.state.set(Field::Cash, 1, boundary.cash);
                for supplier in 1..=SUPPLIERS as usize {
                    let amount = match (stock, supplier) {
                        (_, 1) | (SupplyStock::Empty, _) | (SupplyStock::Mixed, 2) => 0,
                        _ => 100,
                    };
                    let quote = match (stock, supplier) {
                        (_, 1 | 3) => 0,
                        (SupplyStock::Mixed, 2) => boundary.quote,
                        (SupplyStock::Mixed, _) => boundary.quote.saturating_add(9),
                        _ => boundary.quote,
                    };
                    probe.state.set(Field::Bread, supplier, amount);
                    probe.state.set(Field::Price, supplier, u64::from(quote));
                }
                let expected =
                    probe.actor_outcome(SearchExecution::Plane, 32, CacheRefresh::Reset, 1)?;
                for workgroup in [32, 256] {
                    let actual = probe.actor_outcome(
                        SearchExecution::Cached,
                        workgroup,
                        CacheRefresh::Reset,
                        1,
                    )?;
                    assert_eq!(
                        actual, expected,
                        "stock {stock:?}, quote {}, cash {}, wanted {wanted}, workgroup {workgroup}",
                        boundary.quote, boundary.cash
                    );
                    assert_eq!(
                        probe.simulation.device.read_range(
                            &probe.simulation.control,
                            INPUT_MINIMUM_QUOTE_WORD,
                            1
                        )?[0],
                        boundary.quote,
                    );
                    if boundary.cash < u64::from(boundary.quote) {
                        assert_eq!(actual.proposal[ProposalLayout::ACTOR], NO_ACTOR);
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
fn indexed_partial_input_search_preserves_quantities_ties_and_wide_targets_on_gpu() -> Result<()> {
    #[derive(Clone, Copy, Debug)]
    enum PartialInputOffers {
        Ranked,
        Tied,
    }

    for offers in [PartialInputOffers::Ranked, PartialInputOffers::Tied] {
        for rotation in [0, 31, 128, SUPPLIERS - 1] {
            for input_ticks in [1000, u32::MAX] {
                let mut probe =
                    SearchProbe::new(SupplyCase::Capped, rotation, Market::Input(Good::Bread))?;
                probe.state.set(Field::FoodNeed, 1, u64::from(input_ticks));
                probe
                    .state
                    .set(Field::InputTicks, 1, u64::from(input_ticks));
                probe
                    .state
                    .set(Field::Cash, 1, u64::from(u32::MAX) + 10_000);
                let capacity = match input_ticks {
                    u32::MAX => 2,
                    _ => 1,
                };
                let mut actor = probe.state.actors.at(1);
                actor.capacity = capacity;
                probe.state.actors.set(1, actor);
                let mut actors = probe.simulation.device.read(&probe.simulation.actors)?;
                actors[ActorLayout::CAPACITY * probe.state.actors.len() + 1] = capacity;
                probe
                    .simulation
                    .device
                    .write(&probe.simulation.actors, &actors);
                for supplier in 1..=SUPPLIERS as usize {
                    let stock = match supplier {
                        8 | 97 | 257 => 200,
                        _ => (supplier % 7) as u64,
                    };
                    let quote = match offers {
                        PartialInputOffers::Ranked => (1 + supplier % 11) as u64,
                        PartialInputOffers::Tied => 1,
                    };
                    probe.state.set(Field::Bread, supplier, stock);
                    probe.state.set(Field::Price, supplier, quote);
                }
                let expected =
                    probe.actor_outcome(SearchExecution::Plane, 32, CacheRefresh::Reset, 1)?;
                let chosen = expected.proposal[ProposalLayout::ACTOR];
                match offers {
                    PartialInputOffers::Ranked => assert_eq!(chosen, 257),
                    PartialInputOffers::Tied => assert!([8, 97, 257].contains(&chosen)),
                }
                assert_eq!(expected.proposal[ProposalLayout::QUANTITY_LOW], 200);
                assert_eq!(expected.proposal[ProposalLayout::QUANTITY_HIGH], 0);
                for workgroup in [32, 64, 256] {
                    assert_eq!(
                        probe.actor_outcome(
                            SearchExecution::Cached,
                            workgroup,
                            CacheRefresh::Reset,
                            1
                        )?,
                        expected,
                        "offers {offers:?}, rotation {rotation}, input {input_ticks}, workgroup {workgroup}",
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
fn cached_bread_suppliers_preserve_rotations_stock_and_affordability_on_gpu() -> Result<()> {
    for (case, day) in [
        (SupplyCase::Retail, 0),
        (SupplyCase::Sparse, 1),
        (SupplyCase::DistantStock, 19),
        (SupplyCase::DistantEmpty, SUPPLIERS - 1),
    ] {
        let probe = SearchProbe::new(case, day, Market::Input(Good::Bread))?;
        assert_eq!(
            probe.outcome(SearchExecution::Cached)?,
            probe.outcome(SearchExecution::Plane)?,
            "{case:?} day {day}",
        );
    }
    Ok(())
}

#[test]
fn cooperative_market_search_preserves_serial_search_and_shared_fallback_on_gpu() -> Result<()> {
    let cases = [
        SupplyCase::Capped,
        SupplyCase::Sparse,
        SupplyCase::Empty,
        SupplyCase::Unaffordable,
        SupplyCase::MixedAffordability,
        SupplyCase::Retail,
        SupplyCase::RetailSold,
        SupplyCase::Free,
        SupplyCase::HighCash,
        SupplyCase::DistantStock,
        SupplyCase::DistantEmpty,
    ];
    for market in [
        Market::Household,
        Market::Input(Good::Grain),
        Market::Consumer(Good::Linen),
        Market::Consumer(Good::Glass),
        Market::Consumer(Good::Coal),
        Market::Consumer(Good::Wood),
    ] {
        for (index, case) in cases.into_iter().enumerate() {
            let suppliers = match (market, case) {
                (
                    Market::Household,
                    SupplyCase::Sparse | SupplyCase::Empty | SupplyCase::DistantEmpty,
                ) => 2177,
                _ => SUPPLIERS,
            };
            let day = [0, 128, suppliers - 1][index % 3];
            let mut probe = SearchProbe::with_suppliers(case, day, market, suppliers)?;
            if let (Market::Input(Good::Grain), SupplyCase::Capped) = (market, case) {
                for dealer in 1..=suppliers as usize {
                    probe.state.set(Field::EnterpriseModel, dealer, 1);
                    probe.state.set(Field::RetailOpen, dealer, 100);
                }
            }
            let expected = probe.outcome(SearchExecution::Serial)?;
            if let SupplyCase::Capped = case {
                assert_ne!(expected.proposal[ProposalLayout::ACTOR], NO_ACTOR);
                assert!(expected.proposal[ProposalLayout::QUANTITY_LOW] > 0);
            }
            for execution in [SearchExecution::Shared, SearchExecution::Plane] {
                assert_eq!(
                    probe.outcome(execution)?,
                    expected,
                    "{case:?}, {market:?}, day {day}, {execution:?}"
                );
            }
            if let Market::Consumer(Good::Coal | Good::Wood) = market {
                for execution in [SearchExecution::ScalarFuel, SearchExecution::SpatialFuel] {
                    assert_eq!(
                        probe.outcome(execution)?,
                        expected,
                        "{case:?}, {market:?}, day {day}, {execution:?}"
                    );
                }
            }
            if let Market::Household = market {
                use crate::domain::enterprise::FoodAccess;
                let access = match case {
                    SupplyCase::Capped => Some(FoodAccess::Satisfied),
                    SupplyCase::Empty => Some(FoodAccess::MissingSupply),
                    SupplyCase::Unaffordable => Some(FoodAccess::Unaffordable),
                    _ => None,
                };
                if let Some(access) = access {
                    let stride = probe.simulation.config.state_stride as usize;
                    assert_eq!(
                        expected.state[Field::FoodAccess as usize * stride * TICK_WORDS],
                        access as u32,
                        "{case:?}"
                    );
                }
                assert_eq!(
                    probe.outcome(SearchExecution::ScalarHousehold)?,
                    expected,
                    "{case:?}, {market:?}, day {day}, scalar household"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn cooperative_market_search_only_reports_faults_from_the_visited_prefix_on_gpu() -> Result<()> {
    for available in [7, 6] {
        let probe = SearchProbe::new(SupplyCase::LaborPrefix(available), 0, Market::Labor)?;
        let expected = probe.outcome(SearchExecution::Serial)?;
        assert_eq!(expected.faults == 0, available == 7);
        for execution in [SearchExecution::Shared, SearchExecution::Plane] {
            assert_eq!(
                probe.outcome(execution)?,
                expected,
                "eligible prefix {available}, {execution:?}"
            );
        }
    }
    Ok(())
}

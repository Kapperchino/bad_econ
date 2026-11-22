use super::*;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum EmptyPreparation {
    Bounds,
    ScalarSort,
    BitonicSort,
}

#[derive(CubeType, CubeLaunch)]
struct EmptyPreparationInput {
    actors: Box<[u32]>,
    config: EconomyConfig,
    spatial: Box<[u32]>,
    scratch: Box<[u32]>,
    searches: Box<[u32]>,
    #[cube(comptime)]
    preparation: EmptyPreparation,
    #[cube(comptime)]
    mode: ScanMode,
}

struct EmptyPreparationCheck;
struct CheckEmptyPreparation;

impl System for EmptyPreparationCheck {
    type Input = EmptyPreparationInput;
    type Passes = gpu_passes![CheckEmptyPreparation];
}

#[cube]
impl GpuPass<EmptyPreparationInput> for CheckEmptyPreparation {
    type Output = ();

    fn execute(mut input: EmptyPreparationInput) {
        let actors = ActorTable::new(&input.actors);
        match comptime!(input.preparation) {
            EmptyPreparation::Bounds => empty::bounds(
                &actors,
                &input.config,
                &input.spatial,
                &mut input.scratch,
                comptime!(input.mode),
            ),
            EmptyPreparation::ScalarSort => empty::sort_scalar(
                &input.config,
                &input.spatial,
                &mut input.scratch,
                &mut input.searches,
            ),
            EmptyPreparation::BitonicSort => empty::sort(
                &input.config,
                &input.spatial,
                &mut input.scratch,
                &mut input.searches,
                comptime!(input.mode),
            ),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum EmptyKeys {
    Ties,
    Mixed,
}

#[derive(Debug, PartialEq, Eq)]
struct EmptyPreparationState {
    scratch: Vec<u32>,
    searches: Vec<u32>,
}

struct EmptyPreparationBuffers {
    spatial: CubeTensor,
    scratch: CubeTensor,
    searches: CubeTensor,
    count: usize,
}

impl EmptyPreparationBuffers {
    fn new(simulation: &Simulation, count: usize) -> Self {
        let device = &simulation.device;
        let capacity = simulation.config.actors as usize;
        let mut spatial = vec![0; count + 2];
        spatial[1] = count as u32;
        for member in 0..count {
            spatial[2 + member] = ((member * 73 + 19) % simulation.metadata.actors.len()) as u32;
        }
        Self {
            spatial: device.tensor(&spatial),
            scratch: device.empty(markets::scratch_words(capacity)),
            searches: device.empty(18 * capacity + 32),
            count,
        }
    }

    fn run(
        &self,
        simulation: &Simulation,
        keys: EmptyKeys,
        preparation: EmptyPreparation,
        mode: ScanMode,
    ) -> Result<EmptyPreparationState> {
        let device = &simulation.device;
        let capacity = simulation.config.actors as usize;
        let mut scratch = vec![u32::MAX; self.scratch.meta.num_elements()];
        for member in 0..self.count {
            scratch[4 * member] = match member % 7 {
                0 => 0,
                _ => (member as u32 * 17) % 1000 + 1,
            };
            scratch[4 * member + 1] = member as u32;
            scratch[4 * member + 2] = 0;
            scratch[4 * member + 3] = (member % 4) as u32;
            scratch[5 * capacity + member] = match (keys, member % 8) {
                (EmptyKeys::Ties, _) => 16_383,
                (EmptyKeys::Mixed, 0) => 16_383,
                _ => (member as u32 * 193) % 31,
            };
            scratch[7 * capacity + member] = (self.count - member - 1) as u32;
        }
        device.write(&self.scratch, &scratch);
        device.write(
            &self.searches,
            &vec![u32::MAX; self.searches.meta.num_elements()],
        );
        let invocations = match preparation {
            EmptyPreparation::Bounds => empty_bounds_invocations(capacity, mode),
            _ => self.count.max(1),
        };
        EmptyPreparationCheck::launch_elements(
            device,
            invocations,
            EmptyPreparationInputLaunch {
                actors: simulation.actors.buffer_arg(),
                config: simulation.config.launch(),
                spatial: self.spatial.buffer_arg(),
                scratch: self.scratch.buffer_arg(),
                searches: self.searches.buffer_arg(),
                preparation,
                mode,
            },
        );
        Ok(EmptyPreparationState {
            scratch: device.read(&self.scratch)?,
            searches: device.read(&self.searches)?,
        })
    }
}

#[test]
fn plane_empty_bounds_match_scalar_trees_flat_blocks_and_rail_access_on_gpu() -> Result<()> {
    let scenario = Scenario::industrial(8192, 42, Experiment::Transport)?;
    for workgroup in [32, 64, 256] {
        let mut simulation = Simulation::new(scenario.clone(), device(workgroup)?)?;
        simulation.railways(railway::RailStage::Access);
        simulation.config.spatial_side = 1;
        for capacity in [1024, 8192] {
            simulation.config.actors = capacity;
            for availability in [
                railway::RouteAvailability::Available,
                railway::RouteAvailability::Dormant,
            ] {
                simulation
                    .config
                    .railway
                    .replace_route_availability(availability);
                for count in [0, 1, 7, 8, 9, 31, 32, 33, 511, 513, 957, 2048, 2049]
                    .into_iter()
                    .filter(|&count| count <= capacity as usize)
                {
                    let buffers = EmptyPreparationBuffers::new(&simulation, count);
                    let scalar = buffers.run(
                        &simulation,
                        EmptyKeys::Mixed,
                        EmptyPreparation::Bounds,
                        ScanMode::Shared,
                    )?;
                    let plane = buffers.run(
                        &simulation,
                        EmptyKeys::Mixed,
                        EmptyPreparation::Bounds,
                        ScanMode::new(&simulation.device),
                    )?;
                    assert_eq!(
                        plane, scalar,
                        "workgroup {workgroup}, capacity {capacity}, count {count}, {availability:?}"
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
fn bitonic_empty_sort_matches_scalar_morton_ties_and_original_stocked_order_on_gpu() -> Result<()> {
    let scenario = Scenario::industrial(8192, 42, Experiment::Transport)?;
    for workgroup in [32, 64, 128, 256] {
        let mut simulation = Simulation::new(scenario.clone(), device(workgroup)?)?;
        simulation.config.spatial_side = 1;
        for keys in [EmptyKeys::Ties, EmptyKeys::Mixed] {
            for count in [
                0, 1, 7, 8, 31, 32, 33, 257, 957, 1023, 1024, 1025, 2047, 2048, 2049,
            ] {
                let buffers = EmptyPreparationBuffers::new(&simulation, count);
                let scalar = buffers.run(
                    &simulation,
                    keys,
                    EmptyPreparation::ScalarSort,
                    ScanMode::Shared,
                )?;
                for mode in [ScanMode::Shared, ScanMode::new(&simulation.device)] {
                    let bitonic =
                        buffers.run(&simulation, keys, EmptyPreparation::BitonicSort, mode)?;
                    assert_eq!(
                        bitonic, scalar,
                        "workgroup {workgroup}, count {count}, {keys:?}, {mode:?}"
                    );
                    if count > 0 && count <= empty::LIMIT as usize {
                        let capacity = simulation.config.actors as usize;
                        let expected = (0..count)
                            .filter(|member| member % 4 & 1 != 0)
                            .map(|member| member as u32)
                            .collect::<Vec<_>>();
                        assert_eq!(bitonic.scratch[7 * capacity + count], expected.len() as u32);
                        assert_eq!(
                            bitonic.searches[17 * capacity..17 * capacity + expected.len()],
                            expected
                        );
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
fn single_cell_search_geometry_preserves_inconsistent_radius_fallback_on_gpu() -> Result<()> {
    for good in [Good::Coal, Good::Linen] {
        let mut probe =
            SearchProbe::with_suppliers(SupplyCase::Capped, 0, Market::Consumer(good), 193)?;
        let count = probe.state.actors.len();
        let map = probe.simulation.config.map_size;
        let mut actors = probe.simulation.device.read(&probe.simulation.actors)?;
        for actor in 0..count {
            let mut record = probe.state.actors.at(actor);
            record.position = Position {
                x: map - 1,
                y: map - 1,
            };
            actors[ActorLayout::X * count + actor] = record.position.x;
            actors[ActorLayout::Y * count + actor] = record.position.y;
            probe.state.actors.set(actor, record);
            probe.state.set(Field::Price, actor, 1);
        }
        probe
            .simulation
            .device
            .write(&probe.simulation.actors, &actors);
        for radius in [map, map.div_ceil(4)] {
            probe.simulation.config.search_radius = radius;
            for rotation in [0, 31, 192] {
                probe.parameters.settlement_rotation = rotation;
                let expected = probe.outcome(SearchExecution::Serial)?;
                assert_eq!(
                    expected.proposal[ProposalLayout::ACTOR] == NO_ACTOR,
                    radius < map,
                );
                for execution in [
                    SearchExecution::Shared,
                    SearchExecution::Plane,
                    SearchExecution::ScalarFuel,
                    SearchExecution::SpatialFuel,
                ] {
                    assert_eq!(
                        probe.outcome(execution)?,
                        expected,
                        "{good:?}, radius {radius}, rotation {rotation}, {execution:?}",
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
fn indexed_empty_suppliers_preserve_rotated_ties_across_cells_on_gpu() -> Result<()> {
    for good in [Good::Coal, Good::Cloth] {
        let mut probe =
            SearchProbe::with_suppliers(SupplyCase::Empty, 0, Market::Consumer(good), 511)?;
        let count = probe.state.actors.len();
        let map = probe.simulation.config.map_size;
        let radius = map.div_ceil(3);
        let side = map.div_ceil(radius);
        let cells = (side * side) as usize;
        let mut actors = probe.simulation.device.read(&probe.simulation.actors)?;
        for actor in 0..count {
            let mut record = probe.state.actors.at(actor);
            record.position = match actor {
                0 => Position {
                    x: radius - 1,
                    y: radius - 1,
                },
                _ => Position {
                    x: (actor as u32 * 97) % map,
                    y: (actor as u32 * 193) % map,
                },
            };
            actors[ActorLayout::X * count + actor] = record.position.x;
            actors[ActorLayout::Y * count + actor] = record.position.y;
            probe.state.actors.set(actor, record);
            probe.state.set(Field::Price, actor, 1);
        }
        probe
            .simulation
            .device
            .write(&probe.simulation.actors, &actors);
        probe.simulation.config.search_radius = radius;
        probe.simulation.config.spatial_side = side;
        for active in [0usize, 1, 511] {
            let mut members = (1..=active).collect::<Vec<_>>();
            let cell = |actor| {
                let position = probe.state.actors.at(actor).position;
                (position.y / radius * side + position.x / radius) as usize
            };
            members.sort_by_key(|&actor| (cell(actor), actor));
            let segment = cells + 1 + count;
            let mut spatial =
                vec![0; 3 * segment + count.div_ceil(DISCOVERY_BLOCK as usize) + 1 + count];
            for (position, &actor) in members.iter().enumerate() {
                spatial[cell(actor) + 1] += 1;
                spatial[cells + 1 + position] = actor as u32;
            }
            for cell in 1..=cells {
                spatial[cell] += spatial[cell - 1];
            }
            let query = 3 * segment + count.div_ceil(DISCOVERY_BLOCK as usize) + 1;
            spatial[query..query + count]
                .copy_from_slice(&(0..count as u32).rev().collect::<Vec<_>>());
            probe.simulation.spatial = probe.simulation.device.tensor(&spatial);
            for rotation in [0, 31, 128, 510, 1024] {
                probe.parameters.settlement_rotation = rotation;
                let expected = probe.outcome(SearchExecution::Serial)?;
                for execution in [SearchExecution::ScalarFuel, SearchExecution::SpatialFuel] {
                    assert_eq!(
                        probe.outcome(execution)?,
                        expected,
                        "{good:?}, active {active}, rotation {rotation}, {execution:?}",
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
fn supplier_cache_capacity_preserves_rotated_ties_at_the_inline_boundary_on_gpu() -> Result<()> {
    for (good, suppliers, fields, cached) in [
        (Good::Linen, 256, 18, 1),
        (Good::Linen, 257, 19, 1),
        (Good::Grain, 257, 18, 0),
    ] {
        let mut probe =
            SearchProbe::with_suppliers(SupplyCase::Empty, 0, Market::Consumer(good), suppliers)?;
        let count = probe.state.actors.len();
        assert_eq!(
            probe.simulation.searches.requests.meta.num_elements(),
            count * fields
        );
        probe.state.set(Field::inventory(good), 0, 0);
        for firm in 1..count {
            probe.state.set(Field::Price, firm, 1);
        }
        for rotation in [0, suppliers - 1, suppliers] {
            probe.parameters.settlement_rotation = rotation;
            let expected = probe.outcome(SearchExecution::Serial)?;
            assert_eq!(
                expected.proposal[ProposalLayout::ACTOR],
                rotation % suppliers + 1
            );
            let actual =
                probe.outcome_with_cache(SearchExecution::ScalarFuel, 64, CacheRefresh::Retain)?;
            assert_eq!(actual, expected);
            let header = probe.simulation.device.read_range(
                &probe.simulation.searches.requests,
                14 * count,
                1,
            )?[0];
            assert_eq!(header & 1, cached);
        }
    }
    Ok(())
}

#[test]
fn empty_supplier_cache_summary_preserves_empty_single_word_and_cross_word_ties_on_gpu()
-> Result<()> {
    #[derive(Clone, Copy, Debug)]
    enum Ties {
        Empty,
        FirstWord,
        LastWord,
        CrossWord,
    }

    for suppliers in [33u32, 193, 2048] {
        for ties in [
            Ties::Empty,
            Ties::FirstWord,
            Ties::LastWord,
            Ties::CrossWord,
        ] {
            let mut probe = SearchProbe::with_suppliers(
                SupplyCase::Empty,
                0,
                Market::Consumer(Good::Linen),
                suppliers,
            )?;
            let count = probe.state.actors.len();
            let last_word = (suppliers - 1) / 32;
            let mut actors = probe.simulation.device.read(&probe.simulation.actors)?;
            for actor in 0..count {
                let x = match (ties, actor) {
                    (_, 0) => 0,
                    (Ties::Empty, _) => probe.simulation.config.map_size - 1,
                    _ => 0,
                };
                let mut record = probe.state.actors.at(actor);
                record.position = Position { x, y: 0 };
                actors[ActorLayout::X * count + actor] = x;
                actors[ActorLayout::Y * count + actor] = 0;
                probe.state.actors.set(actor, record);
                let member = actor.saturating_sub(1) as u32;
                let price = match ties {
                    Ties::Empty => 1,
                    Ties::FirstWord => match member {
                        1 | 7 | 31 => 1,
                        _ => 10_000,
                    },
                    Ties::LastWord => match member / 32 == last_word {
                        true => 1,
                        false => 10_000,
                    },
                    Ties::CrossWord => match [0, suppliers - 1].contains(&member) {
                        true => 1,
                        false => 10_000,
                    },
                };
                probe.state.set(Field::Price, actor, price);
            }
            probe.simulation.config.search_radius = 1;
            probe
                .simulation
                .device
                .write(&probe.simulation.actors, &actors);
            let cases = [5000, 5000, 10_000, 5000]
                .into_iter()
                .flat_map(|cash| {
                    [0, 1, 7, 31, 32, suppliers - 1, suppliers, u32::MAX]
                        .into_iter()
                        .map(move |rotation| (cash, rotation))
                })
                .collect::<Vec<_>>();
            let mut expected = Vec::new();
            for &(cash, rotation) in &cases {
                probe.state.set(Field::Cash, 0, cash);
                probe.parameters.settlement_rotation = rotation;
                expected.push(probe.outcome(SearchExecution::Serial)?);
            }
            for ((cash, rotation), expected) in cases.into_iter().zip(expected) {
                probe.state.set(Field::Cash, 0, cash);
                probe.parameters.settlement_rotation = rotation;
                let actual = probe.outcome_with_cache(
                    SearchExecution::ScalarFuel,
                    64,
                    CacheRefresh::Retain,
                )?;
                assert_eq!(
                    actual, expected,
                    "{suppliers}, {ties:?}, {cash}, {rotation}"
                );
                let header = probe.simulation.device.read_range(
                    &probe.simulation.searches.requests,
                    14 * count,
                    1,
                )?[0];
                let summary = match ties {
                    Ties::Empty => 0,
                    Ties::CrossWord => 2,
                    _ => 1,
                };
                assert_eq!(header & 0xffff, 1);
                assert_eq!(header >> 30, summary);
                if let Ties::LastWord = ties {
                    assert_eq!((header >> 16) & 63, last_word);
                }
            }
        }
    }
    Ok(())
}

#[test]
fn empty_supplier_cache_tracks_stock_decreases_before_reachable_restocking_on_gpu() -> Result<()> {
    struct SupplierStocks {
        distant: u64,
        unaffordable: u64,
        reachable: u64,
        captured: u32,
    }

    let cases = [
        SupplierStocks {
            distant: 1,
            unaffordable: 1,
            reachable: 0,
            captured: 33,
        },
        SupplierStocks {
            distant: 0,
            unaffordable: 1,
            reachable: 0,
            captured: 17,
        },
        SupplierStocks {
            distant: 0,
            unaffordable: 0,
            reachable: 0,
            captured: 1,
        },
        SupplierStocks {
            distant: 0,
            unaffordable: 0,
            reachable: 1,
            captured: 1,
        },
    ];
    let mut probe =
        SearchProbe::with_suppliers(SupplyCase::Empty, 0, Market::Consumer(Good::Linen), 511)?;
    let count = probe.state.actors.len();
    let mut actors = probe.simulation.device.read(&probe.simulation.actors)?;
    for actor in 0..count {
        let mut record = probe.state.actors.at(actor);
        record.position = match actor {
            510 => Position {
                x: probe.simulation.config.map_size - 1,
                y: probe.simulation.config.map_size - 1,
            },
            _ => Position { x: 0, y: 0 },
        };
        actors[ActorLayout::X * count + actor] = record.position.x;
        actors[ActorLayout::Y * count + actor] = record.position.y;
        probe.state.actors.set(actor, record);
        probe.state.set(
            Field::Price,
            actor,
            match actor {
                511 => 10_000,
                _ => 1,
            },
        );
    }
    probe
        .simulation
        .device
        .write(&probe.simulation.actors, &actors);
    let mut expected = Vec::new();
    for stocks in &cases {
        probe.state.set(Field::Linen, 510, stocks.distant);
        probe.state.set(Field::Linen, 511, stocks.unaffordable);
        probe.state.set(Field::Linen, 1, stocks.reachable);
        expected.push(probe.outcome(SearchExecution::Serial)?);
    }
    for (stocks, expected) in cases.into_iter().zip(expected) {
        probe.state.set(Field::Linen, 510, stocks.distant);
        probe.state.set(Field::Linen, 511, stocks.unaffordable);
        probe.state.set(Field::Linen, 1, stocks.reachable);
        let actual =
            probe.outcome_with_cache(SearchExecution::ScalarFuel, 64, CacheRefresh::Retain)?;
        assert_eq!(actual, expected);
        let requests = probe
            .simulation
            .device
            .read(&probe.simulation.searches.requests)?;
        assert_eq!(requests[14 * count] & 0xffff, stocks.captured);
        if stocks.reachable > 0 {
            assert_eq!(actual.proposal[ProposalLayout::ACTOR], 1);
            assert_eq!(actual.proposal[ProposalLayout::QUANTITY_LOW], 1);
        }
    }
    Ok(())
}

#[test]
fn compacted_stocked_suppliers_preserve_original_rotation_order_on_gpu() -> Result<()> {
    let mut probe =
        SearchProbe::with_suppliers(SupplyCase::Sparse, 0, Market::Consumer(Good::Coal), 511)?;
    let stocked = [1usize, 2, 32, 33, 65, 129, 130, 257, 383, 385];
    let count = probe.state.actors.len();
    let map = probe.simulation.config.map_size;
    let mut actors = probe.simulation.device.read(&probe.simulation.actors)?;
    for actor in 1..count {
        let available = match stocked.binary_search(&actor) {
            Ok(_) => 100,
            Err(_) => 0,
        };
        probe.state.set(Field::Coal, actor, available);
        probe
            .state
            .set(Field::Price, actor, 1 + (511 - actor) as u64);
        let mut record = probe.state.actors.at(actor);
        record.position = Position {
            x: (actor as u32 * 97) % map,
            y: (actor as u32 * 193) % map,
        };
        actors[ActorLayout::X * count + actor] = record.position.x;
        actors[ActorLayout::Y * count + actor] = record.position.y;
        probe.state.actors.set(actor, record);
    }
    probe
        .simulation
        .device
        .write(&probe.simulation.actors, &actors);
    for cash in [3, 5000] {
        probe.state.set(Field::Cash, 0, cash);
        for rotation in [0, 31, 128, 257, 384, 1024] {
            probe.parameters.settlement_rotation = rotation;
            let expected = probe.outcome(SearchExecution::Serial)?;
            for execution in [SearchExecution::ScalarFuel, SearchExecution::SpatialFuel] {
                let actual = probe.outcome(execution)?;
                assert_eq!(
                    actual.proposal, expected.proposal,
                    "cash {cash}, rotation {rotation}, {execution:?}"
                );
                assert_eq!(actual.faults, expected.faults);
                assert!(actual.state == expected.state);
            }
        }
    }
    Ok(())
}

#[test]
fn global_supplier_price_floor_preserves_poor_and_wide_budgets_on_gpu() -> Result<()> {
    for good in [Good::Coal, Good::Cloth] {
        for case in [SupplyCase::Capped, SupplyCase::Empty, SupplyCase::Free] {
            let mut probe = SearchProbe::with_suppliers(case, 0, Market::Consumer(good), 211)?;
            for actor in 1..probe.state.actors.len() {
                let price = match case {
                    SupplyCase::Free => 0,
                    _ => 30 + actor as u64 % 2,
                };
                probe.state.set(Field::Price, actor, price);
            }
            probe.state.set(Field::Price, 1, 0);
            for cash in [0, 1, 29, 30, u64::from(u32::MAX) + 5000] {
                probe.state.set(Field::Cash, 0, cash);
                let expected = probe.outcome(SearchExecution::Shared)?;
                for execution in [SearchExecution::ScalarFuel, SearchExecution::SpatialFuel] {
                    assert_eq!(
                        probe.outcome(execution)?,
                        expected,
                        "{good:?}, {case:?}, cash {cash}, {execution:?}",
                    );
                }
            }
        }
    }
    Ok(())
}

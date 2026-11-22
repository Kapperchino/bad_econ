use super::*;

pub(super) const SUPPLIERS: u32 = 385;

#[derive(Clone, Copy, Debug)]
pub(super) enum SupplyCase {
    Capped,
    Sparse,
    Empty,
    Unaffordable,
    MixedAffordability,
    Retail,
    RetailSold,
    Free,
    HighCash,
    DistantStock,
    DistantEmpty,
    LaborPrefix(u32),
}

#[derive(Clone, Copy, Debug)]
pub(super) enum SearchExecution {
    Serial,
    Prepare,
    Shared,
    Plane,
    Cached,
    Grain,
    ScalarFuel,
    ScalarHousehold,
    SpatialFuel,
}

#[derive(Clone, Copy)]
pub(super) enum CacheRefresh {
    Reset,
    Retain,
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct SearchOutcome {
    pub(super) proposal: Vec<u32>,
    pub(super) state: Vec<u32>,
    pub(super) faults: u32,
}

pub(super) struct SearchProbe {
    pub(super) simulation: Simulation,
    pub(super) state: State,
    pub(super) parameters: EconomyParameters,
}

impl SearchProbe {
    pub(super) fn new(case: SupplyCase, day: u32, market: Market) -> Result<Self> {
        Self::with_suppliers(case, day, market, SUPPLIERS)
    }

    pub(super) fn with_suppliers(
        case: SupplyCase,
        day: u32,
        market: Market,
        suppliers: u32,
    ) -> Result<Self> {
        let good = match market {
            Market::Input(good) | Market::Consumer(good) => good,
            _ => Good::Bread,
        };
        let mut config = Scenario::baseline(1, 42, Experiment::Concentrated)?
            .config()
            .clone();
        config.firms = suppliers;
        config.search_radius = config.map_size;
        config.household_production = HouseholdProduction::industrial();
        let position = match case {
            SupplyCase::DistantStock | SupplyCase::DistantEmpty => Position { x: 0, y: 0 },
            _ => Position {
                x: config.map_size / 2,
                y: config.map_size / 2,
            },
        };
        let input_ticks = match good.inputs()[0] {
            Some(_) => 1,
            None => 0,
        };
        config.population = Some(Population {
            households: vec![HouseholdSeed {
                identity: 1,
                position,
                cash: 5000,
                inventories: Good::food([0, 0, 100]),
                fatigue: 0,
            }],
            firms: (2..=u64::from(suppliers) + 1)
                .map(|identity| FirmSeed {
                    identity,
                    position: match case {
                        SupplyCase::DistantStock | SupplyCase::DistantEmpty => match identity {
                            2..=129 => Position {
                                x: config.map_size - 1,
                                y: config.map_size - 1,
                            },
                            130..=257 => Position {
                                x: config.map_size * 3 / 4,
                                y: 0,
                            },
                            _ => Position { x: 0, y: 1 },
                        },
                        _ => position,
                    },
                    owner: 1,
                    good,
                    cash: 5000,
                    inventories: [0; Good::COUNT],
                    capacity: 1,
                    technique: Technique::new(1, input_ticks, 1).expect("one-minute packet"),
                    wage: 1,
                    price: 1,
                })
                .collect(),
        });
        if let SupplyCase::LaborPrefix(_) = case {
            config.bargaining = Some(crate::domain::bargaining::BargainingPolicy {
                legal_minimum: 1,
                quantum: 1,
                daily_cash_cost: 1_000_000,
                funded_other: 0,
                participating_adults: 1,
                normal_day_minutes: 1440,
                work_cost_per_minute: 0,
                desired_wage: 1000,
                mandate: 500,
                withdrawal: 0,
                alternatives: 0,
                concentration: 0,
                replacements: 0,
            });
        }
        let scenario = Scenario::new(config)?;
        let mut state = scenario.initial()?;
        state.set(Field::FoodNeed, 0, 3);
        state.set(Field::LivingMembers, 0, 3);
        if let Market::Consumer(Good::Coal | Good::Wood) = market {
            state.set(Field::HeatingNeed, 0, 9);
        }
        let bread = match market {
            Market::Consumer(_) => {
                state.get(Field::FoodNeed, 0)
                    * (u64::from(scenario.config().coverage) + scenario.ring() as u64)
            }
            _ => 0,
        };
        state.set(Field::Bread, 0, bread);
        if let Market::Consumer(_) = market {
            state.set(Field::Price, 0, 1);
        }
        state.set(Field::Time, 0, u64::from(u32::MAX));
        if let SupplyCase::HighCash = case {
            state.set(Field::Cash, 0, u64::from(u32::MAX) + 5000);
        }
        for firm in 1..=suppliers as usize {
            let supply = match case {
                SupplyCase::Capped
                | SupplyCase::Unaffordable
                | SupplyCase::Retail
                | SupplyCase::RetailSold
                | SupplyCase::Free
                | SupplyCase::HighCash
                | SupplyCase::DistantStock => 100,
                SupplyCase::MixedAffordability => u64::from(firm > 1) * 100,
                SupplyCase::Sparse => u64::from(firm % 129 == 0) * 100,
                _ => 0,
            };
            let price = match case {
                SupplyCase::Capped => 1 + u64::from(firm <= 8) * 9,
                SupplyCase::Sparse | SupplyCase::Empty => 1 + (firm % 3) as u64,
                SupplyCase::Unaffordable => 5001,
                SupplyCase::MixedAffordability => u64::from(firm > 1) * 5001,
                SupplyCase::Retail | SupplyCase::RetailSold => 1 + (firm % 3) as u64,
                SupplyCase::Free => 0,
                SupplyCase::HighCash => 5001,
                SupplyCase::LaborPrefix(_)
                | SupplyCase::DistantStock
                | SupplyCase::DistantEmpty => 1,
            };
            state.set(Field::inventory(good), firm, supply);
            state.set(Field::Price, firm, price);
            state.set(Field::LaborMinutes, firm, 1);
            state.set(Field::InputTicks, firm, 0);
            state.set(Field::EquipmentCondition, firm, 1000);
            if let SupplyCase::Retail | SupplyCase::RetailSold = case {
                state.set(Field::EnterpriseModel, firm, 1);
                state.set(Field::RetailOpen, firm, u64::from(u32::MAX) + 10);
                let sales = match case {
                    SupplyCase::Retail => u64::from(u32::MAX) + firm as u64 % 10,
                    _ => u64::from(u32::MAX) + 11,
                };
                state.set(Field::Sales, firm, sales);
            }
            if let SupplyCase::LaborPrefix(available) = case {
                state.set(
                    Field::Slots,
                    firm,
                    u64::from(firm <= available as usize || firm == 9 || firm == 10),
                );
            }
        }
        if let SupplyCase::LaborPrefix(_) = case {
            state.set(Field::LaborMinutes, 10, u64::from(u32::MAX));
            state.set(Field::EquipmentCondition, 10, u64::from(u32::MAX));
        }
        let market = match case {
            SupplyCase::LaborPrefix(_) => Market::Labor,
            _ => market,
        };
        let simulation = Simulation::new(scenario, device(64)?)?;
        let mut parameters =
            simulation.parameters(Phase::ProvisionPurchase, market, 0, state.actors.len())?;
        parameters.day = match market {
            Market::Consumer(_) => 0,
            _ => day,
        };
        parameters.settlement_rotation = day;
        let mut spatial = simulation.device.read(&simulation.spatial)?;
        spatial.fill(0);
        spatial[1] = suppliers;
        spatial[2..suppliers as usize + 2].copy_from_slice(&(1..=suppliers).collect::<Vec<u32>>());
        let prices = 3 * (2 + state.actors.len());
        spatial[prices] = (1..=suppliers as usize)
            .map(|firm| state.get(Field::Price, firm) as u32)
            .min()
            .expect("suppliers");
        let query = prices + state.actors.len().div_ceil(DISCOVERY_BLOCK as usize) + 1;
        let mut queries = (0..state.actors.len() as u32).collect::<Vec<_>>();
        if let Market::Consumer(_) = market {
            queries.reverse();
        }
        spatial[query..query + state.actors.len()].copy_from_slice(&queries);
        simulation.device.write(&simulation.spatial, &spatial);
        Ok(Self {
            simulation,
            state,
            parameters,
        })
    }

    pub(super) fn outcome(&self, execution: SearchExecution) -> Result<SearchOutcome> {
        self.outcome_with_workgroup(
            execution,
            match execution {
                SearchExecution::Cached => 256,
                _ => 32,
            },
        )
    }

    pub(super) fn outcome_with_workgroup(
        &self,
        execution: SearchExecution,
        workgroup: u32,
    ) -> Result<SearchOutcome> {
        self.outcome_with_cache(execution, workgroup, CacheRefresh::Reset)
    }

    pub(super) fn outcome_with_cache(
        &self,
        execution: SearchExecution,
        workgroup: u32,
        cache: CacheRefresh,
    ) -> Result<SearchOutcome> {
        self.actor_outcome(execution, workgroup, cache, 0)
    }

    pub(super) fn actor_outcome(
        &self,
        execution: SearchExecution,
        workgroup: u32,
        cache: CacheRefresh,
        buyer: usize,
    ) -> Result<SearchOutcome> {
        let simulation = &self.simulation;
        let device = &simulation.device;
        let count = self.state.actors.len();
        let mut parameters = self.parameters;
        parameters.scan = match execution {
            SearchExecution::Shared => ScanMode::Shared,
            _ => ScanMode::new(device),
        };
        device.write(
            &simulation.state,
            bytemuck::cast_slice(&encode(&self.state.values)),
        );
        let mut control = vec![0; simulation.control.meta.num_elements()];
        control[INPUT_MINIMUM_QUOTE_WORD] = u32::MAX;
        control[SEARCHES_WORD] = 1;
        device.write(&simulation.control, &control);
        device.write(
            &simulation.work.proposals,
            &vec![NO_ACTOR; count * ProposalLayout::WORDS],
        );
        let mut searches = match cache {
            CacheRefresh::Reset => vec![0; simulation.searches.requests.meta.num_elements()],
            CacheRefresh::Retain => device.read(&simulation.searches.requests)?,
        };
        searches[..count * 6].fill(0);
        searches[0] = buyer as u32;
        let coverage = match parameters.market {
            MarketFlag::Input(_) if parameters.good == Good::Grain as u32 => {
                simulation.config.coverage + simulation.config.ring
            }
            _ => simulation.config.coverage + simulation.config.ring - 1,
        };
        let wanted = match parameters.market {
            MarketFlag::Consumer(_) if parameters.good == Good::Coal as u32 => {
                (self.state.get(Field::HeatingNeed, buyer) * u64::from(coverage)).div_ceil(2)
            }
            MarketFlag::Consumer(_) if parameters.good == Good::Wood as u32 => {
                self.state.get(Field::HeatingNeed, buyer) * u64::from(coverage)
            }
            MarketFlag::Consumer(_) => 1,
            _ => self.state.get(Field::FoodNeed, buyer) * u64::from(coverage),
        };
        searches[count + buyer] = wanted as u32;
        searches[2 * count + buyer] = (wanted >> 32) as u32;
        searches[3 * count + buyer] = self.state.get(Field::Cash, buyer) as u32;
        searches[4 * count + buyer] = (self.state.get(Field::Cash, buyer) >> u32::BITS) as u32;
        device.write(&simulation.searches.requests, &searches);
        device.write(
            &simulation.searches.successes,
            &vec![0; orders::search_counts_offset(count) + 2 * orders::search_blocks(count) + 512],
        );
        if matches!(
            (parameters.market, execution),
            (MarketFlag::Consumer(_) | MarketFlag::Household, _) | (_, SearchExecution::Cached)
        ) {
            MarketSearch::launch_elements(
                device,
                count.div_ceil(DISCOVERY_BLOCK as usize) * device.workgroup as usize,
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
                    program: SearchStage::Index.program(&simulation.config),
                },
            );
            if matches!(parameters.market, MarketFlag::Consumer(_)) {
                for stage in [SearchStage::EmptySort, SearchStage::EmptyBounds] {
                    let invocations = match stage {
                        SearchStage::EmptyBounds => {
                            empty_bounds_invocations(count, parameters.scan)
                        }
                        _ => count,
                    };
                    MarketSearch::launch_elements(
                        device,
                        invocations,
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
                }
            }
        }
        if let SearchExecution::Grain = execution {
            for (stage, grid, width) in [
                (
                    SearchStage::GrainFirms,
                    device.count(count),
                    device.workgroup,
                ),
                (
                    SearchStage::GrainDealerTiles,
                    CubeCount::Static(8, 1, 1),
                    256,
                ),
            ] {
                MarketSearch::launch(
                    device,
                    grid,
                    CubeDim::new_1d(width),
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
            }
        }
        let stage = match execution {
            SearchExecution::Prepare => SearchStage::Prepare,
            SearchExecution::ScalarFuel => SearchStage::SearchFuel,
            SearchExecution::ScalarHousehold => SearchStage::SearchHousehold,
            SearchExecution::SpatialFuel => SearchStage::SearchSpatialFuel,
            SearchExecution::Cached => SearchStage::SearchIndexed,
            SearchExecution::Grain => SearchStage::SearchGrain,
            _ => SearchStage::Search,
        };
        match execution {
            SearchExecution::Serial => EconomySystem::launch_single(
                device,
                EconomyInputLaunch {
                    state: simulation.state.buffer_arg(),
                    actors: simulation.actors.buffer_arg(),
                    config: simulation.config.launch(),
                    spatial: simulation.spatial.buffer_arg(),
                    proposals: simulation.work.proposals.buffer_arg(),
                    bundles: simulation.work.scratch.buffer_arg(),
                    control: simulation.control.buffer_arg(),
                    parameters: parameters.launch(),
                    stage: EconomyStage::Propose,
                },
            ),
            _ => MarketSearch::launch(
                device,
                match execution {
                    SearchExecution::SpatialFuel => {
                        CubeCount::Static(count.div_ceil(workgroup as usize) as u32, 1, 1)
                    }
                    _ => CubeCount::Static(1, 1, 1),
                },
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
            ),
        }
        let proposals = device.read(&simulation.work.proposals)?;
        let mut proposal =
            proposals[buyer * ProposalLayout::WORDS..(buyer + 1) * ProposalLayout::WORDS].to_vec();
        match proposal[ProposalLayout::ACTOR] {
            NO_ACTOR => proposal[ProposalLayout::QUANTITY_LOW..].fill(0),
            _ => (),
        }
        Ok(SearchOutcome {
            proposal,
            state: device.read(&simulation.state)?,
            faults: device.read(&simulation.control)?[crate::simulation::flags::FAULT_WORD],
        })
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct PreparedInputs {
    pub(super) queue: Vec<u32>,
    pub(super) payload: Vec<u32>,
    pub(super) statuses: Vec<u32>,
    pub(super) faults: u32,
}

impl SearchProbe {
    pub(super) fn clear_preparation(&self) {
        let simulation = &self.simulation;
        for buffer in [
            &simulation.control,
            &simulation.searches.requests,
            &simulation.searches.successes,
        ] {
            simulation
                .device
                .write(buffer, &vec![0; buffer.meta.num_elements()]);
        }
        simulation.device.write(
            &simulation.work.scratch,
            &vec![u32::MAX; simulation.work.scratch.meta.num_elements()],
        );
    }

    pub(super) fn prepared_inputs(
        &self,
        initialization: market::MarketStart,
        stage: SearchStage,
    ) -> Result<PreparedInputs> {
        let simulation = &self.simulation;
        let device = &simulation.device;
        let count = self.state.actors.len();
        device.write(
            &simulation.state,
            bytemuck::cast_slice(&encode(&self.state.values)),
        );
        initialization.launch(simulation.market_buffers(), &self.parameters);
        let dispatch = match stage {
            SearchStage::PrepareSelected => orders::dispatch(
                &simulation.searches.dispatches,
                orders::PREPARATION_DISPATCH,
                count.div_ceil(device.workgroup as usize),
            ),
            _ => device.count(count).into(),
        };
        MarketSearch::launch(
            device,
            dispatch,
            CubeDim::new_1d(device.workgroup),
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
                parameters: self.parameters.launch(),
                status_reset: StatusReset::Active,
                program: stage.program(&simulation.config),
            },
        );
        orders::prepare_dispatch(
            device,
            &simulation.control,
            &simulation.searches.dispatches,
            count,
        );
        let control = device.read(&simulation.control)?;
        let requests = device.read(&simulation.searches.requests)?;
        let mut queue = requests[..control[SEARCHES_WORD] as usize].to_vec();
        queue.sort_unstable();
        Ok(PreparedInputs {
            queue,
            payload: requests[count..5 * count].to_vec(),
            statuses: device.read_range(&simulation.work.scratch, 4 * count, count)?,
            faults: control[crate::simulation::flags::FAULT_WORD],
        })
    }
}

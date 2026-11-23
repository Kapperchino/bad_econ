use super::*;

#[cube]
impl SearchInput {
    pub(super) fn prepare_buyer(&mut self, invocation: u32) {
        let round = comptime!(match self.program.stage {
            SearchStage::PrepareCandidates => PreparationRound::Capture,
            SearchStage::PrepareSelected => PreparationRound::Selected,
            _ => PreparationRound::First,
        });
        let mut view = EconomyView::new(&mut self.state, &self.actors, &self.config);
        let economy = &mut view;
        let parameters = &self.parameters;
        let spatial: &[u32] = &self.spatial;
        let proposals: &mut [u32] = &mut self.proposals;
        let searches: &mut [u32] = &mut self.searches;
        let successes: &[Atomic<u32>] = &self.successes;
        let scratch: &mut [u32] = &mut self.scratch;
        let control: &[Atomic<u32>] = &self.control;
        let status_reset = comptime!(self.status_reset);
        let config = economy.ledger.config;
        let mut id = invocation;
        let mut searching = false;
        let mut eligible = false;
        let masks = ceildiv(config.actors, 32);
        if invocation < masks + ceildiv(config.actors, orders::COMPACTION_BLOCK) {
            successes[invocation as usize].store(0);
        }
        let count = match comptime!(round) {
            PreparationRound::Selected => control[orders::PREPARATION_ACTORS_WORD].load(),
            _ => config.actors,
        };
        if invocation < count {
            if comptime!(round == PreparationRound::Selected) {
                id = searches[(5 * config.actors + invocation) as usize];
            }
            if comptime!(status_reset == StatusReset::Full) {
                KernelEventStatus::new(comptime!(EventStatus::Inactive))
                    .store(scratch, 4 * config.actors + invocation);
            }
            let market = comptime!(parameters.market);
            if comptime!(
                matches!(market, MarketFlag::Consumer(_)) && round == PreparationRound::Capture
            ) {
                searches[(14 * config.actors + id) as usize] = 0;
            }
            if comptime!(matches!(
                market,
                MarketFlag::Consumer(_) | MarketFlag::Household
            )) {
                searches[(config.actors + id) as usize] = 0;
                searches[(2 * config.actors + id) as usize] = 0;
            }
            let k = economy.actors.kind(id);
            let mut chosen = u32::new(comptime!(i64::from(NO_ACTOR)));
            let mut quantity = ticks(0u32, 0u32);
            let mut trip = 0u32;
            let good = parameters.traded_good();
            let mut wanted = ticks(0u32, 0u32);
            let household_demand = match comptime!(parameters.market) {
                MarketFlag::Input(_) => {
                    let grain = parameters.traded_good() == Good::Grain as u32
                        && comptime!(config.home_mode == Feature::Enabled)
                        && (comptime!(config.economy == crate::domain::Economy::Agrarian)
                            || economy.ledger.number(Field::LandPlots, id) > 0)
                        && economy.ledger.number(Field::LivingMembers, id) > 0;
                    let investment = (parameters.traded_good() == Good::Tools as u32
                        || parameters.traded_good() == Good::Wood as u32)
                        && comptime!(
                            config.home_mode == Feature::Enabled
                                && config.property_mode == Feature::Disabled
                        )
                        && economy.ledger.number(Field::InvestmentGood, id) > 0
                        && economy.ledger.number(Field::LivingMembers, id) > 0;
                    let construction = parameters.traded_good() == Good::Construction as u32
                        && economy.ledger.number(Field::ProjectStatus, id)
                            == comptime!(crate::domain::property::ProjectStatus::Building as u32);
                    u32::cast_from(grain || investment || construction)
                }
                _ => u32::new(1),
            };
            let preparation = match comptime!(round) {
                PreparationRound::Selected => {
                    u32::new(comptime!(i64::from(Preparation::Demand as u32)))
                }
                _ => select(
                    household_demand == 0
                        && k.is(comptime!(ActorKind::Household))
                        && economy.ledger.number(Field::EnterpriseModel, id) == 0,
                    comptime!(Preparation::Inventory as u32),
                    comptime!(Preparation::Demand as u32),
                ),
            };
            if comptime!(round == PreparationRound::Capture) {
                eligible = preparation == comptime!(Preparation::Demand as u32);
            }
            match preparation {
                0 => {
                    let mut purchasing_power = economy.ledger.value(Field::Cash, id);
                    match market {
                        MarketFlag::Labor => {
                            let labor = employment::LaborSearch::new(
                                &*economy,
                                control,
                                parameters,
                                id,
                                comptime!(employment::LaborRound::FromClock),
                            );
                            chosen = labor.peer;
                            quantity = labor.quantity;
                            trip = labor.trip;
                            searching = labor.vacancies;
                        }
                        MarketFlag::Input(_) | MarketFlag::Consumer(_) | MarketFlag::Household => {
                            let order =
                                PurchaseRequest::new(economy, spatial, control, parameters, id);
                            searching = order.search;
                            wanted = order.wanted;
                            purchasing_power = order.purchasing_power;
                        }
                        MarketFlag::Payroll | MarketFlag::Distribution => {
                            let payment =
                                MarketProposal::payment(&*economy, control, id, comptime!(market));
                            chosen = payment.peer;
                            quantity = payment.quantity;
                        }
                    }

                    if comptime!(
                        round == PreparationRound::Capture
                            && market == MarketFlag::Input(InputMarket::Grain)
                    ) {
                        eligible =
                            select(k.is(comptime!(ActorKind::Household)), searching, eligible);
                    }
                    if comptime!(
                        round == PreparationRound::Capture
                            && matches!(market, MarketFlag::Consumer(_))
                    ) {
                        eligible = searching;
                    }
                    if comptime!(
                        round == PreparationRound::Capture
                            && matches!(market, MarketFlag::Consumer(Good::Coal | Good::Wood))
                            && config.food_income
                                == crate::domain::food::FoodIncome::PurchasingPower
                    ) && !searching
                        && k.is(comptime!(ActorKind::Household))
                    {
                        let comfort = consumption::HeatingDemand::new(&economy.ledger, id).reserve(
                            &economy.ledger,
                            control,
                            good,
                        );
                        let held = economy.ledger.stock_with_transit(good, id).record(control);
                        eligible = economy.ledger.number(Field::LivingMembers, id) > 0
                            && !lt(
                                economy.ledger.value(Field::Bread, id),
                                ticks(food::HouseholdRation::new(&economy.ledger, id).minimum(), 0),
                            )
                            && !equal(purchasing_power, zero())
                            && lt(held, comfort);
                    }
                    if searching {
                        BuyerDemand {
                            wanted,
                            purchasing_power,
                        }
                        .store(searches, config.actors, id);
                    }
                    if comptime!(market == MarketFlag::Household) {
                        let result = FoodSearchResult {
                            seller: NO_ACTOR,
                            seen_supply: false,
                            used_fallback: false,
                        };
                        result.record(&mut economy.ledger, id, wanted, purchasing_power);
                    }
                }
                _ => {
                    if comptime!(matches!(
                        round,
                        PreparationRound::First | PreparationRound::Capture
                    )) {
                        let inactive_good = parameters.traded_good();
                        economy
                            .ledger
                            .stock_with_transit(inactive_good, id)
                            .record(control);
                    }
                }
            }
            if comptime!(matches!(
                market,
                MarketFlag::Labor | MarketFlag::Payroll | MarketFlag::Distribution
            )) {
                MarketProposal {
                    peer: chosen,
                    quantity,
                    trip,
                }
                .store(proposals, id);
            }
        }
        let mut packed = u32::cast_from(searching);
        if comptime!(round == PreparationRound::Capture) {
            packed |= u32::cast_from(eligible) << 16;
        }
        let prefix = parallel::scan_prefix(packed, comptime!(parameters.scan));
        let mut searching_total = prefix.total;
        let mut searching_prefix = prefix.inclusive;
        if comptime!(round == PreparationRound::Capture) {
            searching_total &= 0xffff;
            searching_prefix &= 0xffff;
        }
        let mut base = Shared::<u32>::new();
        let mut eligible_base = Shared::<u32>::new();
        if UNIT_POS == 0 {
            *base = match searching_total {
                0 => 0u32,
                _ => control[SEARCHES_WORD].fetch_add(searching_total),
            };
            if comptime!(round == PreparationRound::Capture) {
                let total = prefix.total >> 16;
                *eligible_base = match total {
                    0 => 0u32,
                    _ => control[orders::PREPARATION_ACTORS_WORD].fetch_add(total),
                };
            }
        }
        sync_cube();
        if searching {
            searches[(*base + searching_prefix - 1) as usize] = id;
        }
        if comptime!(round == PreparationRound::Capture) && eligible {
            searches
                [(5 * config.actors + *eligible_base + (prefix.inclusive >> 16) - 1) as usize] = id;
        }
    }
}

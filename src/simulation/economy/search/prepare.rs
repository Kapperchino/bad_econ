use super::*;

#[cube]
pub(super) fn prepare(input: &mut SearchInput, invocation: u32) {
    let round = comptime!(match input.program.stage {
        SearchStage::PrepareCandidates => PreparationRound::Capture,
        SearchStage::PrepareSelected => PreparationRound::Selected,
        _ => PreparationRound::First,
    });
    let mut view = EconomyView::new(&mut input.state, &input.actors, &input.config);
    let economy = &mut view;
    let parameters = &input.parameters;
    let spatial: &mut [u32] = &mut input.spatial;
    let proposals: &mut [u32] = &mut input.proposals;
    let searches: &mut [u32] = &mut input.searches;
    let successes: &[Atomic<u32>] = &input.successes;
    let scratch: &mut [u32] = &mut input.scratch;
    let control: &[Atomic<u32>] = &input.control;
    let status_reset = comptime!(input.status_reset);
    let config = economy.ledger.config;
    let mut id = invocation;
    let mut searching = false;
    let mut eligible = false;
    let masks = ceildiv(config.actors, 32);
    if invocation < masks + ceildiv(config.actors, markets::COMPACTION_BLOCK) {
        successes[invocation as usize].store(0);
    }
    let count = match comptime!(round) {
        PreparationRound::Selected => control[markets::PREPARATION_ACTORS_WORD].load(),
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
        let mut search = false;
        let mut good = 0u32;
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
                if comptime!(matches!(market, MarketFlag::Input(_))) {
                    purchasing_power = enterprises::purchasing_power(&*economy, control, id);
                }
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
                        search = labor.vacancies;
                    }
                    MarketFlag::Input(_) => {
                        good = parameters.traded_good();
                        let demand = InputDemand::new(economy, control, id, &k, good);
                        search = demand.search;
                        wanted = demand.wanted;
                    }
                    MarketFlag::Consumer(_) => {
                        good = parameters.traded_good();
                        if comptime!(matches!(
                            market,
                            MarketFlag::Consumer(Good::Coal | Good::Wood)
                        )) {
                            wanted = consumption::fuel_purchase_target(
                                &economy.ledger,
                                control,
                                parameters,
                                id,
                                good,
                            );
                            let held = add_checked(
                                control,
                                economy.ledger.value_at(good, id),
                                economy
                                    .ledger
                                    .value_at(TRANSIT_FIELDS as u32 + good - 1, id),
                            );
                            search = k.is(comptime!(ActorKind::Household))
                                && economy.ledger.number(Field::LivingMembers, id) > 0
                                && !lt(
                                    economy.ledger.value(Field::Bread, id),
                                    ticks(food::minimum_daily(&economy.ledger, id), 0),
                                )
                                && lt(held, wanted)
                                && !equal(purchasing_power, zero());
                            if search {
                                search = consumption::preferred_fuel(
                                    &*economy, &*spatial, parameters, id, good,
                                ) == good;
                            }
                        } else {
                            let interval = config.consumption_intervals[good as usize - 1];
                            let until_due = (interval - (parameters.day + id) % max(interval, 1))
                                % max(interval, 1);
                            search = k.is(comptime!(ActorKind::Household))
                                && economy.ledger.number(Field::LivingMembers, id) > 0
                                && (until_due <= config.coverage + config.ring
                                    || comptime!(
                                        matches!(
                                            market,
                                            MarketFlag::Consumer(
                                                Good::LuxuryClothes | Good::LuxuryFurniture
                                            )
                                        ) && config.food_income
                                            == crate::domain::food::FoodIncome::PurchasingPower
                                    ));
                            let reserve = ticks(
                                food::reserve_daily(&economy.ledger, id)
                                    * (config.coverage + config.ring - 1),
                                0,
                            );
                            let food = add_checked(
                                control,
                                economy.ledger.value(Field::Bread, id),
                                economy.ledger.value_at(
                                    TRANSIT_FIELDS as u32 + comptime!(Good::Bread as u32) - 1,
                                    id,
                                ),
                            );
                            let cash_reserve = multiply(
                                control,
                                reserve,
                                economy.ledger.number(Field::FoodPrice, id),
                            );
                            search = search
                                && !lt(food, reserve)
                                && lt(cash_reserve, economy.ledger.value(Field::Cash, id));
                            if comptime!(matches!(
                                market,
                                MarketFlag::Consumer(Good::LuxuryClothes | Good::LuxuryFurniture)
                            )) {
                                purchasing_power =
                                    consumption::budget(&economy.ledger, control, id, good);
                            }
                            search = search && !equal(purchasing_power, zero());
                            wanted = ticks(consumption::consumers(&economy.ledger, id), 0);
                            if comptime!(matches!(
                                market,
                                MarketFlag::Consumer(Good::LuxuryClothes | Good::LuxuryFurniture)
                            )) && search
                            {
                                wanted = consumption::durable_target(
                                    &*economy,
                                    &*spatial,
                                    control,
                                    parameters,
                                    id,
                                    good,
                                    purchasing_power,
                                );
                            }
                        }
                    }
                    MarketFlag::Household => {
                        good = comptime!(Good::Bread as u32);
                        search = k.is(comptime!(ActorKind::Household));
                        wanted = ticks(
                            food::purchase_daily(&economy.ledger, parameters, id)
                                * (config.coverage + config.ring - 1u32),
                            0u32,
                        );
                    }
                    MarketFlag::Payroll => {
                        quantity = economy.ledger.value(Field::Due, id);
                        if !equal(quantity, ticks(0u32, 0u32)) {
                            chosen = economy.ledger.number(Field::Employer, id) - 1u32;
                        }
                    }
                    MarketFlag::Distribution => {
                        if !k.is(comptime!(ActorKind::Household)) {
                            quantity = scaled(
                                control,
                                positive(
                                    economy.ledger.value(Field::Cash, id),
                                    maximum(
                                        select(
                                            k.bits() == comptime!(Good::Housing as u32),
                                            zero(),
                                            ticks(config.reserve_low, config.reserve_high),
                                        ),
                                        economy.ledger.value(Field::OperatingReserve, id),
                                    ),
                                ),
                                config.distribution_rate,
                                POLICY_SCALE,
                            );
                            if !equal(quantity, ticks(0u32, 0u32)) {
                                chosen = economy.actors.owner(id);
                            }
                        }
                    }
                }
                if good > 0u32 {
                    wanted = positive(
                        wanted,
                        add_checked(
                            control,
                            economy.ledger.value_at(good, id),
                            economy
                                .ledger
                                .value_at(TRANSIT_FIELDS as u32 + good - 1, id),
                        ),
                    );
                }
                search = search
                    && economy.ledger.number(Field::Time, id)
                        >= select(comptime!(market == MarketFlag::Labor), 1u32, 0);
                if parameters.traded_good() > 0 {
                    purchasing_power = minimum(
                        purchasing_power,
                        positive(
                            economy.ledger.value(Field::Cash, id),
                            economy.ledger.value(Field::CartagePayable, id),
                        ),
                    );
                }
                if good > 0 {
                    search = search && !equal(wanted, zero()) && !equal(purchasing_power, zero());
                }
                searching = search;
                if comptime!(
                    round == PreparationRound::Capture
                        && market == MarketFlag::Input(InputMarket::Grain)
                ) {
                    eligible = select(k.is(comptime!(ActorKind::Household)), searching, eligible);
                }
                if comptime!(
                    round == PreparationRound::Capture && matches!(market, MarketFlag::Consumer(_))
                ) {
                    eligible = searching;
                }
                if comptime!(
                    round == PreparationRound::Capture
                        && matches!(market, MarketFlag::Consumer(Good::Coal | Good::Wood))
                        && config.food_income == crate::domain::food::FoodIncome::PurchasingPower
                ) && !searching
                    && k.is(comptime!(ActorKind::Household))
                {
                    let comfort = consumption::fuel_target(&economy.ledger, control, id, good);
                    let held = add_checked(
                        control,
                        economy.ledger.value_at(good, id),
                        economy
                            .ledger
                            .value_at(TRANSIT_FIELDS as u32 + good - 1, id),
                    );
                    eligible = economy.ledger.number(Field::LivingMembers, id) > 0
                        && !lt(
                            economy.ledger.value(Field::Bread, id),
                            ticks(food::minimum_daily(&economy.ledger, id), 0),
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
                    let failure = select(
                        equal(wanted, zero()),
                        0u32,
                        select(equal(purchasing_power, zero()), 2u32, 1u32),
                    );
                    economy.ledger.set_number(Field::FoodAccess, id, failure);
                }
            }
            _ => {
                if comptime!(matches!(
                    round,
                    PreparationRound::First | PreparationRound::Capture
                )) {
                    let inactive_good = parameters.traded_good();
                    add_checked(
                        control,
                        economy.ledger.value_at(inactive_good, id),
                        economy
                            .ledger
                            .value_at(TRANSIT_FIELDS as u32 + inactive_good - 1, id),
                    );
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
                _ => control[markets::PREPARATION_ACTORS_WORD].fetch_add(total),
            };
        }
    }
    sync_cube();
    if searching {
        searches[(*base + searching_prefix - 1) as usize] = id;
    }
    if comptime!(round == PreparationRound::Capture) && eligible {
        searches[(5 * config.actors + *eligible_base + (prefix.inclusive >> 16) - 1) as usize] = id;
    }
}

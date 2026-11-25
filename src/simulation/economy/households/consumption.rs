use crate::simulation::economy::{market::demand::ConsumerReserves, *};

use pricing::QuoteSource;

#[cube]
pub(in crate::simulation::economy) fn scaled_luxury(config: &EconomyConfig, good: u32) -> bool {
    comptime!(config.food_income == crate::domain::food::FoodIncome::PurchasingPower)
        && (good == comptime!(Good::LuxuryClothes as u32)
            || good == comptime!(Good::LuxuryFurniture as u32))
}

#[cube]
pub(in crate::simulation::economy) fn consumers(ledger: &Ledger<'_>, actor: u32) -> u32 {
    let config = ledger.config;
    match comptime!(config.cohorts) {
        Feature::Enabled => ledger.number(Field::LivingMembers, actor),
        Feature::Disabled => u32::new(1),
    }
}

#[cube]
pub(in crate::simulation::economy) fn durable_use(
    config: &EconomyConfig,
    members: u32,
    stock: Ticks,
    good: u32,
    day: u32,
) -> Ticks {
    if scaled_luxury(config, good) {
        let interval = max(config.market_table[good as usize - 1], 1);
        let whole = divide(stock, interval);
        let remainder = minus(stock, times(whole, interval).value).extract(LOW_WORD);
        plus(whole, ticks(u32::cast_from(day % interval < remainder), 0)).value
    } else {
        minimum(stock, ticks(members, 0))
    }
}

#[cube]
pub(in crate::simulation::economy) fn input_coverage(config: &EconomyConfig, output: u32) -> u32 {
    select(
        output == comptime!(Good::Flour as u32) || output == comptime!(Good::Bread as u32),
        config.coverage + config.ring - 1,
        config.coverage,
    )
}

#[derive(CubeType)]
pub(in crate::simulation::economy) struct HouseholdPurchase<'a, 'b> {
    economy: &'a EconomyView<'b>,
    spatial: &'a [u32],
    control: &'a [Atomic<u32>],
    parameters: &'a EconomyParameters,
    household: u32,
}

#[cube]
impl<'a, 'b> HouseholdPurchase<'a, 'b> {
    pub fn new(
        economy: &'a EconomyView<'b>,
        spatial: &'a [u32],
        control: &'a [Atomic<u32>],
        parameters: &'a EconomyParameters,
        household: u32,
    ) -> Self {
        HouseholdPurchase::<'a, 'b> {
            economy,
            spatial,
            control,
            parameters,
            household,
        }
    }

    pub fn prepare(
        &self,
        order: &mut PurchaseRequest,
        good: u32,
        #[comptime] quotes: QuoteSource,
        #[comptime] reserves: ConsumerReserves,
    ) {
        let economy = self.economy;
        let parameters = self.parameters;
        let config = economy.ledger.config;
        let control = self.control;
        let id = self.household;
        let market = comptime!(parameters.market);
        if comptime!(matches!(market, MarketFlag::Consumer(ConsumerMarket::Fuel))) {
            order.wanted = self.fuel_target(good);
            let held = economy.ledger.stock_with_transit(good, id).record(control);
            order.search = living_household(&economy.ledger, &economy.actors, id)
                && !lt(
                    economy.ledger.value(Field::Bread, id),
                    ticks(food::HouseholdRation::new(&economy.ledger, id).minimum(), 0),
                )
                && lt(held, order.wanted)
                && !equal(order.purchasing_power, zero());
            if order.search {
                order.search = self.preferred_fuel(good, comptime!(quotes)) == good;
            }
        } else {
            let luxury = comptime!(matches!(
                market,
                MarketFlag::Consumer(ConsumerMarket::Luxury)
            ));
            let interval = config.market_table[good as usize - 1];
            let until_due =
                (interval - (parameters.day + id) % max(interval, 1)) % max(interval, 1);
            order.search = until_due <= config.coverage + config.ring
                || (luxury
                    && comptime!(
                        config.food_income == crate::domain::food::FoodIncome::PurchasingPower
                    ));
            match comptime!(reserves) {
                ConsumerReserves::Ledger => {
                    let reserve = ConsumerFoodReserve::calculate(
                        &economy.ledger,
                        control,
                        id,
                        parameters.day,
                    );
                    order.search = order.search
                        && living_household(&economy.ledger, &economy.actors, id)
                        && reserve.food_available
                        && lt(reserve.cash, order.purchasing_power);
                    order.purchasing_power = positive(order.purchasing_power, reserve.required);
                }
                ConsumerReserves::Capture | ConsumerReserves::Cached => {
                    order.purchasing_power = positive(
                        order.purchasing_power,
                        ConsumerFoodReserve::load(config, id),
                    );
                }
            }
            if luxury {
                order.purchasing_power = minimum(order.purchasing_power, self.budget(good));
            }
            order.search = order.search && !equal(order.purchasing_power, zero());
            order.wanted = ticks(consumers(&economy.ledger, id), 0);
            if luxury && order.search {
                order.wanted = self.durable_target(good, order.purchasing_power, comptime!(quotes));
            }
        }
    }

    fn fuel_target(&self, good: u32) -> Ticks {
        let ledger = &self.economy.ledger;
        let demand = HeatingDemand::new(ledger, self.household);
        demand.purchase_target(ledger, self.control, self.parameters, good)
    }

    fn durable_target(&self, good: u32, budget: Ticks, #[comptime] quotes: QuoteSource) -> Ticks {
        let household = self.household;
        let parameters = self.parameters;
        let control = self.control;
        let spatial = self.spatial;
        let economy = self.economy;
        let config = economy.ledger.config;
        let members = consumers(&economy.ledger, household);
        if scaled_luxury(config, good) {
            let quote = pricing::ReplacementMarket::new(spatial, control, parameters.day).quote(
                economy,
                household,
                good,
                comptime!(quotes),
            );
            let price = select(quote > 0, quote, config.maximum_price);
            let interval = config.market_table[good as usize - 1];
            let allowance = times(
                divide_fixed(budget, 180u32),
                interval + config.coverage + config.ring,
            )
            .record(control);
            maximum(divide(allowance, max(price, 1)), ticks(members, 0))
        } else {
            ticks(members, 0)
        }
    }

    fn preferred_fuel(&self, good: u32, #[comptime] quotes: QuoteSource) -> u32 {
        let household = self.household;
        let parameters = self.parameters;
        let spatial = self.spatial;
        let economy = self.economy;
        let config = economy.ledger.config;
        let cells = config.spatial_side * config.spatial_side;
        let segment = 2 * (cells + 1 + config.actors);
        let length = spatial[(segment + cells) as usize];
        let samples = pricing::QuoteSamples::new(length, parameters.day + household);
        let cash = economy.ledger.value(Field::Cash, household);
        let mut selected = good;
        let mut best = u32::new(comptime!(i64::from(u32::MAX)));
        let mut sample = samples.next(spatial, 0, comptime!(quotes));
        while sample < samples.count() {
            let rotated = samples.member(sample);
            let seller = spatial[(segment + cells + 1 + rotated) as usize];
            #[unroll]
            for output in [Good::Coal as u32, Good::Wood as u32] {
                let price = match comptime!(quotes) {
                    QuoteSource::Indexed | QuoteSource::Masked => {
                        let slot = pricing::FirmQuotes::new(cells, length).offer(rotated)
                            + u32::cast_from(output == Good::Wood as u32);
                        spatial[slot as usize]
                    }
                    QuoteSource::FromLedger => pricing::offer_price(economy, seller, output),
                };
                if price > 0 && !lt(cash, ticks(price, 0)) {
                    let trip = economy.actors.distance(household, seller);
                    let units = select(output == comptime!(Good::Coal as u32), 2u32, 1u32);
                    let days = max(ceildiv(trip, config.shipment_speed), 1);
                    let score = price * (2 / units) * (config.ring + 1) + days;
                    if trip <= config.search_radius && score < best {
                        selected = output;
                        best = score;
                    }
                }
            }
            sample = samples.next(spatial, sample + 1, comptime!(quotes));
        }
        selected
    }

    fn budget(&self, good: u32) -> Ticks {
        let ledger = &self.economy.ledger;
        let household = self.household;
        let control = self.control;
        let config = ledger.config;
        let cash = ledger.value(Field::Cash, household);
        let basic = select(
            good == comptime!(Good::LuxuryClothes as u32),
            comptime!(Good::Clothes as u32),
            select(
                good == comptime!(Good::LuxuryFurniture as u32),
                comptime!(Good::Furniture as u32),
                0u32,
            ),
        );
        if basic > 0 {
            let stock = ledger.stock_with_transit(basic, household).record(control);
            let daily = plus(
                times32(
                    food::HouseholdRation::new(ledger, household).reserve(config),
                    ledger.number(Field::FoodPrice, household),
                ),
                times32(
                    rent::household_quote(ledger, household),
                    consumers(ledger, household),
                ),
            )
            .record(control);
            let reserve = times(
                daily,
                max(
                    comptime!(Good::LUXURY_RESERVE_DAYS),
                    config.coverage + config.ring,
                ),
            )
            .record(control);
            select(
                lt(stock, ticks(consumers(ledger, household), 0)),
                zero(),
                positive(cash, reserve),
            )
        } else {
            cash
        }
    }
}

#[cube]
fn living_household(ledger: &Ledger<'_>, actors: &ActorTable<'_>, actor: u32) -> bool {
    actors.kind(actor).is(comptime!(ActorKind::Household))
        && ledger.number(Field::LivingMembers, actor) > 0
}

#[derive(CubeType)]
pub(in crate::simulation::economy) struct ConsumerFoodReserve {
    food_available: bool,
    cash: Ticks,
    required: Ticks,
}

#[cube]
impl ConsumerFoodReserve {
    fn calculate(ledger: &Ledger<'_>, control: &[Atomic<u32>], actor: u32, day: u32) -> Self {
        let config = ledger.config;
        let ration = food::HouseholdRation::new(ledger, actor);
        let reserve = ticks(
            ration.reserve(config) * (config.coverage + config.ring - 1),
            0,
        );
        let food = ledger
            .stock_with_transit(comptime!(Good::Bread as u32), actor)
            .record(control);
        let price = ledger.number(Field::FoodPrice, actor);
        let cash = times(reserve, price).record(control);
        let mut required = zero();
        if comptime!(config.food_income == crate::domain::food::FoodIncome::PurchasingPower) {
            let days = food::HomeKitchen::budget_days(ledger, actor, day);
            let wage = divide_fixed(
                ledger.value(Field::IncomeForecast, actor),
                comptime!(POLICY_SCALE),
            );
            let meals = times32(ration.minimum(), price);
            let seasonal = select(
                food::HomeKitchen::available(ledger, actor),
                times(positive(meals, wage), days).record(control),
                zero(),
            );
            required = maximum(cash, seasonal);
        }
        ConsumerFoodReserve {
            food_available: !lt(food, reserve),
            cash,
            required,
        }
    }

    pub(in crate::simulation::economy) fn capture(
        ledger: &Ledger<'_>,
        actors: &ActorTable<'_>,
        control: &[Atomic<u32>],
        actor: u32,
        day: u32,
    ) -> Ticks {
        let reserve = Self::calculate(ledger, control, actor, day);
        select(
            living_household(ledger, actors, actor) && reserve.food_available,
            reserve.required,
            ticks(u32::MAX, u32::MAX),
        )
    }

    pub(in crate::simulation::economy) fn store(
        table: &mut [u32],
        actors: u32,
        actor: u32,
        reserve: Ticks,
    ) {
        let slot = comptime!(market::farm_offers::MAILBOX_START) + actor as usize;
        table[slot] = reserve.extract(LOW_WORD);
        table[slot + actors as usize] = reserve.extract(HIGH_WORD);
    }

    fn load(config: &EconomyConfig, actor: u32) -> Ticks {
        let slot = comptime!(market::farm_offers::MAILBOX_START) + actor as usize;
        ticks(
            config.market_table[slot],
            config.market_table[slot + config.actors as usize],
        )
    }
}

#[derive(CubeType)]
pub(in crate::simulation::economy) struct HeatingDemand {
    household: u32,
    comfort: u32,
    freezing: u32,
}

#[cube]
impl HeatingDemand {
    pub fn new(ledger: &Ledger<'_>, household: u32) -> Self {
        let config = ledger.config;
        HeatingDemand {
            household,
            comfort: ledger.number(Field::HeatingNeed, household),
            freezing: ledger.number(Field::LivingMembers, household)
                * (max(config.heating_per_person, 2) - 2),
        }
    }

    pub fn reserve(&self, ledger: &Ledger<'_>, control: &[Atomic<u32>], good: u32) -> Ticks {
        self.stock_target(ledger, control, good, self.comfort)
    }

    fn purchase_target(
        &self,
        ledger: &Ledger<'_>,
        control: &[Atomic<u32>],
        parameters: &EconomyParameters,
        good: u32,
    ) -> Ticks {
        let config = ledger.config;
        let need = select(
            comptime!(config.food_income == crate::domain::food::FoodIncome::PurchasingPower)
                && parameters.matching_round + 1 < MATCHING_ROUNDS as u32,
            self.freezing,
            self.comfort,
        );
        self.stock_target(ledger, control, good, need)
    }

    fn stock_target(
        &self,
        ledger: &Ledger<'_>,
        control: &[Atomic<u32>],
        good: u32,
        need: u32,
    ) -> Ticks {
        let household = self.household;
        let config = ledger.config;
        let coal = good == comptime!(Good::Coal as u32);
        let other = select(
            coal,
            comptime!(Good::Wood as u32),
            comptime!(Good::Coal as u32),
        );
        let units = select(coal, 2u32, 1u32);
        let other_units = select(coal, 1u32, 2u32);
        let other_stock = ledger.stock_with_transit(other, household).record(control);
        let target = times32(need, config.coverage + config.ring - 1);
        let remaining = positive(
            target,
            times(minimum(other_stock, target), other_units).record(control),
        );
        divide(plus(remaining, ticks(units - 1, 0)).record(control), units)
    }

    pub fn serve(&self, ledger: &mut Ledger<'_>, control: &[Atomic<u32>]) {
        let household = self.household;
        let config = ledger.config;
        let mut need = self.comfort;
        if comptime!(config.food_income == crate::domain::food::FoodIncome::PurchasingPower) {
            let freezing = self.freezing;
            let reserve = times32(freezing, config.coverage + config.ring - 2);
            let limit = plus(reserve, ticks(need, 0)).record(control);
            let coal = minimum(
                ledger.value_at(comptime!(Good::Coal as u32), household),
                limit,
            );
            let wood = minimum(
                ledger.value_at(comptime!(Good::Wood as u32), household),
                limit,
            );
            let available = plus(times(coal, 2).record(control), wood).record(control);
            need = minimum(
                ticks(need, 0),
                maximum(ticks(freezing, 0), positive(available, reserve)),
            )
            .extract(LOW_WORD);
        }
        let mut served = 0u32;
        if need > 0 {
            let coal_stock = ledger.value_at(comptime!(Good::Coal as u32), household);
            let wood_stock = ledger.value_at(comptime!(Good::Wood as u32), household);
            let coal = minimum(coal_stock, ticks(need / 2, 0));
            let remaining = need - coal.extract(LOW_WORD) * 2;
            let wood = minimum(wood_stock, ticks(remaining, 0));
            let unmet = remaining - wood.extract(LOW_WORD);
            let additional = minimum(minus(coal_stock, coal), ticks((unmet + 1) / 2, 0));
            let burned_coal = plus(coal, additional).record(control);
            ledger.set_at(
                comptime!(Good::Coal as u32),
                household,
                minus(coal_stock, burned_coal),
            );
            ledger.set_at(
                comptime!(Good::Wood as u32),
                household,
                minus(wood_stock, wood),
            );
            ledger.increase_at(
                control,
                CONSUMPTION_FIELDS as u32 + comptime!(Good::Coal as u32) - 1,
                household,
                burned_coal,
            );
            ledger.increase_at(
                control,
                CONSUMPTION_FIELDS as u32 + comptime!(Good::Wood as u32) - 1,
                household,
                wood,
            );
            served = min(
                need,
                burned_coal.extract(LOW_WORD) * 2 + wood.extract(LOW_WORD),
            );
        }
        ledger.set_number(Field::HeatingServed, household, served);
    }
}

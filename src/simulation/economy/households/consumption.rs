use crate::simulation::economy::*;

#[derive(CubeType, Clone, Copy)]
pub(in crate::simulation::economy) struct FuelSearchRotation {
    length: u32,
    samples: u32,
    origin: u32,
    rotation: u32,
}

#[cube]
impl FuelSearchRotation {
    pub(in crate::simulation::economy) fn new(
        length: u32,
        day: u32,
        household: u32,
    ) -> FuelSearchRotation {
        let origin = day + household;
        FuelSearchRotation {
            length,
            samples: min(length, HOUSEHOLD_SEARCH_SAMPLES),
            origin,
            rotation: origin % max(length, 1),
        }
    }

    pub(super) fn samples(&self) -> u32 {
        self.samples
    }

    pub(in crate::simulation::economy) fn member(&self, sample: u32) -> u32 {
        let offset = sample * (self.length / self.samples)
            + sample * (self.length % self.samples) / self.samples;
        if self.origin <= u32::MAX - (max(self.length, 1) - 1) {
            let shifted = self.rotation + offset;
            select(shifted < self.length, shifted, shifted - self.length)
        } else {
            (offset + self.origin) % self.length
        }
    }
}

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
        let interval = max(config.consumption_intervals[good as usize - 1], 1);
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

    pub fn prepare(&self, order: &mut PurchaseRequest, good: u32) {
        let economy = self.economy;
        let parameters = self.parameters;
        let config = economy.ledger.config;
        let control = self.control;
        let id = self.household;
        let market = comptime!(parameters.market);
        let living_household = economy.actors.kind(id).is(comptime!(ActorKind::Household))
            && economy.ledger.number(Field::LivingMembers, id) > 0;
        if comptime!(matches!(
            market,
            MarketFlag::Consumer(Good::Coal | Good::Wood)
        )) {
            order.wanted = self.fuel_target(good);
            let held = economy.ledger.stock_with_transit(good, id).record(control);
            order.search = living_household
                && !lt(
                    economy.ledger.value(Field::Bread, id),
                    ticks(food::HouseholdRation::new(&economy.ledger, id).minimum(), 0),
                )
                && lt(held, order.wanted)
                && !equal(order.purchasing_power, zero());
            if order.search {
                order.search = self.preferred_fuel(good) == good;
            }
        } else {
            let luxury = comptime!(matches!(
                market,
                MarketFlag::Consumer(Good::LuxuryClothes | Good::LuxuryFurniture)
            ));
            let interval = config.consumption_intervals[good as usize - 1];
            let until_due =
                (interval - (parameters.day + id) % max(interval, 1)) % max(interval, 1);
            order.search = living_household
                && (until_due <= config.coverage + config.ring
                    || (luxury
                        && comptime!(
                            config.food_income == crate::domain::food::FoodIncome::PurchasingPower
                        )));
            let reserve = ticks(
                food::HouseholdRation::new(&economy.ledger, id).reserve(config)
                    * (config.coverage + config.ring - 1),
                0,
            );
            let food = economy
                .ledger
                .stock_with_transit(comptime!(Good::Bread as u32), id)
                .record(control);
            let cash_reserve =
                times(reserve, economy.ledger.number(Field::FoodPrice, id)).record(control);
            order.search = order.search
                && !lt(food, reserve)
                && lt(cash_reserve, economy.ledger.value(Field::Cash, id));
            if luxury {
                order.purchasing_power = self.budget(good);
            }
            order.search = order.search && !equal(order.purchasing_power, zero());
            order.wanted = ticks(consumers(&economy.ledger, id), 0);
            if luxury && order.search {
                order.wanted = self.durable_target(good, order.purchasing_power);
            }
        }
    }

    fn fuel_target(&self, good: u32) -> Ticks {
        let ledger = &self.economy.ledger;
        let demand = HeatingDemand::new(ledger, self.household);
        demand.purchase_target(ledger, self.control, self.parameters, good)
    }

    fn durable_target(&self, good: u32, budget: Ticks) -> Ticks {
        let household = self.household;
        let parameters = self.parameters;
        let control = self.control;
        let spatial = self.spatial;
        let economy = self.economy;
        let config = economy.ledger.config;
        let members = consumers(&economy.ledger, household);
        if scaled_luxury(config, good) {
            let quote = pricing::ReplacementMarket::new(spatial, control, parameters.day)
                .quote(economy, household, good);
            let price = select(quote > 0, quote, config.maximum_price);
            let interval = config.consumption_intervals[good as usize - 1];
            let allowance = times(
                divide(budget, 180),
                interval + config.coverage + config.ring,
            )
            .record(control);
            maximum(divide(allowance, max(price, 1)), ticks(members, 0))
        } else {
            ticks(members, 0)
        }
    }

    fn preferred_fuel(&self, good: u32) -> u32 {
        let household = self.household;
        let parameters = self.parameters;
        let spatial = self.spatial;
        let economy = self.economy;
        let config = economy.ledger.config;
        let cells = config.spatial_side * config.spatial_side;
        let segment = 2 * (cells + 1 + config.actors);
        let length = spatial[(segment + cells) as usize];
        let rotation = FuelSearchRotation::new(length, parameters.day, household);
        let cash = economy.ledger.value(Field::Cash, household);
        let mut selected = good;
        let mut best = u32::new(comptime!(i64::from(u32::MAX)));
        for sample in 0..rotation.samples() {
            let rotated = rotation.member(sample);
            let seller = spatial[(segment + cells + 1 + rotated) as usize];
            let output = economy.actors.kind(seller).output();
            let relevant =
                output == comptime!(Good::Coal as u32) || output == comptime!(Good::Wood as u32);
            if relevant {
                let price = economy.ledger.number(Field::Price, seller);
                let trip = economy.actors.distance(household, seller);
                let available = economy.ledger.number(Field::EnterpriseStatus, seller)
                    < comptime!(crate::domain::enterprise::EnterpriseState::Liquidating as u32)
                    && household_production::offers(economy, seller, output)
                    && trip <= config.search_radius
                    && price > 0
                    && !lt(cash, ticks(price, 0));
                let units = select(output == comptime!(Good::Coal as u32), 2u32, 1u32);
                let days = max(ceildiv(trip, config.shipment_speed), 1);
                let score = price * (2 / units) * (config.ring + 1) + days;
                if available && score < best {
                    selected = output;
                    best = score;
                }
            }
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
                times32(config.household_rent, consumers(ledger, household)),
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

use super::*;

#[derive(CubeType, Clone, Copy)]
pub(super) struct FuelSearchRotation {
    length: u32,
    samples: u32,
    origin: u32,
    rotation: u32,
}

#[cube]
impl FuelSearchRotation {
    pub(super) fn new(length: u32, day: u32, household: u32) -> FuelSearchRotation {
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

    pub(super) fn member(&self, sample: u32) -> u32 {
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
pub(super) fn scaled_luxury(config: &EconomyConfig, good: u32) -> bool {
    comptime!(config.food_income == crate::domain::food::FoodIncome::PurchasingPower)
        && (good == comptime!(Good::LuxuryClothes as u32)
            || good == comptime!(Good::LuxuryFurniture as u32))
}

#[cube]
pub(super) fn consumers(state: &mut [u32], config: &EconomyConfig, actor: u32) -> u32 {
    match comptime!(config.cohorts) {
        Feature::Enabled => number(state, config, comptime!(Field::LivingMembers as u32), actor),
        Feature::Disabled => u32::new(1),
    }
}

#[cube]
pub(super) fn durable_target(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    spatial: &[u32],
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
    household: u32,
    good: u32,
    budget: Ticks,
) -> Ticks {
    let members = consumers(state, config, household);
    if scaled_luxury(config, good) {
        let quote = pricing::replacement_quote(
            state, actors, config, spatial, control, parameters, household, good,
        );
        let price = select(quote > 0, quote, config.maximum_price);
        let interval = config.consumption_intervals[good as usize - 1];
        let allowance = multiply(
            control,
            divide(budget, 180),
            interval + config.coverage + config.ring,
        );
        maximum(divide(allowance, max(price, 1)), ticks(members, 0))
    } else {
        ticks(members, 0)
    }
}

#[cube]
pub(super) fn durable_use(
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
pub(super) fn preferred_fuel(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    spatial: &[u32],
    parameters: &EconomyParameters,
    household: u32,
    good: u32,
) -> u32 {
    let cells = config.spatial_side * config.spatial_side;
    let segment = 2 * (cells + 1 + config.actors);
    let length = spatial[(segment + cells) as usize];
    let rotation = FuelSearchRotation::new(length, parameters.day, household);
    let cash = value(state, config, comptime!(Field::Cash as u32), household);
    let mut selected = good;
    let mut best = u32::new(comptime!(i64::from(u32::MAX)));
    for sample in 0..rotation.samples() {
        let rotated = rotation.member(sample);
        let seller = spatial[(segment + cells + 1 + rotated) as usize];
        let output = kind(actors, seller).output();
        let relevant =
            output == comptime!(Good::Coal as u32) || output == comptime!(Good::Wood as u32);
        if relevant {
            let price = number(state, config, comptime!(Field::Price as u32), seller);
            let trip = distance(actors, household, seller);
            let available =
                number(
                    state,
                    config,
                    comptime!(Field::EnterpriseStatus as u32),
                    seller,
                ) < comptime!(crate::domain::enterprise::EnterpriseState::Liquidating as u32)
                    && household_production::offers(state, actors, config, seller, output)
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

#[cube]
pub(super) fn input_coverage(config: &EconomyConfig, output: u32) -> u32 {
    select(
        output == comptime!(Good::Flour as u32) || output == comptime!(Good::Bread as u32),
        config.coverage + config.ring - 1,
        config.coverage,
    )
}

#[cube]
pub(super) fn budget(
    state: &mut [u32],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
    household: u32,
    good: u32,
) -> Ticks {
    let cash = value(state, config, comptime!(Field::Cash as u32), household);
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
        let stock = add_checked(
            control,
            value(state, config, basic, household),
            value(state, config, TRANSIT_FIELDS as u32 + basic - 1, household),
        );
        let daily = add_checked(
            control,
            times32(
                number(state, config, comptime!(Field::FoodNeed as u32), household),
                number(state, config, comptime!(Field::FoodPrice as u32), household),
            ),
            times32(config.household_rent, consumers(state, config, household)),
        );
        let reserve = multiply(
            control,
            daily,
            max(
                comptime!(Good::LUXURY_RESERVE_DAYS),
                config.coverage + config.ring,
            ),
        );
        select(
            lt(stock, ticks(consumers(state, config, household), 0)),
            zero(),
            positive(cash, reserve),
        )
    } else {
        cash
    }
}

#[cube]
pub(super) fn fuel_target(
    state: &mut [u32],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
    household: u32,
    good: u32,
) -> Ticks {
    let need = number(
        state,
        config,
        comptime!(Field::HeatingNeed as u32),
        household,
    );
    fuel_stock_target(state, config, control, household, good, need)
}

#[cube]
pub(super) fn fuel_purchase_target(
    state: &mut [u32],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
    household: u32,
    good: u32,
) -> Ticks {
    let comfort = number(
        state,
        config,
        comptime!(Field::HeatingNeed as u32),
        household,
    );
    let freezing = number(
        state,
        config,
        comptime!(Field::LivingMembers as u32),
        household,
    ) * (max(config.heating_per_person, 2) - 2);
    let need = select(
        comptime!(config.food_income == crate::domain::food::FoodIncome::PurchasingPower)
            && parameters.matching_round + 1 < MATCHING_ROUNDS as u32,
        freezing,
        comfort,
    );
    fuel_stock_target(state, config, control, household, good, need)
}

#[cube]
fn fuel_stock_target(
    state: &mut [u32],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
    household: u32,
    good: u32,
    need: u32,
) -> Ticks {
    let coal = good == comptime!(Good::Coal as u32);
    let other = select(
        coal,
        comptime!(Good::Wood as u32),
        comptime!(Good::Coal as u32),
    );
    let units = select(coal, 2u32, 1u32);
    let other_units = select(coal, 1u32, 2u32);
    let other_stock = add_checked(
        control,
        value(state, config, other, household),
        value(state, config, TRANSIT_FIELDS as u32 + other - 1, household),
    );
    let target = times32(need, config.coverage + config.ring - 1);
    let remaining = positive(
        target,
        multiply(control, minimum(other_stock, target), other_units),
    );
    divide(add_checked(control, remaining, ticks(units - 1, 0)), units)
}

#[cube]
pub(super) fn heat(
    state: &mut [u32],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
    household: u32,
) {
    let mut need = number(
        state,
        config,
        comptime!(Field::HeatingNeed as u32),
        household,
    );
    if comptime!(config.food_income == crate::domain::food::FoodIncome::PurchasingPower) {
        let freezing = number(
            state,
            config,
            comptime!(Field::LivingMembers as u32),
            household,
        ) * (max(config.heating_per_person, 2) - 2);
        let reserve = times32(freezing, config.coverage + config.ring - 2);
        let limit = add_checked(control, reserve, ticks(need, 0));
        let coal = minimum(
            value(state, config, comptime!(Good::Coal as u32), household),
            limit,
        );
        let wood = minimum(
            value(state, config, comptime!(Good::Wood as u32), household),
            limit,
        );
        let available = add_checked(control, multiply(control, coal, 2), wood);
        need = minimum(
            ticks(need, 0),
            maximum(ticks(freezing, 0), positive(available, reserve)),
        )
        .extract(LOW_WORD);
    }
    let mut served = 0u32;
    if need > 0 {
        let coal_stock = value(state, config, comptime!(Good::Coal as u32), household);
        let wood_stock = value(state, config, comptime!(Good::Wood as u32), household);
        let coal = minimum(coal_stock, ticks(need / 2, 0));
        let remaining = need - coal.extract(LOW_WORD) * 2;
        let wood = minimum(wood_stock, ticks(remaining, 0));
        let unmet = remaining - wood.extract(LOW_WORD);
        let additional = minimum(minus(coal_stock, coal), ticks((unmet + 1) / 2, 0));
        let burned_coal = add_checked(control, coal, additional);
        set(
            state,
            config,
            comptime!(Good::Coal as u32),
            household,
            minus(coal_stock, burned_coal),
        );
        set(
            state,
            config,
            comptime!(Good::Wood as u32),
            household,
            minus(wood_stock, wood),
        );
        increase(
            state,
            config,
            control,
            CONSUMPTION_FIELDS as u32 + comptime!(Good::Coal as u32) - 1,
            household,
            burned_coal,
        );
        increase(
            state,
            config,
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
    set32(
        state,
        config,
        comptime!(Field::HeatingServed as u32),
        household,
        served,
    );
}

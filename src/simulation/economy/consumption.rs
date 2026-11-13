use super::*;

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
    let mut available = cash;
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
                number(state, config, comptime!(Field::Price as u32), household),
            ),
            ticks(config.household_rent, 0),
        );
        let reserve = multiply(
            control,
            daily,
            max(
                comptime!(Good::LUXURY_RESERVE_DAYS),
                config.coverage + config.ring,
            ),
        );
        available = select(equal(stock, zero()), zero(), positive(cash, reserve));
    }
    available
}

#[cube]
pub(super) fn fuel_target(
    state: &mut [u32],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
    household: u32,
    good: u32,
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
    let target = times32(
        number(
            state,
            config,
            comptime!(Field::HeatingNeed as u32),
            household,
        ),
        config.coverage + config.ring - 1,
    );
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
    let need = number(
        state,
        config,
        comptime!(Field::HeatingNeed as u32),
        household,
    );
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

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

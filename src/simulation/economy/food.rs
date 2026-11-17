use super::*;

#[repr(u32)]
enum BreadTier {
    Low,
    Medium,
    Rich,
}

#[derive(CubeType)]
pub(super) struct Conversion {
    input: u32,
    output: u32,
}

#[derive(CubeType)]
pub(super) struct ConvertedFood {
    pub quantity: Ticks,
    pub remainder: u32,
}

#[derive(CubeType)]
pub(super) struct HomeFoodWork {
    pub grain: Ticks,
    pub minutes: u32,
}

#[derive(CubeType)]
pub(in crate::simulation) struct BreadDemand {
    pub gross_income: Ticks,
    pub expected_income: Ticks,
    pub savings: Ticks,
    pub minimum: u32,
    pub members: u32,
    pub price: u32,
}

#[cube]
pub(super) fn minimum_daily(state: &mut [u32], config: &EconomyConfig, household: u32) -> u32 {
    let minimum = number(
        state,
        config,
        comptime!(Field::NutritionNeed as u32),
        household,
    );
    select(
        minimum > 0,
        minimum,
        number(state, config, comptime!(Field::FoodNeed as u32), household),
    )
}

#[cube]
pub(super) fn purchase_daily(
    state: &mut [u32],
    config: &EconomyConfig,
    parameters: &EconomyParameters,
    household: u32,
) -> u32 {
    let desired = number(state, config, comptime!(Field::FoodNeed as u32), household);
    let minimum = minimum_daily(state, config, household);
    select(
        comptime!(config.food_income == crate::domain::food::FoodIncome::PurchasingPower)
            && parameters.matching_round + 1 < MATCHING_ROUNDS as u32,
        select(minimum > 0, minimum, desired),
        desired,
    )
}

#[cube]
pub(super) fn consumption_reserve(
    state: &mut [u32],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
    household: u32,
    day: u32,
) -> Ticks {
    let ration = number(
        state,
        config,
        comptime!(Field::NutritionNeed as u32),
        household,
    );
    let mut reserve = times32(ration, config.coverage + config.ring - 2);
    if comptime!(config.food_income == crate::domain::food::FoodIncome::PurchasingPower)
        && number(state, config, comptime!(Field::LandPlots as u32), household) > 0
    {
        let days = 364 - (day % 365 + 365 - config.home.harvest_start) % 365;
        let requirement = times32(ration, days);
        let grain = minimum(
            value(state, config, comptime!(Field::Grain as u32), household),
            requirement,
        );
        let milling = Conversion::milling(config).process(
            control,
            grain,
            number(
                state,
                config,
                comptime!(Field::MillingRemainder as u32),
                household,
            ),
        );
        let flour = add_checked(
            control,
            milling.quantity,
            minimum(
                value(state, config, comptime!(Field::Flour as u32), household),
                requirement,
            ),
        );
        let baking = Conversion::baking(config).process(
            control,
            flour,
            number(
                state,
                config,
                comptime!(Field::BakingRemainder as u32),
                household,
            ),
        );
        reserve = maximum(reserve, positive(requirement, baking.quantity));
    }
    reserve
}

#[cube]
pub(super) fn household_work(
    state: &mut [u32],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
    household: u32,
    missing: u32,
) -> HomeFoodWork {
    let flour = Conversion::baking(config).required(
        control,
        ticks(missing, 0),
        number(
            state,
            config,
            comptime!(Field::BakingRemainder as u32),
            household,
        ),
    );
    let grain = Conversion::milling(config).required(
        control,
        flour,
        number(
            state,
            config,
            comptime!(Field::MillingRemainder as u32),
            household,
        ),
    );
    HomeFoodWork {
        grain,
        minutes: grain.extract(LOW_WORD) * config.home.milling_minutes
            + flour.extract(LOW_WORD) * config.home.baking_minutes,
    }
}

#[cube]
impl Conversion {
    pub fn milling(config: &EconomyConfig) -> Self {
        Conversion {
            input: select(config.wheat_kg_per_hectare > 0, 5u32, 1u32),
            output: select(config.wheat_kg_per_hectare > 0, 4u32, 1u32),
        }
    }

    pub fn baking(config: &EconomyConfig) -> Self {
        Conversion {
            input: select(config.wheat_kg_per_hectare > 0, 25u32, 1u32),
            output: select(config.wheat_kg_per_hectare > 0, 64u32, 1u32),
        }
    }

    pub fn required(&self, control: &[Atomic<u32>], quantity: Ticks, remainder: u32) -> Ticks {
        let missing = positive(multiply(control, quantity, self.input), ticks(remainder, 0));
        divide(
            add_checked(control, missing, ticks(self.output - 1, 0)),
            self.output,
        )
    }

    pub fn process(
        &self,
        control: &[Atomic<u32>],
        quantity: Ticks,
        remainder: u32,
    ) -> ConvertedFood {
        let total = add_checked(
            control,
            multiply(control, quantity, self.output),
            ticks(remainder, 0),
        );
        let produced = divide(total, self.input);
        ConvertedFood {
            quantity: produced,
            remainder: minus(total, multiply(control, produced, self.input)).extract(LOW_WORD),
        }
    }
}

#[cube]
pub(in crate::simulation) fn bread_target(config: &EconomyConfig, request: BreadDemand) -> u32 {
    let gross = select(
        comptime!(config.food_income == crate::domain::food::FoodIncome::PurchasingPower),
        plus(
            divide(request.expected_income, POLICY_SCALE),
            divide(request.savings, 90),
        )
        .value,
        request.gross_income,
    );
    let salary =
        (u64::cast_from(gross.extract(HIGH_WORD)) << 32) | u64::cast_from(gross.extract(LOW_WORD));
    let tier = select(
        salary >= u64::cast_from(config.food_rich_salary),
        comptime!(BreadTier::Rich as u32),
        select(
            salary >= u64::cast_from(config.food_medium_salary),
            comptime!(BreadTier::Medium as u32),
            comptime!(BreadTier::Low as u32),
        ),
    );
    let portions = match tier {
        0 => 1u32,
        1 => 2u32,
        _ => 10u32,
    };
    let nominal = request.members * config.household_need * portions;
    let affordable = (salary / 5) * 4 / u64::cast_from(max(request.price, 1));
    let purchasing = max(
        request.minimum,
        min(
            affordable,
            u64::cast_from(request.members * config.household_need * 10),
        ) as u32,
    );
    select(
        config.food_medium_salary > 0,
        select(
            comptime!(config.food_income == crate::domain::food::FoodIncome::PurchasingPower),
            purchasing,
            nominal,
        ),
        request.minimum,
    )
}

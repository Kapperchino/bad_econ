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
pub(super) fn minimum_daily(ledger: &Ledger<'_>, household: u32) -> u32 {
    let minimum = ledger.number(Field::NutritionNeed, household);
    select(
        minimum > 0,
        minimum,
        ledger.number(Field::FoodNeed, household),
    )
}

#[cube]
pub(super) fn reserve_daily(ledger: &Ledger<'_>, household: u32) -> u32 {
    let config = ledger.config;
    match comptime!(config.food_income) {
        crate::domain::food::FoodIncome::PurchasingPower => minimum_daily(ledger, household),
        crate::domain::food::FoodIncome::Nominal => ledger.number(Field::FoodNeed, household),
    }
}

#[cube]
pub(super) fn purchase_daily(
    ledger: &Ledger<'_>,
    parameters: &EconomyParameters,
    household: u32,
) -> u32 {
    let config = ledger.config;
    let desired = ledger.number(Field::FoodNeed, household);
    let minimum = minimum_daily(ledger, household);
    select(
        comptime!(config.food_income == crate::domain::food::FoodIncome::PurchasingPower)
            && parameters.matching_round + 1 < MATCHING_ROUNDS as u32,
        select(minimum > 0, minimum, desired),
        desired,
    )
}

#[cube]
pub(super) fn consumption_reserve(
    ledger: &Ledger<'_>,
    control: &[Atomic<u32>],
    household: u32,
    day: u32,
) -> Ticks {
    let config = ledger.config;
    let ration = ledger.number(Field::NutritionNeed, household);
    let mut reserve = times32(ration, config.coverage + config.ring - 2);
    if comptime!(config.food_income == crate::domain::food::FoodIncome::PurchasingPower)
        && ledger.number(Field::LandPlots, household) > 0
    {
        let days = 364 - (day % 365 + 365 - config.home.harvest_start) % 365;
        let requirement = times32(ration, days);
        let grain = minimum(ledger.value(Field::Grain, household), requirement);
        let milling = Conversion::milling(config).process(
            control,
            grain,
            ledger.number(Field::MillingRemainder, household),
        );
        let flour = add_checked(
            control,
            milling.quantity,
            minimum(ledger.value(Field::Flour, household), requirement),
        );
        let baking = Conversion::baking(config).process(
            control,
            flour,
            ledger.number(Field::BakingRemainder, household),
        );
        reserve = maximum(reserve, positive(requirement, baking.quantity));
    }
    reserve
}

#[cube]
pub(super) fn household_work(
    ledger: &Ledger<'_>,
    control: &[Atomic<u32>],
    household: u32,
    missing: u32,
) -> HomeFoodWork {
    let config = ledger.config;
    let flour = Conversion::baking(config).required(
        control,
        ticks(missing, 0),
        ledger.number(Field::BakingRemainder, household),
    );
    let grain = Conversion::milling(config).required(
        control,
        flour,
        ledger.number(Field::MillingRemainder, household),
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
    let per_capita = salary
        / u64::cast_from(select(
            comptime!(config.cohorts == Feature::Enabled),
            max(request.members, 1),
            1,
        ));
    let tier = select(
        per_capita >= u64::cast_from(config.food_rich_salary),
        comptime!(BreadTier::Rich as u32),
        select(
            per_capita >= u64::cast_from(config.food_medium_salary),
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

use crate::simulation::economy::*;

#[repr(u32)]
enum BreadTier {
    Low,
    Medium,
    Rich,
}

#[derive(CubeType)]
struct Conversion {
    input: u32,
    output: u32,
}

#[derive(CubeType)]
pub(in crate::simulation::economy) struct ConvertedFood {
    pub quantity: Ticks,
    pub remainder: u32,
}

#[derive(CubeType)]
pub(in crate::simulation::economy) struct HomeFoodWork {
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

#[derive(CubeType)]
pub(in crate::simulation::economy) struct HouseholdRation {
    desired: u32,
    nutrition: u32,
}

#[cube]
impl HouseholdRation {
    pub fn new(ledger: &Ledger<'_>, household: u32) -> Self {
        HouseholdRation {
            desired: ledger.number(Field::FoodNeed, household),
            nutrition: ledger.number(Field::NutritionNeed, household),
        }
    }

    pub fn minimum(&self) -> u32 {
        select(self.nutrition > 0, self.nutrition, self.desired)
    }

    pub fn reserve(&self, config: &EconomyConfig) -> u32 {
        match comptime!(config.food_income) {
            crate::domain::food::FoodIncome::PurchasingPower => self.minimum(),
            crate::domain::food::FoodIncome::Nominal => self.desired,
        }
    }

    pub fn purchase(&self, config: &EconomyConfig, parameters: &EconomyParameters) -> u32 {
        select(
            comptime!(config.food_income == crate::domain::food::FoodIncome::PurchasingPower)
                && parameters.matching_round + 1 < MATCHING_ROUNDS as u32,
            self.minimum(),
            self.desired,
        )
    }

    pub fn consumption_reserve(
        &self,
        ledger: &Ledger<'_>,
        control: &[Atomic<u32>],
        household: u32,
        day: u32,
    ) -> Ticks {
        let config = ledger.config;
        let mut reserve = times32(self.nutrition, config.coverage + config.ring - 2);
        if comptime!(config.food_income == crate::domain::food::FoodIncome::PurchasingPower)
            && ledger.number(Field::LandPlots, household) > 0
        {
            let day = farming::CropCalendar::new(config).on(day);
            let requirement = times32(self.nutrition, day.days_to_harvest() - 1);
            let grain = minimum(ledger.value(Field::Grain, household), requirement);
            let kitchen = HomeKitchen::new(ledger, household);
            let milling = kitchen.mill(control, grain);
            let flour = plus(
                milling.quantity,
                minimum(ledger.value(Field::Flour, household), requirement),
            )
            .record(control);
            let baking = kitchen.bake(control, flour);
            reserve = maximum(reserve, positive(requirement, baking.quantity));
        }
        reserve
    }
}

#[derive(CubeType)]
pub(in crate::simulation::economy) struct HomeKitchen {
    milling: Conversion,
    baking: Conversion,
    milling_remainder: u32,
    baking_remainder: u32,
}

#[cube]
impl HomeKitchen {
    pub fn new(ledger: &Ledger<'_>, household: u32) -> Self {
        let config = ledger.config;
        HomeKitchen {
            milling: Conversion::milling(config),
            baking: Conversion::baking(config),
            milling_remainder: ledger.number(Field::MillingRemainder, household),
            baking_remainder: ledger.number(Field::BakingRemainder, household),
        }
    }

    pub fn flour_for_bread(&self, control: &[Atomic<u32>], bread: Ticks) -> Ticks {
        self.baking.required(control, bread, self.baking_remainder)
    }

    pub fn grain_for_flour(&self, control: &[Atomic<u32>], flour: Ticks) -> Ticks {
        self.milling
            .required(control, flour, self.milling_remainder)
    }

    pub fn mill(&self, control: &[Atomic<u32>], grain: Ticks) -> ConvertedFood {
        self.milling.process(control, grain, self.milling_remainder)
    }

    pub fn bake(&self, control: &[Atomic<u32>], flour: Ticks) -> ConvertedFood {
        self.baking.process(control, flour, self.baking_remainder)
    }

    pub fn work(
        &self,
        config: &EconomyConfig,
        control: &[Atomic<u32>],
        missing: u32,
    ) -> HomeFoodWork {
        let flour = self.flour_for_bread(control, ticks(missing, 0));
        let grain = self.grain_for_flour(control, flour);
        HomeFoodWork {
            grain,
            minutes: grain.extract(LOW_WORD) * config.home.milling_minutes
                + flour.extract(LOW_WORD) * config.home.baking_minutes,
        }
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
        let missing = positive(
            times(quantity, self.input).record(control),
            ticks(remainder, 0),
        );
        divide(
            plus(missing, ticks(self.output - 1, 0)).record(control),
            self.output,
        )
    }

    pub fn process(
        &self,
        control: &[Atomic<u32>],
        quantity: Ticks,
        remainder: u32,
    ) -> ConvertedFood {
        let total = plus(
            times(quantity, self.output).record(control),
            ticks(remainder, 0),
        )
        .record(control);
        let produced = divide(total, self.input);
        ConvertedFood {
            quantity: produced,
            remainder: minus(total, times(produced, self.input).record(control)).extract(LOW_WORD),
        }
    }
}

#[cube]
impl BreadDemand {
    pub fn target(&self, config: &EconomyConfig) -> u32 {
        let gross = select(
            comptime!(config.food_income == crate::domain::food::FoodIncome::PurchasingPower),
            plus(
                divide(self.expected_income, POLICY_SCALE),
                divide(self.savings, 90),
            )
            .value,
            self.gross_income,
        );
        let salary = native(gross);
        let per_capita = salary
            / u64::cast_from(select(
                comptime!(config.cohorts == Feature::Enabled),
                max(self.members, 1),
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
        let nominal = self.members * config.household_need * portions;
        let affordable = (salary / 5) * 4 / u64::cast_from(max(self.price, 1));
        let purchasing = max(
            self.minimum,
            min(
                affordable,
                u64::cast_from(self.members * config.household_need * 10),
            ) as u32,
        );
        select(
            config.food_medium_salary > 0,
            select(
                comptime!(config.food_income == crate::domain::food::FoodIncome::PurchasingPower),
                purchasing,
                nominal,
            ),
            self.minimum,
        )
    }
}

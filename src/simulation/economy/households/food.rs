use crate::simulation::economy::*;

#[repr(u32)]
enum BreadTier {
    Low,
    Medium,
    Rich,
}

#[derive(CubeType)]
pub(in crate::simulation::economy) struct ConvertedFood {
    pub quantity: Ticks,
    pub remainder: u32,
}

#[derive(CubeType)]
pub(in crate::simulation) struct BreadDemand {
    pub gross_income: Ticks,
    pub expected_income: Ticks,
    pub daily_savings: Ticks,
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

    pub fn consumption_reserve(&self, config: &EconomyConfig) -> Ticks {
        times32(self.nutrition, config.coverage + config.ring - 2)
    }
}

#[derive(CubeType)]
pub(in crate::simulation) struct HomeKitchen {
    flour_units: u32,
    bread_units: u32,
    remainder: u32,
}

#[cube]
impl HomeKitchen {
    pub fn available(ledger: &Ledger<'_>, household: u32) -> bool {
        comptime!(ledger.config.home_mode == Feature::Enabled)
            && ledger.number(Field::LandPlots, household) > 0
    }

    pub fn budget_days(ledger: &Ledger<'_>, household: u32, day: u32) -> u32 {
        let seasonal = Self::available(ledger, household) && ledger.config.home.harvest_days < 365;
        select(
            seasonal,
            max(
                90,
                farming::CropCalendar::new(ledger.config)
                    .on(day)
                    .days_to_harvest(),
            ),
            90,
        )
    }

    pub fn flour_capacity(ledger: &Ledger<'_>, household: u32, days: u32) -> Ticks {
        let budget = ledger.number(Field::HouseholdBudget, household);
        let paid = ledger.number(Field::Hours, household);
        times32(
            (max(budget, paid) - paid) / max(ledger.config.home.baking_minutes, 1),
            days,
        )
    }

    pub fn new(ledger: &Ledger<'_>, household: u32) -> Self {
        let calibrated = ledger.config.wheat_kg_per_hectare > 0;
        HomeKitchen {
            flour_units: select(calibrated, 25u32, 1u32),
            bread_units: select(calibrated, 64u32, 1u32),
            remainder: ledger.number(Field::BakingRemainder, household),
        }
    }

    pub fn flour_for_bread(&self, control: &[Atomic<u32>], bread: Ticks) -> Ticks {
        let missing = positive(
            times(bread, self.flour_units).record(control),
            ticks(self.remainder, 0),
        );
        divide(
            plus(missing, ticks(self.bread_units - 1, 0)).record(control),
            self.bread_units,
        )
    }

    fn scaled_bread(&self, flour: Ticks) -> CheckedTicks {
        let scaled = times(flour, self.bread_units);
        let mut total = plus(scaled.value, ticks(self.remainder, 0));
        total.faults.insert(scaled.faults);
        total
    }

    pub fn bread_from_flour(&self, flour: Ticks) -> Ticks {
        divide(self.scaled_bread(flour).value, self.flour_units)
    }

    pub(in crate::simulation::economy) fn bake(
        &self,
        control: &[Atomic<u32>],
        flour: Ticks,
    ) -> ConvertedFood {
        let total = self.scaled_bread(flour).record(control);
        let quantity = divide(total, self.flour_units);
        ConvertedFood {
            quantity,
            remainder: minus(total, times(quantity, self.flour_units).record(control))
                .extract(LOW_WORD),
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
                self.daily_savings,
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
            max(self.minimum, self.members * config.household_need),
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

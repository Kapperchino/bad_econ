use super::{
    Economy, Good, ScenarioConfig,
    enterprise::{Distributor, HouseholdProduction},
    food,
};

#[derive(Clone, Copy)]
enum FoodPricing {
    Production,
    Configured,
}

struct ProductionCycle {
    work_days: u64,
    output_days: u64,
}

impl ProductionCycle {
    fn new(config: &ScenarioConfig, good: Good) -> Self {
        match (good, config.household_production) {
            (
                Good::Grain | Good::Flax | Good::Cotton | Good::Dye,
                HouseholdProduction::Available {
                    harvest_days: harvest_days @ 1..=364,
                    ..
                },
            ) => Self {
                work_days: u64::from(match (good, config.food) {
                    (Good::Grain, Some(food)) => {
                        food.commercial_wheat.paid_days(211.min(365 - harvest_days))
                    }
                    _ => 211.min(365 - harvest_days),
                }),
                output_days: 365,
            },
            _ => Self {
                work_days: 1,
                output_days: 1,
            },
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct StartingPrices {
    goods: [u32; Good::COUNT],
}

impl StartingPrices {
    pub fn production(config: &ScenarioConfig) -> Self {
        Self {
            goods: Good::ALL.map(|good| price(config, good, FoodPricing::Production)),
        }
    }

    pub fn configured(config: &ScenarioConfig) -> Self {
        Self {
            goods: Good::ALL.map(|good| price(config, good, FoodPricing::Configured)),
        }
    }

    pub fn good(&self, good: Good) -> u32 {
        self.goods[good as usize - 1]
    }

    pub fn food(&self) -> [u32; 3] {
        [
            self.good(Good::Grain),
            self.good(Good::Flour),
            self.good(Good::Bread),
        ]
    }

    pub fn retail(&self, distributor: &Distributor, config: &ScenarioConfig) -> u32 {
        let units = u64::from(distributor.units_per_shift);
        let costs = u64::from(self.good(distributor.good)) * units
            + u64::from(config.wage) * u64::from(distributor.handling_minutes);
        (costs * u64::from(1000 + distributor.margin_per_mille))
            .div_ceil(units * 1000)
            .clamp(
                u64::from(config.minimum_price),
                u64::from(config.maximum_price),
            ) as u32
    }

    pub fn household_daily(&self, config: &ScenarioConfig) -> u64 {
        let members = config.demography.initial_household_members;
        let heat = members * super::calendar::Weather::default().heating_units_per_person;
        let bread = config
            .distributors
            .iter()
            .filter(|distributor| distributor.good == Good::Bread)
            .map(|distributor| self.retail(distributor, config))
            .max()
            .unwrap_or(self.good(Good::Bread));
        u64::from(config.household_need) * u64::from(members) * u64::from(bread)
            + u64::from(heat.div_ceil(2)) * u64::from(self.good(Good::Coal))
            + u64::from(config.rent.map_or(0, |rent| rent.household_daily))
    }
}

fn price(config: &ScenarioConfig, good: Good, food_pricing: FoodPricing) -> u32 {
    match (config.economy, good, food_pricing) {
        (_, Good::Grain | Good::Flour | Good::Bread, FoodPricing::Configured) => {
            config.prices[good as usize - 1]
        }
        (_, Good::Housing, _) => config.rent.map_or(1, |rent| rent.household_daily),
        (_, Good::Railway, _) => good.initial_price(),
        (Economy::Agrarian, _, _) => good.initial_price(),
        (Economy::Industrial, _, _) => {
            let technique = food::technique(config, good);
            let cycle = ProductionCycle::new(config, good);
            let labor = u64::from(config.wage) * u64::from(technique.labor_minutes);
            let inputs = good
                .inputs()
                .into_iter()
                .flatten()
                .map(|input| {
                    u64::from(price(config, input, food_pricing)) * u64::from(technique.input_ticks)
                })
                .sum::<u64>();
            ((labor + inputs) * cycle.work_days * 5)
                .div_ceil(u64::from(technique.output_ticks) * cycle.output_days * 4)
                .min(u64::from(config.maximum_price)) as u32
        }
    }
    .clamp(config.minimum_price, config.maximum_price)
}

use super::{
    ActorRows, Economy, Field, Good, ScenarioConfig, State, calendar,
    enterprise::{Distributor, HouseholdProduction},
    food, railway,
    scenario::{generated_firm_identity, generated_good},
};
use crate::arithmetic::add;
use anyhow::Result;
use std::collections::BTreeMap;

pub(crate) struct SupplierCacheLayout {
    pub extra_words: usize,
}

impl SupplierCacheLayout {
    pub const INLINE_WORDS: usize = 8;
    pub const SUPPLIER_LIMIT: usize = 2048;

    pub fn new(actors: &ActorRows) -> Self {
        let mut producers = [0usize; Good::COUNT];
        for firm in actors.firm_indices() {
            producers[actors.at(firm).kind as usize - 1] += 1;
        }
        let largest = Good::ALL
            .into_iter()
            .filter(|good| {
                matches!(good, Good::Coal | Good::Wood) || good.consumption_interval() > 1
            })
            .map(|good| producers[good as usize - 1])
            .max()
            .unwrap_or(0);
        Self {
            extra_words: largest
                .min(Self::SUPPLIER_LIMIT)
                .div_ceil(u32::BITS as usize)
                .saturating_sub(Self::INLINE_WORDS),
        }
    }
}

pub(super) struct StartingReserves {
    household_stock: u64,
    coal: u64,
    wood: u64,
    days: u32,
    buffer_days: u32,
    food: Option<super::food::FoodPolicy>,
}

impl StartingReserves {
    pub fn new(config: &ScenarioConfig) -> Self {
        let delivery_days = config.search_radius.div_ceil(config.shipment_speed).max(1);
        let days = match config.buffer_days {
            0 => 0,
            _ => config.buffer_days.max(config.coverage + delivery_days),
        };
        let heat = config.demography.initial_household_members
            * super::calendar::Weather::default().heating_units_per_person;
        Self {
            household_stock: u64::from(config.household_need)
                * u64::from(config.demography.initial_household_members)
                * u64::from(days)
                * config.food.map_or(1, |food| food.reserve_portions(config)),
            coal: u64::from(heat / 2) * u64::from(days),
            wood: u64::from(heat % 2) * u64::from(days),
            days,
            buffer_days: config.buffer_days,
            food: config.food,
        }
    }

    pub fn household(&self, members: u64) -> [StartingStock; 3] {
        [
            StartingStock {
                good: Good::Coal,
                units: self.coal * members,
            },
            StartingStock {
                good: Good::Wood,
                units: self.wood * members,
            },
            StartingStock {
                good: Good::Bread,
                units: self.household_stock * members,
            },
        ]
    }

    fn inputs(&self, output: Good, daily: u64) -> u64 {
        let days = match output {
            Good::Flour | Good::Bread => self.days,
            _ => self.buffer_days,
        };
        let quantity = daily * u64::from(days);
        self.food
            .map_or(quantity, |food| food.inputs(output, quantity))
    }
}

pub(super) struct StartingStock {
    good: Good,
    units: u64,
}

impl StartingStock {
    pub fn add_to(&self, goods: &mut [u64; Good::COUNT]) -> Result<()> {
        let total = &mut goods[self.good as usize - 1];
        *total = add(*total, self.units)?;
        Ok(())
    }

    pub fn seed(&self, state: &mut State, actor: usize) -> Result<()> {
        state.set(Field::inventory(self.good), actor, self.units);
        self.add_to(&mut state.initial_goods)
    }
}

pub(super) struct StartingFarm {
    pub plots: u64,
    pub inventory: [StartingStock; 2],
}

impl StartingFarm {
    pub fn new(config: &ScenarioConfig, farmers: u32) -> Option<Self> {
        match config.household_production {
            HouseholdProduction::Available {
                annual_yield_per_plot,
                ..
            } => {
                let farmers = u64::from(farmers);
                let plots =
                    farmers * u64::from(config.demography.initial_household_members.div_ceil(2));
                Some(Self {
                    plots,
                    inventory: [
                        StartingStock {
                            good: Good::Grain,
                            units: u64::from(annual_yield_per_plot) * plots,
                        },
                        StartingStock {
                            good: Good::Tools,
                            units: farmers,
                        },
                    ],
                })
            }
            HouseholdProduction::Unavailable => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct StartingProduction {
    pub capacities: [u32; Good::COUNT],
    daily: [u64; Good::COUNT],
    grain_per_dealer: u64,
    stocks: [u64; Good::COUNT],
}

impl StartingProduction {
    pub fn new(config: &ScenarioConfig, retailers: &BTreeMap<u64, Distributor>) -> Self {
        let mut firm_counts = [0u32; Good::COUNT];
        for firm in 0..config.firms as usize {
            if !retailers.contains_key(&(generated_firm_identity(config, firm))) {
                firm_counts[generated_good(config, firm) as usize - 1] += 1;
            }
        }
        let demand = StartingDemand::new(config);
        let grain_dealers = retailers
            .values()
            .filter(|store| store.good == Good::Grain)
            .count() as u64;
        let household_grain = match grain_dealers {
            0 => 0,
            _ => demand.household_grain,
        };
        let daily = Good::ALL.map(|good| {
            let required = demand.daily[good as usize - 1];
            match good {
                Good::Grain => required.saturating_sub(household_grain),
                _ => required,
            }
            .div_ceil(u64::from(firm_counts[good as usize - 1].max(1)))
            .max(1)
        });
        let grain_per_dealer = match household_grain {
            0 => daily[Good::Grain as usize - 1],
            _ => household_grain.div_ceil(grain_dealers),
        };
        let production = Good::ALL.map(|good| {
            match good {
                Good::Grain => demand
                    .production(good)
                    .saturating_sub((household_grain * 5).div_ceil(4)),
                _ => demand.production(good),
            }
            .div_ceil(u64::from(firm_counts[good as usize - 1].max(1)))
            .max(1)
        });
        let capacities = std::array::from_fn(|good| match (Good::ALL[good], config.economy) {
            (Good::Housing, _) => 0,
            (Good::Railway, _) => railway::DEFAULT_CREW_CAPACITY,
            (_, Economy::Agrarian) => config.capacity,
            (_, Economy::Industrial) => (production[good]
                .div_ceil(food::technique(config, Good::ALL[good]).output_ticks.into())
                as u32)
                .max(20)
                .min(config.capacity),
        });
        let stocks = std::array::from_fn(|good| match (Good::ALL[good], config.economy) {
            (Good::Railway, _) => 0,
            (Good::Grain | Good::Flax | Good::Cotton | Good::Dye, _) => {
                production[good]
                    * match config.buffer_days {
                        0 => 0,
                        _ => u64::from(calendar::DAYS_PER_YEAR),
                    }
            }
            (_, Economy::Agrarian) => {
                u64::from(config.capacity)
                    * u64::from(config.packet_output)
                    * u64::from(config.buffer_days)
            }
            (_, Economy::Industrial) => {
                daily[good]
                    * u64::from(
                        config
                            .producer_buffer_days
                            .unwrap_or(config.buffer_days)
                            .min(config.coverage.saturating_sub(1)),
                    )
            }
        });
        Self {
            capacities,
            daily,
            grain_per_dealer,
            stocks,
        }
    }

    pub fn daily(&self, good: Good, retailer: Option<&Distributor>) -> u64 {
        match (good, retailer) {
            (Good::Grain, Some(_)) => self.grain_per_dealer,
            _ => self.daily[good as usize - 1],
        }
    }

    pub fn retail_capacity(&self, config: &ScenarioConfig, store: &Distributor) -> u32 {
        let days = match (store.good, config.household_production) {
            (Good::Grain, HouseholdProduction::Available { harvest_days, .. })
                if config.version >= 5 =>
            {
                store.reserve_days.div_ceil(harvest_days)
            }
            _ => 1,
        };
        (self.daily(store.good, Some(store)) * u64::from(days))
            .div_ceil(u64::from(store.units_per_shift))
            .clamp(1, u64::from(config.capacity)) as u32
    }

    pub fn working_cash(
        &self,
        config: &ScenarioConfig,
        retailer: Option<&Distributor>,
    ) -> Result<u64> {
        match retailer {
            Some(store) if store.good == Good::Grain && config.version >= 5 => {
                let stock = u128::from(self.grain_per_dealer) * u128::from(store.reserve_days);
                Ok(config
                    .firm_cash
                    .max(u64::try_from(stock * u128::from(config.prices[0]))?))
            }
            _ => Ok(config.firm_cash),
        }
    }

    pub fn inventory<'a>(
        &self,
        config: &ScenarioConfig,
        reserves: &'a StartingReserves,
        output: Good,
        retailer: Option<&Distributor>,
    ) -> impl Iterator<Item = StartingStock> + use<'a> {
        let slot = output as usize - 1;
        let daily = self.daily(output, retailer);
        let economy = config.economy;
        let construction_stock = config.version < 6;
        let units = match retailer {
            Some(store) => daily * u64::from(config.buffer_days.min(store.reserve_days)),
            None => self.stocks[slot],
        };
        let inputs = match retailer {
            Some(_) => [None; 2],
            None => output.inputs(),
        };
        std::iter::once(StartingStock {
            good: output,
            units,
        })
        .chain(inputs.into_iter().flatten().map(move |good| StartingStock {
            good,
            units: match economy {
                Economy::Agrarian => units,
                Economy::Industrial => reserves.inputs(output, daily),
            },
        }))
        .filter(move |stock| construction_stock || stock.good != Good::Construction)
    }
}

pub(super) struct StartingDemand {
    pub daily: [u64; Good::COUNT],
    food: u64,
    household_grain: u64,
    calibration: Option<super::food::FoodPolicy>,
}

impl StartingDemand {
    pub fn new(config: &ScenarioConfig) -> Self {
        let households = u64::from(config.population_size());
        let members = config.demography.initial_household_members;
        let adults = members.min(2);
        let dependents = members - adults;
        let requirements = u64::from(adults)
            * u64::from(
                config.demography.food_requirement_per_mille[2]
                    + config.demography.work_food_per_mille,
            )
            + u64::from(dependents) * u64::from(config.demography.food_requirement_per_mille[1]);
        let food = (households * u64::from(config.household_need) * requirements).div_ceil(1000);
        let harvest = match config.household_production {
            HouseholdProduction::Unavailable => 0,
            HouseholdProduction::Available {
                annual_yield_per_plot,
                ..
            } => {
                let plots = match &config.land_access {
                    Some(land) => land
                        .allocations
                        .iter()
                        .map(|grant| u64::from(grant.plots))
                        .sum(),
                    None => {
                        u64::from(
                            super::employment::InitialSettlement::new(config)
                                .farmers(config.population_size()),
                        ) * u64::from(members.div_ceil(2))
                    }
                };
                let farming_households = match &config.land_access {
                    Some(land) => land
                        .allocations
                        .iter()
                        .filter(|grant| grant.plots > 0)
                        .count() as u64,
                    None => u64::from(
                        super::employment::InitialSettlement::new(config)
                            .farmers(config.population_size()),
                    ),
                };
                let crop = plots * u64::from(annual_yield_per_plot);
                let surplus = match config.version {
                    2 => crop,
                    _ => crop - crop.div_ceil(u64::from(food::WheatHusbandry::GROSS_PARTS)),
                };
                let bread = config.food.map_or(surplus, |food| food.bread(surplus));
                (bread / 365).min(
                    (farming_households * u64::from(config.household_need) * requirements)
                        .div_ceil(1000),
                )
            }
        };
        let baking_flour = match config.version {
            2 => 0,
            _ => config
                .food
                .map_or(harvest, |policy| policy.inputs(Good::Bread, harvest)),
        };
        let household_grain = config.food.map_or(baking_flour, |policy| {
            policy.inputs(Good::Flour, baking_flour)
        });
        let mut demand = Self {
            daily: [0; Good::COUNT],
            food,
            household_grain,
            calibration: config.food,
        };
        demand.include(Good::Bread, food.saturating_sub(harvest));
        demand.include(Good::Flour, baking_flour);
        let population = households * u64::from(members);
        demand.include(Good::Wood, population);
        demand.include(Good::Coal, population);
        if config.rent.is_some() {
            demand.daily[Good::Housing as usize - 1] = households;
            demand.include(
                Good::Construction,
                households
                    .div_ceil(u64::from(super::landlords::DEFAULT_HOUSING_UNITS))
                    .div_ceil(u64::from(super::landlords::MAINTENANCE_DAYS)),
            );
        }
        for good in Good::ALL {
            if good.consumption_interval() > 1 {
                let customers = match good.luxury_basic() {
                    Some(_) => households.div_ceil(10),
                    None => households,
                };
                demand.include(
                    good,
                    customers.div_ceil(u64::from(good.consumption_interval())),
                );
            }
        }
        demand
    }

    pub fn production(&self, good: Good) -> u64 {
        let bread = (self.food * 5).div_ceil(4);
        match (self.calibration, good) {
            (Some(food), Good::Grain) => food.inputs(Good::Flour, food.inputs(Good::Bread, bread)),
            (Some(food), Good::Flour) => food.inputs(Good::Bread, bread),
            (_, Good::Grain | Good::Flour | Good::Bread) => bread,
            _ => self.daily[good as usize - 1],
        }
    }

    fn include(&mut self, good: Good, quantity: u64) {
        self.daily[good as usize - 1] += quantity;
        let required = self
            .calibration
            .map_or(quantity, |food| food.inputs(good, quantity));
        for input in good.inputs().into_iter().flatten() {
            self.include(input, required);
        }
    }
}

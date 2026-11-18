use super::{ActorRows, Good, ScenarioConfig, enterprise::HouseholdProduction};

pub(crate) struct SupplierCacheLayout {
    pub extra_words: usize,
}

impl SupplierCacheLayout {
    pub const INLINE_WORDS: usize = 8;
    pub const SUPPLIER_LIMIT: usize = 2048;

    pub fn new(actors: &ActorRows) -> Self {
        let mut producers = [0usize; Good::COUNT];
        for firm in actors.firm_indices() {
            producers[actors.kind[firm] as usize - 1] += 1;
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
    pub household_stock: u64,
    pub coal: u64,
    pub wood: u64,
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

    pub fn inputs(&self, output: Good, daily: u64) -> u64 {
        let days = match output {
            Good::Flour | Good::Bread => self.days,
            _ => self.buffer_days,
        };
        let quantity = daily * u64::from(days);
        self.food
            .map_or(quantity, |food| food.inputs(output, quantity))
    }
}

pub(super) struct StartingDemand {
    pub daily: [u64; Good::COUNT],
    food: u64,
    calibration: Option<super::food::FoodPolicy>,
}

impl StartingDemand {
    pub fn new(config: &ScenarioConfig) -> Self {
        let households = u64::from(config.households);
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
                                .farmers(config.households),
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
                            .farmers(config.households),
                    ),
                };
                let grain = plots * u64::from(annual_yield_per_plot);
                let bread = config.food.map_or(grain, |food| food.bread(grain));
                (bread / 365).min(
                    (farming_households * u64::from(config.household_need) * requirements)
                        .div_ceil(1000),
                )
            }
        };
        let mut demand = Self {
            daily: [0; Good::COUNT],
            food,
            calibration: config.food,
        };
        demand.include(Good::Bread, food.saturating_sub(harvest));
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

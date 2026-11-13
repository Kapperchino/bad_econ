use super::{Good, ScenarioConfig, enterprise::HouseholdProduction};

pub(super) struct StartingDemand {
    pub daily: [u64; Good::COUNT],
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
                        u64::from(config.households.div_ceil(2)) * u64::from(members.div_ceil(2))
                    }
                };
                let farming_households = match &config.land_access {
                    Some(land) => land
                        .allocations
                        .iter()
                        .filter(|grant| grant.plots > 0)
                        .count() as u64,
                    None => u64::from(config.households.div_ceil(2)),
                };
                (plots * u64::from(annual_yield_per_plot) / 365).min(
                    (farming_households * u64::from(config.household_need) * requirements)
                        .div_ceil(1000),
                )
            }
        };
        let mut demand = Self {
            daily: [0; Good::COUNT],
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

    fn include(&mut self, good: Good, quantity: u64) {
        self.daily[good as usize - 1] += quantity;
        for input in good.inputs().into_iter().flatten() {
            self.include(input, quantity);
        }
    }
}

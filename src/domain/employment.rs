use super::{
    Economy, Good, ScenarioConfig,
    calendar::{CalendarDate, HarvestSeason, Weather},
    enterprise::HouseholdProduction,
};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct InitialEmployment {
    employers: Vec<EmployerSlots>,
    workers: usize,
    slots: usize,
    seasonal_days: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct EmployerSlots {
    actor: usize,
    end: usize,
}

impl InitialEmployment {
    pub fn new(
        config: &ScenarioConfig,
        capacities: &[u32; Good::COUNT],
        daily: &[u64; Good::COUNT],
    ) -> Self {
        match (
            config.economy,
            config.household_production,
            &config.population,
            &config.land_access,
        ) {
            (
                Economy::Industrial,
                HouseholdProduction::Available {
                    harvest_start,
                    harvest_days,
                    ..
                },
                None,
                None,
            ) => {
                let season =
                    HarvestSeason::in_window(CalendarDate::default(), harvest_start, harvest_days);
                let seasonal_days = match (season, harvest_days) {
                    (HarvestSeason::Growing, _) | (_, 365) => 0,
                    _ => (harvest_start + 365 - 211.min(365 - harvest_days)) % 365,
                };
                let mut slots = 0;
                let employers = (0..config.firms as usize)
                    .map(|firm| {
                        let good = super::generated_good(config, firm) as usize - 1;
                        let capacity = config
                            .distributors
                            .iter()
                            .find(|store| {
                                store.firm == u64::from(config.households) + firm as u64 + 1
                            })
                            .map(|store| {
                                daily[good]
                                    .div_ceil(u64::from(store.units_per_shift))
                                    .max(1)
                                    .min(u64::from(config.capacity))
                                    as usize
                            })
                            .unwrap_or(
                                daily[good]
                                    .div_ceil(u64::from(config.packet_output))
                                    .max(1)
                                    .min(u64::from(capacities[good]))
                                    as usize,
                            );
                        slots += capacity;
                        EmployerSlots {
                            actor: config.households as usize + firm,
                            end: slots,
                        }
                    })
                    .collect();
                Self {
                    employers,
                    workers: (config.households as usize / 2).min(slots),
                    slots,
                    seasonal_days,
                }
            }
            _ => Self::default(),
        }
    }

    pub fn employer(&self, household: usize) -> Option<usize> {
        match (household % 2, household / 2 < self.workers) {
            (1, true) => {
                let slot = household / 2 * self.slots / self.workers;
                self.employers
                    .get(
                        self.employers
                            .partition_point(|employer| employer.end <= slot),
                    )
                    .map(|employer| employer.actor)
            }
            _ => None,
        }
    }

    pub fn cash(&self, config: &ScenarioConfig, household: usize) -> anyhow::Result<u64> {
        let cash = config.household_liquidity.cash(
            config.household_cash,
            config.seed,
            household as u64 + 1,
        )?;
        let days = match self
            .employer(household)
            .map(|firm| super::generated_good(config, firm - config.households as usize))
        {
            Some(Good::Grain | Good::Flax | Good::Cotton | Good::Dye) => self.seasonal_days,
            _ => 0,
        };
        let members = config.demography.initial_household_members;
        let food =
            u64::from(config.household_need) * u64::from(members) * u64::from(config.prices[2]);
        let coal = (members * Weather::default().heating_units_per_person).div_ceil(2);
        let fuel_price = (config.wage * config.packet_minutes)
            .div_ceil(config.packet_output)
            .clamp(config.minimum_price, config.maximum_price);
        let daily = food
            + u64::from(coal) * u64::from(fuel_price)
            + u64::from(config.rent.map_or(0, |rent| rent.household_daily));
        crate::arithmetic::add(cash, daily * u64::from(days))
    }
}

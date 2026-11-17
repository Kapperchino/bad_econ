use super::{
    Economy, Good, ScenarioConfig,
    calendar::{CalendarDate, HarvestSeason},
    enterprise::HouseholdProduction,
    prices::StartingPrices,
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum InitialSettlement {
    #[default]
    Mixed,
    Rural,
}

impl InitialSettlement {
    pub fn new(config: &ScenarioConfig) -> Self {
        match (
            config.economy,
            config.household_production,
            &config.population,
            &config.land_access,
            config.food.map(|food| food.income),
        ) {
            (
                Economy::Industrial,
                HouseholdProduction::Available { .. },
                None,
                None,
                Some(super::food::FoodIncome::PurchasingPower),
            ) => Self::Rural,
            _ => Self::Mixed,
        }
    }

    fn spacing(self) -> usize {
        match self {
            Self::Mixed => 2,
            Self::Rural => 10,
        }
    }

    pub fn worker(self, household: usize) -> Option<usize> {
        match household % self.spacing() + 1 == self.spacing() {
            true => Some(household / self.spacing()),
            false => None,
        }
    }

    pub fn farmers(self, households: u32) -> u32 {
        households - households / self.spacing() as u32
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct InitialEmployment {
    employers: Vec<EmployerSlots>,
    workers: usize,
    slots: usize,
    seasonal_days: u32,
    household_daily: u64,
    settlement: InitialSettlement,
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
                            .unwrap_or_else(|| {
                                daily[good]
                                    .div_ceil(u64::from(
                                        super::food::technique(config, super::Good::ALL[good])
                                            .output_ticks,
                                    ))
                                    .max(1)
                                    .min(u64::from(capacities[good]))
                                    as usize
                            });
                        slots += capacity;
                        EmployerSlots {
                            actor: config.households as usize + firm,
                            end: slots,
                        }
                    })
                    .collect();
                let settlement = InitialSettlement::new(config);
                Self {
                    employers,
                    workers: (config.households as usize / settlement.spacing()).min(slots),
                    slots,
                    seasonal_days,
                    household_daily: StartingPrices::configured(config).household_daily(config),
                    settlement,
                }
            }
            _ => Self::default(),
        }
    }

    pub fn employer(&self, household: usize) -> Option<usize> {
        match self.settlement.worker(household) {
            Some(worker) if worker < self.workers => {
                let slot = worker * self.slots / self.workers;
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
        crate::arithmetic::add(cash, self.household_daily * u64::from(days))
    }
}

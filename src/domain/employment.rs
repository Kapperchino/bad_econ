use super::{
    Economy, Field, Good, ScenarioConfig, State,
    calendar::{CalendarDate, DAYS_PER_YEAR, HarvestSeason},
    enterprise::{Distributor, EnterpriseModel, HouseholdProduction},
    prices::StartingPrices,
};

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum HiringTerm {
    Day = 1,
    Fortnight = 14,
    Month = 30,
    Year = DAYS_PER_YEAR,
}

impl HiringTerm {
    pub(super) fn at_firm(state: &State, firm: usize, worker_plots: u64) -> anyhow::Result<Self> {
        let good = Good::from_index(state.actors.at(firm).kind)?;
        let model = match state.get(Field::EnterpriseModel, firm) {
            1 => EnterpriseModel::Distributor,
            _ => EnterpriseModel::Producer,
        };
        Ok(match (model, good, worker_plots) {
            (EnterpriseModel::Producer, Good::Grain | Good::Flax | Good::Cotton | Good::Dye, 0) => {
                Self::Year
            }
            (EnterpriseModel::Producer, Good::Grain | Good::Flax | Good::Cotton | Good::Dye, _) => {
                Self::Day
            }
            (EnterpriseModel::Producer, Good::Coal, _) => Self::Month,
            _ => Self::Fortnight,
        })
    }

    pub(super) fn opening_days(self, seed: u64, employer: u64) -> u64 {
        let draw = crate::arithmetic::DrawKey {
            seed,
            day: 0,
            phase: 0,
            actor: employer,
            purpose: 4,
            index: 0,
        }
        .draw();
        1 + draw % self as u64
    }
}

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

    pub(super) fn spacing(self) -> usize {
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
        retailers: &std::collections::BTreeMap<u64, Distributor>,
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
                        let good = super::generated_good(config, firm);
                        let index = good as usize - 1;
                        let actor = config.population_size() as usize + firm;
                        let (output, capacity) = match (retailers.get(&(actor as u64 + 1)), good) {
                            (Some(store), _) => (store.units_per_shift, config.capacity),
                            (None, Good::Railway) => (1, 0),
                            (None, _) => (
                                super::food::technique(config, good).output_ticks,
                                capacities[index],
                            ),
                        };
                        slots += daily[index]
                            .div_ceil(u64::from(output))
                            .max(1)
                            .min(u64::from(capacity)) as usize;
                        EmployerSlots { actor, end: slots }
                    })
                    .collect();
                let settlement = InitialSettlement::new(config);
                Self {
                    employers,
                    workers: (config.population_size() as usize / settlement.spacing()).min(slots),
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
            .map(|firm| super::generated_good(config, firm - config.population_size() as usize))
        {
            Some(Good::Grain | Good::Flax | Good::Cotton | Good::Dye) => self.seasonal_days,
            _ => 0,
        };
        crate::arithmetic::add(cash, self.household_daily * u64::from(days))
    }
}

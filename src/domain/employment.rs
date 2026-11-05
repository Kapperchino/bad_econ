use super::{Economy, Good, ScenarioConfig, enterprise::HouseholdProduction};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct InitialEmployment {
    employers: Vec<EmployerSlots>,
    workers: usize,
    slots: usize,
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
            (Economy::Industrial, HouseholdProduction::Available { .. }, None, None) => {
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
                            .unwrap_or(capacities[good] as usize);
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
}

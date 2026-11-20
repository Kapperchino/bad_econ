use super::{
    Scenario, State,
    demography::{DAYS_PER_YEAR, Person, PersonSeed, Sex},
    employment::InitialSettlement,
    enterprise::HouseholdProduction,
};
use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroU32;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CohortSizes {
    pub workers: NonZeroU32,
    pub farmers: NonZeroU32,
}

impl Default for CohortSizes {
    fn default() -> Self {
        Self {
            workers: NonZeroU32::new(10).unwrap(),
            farmers: NonZeroU32::new(10).unwrap(),
        }
    }
}

impl CohortSizes {
    pub(super) fn is_default(&self) -> bool {
        *self == Self::default()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Role {
    Worker,
    Farmer,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Cohort {
    pub members: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct CohortLayout {
    pub rows: Vec<Cohort>,
    assignments: Vec<u32>,
    pub rental_sites: BTreeMap<usize, Vec<usize>>,
}

impl CohortLayout {
    pub fn new(scenario: &Scenario) -> Result<Self> {
        let config = scenario.config();
        let sizes = config.cohort_sizes;
        match sizes.workers.get().max(sizes.farmers.get()) <= 1024 {
            true => (),
            false => Err(anyhow!("cohort sizes must be in 1..=1024"))?,
        }
        let count = config.population_size();
        let mut individuals = BTreeSet::new();
        let mut housing = BTreeSet::new();
        for firm in count..count + config.firms {
            let site = scenario.source_actor(firm as usize);
            if site.kind == super::Good::Housing as u32 {
                housing.insert(
                    config
                        .property
                        .as_ref()
                        .map(|property| property.region(site.position))
                        .transpose()?
                        .unwrap_or(0),
                );
            }
            let owner = site.owner;
            individuals.insert(scenario.source_actor(owner as usize).identity);
        }
        individuals.extend(
            config
                .ownership
                .iter()
                .flat_map(|entry| entry.shares.iter().map(|share| share.household)),
        );
        if let Some(research) = &config.research {
            individuals.extend(research.labs.iter().map(|lab| lab.researcher));
            individuals.extend(research.strikes.iter().map(|strike| strike.household));
        }
        if let Some(finance) = &config.finance {
            individuals.extend(finance.loans.iter().map(|loan| loan.borrower));
            individuals.extend(finance.support.iter().flat_map(|policy| {
                policy
                    .recipients
                    .iter()
                    .map(|recipient| recipient.household)
            }));
        }
        if let Some(property) = &config.property {
            individuals.extend(property.regions.iter().map(|region| region.landowner));
            individuals.extend(property.buildings.iter().map(|building| building.site));
            individuals.extend(property.orders.iter().map(|order| order.site));
        }
        let grants = config.land_access.as_ref().map(|land| {
            land.allocations
                .iter()
                .filter(|grant| grant.plots > 0)
                .map(|grant| grant.household)
                .collect::<BTreeSet<_>>()
        });
        let settlement = InitialSettlement::new(config);
        let mut rows: Vec<Cohort> = Vec::new();
        let mut assignments = Vec::with_capacity(count as usize);
        let mut open = BTreeMap::new();
        let mut rental_sites = BTreeMap::<usize, Vec<usize>>::new();
        let mut tenants = BTreeMap::<usize, usize>::new();
        for source in 0..count {
            let actor = scenario.source_actor(source as usize);
            let farmer = match &grants {
                Some(grants) => grants.contains(&actor.identity),
                None => {
                    config.population.is_none()
                        && config.household_production != HouseholdProduction::Unavailable
                        && settlement.worker(source as usize).is_none()
                }
            };
            let role = if farmer { Role::Farmer } else { Role::Worker };
            let limit = match role {
                Role::Farmer => sizes.farmers,
                Role::Worker => sizes.workers,
            }
            .get() as usize;
            let region = config
                .property
                .as_ref()
                .map(|property| property.region(actor.position))
                .transpose()?
                .unwrap_or(0);
            let tenant = tenants.entry(region).or_default();
            let rental_site = config.rent.is_some()
                && !housing.contains(&region)
                && (*tenant).is_multiple_of(super::landlords::HOUSEHOLDS_PER_BUILDING);
            *tenant += 1;
            let cell = (
                actor.position.x / config.search_radius,
                actor.position.y / config.search_radius,
            );
            let employer = scenario.initial_employment.employer(source as usize);
            let key = (role, region, cell, employer);
            let row = match rental_site || individuals.contains(&actor.identity) {
                true => rows.len(),
                false => *open.entry(key).or_insert(rows.len()),
            };
            if row == rows.len() {
                rows.push(Cohort {
                    members: Vec::new(),
                });
            }
            rows[row].members.push(source);
            if open.get(&key) == Some(&row) && rows[row].members.len() == limit {
                open.remove(&key);
            }
            if rental_site {
                rental_sites.entry(region).or_default().push(row);
            }
            assignments.push(row as u32);
        }
        Ok(Self {
            rows,
            assignments,
            rental_sites,
        })
    }

    pub fn source(&self, row: usize) -> usize {
        match self.rows.get(row) {
            Some(cohort) => cohort.members[0] as usize,
            None => self.assignments.len() + (row - self.rows.len()),
        }
    }

    pub fn row(&self, source: usize) -> usize {
        match self.assignments.get(source) {
            Some(&row) => row as usize,
            None => self.rows.len() + (source - self.assignments.len()),
        }
    }

    pub fn people(&self, scenario: &Scenario, state: &State) -> Result<Vec<Person>> {
        let policy = &scenario.config().demography;
        let seeds = match policy.people.is_empty() {
            true => self
                .rows
                .iter()
                .enumerate()
                .flat_map(|(row, cohort)| {
                    let household = state.actors.at(row).identity;
                    cohort.members.iter().map(move |&source| {
                        Ok(PersonSeed {
                            identity: state
                                .next_identity
                                .checked_add(u64::from(source))
                                .context("person identity capacity exhausted")?,
                            household,
                            age_days: (20 + source % 25) * DAYS_PER_YEAR,
                            sex: if source % 2 == 0 {
                                Sex::Female
                            } else {
                                Sex::Male
                            },
                        })
                    })
                })
                .collect::<Result<Vec<_>>>()?,
            false => {
                let origins = policy
                    .people
                    .iter()
                    .map(|seed| seed.household)
                    .collect::<BTreeSet<_>>();
                match origins.len() == self.assignments.len()
                    && origins.len() == policy.people.len()
                {
                    true => (),
                    false => Err(anyhow!(
                        "a person-based population requires one demographic seed per starting person"
                    ))?,
                }
                policy
                    .people
                    .iter()
                    .map(|seed| {
                        let row = scenario
                            .member_row(seed.household)
                            .filter(|&row| row < self.rows.len())
                            .context("unknown cohort member")?;
                        Ok(PersonSeed {
                            household: state.actors.at(row).identity,
                            ..seed.clone()
                        })
                    })
                    .collect::<Result<Vec<_>>>()?
            }
        };
        let mut people = policy.from_seeds(state, seeds)?;
        if let Some(population) = &scenario.config().population {
            let sources = policy
                .people
                .iter()
                .map(|seed| (seed.identity, seed.household))
                .collect::<BTreeMap<_, _>>();
            for person in &mut people {
                let source = match sources.get(&person.identity) {
                    Some(identity) => population
                        .index_of(*identity)
                        .context("unknown demographic source")?,
                    None => (person.identity - state.next_identity) as usize,
                };
                person.fatigue = population.households[source].fatigue;
            }
        }
        people.sort_by_key(|person| (person.household, person.identity));
        Ok(people)
    }
}

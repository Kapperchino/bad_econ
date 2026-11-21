use super::{
    Field, Scenario, State, demography::DAYS_PER_YEAR, employment::InitialSettlement,
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
    first: u32,
    pub count: u32,
    pub farmers: u32,
    pub cash: u64,
    ages: [u32; 5],
    female: [u32; 5],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct CohortLayout {
    pub rows: Vec<Cohort>,
    assignments: Vec<u32>,
    grants: BTreeMap<u32, usize>,
    population: u32,
    pub rental_sites: BTreeMap<usize, Vec<usize>>,
}

impl CohortLayout {
    pub fn new(scenario: &Scenario) -> Result<Self> {
        let config = scenario.config();
        let sizes = config.cohort_sizes;
        match sizes.workers.get().max(sizes.farmers.get()) <= 16_384 {
            true => (),
            false => Err(anyhow!("cohort sizes must be in 1..=16384"))?,
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
        let indexed = config.population.is_some() || !config.demography.people.is_empty();
        let grant_ids: BTreeSet<_> = config
            .land_access
            .iter()
            .flat_map(|land| land.allocations.iter().map(|grant| grant.household))
            .collect();
        let mut grant_rows = BTreeMap::new();
        let mut boundaries = individuals.clone();
        boundaries.extend(&grant_ids);
        let mut rows: Vec<Cohort> = Vec::new();
        let mut assignments = Vec::new();
        let mut open = BTreeMap::new();
        let mut rental_sites = BTreeMap::<usize, Vec<usize>>::new();
        let mut tenants = BTreeMap::<usize, usize>::new();
        let stride = (super::BASE_FIELDS + 2 * scenario.ring() * super::Good::COUNT) * 2;
        let farmers = match (config.household_production, &grants) {
            (_, Some(grants)) => grants.len() as u32,
            (HouseholdProduction::Unavailable, _) => 0,
            _ if config.population.is_some() => 0,
            _ => settlement.farmers(count),
        };
        let minimum_rows = u64::from(farmers.div_ceil(sizes.farmers.get()))
            + u64::from((count - farmers.min(count)).div_ceil(sizes.workers.get()))
            + u64::from(config.firms);
        if minimum_rows * stride as u64 > u64::from(u32::MAX) {
            Err(anyhow!(
                "population exceeds GPU address space at these cohort sizes; increase worker and farmer cohort sizes"
            ))?;
        }
        let mut insert = |source: u32, weight: u32| -> Result<()> {
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
            .get();
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
            let row = match rental_site || individuals.contains(&actor.identity) || weight > 1 {
                true => rows.len(),
                false => *open.entry(key).or_insert(rows.len()),
            };
            if row == rows.len() {
                rows.push(Cohort {
                    first: source,
                    count: 0,
                    farmers: 0,
                    cash: 0,
                    ages: [0; 5],
                    female: [0; 5],
                });
            }
            rows[row].count += weight;
            let band = config
                .demography
                .cohort_band((20 + source % 25) * DAYS_PER_YEAR);
            rows[row].ages[band] += weight;
            rows[row].female[band] += weight / 2 + (weight % 2) * u32::from(source % 2 == 0);
            rows[row].farmers += weight * u32::from(settlement.worker(source as usize).is_none());
            if config.population.is_none() {
                let cash = scenario.initial_employment.cash(config, source as usize)?;
                rows[row].cash = crate::arithmetic::add(
                    rows[row].cash,
                    cash.checked_mul(u64::from(weight))
                        .context("cohort opening cash overflow")?,
                )?;
            }
            if open.get(&key) == Some(&row) && rows[row].count == limit {
                open.remove(&key);
            }
            if rental_site {
                rental_sites.entry(region).or_default().push(row);
            }
            if grant_ids.contains(&actor.identity) {
                grant_rows.insert(source, row);
            }
            if indexed {
                assignments.push(row as u32);
            }
            Ok(())
        };
        match (count, indexed) {
            (0..=1_000_000, _) | (_, true) => {
                (0..count).try_for_each(|source| insert(source, 1))?
            }
            _ => {
                let mut start = 0;
                for end in boundaries
                    .iter()
                    .filter_map(|id| u32::try_from(*id).ok())
                    .filter(|&id| (1..=count).contains(&id))
                    .chain(std::iter::once(count + 1))
                {
                    let stop = end - 1;
                    generated_groups(scenario, start, stop, &mut insert)?;
                    if stop < count {
                        insert(stop, 1)?;
                    }
                    start = end;
                }
            }
        }
        Ok(Self {
            rows,
            assignments,
            grants: grant_rows,
            population: count,
            rental_sites,
        })
    }

    pub fn source(&self, row: usize) -> usize {
        match self.rows.get(row) {
            Some(cohort) => cohort.first as usize,
            None => self.population as usize + (row - self.rows.len()),
        }
    }

    pub fn row(&self, source: usize) -> Option<usize> {
        match source.checked_sub(self.population as usize) {
            Some(firm) => Some(self.rows.len() + firm),
            None => self
                .assignments
                .get(source)
                .map(|&row| row as usize)
                .or_else(|| self.grants.get(&(source as u32)).copied())
                .or_else(|| {
                    self.rows
                        .binary_search_by_key(&(source as u32), |row| row.first)
                        .ok()
                }),
        }
    }

    pub fn initialize(&self, scenario: &Scenario, state: &mut State) -> Result<()> {
        let policy = &scenario.config().demography;
        let fields = super::demography::AGE_FIELDS;
        let female = super::demography::FEMALE_FIELDS;
        match policy.people.is_empty() {
            true => {
                for (row, cohort) in self.rows.iter().enumerate() {
                    for (field, count) in fields
                        .into_iter()
                        .zip(cohort.ages)
                        .chain(female.into_iter().zip(cohort.female))
                    {
                        state.set(field, row, u64::from(count));
                    }
                }
            }
            false => {
                let mut identities = state
                    .actors
                    .identities()
                    .iter()
                    .copied()
                    .collect::<BTreeSet<_>>();
                let mut households = BTreeSet::new();
                for seed in &policy.people {
                    let row = scenario
                        .member_row(seed.household)
                        .filter(|&row| row < self.rows.len())
                        .context("unknown cohort member")?;
                    let valid = (1..u64::MAX).contains(&seed.identity)
                        && identities.insert(seed.identity)
                        && households.insert(seed.household)
                        && seed.age_days <= 150 * DAYS_PER_YEAR;
                    if !valid {
                        Err(anyhow!(
                            "invalid demographic age, identity, or duplicate cohort member"
                        ))?;
                    }
                    let band = policy.cohort_band(seed.age_days);
                    let field = fields[band];
                    if seed.sex == super::demography::Sex::Female {
                        state.set(female[band], row, state.get(female[band], row) + 1);
                    }
                    state.set(field, row, state.get(field, row) + 1);
                }
                if households.len() != self.population as usize {
                    Err(anyhow!(
                        "a person-based population requires one demographic seed per starting person"
                    ))?;
                }
            }
        }
        for (row, cohort) in self.rows.iter().enumerate() {
            let workers = state.get(Field::YoungAdults, row) + state.get(Field::OlderAdults, row);
            state.set(Field::EligibleWorkers, row, workers);
            state.set(Field::LivingMembers, row, u64::from(cohort.count));
            state.set(
                Field::Worker,
                row,
                if workers > 0 {
                    row as u64
                } else {
                    u64::from(u32::MAX)
                },
            );
        }
        state.initial_population = u64::from(self.population);
        Ok(())
    }
}

fn generated_groups(
    scenario: &Scenario,
    start: u32,
    end: u32,
    insert: &mut impl FnMut(u32, u32) -> Result<()>,
) -> Result<()> {
    let config = scenario.config();
    let sizes = config.cohort_sizes;
    let spacing = InitialSettlement::new(config).spacing() as u32;
    let mut groups = Vec::new();
    match config.household_production {
        HouseholdProduction::Unavailable => {
            for first in (start..end).step_by(sizes.workers.get() as usize) {
                groups.push((first, (end - first).min(sizes.workers.get())));
            }
        }
        HouseholdProduction::Available { .. } => {
            let farmer_start = start - start / spacing;
            let farmer_end = end - end / spacing;
            let size = match config.land_access {
                Some(_) => sizes.workers.get(),
                None => sizes.farmers.get(),
            };
            for ordinal in (farmer_start..farmer_end).step_by(size as usize) {
                let first = ordinal + ordinal / (spacing - 1);
                groups.push((first, (farmer_end - ordinal).min(size)));
            }
            let mut ordinal = start / spacing;
            while ordinal < end / spacing {
                let source = |offset: u32| (ordinal + offset + 1) * spacing - 1;
                let employer = scenario.initial_employment.employer(source(0) as usize);
                let mut low = 1;
                let mut high = (end / spacing - ordinal).min(sizes.workers.get());
                while low < high {
                    let middle = (low + high).div_ceil(2);
                    match scenario
                        .initial_employment
                        .employer(source(middle - 1) as usize)
                        == employer
                    {
                        true => low = middle,
                        false => high = middle - 1,
                    }
                }
                groups.push((source(0), low));
                ordinal += low;
            }
            groups.sort_unstable_by_key(|&(first, _)| first);
        }
    }
    groups
        .into_iter()
        .try_for_each(|(first, count)| insert(first, count))
}

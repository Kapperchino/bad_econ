use super::{
    Actor, ActorRows, Field, Good, Position, Scenario, State, Technique,
    enterprise::{EnterpriseState, Ownership, OwnershipShare},
    property::PropertyUse,
};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;

pub const HOUSEHOLDS_PER_BUILDING: usize = 32;
pub const MAINTENANCE_DAYS: u32 = 30;
pub const DEFAULT_HOUSING_UNITS: u32 = 8;
pub const EXPANSION_OCCUPANCY_PERCENT: u32 = 95;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct RentalBusiness {
    pub homes: u64,
    pub income: u64,
}

#[derive(Clone)]
pub struct RentalAccount {
    pub landlord: usize,
    pub tenants: Vec<usize>,
}

#[derive(Clone)]
pub struct RentalRegistry {
    pub accounts: Vec<RentalAccount>,
    assignments: Vec<Option<usize>>,
    locations: Vec<RentalLocation>,
    regions: Arc<BTreeMap<usize, RentalRegion>>,
}

pub(crate) enum RentalRefresh {
    Current,
    Changed,
}

pub(crate) enum RentalChanges<'a> {
    #[cfg(test)]
    Compared,
    Selected(&'a [usize]),
}

struct RentalMove {
    actor: usize,
    previous: Option<usize>,
    current: Option<usize>,
}

#[derive(Clone, Copy, Debug)]
enum RentalRefreshMode {
    Current,
    Sparse,
    Dense,
}

#[derive(Clone, PartialEq, Eq)]
struct RentalLocation {
    landlord: usize,
    position: Position,
}

#[derive(PartialEq, Eq)]
struct RentalTenant {
    position: Position,
    household: usize,
    kind: u32,
}

impl RentalTenant {
    fn new(index: usize, actor: &Actor) -> Self {
        Self {
            position: actor.position,
            household: match actor.kind {
                0 => index,
                _ => actor.owner as usize,
            },
            kind: actor.kind,
        }
    }
}

struct RentalHistory<'a> {
    registry: &'a RentalRegistry,
    actors: &'a ActorRows,
}

struct RentalSite {
    account: usize,
    ordinal: usize,
    position: Position,
}

#[derive(Clone, Copy)]
enum RentalAxis {
    Horizontal,
    Vertical,
}

impl RentalAxis {
    fn coordinate(self, position: Position) -> u32 {
        match self {
            Self::Horizontal => position.x,
            Self::Vertical => position.y,
        }
    }

    fn next(self) -> Self {
        match self {
            Self::Horizontal => Self::Vertical,
            Self::Vertical => Self::Horizontal,
        }
    }
}

enum RentalTree {
    Empty,
    Site {
        site: RentalSite,
        axis: RentalAxis,
        lower: Box<RentalTree>,
        upper: Box<RentalTree>,
    },
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct RentalMatch {
    distance: u32,
    priority: usize,
    account: usize,
}

impl RentalTree {
    fn new(mut sites: Vec<RentalSite>, axis: RentalAxis) -> Self {
        match sites.len() {
            0 => Self::Empty,
            count => {
                let middle = count / 2;
                sites.select_nth_unstable_by_key(middle, |site| {
                    (axis.coordinate(site.position), site.account)
                });
                let upper = sites.split_off(middle + 1);
                let site = sites.pop().expect("rental tree median");
                Self::Site {
                    site,
                    axis,
                    lower: Box::new(Self::new(sites, axis.next())),
                    upper: Box::new(Self::new(upper, axis.next())),
                }
            }
        }
    }

    fn nearest(
        &self,
        position: Position,
        start: usize,
        count: usize,
        best: RentalMatch,
    ) -> RentalMatch {
        match self {
            Self::Empty => best,
            Self::Site {
                site,
                axis,
                lower,
                upper,
            } => {
                let candidate = RentalMatch {
                    distance: position.distance(site.position),
                    priority: (site.ordinal + count - start) % count,
                    account: site.account,
                };
                let coordinate = axis.coordinate(position);
                let split = axis.coordinate(site.position);
                let (near, far) = match coordinate < split {
                    true => (lower, upper),
                    false => (upper, lower),
                };
                let nearest = near.nearest(position, start, count, best.min(candidate));
                match coordinate.abs_diff(split) <= nearest.distance {
                    true => far.nearest(position, start, count, nearest),
                    false => nearest,
                }
            }
        }
    }
}

struct RentalRegion {
    tree: RentalTree,
    count: usize,
}

impl RentalRegion {
    fn new(sites: Vec<RentalSite>) -> Self {
        Self {
            count: sites.len(),
            tree: RentalTree::new(sites, RentalAxis::Horizontal),
        }
    }

    fn account(&self, position: Position, household: usize) -> usize {
        self.tree
            .nearest(
                position,
                household % self.count,
                self.count,
                RentalMatch {
                    distance: u32::MAX,
                    priority: usize::MAX,
                    account: usize::MAX,
                },
            )
            .account
    }
}

fn region(scenario: &Scenario, actor: &Actor) -> usize {
    scenario.config().property.as_ref().map_or(0, |property| {
        property
            .region(actor.position)
            .expect("validated property region")
    })
}

impl RentalRegistry {
    pub fn new(scenario: &Scenario, state: &State) -> Self {
        Self::build(scenario, state, None)
    }

    pub(crate) fn updated(
        &self,
        scenario: &Scenario,
        previous: &ActorRows,
        state: &State,
        changes: RentalChanges<'_>,
    ) -> Self {
        let profile = std::env::var_os("BAD_ECON_PROFILE_STAGES").is_some();
        let housing = state
            .actors
            .iter_from(previous.len())
            .any(|actor| actor.kind == Good::Housing as u32);
        match (state.actors.len() >= previous.len(), housing) {
            (true, false) => {
                let started = std::time::Instant::now();
                let additions = state
                    .actors
                    .iter_from(previous.len())
                    .enumerate()
                    .map(|(offset, actor)| {
                        let index = previous.len() + offset;
                        self.regions.get(&region(scenario, &actor)).map(|sites| {
                            sites.account(
                                actor.position,
                                match actor.kind {
                                    0 => index,
                                    _ => actor.owner as usize,
                                },
                            )
                        })
                    })
                    .collect::<Vec<_>>();
                let mut counts = vec![0; self.accounts.len()];
                for &account in additions.iter().flatten() {
                    counts[account] += 1;
                }
                let planned = started.elapsed();
                let started = std::time::Instant::now();
                let mut assignments = Vec::with_capacity(state.actors.len());
                assignments.extend_from_slice(&self.assignments);
                let accounts = self
                    .accounts
                    .iter()
                    .enumerate()
                    .map(|(index, account)| {
                        let mut tenants = Vec::with_capacity(account.tenants.len() + counts[index]);
                        tenants.extend_from_slice(&account.tenants);
                        RentalAccount {
                            landlord: account.landlord,
                            tenants,
                        }
                    })
                    .collect();
                let mut current = Self {
                    accounts,
                    assignments,
                    locations: self.locations.clone(),
                    regions: self.regions.clone(),
                };
                if profile {
                    eprintln!(
                        "RENTALS_PART Clone {:.3}",
                        started.elapsed().as_secs_f64() * 1000.0
                    );
                }
                let started = std::time::Instant::now();
                for (offset, &assignment) in additions.iter().enumerate() {
                    if let Some(account) = assignment {
                        current.accounts[account]
                            .tenants
                            .push(previous.len() + offset);
                    }
                }
                current.assignments.extend_from_slice(&additions);
                if profile {
                    eprintln!(
                        "RENTALS_PART Appends {:.3} actors={}",
                        (planned + started.elapsed()).as_secs_f64() * 1000.0,
                        additions.len()
                    );
                }
                let started = std::time::Instant::now();
                let changes = match changes {
                    #[cfg(test)]
                    RentalChanges::Compared => previous.rental_changes(&state.actors),
                    RentalChanges::Selected(changes) => changes.to_vec(),
                };
                let changed = changes.len();
                current.refresh(scenario, state, changes);
                if profile {
                    eprintln!(
                        "RENTALS_PART Refresh {:.3} actors={changed}",
                        started.elapsed().as_secs_f64() * 1000.0
                    );
                }
                current
            }
            _ => {
                let started = std::time::Instant::now();
                let current = Self::build(
                    scenario,
                    state,
                    Some(RentalHistory {
                        registry: self,
                        actors: previous,
                    }),
                );
                if profile {
                    eprintln!(
                        "RENTALS_PART Build {:.3} housing={housing}",
                        started.elapsed().as_secs_f64() * 1000.0
                    );
                }
                current
            }
        }
    }

    pub(crate) fn refresh(
        &mut self,
        scenario: &Scenario,
        state: &State,
        changes: impl IntoIterator<Item = usize>,
    ) -> RentalRefresh {
        let mut changes = changes.into_iter().collect::<Vec<_>>();
        changes.sort_unstable();
        changes.dedup();
        let current = self.assignments.len() == state.actors.len()
            && changes.iter().all(|&index| {
                let actor = state.actors.at(index);
                match (
                    self.locations
                        .binary_search_by_key(&index, |site| site.landlord),
                    actor.kind == Good::Housing as u32,
                ) {
                    (Ok(site), true) => self.locations[site].position == actor.position,
                    (Err(_), false) => true,
                    _ => false,
                }
            });
        match current {
            false => {
                *self = Self::new(scenario, state);
                RentalRefresh::Changed
            }
            true => {
                let profile = std::env::var_os("BAD_ECON_PROFILE_STAGES").is_some();
                let started = std::time::Instant::now();
                let moves = changes
                    .iter()
                    .filter_map(|&index| {
                        let actor = state.actors.at(index);
                        let assignment =
                            match (actor.kind, self.regions.get(&region(scenario, &actor))) {
                                (kind, Some(sites)) if kind != Good::Housing as u32 => {
                                    let household = match actor.kind {
                                        0 => index,
                                        _ => actor.owner as usize,
                                    };
                                    Some(sites.account(actor.position, household))
                                }
                                _ => None,
                            };
                        (self.assignments[index] != assignment).then_some(RentalMove {
                            actor: index,
                            previous: self.assignments[index],
                            current: assignment,
                        })
                    })
                    .collect::<Vec<_>>();
                let mode = match moves.len() {
                    0 => RentalRefreshMode::Current,
                    1..=255 => RentalRefreshMode::Sparse,
                    _ => RentalRefreshMode::Dense,
                };
                if profile {
                    eprintln!(
                        "RENTALS_PART MovePlan {:.3} actors={} moves={} accounts={} tenants={}",
                        started.elapsed().as_secs_f64() * 1000.0,
                        changes.len(),
                        moves.len(),
                        self.accounts.len(),
                        self.assignments.len()
                    );
                }
                let started = std::time::Instant::now();
                let refresh = match mode {
                    RentalRefreshMode::Current => RentalRefresh::Current,
                    RentalRefreshMode::Sparse => {
                        for movement in moves {
                            if let Some(account) = movement.previous {
                                let tenants = &mut self.accounts[account].tenants;
                                let position = tenants
                                    .binary_search(&movement.actor)
                                    .expect("assigned tenant");
                                tenants.remove(position);
                            }
                            if let Some(account) = movement.current {
                                let tenants = &mut self.accounts[account].tenants;
                                let position = tenants
                                    .binary_search(&movement.actor)
                                    .expect_err("new tenant");
                                tenants.insert(position, movement.actor);
                            }
                            self.assignments[movement.actor] = movement.current;
                        }
                        RentalRefresh::Changed
                    }
                    RentalRefreshMode::Dense => {
                        for movement in moves {
                            self.assignments[movement.actor] = movement.current;
                        }
                        for account in &mut self.accounts {
                            account.tenants.clear();
                        }
                        for (actor, &assignment) in self.assignments.iter().enumerate() {
                            if let Some(account) = assignment {
                                self.accounts[account].tenants.push(actor);
                            }
                        }
                        RentalRefresh::Changed
                    }
                };
                if profile {
                    eprintln!(
                        "RENTALS_PART MoveApply {:.3} mode={mode:?}",
                        started.elapsed().as_secs_f64() * 1000.0
                    );
                }
                refresh
            }
        }
    }

    fn build(scenario: &Scenario, state: &State, history: Option<RentalHistory<'_>>) -> Self {
        let mut buildings = BTreeMap::<usize, Vec<RentalSite>>::new();
        let mut accounts = Vec::<RentalAccount>::new();
        for index in state.actors.firm_indices() {
            let actor = state.actors.at(index);
            if actor.kind == Good::Housing as u32 {
                let sites = buildings.entry(region(scenario, &actor)).or_default();
                sites.push(RentalSite {
                    account: accounts.len(),
                    ordinal: sites.len(),
                    position: actor.position,
                });
                accounts.push(RentalAccount {
                    landlord: index,
                    tenants: Vec::new(),
                });
            }
        }
        let buildings = buildings
            .into_iter()
            .map(|(region, sites)| (region, RentalRegion::new(sites)))
            .collect::<BTreeMap<_, _>>();
        let locations = accounts
            .iter()
            .map(|account| RentalLocation {
                landlord: account.landlord,
                position: state.actors.at(account.landlord).position,
            })
            .collect::<Vec<_>>();
        let history = history.filter(|history| history.registry.locations == locations);
        let mut assignments = Vec::with_capacity(state.actors.len());
        for (index, actor) in state.actors.iter().enumerate() {
            let cached = history.as_ref().and_then(|history| {
                history
                    .actors
                    .get(index)
                    .filter(|previous| {
                        RentalTenant::new(index, previous) == RentalTenant::new(index, &actor)
                    })
                    .map(|_| history.registry.assignments[index])
            });
            let assignment = match (cached, actor.kind, buildings.get(&region(scenario, &actor))) {
                (Some(account), _, _) => account,
                (_, kind, Some(sites)) if kind != Good::Housing as u32 => {
                    let household = match actor.kind {
                        0 => index,
                        _ => actor.owner as usize,
                    };
                    Some(sites.account(actor.position, household))
                }
                _ => None,
            };
            if let Some(account) = assignment {
                accounts[account].tenants.push(index);
            }
            assignments.push(assignment);
        }
        Self {
            accounts,
            assignments,
            locations,
            regions: Arc::new(buildings),
        }
    }

    pub fn words(&self) -> Vec<u32> {
        let mut words = vec![0; (self.accounts.len() * 3).max(1)];
        for (index, account) in self.accounts.iter().enumerate() {
            words[index * 3] = account.landlord as u32;
            words[index * 3 + 1] = words.len() as u32;
            words.extend(account.tenants.iter().map(|&tenant| tenant as u32));
            words[index * 3 + 2] = words.len() as u32;
        }
        words
    }
}

pub fn initialize(scenario: &Scenario, state: &mut State) -> Result<()> {
    let mut households = BTreeMap::<usize, Vec<usize>>::new();
    for (index, actor) in state.actors.iter().enumerate() {
        if actor.kind == 0 && scenario.config().rent.is_some() {
            households
                .entry(region(scenario, &actor))
                .or_default()
                .push(index);
        }
    }
    let mut additions = Vec::new();
    for (region_index, members) in households {
        let existing = state.actors.iter().any(|actor| {
            actor.kind == Good::Housing as u32 && region(scenario, &actor) == region_index
        });
        if !existing {
            let sites: Vec<_> = match scenario.cohorts() {
                Some(layout) => layout
                    .rental_sites
                    .get(&region_index)
                    .cloned()
                    .unwrap_or_default(),
                None => members
                    .chunks(HOUSEHOLDS_PER_BUILDING)
                    .map(|group| group[0])
                    .collect(),
            };
            for site in sites {
                let owner = match &scenario.config().property {
                    Some(property) => state
                        .actors
                        .iter()
                        .position(|actor| {
                            actor.identity == property.regions[region_index].landowner
                        })
                        .context("missing landowner")?,
                    None => site,
                };
                additions.push(Actor {
                    identity: state.next_identity,
                    position: state.actors.at(site).position,
                    kind: Good::Housing as u32,
                    owner: owner as u32,
                    capacity: 0,
                });
                state.next_identity = state
                    .next_identity
                    .checked_add(1)
                    .context("rental identity capacity exhausted")?;
            }
        }
    }
    let before = state.actors.len();
    let after = before + additions.len();
    if after > before {
        let mut values = vec![0; after * scenario.fields()];
        for field in 0..scenario.fields() {
            values[field * after..field * after + before]
                .copy_from_slice(&state.values[field * before..(field + 1) * before]);
        }
        values.extend_from_slice(&state.values[before * scenario.fields()..]);
        state.values = values;
        state.actors.extend(additions);
    }
    for index in before..after {
        let actor = state.actors.at(index);
        state.set_recipe(
            index,
            super::industry::Recipe {
                technique: Technique::housing(match scenario.config().property {
                    Some(_) => 1,
                    None => DEFAULT_HOUSING_UNITS,
                })?,
                inputs: Good::Housing.inputs(),
            },
        );
        state.retain_basic_recipe(index);
        state.set(
            Field::EnterpriseStatus,
            index,
            EnterpriseState::Operating as u64,
        );
        state.set(Field::EquipmentCondition, index, 1000);
        state.set(
            Field::Price,
            index,
            scenario
                .config()
                .rent
                .map_or(1, |rent| {
                    rent.household_daily.clamp(
                        scenario.config().minimum_price,
                        scenario.config().maximum_price,
                    )
                })
                .into(),
        );
        state.set(Field::Wage, index, 1);
        state.set(
            Field::FirstInputCost,
            index,
            Good::Construction.initial_price().into(),
        );
        if scenario.config().people.is_some()
            && state.get(Field::InvestmentFirm, actor.owner as usize) == 0
        {
            state.set(
                Field::InvestmentFirm,
                actor.owner as usize,
                index as u64 + 1,
            );
        }
        state.ownership.push(Ownership {
            firm: actor.identity,
            shares: vec![OwnershipShare {
                household: state.actors.at(actor.owner as usize).identity,
                units: 1_000_000,
            }],
        });
        if scenario.config().property.is_some() {
            state.set(
                Field::PropertyRegion,
                index,
                region(scenario, &actor) as u64 + 1,
            );
            state.set(Field::PropertyUse, index, PropertyUse::Housing as u64);
        }
    }
    for account in RentalRegistry::new(scenario, state).accounts {
        let automatic = account.landlord >= before
            || (scenario.config().population.is_none() && scenario.config().property.is_none());
        let mut plots = 0;
        for tenant in account.tenants {
            state.set(Field::Landlord, tenant, account.landlord as u64 + 1);
            match (
                automatic,
                state.actors.at(tenant).kind,
                state.get(Field::LandPlots, tenant),
            ) {
                (true, 0, 0) => {
                    plots += match scenario.config().property {
                        Some(_) => state.get(Field::OccupiedPlots, tenant),
                        None => u64::from(state.actors.at(tenant).capacity),
                    };
                    state.set(Field::OccupiedPlots, tenant, 0);
                }
                _ => (),
            }
        }
        let site = account.landlord;
        let occupied = match (automatic, &scenario.config().property) {
            (true, Some(_)) => plots,
            (true, None) => plots.div_ceil(state.get(Field::OutputTicks, site)),
            (false, Some(_)) => state.get(Field::OccupiedPlots, site),
            (false, None) => state.get(Field::OccupiedPlots, site).max(1),
        };
        state.set(Field::OccupiedPlots, site, occupied);
        state.set(Field::MaintenanceDays, site, MAINTENANCE_DAYS.into());
        state.set(
            Field::Demand,
            site,
            occupied * state.get(Field::OutputTicks, site),
        );
    }
    Ok(())
}

#[cfg(all(test, feature = "compute"))]
mod tests {
    use super::*;
    use crate::domain::Experiment;
    use crate::simulation::Simulation;

    #[test]
    fn rental_growth_preserves_assignments_and_sorted_tenants_after_gpu_days() -> Result<()> {
        let device = crate::tests::support::device(64)?;
        for scenario in [
            Scenario::industrial(128, 42, Experiment::Baseline)?,
            Scenario::baseline(31, 42, Experiment::Concentrated)?,
        ] {
            let mut simulation = Simulation::new(scenario.clone(), device.clone())?;
            simulation.advance()?;
            let mut state = simulation.download()?;
            let mut cached = RentalRegistry::new(&scenario, &state);
            for step in 0..4u32 {
                let previous = state.actors.clone();
                let mut changes = (1..8).collect::<Vec<_>>();
                for tenant in 1..8 {
                    let mut actor = state.actors.at(tenant);
                    actor.position = Position {
                        x: 71 * tenant as u32 + 11 * step,
                        y: 97 * tenant as u32 + 17 * step,
                    };
                    state.actors.set(tenant, actor);
                }
                let bread = state
                    .actors
                    .firm_indices()
                    .find(|&index| state.actors.at(index).kind == Good::Bread as u32);
                if let Some(firm) = bread {
                    let mut actor = state.actors.at(firm);
                    actor.owner = 2 + step;
                    state.actors.set(firm, actor);
                    changes.push(firm);
                }
                if step == 3 {
                    let building = state
                        .actors
                        .firm_indices()
                        .find(|&index| state.actors.at(index).kind == Good::Housing as u32);
                    if let Some(housing) = building {
                        let mut actor = state.actors.at(housing);
                        actor.position = Position { x: 999, y: 999 };
                        state.actors.set(housing, actor);
                        changes.push(housing);
                    }
                }
                for offset in 0..12u32 {
                    let kind = match (step, offset) {
                        (2, 0) => Good::Housing as u32,
                        (_, 1) => 0,
                        _ => Good::Bread as u32,
                    };
                    state.actors.extend([Actor {
                        identity: state.next_identity,
                        position: Position {
                            x: 50 + offset * 71,
                            y: 100 + step * 97,
                        },
                        kind,
                        owner: 2 + offset,
                        capacity: 1,
                    }]);
                    state.next_identity += 1;
                }
                let selected = cached.updated(
                    &scenario,
                    &previous,
                    &state,
                    RentalChanges::Selected(&changes),
                );
                cached = cached.updated(&scenario, &previous, &state, RentalChanges::Compared);
                assert_eq!(selected.assignments, cached.assignments, "step {step}");
                assert_eq!(selected.words(), cached.words(), "step {step}");
                let rebuilt = RentalRegistry::new(&scenario, &state);
                assert_eq!(cached.assignments, rebuilt.assignments, "step {step}");
                assert_eq!(cached.words(), rebuilt.words(), "step {step}");
                assert!(
                    cached
                        .accounts
                        .iter()
                        .all(|account| account.tenants.is_sorted())
                );
            }
        }
        Ok(())
    }

    #[test]
    fn sparse_and_dense_rental_moves_match_rebuilt_assignments_after_gpu_days() -> Result<()> {
        let scenario = Scenario::industrial(1024, 42, Experiment::Baseline)?;
        let device = crate::tests::support::device(64)?;
        let mut simulation = Simulation::new(scenario.clone(), device)?;
        simulation.advance()?;
        let mut initial = simulation.download()?;
        let housing = initial
            .actors
            .firm_indices()
            .filter(|&index| initial.actors.at(index).kind == Good::Housing as u32)
            .collect::<Vec<_>>();
        assert!(housing.len() >= 2);
        for (ordinal, &index) in housing.iter().enumerate() {
            let mut actor = initial.actors.at(index);
            actor.position = match ordinal {
                0 => Position { x: 0, y: 0 },
                _ => Position { x: 999, y: 999 },
            };
            initial.actors.set(index, actor);
        }
        for index in 0..scenario.config().households as usize {
            let mut actor = initial.actors.at(index);
            actor.position = Position { x: 0, y: 0 };
            initial.actors.set(index, actor);
        }
        let original = RentalRegistry::new(&scenario, &initial);
        for count in [0usize, 3, 255, 256, 513] {
            let mut state = initial.clone();
            for index in 0..count {
                let mut actor = state.actors.at(index);
                actor.position = Position { x: 999, y: 999 };
                state.actors.set(index, actor);
            }
            let rebuilt = RentalRegistry::new(&scenario, &state);
            let changes = (0..count).rev().chain(0..count).collect::<Vec<_>>();
            let selected = original.updated(
                &scenario,
                &initial.actors,
                &state,
                RentalChanges::Selected(&changes),
            );
            let compared =
                original.updated(&scenario, &initial.actors, &state, RentalChanges::Compared);
            assert_eq!(selected.assignments, compared.assignments, "moves {count}");
            assert_eq!(selected.words(), compared.words(), "moves {count}");
            assert_eq!(
                original
                    .assignments
                    .iter()
                    .zip(&rebuilt.assignments)
                    .filter(|(before, after)| before != after)
                    .count(),
                count
            );
            let mut cached = original.clone();
            let refresh = cached.refresh(&scenario, &state, (0..count).rev().chain(0..count));
            match (count, refresh) {
                (0, RentalRefresh::Current) | (1.., RentalRefresh::Changed) => (),
                _ => panic!("rental refresh classification changed"),
            }
            assert_eq!(cached.assignments, rebuilt.assignments, "moves {count}");
            assert_eq!(cached.words(), rebuilt.words(), "moves {count}");
            assert!(
                cached
                    .accounts
                    .iter()
                    .all(|account| account.tenants.is_sorted())
            );
        }
        Ok(())
    }
}

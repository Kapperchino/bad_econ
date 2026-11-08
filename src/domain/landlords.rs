use super::{
    Actor, Field, Good, Position, Scenario, State, Technique,
    enterprise::{EnterpriseState, Ownership, OwnershipShare},
    property::PropertyUse,
};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const HOUSEHOLDS_PER_BUILDING: usize = 32;
pub const MAINTENANCE_DAYS: u32 = 30;
pub const DEFAULT_HOUSING_UNITS: u32 = 8;
pub const EXPANSION_OCCUPANCY_PERCENT: u32 = 95;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct RentalBusiness {
    pub homes: u64,
    pub income: u64,
}

pub struct RentalAccount {
    pub landlord: usize,
    pub tenants: Vec<usize>,
}

pub struct RentalRegistry {
    pub accounts: Vec<RentalAccount>,
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
        let mut buildings = BTreeMap::<usize, Vec<RentalSite>>::new();
        let mut accounts = Vec::<RentalAccount>::new();
        for (index, actor) in state.actors.iter().enumerate() {
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
        for (index, actor) in state.actors.iter().enumerate() {
            match (actor.kind, buildings.get(&region(scenario, &actor))) {
                (kind, Some(sites)) if kind != Good::Housing as u32 => {
                    let household = match actor.kind {
                        0 => index,
                        _ => actor.owner as usize,
                    };
                    let account = sites.account(actor.position, household);
                    accounts[account].tenants.push(index);
                }
                _ => (),
            }
        }
        Self { accounts }
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
            for group in members.chunks(HOUSEHOLDS_PER_BUILDING) {
                let owner = match &scenario.config().property {
                    Some(property) => state
                        .actors
                        .iter()
                        .position(|actor| {
                            actor.identity == property.regions[region_index].landowner
                        })
                        .context("missing landowner")?,
                    None => group[0],
                };
                additions.push(Actor {
                    identity: state.next_identity,
                    position: state.actors.at(group[0]).position,
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
                        None => 1,
                    };
                    state.set(Field::OccupiedPlots, tenant, 0);
                }
                _ => (),
            }
        }
        let site = account.landlord;
        let occupied = match automatic {
            true => match scenario.config().property {
                Some(_) => plots,
                None => plots.div_ceil(state.get(Field::OutputTicks, site)),
            },
            false => match scenario.config().property {
                Some(_) => state.get(Field::OccupiedPlots, site),
                None => state.get(Field::OccupiedPlots, site).max(1),
            },
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

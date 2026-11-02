use super::Good;
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};

#[repr(u32)]
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum EnterpriseState {
    Forming,
    Operating,
    Dormant,
    Liquidating,
    Closed,
}

impl EnterpriseState {
    pub fn from_index(index: u64) -> Result<Self> {
        match index {
            0 => Ok(Self::Forming),
            1 => Ok(Self::Operating),
            2 => Ok(Self::Dormant),
            3 => Ok(Self::Liquidating),
            4 => Ok(Self::Closed),
            _ => Err(anyhow!("unknown enterprise state {index}")),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub enum HouseholdProduction {
    #[default]
    Unavailable,
    Available {
        milling_minutes: u32,
        baking_minutes: u32,
        cultivation_minutes: u32,
        harvest_start: u32,
        harvest_days: u32,
        annual_yield_per_plot: u32,
    },
}
impl HouseholdProduction {
    pub fn industrial() -> Self {
        Self::Available {
            milling_minutes: 30,
            baking_minutes: 30,
            cultivation_minutes: 10,
            harvest_start: 270,
            harvest_days: 30,
            annual_yield_per_plot: 730,
        }
    }
    pub fn new(policy: Self) -> Result<Self> {
        let valid = match policy {
            Self::Unavailable => true,
            Self::Available {
                milling_minutes,
                baking_minutes,
                cultivation_minutes,
                harvest_start,
                harvest_days,
                annual_yield_per_plot,
            } => {
                [milling_minutes, baking_minutes, cultivation_minutes]
                    .into_iter()
                    .all(|minutes| (1..=480).contains(&minutes))
                    && harvest_days > 0
                    && harvest_start
                        .checked_add(harvest_days)
                        .is_some_and(|end| end <= 365)
                    && (1..=10_000).contains(&annual_yield_per_plot)
            }
        };
        match valid {
            true => Ok(policy),
            false => Err(anyhow!("invalid household recipes or harvest season")),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct OwnershipShare {
    pub household: u64,
    pub units: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Ownership {
    pub firm: u64,
    pub shares: Vec<OwnershipShare>,
}
impl Ownership {
    pub fn new(ownership: Self) -> Result<Self> {
        let total = ownership
            .shares
            .iter()
            .map(|share| u64::from(share.units))
            .sum::<u64>();
        let unique = ownership
            .shares
            .iter()
            .map(|share| share.household)
            .collect::<std::collections::BTreeSet<_>>();
        match total == 1_000_000
            && unique.len() == ownership.shares.len()
            && ownership.shares.iter().all(|share| share.units > 0)
        {
            true => Ok(ownership),
            false => Err(anyhow!(
                "ownership requires unique households and one million share units"
            )),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum MigrationState {
    Settled,
    Moving {
        destination: super::Position,
        arrival_day: u64,
        carrier: u64,
        cost: u64,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct LandAccess {
    pub total_plots: u32,
    pub allocations: Vec<LandGrant>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct LandGrant {
    pub household: u64,
    pub plots: u32,
}
impl LandAccess {
    pub fn new(land: Self) -> Result<Self> {
        let allocated = land
            .allocations
            .iter()
            .map(|grant| u64::from(grant.plots))
            .sum::<u64>();
        let unique = land
            .allocations
            .iter()
            .map(|grant| grant.household)
            .collect::<std::collections::BTreeSet<_>>();
        match allocated <= u64::from(land.total_plots)
            && unique.len() == land.allocations.len()
            && land
                .allocations
                .iter()
                .all(|grant| grant.household > 0 && grant.plots <= 1000)
        {
            true => Ok(land),
            false => Err(anyhow!(
                "land allocations exceed finite land or repeat a household"
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkingConditions {
    pub equipment_condition: u32,
    pub safety_provision: u32,
}
impl WorkingConditions {
    pub fn new(conditions: Self) -> Result<Self> {
        match conditions.equipment_condition <= 1000 && conditions.safety_provision <= 900 {
            true => Ok(conditions),
            false => Err(anyhow!("working conditions exceed per-mille bounds")),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Enterprise {
    pub identity: u64,
    pub good: Good,
    pub state: EnterpriseState,
    pub construction_remaining: u32,
    pub ownership: Ownership,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub enum EnterpriseModel {
    #[default]
    Producer,
    Distributor,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Distributor {
    pub firm: u64,
    pub good: Good,
    pub handling_minutes: u32,
    pub units_per_shift: u32,
    pub reserve_days: u32,
    pub margin_per_mille: u32,
}
impl Distributor {
    pub fn new(distributor: Self) -> Result<Self> {
        match (1..=480).contains(&distributor.handling_minutes)
            && (1..=100).contains(&distributor.units_per_shift)
            && (1..=30).contains(&distributor.reserve_days)
            && distributor.margin_per_mille <= 1000
        {
            true => Ok(distributor),
            false => Err(anyhow!(
                "distributor staffing, stock coverage, or margin is invalid"
            )),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct FirmConditions {
    pub firm: u64,
    pub conditions: WorkingConditions,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum FoodAccess {
    Satisfied,
    MissingSupply,
    Unaffordable,
    MatchingFailure,
}

pub struct OwnershipRegistry<'a> {
    pub ownership: &'a [Ownership],
}
impl<'a> OwnershipRegistry<'a> {
    pub fn new(state: &'a super::State) -> Result<Self> {
        let actors = state
            .actors
            .iter()
            .map(|actor| (actor.identity, actor.kind))
            .collect::<std::collections::BTreeMap<_, _>>();
        let mut firms = std::collections::BTreeSet::new();
        for ownership in &state.ownership {
            Ownership::new(ownership.clone())?;
            let valid = firms.insert(ownership.firm)
                && actors.get(&ownership.firm).is_some_and(|&kind| kind > 0)
                && ownership
                    .shares
                    .iter()
                    .all(|share| actors.get(&share.household) == Some(&0));
            match valid {
                true => (),
                false => Err(anyhow!("invalid ownership registry membership"))?,
            }
        }
        match firms.len() == state.actors.iter().filter(|actor| actor.kind > 0).count() {
            true => Ok(Self {
                ownership: &state.ownership,
            }),
            false => Err(anyhow!("every firm requires one ownership record")),
        }
    }
}

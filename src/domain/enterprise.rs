use super::Good;
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EntrepreneurialInterest {
    Uninterested,
    Interested,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ExpansionDisposition {
    Maintain,
    Expand,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum BusinessScale {
    SmallBusiness,
    Industrial,
}

impl BusinessScale {
    pub fn for_business(actor: &super::Actor, homes: u64) -> Self {
        match actor.kind {
            kind if kind == Good::Housing as u32 => match homes {
                0..20 => Self::SmallBusiness,
                _ => Self::Industrial,
            },
            _ => Self::from_workers(actor.capacity),
        }
    }

    pub fn from_workers(workers: u32) -> Self {
        match workers {
            0..20 => Self::SmallBusiness,
            _ => Self::Industrial,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ExpansionPolicy {
    pub small_business_per_mille: u32,
    pub industrial_per_mille: u32,
}

impl Default for ExpansionPolicy {
    fn default() -> Self {
        Self {
            small_business_per_mille: 80,
            industrial_per_mille: 600,
        }
    }
}

impl ExpansionPolicy {
    pub fn new(policy: Self) -> Result<Self> {
        match policy.small_business_per_mille <= policy.industrial_per_mille
            && policy.industrial_per_mille <= 1000
        {
            true => Ok(policy),
            false => Err(anyhow!(
                "expansion probabilities require 0 <= small business <= industrial <= 1000"
            )),
        }
    }

    pub fn disposition(self, seed: u64, owner: u64, scale: BusinessScale) -> ExpansionDisposition {
        let threshold = match scale {
            BusinessScale::SmallBusiness => self.small_business_per_mille,
            BusinessScale::Industrial => self.industrial_per_mille,
        };
        match Self::draw(seed, owner) < threshold {
            true => ExpansionDisposition::Expand,
            false => ExpansionDisposition::Maintain,
        }
    }

    pub fn draw(seed: u64, owner: u64) -> u32 {
        (crate::arithmetic::DrawKey {
            seed,
            day: 0,
            phase: 0,
            actor: owner,
            purpose: 3,
            index: 0,
        }
        .draw()
            % 1000) as u32
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct EntrepreneurshipPolicy {
    pub interested_per_mille: u32,
}

impl Default for EntrepreneurshipPolicy {
    fn default() -> Self {
        Self {
            interested_per_mille: 80,
        }
    }
}

impl EntrepreneurshipPolicy {
    pub(super) fn is_default(&self) -> bool {
        *self == Self::default()
    }

    pub fn new(policy: Self) -> Result<Self> {
        match policy.interested_per_mille <= 1000 {
            true => Ok(policy),
            false => Err(anyhow!("entrepreneurial interest must lie in 0..=1000")),
        }
    }

    pub fn interest(self, seed: u64, household: u64) -> EntrepreneurialInterest {
        let draw = crate::arithmetic::DrawKey {
            seed,
            day: 0,
            phase: 0,
            actor: household,
            purpose: 2,
            index: 0,
        }
        .draw();
        match draw % 1000 < u64::from(self.interested_per_mille) {
            true => EntrepreneurialInterest::Interested,
            false => EntrepreneurialInterest::Uninterested,
        }
    }
}

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

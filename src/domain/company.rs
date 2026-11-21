use super::{
    Good, State,
    enterprise::{EnterpriseModel, Ownership},
};
use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Company {
    pub identity: u64,
    pub sites: Vec<u64>,
}

impl Company {
    pub fn new(company: Self) -> Result<Self> {
        let sites = company.sites.iter().copied().collect::<BTreeSet<_>>();
        match company.identity > 0
            && sites.contains(&company.identity)
            && !sites.contains(&0)
            && sites.len() == company.sites.len()
        {
            true => Ok(company),
            false => Err(anyhow!(
                "a company requires unique sites including its founding site"
            )),
        }
    }

    pub fn initialize(&self, state: &mut State) -> Result<()> {
        let shares = state
            .ownership
            .iter()
            .find(|ownership| ownership.firm == self.identity)
            .context("company references a missing founding site")?
            .shares
            .clone();
        for ownership in &mut state.ownership {
            if self.sites.contains(&ownership.firm) {
                ownership.shares = shares.clone();
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum BusinessLine {
    Production(Good),
    Distribution(Good),
}

impl BusinessLine {
    pub fn from_site(kind: u32, model: u64) -> Result<Self> {
        let good = Good::from_index(kind)?;
        match model {
            value if value == EnterpriseModel::Producer as u64 => Ok(Self::Production(good)),
            value if value == EnterpriseModel::Distributor as u64 => Ok(Self::Distribution(good)),
            _ => Err(anyhow!("unknown enterprise model {model}")),
        }
    }
}

impl std::fmt::Display for BusinessLine {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Production(Good::Housing) => formatter.write_str("Housing landlord"),
            Self::Production(Good::Railway) => formatter.write_str("Railway operator"),
            Self::Production(good) => write!(formatter, "{good} producer"),
            Self::Distribution(good) => write!(formatter, "{good} distributor"),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CompanySize {
    pub sites: usize,
    pub worker_capacity: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AcquisitionScope {
    ExistingLines,
    MayDiversify,
}

impl CompanySize {
    pub fn count_site(&mut self, capacity: u32) {
        self.sites += 1;
        self.worker_capacity += u64::from(capacity);
    }

    pub fn acquisition_scope(self) -> AcquisitionScope {
        match (self.worker_capacity, self.sites) {
            (250.., _) | (_, 25..) => AcquisitionScope::MayDiversify,
            _ => AcquisitionScope::ExistingLines,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct OwnerStubbornness(u64);

impl OwnerStubbornness {
    pub fn from_seed(seed: u64, household: u64) -> Self {
        Self(
            crate::arithmetic::DrawKey {
                seed,
                day: 0,
                phase: 0,
                actor: household,
                purpose: 5,
                index: 0,
            }
            .draw()
                % 1000,
        )
    }

    pub fn per_mille(self) -> u64 {
        self.0
    }
}

pub struct CompanyRegistry<'a> {
    pub companies: &'a [Company],
}

impl<'a> CompanyRegistry<'a> {
    pub fn new(state: &'a State) -> Result<Self> {
        if !state.companies.is_empty() {
            let sites = state
                .companies
                .iter()
                .flat_map(|company| company.sites.iter().copied())
                .collect::<BTreeSet<_>>();
            let actors = state
                .actors
                .iter()
                .filter(|actor| sites.contains(&actor.identity))
                .map(|actor| (actor.identity, actor))
                .collect::<BTreeMap<_, _>>();
            let ownership = state
                .ownership
                .iter()
                .filter(|entry| sites.contains(&entry.firm))
                .map(|entry| (entry.firm, entry))
                .collect::<BTreeMap<_, _>>();
            let mut assigned = BTreeSet::new();
            for company in &state.companies {
                Company::new(company.clone())?;
                let issued = ownership
                    .get(&company.identity)
                    .context("company has no issued ownership")?;
                for site in &company.sites {
                    let actor = actors
                        .get(site)
                        .context("company references a missing facility")?;
                    let owner = state
                        .actors
                        .get(actor.owner as usize)
                        .context("facility has no owner")?;
                    let admitted = actor.kind > 0
                        && assigned.insert(*site)
                        && ownership
                            .get(site)
                            .is_some_and(|entry| entry.shares == issued.shares)
                        && issued
                            .shares
                            .iter()
                            .any(|share| share.household == owner.identity);
                    match admitted {
                        true => (),
                        false => Err(anyhow!(
                            "company facilities must be distinct and share company ownership"
                        ))?,
                    }
                }
            }
        }
        Ok(Self {
            companies: &state.companies,
        })
    }
}

impl State {
    #[cfg(feature = "compute")]
    pub(crate) fn company_founders(&self) -> BTreeMap<u64, u64> {
        self.companies
            .iter()
            .flat_map(|company| company.sites.iter().map(|&id| (id, company.identity)))
            .collect()
    }

    pub fn company(&self, site: u64) -> Option<&Company> {
        self.companies
            .iter()
            .find(|company| company.sites.contains(&site))
    }

    pub fn company_identity(&self, site: u64) -> u64 {
        self.company(site).map_or(site, |company| company.identity)
    }

    pub fn company_sites(&self, site: u64) -> Vec<u64> {
        self.company(site)
            .map_or_else(|| vec![site], |company| company.sites.clone())
    }

    pub fn set_company_ownership(&mut self, ownership: Ownership) {
        let sites = self.company_sites(ownership.firm);
        for issued in &mut self.ownership {
            if sites.contains(&issued.firm) {
                issued.shares = ownership.shares.clone();
            }
        }
    }
}

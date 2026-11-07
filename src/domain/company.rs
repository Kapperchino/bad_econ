use super::{State, enterprise::Ownership};
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

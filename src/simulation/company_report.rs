use super::{
    Simulation, decode,
    layout::{TICK_BYTES, TICK_WORDS},
};
use crate::domain::{
    Actor, Field,
    company::{BusinessLine, CompanySize},
    enterprise::EnterpriseState,
};
use anyhow::Result;
use std::collections::BTreeMap;

pub(crate) struct CompanyReport {
    pub firms: Vec<Actor>,
    pub companies: Vec<Business>,
}

pub(crate) struct Business {
    pub identity: u64,
    pub open_site: u64,
    pub line: Option<BusinessLine>,
    pub size: CompanySize,
    pub profit: i128,
}

impl Simulation {
    pub(crate) fn company_report(&mut self) -> Result<CompanyReport> {
        let actors = &self.metadata.actors;
        let start = actors.firm_indices().next().unwrap_or(actors.len());
        let length = actors.len() - start;
        let read = |field: Field| -> Result<Vec<u64>> {
            Ok(decode(&self.device.read_range(
                &self.state,
                (field as usize * self.config.state_stride as usize + start) * TICK_WORDS,
                length * TICK_WORDS,
            )?))
        };
        let statuses = read(Field::EnterpriseStatus)?;
        let models = read(Field::EnterpriseModel)?;
        let opening = read(Field::BusinessOpeningCash)?;
        let closing = read(Field::BusinessClosingCash)?;
        self.readback_bytes += (length * TICK_BYTES * 4) as u64;
        let founders = self.metadata.company_founders();
        let mut companies = BTreeMap::new();
        let mut firms = Vec::new();
        for index in actors.firm_indices() {
            let actor = actors.at(index);
            let id = founders
                .get(&actor.identity)
                .copied()
                .unwrap_or(actor.identity);
            let row = index - start;
            let business = companies.entry(id).or_insert(Business {
                identity: id,
                open_site: actor.identity,
                line: None,
                size: CompanySize::default(),
                profit: 0,
            });
            business.profit += i128::from(closing[row]) - i128::from(opening[row]);
            if statuses[row] != EnterpriseState::Closed as u64 {
                let line = BusinessLine::from_site(actor.kind, models[row])?;
                business.line = match business.size.sites {
                    0 => Some(line),
                    _ => business.line.filter(|&previous| previous == line),
                };
                if business.size.sites == 0 || actor.identity == id {
                    business.open_site = actor.identity;
                }
                business.size.count_site(actor.capacity);
                firms.push(actor);
            }
        }
        Ok(CompanyReport {
            firms,
            companies: companies
                .into_values()
                .filter(|company| company.size.sites > 0)
                .collect(),
        })
    }
}

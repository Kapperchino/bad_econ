use super::{
    entries::{Transfer, transfer},
    records::PendingRecord,
};
use crate::domain::{Scenario, State, enterprise::Ownership};
use crate::simulation::{Device, decode, encode, enterprise, layout::TICK_WORDS};
use anyhow::{Context, Result, anyhow};
use burn_wgpu::CubeTensor;
use std::collections::{BTreeMap, BTreeSet};

pub(in crate::simulation) struct SettlementBatch {
    pub ownership: Option<Vec<Ownership>>,
    pub readback_bytes: u64,
    indices: CubeTensor,
    values: CubeTensor,
}

impl SettlementBatch {
    pub fn new(
        device: &Device,
        scenario: &Scenario,
        metadata: &State,
        state: &CubeTensor,
        requests: &[PendingRecord],
        day: u64,
    ) -> Result<Option<Self>> {
        match requests
            .iter()
            .any(|record| matches!(record.request >> 24, 4 | 7))
        {
            false => Ok(None),
            true => Self::prepare(device, scenario, metadata, state, requests, day).map(Some),
        }
    }

    fn prepare(
        device: &Device,
        scenario: &Scenario,
        metadata: &State,
        state: &CubeTensor,
        requests: &[PendingRecord],
        day: u64,
    ) -> Result<Self> {
        let mut selected = BTreeSet::new();
        let mut firms = BTreeSet::new();
        for record in requests.iter().filter(|record| record.request > 0) {
            selected.insert(record.actor as usize);
            match record.request >> 24 {
                2 | 7 => {
                    let target = (record.request & 0x00ffffff) as usize - 1;
                    selected.insert(target);
                    if record.request >> 24 == 7 {
                        firms.insert(target);
                    }
                }
                4 => {
                    firms.insert(record.actor as usize);
                }
                _ => (),
            }
        }
        let company_sites = firms
            .iter()
            .flat_map(|&firm| metadata.company_sites(metadata.actors.at(firm).identity))
            .collect::<BTreeSet<_>>();
        for (index, actor) in metadata.actors.iter().enumerate() {
            if company_sites.contains(&actor.identity) {
                firms.insert(index);
                selected.insert(index);
            }
        }
        let owned = metadata
            .ownership
            .iter()
            .map(|ownership| (ownership.firm, ownership))
            .collect::<BTreeMap<_, _>>();
        let mut ownership = Vec::new();
        let mut shareholders = BTreeSet::new();
        for firm in firms {
            let actor = metadata.actors.at(firm);
            let founder = actor.owner as usize;
            selected.insert(founder);
            let issued = owned
                .get(&actor.identity)
                .context("settlement requires issued shares")?;
            ownership.push((*issued).clone());
            for share in &issued.shares {
                if share.household != metadata.actors.at(founder).identity {
                    shareholders.insert(share.household);
                }
            }
        }
        if !shareholders.is_empty() {
            for (index, actor) in metadata.actors.iter().enumerate() {
                if shareholders.remove(&actor.identity) {
                    selected.insert(index);
                }
            }
        }
        match shareholders.is_empty() {
            true => (),
            false => Err(anyhow!("settlement shareholder is missing"))?,
        }
        let selected = selected.into_iter().collect::<Vec<_>>();
        let required = selected.len() as u64 * scenario.fields() as u64 * 16;
        device.reclaim_memory(required)?;
        match device
            .client
            .memory_report(cubecl::MemoryScope::Device)
            .usage()
            .bytes_reserved
            .saturating_add(required)
            <= device.memory_budget.bytes()
        {
            true => (),
            false => Err(anyhow!(
                "settlement staging exceeds the device memory budget"
            ))?,
        }
        let indices = device.tensor(
            &selected
                .iter()
                .map(|&actor| actor as u32)
                .collect::<Vec<_>>(),
        );
        let values = device.empty(selected.len() * scenario.fields() * TICK_WORDS);
        transfer(
            device,
            state,
            &values,
            &indices,
            metadata.actors.len(),
            Transfer::Gather,
        );
        let words = device.read(&values)?;
        let readback_bytes = words.len() as u64 * 4;
        let mut compact = State {
            actors: selected
                .iter()
                .map(|&index| {
                    let mut actor = metadata.actors.at(index);
                    actor.owner =
                        selected.binary_search(&(actor.owner as usize)).unwrap_or(0) as u32;
                    actor
                })
                .collect(),
            values: decode(&words),
            ownership,
            companies: metadata.companies.clone(),
            day,
            next_identity: metadata.next_identity,
            initial_cash: metadata.initial_cash,
            initial_goods: metadata.initial_goods,
            initial_population: metadata.initial_population,
            people: Vec::new(),
            research: None,
        };
        let mut actions = vec![0; selected.len()];
        for record in requests.iter().filter(|record| record.request > 0) {
            let actor = selected
                .binary_search(&(record.actor as usize))
                .ok()
                .context("missing settlement actor")?;
            actions[actor] = match record.request >> 24 {
                kind @ (2 | 7) => {
                    let target = (record.request & 0x00ffffff) as usize - 1;
                    (kind << 24)
                        + selected
                            .binary_search(&target)
                            .ok()
                            .context("missing settlement target")? as u32
                        + 1
                }
                _ => record.request,
            };
        }
        enterprise::transitions(&mut compact, scenario, &actions)?;
        let ownership = match requests.iter().any(|record| record.request >> 24 == 7) {
            false => None,
            true => {
                let changed = compact
                    .ownership
                    .into_iter()
                    .map(|ownership| (ownership.firm, ownership))
                    .collect::<BTreeMap<_, _>>();
                Some(
                    metadata
                        .ownership
                        .iter()
                        .map(|ownership| changed.get(&ownership.firm).unwrap_or(ownership).clone())
                        .collect(),
                )
            }
        };
        Ok(Self {
            ownership,
            readback_bytes,
            indices,
            values: device.tensor(bytemuck::cast_slice(&encode(&compact.values))),
        })
    }

    pub fn apply(&self, device: &Device, state: &CubeTensor, actors: usize) {
        transfer(
            device,
            &self.values,
            state,
            &self.indices,
            actors,
            Transfer::Scatter,
        );
    }
}

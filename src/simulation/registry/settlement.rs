use super::staging::{LivingRows, StagedRows, Staging, Transfer, transfer};
use super::{MigrationRegistry, records::PendingRecord};
use crate::domain::{ActorRows, BASE_FIELDS, Good, Scenario, State, enterprise::Ownership};
use crate::simulation::enterprise::acquisitions::PURCHASE;
use crate::simulation::{Device, encode, enterprise, storage::layout::NO_ACTOR};
use anyhow::{Context, Result, anyhow};
use burn_wgpu::CubeTensor;
use std::collections::{BTreeMap, BTreeSet};
mod moves;
mod tiles;

enum SettlementValues {
    Rows(CubeTensor),
    Changes(tiles::TileChanges),
}

pub(in crate::simulation) struct SettlementBatch {
    pub ownership: Option<Vec<Ownership>>,
    pub companies: Option<Vec<crate::domain::company::Company>>,
    pub owners: Vec<OwnerChange>,
    pub readback_bytes: u64,
    indices: CubeTensor,
    values: SettlementValues,
    moves: moves::IndependentMoves,
}

pub(in crate::simulation) struct OwnerChange {
    pub firm: usize,
    owner: u32,
}

impl SettlementBatch {
    pub fn required(requests: &[PendingRecord]) -> bool {
        requests
            .iter()
            .any(|record| matches!(record.request >> 24, 4 | 5 | 7 | PURCHASE))
    }

    pub fn new(
        device: &Device,
        scenario: &Scenario,
        metadata: &State,
        state: &CubeTensor,
        requests: &[PendingRecord],
        day: u64,
        migrations: &MigrationRegistry,
    ) -> Result<Option<Self>> {
        match Self::required(requests) {
            false => Ok(None),
            true => Self::prepare(device, scenario, metadata, state, requests, day, migrations)
                .map(Some),
        }
    }

    fn prepare(
        device: &Device,
        scenario: &Scenario,
        metadata: &State,
        state: &CubeTensor,
        requests: &[PendingRecord],
        day: u64,
        migrations: &MigrationRegistry,
    ) -> Result<Self> {
        let mut selected = BTreeSet::new();
        let mut firms = BTreeSet::new();
        let mut estates = BTreeSet::new();
        let living = match requests.iter().any(|record| record.request >> 24 == 5) {
            true => LivingRows::read(device, state, scenario.fields(), metadata.actors.len())?,
            false => LivingRows::default(),
        };
        for record in requests
            .iter()
            .filter(|record| record.request > 0 && !matches!(record.request >> 24, 2 | 3))
        {
            selected.insert(record.actor as usize);
            match record.request >> 24 {
                7 | PURCHASE => {
                    selected.insert(record.target());
                    firms.insert(record.target());
                }
                4 => {
                    firms.insert(record.actor as usize);
                }
                5 => {
                    let identity = metadata.actors.at(record.actor as usize).identity;
                    estates.insert(identity);
                    let successor = match migrations
                        .children(identity)
                        .find(|&row| living.contains(row))
                    {
                        Some(heir) => Some(heir),
                        None => enterprise::nearest_living_household(
                            &metadata.actors,
                            scenario.config().property.as_ref(),
                            record.actor as usize,
                            |row| living.contains(row),
                        )?,
                    };
                    selected.extend(successor);
                }
                _ => (),
            }
        }
        let inherited = metadata
            .ownership
            .iter()
            .filter(|ownership| {
                ownership
                    .shares
                    .iter()
                    .any(|share| estates.contains(&share.household))
            })
            .map(|ownership| ownership.firm)
            .collect::<BTreeSet<_>>();
        metadata
            .actors
            .firm_indices()
            .filter(|&index| inherited.contains(&metadata.actors.at(index).identity))
            .for_each(|index| {
                firms.insert(index);
                selected.insert(index);
            });
        firms.extend(
            selected
                .iter()
                .copied()
                .filter(|&index| metadata.actors.at(index).kind > 0),
        );
        let founders = metadata.company_founders();
        let company_of = |index| {
            let id = metadata.actors.at(index).identity;
            founders.get(&id).copied().unwrap_or(id)
        };
        let companies = firms
            .iter()
            .copied()
            .map(company_of)
            .collect::<BTreeSet<_>>();
        metadata
            .actors
            .firm_indices()
            .filter(|&index| companies.contains(&company_of(index)))
            .for_each(|index| {
                firms.insert(index);
                selected.insert(index);
            });
        let owned = metadata
            .ownership
            .iter()
            .map(|ownership| (ownership.firm, ownership))
            .collect::<BTreeMap<_, _>>();
        let mut ownership = Vec::with_capacity(firms.len());
        let mut shareholders = BTreeSet::new();
        for firm in firms {
            let actor = metadata.actors.at(firm);
            let founder = actor.owner as usize;
            let founder_identity = metadata.actors.at(founder).identity;
            selected.insert(founder);
            let issued = owned
                .get(&actor.identity)
                .context("settlement requires issued shares")?;
            ownership.push((*issued).clone());
            shareholders.extend(
                issued
                    .shares
                    .iter()
                    .map(|share| share.household)
                    .filter(|&household| household != founder_identity),
            );
        }
        shareholders.retain(|&identity| {
            match scenario
                .index_of(identity)
                .filter(|&row| metadata.actors.identities().get(row) == Some(&identity))
            {
                Some(row) => {
                    selected.insert(row);
                    false
                }
                None => true,
            }
        });
        if !shareholders.is_empty() {
            selected.extend(
                metadata
                    .actors
                    .identities()
                    .iter()
                    .enumerate()
                    .filter_map(|(index, identity)| shareholders.remove(identity).then_some(index)),
            );
        }
        match shareholders.is_empty() {
            true => (),
            false => Err(anyhow!("settlement shareholder is missing"))?,
        }
        let moves = moves::IndependentMoves::new(device, requests, &mut selected);
        let selected = selected.into_iter().collect::<Vec<_>>();
        if day == crate::simulation::profiling_day()
            && std::env::var_os("BAD_ECON_PROFILE_REGISTRY").is_some()
        {
            let tiles = selected
                .iter()
                .map(|actor| actor / crate::simulation::economy::accounting::audit::TILE_ACTORS)
                .collect::<BTreeSet<_>>()
                .len();
            eprintln!(
                "REGISTRY {day} Settlement actors={} total_actors={} tiles={tiles} total_tiles={}",
                selected.len(),
                metadata.actors.len(),
                metadata
                    .actors
                    .len()
                    .div_ceil(crate::simulation::economy::accounting::audit::TILE_ACTORS),
            );
        }
        let fields = match requests
            .iter()
            .any(|record| matches!(record.request >> 24, 4 | 5))
        {
            true => BASE_FIELDS + 2 * scenario.ring() * Good::COUNT,
            false => BASE_FIELDS,
        };
        let staged = StagedRows::new(
            device,
            state,
            scenario.fields(),
            &selected,
            Staging::Settlement { fields },
        )?;
        let tiles = tiles::GatheredTiles::supported(device, staged.elements)
            .then(|| {
                tiles::GatheredTiles::read(
                    device,
                    state,
                    &staged.indices,
                    staged.stride,
                    staged.elements,
                )
            })
            .transpose()?;
        let readback_bytes = living.readback_bytes()
            + tiles
                .as_ref()
                .map_or(staged.readback_bytes(), |tiles| tiles.readback_bytes());
        let values = match &tiles {
            Some(tiles) => tiles.unpack(staged.elements),
            None => staged.read(device, state)?,
        };
        let mut compact = State {
            actors: selected
                .iter()
                .map(|&index| {
                    let mut actor = metadata.actors.at(index);
                    actor.owner = selected
                        .binary_search(&(actor.owner as usize))
                        .map_or(NO_ACTOR, |row| row as u32);
                    actor
                })
                .collect(),
            values,
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
            if let Ok(actor) = selected.binary_search(&(record.actor as usize)) {
                actions[actor] = match record.request >> 24 {
                    2 | 7 | PURCHASE => record.remap_target(
                        selected
                            .binary_search(&record.target())
                            .ok()
                            .context("missing settlement target")?,
                    ),
                    _ => record.request,
                };
            }
        }
        let previous_owners = compact
            .actors
            .iter()
            .map(|actor| actor.owner)
            .collect::<Vec<_>>();
        enterprise::transitions_in_rows(&mut compact, scenario, &actions, |row| selected[row])?;
        let companies = (compact.companies != metadata.companies).then_some(compact.companies);
        let owners = compact
            .actors
            .iter()
            .enumerate()
            .filter(|(index, actor)| actor.owner != previous_owners[*index])
            .map(|(index, actor)| OwnerChange {
                firm: selected[index],
                owner: selected[actor.owner as usize] as u32,
            })
            .collect();
        let changed = compact
            .ownership
            .into_iter()
            .filter(|issued| owned.get(&issued.firm).copied() != Some(issued))
            .map(|issued| (issued.firm, issued))
            .collect::<BTreeMap<_, _>>();
        let ownership = (!changed.is_empty()).then(|| {
            metadata
                .ownership
                .iter()
                .map(|issued| changed.get(&issued.firm).unwrap_or(issued).clone())
                .collect()
        });
        Ok(Self {
            companies,
            ownership,
            owners,
            readback_bytes,
            indices: staged.indices,
            moves,
            values: match tiles {
                Some(tiles) => SettlementValues::Changes(tiles.changes(device, &compact.values)),
                None => SettlementValues::Rows(
                    device.tensor(bytemuck::cast_slice(&encode(&compact.values))),
                ),
            },
        })
    }

    pub fn update_owners(&self, actors: &mut ActorRows) {
        for change in &self.owners {
            let mut actor = actors.at(change.firm);
            actor.owner = change.owner;
            actors.set(change.firm, actor);
        }
    }

    pub fn indices(&self) -> &CubeTensor {
        &self.indices
    }

    pub fn apply(
        &self,
        gpu: crate::simulation::DeviceEconomy<'_>,
        requests: &CubeTensor,
        control: &CubeTensor,
        day: u32,
    ) {
        self.moves.apply(gpu, requests, control, day);
        let device = gpu.device;
        let state = gpu.state;
        let actors = gpu.config.state_stride as usize;
        match &self.values {
            SettlementValues::Rows(values) => transfer(
                device,
                values,
                state,
                &self.indices,
                actors,
                Transfer::Scatter,
            ),
            SettlementValues::Changes(changes) => {
                changes.apply(device, state, &self.indices, actors)
            }
        }
    }
}

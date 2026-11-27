use super::staging::{StagedRows, Staging, Transfer, transfer};
use super::*;
use crate::domain::Good;
use crate::simulation::{encode, enterprise, storage::layout::TICK_WORDS};
use std::collections::BTreeSet;

pub(in crate::simulation) fn state_capacity(count: usize) -> usize {
    let growth = (count / 64).next_power_of_two().clamp(64, 16_384);
    (count + growth).next_multiple_of(growth)
}

pub(in crate::simulation) fn registry_stride(
    scenario: &Scenario,
    actors: &crate::domain::ActorRows,
    spare_bytes: u64,
) -> usize {
    let count = actors.len();
    let row_words = scenario.fields() * TICK_WORDS;
    let spare_rows = spare_bytes / (row_words * size_of::<u32>() * 2) as u64;
    let households = count - actors.firms();
    let expected_entries =
        households * scenario.config().entrepreneurship.interested_per_mille as usize / 1000;
    let preferred =
        state_capacity(count).max((count + expected_entries.min(count / 8)).next_multiple_of(256));
    preferred
        .min(count.saturating_add(spare_rows as usize))
        .min(u32::MAX as usize / row_words)
        .max(count)
}

pub(in crate::simulation) struct EntryRequests {
    pub transitions: Vec<PendingRecord>,
    entries: Vec<PendingRecord>,
}

impl EntryRequests {
    pub fn new(scenario: &Scenario, actors: usize, requests: &[PendingRecord]) -> Option<Self> {
        let entries = requests
            .iter()
            .copied()
            .filter(|record| Self::is_entry(record.request))
            .collect::<Vec<_>>();
        let sources = entries.iter().all(|record| match record.request >> 24 {
            8 | 9 => (1..=actors as u32).contains(&record.payload()),
            _ => true,
        });
        match (
            scenario.config().finance.as_ref(),
            entries.is_empty(),
            sources,
        ) {
            (None, false, true) => Some(Self {
                transitions: requests
                    .iter()
                    .copied()
                    .filter(|record| !Self::is_entry(record.request))
                    .collect(),
                entries,
            }),
            _ => None,
        }
    }

    fn is_entry(action: u32) -> bool {
        (1..=Good::COUNT as u32).contains(&action) || matches!(action >> 24, 1 | 8 | 9)
    }
}

pub(in crate::simulation) struct EntryGrowth {
    pub requests: EntryRequests,
    pub plan: RegistryPlan,
}

impl EntryGrowth {
    pub fn new(
        scenario: &Scenario,
        metadata: &State,
        migrations: &MigrationRegistry,
        requests: &[PendingRecord],
    ) -> Result<Option<Self>> {
        EntryRequests::new(scenario, metadata.actors.len(), requests)
            .map(|requests| {
                RegistryPlan::with_growth(
                    scenario,
                    metadata,
                    migrations,
                    &requests.transitions,
                    requests.entries.len(),
                )
                .map(|plan| plan.map(|plan| Self { requests, plan }))
            })
            .transpose()
            .map(Option::flatten)
    }
}

pub(in crate::simulation) struct EntryBatch {
    pub metadata: State,
    pub readback_bytes: u64,
    pub formations: Vec<enterprise::HouseholdFormation>,
    indices: CubeTensor,
    values: CubeTensor,
}

impl EntryBatch {
    pub fn new(
        device: &Device,
        scenario: &Scenario,
        metadata: &State,
        state: &CubeTensor,
        entries: &EntryRequests,
        plan: &mut RegistryPlan,
        next_identity: u64,
    ) -> Result<Self> {
        let selected = entries
            .entries
            .iter()
            .flat_map(|record| {
                let source = match record.request >> 24 {
                    8 | 9 => Some(record.target()),
                    _ => None,
                };
                std::iter::once(record.actor as usize).chain(source)
            })
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let fields = crate::domain::BASE_FIELDS;
        let staged = StagedRows::new(device, state, scenario.fields(), &selected, Staging::Entry)?;
        let gathered = staged.read_entry(device, state)?;
        let mut compact = State {
            day: metadata.day,
            actors: selected
                .iter()
                .map(|&index| metadata.actors.at(index))
                .collect(),
            values: gathered.values,
            ownership: Vec::new(),
            companies: Vec::new(),
            next_identity,
            initial_cash: metadata.initial_cash,
            initial_goods: metadata.initial_goods,
            research: metadata.research.clone(),
            people: metadata.people.clone(),
            initial_population: metadata.initial_population,
        };
        let mut compact_requests = vec![0; selected.len()];
        for record in &entries.entries {
            let owner = record.actor as usize;
            let index = selected.binary_search(&owner).expect("selected owner");
            compact_requests[index] = match record.request >> 24 {
                8 | 9 => record.remap_target(
                    selected
                        .binary_search(&record.target())
                        .expect("selected source"),
                ),
                _ => record.request,
            };
        }
        enterprise::enter(&mut compact, scenario, &compact_requests)?;
        let before = metadata.actors.len();
        let global_actor = |index: usize| {
            selected
                .get(index)
                .copied()
                .unwrap_or_else(|| before + index - selected.len())
        };
        let formations = enterprise::form_households(&mut compact, scenario, &compact_requests)?
            .into_iter()
            .map(|formation| enterprise::HouseholdFormation {
                parent: selected[formation.parent],
                household: before + formation.household - selected.len(),
                person: formation.person,
            })
            .collect::<Vec<_>>();
        plan.migrations
            .extend_households(&metadata.actors, &formations);
        let prepared_actors =
            match super::settlement::SettlementBatch::required(&entries.transitions) {
                true => None,
                false => plan.actors.take(),
            };
        let mut grown = State {
            actors: prepared_actors.unwrap_or_else(|| {
                metadata
                    .actors
                    .clone_for_growth(compact.actors.len() - selected.len())
            }),
            day: metadata.day,
            next_identity: metadata.next_identity,
            values: metadata.values.clone(),
            initial_cash: metadata.initial_cash,
            initial_goods: metadata.initial_goods,
            research: metadata.research.clone(),
            people: metadata.people.clone(),
            initial_population: metadata.initial_population,
            ownership: metadata.ownership.clone(),
            companies: metadata.companies.clone(),
        };
        grown
            .actors
            .extend(compact.actors.iter_from(selected.len()).map(|mut actor| {
                actor.owner = global_actor(actor.owner as usize) as u32;
                actor
            }));
        grown.next_identity = compact.next_identity;
        grown.ownership.extend(compact.ownership.iter().cloned());
        let owners = entries.entries.iter().map(|record| {
            selected
                .binary_search(&(record.actor as usize))
                .expect("selected owner")
        });
        let patches = owners
            .chain(selected.len()..compact.actors.len())
            .collect::<Vec<_>>();
        for (offset, record) in entries
            .entries
            .iter()
            .filter(|record| record.request >> 24 != 1)
            .enumerate()
        {
            compact.set(
                Field::InvestmentFirm,
                selected
                    .binary_search(&(record.actor as usize))
                    .expect("selected owner"),
                (before + offset + 1) as u64,
            );
        }
        let indices = patches
            .iter()
            .map(|&index| global_actor(index) as u32)
            .collect::<Vec<_>>();
        let compact_values = &compact.values;
        let compact_count = compact.actors.len();
        let values = (0..fields)
            .flat_map(|field| {
                patches
                    .iter()
                    .map(move |&actor| compact_values[field * compact_count + actor])
            })
            .collect::<Vec<_>>();
        Ok(Self {
            metadata: grown,
            readback_bytes: gathered.readback_bytes,
            formations,
            indices: device.tensor(&indices),
            values: device.tensor(bytemuck::cast_slice(&encode(&values))),
        })
    }

    pub fn indices(&self) -> &CubeTensor {
        &self.indices
    }

    pub fn grow(
        &self,
        device: &Device,
        previous: &CubeTensor,
        fields: usize,
        stride: usize,
    ) -> CubeTensor {
        let before = previous.meta.num_elements() / (fields * TICK_WORDS);
        let grown = match stride == before {
            true => previous.clone(),
            false => {
                let grown = device.state_buffer(stride * fields * TICK_WORDS);
                let width = if before.is_multiple_of(2) && stride.is_multiple_of(2) {
                    4
                } else {
                    2
                };
                let old_width = before * TICK_WORDS / width;
                let new_width = stride * TICK_WORDS / width;
                let groups = new_width.div_ceil(device.workgroup as usize * 4);
                let grid = CubeCount::Static(
                    groups.min(65535) as u32,
                    fields.min(65535) as u32,
                    groups.div_ceil(65535) as u32,
                );
                let dimensions = CubeDim::new_1d(device.workgroup);
                grow_fields::launch(
                    &device.client,
                    grid,
                    dimensions,
                    width,
                    previous.buffer_arg(),
                    grown.buffer_arg(),
                    old_width as u32,
                    new_width as u32,
                );
                grown
            }
        };
        transfer(
            device,
            &self.values,
            &grown,
            &self.indices,
            stride,
            Transfer::Scatter,
        );
        grown
    }
}

#[cube(launch)]
fn grow_fields<N: Size>(
    source: &[Vector<u32, N>],
    target: &mut [Vector<u32, N>],
    old_width: u32,
    new_width: u32,
) {
    let mut field = CUBE_POS_Y;
    let fields = target.len() / new_width as usize;
    let block = CUBE_POS_X + CUBE_POS_Z * CUBE_COUNT_X;
    while (field as usize) < fields {
        #[unroll]
        for part in 0..4u32 {
            let column = (block * 4 + part) * CUBE_DIM + UNIT_POS;
            if column < new_width {
                let mut value = Vector::<u32, N>::new(0);
                if column < old_width {
                    value = source[(field * old_width + column) as usize];
                }
                target[(field * new_width + column) as usize] = value;
            }
        }
        field += CUBE_COUNT_Y;
    }
}

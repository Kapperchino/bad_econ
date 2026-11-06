use super::*;
use crate::domain::Good;
use crate::simulation::{decode, encode, enterprise, layout::TICK_WORDS, words::Ticks};
use std::collections::BTreeSet;

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
        let sources = requests.iter().all(|record| match record.request >> 24 {
            8 | 9 => (1..=actors as u32).contains(&(record.request & 0x00ffffff)),
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
        (1..=Good::COUNT as u32).contains(&action) || matches!(action >> 24, 8 | 9)
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
                RegistryPlan::new(scenario, metadata, migrations, &requests.transitions)
                    .map(|plan| plan.map(|plan| Self { requests, plan }))
            })
            .transpose()
            .map(Option::flatten)
    }
}

pub(in crate::simulation) struct EntryBatch {
    pub metadata: State,
    pub readback_bytes: u64,
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
        next_identity: u64,
    ) -> Result<Self> {
        let selected = entries
            .entries
            .iter()
            .flat_map(|record| {
                let source = match record.request >> 24 {
                    8 | 9 => Some((record.request & 0x00ffffff) as usize - 1),
                    _ => None,
                };
                std::iter::once(record.actor as usize).chain(source)
            })
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
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
            false => Err(anyhow::anyhow!(
                "entry staging exceeds the device memory budget"
            ))?,
        }
        let indices = device.tensor(
            &selected
                .iter()
                .map(|&index| index as u32)
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
            day: metadata.day,
            actors: selected
                .iter()
                .map(|&index| metadata.actors.at(index))
                .collect(),
            values: decode(&words),
            ownership: Vec::new(),
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
                kind @ (8 | 9) => {
                    let source = (record.request & 0x00ffffff) as usize - 1;
                    (kind << 24)
                        + selected.binary_search(&source).expect("selected source") as u32
                        + 1
                }
                _ => record.request,
            };
        }
        enterprise::enter(&mut compact, scenario, &compact_requests)?;
        let mut grown = metadata.clone();
        let before = metadata.actors.len();
        grown
            .actors
            .extend(compact.actors.iter_from(selected.len()).map(|mut actor| {
                actor.owner = selected[actor.owner as usize] as u32;
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
        for (offset, record) in entries.entries.iter().enumerate() {
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
            .map(|&index| match index < selected.len() {
                true => selected[index] as u32,
                false => (before + index - selected.len()) as u32,
            })
            .collect::<Vec<_>>();
        let compact_values = &compact.values;
        let compact_count = compact.actors.len();
        let values = (0..scenario.fields())
            .flat_map(|field| {
                patches
                    .iter()
                    .map(move |&actor| compact_values[field * compact_count + actor])
            })
            .collect::<Vec<_>>();
        Ok(Self {
            metadata: grown,
            readback_bytes,
            indices: device.tensor(&indices),
            values: device.tensor(bytemuck::cast_slice(&encode(&values))),
        })
    }

    pub fn grow(
        &self,
        device: &Device,
        previous: &CubeTensor,
        before: usize,
        fields: usize,
    ) -> CubeTensor {
        let grown = device.empty(self.metadata.actors.len() * fields * TICK_WORDS);
        transfer(
            device,
            previous,
            &grown,
            &self.indices,
            before,
            Transfer::Grow,
        );
        transfer(
            device,
            &self.values,
            &grown,
            &self.indices,
            self.metadata.actors.len(),
            Transfer::Scatter,
        );
        grown
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(super) enum Transfer {
    Gather,
    Grow,
    Scatter,
}

#[derive(CubeType, CubeLaunch)]
struct TransferInput {
    source: Box<[Ticks]>,
    target: Box<[Ticks]>,
    indices: Box<[u32]>,
    actors: u32,
    #[cube(comptime)]
    stage: Transfer,
}

struct RegistryTransfers;
struct TransferRows;

impl System for RegistryTransfers {
    type Input = TransferInput;
    type Passes = gpu_passes![TransferRows];
}

#[cube]
impl GpuPass<TransferInput> for TransferRows {
    type Output = ();

    fn execute(mut input: TransferInput) {
        transfer_rows(
            &input.source,
            &mut input.target,
            &input.indices,
            input.actors,
            comptime!(input.stage),
        );
    }
}

#[cube]
fn transfer_rows(
    source: &[Ticks],
    target: &mut [Ticks],
    indices: &[u32],
    actors: u32,
    #[comptime] stage: Transfer,
) {
    let id = ABSOLUTE_POS;
    match stage {
        Transfer::Gather => {
            if id < target.len() {
                let field = id / indices.len();
                let actor = indices[id % indices.len()] as usize;
                target[id] = source[field * actors as usize + actor];
            }
        }
        Transfer::Grow => {
            if id < target.len() {
                let fields = source.len() / actors as usize;
                let count = target.len() / fields;
                let actor = id % count;
                let field = id / count;
                let mut value = Ticks::new(0);
                if actor < actors as usize {
                    value = source[field * actors as usize + actor];
                }
                target[id] = value;
            }
        }
        Transfer::Scatter => {
            if id < source.len() {
                let field = id / indices.len();
                let actor = indices[id % indices.len()] as usize;
                target[field * actors as usize + actor] = source[id];
            }
        }
    }
}

pub(super) fn transfer(
    device: &Device,
    source: &CubeTensor,
    target: &CubeTensor,
    indices: &CubeTensor,
    actors: usize,
    stage: Transfer,
) {
    let argument = |tensor: &CubeTensor| tensor.clone().into_buffer_arg();
    let count = match stage {
        Transfer::Scatter => source.meta.shape()[0],
        _ => target.meta.shape()[0],
    };
    let cube_count = device.count(count / TICK_WORDS);
    RegistryTransfers::launch(
        device,
        cube_count,
        CubeDim::new_1d(device.workgroup),
        TransferInputLaunch {
            source: argument(source),
            target: argument(target),
            indices: argument(indices),
            actors: actors as u32,
            stage,
        },
    );
}

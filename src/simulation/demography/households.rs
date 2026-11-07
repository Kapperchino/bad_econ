use super::*;
use crate::simulation::{enterprise::HouseholdFormation, pass::GpuPass, system::gpu_passes};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FormationRecord {
    parent: u32,
    household: u32,
    person: [u32; 2],
}

impl FormationRecord {
    const WORDS: usize = size_of::<Self>() / WORD_BYTES;
}

impl DemographyRuntime {
    pub(in crate::simulation) fn form_households(
        &self,
        device: &Device,
        formations: &[HouseholdFormation],
        actors: usize,
    ) -> Result<Self> {
        let required =
            (self.count.max(1) * PersonLayout::WORDS * 2 + actors * 3) as u64 * WORD_BYTES as u64;
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
                "household formation exceeds the device memory budget"
            ))?,
        }
        let mut range_counts = self.range_counts.clone();
        range_counts.resize(actors, 0);
        for formation in formations {
            range_counts[formation.parent] = range_counts[formation.parent]
                .checked_sub(1)
                .context("household formation requires a person")?;
            range_counts[formation.household] = 1;
        }
        let mut offset = 0;
        let ranges = range_counts
            .iter()
            .flat_map(|&count| {
                let range = [offset, count];
                offset += count;
                range
            })
            .collect::<Vec<_>>();
        let mut records = formations
            .iter()
            .map(|formation| FormationRecord {
                parent: formation.parent as u32,
                household: formation.household as u32,
                person: [formation.person as u32, (formation.person >> 32) as u32],
            })
            .collect::<Vec<_>>();
        records.sort_by_key(|record| record.parent);
        let records = device.tensor(bytemuck::cast_slice(&records));
        let faults = device.tensor(&[0]);
        let next = Self {
            people: device.empty(self.count.max(1) * PersonLayout::WORDS),
            audited: device.empty(self.count.max(1) * PersonLayout::WORDS),
            ranges: device.tensor(&ranges),
            births: device.tensor(&vec![0; actors]),
            range_counts,
            ..self.clone()
        };
        HouseholdFormationSystem::launch(
            device,
            device.count(self.range_counts.len()),
            CubeDim::new_1d(device.workgroup),
            FormationInputLaunch {
                source: self.people.clone().into_buffer_arg(),
                target: next.people.clone().into_buffer_arg(),
                previous: self.ranges.clone().into_buffer_arg(),
                ranges: next.ranges.clone().into_buffer_arg(),
                formations: records.into_buffer_arg(),
                faults: faults.clone().into_buffer_arg(),
            },
        );
        match device.read(&faults)?[0] {
            0 => Ok(next),
            _ => Err(anyhow!(
                "household formation requires a living household member"
            )),
        }
    }
}

#[derive(CubeType, CubeLaunch)]
struct FormationInput {
    source: Box<[u32]>,
    target: Box<[u32]>,
    previous: Box<[u32]>,
    ranges: Box<[u32]>,
    formations: Box<[u32]>,
    faults: Box<[Atomic<u32>]>,
}

struct HouseholdFormationSystem;
struct MoveMembers;

impl System for HouseholdFormationSystem {
    type Input = FormationInput;
    type Passes = gpu_passes![MoveMembers];
}

#[cube]
impl GpuPass<FormationInput> for MoveMembers {
    type Output = ();

    fn execute(mut input: FormationInput) {
        let household = ABSOLUTE_POS;
        if household < input.previous.len() / 2 {
            let mut low = 0usize;
            let mut high = input.formations.len() / FormationRecord::WORDS;
            while low < high {
                let middle = (low + high) / 2;
                let preceding =
                    (input.formations[middle * FormationRecord::WORDS] as usize) < household;
                low = select(preceding, middle + 1, low);
                high = select(preceding, high, middle);
            }
            let record = min(low, input.formations.len() / FormationRecord::WORDS - 1)
                * FormationRecord::WORDS;
            let destination = select(
                input.formations[record] as usize == household,
                input.formations[record + 1] as usize,
                household,
            );
            let person_low = input.formations[record + 2];
            let person_high = input.formations[record + 3];
            let start = input.previous[household * 2] as usize;
            let count = input.previous[household * 2 + 1] as usize;
            let mut retained = 0usize;
            let mut moved = 0usize;
            for member in 0..count {
                let source = (start + member) * PersonLayout::WORDS;
                let selected = destination != household
                    && input.source[source + PersonLayout::IDENTITY] == person_low
                    && input.source[source + PersonLayout::IDENTITY + 1] == person_high;
                let owner = select(selected, destination, household);
                let offset = select(selected, 0, retained);
                let target = (input.ranges[owner * 2] as usize + offset) * PersonLayout::WORDS;
                let valid = input.ranges[owner * 2 + 1] as usize > offset;
                if valid {
                    #[unroll]
                    for word in 0..PersonLayout::WORDS {
                        input.target[target + word] = input.source[source + word];
                    }
                    input.target[target + PersonLayout::HOUSEHOLD] = owner as u32;
                }
                if !valid || (selected && input.source[source + PersonLayout::CAUSE] != 0) {
                    input.faults[0].fetch_or(1);
                }
                retained += usize::cast_from(!selected);
                moved += usize::cast_from(selected);
            }
            if moved != usize::cast_from(destination != household) {
                input.faults[0].fetch_or(1);
            }
        }
    }
}

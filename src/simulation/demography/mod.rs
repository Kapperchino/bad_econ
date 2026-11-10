mod households;
mod kernels;
mod systems;
use super::registry::records::PendingRecord;
use super::{Device, layout::WORD_BYTES, system::System};
use crate::domain::{Phase, Scenario, State, demography::*};
use anyhow::{Context, Result, anyhow};
use burn_wgpu::CubeTensor;
use bytemuck::{Pod, Zeroable};
use cubecl::prelude::*;
use std::{collections::HashMap, mem::offset_of};
use systems::*;

const OBSERVATION_SHARDS: usize = 1024;
const OBSERVATION_METRICS: usize = 14;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PersonLayout {
    identity: [u32; 2],
    household: u32,
    age: u32,
    sex: u32,
    cause: u32,
    death_day: u32,
    employer: u32,
    work: u32,
    fatigue: u32,
    exposure: u32,
    cooldown: u32,
}
impl PersonLayout {
    const WORDS: usize = size_of::<Self>() / WORD_BYTES;
    const IDENTITY: usize = offset_of!(Self, identity) / WORD_BYTES;
    const HOUSEHOLD: usize = offset_of!(Self, household) / WORD_BYTES;
    const AGE: usize = offset_of!(Self, age) / WORD_BYTES;
    const SEX: usize = offset_of!(Self, sex) / WORD_BYTES;
    const CAUSE: usize = offset_of!(Self, cause) / WORD_BYTES;
    const DEATH_DAY: usize = offset_of!(Self, death_day) / WORD_BYTES;
    const EMPLOYER: usize = offset_of!(Self, employer) / WORD_BYTES;
    const WORK: usize = offset_of!(Self, work) / WORD_BYTES;
    const FATIGUE: usize = offset_of!(Self, fatigue) / WORD_BYTES;
    const EXPOSURE: usize = offset_of!(Self, exposure) / WORD_BYTES;
    const COOLDOWN: usize = offset_of!(Self, cooldown) / WORD_BYTES;
}

#[derive(CubeType, CubeLaunch, Clone, Copy)]
struct DemographyParameters {
    minimum_age: u32,
    retirement_age: u32,
    births: u32,
    spacing: u32,
    work_food: u32,
    reference_minutes: u32,
    famine: u32,
    recovery: u32,
}
impl DemographyParameters {
    fn launch(self) -> DemographyParametersLaunch {
        DemographyParametersLaunch {
            minimum_age: self.minimum_age,
            retirement_age: self.retirement_age,
            births: self.births,
            spacing: self.spacing,
            work_food: self.work_food,
            reference_minutes: self.reference_minutes,
            famine: self.famine,
            recovery: self.recovery,
        }
    }
}
#[derive(CubeType, CubeLaunch)]
struct DemographyPolicy {
    parameters: DemographyParameters,
    food: Box<[u32]>,
    background: Box<[u32]>,
    occupational: Box<[u32]>,
    vulnerability: Box<[u32]>,
}
#[derive(Clone)]
struct Policy {
    parameters: DemographyParameters,
    food: CubeTensor,
    background: CubeTensor,
    occupational: CubeTensor,
    vulnerability: CubeTensor,
}
impl Policy {
    fn launch(&self) -> DemographyPolicyLaunch {
        DemographyPolicyLaunch {
            parameters: self.parameters.launch(),
            food: self.food.clone().into_buffer_arg(),
            background: self.background.clone().into_buffer_arg(),
            occupational: self.occupational.clone().into_buffer_arg(),
            vulnerability: self.vulnerability.clone().into_buffer_arg(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Stage {
    Observation,
    Requirements,
    Mortality,
}

#[derive(Clone)]
pub struct DemographyRuntime {
    people: CubeTensor,
    audited: CubeTensor,
    ranges: CubeTensor,
    policy: Policy,
    births: CubeTensor,
    count: usize,
    range_counts: Vec<u32>,
    initial_population: u64,
    seed: u64,
}
impl DemographyRuntime {
    pub fn new(device: &Device, scenario: &Scenario, state: &State) -> Result<Self> {
        let indexes = state
            .actors
            .iter()
            .enumerate()
            .map(|(index, actor)| (actor.identity, index as u32))
            .collect::<HashMap<_, _>>();
        let mut encoded = state
            .people
            .iter()
            .map(|person| {
                let household = *indexes
                    .get(&person.household)
                    .context("unknown person household")?;
                let (cause, death_day) = match person.state {
                    LivingState::Living => (0, 0),
                    LivingState::Deceased { cause, day } => (cause as u32, u32::try_from(day)?),
                };
                Ok(PersonLayout {
                    identity: [person.identity as u32, (person.identity >> 32) as u32],
                    household,
                    age: person.age_days,
                    sex: person.sex as u32,
                    cause,
                    death_day,
                    employer: person
                        .employer
                        .map(|employer| indexes.get(&employer).copied().context("unknown employer"))
                        .transpose()?
                        .map_or(0, |index| index + 1),
                    work: person.work_minutes,
                    fatigue: person.fatigue,
                    exposure: person.famine_exposure,
                    cooldown: person.birth_cooldown,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        encoded.sort_by_key(|person| (person.household, person.identity[1], person.identity[0]));
        let mut ranges = vec![0; state.actors.len() * 2];
        for (index, person) in encoded.iter().enumerate() {
            let base = person.household as usize * 2;
            if ranges[base + 1] == 0 {
                ranges[base] = index as u32;
            }
            ranges[base + 1] += 1;
        }
        let count = encoded.len();
        encoded.resize(count.max(1), PersonLayout::zeroed());
        let people = device.tensor(bytemuck::cast_slice(&encoded));
        let audited = device.tensor(bytemuck::cast_slice(&encoded));
        let config = &scenario.config().demography;
        let mortality = &config.mortality;
        let policy = Policy {
            parameters: DemographyParameters {
                minimum_age: config.minimum_work_age_years * DAYS_PER_YEAR,
                retirement_age: config.retirement_age_years * DAYS_PER_YEAR,
                births: config.annual_births_per_million_fertile_women,
                spacing: config.birth_spacing_days,
                work_food: config.work_food_per_mille,
                reference_minutes: mortality.reference_work_minutes_per_year,
                famine: mortality.famine_daily_ppm_at_thirty_days,
                recovery: mortality.exposure_recovery_per_day,
            },
            food: device.tensor(&config.food_requirement_per_mille),
            background: device.tensor(&mortality.background_annual_ppm),
            occupational: device.tensor(&mortality.occupational_annual_ppm),
            vulnerability: device.tensor(&mortality.famine_vulnerability_per_mille),
        };
        Ok(Self {
            people,
            audited,
            ranges: device.tensor(&ranges),
            policy,
            births: device.tensor(&vec![0; state.actors.len()]),
            count,
            range_counts: ranges.chunks_exact(2).map(|range| range[1]).collect(),
            initial_population: state.initial_population,
            seed: scenario.config().seed,
        })
    }
    fn launch(
        &self,
        device: &Device,
        state: &CubeTensor,
        actors: &CubeTensor,
        config: &super::configuration::Configuration,
        day: u64,
        stage: Stage,
    ) -> Result<()> {
        let argument = |tensor: &CubeTensor| tensor.clone().into_buffer_arg();
        let count = self.births.meta.shape()[0];
        let cube_count = device.count(count);
        let samples = match stage {
            Stage::Mortality => super::random::samples(
                device,
                self.seed,
                day,
                super::random::Stream::Demography,
                [self.count, 5],
            )?,
            _ => super::tensor::empty(device, [0]),
        };
        HouseholdDemography::launch(
            device,
            cube_count,
            CubeDim::new_1d(device.workgroup),
            DemographyInputLaunch {
                state: argument(state),
                actors: argument(actors),
                config: config.launch(),
                people: argument(&self.people),
                ranges: argument(&self.ranges),
                policy: self.policy.launch(),
                births: argument(&self.births),
                samples: samples.clone().into_buffer_arg(),
                day: u32::try_from(day)?,
                stage,
            },
        );
        Ok(())
    }
    pub fn before(
        &self,
        device: &Device,
        state: &CubeTensor,
        actors: &CubeTensor,
        config: &super::configuration::Configuration,
        phase: Phase,
        day: u64,
    ) -> Result<()> {
        match phase {
            Phase::Observation => {
                self.launch(device, state, actors, config, day, Stage::Observation)
            }
            Phase::PhysicalClose => {
                self.launch(device, state, actors, config, day, Stage::Requirements)
            }
            _ => Ok(()),
        }
    }
    pub fn after(
        &self,
        device: &Device,
        state: &CubeTensor,
        actors: &CubeTensor,
        config: &super::configuration::Configuration,
        phase: Phase,
        day: u64,
    ) -> Result<()> {
        match phase {
            Phase::PhysicalClose => {
                self.launch(device, state, actors, config, day, Stage::Mortality)
            }
            _ => Ok(()),
        }
    }
    pub fn copy(&self, device: &Device, direction: super::CopyDirection) {
        let (source, target) = match direction {
            super::CopyDirection::Save => (&self.people, &self.audited),
            super::CopyDirection::Restore => (&self.audited, &self.people),
        };
        super::tensor::copy_contiguous(device, source, target);
    }
    pub fn download(
        &self,
        device: &Device,
        state: &State,
        direction: super::CopyDirection,
    ) -> Result<Vec<Person>> {
        let source = match direction {
            super::CopyDirection::Save => &self.people,
            super::CopyDirection::Restore => &self.audited,
        };
        let words = device.read(source)?;
        bytemuck::cast_slice::<u32, PersonLayout>(&words)
            .iter()
            .take(self.count)
            .map(|person| {
                let cause = match person.cause {
                    0 => LivingState::Living,
                    1 => LivingState::Deceased {
                        day: person.death_day.into(),
                        cause: DeathCause::Background,
                    },
                    2 => LivingState::Deceased {
                        day: person.death_day.into(),
                        cause: DeathCause::Occupational,
                    },
                    3 => LivingState::Deceased {
                        day: person.death_day.into(),
                        cause: DeathCause::Famine,
                    },
                    _ => Err(anyhow!("invalid mortality cause"))?,
                };
                Ok(Person {
                    identity: u64::from(person.identity[0]) | (u64::from(person.identity[1]) << 32),
                    household: state.actors.at(person.household as usize).identity,
                    age_days: person.age,
                    sex: match person.sex {
                        0 => Sex::Female,
                        _ => Sex::Male,
                    },
                    state: cause,
                    employer: person
                        .employer
                        .checked_sub(1)
                        .map(|index| state.actors.at(index as usize).identity),
                    work_minutes: person.work,
                    fatigue: person.fatigue,
                    famine_exposure: person.exposure,
                    birth_cooldown: person.cooldown,
                })
            })
            .collect()
    }
    pub fn observation(
        &self,
        device: &Device,
        state: &CubeTensor,
        config: &super::configuration::Configuration,
    ) -> Result<PopulationObservation> {
        let shards = device.tensor(&[0; OBSERVATION_SHARDS * OBSERVATION_METRICS]);
        let output = device.empty(OBSERVATION_METRICS);
        let argument = |tensor: &CubeTensor| tensor.clone().into_buffer_arg();
        let cube_count = device.count(self.count);
        PopulationCounting::launch(
            device,
            cube_count,
            CubeDim::new_1d(device.workgroup),
            PopulationInputLaunch {
                state: argument(state),
                config: config.launch(),
                people: argument(&self.people),
                policy: self.policy.launch(),
                output: argument(&shards),
                count: self.count as u32,
            },
        );
        let input = cubecl::std::tensor::TensorHandle::new_contiguous(
            [OBSERVATION_SHARDS, OBSERVATION_METRICS],
            shards.handle,
            <u32 as cubecl::prelude::Scalar>::elem_type_native(),
        );
        let sums = super::reduction::reduce(
            device,
            &input,
            0,
            cubek::reduce::components::instructions::ReduceOperationConfig::Sum,
            <u64 as cubecl::prelude::Scalar>::elem_type_native(),
        )?;
        PopulationTotals::launch(
            device,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(device.workgroup),
            PopulationTotalsInputLaunch {
                sums: sums.into_arg().into_buffer_arg(),
                output: argument(&output),
            },
        );
        let totals = device.read(&output)?;
        Ok(PopulationObservation {
            occupations: std::array::from_fn(|index| u64::from(totals[8 + index])),
            living: totals[0].into(),
            dependents: totals[1].into(),
            births: self.count as u64 - self.initial_population,
            deaths: [totals[2].into(), totals[3].into(), totals[4].into()],
            work_minutes: totals[5].into(),
            famine_exposure: u64::from(totals[6]) + (u64::from(totals[7]) << 32),
        })
    }
    pub(super) fn grow_births(
        &self,
        device: &Device,
        births: &[PendingRecord],
        next_identity: u64,
        scratch: &CubeTensor,
    ) -> Result<Self> {
        let count = self.count + births.iter().filter(|record| record.birth > 0).count();
        let actors = self.range_counts.len();
        let growth =
            count as u64 * ((PersonLayout::WORDS * 2 + 5) * WORD_BYTES) as u64 + actors as u64 * 8;
        device.reclaim_memory(growth)?;
        let required = growth
            + device
                .client
                .memory_report(cubecl::MemoryScope::Device)
                .usage()
                .bytes_reserved;
        match count <= u32::MAX as usize && required <= device.memory_budget.bytes() {
            true => (),
            false => Err(anyhow!(
                "person registry capacity exhausted: {required} bytes exceed device budget"
            ))?,
        }
        let mut range_counts = self.range_counts.clone();
        for record in births.iter().filter(|record| record.birth > 0) {
            range_counts[record.actor as usize] += 1;
        }
        let grown = Self {
            people: device.empty(count.max(1) * PersonLayout::WORDS),
            audited: device.empty(count.max(1) * PersonLayout::WORDS),
            ranges: device.empty(actors * 2),
            policy: self.policy.clone(),
            births: self.births.clone(),
            count,
            range_counts,
            initial_population: self.initial_population,
            seed: self.seed,
        };
        let argument = |tensor: &CubeTensor| tensor.clone().into_buffer_arg();
        let cube_count = device.count(actors);
        BirthCounting::launch(
            device,
            cube_count.clone(),
            CubeDim::new_1d(device.workgroup),
            BirthCountInputLaunch {
                births: argument(&self.births),
                scratch: argument(scratch),
            },
        );
        super::parallel::scan(
            device,
            scratch,
            actors,
            actors.div_ceil(device.workgroup as usize),
        );
        BirthRanges::launch(
            device,
            cube_count.clone(),
            CubeDim::new_1d(device.workgroup),
            BirthRangeInputLaunch {
                old: argument(&self.ranges),
                births: argument(&self.births),
                scratch: argument(scratch),
                ranges: argument(&grown.ranges),
            },
        );
        PopulationGrowth::launch(
            device,
            device.count(self.count * PersonLayout::WORDS / 4),
            CubeDim::new_1d(device.workgroup),
            PopulationGrowthInputLaunch {
                source: argument(&self.people),
                target: argument(&grown.people),
                old_ranges: argument(&self.ranges),
                new_ranges: argument(&grown.ranges),
                people: self.count as u32,
            },
        );
        NewbornInitialization::launch(
            device,
            cube_count,
            CubeDim::new_1d(device.workgroup),
            NewbornInputLaunch {
                target: argument(&grown.people),
                old_ranges: argument(&self.ranges),
                new_ranges: argument(&grown.ranges),
                births: argument(&self.births),
                next_low: next_identity as u32,
                next_high: (next_identity >> 32) as u32,
            },
        );
        device.reclaim_memory(0)?;
        match device
            .client
            .memory_report(cubecl::MemoryScope::Device)
            .usage()
            .bytes_reserved
            <= device.memory_budget.bytes()
        {
            true => Ok(grown),
            false => Err(anyhow!("person registry allocation exceeded device budget")),
        }
    }
    #[cfg(test)]
    pub fn reference_births(
        &self,
        device: &Device,
        births: &[u32],
        next_identity: u64,
    ) -> Result<Self> {
        let count = self.count + births.iter().filter(|&&birth| birth > 0).count();
        let growth = count as u64 * ((PersonLayout::WORDS * 2 + 5) * WORD_BYTES) as u64
            + births.len() as u64 * 8;
        device.reclaim_memory(growth)?;
        let required = growth
            + device
                .client
                .memory_report(cubecl::MemoryScope::Device)
                .usage()
                .bytes_reserved;
        match count <= u32::MAX as usize && required <= device.memory_budget.bytes() {
            true => (),
            false => Err(anyhow!(
                "person registry capacity exhausted: {required} bytes exceed device budget"
            ))?,
        }
        let mut offset = 0u32;
        let mut ranges = Vec::with_capacity(births.len() * 2);
        let range_counts = self
            .range_counts
            .iter()
            .zip(births)
            .map(|(&count, &birth)| count + u32::from(birth > 0))
            .collect::<Vec<_>>();
        for &count in &range_counts {
            ranges.extend([offset, count]);
            offset += count;
        }
        let grown = Self {
            people: device.empty(count.max(1) * PersonLayout::WORDS),
            audited: device.empty(count.max(1) * PersonLayout::WORDS),
            ranges: device.tensor(&ranges),
            policy: self.policy.clone(),
            births: self.births.clone(),
            count,
            range_counts,
            initial_population: self.initial_population,
            seed: self.seed,
        };
        let argument = |tensor: &CubeTensor| tensor.clone().into_buffer_arg();
        let cube_count = device.count(births.len());
        kernels::original_grow::launch(
            &device.client,
            cube_count,
            CubeDim::new_1d(device.workgroup),
            argument(&self.people),
            argument(&grown.people),
            argument(&self.ranges),
            argument(&grown.ranges),
            argument(&self.births),
            next_identity as u32,
            (next_identity >> 32) as u32,
        );
        device.reclaim_memory(0)?;
        match device
            .client
            .memory_report(cubecl::MemoryScope::Device)
            .usage()
            .bytes_reserved
            <= device.memory_budget.bytes()
        {
            true => Ok(grown),
            false => Err(anyhow!("person registry allocation exceeded device budget")),
        }
    }
    pub fn extend_firms(&mut self, device: &Device, count: usize) {
        self.range_counts.resize(count, 0);
        let ranges = device.empty(count * 2);
        let births = device.empty(count);
        FirmRegistryExtension::launch(
            device,
            device.count(count),
            CubeDim::new_1d(device.workgroup),
            FirmExtensionInputLaunch {
                previous: self.ranges.clone().into_buffer_arg(),
                ranges: ranges.clone().into_buffer_arg(),
                births: births.clone().into_buffer_arg(),
                people: self.count as u32,
            },
        );
        self.ranges = ranges;
        self.births = births;
    }
    pub fn birth_buffer(&self) -> &CubeTensor {
        &self.births
    }
}

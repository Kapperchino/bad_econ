mod cohorts;
mod households;
mod kernels;
#[cfg(test)]
mod remapping_tests;
mod systems;
use super::registry::records::PendingRecord;
use super::{
    Device,
    layout::WORD_BYTES,
    system::{AsBufferArg, System},
};
use crate::domain::{Phase, Scenario, State, demography::*};
use anyhow::{Context, Result, anyhow};
use burn_wgpu::CubeTensor;
use bytemuck::{Pod, Zeroable};
use cubecl::prelude::*;
use std::{
    collections::HashMap,
    mem::offset_of,
    sync::{Arc, Mutex},
};
use systems::*;

const OBSERVATION_SHARDS: usize = 1024;
const OBSERVATION_METRICS: usize = 16;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PersonLayout {
    identity: [u32; 2],
    household: u32,
    age: u32,
    sex_and_cold_exposure: u32,
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
    const SEX_AND_COLD_EXPOSURE: usize = offset_of!(Self, sex_and_cold_exposure) / WORD_BYTES;
    const CAUSE: usize = offset_of!(Self, cause) / WORD_BYTES;
    const DEATH_DAY: usize = offset_of!(Self, death_day) / WORD_BYTES;
    const EMPLOYER: usize = offset_of!(Self, employer) / WORD_BYTES;
    const WORK: usize = offset_of!(Self, work) / WORD_BYTES;
    const FATIGUE: usize = offset_of!(Self, fatigue) / WORD_BYTES;
    const EXPOSURE: usize = offset_of!(Self, exposure) / WORD_BYTES;
    const COOLDOWN: usize = offset_of!(Self, cooldown) / WORD_BYTES;

    fn living_state(&self) -> Result<LivingState> {
        let cause = match self.cause {
            0 => None,
            1 => Some(DeathCause::Background),
            2 => Some(DeathCause::Occupational),
            3 => Some(DeathCause::Famine),
            4 => Some(DeathCause::Cold),
            _ => Err(anyhow!("invalid mortality cause"))?,
        };
        Ok(
            cause.map_or(LivingState::Living, |cause| LivingState::Deceased {
                day: self.death_day.into(),
                cause,
            }),
        )
    }
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
    cold: u32,
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
            cold: self.cold,
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
    cold_vulnerability: Box<[u32]>,
}
#[derive(Clone)]
struct Policy {
    parameters: DemographyParameters,
    food: CubeTensor,
    background: CubeTensor,
    occupational: CubeTensor,
    vulnerability: CubeTensor,
    cold_vulnerability: CubeTensor,
}
impl Policy {
    fn launch(&self) -> DemographyPolicyLaunch {
        DemographyPolicyLaunch {
            parameters: self.parameters.launch(),
            food: self.food.buffer_arg(),
            background: self.background.buffer_arg(),
            occupational: self.occupational.buffer_arg(),
            vulnerability: self.vulnerability.buffer_arg(),
            cold_vulnerability: self.cold_vulnerability.buffer_arg(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum PopulationModel {
    Cohorts,
    Individuals,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Stage {
    Observation,
    Requirements,
    Diet,
    Mortality,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PersonSnapshot {
    Current,
    Dirty,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum WorkerRemapping {
    Births,
    Formations,
}

#[derive(Clone, Copy)]
enum WorkerSelection {
    Identity(u64),
    Eligible,
    Inactive,
}

#[derive(Clone)]
pub struct DemographyRuntime {
    people: CubeTensor,
    audited: CubeTensor,
    snapshot: Arc<Mutex<PersonSnapshot>>,
    ranges: CubeTensor,
    policy: Policy,
    births: CubeTensor,
    count: usize,
    range_counts: Vec<u32>,
    initial_population: u64,
    seed: u64,
}
impl DemographyRuntime {
    pub fn new(device: &Device, scenario: &Scenario, state: &mut State) -> Result<Self> {
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
                    sex_and_cold_exposure: person.sex as u32 | (person.cold_exposure << 1),
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
        let reordered = encoded.iter().zip(&state.people).any(|(person, original)| {
            u64::from(person.identity[0]) | (u64::from(person.identity[1]) << 32)
                != original.identity
        });
        let workers = state
            .actors
            .iter()
            .enumerate()
            .map(|(index, actor)| {
                let selected = state.get(crate::domain::Field::Worker, index);
                match (actor.kind, selected) {
                    (0, selected) if selected != u64::from(u32::MAX) => {
                        match state
                            .people
                            .get(selected as usize)
                            .filter(|person| person.household == actor.identity)
                        {
                            Some(person) => WorkerSelection::Identity(person.identity),
                            None if reordered
                                || state.day > 0
                                || state.get(crate::domain::Field::ParentHousehold, index) > 0
                                || state
                                    .people
                                    .get(selected as usize)
                                    .and_then(|person| indexes.get(&person.household))
                                    .is_some_and(|&household| {
                                        state.get(
                                            crate::domain::Field::ParentHousehold,
                                            household as usize,
                                        ) == actor.identity
                                    }) =>
                            {
                                WorkerSelection::Eligible
                            }
                            None => WorkerSelection::Inactive,
                        }
                    }
                    _ => WorkerSelection::Inactive,
                }
            })
            .collect::<Vec<_>>();
        let mut selected = vec![None; state.actors.len()];
        let policy = &scenario.config().demography;
        for (index, person) in encoded.iter().enumerate() {
            let household = person.household as usize;
            let identity = u64::from(person.identity[0]) | (u64::from(person.identity[1]) << 32);
            let choose = match workers[household] {
                WorkerSelection::Identity(expected) => identity == expected,
                WorkerSelection::Eligible => {
                    selected[household].is_none()
                        && person.cause == 0
                        && person.age >= policy.minimum_work_age_years * DAYS_PER_YEAR
                        && person.age < policy.retirement_age_years * DAYS_PER_YEAR
                }
                WorkerSelection::Inactive => false,
            };
            if choose {
                selected[household] = Some(index as u64);
            }
        }
        for (household, worker) in workers.into_iter().enumerate() {
            match worker {
                WorkerSelection::Inactive => (),
                _ => state.set(
                    crate::domain::Field::Worker,
                    household,
                    selected[household].unwrap_or(u64::from(u32::MAX)),
                ),
            }
        }
        if reordered {
            state
                .people
                .sort_by_key(|person| (indexes[&person.household], person.identity));
        }
        let mut ranges = vec![0; state.actors.len() * 2];
        for (index, person) in encoded.iter().enumerate() {
            let base = person.household as usize * 2;
            if ranges[base + 1] == 0 {
                ranges[base] = index as u32;
            }
            ranges[base + 1] += 1;
        }
        let count = encoded.len();
        let people = device.record_tensor(&encoded);
        let audited = device.record_tensor(&encoded);
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
                cold: mortality.cold_daily_ppm_at_seven_days,
            },
            food: device.tensor(&config.food_requirement_per_mille),
            background: device.tensor(&mortality.background_annual_ppm),
            occupational: device.tensor(&mortality.occupational_annual_ppm),
            vulnerability: device.tensor(&mortality.famine_vulnerability_per_mille),
            cold_vulnerability: device.tensor(&mortality.cold_vulnerability_per_mille),
        };
        Ok(Self {
            people,
            audited,
            snapshot: Arc::new(Mutex::new(PersonSnapshot::Current)),
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
        let count = self.range_counts.len();
        let model = if self.count == 0 {
            PopulationModel::Cohorts
        } else {
            PopulationModel::Individuals
        };
        let samples = match stage {
            Stage::Mortality => super::random::samples(
                device,
                self.seed,
                day,
                super::random::Stream::Demography,
                match model {
                    PopulationModel::Individuals => [self.count, 5],
                    PopulationModel::Cohorts => [count, 1],
                },
            )?,
            _ => super::tensor::empty(device, [0]),
        };
        *self.snapshot.lock().expect("person snapshot state") = PersonSnapshot::Dirty;
        HouseholdDemography::launch_elements(
            device,
            count,
            DemographyInputLaunch {
                state: state.buffer_arg(),
                actors: actors.buffer_arg(),
                config: config.launch(),
                people: self.people.buffer_arg(),
                ranges: self.ranges.buffer_arg(),
                policy: self.policy.launch(),
                births: self.births.buffer_arg(),
                samples: samples.buffer_arg(),
                day: u32::try_from(day)?,
                stage,
                model,
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
            Phase::WorkPayment if config.food_medium_salary > 0 => {
                self.launch(device, state, actors, config, day, Stage::Diet)
            }
            Phase::PhysicalClose => {
                self.launch(device, state, actors, config, day, Stage::Mortality)
            }
            _ => Ok(()),
        }
    }
    pub fn copy(&self, device: &Device, direction: super::CopyDirection) {
        let mut snapshot = self.snapshot.lock().expect("person snapshot state");
        match (direction, *snapshot) {
            (super::CopyDirection::Save, PersonSnapshot::Dirty) => {
                super::tensor::copy_contiguous(device, &self.people, &self.audited);
            }
            (super::CopyDirection::Restore, _) => {
                super::tensor::copy_contiguous(device, &self.audited, &self.people);
            }
            (super::CopyDirection::Save, PersonSnapshot::Current) => (),
        }
        *snapshot = PersonSnapshot::Current;
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
                let living_state = person.living_state()?;
                Ok(Person {
                    identity: u64::from(person.identity[0]) | (u64::from(person.identity[1]) << 32),
                    household: state.actors.at(person.household as usize).identity,
                    age_days: person.age,
                    sex: match person.sex_and_cold_exposure & 1 {
                        0 => Sex::Female,
                        _ => Sex::Male,
                    },
                    state: living_state,
                    employer: person
                        .employer
                        .checked_sub(1)
                        .map(|index| state.actors.at(index as usize).identity),
                    work_minutes: person.work,
                    fatigue: person.fatigue,
                    famine_exposure: person.exposure,
                    cold_exposure: person.sex_and_cold_exposure >> 1,
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
        match self.count {
            0 => cohorts::observation(device, state, config, self.initial_population),
            _ => self.individual_observation(device, state, config),
        }
    }
    fn individual_observation(
        &self,
        device: &Device,
        state: &CubeTensor,
        config: &super::configuration::Configuration,
    ) -> Result<PopulationObservation> {
        let shards = device.tensor(&[0; OBSERVATION_SHARDS * OBSERVATION_METRICS]);
        let output = device.empty(OBSERVATION_METRICS);
        PopulationCounting::launch_elements(
            device,
            self.count,
            PopulationInputLaunch {
                state: state.buffer_arg(),
                config: config.launch(),
                people: self.people.buffer_arg(),
                policy: self.policy.launch(),
                output: shards.buffer_arg(),
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
                output: output.buffer_arg(),
            },
        );
        let totals = device.read(&output)?;
        let classified = [totals[0], totals[2], totals[3], totals[4]]
            .into_iter()
            .map(u64::from)
            .sum::<u64>();
        let cold_deaths = (self.count as u64)
            .checked_sub(classified)
            .context("classified population exceeds the person registry")?;
        Ok(PopulationObservation {
            occupations: std::array::from_fn(|index| u64::from(totals[8 + index])),
            living: totals[0].into(),
            dependents: totals[1].into(),
            births: self.count as u64 - self.initial_population,
            deaths: [
                totals[2].into(),
                totals[3].into(),
                totals[4].into(),
                cold_deaths,
            ],
            work_minutes: totals[5].into(),
            famine_exposure: u64::from(totals[6]) + (u64::from(totals[7]) << 32),
            cold_exposure: u64::from(totals[14]) + (u64::from(totals[15]) << 32),
        })
    }
    pub(super) fn remap_workers_from(
        &self,
        device: &Device,
        state: &CubeTensor,
        stride: u32,
        previous: &Self,
    ) {
        if !super::market_graphs::same_buffer(&self.people, &previous.people) {
            let remapping = match self
                .range_counts
                .iter()
                .skip(previous.range_counts.len())
                .any(|&count| count > 0)
            {
                true => WorkerRemapping::Formations,
                false => WorkerRemapping::Births,
            };
            WorkerRemappingSystem::launch_elements(
                device,
                self.range_counts.len(),
                WorkerRemappingInputLaunch {
                    state: state.buffer_arg(),
                    previous_people: previous.people.buffer_arg(),
                    people: self.people.buffer_arg(),
                    previous_ranges: previous.ranges.buffer_arg(),
                    ranges: self.ranges.buffer_arg(),
                    stride,
                    minimum_age: self.policy.parameters.minimum_age,
                    retirement_age: self.policy.parameters.retirement_age,
                    remapping,
                },
            );
        }
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
        let required = growth.saturating_add(device.reclaim_memory(growth)?.bytes_reserved);
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
            snapshot: Arc::new(Mutex::new(PersonSnapshot::Current)),
            ranges: device.empty(actors * 2),
            policy: self.policy.clone(),
            births: self.births.clone(),
            count,
            range_counts,
            initial_population: self.initial_population,
            seed: self.seed,
        };
        let cube_count = device.count(actors);
        BirthCounting::launch(
            device,
            cube_count.clone(),
            CubeDim::new_1d(device.workgroup),
            BirthCountInputLaunch {
                births: self.births.buffer_arg(),
                scratch: scratch.buffer_arg(),
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
                old: self.ranges.buffer_arg(),
                births: self.births.buffer_arg(),
                scratch: scratch.buffer_arg(),
                ranges: grown.ranges.buffer_arg(),
            },
        );
        PopulationGrowth::launch_elements(
            device,
            self.count * PersonLayout::WORDS / 4,
            PopulationGrowthInputLaunch {
                source: self.people.buffer_arg(),
                target: grown.people.buffer_arg(),
                audited: grown.audited.buffer_arg(),
                old_ranges: self.ranges.buffer_arg(),
                new_ranges: grown.ranges.buffer_arg(),
                people: self.count as u32,
            },
        );
        NewbornInitialization::launch(
            device,
            cube_count,
            CubeDim::new_1d(device.workgroup),
            NewbornInputLaunch {
                target: grown.people.buffer_arg(),
                audited: grown.audited.buffer_arg(),
                old_ranges: self.ranges.buffer_arg(),
                new_ranges: grown.ranges.buffer_arg(),
                births: self.births.buffer_arg(),
                next_low: next_identity as u32,
                next_high: (next_identity >> 32) as u32,
            },
        );
        match device.reclaim_memory(0)?.bytes_reserved <= device.memory_budget.bytes() {
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
        let required = growth.saturating_add(device.reclaim_memory(growth)?.bytes_reserved);
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
            snapshot: Arc::new(Mutex::new(PersonSnapshot::Dirty)),
            ranges: device.tensor(&ranges),
            policy: self.policy.clone(),
            births: self.births.clone(),
            count,
            range_counts,
            initial_population: self.initial_population,
            seed: self.seed,
        };
        let cube_count = device.count(births.len());
        kernels::original_grow::launch(
            &device.client,
            cube_count,
            CubeDim::new_1d(device.workgroup),
            self.people.buffer_arg(),
            grown.people.buffer_arg(),
            self.ranges.buffer_arg(),
            grown.ranges.buffer_arg(),
            self.births.buffer_arg(),
            next_identity as u32,
            (next_identity >> 32) as u32,
        );
        match device.reclaim_memory(0)?.bytes_reserved <= device.memory_budget.bytes() {
            true => Ok(grown),
            false => Err(anyhow!("person registry allocation exceeded device budget")),
        }
    }
    pub fn extend_firms(&mut self, device: &Device, count: usize) {
        self.range_counts.resize(count, 0);
        let ranges = device.empty(count * 2);
        let births = device.empty(count);
        FirmRegistryExtension::launch_elements(
            device,
            count,
            FirmExtensionInputLaunch {
                previous: self.ranges.buffer_arg(),
                ranges: ranges.buffer_arg(),
                births: births.buffer_arg(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::Experiment;
    use crate::simulation::Simulation;

    #[test]
    fn fused_person_growth_snapshots_preserve_shared_mutations_formations_and_restore_on_gpu()
    -> Result<()> {
        use crate::simulation::{CopyDirection, MemoryBudget, enterprise::HouseholdFormation};

        let scenario = Scenario::baseline(97, 42, Experiment::Baseline)?;
        for workgroup in [32, 256] {
            let simulation = Simulation::new(
                scenario.clone(),
                crate::simulation::tests::device(workgroup)?,
            )?;
            let actors = simulation.actors().len();
            let mut births = vec![0; actors];
            births[0] = 1;
            births[31] = 2;
            births[96] = 1;
            let records = births
                .iter()
                .enumerate()
                .filter(|(_, birth)| **birth > 0)
                .map(|(actor, &birth)| PendingRecord {
                    actor: actor as u32,
                    request: 0,
                    birth,
                })
                .collect::<Vec<_>>();
            simulation
                .device
                .write(simulation.demography.birth_buffer(), &births);
            let fused = simulation.demography.grow_births(
                &simulation.device,
                &records,
                simulation.metadata.next_identity,
                &simulation.scratch,
            )?;
            let reference = simulation.demography.reference_births(
                &simulation.device,
                &births,
                simulation.metadata.next_identity,
            )?;
            reference.copy(&simulation.device, CopyDirection::Save);
            assert!(!Arc::ptr_eq(
                &simulation.demography.snapshot,
                &fused.snapshot
            ));
            assert_eq!(
                *fused.snapshot.lock().expect("person snapshot state"),
                PersonSnapshot::Current,
            );
            let grown = simulation.device.read(&fused.people)?;
            assert_eq!(grown, simulation.device.read(&fused.audited)?);
            assert_eq!(grown, simulation.device.read(&reference.people)?);
            assert_eq!(grown, simulation.device.read(&reference.audited)?);
            let shared = fused.clone();
            assert!(Arc::ptr_eq(&shared.snapshot, &fused.snapshot));
            let reference_state = simulation
                .device
                .tensor(&simulation.device.read(&simulation.state)?);
            fused.before(
                &simulation.device,
                &simulation.state,
                &simulation.actors,
                &simulation.config,
                Phase::Observation,
                0,
            )?;
            reference.before(
                &simulation.device,
                &reference_state,
                &simulation.actors,
                &simulation.config,
                Phase::Observation,
                0,
            )?;
            assert_eq!(
                *shared.snapshot.lock().expect("person snapshot state"),
                PersonSnapshot::Dirty,
            );
            assert_eq!(
                *simulation
                    .demography
                    .snapshot
                    .lock()
                    .expect("person snapshot state"),
                PersonSnapshot::Current,
            );
            shared.copy(&simulation.device, CopyDirection::Save);
            reference.copy(&simulation.device, CopyDirection::Save);
            assert_eq!(
                *fused.snapshot.lock().expect("person snapshot state"),
                PersonSnapshot::Current,
            );
            let observed = simulation.device.read(&fused.people)?;
            assert_eq!(observed, simulation.device.read(&fused.audited)?);
            assert_eq!(observed, simulation.device.read(&reference.audited)?);
            let person = bytemuck::cast_slice::<u32, PersonLayout>(&observed)[0];
            let formations = [HouseholdFormation {
                parent: 0,
                household: actors,
                person: u64::from(person.identity[0]) | (u64::from(person.identity[1]) << 32),
            }];
            let formed = fused.form_households(&simulation.device, &formations, actors + 1)?;
            let expected =
                reference.form_households(&simulation.device, &formations, actors + 1)?;
            assert!(!Arc::ptr_eq(&formed.snapshot, &fused.snapshot));
            assert_eq!(
                *formed.snapshot.lock().expect("person snapshot state"),
                PersonSnapshot::Dirty,
            );
            formed.copy(&simulation.device, CopyDirection::Save);
            expected.copy(&simulation.device, CopyDirection::Save);
            let households = simulation.device.read(&formed.people)?;
            assert_eq!(households, simulation.device.read(&formed.audited)?);
            assert_eq!(households, simulation.device.read(&expected.audited)?);
            let mut corrupted = households.clone();
            corrupted[PersonLayout::AGE] ^= 0x1234_5678;
            simulation.device.write(&formed.people, &corrupted);
            formed.copy(&simulation.device, CopyDirection::Restore);
            assert_eq!(households, simulation.device.read(&formed.people)?);
            let mut constrained = simulation.device.clone();
            constrained.memory_budget = MemoryBudget::Reported { available_bytes: 2 };
            assert!(
                fused
                    .grow_births(
                        &constrained,
                        &records,
                        simulation.metadata.next_identity + 3,
                        &simulation.scratch,
                    )
                    .is_err()
            );
            assert_eq!(observed, simulation.device.read(&fused.people)?);
            assert_eq!(observed, simulation.device.read(&fused.audited)?);
            assert_eq!(
                *fused.snapshot.lock().expect("person snapshot state"),
                PersonSnapshot::Current,
            );
        }
        Ok(())
    }
}

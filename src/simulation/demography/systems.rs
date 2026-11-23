use super::{
    DemographyPolicy, OBSERVATION_METRICS, OBSERVATION_SHARDS, PersonLayout as P, PopulationModel,
    Stage, WorkerRemapping,
};
use crate::domain::Field;
use crate::simulation::{
    economy::configuration::{EconomyConfig, Feature},
    pass::GpuPass,
    storage::{
        state::{ActorBalances, Ledger, ReadLedger, ReadLedgerExpand},
        words::*,
    },
    system::{System, gpu_passes},
};
use cubecl::prelude::*;

#[derive(CubeType, CubeLaunch)]
pub(super) struct DemographyInput {
    pub state: Box<[u32]>,
    pub actors: Box<[u32]>,
    pub config: EconomyConfig,
    pub people: Box<[u32]>,
    pub ranges: Box<[u32]>,
    pub policy: DemographyPolicy,
    pub births: Box<[u32]>,
    pub samples: Box<[f32]>,
    pub day: u32,
    #[cube(comptime)]
    pub stage: Stage,
    #[cube(comptime)]
    pub model: PopulationModel,
}

pub(super) struct HouseholdDemography;
pub(super) struct AdvanceHouseholds;

impl System for HouseholdDemography {
    type Input = DemographyInput;
    type Passes = gpu_passes![AdvanceHouseholds];
}

#[cube]
impl GpuPass<DemographyInput> for AdvanceHouseholds {
    type Output = ();

    fn execute(mut input: DemographyInput) {
        input.advance();
    }
}

#[derive(CubeType, CubeLaunch)]
pub(super) struct PopulationInput {
    pub state: Box<[u32]>,
    pub config: EconomyConfig,
    pub people: Box<[u32]>,
    pub policy: DemographyPolicy,
    pub output: Box<[Atomic<u32>]>,
    pub count: u32,
}

pub(super) struct PopulationCounting;
pub(super) struct CountPopulation;

impl System for PopulationCounting {
    type Input = PopulationInput;
    type Passes = gpu_passes![CountPopulation];
}

#[cube]
impl GpuPass<PopulationInput> for CountPopulation {
    type Output = ();

    fn execute(mut input: PopulationInput) {
        let ledger = Ledger::new(&mut input.state, &input.config);
        let config = ledger.config;
        let index = ABSOLUTE_POS;
        if index < input.count as usize {
            let base = index * P::WORDS;
            let shard = index % OBSERVATION_SHARDS * OBSERVATION_METRICS;
            let cause = input.people[base + P::CAUSE];
            if cause == 0 {
                input.output[shard].fetch_add(1);
                let age = input.people[base + P::AGE];
                let household = input.people[base + P::HOUSEHOLD];
                let selected = ledger.number(Field::Worker, household) == index as u32;
                let firm = ledger.number(Field::InvestmentFirm, household);
                let owner =
                    firm > 0 && ledger.number(Field::EnterpriseStatus, max(firm, 1) - 1) < 4;
                let farmer = ledger.number(Field::LandPlots, household) > 0;
                let dependent = age < input.policy.parameters.minimum_age
                    || age >= input.policy.parameters.retirement_age;
                let participant = selected || comptime!(config.cohorts == Feature::Enabled);
                let mut category = select(participant, 3u32, 5u32);
                category = select(input.people[base + P::EMPLOYER] > 0, 1u32, category);
                category = select(participant && farmer, 0u32, category);
                category = select(selected && owner, 2u32, category);
                category = select(dependent, 4u32, category);
                input.output[shard + 8 + category as usize].fetch_add(1);
                input.output[shard + 1].fetch_add(u32::cast_from(dependent));
            } else {
                match cause {
                    1 => {
                        input.output[shard + 2].fetch_add(1);
                    }
                    2 => {
                        input.output[shard + 3].fetch_add(1);
                    }
                    3 => {
                        input.output[shard + 4].fetch_add(1);
                    }
                    _ => (),
                }
            }
            let work = input.people[base + P::WORK];
            if work > 0 {
                input.output[shard + 5].fetch_add(work);
            }
            let exposure = input.people[base + P::EXPOSURE];
            let mut famine_carry = 0u32;
            if exposure > 0 {
                let previous = input.output[shard + 6].fetch_add(exposure);
                famine_carry = u32::cast_from(exposure > u32::MAX - previous);
            }
            if famine_carry > 0 {
                input.output[shard + 7].fetch_add(famine_carry);
            }
            let cold_exposure = input.people[base + P::SEX_AND_COLD_EXPOSURE] >> 1;
            let mut cold_carry = 0u32;
            if cold_exposure > 0 {
                let prior_cold = input.output[shard + 14].fetch_add(cold_exposure);
                cold_carry = u32::cast_from(cold_exposure > u32::MAX - prior_cold);
            }
            if cold_carry > 0 {
                input.output[shard + 15].fetch_add(cold_carry);
            }
        }
    }
}

#[derive(CubeType, CubeLaunch)]
pub(super) struct PopulationTotalsInput {
    pub sums: Box<[u64]>,
    pub output: Box<[u32]>,
}

#[derive(CubeType, CubeLaunch)]
pub(super) struct WorkerRemappingInput {
    pub state: Box<[u32]>,
    pub previous_people: Box<[Vector<u32, Const<4>>]>,
    pub people: Box<[Vector<u32, Const<4>>]>,
    pub previous_ranges: Box<[Vector<u32, Const<2>>]>,
    pub ranges: Box<[Vector<u32, Const<2>>]>,
    pub stride: u32,
    pub minimum_age: u32,
    pub retirement_age: u32,
    #[cube(comptime)]
    pub remapping: WorkerRemapping,
}

pub(super) struct WorkerRemappingSystem;
pub(super) struct RemapWorkers;

impl System for WorkerRemappingSystem {
    type Input = WorkerRemappingInput;
    type Passes = gpu_passes![RemapWorkers];
}

#[cube]
impl GpuPass<WorkerRemappingInput> for RemapWorkers {
    type Output = ();

    fn execute(mut input: WorkerRemappingInput) {
        let household = ABSOLUTE_POS;
        if household < input.ranges.len() {
            let current = input.ranges[household];
            let start = current.extract(0usize);
            let count = current.extract(1usize);
            let mut old_start = 0u32;
            let mut old_count = 0u32;
            if household < input.previous_ranges.len() {
                let old = input.previous_ranges[household];
                old_start = old.extract(0usize);
                old_count = old.extract(1usize);
            }
            if count > 0 || old_count > 0 {
                let mut balances = ActorBalances::new(&mut input.state, input.stride);
                let selected = balances.number(Field::Worker, household as u32);
                let mut worker = selected;
                if old_count > 0 && selected >= old_start && selected - old_start < old_count {
                    match comptime!(input.remapping) {
                        WorkerRemapping::Births => worker = start + selected - old_start,
                        WorkerRemapping::Formations => {
                            worker = u32::new(comptime!(i64::from(u32::MAX)));
                            let identity =
                                input.previous_people[selected as usize * (P::WORDS / 4)];
                            for offset in 0..count {
                                let person =
                                    input.people[(start + offset) as usize * (P::WORDS / 4)];
                                if person.extract(P::IDENTITY) == identity.extract(P::IDENTITY)
                                    && person.extract(P::IDENTITY + 1)
                                        == identity.extract(P::IDENTITY + 1)
                                {
                                    worker = start + offset;
                                }
                            }
                        }
                    }
                }
                if old_count == 0 {
                    worker = u32::new(comptime!(i64::from(u32::MAX)));
                }
                if worker == u32::MAX && selected != u32::MAX {
                    for offset in 0..count {
                        let header = input.people[(start + offset) as usize * (P::WORDS / 4)];
                        let status = input.people[(start + offset) as usize * (P::WORDS / 4) + 1];
                        let age = header.extract(P::AGE);
                        let eligible = age >= input.minimum_age
                            && age < input.retirement_age
                            && status.extract(P::CAUSE - 4) == 0;
                        if eligible && worker == u32::MAX {
                            worker = start + offset;
                        }
                    }
                }
                balances.set_number(Field::Worker, household as u32, worker);
            }
        }
    }
}

pub(super) struct PopulationTotals;
pub(super) struct WritePopulationTotals;

impl System for PopulationTotals {
    type Input = PopulationTotalsInput;
    type Passes = gpu_passes![WritePopulationTotals];
}

#[cube]
impl GpuPass<PopulationTotalsInput> for WritePopulationTotals {
    type Output = ();

    fn execute(mut input: PopulationTotalsInput) {
        let metric = ABSOLUTE_POS;
        if metric < OBSERVATION_METRICS && metric != 7 && metric != 15 {
            input.output[metric] = input.sums[metric] as u32;
            if metric == 6 {
                input.output[7usize] = ((input.sums[6usize] >> 32) + input.sums[7usize]) as u32;
            }
            if metric == 14 {
                input.output[15usize] = ((input.sums[14usize] >> 32) + input.sums[15usize]) as u32;
            }
        }
    }
}

#[derive(CubeType, CubeLaunch)]
pub(super) struct BirthCountInput {
    pub births: Box<[u32]>,
    pub scratch: Box<[u32]>,
}

pub(super) struct BirthCounting;
pub(super) struct CountBirths;

impl System for BirthCounting {
    type Input = BirthCountInput;
    type Passes = gpu_passes![CountBirths];
}

#[cube]
impl GpuPass<BirthCountInput> for CountBirths {
    type Output = ();

    fn execute(mut input: BirthCountInput) {
        let actor = ABSOLUTE_POS;
        let mut present = 0u32;
        if actor < input.births.len() {
            present = u32::cast_from(input.births[actor] > 0);
        }
        let prefix = crate::simulation::parallel::prefix_sum::<u32>(present);
        if actor < input.births.len() {
            input.scratch[actor] = prefix.inclusive;
        }
        if UNIT_POS == 0 {
            input.scratch[input.births.len() + CUBE_POS] = prefix.total;
        }
    }
}

#[derive(CubeType, CubeLaunch)]
pub(super) struct BirthRangeInput {
    pub old: Box<[u32]>,
    pub births: Box<[u32]>,
    pub scratch: Box<[u32]>,
    pub ranges: Box<[u32]>,
}

pub(super) struct BirthRanges;
pub(super) struct AssignBirthRanges;

impl System for BirthRanges {
    type Input = BirthRangeInput;
    type Passes = gpu_passes![AssignBirthRanges];
}

#[cube]
impl GpuPass<BirthRangeInput> for AssignBirthRanges {
    type Output = ();

    fn execute(mut input: BirthRangeInput) {
        let actor = ABSOLUTE_POS;
        if actor < input.births.len() {
            let birth = u32::cast_from(input.births[actor] > 0);
            let mut offset = input.scratch[actor] - birth;
            if CUBE_POS > 0 {
                offset += input.scratch[input.births.len() + CUBE_POS - 1];
            }
            input.ranges[actor * 2] = input.old[actor * 2] + offset;
            input.ranges[actor * 2 + 1] = input.old[actor * 2 + 1] + birth;
        }
    }
}

#[derive(CubeType, CubeLaunch)]
pub(super) struct FirmExtensionInput {
    pub previous: Box<[Vector<u32, Const<2>>]>,
    pub ranges: Box<[Vector<u32, Const<2>>]>,
    pub births: Box<[u32]>,
    pub people: u32,
}

pub(super) struct FirmRegistryExtension;
pub(super) struct ExtendFirmRanges;

impl System for FirmRegistryExtension {
    type Input = FirmExtensionInput;
    type Passes = gpu_passes![ExtendFirmRanges];
}

#[cube]
impl GpuPass<FirmExtensionInput> for ExtendFirmRanges {
    type Output = ();

    fn execute(mut input: FirmExtensionInput) {
        let actor = ABSOLUTE_POS;
        if actor < input.ranges.len() {
            let mut range = Vector::<u32, Const<2>>::new(0);
            range.insert(0usize, input.people);
            if actor < input.previous.len() {
                range = input.previous[actor];
            }
            input.ranges[actor] = range;
            input.births[actor] = 0;
        }
    }
}

#[derive(CubeType)]
struct PopulationCopies<'a> {
    active: &'a mut [Vector<u32, Const<4>>],
    audited: &'a mut [Vector<u32, Const<4>>],
}

#[cube]
impl<'a> PopulationCopies<'a> {
    fn new(
        active: &'a mut [Vector<u32, Const<4>>],
        audited: &'a mut [Vector<u32, Const<4>>],
    ) -> Self {
        PopulationCopies::<'a> { active, audited }
    }

    fn write(&mut self, offset: usize, words: Vector<u32, Const<4>>) {
        self.active[offset] = words;
        self.audited[offset] = words;
    }
}

#[derive(CubeType, CubeLaunch)]
pub(super) struct PopulationGrowthInput {
    pub source: Box<[Vector<u32, Const<4>>]>,
    pub target: Box<[Vector<u32, Const<4>>]>,
    pub audited: Box<[Vector<u32, Const<4>>]>,
    pub old_ranges: Box<[u32]>,
    pub new_ranges: Box<[u32]>,
    pub people: u32,
}

pub(super) struct PopulationGrowth;
pub(super) struct CopyPeople;

impl System for PopulationGrowth {
    type Input = PopulationGrowthInput;
    type Passes = gpu_passes![CopyPeople];
}

#[cube]
impl GpuPass<PopulationGrowthInput> for CopyPeople {
    type Output = ();

    fn execute(mut input: PopulationGrowthInput) {
        let index = ABSOLUTE_POS;
        if index < input.people as usize * (P::WORDS / 4) {
            let person = index / (P::WORDS / 4);
            let household = input.source[person * (P::WORDS / 4)].extract(P::HOUSEHOLD) as usize;
            let offset = input.new_ranges[household * 2] - input.old_ranges[household * 2];
            let destination = index + offset as usize * (P::WORDS / 4);
            let value = input.source[index];
            let mut copies = PopulationCopies::new(&mut input.target, &mut input.audited);
            copies.write(destination, value);
        }
    }
}

#[derive(CubeType, CubeLaunch)]
pub(super) struct NewbornInput {
    pub target: Box<[Vector<u32, Const<4>>]>,
    pub audited: Box<[Vector<u32, Const<4>>]>,
    pub old_ranges: Box<[u32]>,
    pub new_ranges: Box<[u32]>,
    pub births: Box<[u32]>,
    pub next_low: u32,
    pub next_high: u32,
}

pub(super) struct NewbornInitialization;
pub(super) struct WriteNewborns;

impl System for NewbornInitialization {
    type Input = NewbornInput;
    type Passes = gpu_passes![WriteNewborns];
}

#[cube]
impl GpuPass<NewbornInput> for WriteNewborns {
    type Output = ();

    fn execute(mut input: NewbornInput) {
        let household = ABSOLUTE_POS;
        if household < input.births.len() && input.births[household] > 0 {
            let start = input.old_ranges[household * 2];
            let count = input.old_ranges[household * 2 + 1];
            let destination = input.new_ranges[household * 2];
            let base = (destination + count) as usize * (P::WORDS / 4);
            let identity = plus(
                ticks(input.next_low, input.next_high),
                ticks(destination - start, 0),
            )
            .value;
            let mut header = Vector::<u32, Const<4>>::new(0);
            header.insert(P::IDENTITY, identity.extract(0usize));
            header.insert(P::IDENTITY + 1, identity.extract(1usize));
            header.insert(P::HOUSEHOLD, household as u32);
            let mut status = Vector::<u32, Const<4>>::new(0);
            status.insert(P::SEX_AND_COLD_EXPOSURE - 4, input.births[household] - 1);
            let mut copies = PopulationCopies::new(&mut input.target, &mut input.audited);
            copies.write(base, header);
            copies.write(base + 1, status);
            copies.write(base + 2, Vector::new(0));
        }
    }
}

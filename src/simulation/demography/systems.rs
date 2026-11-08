use super::{DemographyPolicy, Stage, kernels};
use crate::simulation::{
    configuration::EconomyConfig,
    pass::GpuPass,
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
        kernels::step(
            &mut input.state,
            &input.actors,
            &input.config,
            &mut input.people,
            &input.ranges,
            &input.policy,
            &mut input.births,
            &input.samples,
            input.day,
            comptime!(input.stage),
        );
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
        kernels::observe(
            &input.people,
            &mut input.state,
            &input.config,
            &input.policy,
            &input.output,
            input.count,
        );
    }
}

#[derive(CubeType, CubeLaunch)]
pub(super) struct PopulationTotalsInput {
    pub sums: Box<[u64]>,
    pub output: Box<[u32]>,
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
        kernels::observation_totals(&input.sums, &mut input.output);
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
        kernels::birth_prefix(&input.births, &mut input.scratch);
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
        kernels::birth_ranges(&input.old, &input.births, &input.scratch, &mut input.ranges);
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

#[derive(CubeType, CubeLaunch)]
pub(super) struct PopulationGrowthInput {
    pub source: Box<[Vector<u32, Const<4>>]>,
    pub target: Box<[Vector<u32, Const<4>>]>,
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
        kernels::grow(
            &input.source,
            &mut input.target,
            &input.old_ranges,
            &input.new_ranges,
            input.people,
        );
    }
}

#[derive(CubeType, CubeLaunch)]
pub(super) struct NewbornInput {
    pub target: Box<[Vector<u32, Const<4>>]>,
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
        kernels::newborns(
            &mut input.target,
            &input.old_ranges,
            &input.new_ranges,
            &input.births,
            input.next_low,
            input.next_high,
        );
    }
}

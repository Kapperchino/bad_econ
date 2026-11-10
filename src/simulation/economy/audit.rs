use super::*;
use crate::domain::ActorRows;
use crate::simulation::{Device, configuration::Configuration};
use anyhow::Result;
use burn_wgpu::CubeTensor;
use bytemuck::{Pod, Zeroable};

pub(in crate::simulation) const TILE_ACTORS: usize = 4096;

fn dispatch(device: &Device, groups: usize, tasks: usize) -> CubeCount {
    let width = groups
        .max(1)
        .min(device.client.properties().hardware.max_cube_count.0 as usize);
    CubeCount::Static(
        width as u32,
        tasks as u32,
        groups.max(1).div_ceil(width) as u32,
    )
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Metric {
    field: u32,
    kind: u32,
    operation: u32,
    source: u32,
}

impl Metric {
    fn population(&self) -> MetricPopulation {
        match self.kind {
            0 | NO_ACTOR => MetricPopulation::Actors,
            _ => MetricPopulation::Firms,
        }
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum MetricPopulation {
    Actors,
    Firms,
}

struct MetricSet {
    ids: CubeTensor,
    groups: usize,
    population: MetricPopulation,
}

impl MetricSet {
    fn new(
        device: &Device,
        metrics: &[Metric],
        actors: &ActorRows,
        population: MetricPopulation,
    ) -> Self {
        let ids = metrics
            .iter()
            .enumerate()
            .filter_map(|(index, metric)| {
                (metric.population() == population).then_some(index as u32)
            })
            .collect::<Vec<_>>();
        let count = match population {
            MetricPopulation::Actors => actors.len(),
            MetricPopulation::Firms => actors.firms(),
        };
        Self {
            ids: device.tensor(&ids),
            groups: count.div_ceil(TILE_ACTORS).max(1),
            population,
        }
    }
}

pub(in crate::simulation) struct AuditInput<'a> {
    pub state: &'a CubeTensor,
    pub actors: &'a CubeTensor,
    pub config: &'a Configuration,
    pub spatial: &'a CubeTensor,
    pub partials: &'a CubeTensor,
    pub summary: &'a CubeTensor,
    pub control: &'a CubeTensor,
    pub rings: RingValidation<'a>,
}

pub(in crate::simulation) enum RingValidation<'a> {
    All,
    Unchanged,
    Selected(&'a CubeTensor),
}

#[derive(Clone, Copy)]
#[repr(u32)]
enum Source {
    Field,
    Cash,
    Inventory,
    Produced,
    Firm,
    Distributor,
    Closure,
    Square,
    Homeless,
    Labor,
    ActiveFirm,
    BusinessType,
    BusinessState,
}

pub(in crate::simulation) struct AuditBuffers {
    metrics: CubeTensor,
    sets: [MetricSet; 2],
    groups: CubeTensor,
}

impl AuditBuffers {
    pub fn new(device: &Device, actors: &ActorRows) -> Self {
        use SummaryLayout as S;
        let metrics = (0..S::METRICS as u32)
            .map(|metric| {
                let mut definition = Metric {
                    field: 0,
                    kind: NO_ACTOR,
                    operation: 0,
                    source: Source::Field as u32,
                };
                match metric {
                    S::WORKFORCE | S::SUBSISTENCE | S::INACTIVE => {
                        definition.source = Source::Labor as u32;
                        definition.field = metric - S::WORKFORCE;
                        definition.kind = 0;
                    }
                    S::HOUSEHOLD_CASH => {
                        definition.field = Field::Cash as u32;
                        definition.kind = 0;
                    }
                    metric if (S::BUSINESS_TYPES..S::BUSINESS_STATES).contains(&metric) => {
                        definition.source = Source::BusinessType as u32;
                        definition.field = metric - S::BUSINESS_TYPES;
                        definition.kind = Good::COUNT as u32 + 1;
                    }
                    metric if (S::BUSINESS_STATES..S::FIRM_CASH).contains(&metric) => {
                        definition.source = Source::BusinessState as u32;
                        definition.field = metric - S::BUSINESS_STATES;
                        definition.kind = Good::COUNT as u32 + 1;
                    }
                    metric if metric >= S::FIRM_CASH => {
                        let group = (metric - S::FIRM_CASH) / Good::COUNT as u32;
                        definition.kind = (metric - S::FIRM_CASH) % Good::COUNT as u32 + 1;
                        definition.field = match group {
                            0 => Field::Cash,
                            1 => Field::BusinessOpeningCash,
                            2 => Field::BusinessClosingCash,
                            _ => Field::BusinessRevenue,
                        } as u32;
                    }
                    metric if (S::ACTIVE_BY_GOOD..S::BUSINESS_TYPES).contains(&metric) => {
                        definition.kind = metric - S::ACTIVE_BY_GOOD + 1;
                        definition.source = Source::ActiveFirm as u32;
                    }
                    S::CASH => definition.source = Source::Cash as u32,
                    S::PAYROLL_RESERVES => definition.field = Field::PayrollReserve as u32,
                    S::INCOME_TAX => definition.field = Field::TaxTreasury as u32,
                    S::EMPLOYEE_CHARGES => definition.field = Field::ChargeTreasury as u32,
                    S::RENT_COLLECTED => definition.field = Field::RentTreasury as u32,
                    S::RENT_ARREARS => definition.field = Field::RentArrears as u32,
                    S::HOMELESS_HOUSEHOLDS => {
                        definition.source = Source::Homeless as u32;
                        definition.kind = 0;
                    }
                    S::EMPLOYED => {
                        definition.field = Field::Employed as u32;
                        definition.kind = 0;
                    }
                    S::SHORTAGES => {
                        definition.field = Field::Shortage as u32;
                        definition.kind = 0;
                    }
                    S::CONSUMED => {
                        definition.field = Field::Consumed as u32;
                        definition.kind = 0;
                    }
                    S::DISTRIBUTORS => {
                        definition.source = Source::Distributor as u32;
                        definition.kind = Good::COUNT as u32 + 1;
                    }
                    S::CLOSURES => {
                        definition.source = Source::Closure as u32;
                        definition.kind = Good::COUNT as u32 + 1;
                    }
                    S::MINIMUM_WAGE | S::MAXIMUM_WAGE | S::MINIMUM_PRICE | S::MAXIMUM_PRICE => {
                        definition.kind = Good::COUNT as u32 + 1;
                        definition.field = match metric {
                            S::MINIMUM_WAGE | S::MAXIMUM_WAGE => Field::Wage as u32,
                            _ => Field::Price as u32,
                        };
                        definition.operation = match metric {
                            S::MINIMUM_WAGE | S::MINIMUM_PRICE => 1,
                            _ => 2,
                        };
                    }
                    _ => {
                        let groups = [
                            (S::INVENTORIES, 1, Source::Inventory),
                            (S::TRANSIT, TRANSIT_FIELDS as u32, Source::Field),
                            (S::PRODUCED, PRODUCTION_FIELDS as u32, Source::Produced),
                            (S::USED, USED_FIELDS as u32, Source::Field),
                            (
                                S::HOUSEHOLD_PRODUCTION,
                                PRODUCTION_FIELDS as u32,
                                Source::Field,
                            ),
                            (S::CONSUMPTION, CONSUMPTION_FIELDS as u32, Source::Field),
                        ];
                        let group = groups.into_iter().find(|(start, _, _)| {
                            (*start..*start + Good::COUNT as u32).contains(&metric)
                        });
                        match group {
                            Some((start, field, source)) => {
                                definition.field = field + metric - start;
                                definition.source = source as u32;
                            }
                            None => {
                                let groups = [
                                    S::FIRMS_BY_GOOD,
                                    S::WAGE_MINIMA,
                                    S::WAGE_MAXIMA,
                                    S::WAGE_TOTALS,
                                    S::WAGE_SQUARES,
                                    S::PRICE_MINIMA,
                                    S::PRICE_MAXIMA,
                                    S::PRICE_TOTALS,
                                    S::PRICE_SQUARES,
                                ];
                                let start = groups
                                    .into_iter()
                                    .find(|start| {
                                        (*start..*start + Good::COUNT as u32).contains(&metric)
                                    })
                                    .expect("summary metric");
                                definition.kind = metric - start + 1;
                                definition.field = match start {
                                    S::WAGE_MINIMA
                                    | S::WAGE_MAXIMA
                                    | S::WAGE_TOTALS
                                    | S::WAGE_SQUARES => Field::Wage as u32,
                                    _ => Field::Price as u32,
                                };
                                definition.source = match start {
                                    S::FIRMS_BY_GOOD => Source::Firm,
                                    S::WAGE_SQUARES | S::PRICE_SQUARES => Source::Square,
                                    _ => Source::Field,
                                } as u32;
                                definition.operation = match start {
                                    S::WAGE_MINIMA | S::PRICE_MINIMA => 1,
                                    S::WAGE_MAXIMA | S::PRICE_MAXIMA => 2,
                                    _ => 0,
                                };
                            }
                        }
                    }
                }
                definition
            })
            .collect::<Vec<_>>();
        let sets = [MetricPopulation::Actors, MetricPopulation::Firms]
            .map(|population| MetricSet::new(device, &metrics, actors, population));
        let groups = metrics
            .iter()
            .map(|metric| {
                sets.iter()
                    .find(|set| set.population == metric.population())
                    .expect("metric population")
                    .groups as u32
            })
            .collect::<Vec<_>>();
        Self {
            metrics: device.tensor(bytemuck::cast_slice(&metrics)),
            sets,
            groups: device.tensor(&groups),
        }
    }

    pub fn summarize(&self, device: &Device, input: AuditInput<'_>) -> Result<()> {
        let count = input.actors.meta.num_elements() / ActorLayout::WORDS;
        let argument = |tensor: &CubeTensor| tensor.clone().into_buffer_arg();
        ActorValidation::launch(
            device,
            dispatch(
                device,
                count.div_ceil(device.workgroup as usize),
                match input.rings {
                    RingValidation::All => Good::COUNT + 1,
                    RingValidation::Unchanged | RingValidation::Selected(_) => 1,
                },
            ),
            CubeDim::new_1d(device.workgroup),
            ValidationInputLaunch {
                state: argument(input.state),
                actors: argument(input.actors),
                config: input.config.launch(),
                control: argument(input.control),
            },
        );
        if let RingValidation::Selected(indices) = input.rings {
            SelectedValidation::launch(
                device,
                dispatch(
                    device,
                    indices
                        .meta
                        .num_elements()
                        .div_ceil(device.workgroup as usize),
                    Good::COUNT,
                ),
                CubeDim::new_1d(device.workgroup),
                SelectedValidationInputLaunch {
                    state: argument(input.state),
                    indices: argument(indices),
                    config: input.config.launch(),
                    control: argument(input.control),
                },
            );
        }
        for set in &self.sets {
            MetricSummary::launch(
                device,
                dispatch(device, set.groups, set.ids.meta.num_elements()),
                CubeDim::new_1d(device.workgroup),
                SummaryInputLaunch {
                    state: argument(input.state),
                    actors: argument(input.actors),
                    config: input.config.launch(),
                    spatial: argument(input.spatial),
                    metrics: argument(&self.metrics),
                    ids: argument(&set.ids),
                    output: argument(input.partials),
                    control: argument(input.control),
                    groups: set.groups as u32,
                    population: set.population,
                },
            );
        }
        parallel::reduction::summary(
            device,
            input.partials,
            input.summary,
            input.control,
            &self.groups,
        )
    }
}

#[cube]
fn value(state: &[Ticks], config: &EconomyConfig, field: u32, actor: u32) -> Ticks {
    state[(field * config.actors + actor) as usize]
}

#[cube]
fn number(state: &[Ticks], config: &EconomyConfig, field: u32, actor: u32) -> u32 {
    value(state, config, field, actor).extract(LOW_WORD)
}

#[cube]
fn checked_sum(a: u64, b: u64, control: &[Atomic<u32>]) -> u64 {
    let total = a + b;
    if total < a {
        KernelFaults::new(comptime!(FaultFlags::ARITHMETIC_OVERFLOW)).record(control);
    }
    total
}

#[cube]
fn native(value: Ticks) -> u64 {
    u64::cast_from(value.extract(LOW_WORD)) | (u64::cast_from(value.extract(HIGH_WORD)) << 32)
}

#[derive(CubeType, CubeLaunch)]
struct SummaryInput {
    state: Box<[Ticks]>,
    actors: Box<[u32]>,
    config: EconomyConfig,
    spatial: Box<[u32]>,
    metrics: Box<[Vector<u32, Const<4>>]>,
    ids: Box<[u32]>,
    output: Box<[Ticks]>,
    control: Box<[Atomic<u32>]>,
    groups: u32,
    #[cube(comptime)]
    population: MetricPopulation,
}

struct MetricSummary;
struct SummarizeMetrics;

impl System for MetricSummary {
    type Input = SummaryInput;
    type Passes = gpu_passes![SummarizeMetrics];
}

#[cube]
impl GpuPass<SummaryInput> for SummarizeMetrics {
    type Output = ();

    fn execute(mut input: SummaryInput) {
        summarize_metrics(&mut input);
    }
}

#[cube]
fn summarize_metrics(input: &mut SummaryInput) {
    let state = &input.state;
    let actors = &input.actors;
    let config = &input.config;
    let control = &input.control;
    let index = CUBE_POS_Y;
    let group = CUBE_POS_X + CUBE_POS_Z * CUBE_COUNT_X;
    if group < input.groups {
        let metric = input.ids[index as usize];
        let definition = input.metrics[metric as usize];
        let field = definition.extract(0usize);
        let selected_kind = definition.extract(1usize);
        let operation = definition.extract(2usize);
        let source = definition.extract(3usize);
        let identity = select(operation == 1, u64::MAX, 0u64);
        let mut total = identity;
        let cells = config.spatial_side * config.spatial_side;
        let firm_segment = 2 * (cells + 1 + config.actors);
        let count = match comptime!(input.population) {
            MetricPopulation::Actors => config.actors,
            MetricPopulation::Firms => input.spatial[(firm_segment + cells) as usize],
        };
        let mut member = group * TILE_ACTORS as u32 + UNIT_POS;
        while member < min((group + 1) * TILE_ACTORS as u32, count) {
            let actor = match comptime!(input.population) {
                MetricPopulation::Actors => member,
                MetricPopulation::Firms => {
                    input.spatial[(firm_segment + cells + 1 + member) as usize]
                }
            };
            let actor_kind = kind(actors, actor).bits();
            let selected = selected_kind == NO_ACTOR
                || selected_kind == actor_kind
                || (selected_kind == Good::COUNT as u32 + 1 && actor_kind > 0);
            if selected {
                let mut amount = native(value(state, config, field, actor));
                match source {
                    1 => {
                        for coordinate in [
                            Field::Escrow as u32,
                            Field::PayrollReserve as u32,
                            Field::TaxTreasury as u32,
                            Field::ChargeTreasury as u32,
                            Field::RentPending as u32,
                        ] {
                            amount = checked_sum(
                                amount,
                                native(value(state, config, coordinate, actor)),
                                control,
                            );
                        }
                    }
                    2 => {
                        if actor_kind == field {
                            amount = checked_sum(
                                amount,
                                native(value(
                                    state,
                                    config,
                                    comptime!(Field::Pipeline as u32),
                                    actor,
                                )),
                                control,
                            );
                        }
                    }
                    3 => {
                        if actor_kind == field - PRODUCTION_FIELDS as u32 + 1 {
                            amount = checked_sum(
                                amount,
                                native(value(
                                    state,
                                    config,
                                    comptime!(Field::Produced as u32),
                                    actor,
                                )),
                                control,
                            );
                        }
                    }
                    4 => amount = 1,
                    5 => {
                        amount = u64::cast_from(
                            number(
                                state,
                                config,
                                comptime!(Field::EnterpriseModel as u32),
                                actor,
                            ) == 1
                                && number(
                                    state,
                                    config,
                                    comptime!(Field::EnterpriseStatus as u32),
                                    actor,
                                ) != comptime!(
                                    crate::domain::enterprise::EnterpriseState::Closed as u32
                                ),
                        )
                    }
                    6 => {
                        amount = u64::cast_from(
                            number(
                                state,
                                config,
                                comptime!(Field::EnterpriseStatus as u32),
                                actor,
                            ) == 4,
                        )
                    }
                    7 => {
                        amount &= 0xffff_ffff;
                        amount *= amount;
                    }
                    9 => {
                        let working = number(state, config, comptime!(Field::Worker as u32), actor)
                            != NO_ACTOR
                            && number(state, config, comptime!(Field::LivingMembers as u32), actor)
                                > 0;
                        let subsistence =
                            comptime!(config.economy == crate::domain::Economy::Industrial)
                                && number(state, config, comptime!(Field::LandPlots as u32), actor)
                                    > 0;
                        let category = select(working, select(subsistence, 1u32, 0u32), 2u32);
                        amount = u64::cast_from(category == field);
                    }
                    10 => {
                        amount = u64::cast_from(
                            number(
                                state,
                                config,
                                comptime!(Field::EnterpriseStatus as u32),
                                actor,
                            ) < 4,
                        );
                    }
                    11 => {
                        let category = select(
                            actor_kind == comptime!(Good::Housing as u32),
                            2u32,
                            number(
                                state,
                                config,
                                comptime!(Field::EnterpriseModel as u32),
                                actor,
                            ),
                        );
                        amount = u64::cast_from(
                            category == field
                                && number(
                                    state,
                                    config,
                                    comptime!(Field::EnterpriseStatus as u32),
                                    actor,
                                ) < 4,
                        );
                    }
                    12 => {
                        amount = u64::cast_from(
                            number(
                                state,
                                config,
                                comptime!(Field::EnterpriseStatus as u32),
                                actor,
                            ) == field,
                        );
                    }
                    8 => {
                        amount = u64::cast_from(
                            number(state, config, comptime!(Field::LivingMembers as u32), actor)
                                > 0
                                && number(
                                    state,
                                    config,
                                    comptime!(Field::HousingStatus as u32),
                                    actor,
                                ) == comptime!(crate::domain::HousingState::Homeless as u32),
                        );
                    }
                    _ => (),
                }
                total = match operation {
                    1 => min(total, amount),
                    2 => max(total, amount),
                    _ => checked_sum(total, amount, control),
                };
            }
            member += CUBE_DIM;
        }
        let mut shared = Shared::<[u64]>::new_slice(256usize);
        shared[UNIT_POS as usize] = total;
        sync_cube();
        let mut stride = CUBE_DIM / 2;
        while stride > 0 {
            if UNIT_POS < stride {
                let a = shared[UNIT_POS as usize];
                let b = shared[(UNIT_POS + stride) as usize];
                shared[UNIT_POS as usize] = match operation {
                    1 => min(a, b),
                    2 => max(a, b),
                    _ => checked_sum(a, b, control),
                };
            }
            sync_cube();
            stride /= 2;
        }
        if UNIT_POS == 0 {
            let base = group * SummaryLayout::METRICS as u32 + metric;
            input.output[base as usize] = ticks(
                u32::cast_from(shared[0usize]),
                u32::cast_from(shared[0usize] >> 32),
            );
        }
    }
}

#[derive(CubeType, CubeLaunch)]
struct ValidationInput {
    state: Box<[Ticks]>,
    actors: Box<[u32]>,
    config: EconomyConfig,
    control: Box<[Atomic<u32>]>,
}

struct ActorValidation;
struct ValidateActors;

#[derive(CubeType, CubeLaunch)]
struct SelectedValidationInput {
    state: Box<[Ticks]>,
    indices: Box<[u32]>,
    config: EconomyConfig,
    control: Box<[Atomic<u32>]>,
}

struct SelectedValidation;
struct ValidateSelected;

impl System for SelectedValidation {
    type Input = SelectedValidationInput;
    type Passes = gpu_passes![ValidateSelected];
}

#[cube]
impl GpuPass<SelectedValidationInput> for ValidateSelected {
    type Output = ();

    fn execute(input: SelectedValidationInput) {
        let index = (CUBE_POS_X + CUBE_POS_Z * CUBE_COUNT_X) * CUBE_DIM + UNIT_POS;
        if index < input.indices.len() as u32 {
            validate_ring(
                &input.state,
                &input.config,
                &input.control,
                input.indices[index as usize],
                CUBE_POS_Y,
            );
        }
    }
}

impl System for ActorValidation {
    type Input = ValidationInput;
    type Passes = gpu_passes![ValidateActors];
}

#[cube]
impl GpuPass<ValidationInput> for ValidateActors {
    type Output = ();

    fn execute(input: ValidationInput) {
        validate_actors(&input.state, &input.actors, &input.config, &input.control);
    }
}

#[cube]
fn validate_actors(
    state: &[Ticks],
    actors: &[u32],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
) {
    let actor = (CUBE_POS_X + CUBE_POS_Z * CUBE_COUNT_X) * CUBE_DIM + UNIT_POS;
    let task = CUBE_POS_Y;
    match task {
        0 => validate_actor(state, actors, config, control, actor),
        _ => {
            if actor < config.actors {
                validate_ring(state, config, control, actor, task - 1);
            }
        }
    }
}

#[cube]
fn validate_ring(
    state: &[Ticks],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
    actor: u32,
    good: u32,
) {
    let held = value(state, config, TRANSIT_FIELDS as u32 + good, actor);
    let mut in_ring = zero();
    for slot in 0u32..config.ring {
        in_ring = add_checked(
            control,
            in_ring,
            value(
                state,
                config,
                BASE_FIELDS as u32 + good * config.ring + slot,
                actor,
            ),
        );
    }
    if !equal(in_ring, held) {
        KernelFaults::new(comptime!(FaultFlags::TRANSIT_MISMATCH)).record(control);
    }
}

#[cube]
fn validate_actor(
    state: &[Ticks],
    actors: &[u32],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
    actor: u32,
) {
    if actor < config.actors {
        let payroll_reserve = value(
            state,
            config,
            comptime!(Field::PayrollReserve as u32),
            actor,
        );
        let tax = value(state, config, comptime!(Field::TaxTreasury as u32), actor);
        let charge = value(
            state,
            config,
            comptime!(Field::ChargeTreasury as u32),
            actor,
        );
        if comptime!(config.payroll == Feature::Enabled) {
            let accrued = PayrollAccrual::new(
                value(state, config, comptime!(Field::GrossIncome as u32), actor),
                config,
                control,
            );
            if !equal(payroll_reserve, accrued.reserve)
                || !equal(tax, accrued.tax)
                || !equal(charge, accrued.charge)
            {
                KernelFaults::new(comptime!(FaultFlags::PAYROLL_MISMATCH)).record(control);
            }
        }
        if !equal(
            value(state, config, comptime!(Field::Escrow as u32), actor),
            zero(),
        ) || !equal(
            value(state, config, comptime!(Field::Pipeline as u32), actor),
            zero(),
        ) || !equal(
            value(state, config, comptime!(Field::Due as u32), actor),
            zero(),
        ) {
            KernelFaults::new(comptime!(FaultFlags::UNSETTLED_COMMITMENTS)).record(control);
        }
        if kind(actors, actor).is(comptime!(ActorKind::Household)) {
            let cash = value(state, config, comptime!(Field::Cash as u32), actor);
            let mut bin = WORD_BITS - cash.extract(LOW_WORD).leading_zeros();
            if cash.extract(HIGH_WORD) > 0u32 {
                bin = TICK_BITS - cash.extract(HIGH_WORD).leading_zeros();
            }
            control[(SummaryLayout::WEALTH_START as u32
                + min(bin, SummaryLayout::WEALTH_BINS as u32 - 1)) as usize]
                .fetch_add(1u32);
            let x = actors[ActorLayout::X * (actors.len() / ActorLayout::WORDS) + actor as usize]
                * SummaryLayout::REGION_SIDE as u32
                / config.map_size;
            let y = actors[ActorLayout::Y * (actors.len() / ActorLayout::WORDS) + actor as usize]
                * SummaryLayout::REGION_SIDE as u32
                / config.map_size;
            control[(SummaryLayout::REGIONS_START as u32
                + y * SummaryLayout::REGION_SIDE as u32
                + x) as usize]
                .fetch_add(number(
                    state,
                    config,
                    comptime!(Field::Shortage as u32),
                    actor,
                ));
        }
        let mut transit = zero();
        for good in 0u32..Good::COUNT as u32 {
            let held = value(state, config, TRANSIT_FIELDS as u32 + good, actor);
            transit = add_checked(control, transit, held);
        }
        if !equal(
            transit,
            value(state, config, comptime!(Field::Transit as u32), actor),
        ) {
            KernelFaults::new(comptime!(FaultFlags::TRANSIT_MISMATCH)).record(control);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        domain::{Experiment, Scenario},
        simulation::{Simulation, encode, registry, tests::device},
    };

    #[derive(Clone, Copy)]
    enum TransitCase {
        Unchanged,
        Mismatch,
        RingOverflow,
        TotalOverflow,
    }

    #[test]
    fn sparse_ring_validation_matches_full_snapshots_and_faults_on_gpu() -> Result<()> {
        let scenario = Scenario::baseline(97, 42, Experiment::Transport)?;
        let workgroup = 64;
        let mut simulation = Simulation::new(scenario.clone(), device(workgroup)?)?;
        let count = scenario.actors();
        let indices = simulation
            .device
            .tensor(&(count as u32 - 65..count as u32).collect::<Vec<_>>());
        for case in [
            TransitCase::Unchanged,
            TransitCase::Mismatch,
            TransitCase::RingOverflow,
            TransitCase::TotalOverflow,
        ] {
            let mut state = scenario.initial()?;
            let ring = BASE_FIELDS + scenario.ring() * (Good::COUNT - 1);
            let expected = match case {
                TransitCase::Unchanged => FaultFlags::empty(),
                TransitCase::Mismatch => {
                    state.values[ring * count + count - 1] = 7;
                    FaultFlags::TRANSIT_MISMATCH
                }
                TransitCase::RingOverflow => {
                    state.values[ring * count + count - 1] = u64::MAX;
                    state.values[(ring + 1) * count + count - 1] = 1;
                    state.values[(TRANSIT_FIELDS + Good::COUNT - 1) * count + count - 1] = 7;
                    state.set(Field::Transit, count - 1, 7);
                    FaultFlags::ARITHMETIC_OVERFLOW | FaultFlags::TRANSIT_MISMATCH
                }
                TransitCase::TotalOverflow => {
                    state.values[BASE_FIELDS * count] = u64::MAX;
                    state.values[(BASE_FIELDS + scenario.ring()) * count] = 1;
                    state.values[TRANSIT_FIELDS * count] = u64::MAX;
                    state.values[(TRANSIT_FIELDS + 1) * count] = 1;
                    state.set(Field::Transit, 0, 0);
                    FaultFlags::ARITHMETIC_OVERFLOW
                }
            };
            simulation.device.write(
                &simulation.state,
                bytemuck::cast_slice(&encode(&state.values)),
            );
            let sparse = simulation.observe_rings(RingValidation::Selected(&indices));
            let summary =
                simulation
                    .device
                    .read_range(&simulation.summary, 0, SummaryLayout::WORDS)?;
            let control = simulation.device.read(&simulation.control)?;
            let full = simulation.observe();
            assert_eq!(sparse.is_ok(), full.is_ok());
            assert_eq!(
                summary,
                simulation
                    .device
                    .read_range(&simulation.summary, 0, SummaryLayout::WORDS,)?
            );
            assert_eq!(control, simulation.device.read(&simulation.control)?);
            assert_eq!(FaultFlags::read(&control)?, expected);
            if let (Ok(sparse), Ok(full)) = (sparse, full) {
                assert_eq!(sparse, full);
                assert_eq!(simulation.observe_rings(RingValidation::Unchanged)?, full);
            }
        }
        Ok(())
    }

    #[test]
    fn direct_registry_commit_revalidates_rings_after_a_previous_audit_on_gpu() -> Result<()> {
        let scenario = Scenario::baseline(97, 42, Experiment::Transport)?;
        let mut simulation = Simulation::new(scenario.clone(), device(64)?)?;
        let _ = simulation.observe()?;
        let mut state = scenario.initial()?;
        let firm = scenario.config().households as usize;
        state.set(Field::PendingCapacity, firm, 1);
        state.values[BASE_FIELDS * state.actors.len()] = 7;
        simulation.device.write(
            &simulation.state,
            bytemuck::cast_slice(&encode(&state.values)),
        );
        let mut requests = vec![0; state.actors.len()];
        requests[firm] = (6 << 24) | 1;
        simulation
            .device
            .write(&simulation.enterprise_requests, &requests);
        let pending =
            registry::records::PendingRecords::from_dense(&requests, &vec![0; requests.len()])?;
        let plan = registry::RegistryPlan::new(
            &scenario,
            &simulation.metadata,
            &simulation.migrations,
            &pending.records,
        )?
        .expect("capacity registry plan");
        assert!(simulation.commit_registry(plan, &pending.records).is_err());
        assert_eq!(
            FaultFlags::read(&simulation.device.read(&simulation.control)?)?,
            FaultFlags::TRANSIT_MISMATCH,
        );
        Ok(())
    }
}

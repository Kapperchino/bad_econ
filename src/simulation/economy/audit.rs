use super::*;
use crate::simulation::{Device, configuration::Configuration};
use burn_wgpu::CubeTensor;
use bytemuck::{Pod, Zeroable};

pub(in crate::simulation) const TILE_ACTORS: usize = 4096;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Metric {
    field: u32,
    kind: u32,
    operation: u32,
    source: u32,
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
}

pub(in crate::simulation) struct AuditBuffers {
    metrics: CubeTensor,
}

impl AuditBuffers {
    pub fn new(device: &Device) -> Self {
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
                    S::CASH => definition.source = Source::Cash as u32,
                    S::PAYROLL_RESERVES => definition.field = Field::PayrollReserve as u32,
                    S::INCOME_TAX => definition.field = Field::TaxTreasury as u32,
                    S::EMPLOYEE_CHARGES => definition.field = Field::ChargeTreasury as u32,
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
        Self {
            metrics: device.tensor(bytemuck::cast_slice(&metrics)),
        }
    }

    pub fn summarize(
        &self,
        device: &Device,
        state: &CubeTensor,
        actors: &CubeTensor,
        config: &Configuration,
        output: &CubeTensor,
        control: &CubeTensor,
    ) {
        let count = actors.meta.num_elements() / ActorLayout::WORDS;
        let argument = |tensor: &CubeTensor| tensor.clone().into_buffer_arg();
        ActorValidation::launch(
            device,
            device.count(count * (Good::COUNT + 1)),
            CubeDim::new_1d(device.workgroup),
            ValidationInputLaunch {
                state: argument(state),
                actors: argument(actors),
                config: config.launch(),
                control: argument(control),
            },
        );
        let groups = count.div_ceil(TILE_ACTORS);
        MetricSummary::launch(
            device,
            device.count(groups * SummaryLayout::METRICS * device.workgroup as usize),
            CubeDim::new_1d(device.workgroup),
            SummaryInputLaunch {
                state: argument(state),
                actors: argument(actors),
                config: config.launch(),
                metrics: argument(&self.metrics),
                output: argument(output),
                control: argument(control),
                groups: groups as u32,
            },
        );
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
    metrics: Box<[Vector<u32, Const<4>>]>,
    output: Box<[Ticks]>,
    control: Box<[Atomic<u32>]>,
    groups: u32,
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
        summarize_metrics(
            &input.state,
            &input.actors,
            &input.config,
            &input.metrics,
            &mut input.output,
            &input.control,
            input.groups,
        );
    }
}

#[cube]
fn summarize_metrics(
    state: &[Ticks],
    actors: &[u32],
    config: &EconomyConfig,
    metrics: &[Vector<u32, Const<4>>],
    output: &mut [Ticks],
    control: &[Atomic<u32>],
    groups: u32,
) {
    let block = CUBE_POS as u32;
    let metric = block / groups;
    let group = block % groups;
    if metric < SummaryLayout::METRICS as u32 {
        let definition = metrics[metric as usize];
        let field = definition.extract(0usize);
        let selected_kind = definition.extract(1usize);
        let operation = definition.extract(2usize);
        let source = definition.extract(3usize);
        let identity = select(operation == 1, u64::MAX, 0u64);
        let mut total = identity;
        let mut actor = group * TILE_ACTORS as u32 + UNIT_POS;
        while actor < min((group + 1) * TILE_ACTORS as u32, config.actors) {
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
                            ) == 1,
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
                    _ => (),
                }
                total = match operation {
                    1 => min(total, amount),
                    2 => max(total, amount),
                    _ => checked_sum(total, amount, control),
                };
            }
            actor += CUBE_DIM;
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
            output[base as usize] = ticks(
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
    let index = ABSOLUTE_POS as u32;
    let actor = index % config.actors;
    let task = index / config.actors;
    match task {
        0 => validate_actor(state, actors, config, control, actor),
        _ => {
            if task <= Good::COUNT as u32 {
                let good = task - 1;
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
        }
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

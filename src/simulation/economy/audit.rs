use super::*;
use crate::domain::ActorRows;
use crate::simulation::parallel::reduction::{checked_sum, combine_metrics, reduce_metric};
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
    fn definition(metric: u32) -> Self {
        use SummaryLayout as S;
        let goods = Good::COUNT as u32;
        let firms = goods + 1;
        let sum = |source: Source, field, kind| Self {
            source: source as u32,
            field,
            kind,
            operation: 0,
        };
        let field = |coordinate: Field, kind| sum(Source::Field, coordinate as u32, kind);
        match metric {
            S::CASH => sum(Source::Cash, 0, NO_ACTOR),
            S::PAYROLL_RESERVES => field(Field::PayrollReserve, NO_ACTOR),
            S::INCOME_TAX => field(Field::TaxTreasury, NO_ACTOR),
            S::EMPLOYEE_CHARGES => field(Field::ChargeTreasury, NO_ACTOR),
            S::RENT_COLLECTED => field(Field::RentTreasury, NO_ACTOR),
            S::RENT_ARREARS => field(Field::RentArrears, NO_ACTOR),
            S::HOMELESS_HOUSEHOLDS => sum(Source::Homeless, 0, 0),
            S::EMPLOYED => field(Field::Employed, 0),
            S::SHORTAGES => field(Field::Shortage, 0),
            S::INVENTORIES..S::TRANSIT => {
                sum(Source::Inventory, 1 + metric - S::INVENTORIES, NO_ACTOR)
            }
            S::TRANSIT..S::PRODUCED => sum(
                Source::Field,
                TRANSIT_FIELDS as u32 + metric - S::TRANSIT,
                NO_ACTOR,
            ),
            S::PRODUCED..S::USED => sum(
                Source::Produced,
                PRODUCTION_FIELDS as u32 + metric - S::PRODUCED,
                NO_ACTOR,
            ),
            S::USED..S::CONSUMED => sum(
                Source::Field,
                USED_FIELDS as u32 + metric - S::USED,
                NO_ACTOR,
            ),
            S::CONSUMED => field(Field::Consumed, 0),
            S::HOUSEHOLD_PRODUCTION..S::DISTRIBUTORS => sum(
                Source::Field,
                PRODUCTION_FIELDS as u32 + metric - S::HOUSEHOLD_PRODUCTION,
                NO_ACTOR,
            ),
            S::DISTRIBUTORS => sum(Source::Distributor, 0, firms),
            S::CLOSURES => sum(Source::Closure, 0, firms),
            S::CONSUMPTION..S::MINIMUM_WAGE => sum(
                Source::Field,
                CONSUMPTION_FIELDS as u32 + metric - S::CONSUMPTION,
                NO_ACTOR,
            ),
            S::MINIMUM_WAGE | S::MAXIMUM_WAGE | S::MINIMUM_PRICE | S::MAXIMUM_PRICE => {
                let quote = match metric {
                    S::MINIMUM_WAGE | S::MAXIMUM_WAGE => {
                        sum(Source::QuoteField, Field::Wage as u32, firms)
                    }
                    _ => sum(Source::PriceField, Field::Price as u32, firms),
                };
                Self {
                    operation: match metric {
                        S::MINIMUM_WAGE | S::MINIMUM_PRICE => 1,
                        _ => 2,
                    },
                    ..quote
                }
            }
            S::FIRMS_BY_GOOD..S::WORKFORCE => {
                let index = (metric - S::FIRMS_BY_GOOD) % goods;
                let (coordinate, source, operation) = match metric - index {
                    S::FIRMS_BY_GOOD => (Field::Price, Source::QuotedFirm, 0),
                    S::WAGE_MINIMA => (Field::Wage, Source::QuoteField, 1),
                    S::WAGE_MAXIMA => (Field::Wage, Source::QuoteField, 2),
                    S::WAGE_TOTALS => (Field::Wage, Source::QuoteField, 0),
                    S::WAGE_SQUARES => (Field::Wage, Source::QuoteSquare, 0),
                    S::PRICE_MINIMA => (Field::Price, Source::PriceField, 1),
                    S::PRICE_MAXIMA => (Field::Price, Source::PriceField, 2),
                    S::PRICE_TOTALS => (Field::Price, Source::PriceField, 0),
                    S::PRICE_SQUARES => (Field::Price, Source::PriceSquare, 0),
                    _ => unreachable!("summary quote metric"),
                };
                Self {
                    operation,
                    ..sum(source, coordinate as u32, index + 1)
                }
            }
            S::WORKFORCE | S::SUBSISTENCE | S::INACTIVE => {
                sum(Source::Labor, metric - S::WORKFORCE, 0)
            }
            S::RURAL_EMPLOYED => sum(Source::RuralEmployed, Field::Employed as u32, 0),
            S::TRANSPORT_WORKERS => sum(Source::TransportEmployment, Field::Employed as u32, 0),
            S::CARTAGE_MINUTES => field(Field::CartageMinutes, NO_ACTOR),
            S::CARTAGE_RECEIPTS => field(Field::CartageIncome, NO_ACTOR),
            S::AWAITING_PICKUP => field(Field::RoadFreight, NO_ACTOR),
            S::HOUSEHOLD_CASH => field(Field::Cash, 0),
            S::ACTIVE_BY_GOOD..S::BUSINESS_TYPES => {
                sum(Source::ActiveFirm, 0, metric - S::ACTIVE_BY_GOOD + 1)
            }
            S::BUSINESS_TYPES..S::BUSINESS_STATES => {
                sum(Source::BusinessType, metric - S::BUSINESS_TYPES, firms)
            }
            S::BUSINESS_STATES..S::FIRM_CASH => {
                sum(Source::BusinessState, metric - S::BUSINESS_STATES, firms)
            }
            S::FIRM_CASH..S::PRICE_COUNTS => {
                let index = (metric - S::FIRM_CASH) % goods;
                let coordinate = match metric - index {
                    S::FIRM_CASH => Field::Cash,
                    S::BUSINESS_OPENING => Field::BusinessOpeningCash,
                    S::BUSINESS_CLOSING => Field::BusinessClosingCash,
                    S::BUSINESS_REVENUE => Field::BusinessRevenue,
                    _ => unreachable!("summary firm metric"),
                };
                field(coordinate, index + 1)
            }
            S::PRICE_COUNTS.. => sum(Source::PricedFirm, 0, metric - S::PRICE_COUNTS + 1),
        }
    }

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
    pub ring_mask: Option<&'a CubeTensor>,
    pub rings: RingValidation<'a>,
}

pub(in crate::simulation) enum RingValidation<'a> {
    All,
    Unchanged,
    Selected(&'a CubeTensor),
    Changed {
        entries: &'a CubeTensor,
        settlements: &'a CubeTensor,
    },
}

#[derive(Debug)]
pub(in crate::simulation) enum RegistryObservation {
    Existing,
    Updated,
}

impl RegistryObservation {
    pub(in crate::simulation) fn new(
        audit: crate::simulation::RegistryAudit,
        records: &[crate::simulation::registry::records::PendingRecord],
        settlement: Option<&crate::simulation::registry::settlement::SettlementBatch>,
    ) -> Self {
        match (
            audit,
            settlement,
            records
                .iter()
                .all(|record| matches!(record.request >> 24, 0 | 5 | 6)),
        ) {
            #[cfg(test)]
            (crate::simulation::RegistryAudit::Full, _, _) => Self::Updated,
            (_, None, true) => Self::Existing,
            _ => Self::Updated,
        }
    }
}

#[derive(Clone, Copy)]
#[repr(u32)]
enum Source {
    Field,
    Cash,
    Inventory,
    Produced,
    QuotedFirm,
    Distributor,
    Closure,
    QuoteSquare,
    Homeless,
    Labor,
    ActiveFirm,
    BusinessType,
    BusinessState,
    QuoteField,
    PriceField,
    PriceSquare,
    PricedFirm,
    RuralEmployed,
    TransportEmployment,
}

pub(in crate::simulation) struct AuditBuffers {
    metrics: CubeTensor,
    sets: [MetricSet; 2],
    groups: CubeTensor,
}

impl AuditBuffers {
    pub fn new(device: &Device, actors: &ActorRows) -> Self {
        let metrics = (0..SummaryLayout::METRICS as u32)
            .map(Metric::definition)
            .collect::<Vec<_>>();
        let sets = [MetricPopulation::Actors, MetricPopulation::Firms]
            .map(|population| MetricSet::new(device, &metrics, actors, population));
        let groups = metrics
            .iter()
            .map(|metric| sets[metric.population() as usize].groups as u32)
            .collect::<Vec<_>>();
        Self {
            metrics: device.tensor(bytemuck::cast_slice(&metrics)),
            sets,
            groups: device.tensor(&groups),
        }
    }

    pub fn summarize(&self, device: &Device, input: AuditInput<'_>) -> Result<()> {
        let count = input.actors.meta.num_elements() / ActorLayout::WORDS;
        let ring_tracking = match input.ring_mask {
            Some(_) => RingTracking::Observed,
            None => RingTracking::Untracked,
        };
        let hardware = &device.client.properties().hardware;
        let actor_ring_tracking = match (
            &input.rings,
            ring_tracking,
            parallel::ScanMode::new(device),
            input.config.state_stride % 32,
            hardware.plane_size_min,
            hardware.plane_size_max,
        ) {
            (RingValidation::All, RingTracking::Observed, parallel::ScanMode::Plane, 0, 32, 32) => {
                RingTracking::ObservedPlane
            }
            _ => ring_tracking,
        };
        if let (RingValidation::All, Some(mask)) = (&input.rings, input.ring_mask) {
            crate::simulation::tensor::reset_ring_mask(
                device,
                mask,
                crate::simulation::tensor::RingMaskState::Empty,
            );
        }
        let ring_mask = input.ring_mask.unwrap_or(input.control);
        let validation_workgroup = 128;
        ActorValidation::launch(
            device,
            dispatch(
                device,
                count.div_ceil(validation_workgroup as usize),
                match input.rings {
                    RingValidation::All => Good::COUNT + 1,
                    RingValidation::Unchanged
                    | RingValidation::Selected(_)
                    | RingValidation::Changed { .. } => 1,
                },
            ),
            CubeDim::new_1d(validation_workgroup),
            ValidationInputLaunch {
                state: input.state.buffer_arg(),
                actors: input.actors.buffer_arg(),
                config: input.config.launch(),
                control: input.control.buffer_arg(),
                ring_mask: ring_mask.buffer_arg(),
                ring_tracking: actor_ring_tracking,
            },
        );
        let selected = |indices: &CubeTensor| {
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
                    state: input.state.buffer_arg(),
                    indices: indices.buffer_arg(),
                    config: input.config.launch(),
                    control: input.control.buffer_arg(),
                    ring_mask: ring_mask.buffer_arg(),
                    ring_tracking,
                },
            );
        };
        match input.rings {
            RingValidation::Selected(indices) => selected(indices),
            RingValidation::Changed {
                entries,
                settlements,
            } => {
                selected(entries);
                selected(settlements);
            }
            _ => (),
        }
        for set in &self.sets {
            MetricSummary::launch(
                device,
                dispatch(device, set.groups, set.ids.meta.num_elements()),
                CubeDim::new_1d(device.workgroup),
                SummaryInputLaunch {
                    state: input.state.buffer_arg(),
                    actors: input.actors.buffer_arg(),
                    config: input.config.launch(),
                    spatial: input.spatial.buffer_arg(),
                    metrics: self.metrics.buffer_arg(),
                    ids: set.ids.buffer_arg(),
                    output: input.partials.buffer_arg(),
                    control: input.control.buffer_arg(),
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
    state[(field * config.state_stride + actor) as usize]
}

#[cube]
fn number(state: &[Ticks], config: &EconomyConfig, field: u32, actor: u32) -> u32 {
    value(state, config, field, actor).extract(LOW_WORD)
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
        if source == comptime!(Source::Field as u32) {
            while member < min((group + 1) * TILE_ACTORS as u32, count) {
                let actor = match comptime!(input.population) {
                    MetricPopulation::Actors => member,
                    MetricPopulation::Firms => {
                        input.spatial[(firm_segment + cells + 1 + member) as usize]
                    }
                };
                let mut selected = true;
                if selected_kind != NO_ACTOR {
                    let actor_kind = kind(actors, actor).bits();
                    selected = selected_kind == actor_kind
                        || (selected_kind == Good::COUNT as u32 + 1 && actor_kind > 0);
                }
                if selected {
                    total = checked_sum(total, native(value(state, config, field, actor)), control);
                }
                member += CUBE_DIM;
            }
        }
        let priced = match source {
            14 | 15 | 16 => true,
            _ => false,
        };
        let quoted = priced
            || source == comptime!(Source::QuoteField as u32)
            || source == comptime!(Source::QuoteSquare as u32)
            || source == comptime!(Source::QuotedFirm as u32);
        while member < min((group + 1) * TILE_ACTORS as u32, count) {
            let actor = match comptime!(input.population) {
                MetricPopulation::Actors => member,
                MetricPopulation::Firms => {
                    input.spatial[(firm_segment + cells + 1 + member) as usize]
                }
            };
            let actor_kind = kind(actors, actor).bits();
            let selected_kind = selected_kind == NO_ACTOR
                || selected_kind == actor_kind
                || (selected_kind == Good::COUNT as u32 + 1 && actor_kind > 0);
            let mut selected = selected_kind;
            if quoted && selected {
                let status = number(
                    state,
                    config,
                    comptime!(Field::EnterpriseStatus as u32),
                    actor,
                );
                selected = status
                    > comptime!(crate::domain::enterprise::EnterpriseState::Forming as u32)
                    && status
                        < comptime!(crate::domain::enterprise::EnterpriseState::Liquidating as u32);
            }
            if priced && selected {
                selected = actor_kind == comptime!(Good::Housing as u32)
                    || actor_kind == comptime!(Good::Railway as u32)
                    || native(value(state, config, actor_kind, actor)) > 0;
            }
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
                    4 | 16 => amount = 1,
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
                    7 | 15 => {
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
                        if comptime!(config.cohorts == Feature::Enabled) {
                            let living = number(
                                state,
                                config,
                                comptime!(Field::LivingMembers as u32),
                                actor,
                            );
                            let workers = min(
                                living,
                                number(
                                    state,
                                    config,
                                    comptime!(Field::EligibleWorkers as u32),
                                    actor,
                                ),
                            );
                            amount = u64::cast_from(select(
                                field == 2,
                                living - workers,
                                select(category == field, workers, 0),
                            ));
                        }
                    }
                    18 => {
                        let employer =
                            number(state, config, comptime!(Field::Employer as u32), actor);
                        amount *= u64::cast_from(
                            employer > 0
                                && employer <= config.actors
                                && kind(actors, min(max(employer, 1), config.actors) - 1).bits()
                                    == comptime!(Good::Railway as u32),
                        );
                    }
                    17 => {
                        amount *= u64::cast_from(
                            comptime!(config.economy == crate::domain::Economy::Industrial)
                                && number(state, config, comptime!(Field::LandPlots as u32), actor)
                                    > 0,
                        );
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
                total = combine_metrics(total, amount, operation, control);
            }
            member += CUBE_DIM;
        }
        let total = reduce_metric(total, operation, control);
        if UNIT_POS == 0 {
            let base = group * SummaryLayout::METRICS as u32 + metric;
            input.output[base as usize] = ticks(u32::cast_from(total), u32::cast_from(total >> 32));
        }
    }
}

#[derive(CubeType, CubeLaunch)]
struct ValidationInput {
    state: Box<[Ticks]>,
    actors: Box<[u32]>,
    config: EconomyConfig,
    control: Box<[Atomic<u32>]>,
    ring_mask: Box<[Atomic<u32>]>,
    #[cube(comptime)]
    ring_tracking: RingTracking,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum RingTracking {
    Untracked,
    Observed,
    ObservedPlane,
}

struct ActorValidation;
struct ValidateActors;

#[derive(CubeType, CubeLaunch)]
struct SelectedValidationInput {
    state: Box<[Ticks]>,
    indices: Box<[u32]>,
    config: EconomyConfig,
    control: Box<[Atomic<u32>]>,
    ring_mask: Box<[Atomic<u32>]>,
    #[cube(comptime)]
    ring_tracking: RingTracking,
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
                &input.ring_mask,
                input.indices[index as usize],
                CUBE_POS_Y,
                comptime!(input.ring_tracking),
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
        validate_actors(
            &input.state,
            &input.actors,
            &input.config,
            &input.control,
            &input.ring_mask,
            comptime!(input.ring_tracking),
        );
    }
}

#[cube]
fn validate_actors(
    state: &[Ticks],
    actors: &[u32],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
    ring_mask: &[Atomic<u32>],
    #[comptime] ring_tracking: RingTracking,
) {
    let actor = (CUBE_POS_X + CUBE_POS_Z * CUBE_COUNT_X) * CUBE_DIM + UNIT_POS;
    let task = CUBE_POS_Y;
    match task {
        0 => {
            validate_actor(state, actors, config, control, actor);
            household_histograms(state, actors, config, control, actor);
        }
        _ => match comptime!(ring_tracking) {
            RingTracking::ObservedPlane => {
                validate_ring(
                    state,
                    config,
                    control,
                    ring_mask,
                    actor,
                    task - 1,
                    ring_tracking,
                );
            }
            _ => {
                if actor < config.actors {
                    validate_ring(
                        state,
                        config,
                        control,
                        ring_mask,
                        actor,
                        task - 1,
                        ring_tracking,
                    );
                }
            }
        },
    }
}

#[cube]
fn validate_ring(
    state: &[Ticks],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
    ring_mask: &[Atomic<u32>],
    actor: u32,
    good: u32,
    #[comptime] ring_tracking: RingTracking,
) {
    let mut held = 0u64;
    if comptime!(ring_tracking != RingTracking::ObservedPlane) || actor < config.actors {
        held = native(value(state, config, TRANSIT_FIELDS as u32 + good, actor));
    }
    let mut in_ring = 0u64;
    for slot in 0u32..config.ring {
        let field = BASE_FIELDS as u32 + good * config.ring + slot;
        let mut waiting = zero();
        let mut amount = zero();
        if comptime!(ring_tracking != RingTracking::ObservedPlane) || actor < config.actors {
            amount = value(state, config, field, actor);
            waiting = value(
                state,
                config,
                field + Good::COUNT as u32 * config.ring,
                actor,
            );
        }
        let block = (field * config.state_stride + actor) * 2
            / crate::simulation::tensor::SNAPSHOT_BLOCK_WORDS as u32;
        match comptime!(ring_tracking) {
            RingTracking::ObservedPlane => {
                let present = plane_any(!equal(amount, zero()));
                if present && UNIT_POS % 32 == 0 {
                    ring_mask[block as usize].store(1);
                }
            }
            RingTracking::Observed => {
                if !equal(amount, zero()) {
                    ring_mask[block as usize].store(1);
                }
            }
            RingTracking::Untracked => (),
        }
        in_ring = checked_sum(
            checked_sum(in_ring, native(amount), control),
            native(waiting),
            control,
        );
    }
    if in_ring != held {
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
                employment::payroll_members(actors, config, actor),
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
        let mut transit = 0u64;
        for good in 0u32..Good::COUNT as u32 {
            let held = value(state, config, TRANSIT_FIELDS as u32 + good, actor);
            transit = checked_sum(transit, native(held), control);
        }
        if transit
            != native(value(
                state,
                config,
                comptime!(Field::Transit as u32),
                actor,
            ))
        {
            KernelFaults::new(comptime!(FaultFlags::TRANSIT_MISMATCH)).record(control);
        }
    }
}

#[cube]
fn household_histograms(
    state: &[Ticks],
    actors: &[u32],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
    actor: u32,
) {
    let histograms = Shared::<[Atomic<u32>]>::new_slice(comptime!(
        SummaryLayout::WEALTH_BINS + SummaryLayout::REGIONS
    ));
    let mut index = UNIT_POS;
    while index < SummaryLayout::WEALTH_BINS as u32 + SummaryLayout::REGIONS as u32 {
        histograms[index as usize].store(0);
        index += CUBE_DIM;
    }
    sync_cube();
    if actor < config.actors && kind(actors, actor).is(comptime!(ActorKind::Household)) {
        let members = select(
            comptime!(config.cohorts == Feature::Enabled),
            number(state, config, comptime!(Field::LivingMembers as u32), actor),
            1,
        );
        let cash = divide(
            value(state, config, comptime!(Field::Cash as u32), actor),
            max(members, 1),
        );
        let mut bin = WORD_BITS - cash.extract(LOW_WORD).leading_zeros();
        if cash.extract(HIGH_WORD) > 0u32 {
            bin = TICK_BITS - cash.extract(HIGH_WORD).leading_zeros();
        }
        histograms[min(bin, SummaryLayout::WEALTH_BINS as u32 - 1) as usize].fetch_add(members);
        let shortage = number(state, config, comptime!(Field::Shortage as u32), actor);
        if shortage > 0 {
            let x = actors[ActorLayout::X * (actors.len() / ActorLayout::WORDS) + actor as usize]
                * SummaryLayout::REGION_SIDE as u32
                / config.map_size;
            let y = actors[ActorLayout::Y * (actors.len() / ActorLayout::WORDS) + actor as usize]
                * SummaryLayout::REGION_SIDE as u32
                / config.map_size;
            histograms[(SummaryLayout::WEALTH_BINS as u32
                + y * SummaryLayout::REGION_SIDE as u32
                + x) as usize]
                .fetch_add(shortage);
        }
    }
    sync_cube();
    index = UNIT_POS;
    while index < SummaryLayout::WEALTH_BINS as u32 + SummaryLayout::REGIONS as u32 {
        let count = histograms[index as usize].load();
        let target = select(
            index < SummaryLayout::WEALTH_BINS as u32,
            SummaryLayout::WEALTH_START as u32 + index,
            SummaryLayout::REGIONS_START as u32 + index - SummaryLayout::WEALTH_BINS as u32,
        );
        if count > 0 {
            control[target as usize].fetch_add(count);
        }
        index += CUBE_DIM;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        domain::{Experiment, Scenario},
        simulation::{Simulation, encode, registry, tests::device},
    };

    #[test]
    fn plane_ring_masks_match_individual_marks_with_partial_actor_groups_on_gpu() -> Result<()> {
        let scenario = Scenario::baseline(4097, 42, Experiment::Baseline)?;
        for workgroup in [32, 256] {
            let simulation = Simulation::new(scenario.clone(), device(workgroup)?)?;
            let mut state = scenario.initial()?;
            let count = state.actors.len();
            let good = Good::COUNT - 1;
            let field = BASE_FIELDS + good * scenario.ring() + scenario.ring() - 1;
            for actor in [0, 31, 32, count - 1] {
                let amount = actor as u64 + 1;
                state.values[field * count + actor] = amount;
                state.values[(TRANSIT_FIELDS + good) * count + actor] = amount;
                state.set(Field::Transit, actor, amount);
            }
            simulation.device.write(
                &simulation.state,
                bytemuck::cast_slice(&encode(&crate::simulation::tensor::state_values(
                    &state.values,
                    count,
                    simulation.config.state_stride as usize,
                ))),
            );
            let mask = simulation.snapshot_copy.ring_mask().expect("ring mask");
            let hardware = &simulation.device.client.properties().hardware;
            let tracking = match (
                parallel::ScanMode::new(&simulation.device),
                hardware.plane_size_min,
                hardware.plane_size_max,
            ) {
                (parallel::ScanMode::Plane, 32, 32) => RingTracking::ObservedPlane,
                _ => RingTracking::Observed,
            };
            crate::simulation::tensor::reset_ring_mask(
                &simulation.device,
                mask,
                crate::simulation::tensor::RingMaskState::Empty,
            );
            ActorValidation::launch(
                &simulation.device,
                dispatch(
                    &simulation.device,
                    count.div_ceil(workgroup as usize),
                    Good::COUNT + 1,
                ),
                CubeDim::new_1d(workgroup),
                ValidationInputLaunch {
                    state: simulation.state.buffer_arg(),
                    actors: simulation.actors.buffer_arg(),
                    config: simulation.config.launch(),
                    control: simulation.control.buffer_arg(),
                    ring_mask: mask.buffer_arg(),
                    ring_tracking: tracking,
                },
            );
            let coalesced = simulation.device.read(mask)?;
            crate::simulation::tensor::reset_ring_mask(
                &simulation.device,
                mask,
                crate::simulation::tensor::RingMaskState::Empty,
            );
            let indices = simulation
                .device
                .tensor(&(0..count as u32).collect::<Vec<_>>());
            SelectedValidation::launch(
                &simulation.device,
                dispatch(
                    &simulation.device,
                    count.div_ceil(workgroup as usize),
                    Good::COUNT,
                ),
                CubeDim::new_1d(workgroup),
                SelectedValidationInputLaunch {
                    state: simulation.state.buffer_arg(),
                    indices: indices.into_buffer_arg(),
                    config: simulation.config.launch(),
                    control: simulation.control.buffer_arg(),
                    ring_mask: mask.buffer_arg(),
                    ring_tracking: RingTracking::Observed,
                },
            );
            assert_eq!(coalesced, simulation.device.read(mask)?);
            assert!(coalesced.iter().any(|&value| value > 0));
            assert_eq!(
                FaultFlags::read(&simulation.device.read(&simulation.control)?)?,
                FaultFlags::empty()
            );
        }
        Ok(())
    }

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
        let entries = simulation.device.tensor(&[0]);
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
            let combined = simulation.observe_rings(RingValidation::Changed {
                entries: &entries,
                settlements: &indices,
            });
            assert_eq!(combined.is_ok(), full.is_ok());
            assert_eq!(control, simulation.device.read(&simulation.control)?);
            assert_eq!(
                summary,
                simulation
                    .device
                    .read_range(&simulation.summary, 0, SummaryLayout::WORDS)?,
            );
            if let (Ok(sparse), Ok(full), Ok(combined)) = (sparse, full, combined) {
                assert_eq!(sparse, full);
                assert_eq!(combined, full);
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

    #[derive(Clone, Copy)]
    enum RegistryCase {
        Capacity,
        Estate,
        Migration,
    }

    struct RegistryResult {
        state: crate::domain::State,
        snapshot: crate::domain::ObservationSnapshot,
    }

    #[test]
    fn capacity_snapshot_reuse_matches_full_birth_and_migration_audits_on_gpu() -> Result<()> {
        let scenario = Scenario::baseline(129, 42, Experiment::Baseline)?;
        let count = scenario.actors();
        let firm = scenario.config().households as usize;
        for case in [
            RegistryCase::Capacity,
            RegistryCase::Estate,
            RegistryCase::Migration,
        ] {
            let mut initial = scenario.initial()?;
            let mut requests = vec![0; count];
            let mut births = vec![0; count];
            match case {
                RegistryCase::Capacity => {
                    initial.set(Field::PendingCapacity, firm, 2);
                    requests[firm] = (6 << 24) | 2;
                    births[0] = 1;
                }
                RegistryCase::Estate => {
                    initial.set(Field::EstateSettled, 0, 1);
                    requests[0] = 5 << 24;
                }
                RegistryCase::Migration => requests[0] = (2 << 24) | (firm as u32 + 1),
            }
            let records = registry::records::PendingRecords::from_dense(&requests, &births)?;
            let mut outcomes = Vec::new();
            for audit in [
                crate::simulation::RegistryAudit::Validated,
                crate::simulation::RegistryAudit::Full,
            ] {
                let mut simulation = Simulation::new(scenario.clone(), device(64)?)?;
                simulation.device.write(
                    &simulation.state,
                    bytemuck::cast_slice(&encode(&initial.values)),
                );
                simulation
                    .device
                    .write(&simulation.enterprise_requests, &requests);
                simulation
                    .device
                    .write(simulation.demography.birth_buffer(), &births);
                simulation.snapshot = simulation.observe()?;
                let before = simulation.snapshot.economy.household_cash;
                let plan = registry::RegistryPlan::new(
                    &scenario,
                    &simulation.metadata,
                    &simulation.migrations,
                    &records.records,
                )?
                .expect("supported registry changes");
                simulation.commit_registry_audited(plan, &records.records, audit)?;
                match case {
                    RegistryCase::Capacity | RegistryCase::Estate => {
                        assert_eq!(simulation.snapshot.economy.household_cash, before);
                    }
                    RegistryCase::Migration => {
                        assert_eq!(simulation.snapshot.economy.household_cash, before - 100);
                    }
                }
                outcomes.push(RegistryResult {
                    state: simulation.download()?,
                    snapshot: simulation.snapshot,
                });
            }
            assert_eq!(outcomes[0].state, outcomes[1].state);
            assert_eq!(outcomes[0].snapshot, outcomes[1].snapshot);
        }
        Ok(())
    }

    #[test]
    fn reused_capacity_snapshot_preserves_existing_gpu_faults_on_gpu() -> Result<()> {
        let scenario = Scenario::baseline(97, 42, Experiment::Baseline)?;
        let mut simulation = Simulation::new(scenario.clone(), device(64)?)?;
        let mut requests = vec![0; scenario.actors()];
        let firm = scenario.config().households as usize;
        requests[firm] = (6 << 24) | 1;
        simulation
            .device
            .write(&simulation.enterprise_requests, &requests);
        let records =
            registry::records::PendingRecords::from_dense(&requests, &vec![0; requests.len()])?;
        let plan = registry::RegistryPlan::new(
            &scenario,
            &simulation.metadata,
            &simulation.migrations,
            &records.records,
        )?
        .expect("capacity registry plan");
        let mut control = simulation.device.read(&simulation.control)?;
        control[crate::simulation::flags::FAULT_WORD] = FaultFlags::ARITHMETIC_OVERFLOW.bits();
        simulation.device.write(&simulation.control, &control);
        let actors = simulation.metadata.actors.clone();
        let error = simulation
            .commit_registry_audited(
                plan,
                &records.records,
                crate::simulation::RegistryAudit::Validated,
            )
            .expect_err("the existing GPU fault must propagate");
        assert!(error.to_string().contains("ARITHMETIC_OVERFLOW"));
        assert_eq!(simulation.metadata.actors, actors);
        Ok(())
    }
}

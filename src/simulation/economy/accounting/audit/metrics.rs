use super::*;
use crate::domain::ActorRows;
use crate::simulation::parallel::reduction::{combine_metrics, reduce_metric};
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(CubeType, Clone, Copy, Pod, Zeroable)]
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

    pub fn summarize(&self, input: AuditInput<'_>) -> Result<()> {
        input.check_actors();
        let DeviceEconomy {
            device,
            state,
            actors,
            config,
        } = input.economy;
        for set in &self.sets {
            MetricSummary::launch(
                device,
                dispatch(device, set.groups, set.ids.meta.num_elements()),
                CubeDim::new_1d(device.workgroup),
                SummaryInputLaunch {
                    state: state.buffer_arg(),
                    actors: actors.buffer_arg(),
                    config: config.launch(),
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
        let group = CUBE_POS_X + CUBE_POS_Z * CUBE_COUNT_X;
        if group < input.groups {
            let metric = input.ids[CUBE_POS_Y as usize];
            let total = input.measure(metric, group);
            if UNIT_POS == 0 {
                let base = group * SummaryLayout::METRICS as u32 + metric;
                input.output[base as usize] =
                    ticks(u32::cast_from(total), u32::cast_from(total >> 32));
            }
        }
    }
}

#[cube]
impl Metric {
    fn unpack(vector: Vector<u32, Const<4>>) -> Self {
        Metric {
            field: vector.extract(0usize),
            kind: vector.extract(1usize),
            operation: vector.extract(2usize),
            source: vector.extract(3usize),
        }
    }

    fn identity(&self) -> u64 {
        select(self.operation == 1, u64::MAX, 0u64)
    }

    fn includes_kind(&self, kind: u32) -> bool {
        self.kind == NO_ACTOR
            || self.kind == kind
            || (self.kind == Good::COUNT as u32 + 1 && kind > 0)
    }

    fn needs_stock(&self) -> bool {
        match self.source {
            14 | 15 | 16 => true,
            _ => false,
        }
    }

    fn needs_quote(&self) -> bool {
        self.needs_stock()
            || self.source == comptime!(Source::QuoteField as u32)
            || self.source == comptime!(Source::QuoteSquare as u32)
            || self.source == comptime!(Source::QuotedFirm as u32)
    }

    fn value(
        &self,
        ledger: &TickLedger<'_>,
        actors: &ActorTable<'_>,
        control: &[Atomic<u32>],
        actor: u32,
        actor_kind: u32,
    ) -> u64 {
        let config = ledger.config;
        let mut amount = native(ledger.value_at(self.field, actor));
        match self.source {
            1 => {
                for coordinate in [
                    Field::Escrow as u32,
                    Field::PayrollReserve as u32,
                    Field::TaxTreasury as u32,
                    Field::ChargeTreasury as u32,
                    Field::RentPending as u32,
                ] {
                    amount =
                        checked_sum(amount, native(ledger.value_at(coordinate, actor)), control);
                }
            }
            2 => {
                if actor_kind == self.field {
                    amount = checked_sum(
                        amount,
                        native(ledger.value(Field::Pipeline, actor)),
                        control,
                    );
                }
            }
            3 => {
                if actor_kind == self.field - PRODUCTION_FIELDS as u32 + 1 {
                    amount = checked_sum(
                        amount,
                        native(ledger.value(Field::Produced, actor)),
                        control,
                    );
                }
            }
            4 | 16 => amount = 1,
            5 => {
                amount = u64::cast_from(
                    ledger.number(Field::EnterpriseModel, actor) == 1
                        && ledger.number(Field::EnterpriseStatus, actor)
                            != comptime!(crate::domain::enterprise::EnterpriseState::Closed as u32),
                )
            }
            6 => amount = u64::cast_from(ledger.number(Field::EnterpriseStatus, actor) == 4),
            7 | 15 => {
                amount &= 0xffff_ffff;
                amount *= amount;
            }
            9 => {
                let working = ledger.number(Field::Worker, actor) != NO_ACTOR
                    && ledger.number(Field::LivingMembers, actor) > 0;
                let subsistence = comptime!(config.economy == crate::domain::Economy::Industrial)
                    && ledger.number(Field::LandPlots, actor) > 0;
                let category = select(working, select(subsistence, 1u32, 0u32), 2u32);
                amount = u64::cast_from(category == self.field);
                if comptime!(config.cohorts == Feature::Enabled) {
                    let living = ledger.number(Field::LivingMembers, actor);
                    let workers = min(living, ledger.number(Field::EligibleWorkers, actor));
                    amount = u64::cast_from(select(
                        self.field == 2,
                        living - workers,
                        select(category == self.field, workers, 0),
                    ));
                }
            }
            18 => {
                let employer = ledger.number(Field::Employer, actor);
                amount *= u64::cast_from(
                    employer > 0
                        && employer <= config.actors
                        && actors.kind(min(max(employer, 1), config.actors) - 1).bits()
                            == comptime!(Good::Railway as u32),
                );
            }
            17 => {
                amount *= u64::cast_from(
                    comptime!(config.economy == crate::domain::Economy::Industrial)
                        && ledger.number(Field::LandPlots, actor) > 0,
                );
            }
            10 => {
                amount = u64::cast_from(ledger.number(Field::EnterpriseStatus, actor) < 4);
            }
            11 => {
                let category = select(
                    actor_kind == comptime!(Good::Housing as u32),
                    2u32,
                    ledger.number(Field::EnterpriseModel, actor),
                );
                amount = u64::cast_from(
                    category == self.field && ledger.number(Field::EnterpriseStatus, actor) < 4,
                );
            }
            12 => {
                amount =
                    u64::cast_from(ledger.number(Field::EnterpriseStatus, actor) == self.field);
            }
            8 => {
                amount = u64::cast_from(
                    ledger.number(Field::LivingMembers, actor) > 0
                        && ledger.number(Field::HousingStatus, actor)
                            == comptime!(crate::domain::HousingState::Homeless as u32),
                );
            }
            _ => (),
        }
        amount
    }
}

#[cube]
impl SummaryInput {
    fn firm_index_offset(&self) -> u32 {
        let cells = self.config.spatial_side * self.config.spatial_side;
        2 * (cells + 1 + self.config.actors) + cells
    }

    fn members(&self, firms: u32) -> u32 {
        match comptime!(self.population) {
            MetricPopulation::Actors => self.config.actors,
            MetricPopulation::Firms => self.spatial[firms as usize],
        }
    }

    fn member_actor(&self, firms: u32, member: u32) -> u32 {
        match comptime!(self.population) {
            MetricPopulation::Actors => member,
            MetricPopulation::Firms => self.spatial[(firms + 1 + member) as usize],
        }
    }

    fn measure(&self, metric: u32, group: u32) -> u64 {
        let ledger = TickLedger::new(&self.state, &self.config);
        let actors = ActorTable::new(&self.actors);
        let control: &[Atomic<u32>] = &self.control;
        let definition = Metric::unpack(self.metrics[metric as usize]);
        let mut total = definition.identity();
        let firms = self.firm_index_offset();
        let end = min((group + 1) * TILE_ACTORS as u32, self.members(firms));
        let mut member = group * TILE_ACTORS as u32 + UNIT_POS;
        if definition.source == comptime!(Source::Field as u32) {
            while member < end {
                let actor = self.member_actor(firms, member);
                let mut selected = true;
                if definition.kind != NO_ACTOR {
                    selected = definition.includes_kind(actors.kind(actor).bits());
                }
                if selected {
                    total = checked_sum(
                        total,
                        native(ledger.value_at(definition.field, actor)),
                        control,
                    );
                }
                member += CUBE_DIM;
            }
        }
        let priced = definition.needs_stock();
        let quoted = definition.needs_quote();
        while member < end {
            let actor = self.member_actor(firms, member);
            let actor_kind = actors.kind(actor).bits();
            let mut selected = definition.includes_kind(actor_kind);
            if quoted && selected {
                let status = ledger.number(Field::EnterpriseStatus, actor);
                selected = status
                    > comptime!(crate::domain::enterprise::EnterpriseState::Forming as u32)
                    && status
                        < comptime!(crate::domain::enterprise::EnterpriseState::Liquidating as u32);
            }
            if priced && selected {
                selected = pricing::stocked::<TickLedger>(&ledger, actor, actor_kind);
            }
            if selected {
                total = combine_metrics(
                    total,
                    definition.value(&ledger, &actors, control, actor, actor_kind),
                    definition.operation,
                    control,
                );
            }
            if priced && ledger.number(Field::EnterpriseModel, actor) == 1 {
                let status = ledger.number(Field::EnterpriseStatus, actor);
                let trading = status
                    > comptime!(crate::domain::enterprise::EnterpriseState::Forming as u32)
                    && status
                        < comptime!(crate::domain::enterprise::EnterpriseState::Liquidating as u32);
                if trading {
                    let global = definition.kind > Good::COUNT as u32;
                    let first = select(global, 1, definition.kind);
                    let last = select(global, Good::COUNT as u32, definition.kind);
                    for good in first..last + 1 {
                        if good != actor_kind && !equal(ledger.value_at(good, actor), zero()) {
                            let quote = ledger.number_at(
                                pricing::retail_field(
                                    &self.config,
                                    good,
                                    comptime!(crate::domain::RetailAccount::Quote),
                                ),
                                actor,
                            );
                            if quote > 0 {
                                let price = u64::cast_from(quote);
                                let amount = match definition.source {
                                    16 => 1u64,
                                    15 => price * price,
                                    _ => price,
                                };
                                total =
                                    combine_metrics(total, amount, definition.operation, control);
                            }
                        }
                    }
                }
            }
            member += CUBE_DIM;
        }
        reduce_metric(total, definition.operation, control)
    }
}

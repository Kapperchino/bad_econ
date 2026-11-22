use crate::simulation::{
    economy::*,
    state::{EconomyViewExpand, LedgerExpand},
};

#[cube]
pub(in crate::simulation::economy) fn wealth_bin<L: ReadLedger>(
    ledger: &L,
    actor: u32,
    members: u32,
) -> u32 {
    let cash = divide(ledger.value(Field::Cash, actor), max(members, 1));
    let high = cash.extract(HIGH_WORD);
    let bits = select(
        high > 0,
        TICK_BITS - high.leading_zeros(),
        WORD_BITS - cash.extract(LOW_WORD).leading_zeros(),
    );
    min(bits, SummaryLayout::WEALTH_BINS as u32 - 1)
}

#[cube]
pub(in crate::simulation::economy) fn histogram_region(
    actors: &ActorTable<'_>,
    config: &EconomyConfig,
    actor: u32,
) -> u32 {
    let side = SummaryLayout::REGION_SIDE as u32;
    let x = actors.x(actor) * side / config.map_size;
    let y = actors.y(actor) * side / config.map_size;
    y * side + x
}

#[derive(CubeType)]
struct ObservationTotals<'a> {
    values: Array<Ticks>,
    control: &'a [Atomic<u32>],
}

#[cube]
impl<'a> ObservationTotals<'a> {
    fn new(control: &'a [Atomic<u32>]) -> Self {
        let mut totals = Array::<Ticks>::new(SummaryLayout::METRICS);
        for metric in 0..SummaryLayout::METRICS {
            totals[metric] = zero();
        }
        totals[SummaryLayout::MINIMUM_WAGE as usize] = ticks(u32::MAX, u32::MAX);
        totals[SummaryLayout::MINIMUM_PRICE as usize] = ticks(u32::MAX, u32::MAX);
        for good in 0u32..Good::COUNT as u32 {
            totals[(SummaryLayout::WAGE_MINIMA + good) as usize] = ticks(u32::MAX, u32::MAX);
            totals[(SummaryLayout::PRICE_MINIMA + good) as usize] = ticks(u32::MAX, u32::MAX);
        }
        ObservationTotals::<'a> {
            values: totals,
            control,
        }
    }

    fn add(&mut self, metric: usize, value: Ticks) {
        self.values[metric] = plus(self.values[metric], value).record(self.control);
    }

    fn write(&self, bundles: &mut [u32], id: u32) {
        for metric in 0u32..SummaryLayout::METRICS as u32 {
            let base = (id * SummaryLayout::METRICS as u32 + metric) * TICK_WORDS as u32;
            bundles[(base) as usize] = self.values[(metric) as usize].extract(LOW_WORD);
            bundles[(base + 1u32) as usize] = self.values[(metric) as usize].extract(HIGH_WORD);
        }
    }
}

#[cube]
impl<'a> EconomyView<'a> {
    pub(in crate::simulation::economy) fn summarize(
        &self,
        bundles: &mut [u32],
        control: &[Atomic<u32>],
        id: u32,
    ) {
        let config = self.ledger.config;
        let groups = ceildiv(config.actors, SummaryLayout::ACTORS_PER_GROUP);
        if id < groups {
            let mut totals = ObservationTotals::new(control);
            let count = config.actors;
            for offset in 0..ceildiv(count - id, groups) {
                let actor = id + offset * groups;
                totals.add(
                    SummaryLayout::CASH as usize,
                    plus(
                        self.ledger.value(Field::Cash, actor),
                        self.ledger.value(Field::Escrow, actor),
                    )
                    .record(control),
                );
                let rent = self.ledger.value(Field::RentTreasury, actor);
                totals.add(SummaryLayout::RENT_COLLECTED as usize, rent);
                totals.add(
                    SummaryLayout::RENT_ARREARS as usize,
                    self.ledger.value(Field::RentArrears, actor),
                );
                totals.add(
                    SummaryLayout::CASH as usize,
                    self.ledger.value(Field::RentPending, actor),
                );
                let payroll_reserve = self.ledger.value(Field::PayrollReserve, actor);
                let tax = self.ledger.value(Field::TaxTreasury, actor);
                let charge = self.ledger.value(Field::ChargeTreasury, actor);
                totals.add(SummaryLayout::PAYROLL_RESERVES as usize, payroll_reserve);
                totals.add(SummaryLayout::INCOME_TAX as usize, tax);
                totals.add(SummaryLayout::EMPLOYEE_CHARGES as usize, charge);
                totals.add(
                    SummaryLayout::CASH as usize,
                    plus(payroll_reserve, plus(tax, charge).record(control)).record(control),
                );
                if comptime!(config.payroll == Feature::Enabled) {
                    let accrued = PayrollAccrual::new(
                        self.ledger.value(Field::GrossIncome, actor),
                        employment::payroll_members(&self.actors, config, actor),
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
                for g in 0u32..Good::COUNT as u32 {
                    totals.add(
                        (SummaryLayout::INVENTORIES + g) as usize,
                        self.ledger.value_at(g + 1u32, actor),
                    );
                }
                if !equal(self.ledger.value(Field::Escrow, actor), zero())
                    || !equal(self.ledger.value(Field::Pipeline, actor), zero())
                    || !equal(self.ledger.value(Field::Due, actor), zero())
                {
                    KernelFaults::new(comptime!(FaultFlags::UNSETTLED_COMMITMENTS)).record(control);
                }
                let employer = self.ledger.number(Field::Employer, actor);
                if employer > 0
                    && employer <= config.actors
                    && self.actors.kind(max(employer, 1) - 1).bits()
                        == comptime!(Good::Railway as u32)
                {
                    totals.add(
                        SummaryLayout::TRANSPORT_WORKERS as usize,
                        self.ledger.value(Field::Employed, actor),
                    );
                }
                totals.add(
                    SummaryLayout::CARTAGE_MINUTES as usize,
                    self.ledger.value(Field::CartageMinutes, actor),
                );
                totals.add(
                    SummaryLayout::CARTAGE_RECEIPTS as usize,
                    self.ledger.value(Field::CartageIncome, actor),
                );
                totals.add(
                    SummaryLayout::AWAITING_PICKUP as usize,
                    self.ledger.value(Field::RoadFreight, actor),
                );
                let k = self.actors.kind(actor);
                match k.bits() {
                    ActorKind::HOUSEHOLD_BITS => {
                        totals.add(
                            SummaryLayout::HOMELESS_HOUSEHOLDS as usize,
                            ticks(
                                u32::cast_from(
                                    self.ledger.number(Field::LivingMembers, actor) > 0
                                        && self.ledger.number(Field::HousingStatus, actor)
                                            == comptime!(
                                                crate::domain::HousingState::Homeless as u32
                                            ),
                                ),
                                0,
                            ),
                        );
                        totals.add(
                            SummaryLayout::EMPLOYED as usize,
                            self.ledger.value(Field::Employed, actor),
                        );
                        totals.add(
                            SummaryLayout::SHORTAGES as usize,
                            self.ledger.value(Field::Shortage, actor),
                        );
                        totals.add(
                            SummaryLayout::CONSUMED as usize,
                            self.ledger.value(Field::Consumed, actor),
                        );
                        let members = consumption::consumers(&self.ledger, actor);
                        let bin = wealth_bin::<Ledger>(&self.ledger, actor, members);
                        control[(SummaryLayout::WEALTH_START as u32 + bin) as usize]
                            .fetch_add(members);
                        let region = histogram_region(&self.actors, config, actor);
                        control[(SummaryLayout::REGIONS_START as u32 + region) as usize]
                            .fetch_add(self.ledger.number(Field::Shortage, actor));
                    }
                    _ => {
                        totals.add(
                            (SummaryLayout::INVENTORIES + k.output() - 1) as usize,
                            self.ledger.value(Field::Pipeline, actor),
                        );
                        totals.add(
                            (SummaryLayout::PRODUCED + k.output() - 1) as usize,
                            self.ledger.value(Field::Produced, actor),
                        );
                        totals.add(
                            SummaryLayout::DISTRIBUTORS as usize,
                            ticks(
                                u32::cast_from(
                                    self.ledger.number(Field::EnterpriseModel, actor) == 1
                                        && self.ledger.number(Field::EnterpriseStatus, actor)
                                            != comptime!(
                                                crate::domain::enterprise::EnterpriseState::Closed
                                                    as u32
                                            ),
                                ),
                                0,
                            ),
                        );
                        totals.add(
                            SummaryLayout::CLOSURES as usize,
                            ticks(
                                u32::cast_from(
                                    self.ledger.number(Field::EnterpriseStatus, actor) == 4,
                                ),
                                0,
                            ),
                        );
                        let status = self.ledger.number(Field::EnterpriseStatus, actor);
                        let trading = status
                            > comptime!(crate::domain::enterprise::EnterpriseState::Forming as u32)
                            && status
                                < comptime!(
                                    crate::domain::enterprise::EnterpriseState::Liquidating as u32
                                );
                        if trading {
                            let good = k.output() - 1;
                            let wage = self.ledger.value(Field::Wage, actor);
                            totals.add((SummaryLayout::FIRMS_BY_GOOD + good) as usize, ticks(1, 0));
                            totals.values[(SummaryLayout::WAGE_MINIMA + good) as usize] = minimum(
                                totals.values[(SummaryLayout::WAGE_MINIMA + good) as usize],
                                wage,
                            );
                            totals.values[(SummaryLayout::WAGE_MAXIMA + good) as usize] = maximum(
                                totals.values[(SummaryLayout::WAGE_MAXIMA + good) as usize],
                                wage,
                            );
                            totals.add((SummaryLayout::WAGE_TOTALS + good) as usize, wage);
                            totals.add(
                                (SummaryLayout::WAGE_SQUARES + good) as usize,
                                times32(wage.extract(LOW_WORD), wage.extract(LOW_WORD)),
                            );
                            totals.values[SummaryLayout::MINIMUM_WAGE as usize] = minimum(
                                totals.values[SummaryLayout::MINIMUM_WAGE as usize],
                                self.ledger.value(Field::Wage, actor),
                            );
                            totals.values[SummaryLayout::MAXIMUM_WAGE as usize] = select(
                                lt(
                                    totals.values[SummaryLayout::MAXIMUM_WAGE as usize],
                                    self.ledger.value(Field::Wage, actor),
                                ),
                                self.ledger.value(Field::Wage, actor),
                                totals.values[SummaryLayout::MAXIMUM_WAGE as usize],
                            );
                            if pricing::stocked::<Ledger>(&self.ledger, actor, k.output()) {
                                let price = self.ledger.value(Field::Price, actor);
                                totals.add(
                                    (SummaryLayout::PRICE_COUNTS + good) as usize,
                                    ticks(1, 0),
                                );
                                totals.values[(SummaryLayout::PRICE_MINIMA + good) as usize] =
                                    minimum(
                                        totals.values
                                            [(SummaryLayout::PRICE_MINIMA + good) as usize],
                                        price,
                                    );
                                totals.values[(SummaryLayout::PRICE_MAXIMA + good) as usize] =
                                    maximum(
                                        totals.values
                                            [(SummaryLayout::PRICE_MAXIMA + good) as usize],
                                        price,
                                    );
                                totals.add((SummaryLayout::PRICE_TOTALS + good) as usize, price);
                                totals.add(
                                    (SummaryLayout::PRICE_SQUARES + good) as usize,
                                    times32(price.extract(LOW_WORD), price.extract(LOW_WORD)),
                                );
                                totals.values[SummaryLayout::MINIMUM_PRICE as usize] = minimum(
                                    totals.values[SummaryLayout::MINIMUM_PRICE as usize],
                                    price,
                                );
                                totals.values[SummaryLayout::MAXIMUM_PRICE as usize] = maximum(
                                    totals.values[SummaryLayout::MAXIMUM_PRICE as usize],
                                    price,
                                );
                            }
                        }
                    }
                }
                match k.bits() {
                    ActorKind::HOUSEHOLD_BITS => {
                        let working = self.ledger.number(Field::Worker, actor) != NO_ACTOR
                            && self.ledger.number(Field::LivingMembers, actor) > 0;
                        let subsistence =
                            comptime!(config.economy == crate::domain::Economy::Industrial)
                                && self.ledger.number(Field::LandPlots, actor) > 0;
                        let category = select(working, select(subsistence, 1u32, 0u32), 2u32);
                        let metric = (SummaryLayout::WORKFORCE + category) as usize;
                        if comptime!(config.cohorts == Feature::Enabled) {
                            let living = self.ledger.number(Field::LivingMembers, actor);
                            let workers =
                                min(living, self.ledger.number(Field::EligibleWorkers, actor));
                            let inactive = comptime!(SummaryLayout::INACTIVE as usize);
                            totals.add(inactive, ticks(living - workers, 0));
                            totals.add(metric, ticks(workers, 0));
                        } else {
                            totals.add(metric, ticks(1, 0));
                        }
                        totals.add(
                            SummaryLayout::RURAL_EMPLOYED as usize,
                            select(
                                subsistence,
                                self.ledger.value(Field::Employed, actor),
                                zero(),
                            ),
                        );
                        totals.add(
                            SummaryLayout::HOUSEHOLD_CASH as usize,
                            self.ledger.value(Field::Cash, actor),
                        );
                    }
                    _ => {
                        let good = k.output() - 1;
                        let status = self.ledger.number(Field::EnterpriseStatus, actor);
                        if status < 4 {
                            let category = select(
                                k.output() == comptime!(Good::Housing as u32),
                                2u32,
                                self.ledger.number(Field::EnterpriseModel, actor),
                            );
                            totals
                                .add((SummaryLayout::ACTIVE_BY_GOOD + good) as usize, ticks(1, 0));
                            totals.add(
                                (SummaryLayout::BUSINESS_TYPES + category) as usize,
                                ticks(1, 0),
                            );
                            totals.add(
                                (SummaryLayout::BUSINESS_STATES + status) as usize,
                                ticks(1, 0),
                            );
                        }
                        totals.add(
                            (SummaryLayout::FIRM_CASH + good) as usize,
                            self.ledger.value(Field::Cash, actor),
                        );
                        totals.add(
                            (SummaryLayout::BUSINESS_OPENING + good) as usize,
                            self.ledger.value(Field::BusinessOpeningCash, actor),
                        );
                        totals.add(
                            (SummaryLayout::BUSINESS_CLOSING + good) as usize,
                            self.ledger.value(Field::BusinessClosingCash, actor),
                        );
                        totals.add(
                            (SummaryLayout::BUSINESS_REVENUE + good) as usize,
                            self.ledger.value(Field::BusinessRevenue, actor),
                        );
                    }
                }
                let mut transit = ticks(0u32, 0u32);
                for good in 0u32..Good::COUNT as u32 {
                    let held = self.ledger.value_at(TRANSIT_FIELDS as u32 + good, actor);
                    totals.add((SummaryLayout::TRANSIT + good) as usize, held);
                    let household_output =
                        self.ledger.value_at(PRODUCTION_FIELDS as u32 + good, actor);
                    totals.add(
                        (SummaryLayout::HOUSEHOLD_PRODUCTION + good) as usize,
                        household_output,
                    );
                    totals.add((SummaryLayout::PRODUCED + good) as usize, household_output);
                    totals.add(
                        (SummaryLayout::USED + good) as usize,
                        self.ledger.value_at(USED_FIELDS as u32 + good, actor),
                    );
                    totals.add(
                        (SummaryLayout::CONSUMPTION + good) as usize,
                        self.ledger
                            .value_at(CONSUMPTION_FIELDS as u32 + good, actor),
                    );
                    let mut in_ring = zero();
                    for slot in 0u32..config.ring {
                        in_ring = plus(
                            in_ring,
                            self.ledger.value_at(
                                BASE_FIELDS as u32
                                    + (Good::COUNT as u32 + good) * config.ring
                                    + slot,
                                actor,
                            ),
                        )
                        .record(control);
                        in_ring = plus(
                            in_ring,
                            self.ledger
                                .value_at(BASE_FIELDS as u32 + good * config.ring + slot, actor),
                        )
                        .record(control);
                    }
                    if !equal(in_ring, held) {
                        KernelFaults::new(comptime!(FaultFlags::TRANSIT_MISMATCH)).record(control);
                    }
                    transit = plus(transit, held).record(control);
                }
                if !equal(transit, self.ledger.value(Field::Transit, actor)) {
                    KernelFaults::new(comptime!(FaultFlags::TRANSIT_MISMATCH)).record(control);
                }
            }
            totals.write(bundles, id);
        }
    }
}

#[cube]
impl<'a> Ledger<'a> {
    pub(in crate::simulation::economy) fn inspect(
        &self,
        proposals: &mut [u32],
        parameters: &EconomyParameters,
        id: u32,
    ) {
        if id < self.config.fields {
            store(proposals, id, self.value_at(id, parameters.offset));
        }
    }
}

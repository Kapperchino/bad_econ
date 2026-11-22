use super::*;

#[cube]
pub(super) fn wealth_bin<L: ReadLedger>(ledger: &L, actor: u32, members: u32) -> u32 {
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
pub(super) fn histogram_region<A: ActorMetadata>(
    actors: &A,
    config: &EconomyConfig,
    actor: u32,
) -> u32 {
    let side = SummaryLayout::REGION_SIDE as u32;
    let x = actors.x(actor) * side / config.map_size;
    let y = actors.y(actor) * side / config.map_size;
    y * side + x
}

#[cube]
pub(super) fn summarize(
    economy: &EconomyView<'_>,
    bundles: &mut [u32],
    control: &[Atomic<u32>],
    id: u32,
) {
    let config = economy.ledger.config;
    let groups = ceildiv(config.actors, SummaryLayout::ACTORS_PER_GROUP);
    if id < groups {
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
        let count = config.actors;
        for offset in 0..ceildiv(count - id, groups) {
            let actor = id + offset * groups;
            totals[SummaryLayout::CASH as usize] = add_checked(
                control,
                totals[SummaryLayout::CASH as usize],
                add_checked(
                    control,
                    economy.ledger.value(Field::Cash, actor),
                    economy.ledger.value(Field::Escrow, actor),
                ),
            );
            let rent = economy.ledger.value(Field::RentTreasury, actor);
            totals[SummaryLayout::RENT_COLLECTED as usize] = add_checked(
                control,
                totals[SummaryLayout::RENT_COLLECTED as usize],
                rent,
            );
            totals[SummaryLayout::RENT_ARREARS as usize] = add_checked(
                control,
                totals[SummaryLayout::RENT_ARREARS as usize],
                economy.ledger.value(Field::RentArrears, actor),
            );
            totals[SummaryLayout::CASH as usize] = add_checked(
                control,
                totals[SummaryLayout::CASH as usize],
                economy.ledger.value(Field::RentPending, actor),
            );
            let payroll_reserve = economy.ledger.value(Field::PayrollReserve, actor);
            let tax = economy.ledger.value(Field::TaxTreasury, actor);
            let charge = economy.ledger.value(Field::ChargeTreasury, actor);
            totals[SummaryLayout::PAYROLL_RESERVES as usize] = add_checked(
                control,
                totals[SummaryLayout::PAYROLL_RESERVES as usize],
                payroll_reserve,
            );
            totals[SummaryLayout::INCOME_TAX as usize] =
                add_checked(control, totals[SummaryLayout::INCOME_TAX as usize], tax);
            totals[SummaryLayout::EMPLOYEE_CHARGES as usize] = add_checked(
                control,
                totals[SummaryLayout::EMPLOYEE_CHARGES as usize],
                charge,
            );
            totals[SummaryLayout::CASH as usize] = add_checked(
                control,
                totals[SummaryLayout::CASH as usize],
                add_checked(control, payroll_reserve, add_checked(control, tax, charge)),
            );
            if comptime!(config.payroll == Feature::Enabled) {
                let accrued = PayrollAccrual::new(
                    economy.ledger.value(Field::GrossIncome, actor),
                    employment::payroll_members(&economy.actors, config, actor),
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
                totals[(SummaryLayout::INVENTORIES + g) as usize] = add_checked(
                    control,
                    totals[(SummaryLayout::INVENTORIES + g) as usize],
                    economy.ledger.value_at(g + 1u32, actor),
                );
            }
            if !equal(economy.ledger.value(Field::Escrow, actor), zero())
                || !equal(economy.ledger.value(Field::Pipeline, actor), zero())
                || !equal(economy.ledger.value(Field::Due, actor), zero())
            {
                KernelFaults::new(comptime!(FaultFlags::UNSETTLED_COMMITMENTS)).record(control);
            }
            let employer = economy.ledger.number(Field::Employer, actor);
            if employer > 0
                && employer <= config.actors
                && economy.actors.kind(max(employer, 1) - 1).bits()
                    == comptime!(Good::Railway as u32)
            {
                totals[SummaryLayout::TRANSPORT_WORKERS as usize] = add_checked(
                    control,
                    totals[SummaryLayout::TRANSPORT_WORKERS as usize],
                    economy.ledger.value(Field::Employed, actor),
                );
            }
            totals[SummaryLayout::CARTAGE_MINUTES as usize] = add_checked(
                control,
                totals[SummaryLayout::CARTAGE_MINUTES as usize],
                economy.ledger.value(Field::CartageMinutes, actor),
            );
            totals[SummaryLayout::CARTAGE_RECEIPTS as usize] = add_checked(
                control,
                totals[SummaryLayout::CARTAGE_RECEIPTS as usize],
                economy.ledger.value(Field::CartageIncome, actor),
            );
            totals[SummaryLayout::AWAITING_PICKUP as usize] = add_checked(
                control,
                totals[SummaryLayout::AWAITING_PICKUP as usize],
                economy.ledger.value(Field::RoadFreight, actor),
            );
            let k = economy.actors.kind(actor);
            match k.bits() {
                ActorKind::HOUSEHOLD_BITS => {
                    totals[SummaryLayout::HOMELESS_HOUSEHOLDS as usize] = add_checked(
                        control,
                        totals[SummaryLayout::HOMELESS_HOUSEHOLDS as usize],
                        ticks(
                            u32::cast_from(
                                economy.ledger.number(Field::LivingMembers, actor) > 0
                                    && economy.ledger.number(Field::HousingStatus, actor)
                                        == comptime!(crate::domain::HousingState::Homeless as u32),
                            ),
                            0,
                        ),
                    );
                    totals[SummaryLayout::EMPLOYED as usize] = add_checked(
                        control,
                        totals[SummaryLayout::EMPLOYED as usize],
                        economy.ledger.value(Field::Employed, actor),
                    );
                    totals[SummaryLayout::SHORTAGES as usize] = add_checked(
                        control,
                        totals[SummaryLayout::SHORTAGES as usize],
                        economy.ledger.value(Field::Shortage, actor),
                    );
                    totals[SummaryLayout::CONSUMED as usize] = add_checked(
                        control,
                        totals[SummaryLayout::CONSUMED as usize],
                        economy.ledger.value(Field::Consumed, actor),
                    );
                    let members = consumption::consumers(&economy.ledger, actor);
                    let bin = wealth_bin::<Ledger>(&economy.ledger, actor, members);
                    control[(SummaryLayout::WEALTH_START as u32 + bin) as usize].fetch_add(members);
                    let region = histogram_region::<ActorTable>(&economy.actors, config, actor);
                    control[(SummaryLayout::REGIONS_START as u32 + region) as usize]
                        .fetch_add(economy.ledger.number(Field::Shortage, actor));
                }
                _ => {
                    totals[(SummaryLayout::INVENTORIES + k.output() - 1) as usize] = add_checked(
                        control,
                        totals[(SummaryLayout::INVENTORIES + k.output() - 1) as usize],
                        economy.ledger.value(Field::Pipeline, actor),
                    );
                    totals[(SummaryLayout::PRODUCED + k.output() - 1) as usize] = add_checked(
                        control,
                        totals[(SummaryLayout::PRODUCED + k.output() - 1) as usize],
                        economy.ledger.value(Field::Produced, actor),
                    );
                    totals[SummaryLayout::DISTRIBUTORS as usize] = add_checked(
                        control,
                        totals[SummaryLayout::DISTRIBUTORS as usize],
                        ticks(
                            u32::cast_from(
                                economy.ledger.number(Field::EnterpriseModel, actor) == 1
                                    && economy.ledger.number(Field::EnterpriseStatus, actor)
                                        != comptime!(
                                            crate::domain::enterprise::EnterpriseState::Closed
                                                as u32
                                        ),
                            ),
                            0,
                        ),
                    );
                    totals[SummaryLayout::CLOSURES as usize] = add_checked(
                        control,
                        totals[SummaryLayout::CLOSURES as usize],
                        ticks(
                            u32::cast_from(
                                economy.ledger.number(Field::EnterpriseStatus, actor) == 4,
                            ),
                            0,
                        ),
                    );
                    let status = economy.ledger.number(Field::EnterpriseStatus, actor);
                    let trading = status
                        > comptime!(crate::domain::enterprise::EnterpriseState::Forming as u32)
                        && status
                            < comptime!(
                                crate::domain::enterprise::EnterpriseState::Liquidating as u32
                            );
                    if trading {
                        let good = k.output() - 1;
                        let wage = economy.ledger.value(Field::Wage, actor);
                        totals[(SummaryLayout::FIRMS_BY_GOOD + good) as usize] = add_checked(
                            control,
                            totals[(SummaryLayout::FIRMS_BY_GOOD + good) as usize],
                            ticks(1, 0),
                        );
                        totals[(SummaryLayout::WAGE_MINIMA + good) as usize] =
                            minimum(totals[(SummaryLayout::WAGE_MINIMA + good) as usize], wage);
                        totals[(SummaryLayout::WAGE_MAXIMA + good) as usize] =
                            maximum(totals[(SummaryLayout::WAGE_MAXIMA + good) as usize], wage);
                        totals[(SummaryLayout::WAGE_TOTALS + good) as usize] = add_checked(
                            control,
                            totals[(SummaryLayout::WAGE_TOTALS + good) as usize],
                            wage,
                        );
                        totals[(SummaryLayout::WAGE_SQUARES + good) as usize] = add_checked(
                            control,
                            totals[(SummaryLayout::WAGE_SQUARES + good) as usize],
                            times32(wage.extract(LOW_WORD), wage.extract(LOW_WORD)),
                        );
                        totals[SummaryLayout::MINIMUM_WAGE as usize] = minimum(
                            totals[SummaryLayout::MINIMUM_WAGE as usize],
                            economy.ledger.value(Field::Wage, actor),
                        );
                        totals[SummaryLayout::MAXIMUM_WAGE as usize] = select(
                            lt(
                                totals[SummaryLayout::MAXIMUM_WAGE as usize],
                                economy.ledger.value(Field::Wage, actor),
                            ),
                            economy.ledger.value(Field::Wage, actor),
                            totals[SummaryLayout::MAXIMUM_WAGE as usize],
                        );
                        if pricing::stocked::<Ledger>(&economy.ledger, actor, k.output()) {
                            let price = economy.ledger.value(Field::Price, actor);
                            totals[(SummaryLayout::PRICE_COUNTS + good) as usize] = add_checked(
                                control,
                                totals[(SummaryLayout::PRICE_COUNTS + good) as usize],
                                ticks(1, 0),
                            );
                            totals[(SummaryLayout::PRICE_MINIMA + good) as usize] = minimum(
                                totals[(SummaryLayout::PRICE_MINIMA + good) as usize],
                                price,
                            );
                            totals[(SummaryLayout::PRICE_MAXIMA + good) as usize] = maximum(
                                totals[(SummaryLayout::PRICE_MAXIMA + good) as usize],
                                price,
                            );
                            totals[(SummaryLayout::PRICE_TOTALS + good) as usize] = add_checked(
                                control,
                                totals[(SummaryLayout::PRICE_TOTALS + good) as usize],
                                price,
                            );
                            totals[(SummaryLayout::PRICE_SQUARES + good) as usize] = add_checked(
                                control,
                                totals[(SummaryLayout::PRICE_SQUARES + good) as usize],
                                times32(price.extract(LOW_WORD), price.extract(LOW_WORD)),
                            );
                            totals[SummaryLayout::MINIMUM_PRICE as usize] =
                                minimum(totals[SummaryLayout::MINIMUM_PRICE as usize], price);
                            totals[SummaryLayout::MAXIMUM_PRICE as usize] =
                                maximum(totals[SummaryLayout::MAXIMUM_PRICE as usize], price);
                        }
                    }
                }
            }
            match k.bits() {
                ActorKind::HOUSEHOLD_BITS => {
                    let working = economy.ledger.number(Field::Worker, actor) != NO_ACTOR
                        && economy.ledger.number(Field::LivingMembers, actor) > 0;
                    let subsistence =
                        comptime!(config.economy == crate::domain::Economy::Industrial)
                            && economy.ledger.number(Field::LandPlots, actor) > 0;
                    let category = select(working, select(subsistence, 1u32, 0u32), 2u32);
                    let metric = (SummaryLayout::WORKFORCE + category) as usize;
                    if comptime!(config.cohorts == Feature::Enabled) {
                        let living = economy.ledger.number(Field::LivingMembers, actor);
                        let workers =
                            min(living, economy.ledger.number(Field::EligibleWorkers, actor));
                        let inactive = comptime!(SummaryLayout::INACTIVE as usize);
                        totals[inactive] =
                            add_checked(control, totals[inactive], ticks(living - workers, 0));
                        totals[metric] = add_checked(control, totals[metric], ticks(workers, 0));
                    } else {
                        totals[metric] = add_checked(control, totals[metric], ticks(1, 0));
                    }
                    totals[SummaryLayout::RURAL_EMPLOYED as usize] = add_checked(
                        control,
                        totals[SummaryLayout::RURAL_EMPLOYED as usize],
                        select(
                            subsistence,
                            economy.ledger.value(Field::Employed, actor),
                            zero(),
                        ),
                    );
                    totals[SummaryLayout::HOUSEHOLD_CASH as usize] = add_checked(
                        control,
                        totals[SummaryLayout::HOUSEHOLD_CASH as usize],
                        economy.ledger.value(Field::Cash, actor),
                    );
                }
                _ => {
                    let good = k.output() - 1;
                    let status = economy.ledger.number(Field::EnterpriseStatus, actor);
                    if status < 4 {
                        let category = select(
                            k.output() == comptime!(Good::Housing as u32),
                            2u32,
                            economy.ledger.number(Field::EnterpriseModel, actor),
                        );
                        totals[(SummaryLayout::ACTIVE_BY_GOOD + good) as usize] = add_checked(
                            control,
                            totals[(SummaryLayout::ACTIVE_BY_GOOD + good) as usize],
                            ticks(1, 0),
                        );
                        totals[(SummaryLayout::BUSINESS_TYPES + category) as usize] = add_checked(
                            control,
                            totals[(SummaryLayout::BUSINESS_TYPES + category) as usize],
                            ticks(1, 0),
                        );
                        totals[(SummaryLayout::BUSINESS_STATES + status) as usize] = add_checked(
                            control,
                            totals[(SummaryLayout::BUSINESS_STATES + status) as usize],
                            ticks(1, 0),
                        );
                    }
                    totals[(SummaryLayout::FIRM_CASH + good) as usize] = add_checked(
                        control,
                        totals[(SummaryLayout::FIRM_CASH + good) as usize],
                        economy.ledger.value(Field::Cash, actor),
                    );
                    totals[(SummaryLayout::BUSINESS_OPENING + good) as usize] = add_checked(
                        control,
                        totals[(SummaryLayout::BUSINESS_OPENING + good) as usize],
                        economy.ledger.value(Field::BusinessOpeningCash, actor),
                    );
                    totals[(SummaryLayout::BUSINESS_CLOSING + good) as usize] = add_checked(
                        control,
                        totals[(SummaryLayout::BUSINESS_CLOSING + good) as usize],
                        economy.ledger.value(Field::BusinessClosingCash, actor),
                    );
                    totals[(SummaryLayout::BUSINESS_REVENUE + good) as usize] = add_checked(
                        control,
                        totals[(SummaryLayout::BUSINESS_REVENUE + good) as usize],
                        economy.ledger.value(Field::BusinessRevenue, actor),
                    );
                }
            }
            let mut transit = ticks(0u32, 0u32);
            for good in 0u32..Good::COUNT as u32 {
                let held = economy.ledger.value_at(TRANSIT_FIELDS as u32 + good, actor);
                totals[(SummaryLayout::TRANSIT + good) as usize] = add_checked(
                    control,
                    totals[(SummaryLayout::TRANSIT + good) as usize],
                    held,
                );
                let household_output = economy
                    .ledger
                    .value_at(PRODUCTION_FIELDS as u32 + good, actor);
                totals[(SummaryLayout::HOUSEHOLD_PRODUCTION + good) as usize] = add_checked(
                    control,
                    totals[(SummaryLayout::HOUSEHOLD_PRODUCTION + good) as usize],
                    household_output,
                );
                totals[(SummaryLayout::PRODUCED + good) as usize] = add_checked(
                    control,
                    totals[(SummaryLayout::PRODUCED + good) as usize],
                    household_output,
                );
                totals[(SummaryLayout::USED + good) as usize] = add_checked(
                    control,
                    totals[(SummaryLayout::USED + good) as usize],
                    economy.ledger.value_at(USED_FIELDS as u32 + good, actor),
                );
                totals[(SummaryLayout::CONSUMPTION + good) as usize] = add_checked(
                    control,
                    totals[(SummaryLayout::CONSUMPTION + good) as usize],
                    economy
                        .ledger
                        .value_at(CONSUMPTION_FIELDS as u32 + good, actor),
                );
                let mut in_ring = zero();
                for slot in 0u32..config.ring {
                    in_ring = add_checked(
                        control,
                        in_ring,
                        economy.ledger.value_at(
                            BASE_FIELDS as u32 + (Good::COUNT as u32 + good) * config.ring + slot,
                            actor,
                        ),
                    );
                    in_ring = add_checked(
                        control,
                        in_ring,
                        economy
                            .ledger
                            .value_at(BASE_FIELDS as u32 + good * config.ring + slot, actor),
                    );
                }
                if !equal(in_ring, held) {
                    KernelFaults::new(comptime!(FaultFlags::TRANSIT_MISMATCH)).record(control);
                }
                transit = add_checked(control, transit, held);
            }
            if !equal(transit, economy.ledger.value(Field::Transit, actor)) {
                KernelFaults::new(comptime!(FaultFlags::TRANSIT_MISMATCH)).record(control);
            }
        }
        for metric in 0u32..SummaryLayout::METRICS as u32 {
            let base = (id * SummaryLayout::METRICS as u32 + metric) * TICK_WORDS as u32;
            bundles[(base) as usize] = totals[(metric) as usize].extract(LOW_WORD);
            bundles[(base + 1u32) as usize] = totals[(metric) as usize].extract(HIGH_WORD);
        }
    }
}

#[cube]
pub(super) fn inspect(
    ledger: &Ledger<'_>,
    proposals: &mut [u32],
    parameters: &EconomyParameters,
    id: u32,
) {
    let config = ledger.config;
    if id < config.fields {
        let item = ledger.value_at(id, parameters.offset);
        proposals[(id * TICK_WORDS as u32) as usize] = item.extract(LOW_WORD);
        proposals[(id * TICK_WORDS as u32 + 1u32) as usize] = item.extract(HIGH_WORD);
    }
}

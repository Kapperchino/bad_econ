use super::*;

#[cube]
pub(super) fn local(
    economy: &mut EconomyView<'_>,
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
    id: u32,
) {
    if id < economy.ledger.config.actors {
        match comptime!(parameters.phase) {
            Phase::Opening => open_day(&mut economy.ledger, control, id, parameters.day),
            Phase::Proposal => plan_production(economy, control, id, parameters.day),
            Phase::WorkPayment => finish_work(economy, control, id, parameters.day),
            Phase::PhysicalClose => close_production(economy, control, id, parameters.day),
            Phase::AccountsDecisions => update_accounts(economy, control, id, parameters.day),
            _ => {}
        }
    }
}

#[cube]
fn open_day(ledger: &mut Ledger<'_>, control: &[Atomic<u32>], id: u32, day: u32) {
    let config = ledger.config;
    for good in 1u32..Good::COUNT as u32 + 1 {
        let slot = BASE_FIELDS as u32 + (good - 1) * config.ring + day % config.ring;
        let delivery = ledger.value_at(slot, id);
        ledger.increase_at(control, good, id, delivery);
        let transit_field = TRANSIT_FIELDS as u32 + good - 1;
        let available = ledger.value_at(transit_field, id);
        if lt(available, delivery) {
            KernelFaults::new(comptime!(FaultFlags::TRANSIT_UNDERFLOW)).record(control);
        }
        ledger.set_at(transit_field, id, minus(available, delivery));
        let total = ledger.value(Field::Transit, id);
        ledger.set(Field::Transit, id, minus(total, delivery));
        ledger.set_number_at(slot, id, 0u32);
    }
    ledger.set_number(Field::Employed, id, 0u32);
    ledger.set_number(Field::Employer, id, 0u32);
    ledger.set_number(Field::Sales, id, 0u32);
    ledger.set_number(Field::GrainSales, id, 0u32);
    ledger.set_number(Field::Unfilled, id, 0u32);
    ledger.set_number(Field::Shortage, id, 0u32);
    ledger.set_number(Field::Purchased, id, 0u32);
    ledger.set_number(Field::Hours, id, 0u32);
    ledger.set_number(Field::CartageIncome, id, 0);
    ledger.set_number(Field::HomeMinutes, id, 0u32);
    ledger.set_number(Field::ResearchMinutes, id, 0u32);
}

#[cube]
fn plan_production(economy: &mut EconomyView<'_>, control: &[Atomic<u32>], id: u32, day: u32) {
    let config = economy.ledger.config;
    let kind = economy.actors.kind(id);
    if !kind.is(comptime!(ActorKind::Household)) {
        let cash = economy.ledger.value(Field::Cash, id);
        let rent = economy.ledger.value(Field::RentalIncome, id);
        economy.ledger.set(Field::BusinessOpeningCash, id, cash);
        economy.ledger.set(Field::BusinessOpeningRent, id, rent);
    }
    if kind.bits() == comptime!(Good::Railway as u32) {
        railway::offer_crew_jobs(economy, control, id);
    }
    if !kind.is(comptime!(ActorKind::Household))
        && kind.bits() != comptime!(Good::Housing as u32)
        && kind.bits() != comptime!(Good::Railway as u32)
    {
        if comptime!(config.home_mode == Feature::Enabled) {
            enterprises::available_method(economy, control, id);
        }
        let mut demand = economy.ledger.value(Field::Demand, id);
        if kind.bits() <= comptime!(Good::Bread as u32) {
            demand = farming::production_target(control, demand);
        }
        let mut desired = proposed_starts(
            KernelStarts {
                coverage: u32::cast_from(config.coverage),
                demand,
                finished: economy.ledger.value_at(kind.output(), id),
                pipeline: economy.ledger.value(Field::Pipeline, id),
            },
            control,
        );
        if farming::farm(&economy.ledger, id, kind.output()) {
            desired = farming::starts(&economy.ledger, control, id, kind.output(), day);
        }
        let quotient = divide(
            add_checked(
                control,
                desired,
                ticks(economy.ledger.number(Field::OutputTicks, id) - 1u32, 0u32),
            ),
            economy.ledger.number(Field::OutputTicks, id),
        );
        let mut planned = minimum(
            maximum(quotient, trial_packet(&economy.ledger, id, kind.output())),
            ticks(economy.actors.geographic_capacity(id), 0u32),
        );
        for input in 0u32..2 {
            let good = economy
                .ledger
                .number_at(comptime!(Field::FirstInput as u32) + input, id);
            if good > 0 {
                planned = minimum(
                    planned,
                    divide(
                        economy.ledger.value_at(good, id),
                        economy.ledger.number(Field::InputTicks, id),
                    ),
                );
            }
        }
        let budget = employment::ShiftBudget::new(&economy.ledger, control, id);
        let funded =
            farming::funded_shifts(&economy.ledger, control, id, kind.output(), day, &budget);
        planned = minimum(planned, funded);
        let distributor = economy.ledger.number(Field::EnterpriseModel, id) == 1;
        if distributor {
            let stock = economy.ledger.value_at(kind.output(), id);
            let demand = maximum(economy.ledger.value(Field::Demand, id), ticks(1, 0));
            let units = max(economy.ledger.number(Field::OutputTicks, id), 1);
            let shifts = divide(
                add_checked(control, minimum(stock, demand), ticks(units - 1, 0)),
                units,
            );
            let capacity = ticks(economy.actors.capacity(id), 0);
            planned = minimum(minimum(shifts, funded), capacity);
        }
        let status = economy.ledger.number(Field::EnterpriseStatus, id);
        planned = select(
            status == comptime!(crate::domain::enterprise::EnterpriseState::Operating as u32),
            planned,
            zero(),
        );
        economy.ledger.set(Field::Slots, id, planned);
        economy.ledger.set(Field::Plan, id, planned);
    }
}

#[cube]
fn finish_work(economy: &mut EconomyView<'_>, control: &[Atomic<u32>], id: u32, day: u32) {
    let config = economy.ledger.config;
    let kind = economy.actors.kind(id);
    economy.ledger.set_number(Field::CartageMinutes, id, 0);
    if kind.bits() == comptime!(Good::Railway as u32) {
        let crew = economy.ledger.number(Field::Plan, id) - economy.ledger.number(Field::Slots, id);
        let shift = economy.ledger.number(Field::LaborMinutes, id);
        economy
            .ledger
            .set_number(Field::CartageCapacity, id, crew * shift);
        economy.ledger.set_number(Field::Pipeline, id, 0);
    }
    if kind.is(comptime!(ActorKind::Household)) {
        employment::finish_day(economy, control, id);
    }
    if economy.ledger.number(Field::EnterpriseModel, id) == 1 {
        let services = economy.ledger.value(Field::Pipeline, id);
        let services = property::scaled_output(&mut economy.ledger, control, id, services);
        economy.ledger.set(Field::RetailOpen, id, services);
        economy.ledger.set_number(Field::Pipeline, id, 0);
    }
    if kind.is(comptime!(ActorKind::Household))
        && comptime!(config.home_mode == Feature::Enabled)
        && economy.ledger.number(Field::LivingMembers, id) > 0
    {
        enterprises::expand(economy, control, id);
        if comptime!(config.property_mode == Feature::Disabled) {
            enterprises::construct_home(&mut economy.ledger, control, id);
        }
        let rural = comptime!(config.economy == crate::domain::Economy::Agrarian)
            || economy.ledger.number(Field::LandPlots, id) > 0;
        let ration = times32(
            economy.ledger.number(Field::NutritionNeed, id),
            economy.ledger.number(Field::FoodPrice, id),
        );
        let unfunded = lt(economy.ledger.value(Field::Cash, id), ration);
        household_production::open_grain_quote(&mut economy.ledger, id);
        if rural || unfunded {
            household_production::produce(&mut economy.ledger, control, id, day);
        }
        household_production::reserve_grain(&mut economy.ledger, control, id, day);
    }
}

#[allow(
    clippy::manual_is_multiple_of,
    reason = "CubeCL supports integer remainder"
)]
#[cube]
fn close_production(economy: &mut EconomyView<'_>, control: &[Atomic<u32>], id: u32, day: u32) {
    let config = economy.ledger.config;
    let kind = economy.actors.kind(id);
    match kind.bits() {
        ActorKind::HOUSEHOLD_BITS => {
            let income = observed_blend(
                economy.ledger.value(Field::IncomeForecast, id),
                KernelObservation {
                    kind: comptime!(ObservationKind::Observed as u32),
                    value: multiply(
                        control,
                        add_checked(
                            control,
                            positive(
                                economy.ledger.value(Field::GrossIncome, id),
                                economy.ledger.value(Field::OpeningGrossIncome, id),
                            ),
                            economy.ledger.value(Field::CartageIncome, id),
                        ),
                        POLICY_SCALE,
                    ),
                },
                POLICY_SCALE / 30,
                control,
            );
            economy.ledger.set(Field::IncomeForecast, id, income);
            if config.heating_per_person > 0 {
                consumption::heat(&mut economy.ledger, control, id);
            }
            let bread = economy.ledger.value(Field::Bread, id);
            let minimum_ration = economy.ledger.value(Field::NutritionNeed, id);
            let reserve = food::consumption_reserve(&economy.ledger, control, id, day);
            let available = maximum(minimum_ration, positive(bread, reserve));
            let consumed = minimum(
                minimum(bread, available),
                economy.ledger.value(Field::FoodNeed, id),
            );
            let remaining = minus(economy.ledger.value(Field::Bread, id), consumed);
            economy.ledger.set(Field::Bread, id, remaining);
            economy
                .ledger
                .increase(control, Field::Consumed, id, consumed);
            economy.ledger.increase_at(
                control,
                CONSUMPTION_FIELDS as u32 + comptime!(Good::Bread as u32) - 1,
                id,
                consumed,
            );
            if comptime!(config.economy == crate::domain::Economy::Industrial) {
                for offset in 0..Good::COUNT {
                    let interval = config.consumption_intervals[offset];
                    if interval > 1
                        && ((day + id) % max(interval, 1) == 0
                            || consumption::scaled_luxury(config, offset as u32 + 1))
                        && economy.ledger.number(Field::LivingMembers, id) > 0
                    {
                        let good = offset as u32 + 1;
                        let stock = economy.ledger.value_at(good, id);
                        let members = consumption::consumers(&economy.ledger, id);
                        let used = consumption::durable_use(config, members, stock, good, day + id);
                        economy.ledger.set_at(good, id, minus(stock, used));
                        economy.ledger.increase_at(
                            control,
                            CONSUMPTION_FIELDS as u32 + offset as u32,
                            id,
                            used,
                        );
                    }
                }
            }
            let shortage = economy.ledger.number(Field::FoodNeed, id) - consumed.extract(LOW_WORD);
            economy.ledger.set_number(Field::Shortage, id, shortage);
            let previous_fatigue = economy.ledger.number(Field::Fatigue, id);
            let mut fatigue = min(previous_fatigue + config.work_fatigue, POLICY_SCALE);
            let work = economy.ledger.number(Field::Hours, id)
                + economy.ledger.number(Field::CartageMinutes, id);
            if work == 0 {
                fatigue = max(previous_fatigue, config.rest_recovery) - config.rest_recovery;
            }
            if comptime!(config.household_policy == Feature::Enabled) {
                let exertion = work + economy.ledger.number(Field::ResearchMinutes, id);
                fatigue = next_fatigue(KernelFatigue {
                    fatigue: previous_fatigue,
                    response: config.fatigue_response,
                    exertion_minutes: exertion,
                    rest_minutes: config.rest_minutes,
                });
            }
            economy.ledger.set_number(Field::Fatigue, id, fatigue);
        }
        _ => {
            let completed = economy.ledger.value(Field::Pipeline, id);
            let output = farming::finish(
                &mut economy.ledger,
                control,
                id,
                day,
                kind.output(),
                completed,
            );
            economy
                .ledger
                .increase(control, Field::UnscaledProduced, id, output.unscaled);
            economy
                .ledger
                .increase_at(control, kind.output(), id, output.finished);
            economy
                .ledger
                .increase(control, Field::Produced, id, output.finished);
            economy.ledger.set_number(Field::Pipeline, id, 0u32);
        }
    }
}

#[cube]
fn update_accounts(economy: &mut EconomyView<'_>, control: &[Atomic<u32>], id: u32, day: u32) {
    let config = economy.ledger.config;
    let kind = economy.actors.kind(id);
    if !kind.is(comptime!(ActorKind::Household)) {
        let business = u32::cast_from(kind.bits() == comptime!(Good::Housing as u32))
            + 2 * u32::cast_from(kind.bits() == comptime!(Good::Railway as u32));
        let revenue = match business {
            2 => zero(),
            1 => minus(
                economy.ledger.value(Field::RentalIncome, id),
                economy.ledger.value(Field::BusinessOpeningRent, id),
            ),
            _ => multiply(
                control,
                economy.ledger.value(Field::Sales, id),
                economy.ledger.number(Field::Price, id),
            ),
        };
        let cash = economy.ledger.value(Field::Cash, id);
        economy.ledger.set(Field::BusinessClosingCash, id, cash);
        let rail_revenue = economy.ledger.value(Field::RailRevenue, id);
        let cartage_revenue = economy.ledger.value(Field::CartageIncome, id);
        economy.ledger.set(
            Field::BusinessRevenue,
            id,
            add_checked(
                control,
                add_checked(control, revenue, rail_revenue),
                cartage_revenue,
            ),
        );
    }
    if kind.is(comptime!(ActorKind::Household)) && comptime!(config.home_mode == Feature::Enabled) {
        enterprises::household_quote(&mut economy.ledger, control, id);
        household_production::close_grain_sales(&mut economy.ledger, control, id, day);
    }
    if kind.bits() == comptime!(Good::Railway as u32) {
        enterprises::operating_plan(economy, control, id);
        employment::adjust_wage(&mut economy.ledger, id);
        if comptime!(config.home_mode == Feature::Enabled) {
            enterprises::lifecycle(economy, control, id, day);
        }
    }
    if !kind.is(comptime!(ActorKind::Household))
        && kind.bits() != comptime!(Good::Housing as u32)
        && kind.bits() != comptime!(Good::Railway as u32)
    {
        if comptime!(config.home_mode == Feature::Enabled) {
            let packets =
                economy.ledger.number(Field::Plan, id) - economy.ledger.number(Field::Slots, id);
            enterprises::maintain(&mut economy.ledger, control, id, packets);
        }
        if comptime!(config.home_mode == Feature::Enabled)
            || economy.ledger.number(Field::EnterpriseModel, id)
                == comptime!(crate::domain::enterprise::EnterpriseModel::Distributor as u32)
        {
            enterprises::lifecycle(economy, control, id, day);
        }
        let old = economy.ledger.value(Field::Demand, id);
        let signal = add_checked(
            control,
            economy.ledger.value(Field::Sales, id),
            economy.ledger.value(Field::Unfilled, id),
        );
        let demand = observed_blend(
            old,
            KernelObservation {
                kind: comptime!(ObservationKind::Observed as u32),
                value: signal,
            },
            config.forecast_weight,
            control,
        );
        let target = pricing::inventory_target(&*economy, control, id, kind.output(), demand, day);
        let excess = positive(economy.ledger.value_at(kind.output(), id), target);
        let unfilled = economy.ledger.value(Field::Unfilled, id);
        let positive_sign = !lt(unfilled, excess);
        let magnitude = select(
            positive_sign,
            minus(unfilled, excess),
            minus(excess, unfilled),
        );
        let denominator = maximum(demand, ticks(1, 0));
        let scaled = fraction_index(magnitude, denominator) * config.price_response / POLICY_SCALE;
        let table_index = select(
            positive_sign,
            PRICE_INDEX_LIMIT + scaled,
            PRICE_INDEX_LIMIT - scaled,
        );
        let response = pricing::response(
            &economy.ledger,
            control,
            id,
            kind.output(),
            excess,
            config.price_table[table_index as usize],
        );
        pricing::adjust(&mut economy.ledger, control, id, response);
        economy.ledger.set(Field::Demand, id, demand);
        enterprises::operating_plan(economy, control, id);

        employment::adjust_wage(&mut economy.ledger, id);
    }
}

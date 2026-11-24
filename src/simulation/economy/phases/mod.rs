pub(in crate::simulation) mod opening;

use crate::simulation::economy::*;

const CONSTRUCTION_KIND: u32 = Good::Construction as u32;

#[derive(CubeType)]
pub(super) struct ActorDay<'a> {
    actor: u32,
    day: u32,
    control: &'a [Atomic<u32>],
}

#[cube]
impl<'a> ActorDay<'a> {
    pub(super) fn new(actor: u32, day: u32, control: &'a [Atomic<u32>]) -> Self {
        ActorDay::<'a> {
            actor,
            day,
            control,
        }
    }

    pub(super) fn advance(&self, economy: &mut EconomyView<'_>, #[comptime] phase: Phase) {
        if self.actor < economy.ledger.config.actors {
            match comptime!(phase) {
                Phase::Opening => self.open_day(&mut economy.ledger),
                Phase::Proposal => self.plan_production(economy),
                Phase::WorkPayment => self.finish_work(economy),
                Phase::PhysicalClose => self.close_production(economy),
                Phase::AccountsDecisions => self.update_accounts(economy),
                _ => {}
            }
        }
    }

    fn open_day(&self, ledger: &mut Ledger<'_>) {
        let id = self.actor;
        let control = self.control;
        let day = self.day;
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

    fn plan_production(&self, economy: &mut EconomyView<'_>) {
        let id = self.actor;
        let control = self.control;
        let day = self.day;
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
                enterprises::FirmPolicy::new(id, control).select_method(economy);
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
            if kind.output() == comptime!(Good::Construction as u32) {
                desired = demand;
            }
            if farming::FarmSite::new(id, kind.output()).is_farm(&economy.ledger) {
                desired = farming::FarmProduction::new(
                    &mut economy.ledger,
                    control,
                    farming::FarmSite::new(id, kind.output()),
                    day,
                )
                .starts();
            }
            let quotient = divide(
                plus(
                    desired,
                    ticks(economy.ledger.number(Field::OutputTicks, id) - 1u32, 0u32),
                )
                .record(control),
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
            let funded = farming::FarmProduction::new(
                &mut economy.ledger,
                control,
                farming::FarmSite::new(id, kind.output()),
                day,
            )
            .funded_shifts(&budget);
            planned = minimum(planned, funded);
            let distributor = economy.ledger.number(Field::EnterpriseModel, id) == 1;
            if distributor {
                let stock = economy.ledger.value_at(kind.output(), id);
                let mut demand = maximum(economy.ledger.value(Field::Demand, id), ticks(1, 0));
                if kind.output() == comptime!(Good::Grain as u32) {
                    let days = max(
                        min(
                            economy.ledger.number(Field::StockCoverage, id),
                            config.coverage,
                        ),
                        1,
                    );
                    let daily_stock = CheckedTicks::ceiling_ratio(stock, 1, days).record(control);
                    demand = maximum(demand, daily_stock);
                }
                let units = max(economy.ledger.number(Field::OutputTicks, id), 1);
                let sales = market::distribution::handling_demand(
                    &*economy,
                    control,
                    id,
                    minimum(stock, demand),
                );
                let shifts = divide(plus(sales, ticks(units - 1, 0)).record(control), units);
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

    fn finish_work(&self, economy: &mut EconomyView<'_>) {
        let id = self.actor;
        let control = self.control;
        let day = self.day;
        let config = economy.ledger.config;
        let kind = economy.actors.kind(id);
        economy.ledger.set_number(Field::CartageMinutes, id, 0);
        if kind.bits() == comptime!(Good::Railway as u32) {
            let crew =
                economy.ledger.number(Field::Plan, id) - economy.ledger.number(Field::Slots, id);
            let shift = economy.ledger.number(Field::LaborMinutes, id);
            economy
                .ledger
                .set_number(Field::CartageCapacity, id, crew * shift);
            economy.ledger.set_number(Field::Pipeline, id, 0);
        }
        if kind.is(comptime!(ActorKind::Household)) {
            employment::finish_day(economy, control, id);
        }
        if kind.bits() == comptime!(Good::Grain as u32) {
            farming::FarmSite::new(id, kind.output()).offer_wheat(
                &mut economy.ledger,
                control,
                day,
            );
        }
        if economy.ledger.number(Field::EnterpriseModel, id) == 1 {
            market::distribution::open(economy, control, id);
            let services = economy.ledger.value(Field::Pipeline, id);
            let services = property::PropertySite::new(id).scale_output(
                &mut economy.ledger,
                control,
                services,
            );
            economy.ledger.set(Field::RetailOpen, id, services);
            economy.ledger.set_number(Field::Pipeline, id, 0);
        }
        if kind.output() == comptime!(Good::Construction as u32) {
            let completed = economy.ledger.value(Field::Pipeline, id);
            let available = property::PropertySite::new(id).scale_output(
                &mut economy.ledger,
                control,
                completed,
            );
            economy.ledger.set(Field::Pipeline, id, available);
            economy
                .ledger
                .increase(control, Field::Produced, id, available);
            economy
                .ledger
                .increase(control, Field::UnscaledProduced, id, completed);
        }
        if kind.is(comptime!(ActorKind::Household))
            && comptime!(config.home_mode == Feature::Enabled)
            && economy.ledger.number(Field::LivingMembers, id) > 0
        {
            enterprises::expand(economy, control, id);
            let has_kitchen = food::HomeKitchen::available(&economy.ledger, id);
            let mut farm =
                household_production::HouseholdFarm::new(&mut economy.ledger, control, id, day);
            farm.open_grain_quote();
            if has_kitchen {
                farm.produce();
                farm.reserve_grain();
            } else {
                economy.ledger.set(Field::GrainReserve, id, zero());
            }
        }
    }

    #[allow(
        clippy::manual_is_multiple_of,
        reason = "CubeCL supports integer remainder"
    )]
    fn close_production(&self, economy: &mut EconomyView<'_>) {
        let id = self.actor;
        let control = self.control;
        let day = self.day;
        let config = economy.ledger.config;
        let kind = economy.actors.kind(id);
        match kind.bits() {
            ActorKind::HOUSEHOLD_BITS => {
                let prior_income = economy.ledger.value(Field::IncomeForecast, id);
                let income = KernelObservation {
                    kind: comptime!(ObservationKind::Observed as u32),
                    value: times(
                        plus(
                            positive(
                                economy.ledger.value(Field::GrossIncome, id),
                                economy.ledger.value(Field::OpeningGrossIncome, id),
                            ),
                            economy.ledger.value(Field::CartageIncome, id),
                        )
                        .record(control),
                        POLICY_SCALE,
                    )
                    .record(control),
                }
                .blend(prior_income, POLICY_SCALE / 30, control);
                economy.ledger.set(Field::IncomeForecast, id, income);
                if config.heating_per_person > 0 {
                    consumption::HeatingDemand::new(&economy.ledger, id)
                        .serve(&mut economy.ledger, control);
                }
                let bread = economy.ledger.value(Field::Bread, id);
                let minimum_ration = economy.ledger.value(Field::NutritionNeed, id);
                let reserve =
                    food::HouseholdRation::new(&economy.ledger, id).consumption_reserve(config);
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
                        let interval = config.market_table[offset];
                        if interval > 1
                            && ((day + id) % max(interval, 1) == 0
                                || consumption::scaled_luxury(config, offset as u32 + 1))
                            && economy.ledger.number(Field::LivingMembers, id) > 0
                        {
                            let good = offset as u32 + 1;
                            let stock = economy.ledger.value_at(good, id);
                            let members = consumption::consumers(&economy.ledger, id);
                            let used =
                                consumption::durable_use(config, members, stock, good, day + id);
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
                let shortage =
                    economy.ledger.number(Field::FoodNeed, id) - consumed.extract(LOW_WORD);
                economy.ledger.set_number(Field::Shortage, id, shortage);
                let previous_fatigue = economy.ledger.number(Field::Fatigue, id);
                let mut fatigue = min(previous_fatigue + config.work_fatigue, POLICY_SCALE);
                let work = economy.ledger.number(Field::Hours, id)
                    + economy.ledger.number(Field::CartageMinutes, id);
                if work == 0 {
                    fatigue = max(previous_fatigue, config.rest_recovery) - config.rest_recovery;
                }
                let workers = match comptime!(config.cohorts) {
                    Feature::Enabled => max(economy.ledger.number(Field::EligibleWorkers, id), 1),
                    Feature::Disabled => 1u32,
                };
                let home = economy.ledger.number(Field::HomeMinutes, id);
                let research = economy.ledger.number(Field::ResearchMinutes, id);
                let exertion = match comptime!(config.cohorts) {
                    Feature::Enabled => (work + home + research) / workers,
                    Feature::Disabled => work + research,
                };
                if comptime!(config.household_policy == Feature::Enabled) {
                    fatigue = KernelFatigue {
                        fatigue: previous_fatigue,
                        response: config.fatigue_response,
                        exertion_minutes: exertion,
                        rest_minutes: config.rest_minutes,
                    }
                    .next_fatigue();
                } else if comptime!(config.cohorts == Feature::Enabled) {
                    let tired = previous_fatigue + exertion * config.work_fatigue / 480;
                    fatigue = min(
                        max(tired, config.rest_recovery) - config.rest_recovery,
                        POLICY_SCALE,
                    );
                }
                economy.ledger.set_number(Field::Fatigue, id, fatigue);
            }
            self::CONSTRUCTION_KIND => {
                self.close_construction(&mut economy.ledger);
            }
            _ => {
                let completed = economy.ledger.value(Field::Pipeline, id);
                let mut farm = farming::FarmProduction::new(
                    &mut economy.ledger,
                    control,
                    farming::FarmSite::new(id, kind.output()),
                    day,
                );
                let output = farm.finish(completed);
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

    fn close_construction(&self, ledger: &mut Ledger<'_>) {
        let id = self.actor;
        let control = self.control;
        let unused = ledger.value(Field::Pipeline, id);
        if !equal(unused, zero()) {
            let shifts = ledger.number(Field::Plan, id) - ledger.number(Field::Slots, id);
            let reserved = shifts * ledger.number(Field::InputTicks, id);
            let available = plus(unused, ledger.value(Field::Sales, id)).record(control);
            let refund = ratio(unused, reserved, available.extract(LOW_WORD)).record(control);
            for input in 0u32..2u32 {
                let good = ledger.number_at(comptime!(Field::FirstInput as u32) + input, id);
                if good > 0 {
                    ledger.increase_at(control, good, id, refund);
                    let used = USED_FIELDS as u32 + good - 1;
                    ledger.set_at(used, id, minus(ledger.value_at(used, id), refund));
                    ledger.set(
                        Field::Used,
                        id,
                        minus(ledger.value(Field::Used, id), refund),
                    );
                }
            }
            ledger.increase_at(
                control,
                CONSUMPTION_FIELDS as u32 + comptime!(Good::Construction as u32) - 1,
                id,
                unused,
            );
        }
        ledger.set(Field::Pipeline, id, zero());
    }

    fn update_accounts(&self, economy: &mut EconomyView<'_>) {
        let id = self.actor;
        let control = self.control;
        let day = self.day;
        let config = economy.ledger.config;
        let kind = economy.actors.kind(id);
        if !kind.is(comptime!(ActorKind::Household)) {
            let business = u32::cast_from(kind.bits() == comptime!(Good::Housing as u32))
                + 2 * u32::cast_from(kind.bits() == comptime!(Good::Railway as u32));
            let revenue = match select(pricing::distributor(&economy.ledger, id), 3, business) {
                3 => economy.ledger.value(Field::BusinessRevenue, id),
                2 => zero(),
                1 => minus(
                    economy.ledger.value(Field::RentalIncome, id),
                    economy.ledger.value(Field::BusinessOpeningRent, id),
                ),
                _ => times(
                    economy.ledger.value(Field::Sales, id),
                    economy.ledger.number(Field::Price, id),
                )
                .record(control),
            };
            let cash = economy.ledger.value(Field::Cash, id);
            economy.ledger.set(Field::BusinessClosingCash, id, cash);
            let rail_revenue = economy.ledger.value(Field::RailRevenue, id);
            let cartage_revenue = economy.ledger.value(Field::CartageIncome, id);
            economy.ledger.set(
                Field::BusinessRevenue,
                id,
                plus(plus(revenue, rail_revenue).record(control), cartage_revenue).record(control),
            );
        }
        if kind.is(comptime!(ActorKind::Household))
            && comptime!(config.home_mode == Feature::Enabled)
        {
            enterprises::household_quote(&mut economy.ledger, control, id);
            household_production::HouseholdFarm::new(&mut economy.ledger, control, id, day)
                .close_grain_sales();
        }
        let firm = enterprises::FirmPolicy::new(id, control);
        if kind.bits() == comptime!(Good::Railway as u32) {
            firm.reserve_operating_costs(economy);
            employment::adjust_wage(&mut economy.ledger, id);
            if comptime!(config.home_mode == Feature::Enabled) {
                firm.advance_lifecycle(economy, day);
            }
        }
        if !kind.is(comptime!(ActorKind::Household))
            && kind.bits() != comptime!(Good::Housing as u32)
            && kind.bits() != comptime!(Good::Railway as u32)
        {
            if comptime!(config.home_mode == Feature::Enabled) {
                let packets = economy.ledger.number(Field::Plan, id)
                    - economy.ledger.number(Field::Slots, id);
                firm.maintain(&mut economy.ledger, packets);
            }
            if comptime!(config.home_mode == Feature::Enabled)
                || economy.ledger.number(Field::EnterpriseModel, id)
                    == comptime!(crate::domain::enterprise::EnterpriseModel::Distributor as u32)
            {
                firm.advance_lifecycle(economy, day);
            }
            let old = economy.ledger.value(Field::Demand, id);
            let mut signal = plus(
                economy.ledger.value(Field::Sales, id),
                economy.ledger.value(Field::Unfilled, id),
            )
            .record(control);
            if pricing::distributor(&economy.ledger, id) {
                market::distribution::close(economy, control, id);
                signal = economy.ledger.value_at(
                    pricing::retail_field(
                        config,
                        kind.output(),
                        comptime!(crate::domain::RetailAccount::Orders),
                    ),
                    id,
                );
            }
            let demand = KernelObservation {
                kind: comptime!(ObservationKind::Observed as u32),
                value: signal,
            }
            .blend(old, config.forecast_weight, control);
            let pricing = pricing::FirmPricing::new(id, kind.output(), control);
            let target = pricing.inventory_target(&*economy, demand, day);
            let stock = economy.ledger.value_at(kind.output(), id);
            let mut excess = positive(stock, target);
            let mut unfilled = economy.ledger.value(Field::Unfilled, id);
            if comptime!(
                config.economy == crate::domain::Economy::Industrial
                    && config.home_mode == Feature::Enabled
            ) && config.home.harvest_days < 365
                && kind.output() == comptime!(Good::Grain as u32)
                && farming::FarmSite::new(id, kind.output()).is_farm(&economy.ledger)
            {
                let days = farming::CropCalendar::new(config).on(day).days_to_harvest();
                let shortage = divide(positive(target, stock), days);
                excess = divide(excess, days);
                unfilled = maximum(unfilled, shortage);
            }
            let positive_sign = !lt(unfilled, excess);
            let magnitude = select(
                positive_sign,
                minus(unfilled, excess),
                minus(excess, unfilled),
            );
            let denominator = maximum(demand, ticks(1, 0));
            let scaled =
                fraction_index(magnitude, denominator) * config.price_response / POLICY_SCALE;
            let table_index = select(
                positive_sign,
                PRICE_INDEX_LIMIT + scaled,
                PRICE_INDEX_LIMIT - scaled,
            );
            pricing.reprice(
                &mut economy.ledger,
                excess,
                config.price_table[table_index as usize],
            );
            economy.ledger.set(Field::Demand, id, demand);
            firm.reserve_operating_costs(economy);

            employment::adjust_wage(&mut economy.ledger, id);
        }
    }
}

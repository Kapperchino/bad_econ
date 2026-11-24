use crate::simulation::economy::*;

#[cube]
pub(in crate::simulation::economy) fn reserve(
    economy: &EconomyView<'_>,
    actor: u32,
    good: u32,
) -> Ticks {
    let config = economy.ledger.config;
    let mut held = zero();
    if economy
        .actors
        .kind(actor)
        .is(comptime!(ActorKind::Household))
        && good == comptime!(Good::Bread as u32)
    {
        held = ticks(
            economy.ledger.number(Field::FoodNeed, actor) * (config.coverage + config.ring),
            0,
        );
    }
    if good == comptime!(Good::Grain as u32) {
        held = economy.ledger.value(Field::GrainReserve, actor);
    }
    if economy.ledger.number(Field::EnterpriseModel, actor) == 1 {
        let capacity = economy.ledger.value(Field::RetailOpen, actor);
        let sales = economy.ledger.value(Field::Sales, actor);
        let available = positive(capacity, sales);
        held = positive(economy.ledger.value_at(good, actor), available);
    }
    held
}

#[derive(CubeType)]
pub(in crate::simulation::economy) struct TradeFields {
    pub stock: u32,
    pub price: u32,
    pub sales: u32,
}

#[cube]
impl TradeFields {
    pub fn new(actors: &ActorTable<'_>, config: &EconomyConfig, seller: u32, good: u32) -> Self {
        let grain = good == comptime!(Good::Grain as u32)
            && actors.kind(seller).is(comptime!(ActorKind::Household));
        TradeFields {
            stock: trade_stock_field(good),
            price: select(
                grain,
                comptime!(Field::GrainQuote as u32),
                select(
                    good > 0
                        && !actors.kind(seller).is(comptime!(ActorKind::Household))
                        && actors.kind(seller).output() != good,
                    pricing::retail_field(
                        config,
                        good,
                        comptime!(crate::domain::RetailAccount::Quote),
                    ),
                    comptime!(Field::Price as u32),
                ),
            ),
            sales: select(
                grain,
                comptime!(Field::GrainSales as u32),
                comptime!(Field::Sales as u32),
            ),
        }
    }
}

#[cube]
pub(in crate::simulation::economy) fn trade_stock_field(good: u32) -> u32 {
    select(
        good == comptime!(Good::Construction as u32),
        comptime!(Field::Pipeline as u32),
        good,
    )
}

#[cube]
pub(in crate::simulation::economy) fn supplier(
    economy: &EconomyView<'_>,
    actor: u32,
    good: u32,
) -> bool {
    let config = economy.ledger.config;
    let k = economy.actors.kind(actor);
    let household_good = good == comptime!(Good::Bread as u32)
        || (comptime!(config.home_mode == Feature::Enabled)
            && good == comptime!(Good::Grain as u32)
            && economy.ledger.number(Field::LivingMembers, actor) > 0
            && economy.ledger.number(Field::GrainQuote, actor) > 0);
    let dealer = pricing::distributor(&economy.ledger, actor);
    if good == comptime!(Good::Construction as u32) {
        k.output() == good && !dealer
    } else {
        dealer || k.output() == good || (k.is(comptime!(ActorKind::Household)) && household_good)
    }
}

#[cube]
pub(in crate::simulation::economy) fn offers(
    economy: &EconomyView<'_>,
    actor: u32,
    good: u32,
) -> bool {
    supplier(economy, actor, good)
        && lt(
            reserve(economy, actor, good),
            economy.ledger.value_at(trade_stock_field(good), actor),
        )
}

#[cube]
pub(in crate::simulation::economy) fn counterparty(
    economy: &EconomyView<'_>,
    buyer: u32,
    seller: u32,
    good: u32,
) -> bool {
    let buyer_is_dealer = pricing::distributor(&economy.ledger, buyer);
    let seller_is_dealer = pricing::distributor(&economy.ledger, seller);
    let output = economy.actors.kind(seller).output();
    if good == comptime!(Good::Construction as u32) {
        output == good && !seller_is_dealer
    } else if buyer_is_dealer {
        !seller_is_dealer
    } else if good == comptime!(Good::Grain as u32) {
        seller_is_dealer
            || (economy.actors.kind(buyer).output() == comptime!(Good::Flour as u32)
                && farming::FarmSite::new(seller, output).sells_direct(&economy.ledger))
    } else {
        output > 0 || comptime!(economy.ledger.config.home_mode == Feature::Disabled)
    }
}

const SEED_RETURN: u32 = crate::domain::food::WheatHusbandry::GROSS_PARTS;
const SALES_WINDOW: u32 = 30;

#[cube]
fn seed_required(control: &[Atomic<u32>], work: Ticks, minutes: u32) -> Ticks {
    let batch = minutes * comptime!(SEED_RETURN);
    divide(plus(work, ticks(batch - 1, 0)).record(control), batch)
}

#[cube]
pub(in crate::simulation::economy) fn seed_reserve(
    ledger: &Ledger<'_>,
    control: &[Atomic<u32>],
    actor: u32,
) -> Ticks {
    let plots = ledger.number(Field::LandPlots, actor);
    let minutes = ledger.config.home.cultivation_minutes;
    let annual = times32(plots, ledger.config.home.annual_yield_per_plot * minutes);
    positive(
        seed_required(control, annual, minutes),
        seed_required(control, ledger.value(Field::CropWork, actor), minutes),
    )
}

#[derive(CubeType)]
pub(in crate::simulation::economy) struct HouseholdFarm<'a> {
    ledger: Ledger<'a>,
    control: &'a [Atomic<u32>],
    actor: u32,
    day: u32,
    date: farming::CropDay,
}

#[cube]
impl<'a> HouseholdFarm<'a> {
    pub fn new(
        ledger: &'a mut Ledger<'_>,
        control: &'a [Atomic<u32>],
        actor: u32,
        day: u32,
    ) -> Self {
        let date = farming::CropCalendar::new(ledger.config).on(day);
        HouseholdFarm::<'a> {
            ledger: Ledger::new(ledger.words, ledger.config),
            control,
            actor,
            day,
            date,
        }
    }

    fn output(&mut self, good: u32, amount: Ticks) {
        let actor = self.actor;
        let control = self.control;
        self.ledger.increase_at(control, good, actor, amount);
        self.ledger
            .increase_at(control, PRODUCTION_FIELDS as u32 + good - 1, actor, amount);
    }

    fn grain_to_fund_meals(&self, target: Ticks) -> Ticks {
        let actor = self.actor;
        let kitchen = food::HomeKitchen::new(&self.ledger, actor);
        let bread = self
            .ledger
            .stock_with_transit(comptime!(Good::Bread as u32), actor)
            .record(self.control);
        let missing = positive(target, bread);
        let flour = minimum(
            missing,
            self.ledger
                .stock_with_transit(comptime!(Good::Flour as u32), actor)
                .record(self.control),
        );
        let loaves = positive(missing, kitchen.bread_from_flour(flour));
        let quote = max(self.ledger.number(Field::GrainQuote, actor), 1);
        let cost = times(loaves, self.ledger.number(Field::FoodPrice, actor)).record(self.control);
        let available = positive(
            self.ledger.value(Field::Cash, actor),
            self.ledger.value(Field::CartagePayable, actor),
        );
        CheckedTicks::ceiling_ratio(positive(cost, available), 1, quote).record(self.control)
    }

    pub fn open_grain_quote(&mut self) {
        let actor = self.actor;
        let config = self.ledger.config;
        let quote = self.ledger.number(Field::GrainQuote, actor);
        if quote == 0 {
            self.ledger
                .set_number(Field::GrainQuote, actor, config.home.grain_price);
        }
    }

    pub fn reserve_grain(&mut self) {
        let actor = self.actor;
        let control = self.control;
        let seed = seed_reserve(&self.ledger, control, actor);
        let config = self.ledger.config;
        let reserve =
            if comptime!(config.food_income == crate::domain::food::FoodIncome::PurchasingPower) {
                let stock = self.ledger.value(Field::Grain, actor);
                let daily_lot = CheckedTicks::ceiling_ratio(
                    positive(stock, seed),
                    1,
                    self.date.days_to_harvest(),
                )
                .record(control);
                let meals = times32(
                    self.ledger.number(Field::NutritionNeed, actor),
                    config.coverage + config.ring,
                );
                let food_budget = self.grain_to_fund_meals(times(meals, 2).record(control));
                maximum(seed, positive(stock, maximum(daily_lot, food_budget)))
            } else {
                seed
            };
        self.ledger.set(Field::GrainReserve, actor, reserve);
    }

    pub fn close_grain_sales(&mut self) {
        let actor = self.actor;
        let control = self.control;
        let config = self.ledger.config;
        let sales = self.ledger.value(Field::GrainSales, actor);
        let previous = self.ledger.value(Field::ExpectedGrainSales, actor);
        let forecast = plus(
            ratio(
                previous,
                comptime!(SALES_WINDOW - 1),
                comptime!(SALES_WINDOW),
            )
            .record(control),
            ratio(sales, 365, comptime!(SALES_WINDOW)).record(control),
        )
        .record(control);
        self.ledger.set(Field::ExpectedGrainSales, actor, forecast);
        let quote = self.ledger.number(Field::GrainQuote, actor);
        if self.day % comptime!(SALES_WINDOW)
            == comptime!(SALES_WINDOW - 1) - actor % comptime!(SALES_WINDOW)
            && config.price_response > 0
            && quote > 0
        {
            let surplus = positive(
                self.ledger.value(Field::Grain, actor),
                seed_reserve(&self.ledger, control, actor),
            );
            let days = self.date.days_to_harvest();
            let expected = ratio(forecast, days, 365).record(control);
            let daily = max(
                self.ledger.number(Field::FoodNeed, actor),
                self.ledger.number(Field::NutritionNeed, actor),
            );
            let for_sale = minimum(surplus, self.grain_to_fund_meals(times32(daily, days)));
            let step = max(ceildiv(quote * config.price_response, 20 * POLICY_SCALE), 1);
            let production_cost = min(
                config.maximum_price,
                ceildiv(
                    config.starting_wage * config.home.cultivation_minutes * comptime!(SEED_RETURN),
                    comptime!(SEED_RETURN - 1),
                ),
            );
            let floor = min(quote, max(config.minimum_price, production_cost));
            let cheaper = max(quote, step + floor) - step;
            let dearer = min(quote + step, config.maximum_price);
            let next = select(
                lt(expected, for_sale),
                cheaper,
                select(lt(surplus, expected), dearer, quote),
            );
            self.ledger.set_number(Field::GrainQuote, actor, next);
        }
    }

    fn cultivate(&mut self, plots: u32, time: u32) -> u32 {
        let actor = self.actor;
        let control = self.control;
        let config = self.ledger.config;
        let minutes = config.home.cultivation_minutes;
        let standing = self.ledger.value(Field::CropWork, actor);
        let annual = times32(plots, config.home.annual_yield_per_plot);
        let grain = self.ledger.value(Field::Grain, actor);
        let paid_minutes =
            self.ledger.number(Field::Hours, actor) + self.ledger.number(Field::HomeMinutes, actor);
        let earnings = positive(
            self.ledger.value(Field::GrossIncome, actor),
            self.ledger.value(Field::OpeningGrossIncome, actor),
        );
        let wage = divide(earnings, max(paid_minutes, 1));
        let price = self.ledger.number(Field::GrainQuote, actor);
        let profitable = paid_minutes == 0
            || !lt(
                times32(price, comptime!(SEED_RETURN - 1)),
                times(wage, minutes * comptime!(SEED_RETURN)).record(control),
            );
        let sales = self.ledger.value(Field::ExpectedGrainSales, actor);
        let horizon = self.date.days_to_harvest() + 365;
        let daily = max(
            self.ledger.number(Field::FoodNeed, actor),
            self.ledger.number(Field::NutritionNeed, actor),
        );
        let demand = maximum(
            self.grain_to_fund_meals(times32(daily, horizon)),
            select(
                profitable,
                ratio(maximum(sales, ticks(plots, 0)), horizon, 365).record(control),
                zero(),
            ),
        );
        let seed = seed_required(control, times(annual, minutes).record(control), minutes);
        let sown_seed = seed_required(control, standing, minutes);
        let needed = positive(
            plus(demand, seed).record(control),
            plus(grain, sown_seed).record(control),
        );
        let target = minimum(
            annual,
            ratio(needed, comptime!(SEED_RETURN), comptime!(SEED_RETURN - 1)).record(control),
        );
        let seed_funded_work = times(
            plus(sown_seed, minimum(grain, seed)).record(control),
            minutes * comptime!(SEED_RETURN),
        )
        .record(control);
        let remaining = positive(
            minimum(times(target, minutes).record(control), seed_funded_work),
            standing,
        );
        let pace = ceildiv(
            config.home.annual_yield_per_plot * minutes,
            max(self.date.growing_days, 1),
        );
        let work = minimum(minimum(ticks(time, 0), times32(plots, pace)), remaining);
        let planted = plus(standing, work).record(control);
        let spent = minus(seed_required(control, planted, minutes), sown_seed);
        self.ledger.set(Field::CropWork, actor, planted);
        self.ledger.set(Field::Grain, actor, minus(grain, spent));
        self.ledger
            .increase_at(control, USED_FIELDS as u32, actor, spent);
        self.ledger.increase(control, Field::Used, actor, spent);
        work.extract(LOW_WORD)
    }

    fn harvest(&mut self, plots: u32) {
        let actor = self.actor;
        let control = self.control;
        let config = self.ledger.config;
        let crop = self.ledger.value(Field::CropWork, actor);
        let yield_ticks = minimum(
            divide(crop, max(config.home.cultivation_minutes, 1)),
            times32(plots, config.home.annual_yield_per_plot),
        );
        self.output(comptime!(Good::Grain as u32), yield_ticks);
        self.ledger.set(
            Field::CropWork,
            actor,
            minus(
                crop,
                times(yield_ticks, config.home.cultivation_minutes).record(control),
            ),
        );
        self.ledger
            .set_number(Field::HarvestYear, actor, self.date.year);
    }

    fn bake_bread(&mut self, mut time: u32) -> u32 {
        let actor = self.actor;
        let control = self.control;
        let config = self.ledger.config;
        let surplus_days = comptime!(match config.economy {
            crate::domain::Economy::Agrarian => 2u32,
            crate::domain::Economy::Industrial => 0u32,
        });
        let target = ticks(
            self.ledger.number(Field::FoodNeed, actor)
                * (config.coverage + config.ring + surplus_days),
            0,
        );
        let bread_needed = positive(target, self.ledger.value(Field::Bread, actor));
        let kitchen = food::HomeKitchen::new(&self.ledger, actor);
        let baked = minimum(
            minimum(
                kitchen.flour_for_bread(control, bread_needed),
                self.ledger.value(Field::Flour, actor),
            ),
            ticks(time / max(config.home.baking_minutes, 1), 0),
        );
        let bread_output = kitchen.bake(control, baked);
        self.ledger
            .set_number(Field::BakingRemainder, actor, bread_output.remainder);
        time -= baked.extract(LOW_WORD) * config.home.baking_minutes;
        let flour = self.ledger.value(Field::Flour, actor);
        self.ledger.set(Field::Flour, actor, minus(flour, baked));
        self.ledger
            .increase_at(control, USED_FIELDS as u32 + 1, actor, baked);
        self.ledger.increase(control, Field::Used, actor, baked);
        self.output(comptime!(Good::Bread as u32), bread_output.quantity);
        time
    }

    fn gather_firewood(&mut self, plots: u32, mut time: u32) -> u32 {
        let actor = self.actor;
        let control = self.control;
        let wood = plus(
            self.ledger.value_at(comptime!(Good::Wood as u32), actor),
            self.ledger.value_at(
                TRANSIT_FIELDS as u32 + comptime!(Good::Wood as u32) - 1,
                actor,
            ),
        )
        .record(control);
        let needed = positive(
            consumption::HeatingDemand::new(&self.ledger, actor).reserve(
                &self.ledger,
                control,
                comptime!(Good::Wood as u32),
            ),
            wood,
        );
        let minutes = comptime!(crate::domain::enterprise::HouseholdProduction::FIREWOOD_MINUTES);
        let gathered = select(
            plots > 0 && !equal(self.ledger.value(Field::Tools, actor), zero()),
            minimum(needed, ticks(time / minutes, 0)),
            zero(),
        );
        time -= gathered.extract(LOW_WORD) * minutes;
        self.output(comptime!(Good::Wood as u32), gathered);
        time
    }

    pub fn produce(&mut self) {
        let actor = self.actor;
        let selected_time = self.ledger.number(Field::Time, actor);
        let available = selected_time + self.ledger.number(Field::HomeTime, actor);
        let mut time = available;
        let plots = self.ledger.number(Field::LandPlots, actor);
        let land_with_tools = plots > 0 && !equal(self.ledger.value(Field::Tools, actor), zero());
        if land_with_tools {
            if self.date.cultivating() {
                time -= self.cultivate(plots, time);
            }
            if self.date.harvesting()
                && self.ledger.number(Field::HarvestYear, actor) != self.date.year
            {
                self.harvest(plots);
            }
        }
        time = self.bake_bread(time);
        time = self.gather_firewood(plots, time);
        let construction = self.ledger.number(Field::HomeMinutes, actor);
        self.ledger
            .set_number(Field::HomeMinutes, actor, construction + available - time);
        self.ledger
            .set_number(Field::Time, actor, min(time, selected_time));
        self.ledger.set_number(
            Field::HomeTime,
            actor,
            max(time, selected_time) - selected_time,
        );
    }
}

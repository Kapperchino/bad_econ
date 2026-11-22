use crate::domain::enterprise::EnterpriseModel;
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
    if economy
        .actors
        .kind(actor)
        .is(comptime!(ActorKind::Household))
        && good == comptime!(Good::Grain as u32)
    {
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
    pub price: u32,
    pub sales: u32,
}

#[cube]
impl TradeFields {
    pub fn new(actors: &ActorTable<'_>, seller: u32, good: u32) -> Self {
        let grain = good == comptime!(Good::Grain as u32)
            && actors.kind(seller).is(comptime!(ActorKind::Household));
        TradeFields {
            price: select(
                grain,
                comptime!(Field::GrainQuote as u32),
                comptime!(Field::Price as u32),
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
    k.output() == good || (k.is(comptime!(ActorKind::Household)) && household_good)
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
            economy.ledger.value_at(good, actor),
        )
}

#[cube]
pub(in crate::simulation::economy) fn grain_dealer(economy: &EconomyView<'_>, buyer: u32) -> bool {
    economy.actors.kind(buyer).output() == comptime!(Good::Grain as u32)
        && economy.ledger.number(Field::EnterpriseModel, buyer)
            == comptime!(EnterpriseModel::Distributor as u32)
}

#[cube]
pub(in crate::simulation::economy) fn grain_counterparty(
    economy: &EconomyView<'_>,
    buyer: u32,
    seller: u32,
) -> bool {
    let buyer_is_dealer = grain_dealer(economy, buyer);
    let seller_is_dealer = economy.ledger.number(Field::EnterpriseModel, seller)
        == comptime!(EnterpriseModel::Distributor as u32);
    select(
        economy
            .actors
            .kind(seller)
            .is(comptime!(ActorKind::Household)),
        buyer_is_dealer,
        !(seller_is_dealer && buyer_is_dealer),
    )
}

const SEED_RETURN: u32 = crate::domain::food::WheatHusbandry::GROSS_PARTS;
const SALES_WINDOW: u32 = 30;

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

    fn grain_for_food(&self, days: u32) -> Ticks {
        let actor = self.actor;
        let control = self.control;
        let config = self.ledger.config;
        let ration = food::HouseholdRation::new(&self.ledger, actor);
        let bread = positive(
            times32(ration.reserve(config), days),
            self.ledger.value(Field::Bread, actor),
        );
        let kitchen = food::HomeKitchen::new(&self.ledger, actor);
        let flour = kitchen.flour_for_bread(control, bread);
        kitchen.grain_for_flour(
            control,
            positive(flour, self.ledger.value(Field::Flour, actor)),
        )
    }

    fn seed_required(&self, work: Ticks, minutes: u32) -> Ticks {
        let control = self.control;
        let batch = minutes * comptime!(SEED_RETURN);
        divide(plus(work, ticks(batch - 1, 0)).record(control), batch)
    }

    fn unplanted_seed(&self, plots: u32) -> Ticks {
        let actor = self.actor;
        let config = self.ledger.config;
        let minutes = config.home.cultivation_minutes;
        let annual = times32(plots, config.home.annual_yield_per_plot * minutes);
        let sown = self.ledger.value(Field::CropWork, actor);
        positive(
            self.seed_required(annual, minutes),
            self.seed_required(sown, minutes),
        )
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
        let config = self.ledger.config;
        let plots = self.ledger.number(Field::LandPlots, actor);
        let seed = self.unplanted_seed(plots);
        let days = select(
            plots > 0,
            self.date.days_to_harvest(),
            config.coverage + config.ring,
        );
        let food = self.grain_for_food(days);
        self.ledger
            .set(Field::GrainReserve, actor, plus(seed, food).record(control));
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
        let stock = self.ledger.value(Field::Grain, actor);
        let reserve = self.ledger.value(Field::GrainReserve, actor);
        let surplus = positive(stock, reserve);
        let expected = ratio(forecast, self.date.days_to_harvest(), 365).record(control);
        let quote = self.ledger.number(Field::GrainQuote, actor);
        if self.day % comptime!(SALES_WINDOW) == comptime!(SALES_WINDOW - 1)
            && config.price_response > 0
            && quote > 0
        {
            let step = max(ceildiv(quote * config.price_response, 20 * POLICY_SCALE), 1);
            let cheaper = max(quote, step + config.minimum_price) - step;
            let dearer = min(quote + step, config.maximum_price);
            let next = select(
                lt(expected, surplus),
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
        let demand = plus(
            self.grain_for_food(horizon),
            select(
                profitable,
                ratio(maximum(sales, ticks(plots, 0)), horizon, 365).record(control),
                zero(),
            ),
        )
        .record(control);
        let seed = self.seed_required(times(annual, minutes).record(control), minutes);
        let sown_seed = self.seed_required(standing, minutes);
        let needed = positive(
            plus(demand, seed).record(control),
            plus(grain, sown_seed).record(control),
        );
        let target = minimum(
            annual,
            ratio(needed, comptime!(SEED_RETURN), comptime!(SEED_RETURN - 1)).record(control),
        );
        let seed_available = positive(grain, self.grain_for_food(1));
        let seed_funded_work = times(
            plus(sown_seed, minimum(seed_available, seed)).record(control),
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
        let spent = minus(self.seed_required(planted, minutes), sown_seed);
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

    fn prepare_food(&mut self, plots: u32, mut time: u32) -> u32 {
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
        let flour_needed = positive(
            kitchen.flour_for_bread(control, bread_needed),
            self.ledger.value(Field::Flour, actor),
        );
        let missing = positive(
            self.ledger.value(Field::NutritionNeed, actor),
            self.ledger.value(Field::Bread, actor),
        );
        let basic_flour = kitchen.flour_for_bread(control, missing);
        let baking_time = minimum(
            ticks(time, 0),
            times(basic_flour, config.home.baking_minutes).record(control),
        )
        .extract(LOW_WORD);
        let grain = self.ledger.value(Field::Grain, actor);
        let seed = self.unplanted_seed(plots);
        let household_food = self.grain_for_food(config.coverage + config.ring);
        let available_grain = minimum(grain, maximum(positive(grain, seed), household_food));
        let milled = minimum(
            minimum(
                kitchen.grain_for_flour(control, flour_needed),
                available_grain,
            ),
            ticks(
                (time - baking_time) / max(config.home.milling_minutes, 1),
                0,
            ),
        );
        let flour_output = kitchen.mill(control, milled);
        self.ledger
            .set_number(Field::MillingRemainder, actor, flour_output.remainder);
        let milled_minutes = milled.extract(LOW_WORD) * config.home.milling_minutes;
        time -= milled_minutes;
        self.ledger.set(Field::Grain, actor, minus(grain, milled));
        self.ledger
            .increase_at(control, USED_FIELDS as u32, actor, milled);
        self.output(comptime!(Good::Flour as u32), flour_output.quantity);
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
        self.ledger.increase(
            control,
            Field::Used,
            actor,
            plus(milled, baked).record(control),
        );
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
        time = self.prepare_food(plots, time);
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

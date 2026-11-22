use super::*;
use crate::domain::enterprise::EnterpriseModel;
use crate::simulation::configuration::EconomyConfig;

#[cube]
pub(super) fn reserve(economy: &EconomyView<'_>, actor: u32, good: u32) -> Ticks {
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
pub(super) struct TradeFields {
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
pub(super) fn supplier(economy: &EconomyView<'_>, actor: u32, good: u32) -> bool {
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
pub(super) fn offers(economy: &EconomyView<'_>, actor: u32, good: u32) -> bool {
    supplier(economy, actor, good)
        && lt(
            reserve(economy, actor, good),
            economy.ledger.value_at(good, actor),
        )
}

#[cube]
pub(super) fn grain_dealer(economy: &EconomyView<'_>, buyer: u32) -> bool {
    economy.actors.kind(buyer).output() == comptime!(Good::Grain as u32)
        && economy.ledger.number(Field::EnterpriseModel, buyer)
            == comptime!(EnterpriseModel::Distributor as u32)
}

#[cube]
pub(super) fn grain_counterparty(economy: &EconomyView<'_>, buyer: u32, seller: u32) -> bool {
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

#[cube]
fn output(ledger: &mut Ledger<'_>, control: &[Atomic<u32>], actor: u32, good: u32, amount: Ticks) {
    ledger.increase_at(control, good, actor, amount);
    ledger.increase_at(control, PRODUCTION_FIELDS as u32 + good - 1, actor, amount);
}

const SEED_RETURN: u32 = crate::domain::food::WheatHusbandry::GROSS_PARTS;
const SALES_WINDOW: u32 = 30;

#[cube]
fn days_to_harvest(config: &EconomyConfig, day: u32) -> u32 {
    365 - (day % 365 + 365 - config.home.harvest_start) % 365
}

#[cube]
fn grain_for_food(ledger: &Ledger<'_>, control: &[Atomic<u32>], actor: u32, days: u32) -> Ticks {
    let config = ledger.config;
    let bread = positive(
        times32(food::reserve_daily(ledger, actor), days),
        ledger.value(Field::Bread, actor),
    );
    let flour = food::Conversion::baking(config).required(
        control,
        bread,
        ledger.number(Field::BakingRemainder, actor),
    );
    food::Conversion::milling(config).required(
        control,
        positive(flour, ledger.value(Field::Flour, actor)),
        ledger.number(Field::MillingRemainder, actor),
    )
}

#[cube]
fn seed_required(control: &[Atomic<u32>], work: Ticks, minutes: u32) -> Ticks {
    let batch = minutes * comptime!(SEED_RETURN);
    divide(add_checked(control, work, ticks(batch - 1, 0)), batch)
}

#[cube]
fn unplanted_seed(ledger: &Ledger<'_>, control: &[Atomic<u32>], actor: u32, plots: u32) -> Ticks {
    let config = ledger.config;
    let minutes = config.home.cultivation_minutes;
    let annual = times32(plots, config.home.annual_yield_per_plot * minutes);
    let sown = ledger.value(Field::CropWork, actor);
    positive(
        seed_required(control, annual, minutes),
        seed_required(control, sown, minutes),
    )
}

#[cube]
pub(super) fn open_grain_quote(ledger: &mut Ledger<'_>, actor: u32) {
    let config = ledger.config;
    let quote = ledger.number(Field::GrainQuote, actor);
    if quote == 0 {
        ledger.set_number(Field::GrainQuote, actor, config.home.grain_price);
    }
}

#[cube]
pub(super) fn reserve_grain(
    ledger: &mut Ledger<'_>,
    control: &[Atomic<u32>],
    actor: u32,
    day: u32,
) {
    let config = ledger.config;
    let plots = ledger.number(Field::LandPlots, actor);
    let seed = unplanted_seed(&*ledger, control, actor, plots);
    let days = select(
        plots > 0,
        days_to_harvest(config, day),
        config.coverage + config.ring,
    );
    let food = grain_for_food(&*ledger, control, actor, days);
    ledger.set(Field::GrainReserve, actor, add_checked(control, seed, food));
}

#[cube]
pub(super) fn close_grain_sales(
    ledger: &mut Ledger<'_>,
    control: &[Atomic<u32>],
    actor: u32,
    day: u32,
) {
    let config = ledger.config;
    let sales = ledger.value(Field::GrainSales, actor);
    let previous = ledger.value(Field::ExpectedGrainSales, actor);
    let forecast = add_checked(
        control,
        scaled(
            control,
            previous,
            comptime!(SALES_WINDOW - 1),
            comptime!(SALES_WINDOW),
        ),
        scaled(control, sales, 365, comptime!(SALES_WINDOW)),
    );
    ledger.set(Field::ExpectedGrainSales, actor, forecast);
    let stock = ledger.value(Field::Grain, actor);
    let reserve = ledger.value(Field::GrainReserve, actor);
    let surplus = positive(stock, reserve);
    let expected = scaled(control, forecast, days_to_harvest(config, day), 365);
    let quote = ledger.number(Field::GrainQuote, actor);
    if day % comptime!(SALES_WINDOW) == comptime!(SALES_WINDOW - 1)
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
        ledger.set_number(Field::GrainQuote, actor, next);
    }
}

#[cube]
fn cultivate(
    ledger: &mut Ledger<'_>,
    control: &[Atomic<u32>],
    actor: u32,
    day: u32,
    plots: u32,
    time: u32,
) -> u32 {
    let config = ledger.config;
    let minutes = config.home.cultivation_minutes;
    let standing = ledger.value(Field::CropWork, actor);
    let annual = times32(plots, config.home.annual_yield_per_plot);
    let grain = ledger.value(Field::Grain, actor);
    let paid_minutes =
        ledger.number(Field::Hours, actor) + ledger.number(Field::HomeMinutes, actor);
    let earnings = positive(
        ledger.value(Field::GrossIncome, actor),
        ledger.value(Field::OpeningGrossIncome, actor),
    );
    let wage = divide(earnings, max(paid_minutes, 1));
    let price = ledger.number(Field::GrainQuote, actor);
    let profitable = paid_minutes == 0
        || !lt(
            times32(price, comptime!(SEED_RETURN - 1)),
            multiply(control, wage, minutes * comptime!(SEED_RETURN)),
        );
    let sales = ledger.value(Field::ExpectedGrainSales, actor);
    let horizon = days_to_harvest(config, day) + 365;
    let demand = add_checked(
        control,
        grain_for_food(&*ledger, control, actor, horizon),
        select(
            profitable,
            scaled(control, maximum(sales, ticks(plots, 0)), horizon, 365),
            zero(),
        ),
    );
    let seed = seed_required(control, multiply(control, annual, minutes), minutes);
    let sown_seed = seed_required(control, standing, minutes);
    let needed = positive(
        add_checked(control, demand, seed),
        add_checked(control, grain, sown_seed),
    );
    let target = minimum(
        annual,
        scaled(
            control,
            needed,
            comptime!(SEED_RETURN),
            comptime!(SEED_RETURN - 1),
        ),
    );
    let seed_available = positive(grain, grain_for_food(&*ledger, control, actor, 1));
    let seed_funded_work = multiply(
        control,
        add_checked(control, sown_seed, minimum(seed_available, seed)),
        minutes * comptime!(SEED_RETURN),
    );
    let remaining = positive(
        minimum(multiply(control, target, minutes), seed_funded_work),
        standing,
    );
    let pace = ceildiv(
        config.home.annual_yield_per_plot * minutes,
        max(farming::growing_days(config), 1),
    );
    let work = minimum(minimum(ticks(time, 0), times32(plots, pace)), remaining);
    let planted = add_checked(control, standing, work);
    let spent = minus(seed_required(control, planted, minutes), sown_seed);
    ledger.set(Field::CropWork, actor, planted);
    ledger.set(Field::Grain, actor, minus(grain, spent));
    ledger.increase_at(control, USED_FIELDS as u32, actor, spent);
    ledger.increase(control, Field::Used, actor, spent);
    work.extract(LOW_WORD)
}

#[cube]
pub(super) fn produce(ledger: &mut Ledger<'_>, control: &[Atomic<u32>], actor: u32, day: u32) {
    let config = ledger.config;
    let selected_time = ledger.number(Field::Time, actor);
    let available = selected_time + ledger.number(Field::HomeTime, actor);
    let mut time = available;
    let plots = ledger.number(Field::LandPlots, actor);
    let harvest = farming::harvesting(config, day);
    let year = farming::year(config, day);
    let tools = !equal(ledger.value(Field::Tools, actor), zero());
    if plots > 0 && tools {
        if farming::cultivating(config, day) {
            time -= cultivate(ledger, control, actor, day, plots, time);
        }
        if harvest && ledger.number(Field::HarvestYear, actor) != year {
            let crop = ledger.value(Field::CropWork, actor);
            let yield_ticks = minimum(
                divide(crop, max(config.home.cultivation_minutes, 1)),
                times32(plots, config.home.annual_yield_per_plot),
            );
            output(
                ledger,
                control,
                actor,
                comptime!(Good::Grain as u32),
                yield_ticks,
            );
            ledger.set(
                Field::CropWork,
                actor,
                minus(
                    crop,
                    multiply(control, yield_ticks, config.home.cultivation_minutes),
                ),
            );
            put_harvest(ledger, actor, year);
        }
    }
    let surplus_days = comptime!(match config.economy {
        crate::domain::Economy::Agrarian => 2u32,
        crate::domain::Economy::Industrial => 0u32,
    });
    let target = ticks(
        ledger.number(Field::FoodNeed, actor) * (config.coverage + config.ring + surplus_days),
        0,
    );
    let bread_needed = positive(target, ledger.value(Field::Bread, actor));
    let milling = food::Conversion::milling(config);
    let baking = food::Conversion::baking(config);
    let milling_remainder = ledger.number(Field::MillingRemainder, actor);
    let baking_remainder = ledger.number(Field::BakingRemainder, actor);
    let flour_needed = positive(
        baking.required(control, bread_needed, baking_remainder),
        ledger.value(Field::Flour, actor),
    );
    let missing = positive(
        ledger.value(Field::NutritionNeed, actor),
        ledger.value(Field::Bread, actor),
    );
    let basic_flour = baking.required(control, missing, baking_remainder);
    let baking_time = minimum(
        ticks(time, 0),
        multiply(control, basic_flour, config.home.baking_minutes),
    )
    .extract(LOW_WORD);
    let grain = ledger.value(Field::Grain, actor);
    let seed = unplanted_seed(&*ledger, control, actor, plots);
    let household_food = grain_for_food(&*ledger, control, actor, config.coverage + config.ring);
    let available_grain = minimum(grain, maximum(positive(grain, seed), household_food));
    let milled = minimum(
        minimum(
            milling.required(control, flour_needed, milling_remainder),
            available_grain,
        ),
        ticks(
            (time - baking_time) / max(config.home.milling_minutes, 1),
            0,
        ),
    );
    let flour_output = milling.process(control, milled, milling_remainder);
    ledger.set_number(Field::MillingRemainder, actor, flour_output.remainder);
    let milled_minutes = milled.extract(LOW_WORD) * config.home.milling_minutes;
    time -= milled_minutes;
    ledger.set(Field::Grain, actor, minus(grain, milled));
    ledger.increase_at(control, USED_FIELDS as u32, actor, milled);
    output(
        ledger,
        control,
        actor,
        comptime!(Good::Flour as u32),
        flour_output.quantity,
    );
    let baked = minimum(
        minimum(
            baking.required(control, bread_needed, baking_remainder),
            ledger.value(Field::Flour, actor),
        ),
        ticks(time / max(config.home.baking_minutes, 1), 0),
    );
    let bread_output = baking.process(control, baked, baking_remainder);
    ledger.set_number(Field::BakingRemainder, actor, bread_output.remainder);
    time -= baked.extract(LOW_WORD) * config.home.baking_minutes;
    let flour = ledger.value(Field::Flour, actor);
    ledger.set(Field::Flour, actor, minus(flour, baked));
    ledger.increase_at(control, USED_FIELDS as u32 + 1, actor, baked);
    ledger.increase(
        control,
        Field::Used,
        actor,
        add_checked(control, milled, baked),
    );
    output(
        ledger,
        control,
        actor,
        comptime!(Good::Bread as u32),
        bread_output.quantity,
    );
    let wood = add_checked(
        control,
        ledger.value_at(comptime!(Good::Wood as u32), actor),
        ledger.value_at(
            TRANSIT_FIELDS as u32 + comptime!(Good::Wood as u32) - 1,
            actor,
        ),
    );
    let needed = positive(
        consumption::fuel_target(&*ledger, control, actor, comptime!(Good::Wood as u32)),
        wood,
    );
    let minutes = comptime!(crate::domain::enterprise::HouseholdProduction::FIREWOOD_MINUTES);
    let gathered = select(
        plots > 0 && tools,
        minimum(needed, ticks(time / minutes, 0)),
        zero(),
    );
    time -= gathered.extract(LOW_WORD) * minutes;
    output(
        ledger,
        control,
        actor,
        comptime!(Good::Wood as u32),
        gathered,
    );
    let construction = ledger.number(Field::HomeMinutes, actor);
    ledger.set_number(Field::HomeMinutes, actor, construction + available - time);
    ledger.set_number(Field::Time, actor, min(time, selected_time));
    ledger.set_number(
        Field::HomeTime,
        actor,
        max(time, selected_time) - selected_time,
    );
}

#[cube]
fn put_harvest(ledger: &mut Ledger<'_>, actor: u32, year: u32) {
    ledger.set_number(Field::HarvestYear, actor, year);
}

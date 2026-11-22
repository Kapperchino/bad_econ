use super::*;
use crate::domain::enterprise::EnterpriseModel;
use crate::simulation::configuration::EconomyConfig;

#[cube]
pub(super) fn reserve(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    actor: u32,
    good: u32,
) -> Ticks {
    let mut held = zero();
    if kind(actors, actor).is(comptime!(ActorKind::Household))
        && good == comptime!(Good::Bread as u32)
    {
        held = ticks(
            number(state, config, comptime!(Field::FoodNeed as u32), actor)
                * (config.coverage + config.ring),
            0,
        );
    }
    if kind(actors, actor).is(comptime!(ActorKind::Household))
        && good == comptime!(Good::Grain as u32)
    {
        held = value(state, config, comptime!(Field::GrainReserve as u32), actor);
    }
    if number(
        state,
        config,
        comptime!(Field::EnterpriseModel as u32),
        actor,
    ) == 1
    {
        let capacity = value(state, config, comptime!(Field::RetailOpen as u32), actor);
        let sales = value(state, config, comptime!(Field::Sales as u32), actor);
        let available = positive(capacity, sales);
        held = positive(value(state, config, good, actor), available);
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
    pub fn new(actors: &[u32], seller: u32, good: u32) -> Self {
        let grain = good == comptime!(Good::Grain as u32)
            && kind(actors, seller).is(comptime!(ActorKind::Household));
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
pub(super) fn supplier(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    actor: u32,
    good: u32,
) -> bool {
    let k = kind(actors, actor);
    let household_good = good == comptime!(Good::Bread as u32)
        || (comptime!(config.home_mode == Feature::Enabled)
            && good == comptime!(Good::Grain as u32)
            && number(state, config, comptime!(Field::LivingMembers as u32), actor) > 0
            && number(state, config, comptime!(Field::GrainQuote as u32), actor) > 0);
    k.output() == good || (k.is(comptime!(ActorKind::Household)) && household_good)
}

#[cube]
pub(super) fn offers(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    actor: u32,
    good: u32,
) -> bool {
    supplier(state, actors, config, actor, good)
        && lt(
            reserve(state, actors, config, actor, good),
            value(state, config, good, actor),
        )
}

#[cube]
pub(super) fn grain_dealer(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    buyer: u32,
) -> bool {
    kind(actors, buyer).output() == comptime!(Good::Grain as u32)
        && number(
            state,
            config,
            comptime!(Field::EnterpriseModel as u32),
            buyer,
        ) == comptime!(EnterpriseModel::Distributor as u32)
}

#[cube]
pub(super) fn grain_counterparty(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    buyer: u32,
    seller: u32,
) -> bool {
    let buyer_is_dealer = grain_dealer(state, actors, config, buyer);
    let seller_is_dealer = number(
        state,
        config,
        comptime!(Field::EnterpriseModel as u32),
        seller,
    ) == comptime!(EnterpriseModel::Distributor as u32);
    select(
        kind(actors, seller).is(comptime!(ActorKind::Household)),
        buyer_is_dealer,
        !(seller_is_dealer && buyer_is_dealer),
    )
}

#[cube]
fn output(
    state: &mut [u32],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
    actor: u32,
    good: u32,
    amount: Ticks,
) {
    increase(state, config, control, good, actor, amount);
    increase(
        state,
        config,
        control,
        PRODUCTION_FIELDS as u32 + good - 1,
        actor,
        amount,
    );
}

const SEED_RETURN: u32 = crate::domain::food::WheatHusbandry::GROSS_PARTS;
const SALES_WINDOW: u32 = 30;

#[cube]
fn days_to_harvest(config: &EconomyConfig, day: u32) -> u32 {
    365 - (day % 365 + 365 - config.home.harvest_start) % 365
}

#[cube]
fn grain_for_food(
    state: &mut [u32],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
    actor: u32,
    days: u32,
) -> Ticks {
    let bread = positive(
        times32(food::reserve_daily(state, config, actor), days),
        value(state, config, comptime!(Field::Bread as u32), actor),
    );
    let flour = food::Conversion::baking(config).required(
        control,
        bread,
        number(
            state,
            config,
            comptime!(Field::BakingRemainder as u32),
            actor,
        ),
    );
    food::Conversion::milling(config).required(
        control,
        positive(
            flour,
            value(state, config, comptime!(Field::Flour as u32), actor),
        ),
        number(
            state,
            config,
            comptime!(Field::MillingRemainder as u32),
            actor,
        ),
    )
}

#[cube]
fn seed_required(control: &[Atomic<u32>], work: Ticks, minutes: u32) -> Ticks {
    let batch = minutes * comptime!(SEED_RETURN);
    divide(add_checked(control, work, ticks(batch - 1, 0)), batch)
}

#[cube]
fn unplanted_seed(
    state: &mut [u32],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
    actor: u32,
    plots: u32,
) -> Ticks {
    let minutes = config.home.cultivation_minutes;
    let annual = times32(plots, config.home.annual_yield_per_plot * minutes);
    let sown = value(state, config, comptime!(Field::CropWork as u32), actor);
    positive(
        seed_required(control, annual, minutes),
        seed_required(control, sown, minutes),
    )
}

#[cube]
pub(super) fn open_grain_quote(state: &mut [u32], config: &EconomyConfig, actor: u32) {
    let quote = number(state, config, comptime!(Field::GrainQuote as u32), actor);
    if quote == 0 {
        set32(
            state,
            config,
            comptime!(Field::GrainQuote as u32),
            actor,
            config.home.grain_price,
        );
    }
}

#[cube]
pub(super) fn reserve_grain(
    state: &mut [u32],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
    actor: u32,
    day: u32,
) {
    let plots = number(state, config, comptime!(Field::LandPlots as u32), actor);
    let seed = unplanted_seed(state, config, control, actor, plots);
    let days = select(
        plots > 0,
        days_to_harvest(config, day),
        config.coverage + config.ring,
    );
    let food = grain_for_food(state, config, control, actor, days);
    set(
        state,
        config,
        comptime!(Field::GrainReserve as u32),
        actor,
        add_checked(control, seed, food),
    );
}

#[cube]
pub(super) fn close_grain_sales(
    state: &mut [u32],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
    actor: u32,
    day: u32,
) {
    let sales = value(state, config, comptime!(Field::GrainSales as u32), actor);
    let previous = value(
        state,
        config,
        comptime!(Field::ExpectedGrainSales as u32),
        actor,
    );
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
    set(
        state,
        config,
        comptime!(Field::ExpectedGrainSales as u32),
        actor,
        forecast,
    );
    let stock = value(state, config, comptime!(Field::Grain as u32), actor);
    let reserve = value(state, config, comptime!(Field::GrainReserve as u32), actor);
    let surplus = positive(stock, reserve);
    let expected = scaled(control, forecast, days_to_harvest(config, day), 365);
    let quote = number(state, config, comptime!(Field::GrainQuote as u32), actor);
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
        set32(
            state,
            config,
            comptime!(Field::GrainQuote as u32),
            actor,
            next,
        );
    }
}

#[cube]
fn cultivate(
    state: &mut [u32],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
    actor: u32,
    day: u32,
    plots: u32,
    time: u32,
) -> u32 {
    let minutes = config.home.cultivation_minutes;
    let standing = value(state, config, comptime!(Field::CropWork as u32), actor);
    let annual = times32(plots, config.home.annual_yield_per_plot);
    let grain = value(state, config, comptime!(Field::Grain as u32), actor);
    let paid_minutes = number(state, config, comptime!(Field::Hours as u32), actor)
        + number(state, config, comptime!(Field::HomeMinutes as u32), actor);
    let earnings = positive(
        value(state, config, comptime!(Field::GrossIncome as u32), actor),
        value(
            state,
            config,
            comptime!(Field::OpeningGrossIncome as u32),
            actor,
        ),
    );
    let wage = divide(earnings, max(paid_minutes, 1));
    let price = number(state, config, comptime!(Field::GrainQuote as u32), actor);
    let profitable = paid_minutes == 0
        || !lt(
            times32(price, comptime!(SEED_RETURN - 1)),
            multiply(control, wage, minutes * comptime!(SEED_RETURN)),
        );
    let sales = value(
        state,
        config,
        comptime!(Field::ExpectedGrainSales as u32),
        actor,
    );
    let horizon = days_to_harvest(config, day) + 365;
    let demand = add_checked(
        control,
        grain_for_food(state, config, control, actor, horizon),
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
    let seed_available = positive(grain, grain_for_food(state, config, control, actor, 1));
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
    set(
        state,
        config,
        comptime!(Field::CropWork as u32),
        actor,
        planted,
    );
    set(
        state,
        config,
        comptime!(Field::Grain as u32),
        actor,
        minus(grain, spent),
    );
    increase(state, config, control, USED_FIELDS as u32, actor, spent);
    increase(
        state,
        config,
        control,
        comptime!(Field::Used as u32),
        actor,
        spent,
    );
    work.extract(LOW_WORD)
}

#[cube]
pub(super) fn produce(
    state: &mut [u32],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
    actor: u32,
    day: u32,
) {
    let selected_time = number(state, config, comptime!(Field::Time as u32), actor);
    let available = selected_time + number(state, config, comptime!(Field::HomeTime as u32), actor);
    let mut time = available;
    let plots = number(state, config, comptime!(Field::LandPlots as u32), actor);
    let harvest = farming::harvesting(config, day);
    let year = farming::year(config, day);
    let tools = !equal(
        value(state, config, comptime!(Field::Tools as u32), actor),
        zero(),
    );
    if plots > 0 && tools {
        if farming::cultivating(config, day) {
            time -= cultivate(state, config, control, actor, day, plots, time);
        }
        if harvest && number(state, config, comptime!(Field::HarvestYear as u32), actor) != year {
            let crop = value(state, config, comptime!(Field::CropWork as u32), actor);
            let yield_ticks = minimum(
                divide(crop, max(config.home.cultivation_minutes, 1)),
                times32(plots, config.home.annual_yield_per_plot),
            );
            output(
                state,
                config,
                control,
                actor,
                comptime!(Good::Grain as u32),
                yield_ticks,
            );
            set(
                state,
                config,
                comptime!(Field::CropWork as u32),
                actor,
                minus(
                    crop,
                    multiply(control, yield_ticks, config.home.cultivation_minutes),
                ),
            );
            put_harvest(state, config, actor, year);
        }
    }
    let surplus_days = comptime!(match config.economy {
        crate::domain::Economy::Agrarian => 2u32,
        crate::domain::Economy::Industrial => 0u32,
    });
    let target = ticks(
        number(state, config, comptime!(Field::FoodNeed as u32), actor)
            * (config.coverage + config.ring + surplus_days),
        0,
    );
    let bread_needed = positive(
        target,
        value(state, config, comptime!(Field::Bread as u32), actor),
    );
    let milling = food::Conversion::milling(config);
    let baking = food::Conversion::baking(config);
    let milling_remainder = number(
        state,
        config,
        comptime!(Field::MillingRemainder as u32),
        actor,
    );
    let baking_remainder = number(
        state,
        config,
        comptime!(Field::BakingRemainder as u32),
        actor,
    );
    let flour_needed = positive(
        baking.required(control, bread_needed, baking_remainder),
        value(state, config, comptime!(Field::Flour as u32), actor),
    );
    let missing = positive(
        value(state, config, comptime!(Field::NutritionNeed as u32), actor),
        value(state, config, comptime!(Field::Bread as u32), actor),
    );
    let basic_flour = baking.required(control, missing, baking_remainder);
    let baking_time = minimum(
        ticks(time, 0),
        multiply(control, basic_flour, config.home.baking_minutes),
    )
    .extract(LOW_WORD);
    let grain = value(state, config, comptime!(Field::Grain as u32), actor);
    let seed = unplanted_seed(state, config, control, actor, plots);
    let household_food =
        grain_for_food(state, config, control, actor, config.coverage + config.ring);
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
    set32(
        state,
        config,
        comptime!(Field::MillingRemainder as u32),
        actor,
        flour_output.remainder,
    );
    let milled_minutes = milled.extract(LOW_WORD) * config.home.milling_minutes;
    time -= milled_minutes;
    set(
        state,
        config,
        comptime!(Field::Grain as u32),
        actor,
        minus(grain, milled),
    );
    increase(state, config, control, USED_FIELDS as u32, actor, milled);
    output(
        state,
        config,
        control,
        actor,
        comptime!(Good::Flour as u32),
        flour_output.quantity,
    );
    let baked = minimum(
        minimum(
            baking.required(control, bread_needed, baking_remainder),
            value(state, config, comptime!(Field::Flour as u32), actor),
        ),
        ticks(time / max(config.home.baking_minutes, 1), 0),
    );
    let bread_output = baking.process(control, baked, baking_remainder);
    set32(
        state,
        config,
        comptime!(Field::BakingRemainder as u32),
        actor,
        bread_output.remainder,
    );
    time -= baked.extract(LOW_WORD) * config.home.baking_minutes;
    let flour = value(state, config, comptime!(Field::Flour as u32), actor);
    set(
        state,
        config,
        comptime!(Field::Flour as u32),
        actor,
        minus(flour, baked),
    );
    increase(state, config, control, USED_FIELDS as u32 + 1, actor, baked);
    increase(
        state,
        config,
        control,
        comptime!(Field::Used as u32),
        actor,
        add_checked(control, milled, baked),
    );
    output(
        state,
        config,
        control,
        actor,
        comptime!(Good::Bread as u32),
        bread_output.quantity,
    );
    let wood = add_checked(
        control,
        value(state, config, comptime!(Good::Wood as u32), actor),
        value(
            state,
            config,
            TRANSIT_FIELDS as u32 + comptime!(Good::Wood as u32) - 1,
            actor,
        ),
    );
    let needed = positive(
        consumption::fuel_target(state, config, control, actor, comptime!(Good::Wood as u32)),
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
        state,
        config,
        control,
        actor,
        comptime!(Good::Wood as u32),
        gathered,
    );
    let construction = number(state, config, comptime!(Field::HomeMinutes as u32), actor);
    set32(
        state,
        config,
        comptime!(Field::HomeMinutes as u32),
        actor,
        construction + available - time,
    );
    set32(
        state,
        config,
        comptime!(Field::Time as u32),
        actor,
        min(time, selected_time),
    );
    set32(
        state,
        config,
        comptime!(Field::HomeTime as u32),
        actor,
        max(time, selected_time) - selected_time,
    );
}

#[cube]
fn put_harvest(state: &mut [u32], config: &EconomyConfig, actor: u32, year: u32) {
    set32(
        state,
        config,
        comptime!(Field::HarvestYear as u32),
        actor,
        year,
    );
}

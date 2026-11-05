use super::*;
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

#[cube]
pub(super) fn offers(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    actor: u32,
    good: u32,
) -> bool {
    let k = kind(actors, actor);
    let producer = k.output() == good
        || (k.is(comptime!(ActorKind::Household)) && good == comptime!(Good::Bread as u32));
    producer
        && lt(
            reserve(state, actors, config, actor, good),
            value(state, config, good, actor),
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
    let season = day % 365;
    let harvest = season >= config.home.harvest_start
        && season < config.home.harvest_start + config.home.harvest_days;
    let tools = !equal(
        value(state, config, comptime!(Field::Tools as u32), actor),
        zero(),
    );
    if plots > 0 && tools {
        let cultivated = min(time, plots * 20);
        increase(
            state,
            config,
            control,
            comptime!(Field::CropWork as u32),
            actor,
            ticks(cultivated, 0),
        );
        time -= cultivated;
        if harvest
            && number(state, config, comptime!(Field::HarvestYear as u32), actor) != day / 365 + 1
        {
            let crop = value(state, config, comptime!(Field::CropWork as u32), actor);
            let yield_ticks = minimum(
                divide(crop, max(config.home.cultivation_minutes, 1)),
                ticks(plots * config.home.annual_yield_per_plot, 0),
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
            put_harvest(state, config, actor, day / 365 + 1);
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
    let flour_needed = positive(
        bread_needed,
        value(state, config, comptime!(Field::Flour as u32), actor),
    );
    let milled = minimum(
        minimum(
            flour_needed,
            value(state, config, comptime!(Field::Grain as u32), actor),
        ),
        ticks(time / max(config.home.milling_minutes, 1), 0),
    );
    let milled_minutes = milled.extract(LOW_WORD) * config.home.milling_minutes;
    time -= milled_minutes;
    let grain = value(state, config, comptime!(Field::Grain as u32), actor);
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
        milled,
    );
    let baked = minimum(
        minimum(
            bread_needed,
            value(state, config, comptime!(Field::Flour as u32), actor),
        ),
        ticks(time / max(config.home.baking_minutes, 1), 0),
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
        baked,
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

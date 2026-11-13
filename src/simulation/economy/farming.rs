use super::*;

#[repr(u32)]
#[derive(Clone, Copy)]
enum CropStage {
    Dormant,
    Growing,
    Harvesting,
    Continuous,
}

#[cube]
pub(super) fn growing_days(config: &EconomyConfig) -> u32 {
    min(211, 365 - config.home.harvest_days)
}

#[cube]
fn elapsed(config: &EconomyConfig, day: u32) -> u32 {
    (day % 365 + 365 - config.home.harvest_start) % 365
}

#[cube]
fn stage(config: &EconomyConfig, day: u32) -> u32 {
    let offset = elapsed(config, day);
    let selected = select(
        offset >= 365 - growing_days(config),
        comptime!(CropStage::Growing as u32),
        comptime!(CropStage::Dormant as u32),
    );
    let selected = select(
        offset < config.home.harvest_days,
        comptime!(CropStage::Harvesting as u32),
        selected,
    );
    select(
        config.home.harvest_days == 365,
        comptime!(CropStage::Continuous as u32),
        selected,
    )
}

#[cube]
pub(super) fn cultivating(config: &EconomyConfig, day: u32) -> bool {
    let phase = stage(config, day);
    phase == comptime!(CropStage::Growing as u32)
        || phase == comptime!(CropStage::Continuous as u32)
}

#[cube]
pub(super) fn harvesting(config: &EconomyConfig, day: u32) -> bool {
    let phase = stage(config, day);
    phase == comptime!(CropStage::Harvesting as u32)
        || phase == comptime!(CropStage::Continuous as u32)
}

#[cube]
pub(super) fn year(config: &EconomyConfig, day: u32) -> u32 {
    day / 365 + u32::cast_from(day % 365 >= config.home.harvest_start) + 1
}

#[cube]
pub(super) fn farm(state: &mut [u32], config: &EconomyConfig, actor: u32, good: u32) -> bool {
    let crop = good == comptime!(Good::Grain as u32)
        || good == comptime!(Good::Flax as u32)
        || good == comptime!(Good::Cotton as u32)
        || good == comptime!(Good::Dye as u32);
    crop && number(
        state,
        config,
        comptime!(Field::EnterpriseModel as u32),
        actor,
    ) == 0
}

#[cube]
pub(super) fn starts(state: &mut [u32], config: &EconomyConfig, actor: u32, day: u32) -> Ticks {
    select(
        cultivating(config, day),
        maximum(
            value(state, config, comptime!(Field::Demand as u32), actor),
            ticks(1, 0),
        ),
        zero(),
    )
}

#[cube]
pub(super) fn inventory_target(
    config: &EconomyConfig,
    control: &[Atomic<u32>],
    demand: Ticks,
    day: u32,
) -> Ticks {
    multiply(control, demand, 365 - elapsed(config, day))
}

#[cube]
pub(super) fn harvest(
    state: &mut [u32],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
    actor: u32,
    day: u32,
    completed: Ticks,
) -> Ticks {
    let pending = add_checked(
        control,
        value(state, config, comptime!(Field::CropWork as u32), actor),
        completed,
    );
    let ready = harvesting(config, day);
    let mut yield_ticks = zero();
    if ready {
        let denominator = max(growing_days(config), 1);
        let multiplier = select(config.home.harvest_days == 365, 1u32, 365u32);
        let whole = divide(pending, denominator);
        let remainder = minus(pending, multiply(control, whole, denominator));
        yield_ticks = add_checked(
            control,
            multiply(control, whole, multiplier),
            divide(multiply(control, remainder, multiplier), denominator),
        );
    }
    set(
        state,
        config,
        comptime!(Field::CropWork as u32),
        actor,
        select(ready, zero(), pending),
    );
    yield_ticks
}

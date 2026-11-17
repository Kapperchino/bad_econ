use super::*;

#[repr(u32)]
#[derive(Clone, Copy)]
enum CropStage {
    Dormant,
    Growing,
    Harvesting,
    Continuous,
}

#[derive(CubeType)]
pub(super) struct HarvestYield {
    multiplier: u32,
    divisor: u32,
}

#[cube]
impl HarvestYield {
    pub fn new(config: &EconomyConfig) -> Self {
        HarvestYield {
            multiplier: select(config.home.harvest_days == 365, 1u32, 365u32),
            divisor: max(growing_days(config), 1),
        }
    }

    fn quantity(&self, control: &[Atomic<u32>], completed: Ticks) -> Ticks {
        let whole = divide(completed, self.divisor);
        let remainder = minus(completed, multiply(control, whole, self.divisor));
        add_checked(
            control,
            multiply(control, whole, self.multiplier),
            divide(multiply(control, remainder, self.multiplier), self.divisor),
        )
    }
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
pub(super) fn production_target(control: &[Atomic<u32>], demand: Ticks) -> Ticks {
    divide(
        add_checked(control, multiply(control, demand, 5), ticks(3, 0)),
        4,
    )
}

#[cube]
pub(super) fn starts(
    state: &mut [u32],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
    actor: u32,
    good: u32,
    day: u32,
) -> Ticks {
    let demand = value(state, config, comptime!(Field::Demand as u32), actor);
    let buffered = production_target(control, demand);
    let days = 365 - elapsed(config, day);
    let target = multiply(control, buffered, days + 365);
    let pending = HarvestYield::new(config).quantity(
        control,
        value(state, config, comptime!(Field::CropWork as u32), actor),
    );
    let available = add_checked(control, value(state, config, good, actor), pending);
    let required = positive(target, available);
    let work = divide(
        add_checked(
            control,
            multiply(control, required, growing_days(config)),
            ticks(364, 0),
        ),
        365,
    );
    let remaining = max(min(days, growing_days(config)), 1);
    let daily = divide(
        add_checked(control, work, ticks(remaining - 1, 0)),
        remaining,
    );
    let recurring = proposed_starts(
        KernelStarts {
            coverage: u32::cast_from(config.coverage),
            demand: buffered,
            finished: value(state, config, good, actor),
            pipeline: value(state, config, comptime!(Field::Pipeline as u32), actor),
        },
        control,
    );
    select(
        cultivating(config, day),
        select(config.home.harvest_days == 365, recurring, daily),
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
    multiply(
        control,
        production_target(control, demand),
        365 - elapsed(config, day),
    )
}

#[cube]
pub(super) fn harvest(
    state: &mut [u32],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
    actor: u32,
    day: u32,
    completed: Ticks,
    good: u32,
) -> Ticks {
    let pending = add_checked(
        control,
        value(state, config, comptime!(Field::CropWork as u32), actor),
        completed,
    );
    let ready = harvesting(config, day);
    let yield_ticks = if ready {
        let available = HarvestYield::new(config).quantity(control, pending);
        wheat_harvest(state, config, control, actor, day, good, available)
    } else {
        zero()
    };
    set(
        state,
        config,
        comptime!(Field::CropWork as u32),
        actor,
        select(ready, zero(), pending),
    );
    yield_ticks
}

#[cube]
fn wheat_harvest(
    state: &mut [u32],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
    actor: u32,
    day: u32,
    good: u32,
    available: Ticks,
) -> Ticks {
    let metric_crop = select(config.wheat_kg_per_hectare > 0, good, 0u32);
    if metric_crop == comptime!(Good::Grain as u32) {
        let hectares = add_checked(
            control,
            value(state, config, comptime!(Field::FarmHectares as u32), actor),
            value(state, config, comptime!(Field::OccupiedPlots as u32), actor),
        );
        let harvest_year = year(config, day);
        let previous_year = number(state, config, comptime!(Field::HarvestYear as u32), actor);
        let previous = select(
            previous_year == harvest_year,
            value(
                state,
                config,
                comptime!(Field::HarvestedThisYear as u32),
                actor,
            ),
            zero(),
        );
        let method = number(state, config, comptime!(Field::ActiveMethod as u32), actor);
        let yield_per_hectare = config.wheat_kg_per_hectare * select(method > 0, 2u32, 1u32);
        let maximum = multiply(control, hectares, yield_per_hectare);
        let quantity = minimum(available, positive(maximum, previous));
        set32(
            state,
            config,
            comptime!(Field::HarvestYear as u32),
            actor,
            harvest_year,
        );
        let total = add_checked(control, previous, quantity);
        set(
            state,
            config,
            comptime!(Field::HarvestedThisYear as u32),
            actor,
            total,
        );
        quantity
    } else {
        available
    }
}

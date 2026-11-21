use super::*;
use crate::domain::food::WheatHusbandry;

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
fn rotated_wheat(config: &EconomyConfig, good: u32) -> bool {
    comptime!(config.commercial_wheat == WheatHusbandry::NorfolkFourCourse)
        && good == comptime!(Good::Grain as u32)
}

#[cube]
pub(in crate::simulation) fn paid_days(config: &EconomyConfig, good: u32) -> u32 {
    let days = growing_days(config);
    select(
        rotated_wheat(config, good),
        days + ceildiv(days, comptime!(WheatHusbandry::HARVEST_WORK_PER_PACKET)),
        days,
    )
}

#[cube]
pub(super) fn fieldwork(config: &EconomyConfig, day: u32, good: u32) -> bool {
    cultivating(config, day) || (rotated_wheat(config, good) && harvesting(config, day))
}

#[cube]
fn hectares(
    state: &mut [u32],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
    actor: u32,
) -> Ticks {
    add_checked(
        control,
        value(state, config, comptime!(Field::FarmHectares as u32), actor),
        value(state, config, comptime!(Field::OccupiedPlots as u32), actor),
    )
}

#[cube]
fn rotation_yield(
    state: &mut [u32],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
    actor: u32,
) -> Ticks {
    scaled(
        control,
        hectares(state, config, control, actor),
        config.wheat_kg_per_hectare * comptime!(WheatHusbandry::MARKETABLE_PARTS),
        comptime!(WheatHusbandry::GROSS_PARTS * WheatHusbandry::ROTATION_FIELDS),
    )
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
fn harvested(state: &mut [u32], config: &EconomyConfig, actor: u32, day: u32) -> Ticks {
    let previous_year = number(state, config, comptime!(Field::HarvestYear as u32), actor);
    select(
        previous_year == year(config, day),
        value(
            state,
            config,
            comptime!(Field::HarvestedThisYear as u32),
            actor,
        ),
        zero(),
    )
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
    let standing = value(state, config, comptime!(Field::CropWork as u32), actor);
    let pending = HarvestYield::new(config).quantity(control, standing);
    let available = add_checked(control, value(state, config, good, actor), pending);
    let mut required = positive(target, available);
    let mut recurring = proposed_starts(
        KernelStarts {
            coverage: u32::cast_from(config.coverage),
            demand: buffered,
            finished: value(state, config, good, actor),
            pipeline: value(state, config, comptime!(Field::Pipeline as u32), actor),
        },
        control,
    );
    if rotated_wheat(config, good) {
        let maximum = rotation_yield(state, config, control, actor);
        required = minimum(required, positive(maximum, pending));
        if config.home.harvest_days == 365 {
            recurring = minimum(
                recurring,
                positive(maximum, harvested(state, config, actor, day)),
            );
        }
    }
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
    let desired = select(
        cultivating(config, day),
        select(config.home.harvest_days == 365, recurring, daily),
        zero(),
    );
    if rotated_wheat(config, good) && harvesting(config, day) && config.home.harvest_days < 365 {
        let pace = comptime!(WheatHusbandry::HARVEST_WORK_PER_PACKET);
        divide(add_checked(control, standing, ticks(pace - 1, 0)), pace)
    } else {
        desired
    }
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
fn harvest(
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
    let yield_ticks = if ready {
        HarvestYield::new(config).quantity(control, pending)
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
fn reap(
    state: &mut [u32],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
    actor: u32,
    day: u32,
    completed: Ticks,
) -> Ticks {
    let standing = value(state, config, comptime!(Field::CropWork as u32), actor);
    let reaped = multiply(
        control,
        completed,
        comptime!(WheatHusbandry::HARVEST_WORK_PER_PACKET),
    );
    let left = positive(standing, reaped);
    let wheat = HarvestYield::new(config);
    let recovered = minus(
        wheat.quantity(control, standing),
        wheat.quantity(control, left),
    );
    let last_day = elapsed(config, day) + 1 == config.home.harvest_days;
    set(
        state,
        config,
        comptime!(Field::CropWork as u32),
        actor,
        select(last_day, zero(), left),
    );
    recovered
}

#[derive(CubeType)]
pub(super) struct FarmOutput {
    pub unscaled: Ticks,
    pub finished: Ticks,
}

#[cube]
pub(super) fn finish(
    state: &mut [u32],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
    actor: u32,
    day: u32,
    good: u32,
    completed: Ticks,
) -> FarmOutput {
    let crop = farm(state, config, actor, good);
    let ready = harvesting(config, day);
    let mut available = completed;
    if crop {
        available = if rotated_wheat(config, good) && ready && config.home.harvest_days < 365 {
            reap(state, config, control, actor, day, completed)
        } else {
            harvest(state, config, control, actor, day, completed)
        };
    }
    let mut unscaled = available;
    let finished = if crop
        && ready
        && config.wheat_kg_per_hectare > 0
        && good == comptime!(Good::Grain as u32)
    {
        let previous = harvested(state, config, actor, day);
        let maximum = if rotated_wheat(config, good) {
            rotation_yield(state, config, control, actor)
        } else {
            let method = number(state, config, comptime!(Field::ActiveMethod as u32), actor);
            let yield_per_hectare = config.wheat_kg_per_hectare * select(method > 0, 2u32, 1u32);
            multiply(
                control,
                hectares(state, config, control, actor),
                yield_per_hectare,
            )
        };
        let remaining = positive(maximum, previous);
        unscaled = minimum(available, remaining);
        let scaled = property::scaled_output(state, config, control, actor, unscaled);
        let finished = select(
            rotated_wheat(config, good),
            minimum(scaled, remaining),
            scaled,
        );
        if rotated_wheat(config, good) && equal(finished, remaining) {
            set32(state, config, comptime!(Field::CropWork as u32), actor, 0);
        }
        set32(
            state,
            config,
            comptime!(Field::HarvestYear as u32),
            actor,
            year(config, day),
        );
        let total = add_checked(
            control,
            previous,
            select(rotated_wheat(config, good), finished, unscaled),
        );
        set(
            state,
            config,
            comptime!(Field::HarvestedThisYear as u32),
            actor,
            total,
        );
        finished
    } else {
        property::scaled_output(state, config, control, actor, available)
    };
    FarmOutput { unscaled, finished }
}

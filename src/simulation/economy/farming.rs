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
fn hectares(ledger: &Ledger<'_>, control: &[Atomic<u32>], actor: u32) -> Ticks {
    add_checked(
        control,
        ledger.value(Field::FarmHectares, actor),
        ledger.value(Field::OccupiedPlots, actor),
    )
}

#[cube]
fn rotation_yield(ledger: &Ledger<'_>, control: &[Atomic<u32>], actor: u32) -> Ticks {
    let config = ledger.config;
    scaled(
        control,
        hectares(ledger, control, actor),
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
pub(super) fn funded_shifts(
    ledger: &Ledger<'_>,
    control: &[Atomic<u32>],
    firm: u32,
    good: u32,
    day: u32,
    budget: &employment::ShiftBudget,
) -> Ticks {
    let config = ledger.config;
    let mut funded = divide(budget.cash, budget.pay);
    if rotated_wheat(config, good)
        && config.home.harvest_days < 365
        && cultivating(config, day)
        && farm(ledger, firm, good)
    {
        let rent = times32(config.firm_rent, 365 - elapsed(config, day));
        let shifts = divide(positive(budget.cash, rent), budget.pay);
        let units = ledger.number(Field::OutputTicks, firm);
        let standing = ledger.value(Field::CropWork, firm);
        let pace = comptime!(WheatHusbandry::HARVEST_WORK_PER_PACKET);
        let harvest_batch = units * pace;
        let committed = divide(standing, harvest_batch);
        let partial =
            minus(standing, multiply(control, committed, harvest_batch)).extract(LOW_WORD);
        let unreserved = minimum(positive(shifts, committed), ticks(u32::MAX, 0)).extract(LOW_WORD);
        let remaining_work = positive(times32(unreserved, pace), ticks(ceildiv(partial, units), 0));
        funded = divide(remaining_work, pace + 1);
    }
    funded
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
fn harvested(ledger: &Ledger<'_>, actor: u32, day: u32) -> Ticks {
    let config = ledger.config;
    let previous_year = ledger.number(Field::HarvestYear, actor);
    select(
        previous_year == year(config, day),
        ledger.value(Field::HarvestedThisYear, actor),
        zero(),
    )
}

#[cube]
pub(super) fn farm(ledger: &Ledger<'_>, actor: u32, good: u32) -> bool {
    let crop = good == comptime!(Good::Grain as u32)
        || good == comptime!(Good::Flax as u32)
        || good == comptime!(Good::Cotton as u32)
        || good == comptime!(Good::Dye as u32);
    crop && ledger.number(Field::EnterpriseModel, actor) == 0
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
    ledger: &Ledger<'_>,
    control: &[Atomic<u32>],
    actor: u32,
    good: u32,
    day: u32,
) -> Ticks {
    let config = ledger.config;
    let demand = ledger.value(Field::Demand, actor);
    let buffered = production_target(control, demand);
    let days = 365 - elapsed(config, day);
    let target = multiply(control, buffered, days + 365);
    let standing = ledger.value(Field::CropWork, actor);
    let pending = HarvestYield::new(config).quantity(control, standing);
    let available = add_checked(control, ledger.value_at(good, actor), pending);
    let mut required = positive(target, available);
    let mut recurring = proposed_starts(
        KernelStarts {
            coverage: u32::cast_from(config.coverage),
            demand: buffered,
            finished: ledger.value_at(good, actor),
            pipeline: ledger.value(Field::Pipeline, actor),
        },
        control,
    );
    if rotated_wheat(config, good) {
        let maximum = rotation_yield(ledger, control, actor);
        required = minimum(required, positive(maximum, pending));
        if config.home.harvest_days == 365 {
            recurring = minimum(recurring, positive(maximum, harvested(ledger, actor, day)));
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
    ledger: &mut Ledger<'_>,
    control: &[Atomic<u32>],
    actor: u32,
    day: u32,
    completed: Ticks,
) -> Ticks {
    let config = ledger.config;
    let pending = add_checked(control, ledger.value(Field::CropWork, actor), completed);
    let ready = harvesting(config, day);
    let yield_ticks = if ready {
        HarvestYield::new(config).quantity(control, pending)
    } else {
        zero()
    };
    ledger.set(Field::CropWork, actor, select(ready, zero(), pending));
    yield_ticks
}

#[cube]
fn reap(
    ledger: &mut Ledger<'_>,
    control: &[Atomic<u32>],
    actor: u32,
    day: u32,
    completed: Ticks,
) -> Ticks {
    let config = ledger.config;
    let standing = ledger.value(Field::CropWork, actor);
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
    ledger.set(Field::CropWork, actor, select(last_day, zero(), left));
    recovered
}

#[derive(CubeType)]
pub(super) struct FarmOutput {
    pub unscaled: Ticks,
    pub finished: Ticks,
}

#[cube]
pub(super) fn finish(
    ledger: &mut Ledger<'_>,
    control: &[Atomic<u32>],
    actor: u32,
    day: u32,
    good: u32,
    completed: Ticks,
) -> FarmOutput {
    let config = ledger.config;
    let crop = farm(&*ledger, actor, good);
    let ready = harvesting(config, day);
    let mut available = completed;
    if crop {
        available = if rotated_wheat(config, good) && ready && config.home.harvest_days < 365 {
            reap(ledger, control, actor, day, completed)
        } else {
            harvest(ledger, control, actor, day, completed)
        };
    }
    let mut unscaled = available;
    let finished = if crop
        && ready
        && config.wheat_kg_per_hectare > 0
        && good == comptime!(Good::Grain as u32)
    {
        let previous = harvested(&*ledger, actor, day);
        let maximum = if rotated_wheat(config, good) {
            rotation_yield(&*ledger, control, actor)
        } else {
            let method = ledger.number(Field::ActiveMethod, actor);
            let yield_per_hectare = config.wheat_kg_per_hectare * select(method > 0, 2u32, 1u32);
            multiply(
                control,
                hectares(&*ledger, control, actor),
                yield_per_hectare,
            )
        };
        let remaining = positive(maximum, previous);
        unscaled = minimum(available, remaining);
        let scaled = property::scaled_output(ledger, control, actor, unscaled);
        let finished = select(
            rotated_wheat(config, good),
            minimum(scaled, remaining),
            scaled,
        );
        if rotated_wheat(config, good) && equal(finished, remaining) {
            ledger.set_number(Field::CropWork, actor, 0);
        }
        ledger.set_number(Field::HarvestYear, actor, year(config, day));
        let total = add_checked(
            control,
            previous,
            select(rotated_wheat(config, good), finished, unscaled),
        );
        ledger.set(Field::HarvestedThisYear, actor, total);
        finished
    } else {
        property::scaled_output(ledger, control, actor, available)
    };
    FarmOutput { unscaled, finished }
}

use crate::domain::food::WheatHusbandry;
use crate::simulation::economy::*;

#[repr(u32)]
#[derive(Clone, Copy)]
enum CropStage {
    Dormant,
    Growing,
    Harvesting,
    Continuous,
}

#[derive(CubeType)]
pub(in crate::simulation) struct CropCalendar {
    harvest_start: u32,
    harvest_days: u32,
    growing_days: u32,
    wheat_rotation: bool,
}

#[derive(CubeType)]
pub(in crate::simulation::economy) struct CropDay {
    harvest_days: u32,
    elapsed: u32,
    stage: u32,
    wheat_rotation: bool,
    pub growing_days: u32,
    pub year: u32,
}

#[cube]
impl CropCalendar {
    pub fn new(config: &EconomyConfig) -> Self {
        CropCalendar {
            harvest_start: config.home.harvest_start,
            harvest_days: config.home.harvest_days,
            growing_days: min(211, 365 - config.home.harvest_days),
            wheat_rotation: comptime!(config.commercial_wheat == WheatHusbandry::NorfolkFourCourse),
        }
    }

    pub fn paid_days(&self, good: u32) -> u32 {
        let days = self.growing_days;
        select(
            self.wheat_rotation && good == comptime!(Good::Grain as u32),
            days + ceildiv(days, comptime!(WheatHusbandry::HARVEST_WORK_PER_PACKET)),
            days,
        )
    }

    pub(in crate::simulation::economy) fn on(&self, day: u32) -> CropDay {
        let elapsed = (day % 365 + 365 - self.harvest_start) % 365;
        let selected = select(
            elapsed >= 365 - self.growing_days,
            comptime!(CropStage::Growing as u32),
            comptime!(CropStage::Dormant as u32),
        );
        let selected = select(
            elapsed < self.harvest_days,
            comptime!(CropStage::Harvesting as u32),
            selected,
        );
        CropDay {
            harvest_days: self.harvest_days,
            elapsed,
            year: day / 365 + u32::cast_from(day % 365 >= self.harvest_start) + 1,
            growing_days: self.growing_days,
            stage: select(
                self.harvest_days == 365,
                comptime!(CropStage::Continuous as u32),
                selected,
            ),
            wheat_rotation: self.wheat_rotation,
        }
    }
}

#[cube]
impl CropDay {
    fn continuous(&self) -> bool {
        self.stage == comptime!(CropStage::Continuous as u32)
    }

    pub fn cultivating(&self) -> bool {
        self.stage == comptime!(CropStage::Growing as u32) || self.continuous()
    }

    pub fn harvesting(&self) -> bool {
        self.stage == comptime!(CropStage::Harvesting as u32) || self.continuous()
    }

    pub fn days_to_harvest(&self) -> u32 {
        365 - self.elapsed
    }

    fn last_harvest_day(&self) -> bool {
        self.elapsed + 1 == self.harvest_days
    }

    fn rotates_wheat(&self, good: u32) -> bool {
        self.wheat_rotation && good == comptime!(Good::Grain as u32)
    }

    pub fn fieldwork(&self, good: u32) -> bool {
        self.cultivating() || (self.rotates_wheat(good) && self.harvesting())
    }

    pub fn inventory_target(&self, control: &[Atomic<u32>], demand: Ticks) -> Ticks {
        times(production_target(control, demand), self.days_to_harvest()).record(control)
    }
}

#[derive(CubeType)]
struct HarvestYield {
    multiplier: u32,
    divisor: u32,
}

#[cube]
impl HarvestYield {
    fn new(day: &CropDay) -> Self {
        HarvestYield {
            multiplier: select(day.continuous(), 1u32, 365u32),
            divisor: max(day.growing_days, 1),
        }
    }

    fn quantity(&self, control: &[Atomic<u32>], completed: Ticks) -> Ticks {
        let whole = divide(completed, self.divisor);
        let remainder = minus(completed, times(whole, self.divisor).record(control));
        plus(
            times(whole, self.multiplier).record(control),
            divide(
                times(remainder, self.multiplier).record(control),
                self.divisor,
            ),
        )
        .record(control)
    }
}

#[derive(CubeType)]
pub(in crate::simulation::economy) struct FarmSite {
    actor: u32,
    good: u32,
}

#[cube]
impl FarmSite {
    pub fn new(actor: u32, good: u32) -> Self {
        FarmSite { actor, good }
    }

    pub fn is_farm(&self, ledger: &Ledger<'_>) -> bool {
        let crop = self.good == comptime!(Good::Grain as u32)
            || self.good == comptime!(Good::Flax as u32)
            || self.good == comptime!(Good::Cotton as u32)
            || self.good == comptime!(Good::Dye as u32);
        crop && ledger.number(Field::EnterpriseModel, self.actor) == 0
    }

    fn hectares(&self, ledger: &Ledger<'_>, control: &[Atomic<u32>]) -> Ticks {
        plus(
            ledger.value(Field::FarmHectares, self.actor),
            ledger.value(Field::OccupiedPlots, self.actor),
        )
        .record(control)
    }

    fn rotation_yield(&self, ledger: &Ledger<'_>, control: &[Atomic<u32>]) -> Ticks {
        let config = ledger.config;
        ratio(
            self.hectares(ledger, control),
            config.wheat_kg_per_hectare * comptime!(WheatHusbandry::MARKETABLE_PARTS),
            comptime!(WheatHusbandry::GROSS_PARTS * WheatHusbandry::ROTATION_FIELDS),
        )
        .record(control)
    }

    fn harvested(&self, ledger: &Ledger<'_>, year: u32) -> Ticks {
        let previous_year = ledger.number(Field::HarvestYear, self.actor);
        select(
            previous_year == year,
            ledger.value(Field::HarvestedThisYear, self.actor),
            zero(),
        )
    }
}

#[derive(CubeType)]
pub(in crate::simulation::economy) struct FarmOutput {
    pub unscaled: Ticks,
    pub finished: Ticks,
}

#[derive(CubeType)]
pub(in crate::simulation::economy) struct FarmProduction<'a> {
    ledger: Ledger<'a>,
    control: &'a [Atomic<u32>],
    site: FarmSite,
    day: CropDay,
}

#[cube]
impl<'a> FarmProduction<'a> {
    pub fn new(
        ledger: &'a mut Ledger<'_>,
        control: &'a [Atomic<u32>],
        site: FarmSite,
        day: u32,
    ) -> Self {
        let date = CropCalendar::new(ledger.config).on(day);
        FarmProduction::<'a> {
            ledger: Ledger::new(ledger.words, ledger.config),
            control,
            site,
            day: date,
        }
    }

    pub fn funded_shifts(&self, budget: &employment::ShiftBudget) -> Ticks {
        let actor = self.site.actor;
        let control = self.control;
        let ledger = &self.ledger;
        let good = self.site.good;
        let config = ledger.config;
        let mut funded = divide(budget.cash, budget.pay);
        if self.day.rotates_wheat(good)
            && !self.day.continuous()
            && self.day.cultivating()
            && self.site.is_farm(ledger)
        {
            let rent = times32(config.firm_rent, self.day.days_to_harvest());
            let shifts = divide(positive(budget.cash, rent), budget.pay);
            let units = ledger.number(Field::OutputTicks, actor);
            let standing = ledger.value(Field::CropWork, actor);
            let pace = comptime!(WheatHusbandry::HARVEST_WORK_PER_PACKET);
            let harvest_batch = units * pace;
            let committed = divide(standing, harvest_batch);
            let partial =
                minus(standing, times(committed, harvest_batch).record(control)).extract(LOW_WORD);
            let unreserved =
                minimum(positive(shifts, committed), ticks(u32::MAX, 0)).extract(LOW_WORD);
            let remaining_work =
                positive(times32(unreserved, pace), ticks(ceildiv(partial, units), 0));
            funded = divide(remaining_work, pace + 1);
        }
        funded
    }

    pub fn starts(&self) -> Ticks {
        let actor = self.site.actor;
        let control = self.control;
        let ledger = &self.ledger;
        let good = self.site.good;
        let config = ledger.config;
        let demand = ledger.value(Field::Demand, actor);
        let buffered = production_target(control, demand);
        let days = self.day.days_to_harvest();
        let target = times(buffered, days + 365).record(control);
        let standing = ledger.value(Field::CropWork, actor);
        let pending = HarvestYield::new(&self.day).quantity(control, standing);
        let available = plus(ledger.value_at(good, actor), pending).record(control);
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
        if self.day.rotates_wheat(good) {
            let maximum = self.site.rotation_yield(ledger, control);
            required = minimum(required, positive(maximum, pending));
            if self.day.continuous() {
                recurring = minimum(
                    recurring,
                    positive(maximum, self.site.harvested(ledger, self.day.year)),
                );
            }
        }
        let work = divide(
            plus(
                times(required, self.day.growing_days).record(control),
                ticks(364, 0),
            )
            .record(control),
            365,
        );
        let remaining = max(min(days, self.day.growing_days), 1);
        let daily = divide(
            plus(work, ticks(remaining - 1, 0)).record(control),
            remaining,
        );
        let desired = select(
            self.day.cultivating(),
            select(self.day.continuous(), recurring, daily),
            zero(),
        );
        if self.day.rotates_wheat(good) && self.day.harvesting() && !self.day.continuous() {
            let pace = comptime!(WheatHusbandry::HARVEST_WORK_PER_PACKET);
            divide(plus(standing, ticks(pace - 1, 0)).record(control), pace)
        } else {
            desired
        }
    }

    fn harvest(&mut self, completed: Ticks) -> Ticks {
        let actor = self.site.actor;
        let control = self.control;
        let ledger = &mut self.ledger;
        let pending = plus(ledger.value(Field::CropWork, actor), completed).record(control);
        let ready = self.day.harvesting();
        let yield_ticks = if ready {
            HarvestYield::new(&self.day).quantity(control, pending)
        } else {
            zero()
        };
        ledger.set(Field::CropWork, actor, select(ready, zero(), pending));
        yield_ticks
    }

    fn reap(&mut self, completed: Ticks) -> Ticks {
        let actor = self.site.actor;
        let control = self.control;
        let ledger = &mut self.ledger;
        let standing = ledger.value(Field::CropWork, actor);
        let reaped = times(
            completed,
            comptime!(WheatHusbandry::HARVEST_WORK_PER_PACKET),
        )
        .record(control);
        let left = positive(standing, reaped);
        let wheat = HarvestYield::new(&self.day);
        let recovered = minus(
            wheat.quantity(control, standing),
            wheat.quantity(control, left),
        );
        let last_day = self.day.last_harvest_day();
        ledger.set(Field::CropWork, actor, select(last_day, zero(), left));
        recovered
    }

    pub fn finish(&mut self, completed: Ticks) -> FarmOutput {
        let actor = self.site.actor;
        let control = self.control;
        let good = self.site.good;
        let config = self.ledger.config;
        let crop = self.site.is_farm(&self.ledger);
        let ready = self.day.harvesting();
        let mut available = completed;
        if crop {
            available = if self.day.rotates_wheat(good) && ready && !self.day.continuous() {
                self.reap(completed)
            } else {
                self.harvest(completed)
            };
        }
        let ledger = &mut self.ledger;
        let mut unscaled = available;
        let finished = if crop
            && ready
            && config.wheat_kg_per_hectare > 0
            && good == comptime!(Good::Grain as u32)
        {
            let previous = self.site.harvested(&*ledger, self.day.year);
            let maximum = if self.day.rotates_wheat(good) {
                self.site.rotation_yield(&*ledger, control)
            } else {
                let method = ledger.number(Field::ActiveMethod, actor);
                let yield_per_hectare =
                    config.wheat_kg_per_hectare * select(method > 0, 2u32, 1u32);
                times(self.site.hectares(&*ledger, control), yield_per_hectare).record(control)
            };
            let remaining = positive(maximum, previous);
            unscaled = minimum(available, remaining);
            let scaled = property::PropertySite::new(actor).scale_output(ledger, control, unscaled);
            let finished = select(
                self.day.rotates_wheat(good),
                minimum(scaled, remaining),
                scaled,
            );
            if self.day.rotates_wheat(good) && equal(finished, remaining) {
                ledger.set_number(Field::CropWork, actor, 0);
            }
            ledger.set_number(Field::HarvestYear, actor, self.day.year);
            let total = plus(
                previous,
                select(self.day.rotates_wheat(good), finished, unscaled),
            )
            .record(control);
            ledger.set(Field::HarvestedThisYear, actor, total);
            finished
        } else {
            property::PropertySite::new(actor).scale_output(ledger, control, available)
        };
        FarmOutput { unscaled, finished }
    }
}

#[cube]
pub(in crate::simulation::economy) fn production_target(
    control: &[Atomic<u32>],
    demand: Ticks,
) -> Ticks {
    divide(
        plus(times(demand, 5).record(control), ticks(3, 0)).record(control),
        4,
    )
}

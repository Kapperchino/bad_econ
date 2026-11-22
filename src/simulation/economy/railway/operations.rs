use super::*;

#[cube]
fn material(index: u32) -> u32 {
    let mut good = u32::new(comptime!(Good::Coal as i64));
    match index {
        1 => good = comptime!(Good::Steel as u32),
        2 => good = comptime!(Good::Locomotives as u32),
        _ => (),
    }
    good
}

#[cube]
fn supplied(ledger: &Ledger<'_>, firm: u32, units: u32, offset: u32) -> bool {
    let config = ledger.config;
    let mut funded = true;
    for index in 0..3u32 {
        let required = times32(units, config.railway.materials[(offset + index) as usize]);
        funded = funded && !lt(ledger.value_at(material(index), firm), required);
    }
    funded
}

#[cube]
fn consume(ledger: &mut Ledger<'_>, control: &[Atomic<u32>], firm: u32, units: u32, offset: u32) {
    let config = ledger.config;
    for index in 0..3u32 {
        let good = material(index);
        let required = times32(units, config.railway.materials[(offset + index) as usize]);
        let remaining = minus(ledger.value_at(good, firm), required);
        ledger.set_at(good, firm, remaining);
        ledger.increase_at(control, USED_FIELDS as u32 + good - 1, firm, required);
        ledger.increase(control, Field::Used, firm, required);
    }
}

#[cube]
pub(in crate::simulation::economy) fn offer_crew_jobs(
    economy: &mut EconomyView<'_>,
    control: &[Atomic<u32>],
    firm: u32,
) {
    let config = economy.ledger.config;
    let built = economy.ledger.number(Field::RailBuilt, firm);
    let mut segments = 0;
    for index in 0..config.railway.segment_count {
        let company = config.railway.segments[index as usize * SEGMENT_WORDS + 2];
        segments +=
            u32::cast_from(config.railway.companies[company as usize * COMPANY_WORDS] == firm);
    }
    let minutes = economy.ledger.number(Field::LaborMinutes, firm);
    let cart_crews = 1 + ceildiv(
        economy.ledger.number(Field::CartageMinutes, firm),
        max(minutes, 1),
    );
    let requested = cart_crews
        + ceildiv(built, comptime!(crate::domain::railway::SEGMENTS_PER_CREW))
        + u32::cast_from(segments > built);
    let limit = economy.actors.capacity(firm);
    let budget = employment::ShiftBudget::new(&economy.ledger, control, firm);
    let funded = divide(budget.cash, max(budget.pay, 1));
    let operating = economy.ledger.number(Field::EnterpriseStatus, firm)
        == comptime!(crate::domain::enterprise::EnterpriseState::Operating as u32);
    let jobs = minimum(
        funded,
        ticks(select(operating, min(requested, limit), 0), 0),
    );
    economy.ledger.set(Field::Plan, firm, jobs);
    economy.ledger.set(Field::Slots, firm, jobs);
}

#[cube]
pub(super) fn infrastructure(economy: &mut EconomyView<'_>, control: &[Atomic<u32>]) {
    let config = economy.ledger.config;
    let company = CUBE_POS as u32;
    if company < config.railway.company_count {
        let firm = config.railway.companies[company as usize * COMPANY_WORDS];
        let built = economy.ledger.number(Field::RailBuilt, firm);
        let mut units = 0u32;
        let mut next = 0u32;
        let mut edge = UNIT_POS;
        while edge < config.railway.segment_count {
            let base = edge as usize * SEGMENT_WORDS;
            let own = config.railway.segments[base + 2] == company;
            let ordinal = config.railway.segments[base + 4];
            let length = ceildiv(
                config.railway.segments[base + 3],
                config.railway.length_unit,
            );
            units += select(own && ordinal < built, length, 0);
            next += select(own && ordinal == built, length, 0);
            edge += CUBE_DIM;
        }
        let mut region = UNIT_POS;
        while region < config.railway.region_count {
            if config.railway.regions[region as usize * REGION_WORDS] == company {
                let level =
                    regional_number(&economy.ledger, region, comptime!(RailCapacityField::Level));
                units += max(level, 1) - 1;
            }
            region += CUBE_DIM;
        }
        units = parallel::reduce::<u32>(units, comptime!(Reduction::Sum));
        next = parallel::reduce::<u32>(next, comptime!(Reduction::Sum));
        let mut built_count = Shared::<u32>::new();
        let mut service = Shared::<u32>::new();
        let mut budget = Shared::<u32>::new();
        if UNIT_POS == 0 {
            let active = economy.ledger.number(Field::EnterpriseStatus, firm)
                == comptime!(crate::domain::enterprise::EnterpriseState::Operating as u32);
            let crew = economy.ledger.number(Field::Plan, firm)
                - economy.ledger.number(Field::Slots, firm);
            let upkeep_crew = ceildiv(built, comptime!(crate::domain::railway::SEGMENTS_PER_CREW));
            let maintained =
                active && crew >= max(upkeep_crew, 1) && supplied(&economy.ledger, firm, units, 3);
            if maintained {
                consume(&mut economy.ledger, control, firm, units, 3);
            }
            let build = maintained
                && crew > upkeep_crew
                && next > 0
                && supplied(&economy.ledger, firm, next, 0);
            if build {
                consume(&mut economy.ledger, control, firm, next, 0);
            }
            let rail_crews = select(maintained, upkeep_crew, 0) + u32::cast_from(build);
            let shift = economy.ledger.number(Field::LaborMinutes, firm);
            economy
                .ledger
                .set_number(Field::CartageCapacity, firm, (crew - rail_crews) * shift);
            *built_count = built + u32::cast_from(build);
            *service = u32::cast_from(maintained);
            let mut affordable = ticks(config.railway.region_count, 0);
            for index in 0..3u32 {
                affordable = minimum(
                    affordable,
                    divide(
                        economy.ledger.value_at(material(index), firm),
                        config.railway.materials[index as usize],
                    ),
                );
            }
            *budget = affordable.extract(LOW_WORD);
            economy
                .ledger
                .set_number(Field::RailBuilt, firm, *built_count);
            economy.ledger.set_number(
                Field::RailOperating,
                firm,
                select(maintained, *built_count, 0),
            );
            let expired = economy.ledger.value(Field::Railway, firm);
            record_use(&mut economy.ledger, control, firm, expired);
        }
        sync_storage();
        let expanding = enterprises::expands(&*economy, firm);
        let maintained = *service != 0;
        let mut remaining = *budget;
        let mut upgrades = 0u32;
        let mut capacity = 0u64;
        let mut cursor = 0u32;
        while cursor < config.railway.region_count {
            let region = cursor + UNIT_POS;
            let mut own = false;
            let mut level = 0u32;
            let mut requested = 0u32;
            if region < config.railway.region_count {
                let base = region as usize * REGION_WORDS;
                own = config.railway.regions[base] == company;
                if own {
                    level = max(
                        regional_number(
                            &economy.ledger,
                            region,
                            comptime!(RailCapacityField::Level),
                        ),
                        u32::cast_from(config.railway.regions[base + 2] < *built_count),
                    );
                    let demand = regional_value(
                        &economy.ledger,
                        region,
                        comptime!(RailCapacityField::Demand),
                    );
                    requested = u32::cast_from(
                        maintained
                            && expanding
                            && level > 0
                            && level < 1000
                            && lt(times32(level, config.railway.units_per_level), demand),
                    );
                }
            }
            let prefix = parallel::prefix_sum::<u32>(requested);
            level += u32::cast_from(requested > 0 && prefix.inclusive <= remaining);
            let applied = min(remaining, prefix.total);
            remaining -= applied;
            upgrades += applied;
            if own {
                let units = level * config.railway.units_per_level;
                regional_set(
                    &mut economy.ledger,
                    region,
                    ticks(level, 0),
                    comptime!(RailCapacityField::Level),
                );
                regional_set(
                    &mut economy.ledger,
                    region,
                    ticks(select(maintained, units, 0), 0),
                    comptime!(RailCapacityField::Available),
                );
                capacity += u64::cast_from(units);
            }
            cursor += CUBE_DIM;
        }
        let total = parallel::sum_wide(capacity);
        if UNIT_POS == 0 {
            consume(&mut economy.ledger, control, firm, upgrades, 0);
            let capacity = ticks(
                u32::cast_from(total.value),
                u32::cast_from(total.value >> 32),
            );
            let production = select(maintained, capacity, zero());
            economy.ledger.set(Field::RailCapacity, firm, capacity);
            economy.ledger.set(Field::Railway, firm, production);
            economy
                .ledger
                .increase(control, Field::Produced, firm, production);
            economy
                .ledger
                .increase(control, Field::UnscaledProduced, firm, production);
            economy.ledger.set_number(Field::RailRevenue, firm, 0);
        }
    }
}

#[cube]
pub(in crate::simulation::economy) fn demand(
    economy: &mut EconomyView<'_>,
    firm: u32,
    good: u32,
) -> Ticks {
    let config = economy.ledger.config;
    let mut wanted = zero();
    let requested_material = good == comptime!(Good::Coal as u32)
        || good == comptime!(Good::Steel as u32)
        || good == comptime!(Good::Locomotives as u32);
    if comptime!(config.railway.enabled == Feature::Enabled)
        && requested_material
        && economy.actors.kind(firm).bits() == comptime!(Good::Railway as u32)
    {
        for company in 0..config.railway.company_count {
            if config.railway.companies[company as usize * COMPANY_WORDS] == firm {
                let built = economy.ledger.number(Field::RailBuilt, firm);
                let mut units = 0;
                let mut next = 0;
                for edge in 0..config.railway.segment_count {
                    let base = edge as usize * SEGMENT_WORDS;
                    let own = config.railway.segments[base + 2] == company;
                    let ordinal = config.railway.segments[base + 4];
                    let length = ceildiv(
                        config.railway.segments[base + 3],
                        config.railway.length_unit,
                    );
                    units += select(own && ordinal < built, length, 0);
                    next += select(own && ordinal == built, length, 0);
                }
                let expanding = enterprises::expands(&*economy, firm);
                for region in 0..config.railway.region_count {
                    if config.railway.regions[region as usize * REGION_WORDS] == company {
                        let level = regional_number(
                            &economy.ledger,
                            region,
                            comptime!(RailCapacityField::Level),
                        );
                        let demand = regional_value(
                            &economy.ledger,
                            region,
                            comptime!(RailCapacityField::Demand),
                        );
                        units += max(level, 1) - 1;
                        let upgrade = expanding
                            && level > 0
                            && level < 1000
                            && lt(times32(level, config.railway.units_per_level), demand);
                        next += u32::cast_from(upgrade);
                    }
                }
                for index in 0..3u32 {
                    if good == material(index) {
                        let construction = times32(next, config.railway.materials[index as usize]);
                        let upkeep = times(
                            times32(units + next, config.railway.materials[index as usize + 3]),
                            config.coverage + config.ring,
                        )
                        .value;
                        wanted = plus(construction, upkeep).value;
                    }
                }
            }
        }
    }
    wanted
}

#[cube]
pub(super) fn dormant_demand(ledger: &mut Ledger<'_>) {
    let config = ledger.config;
    let region = ABSOLUTE_POS as u32;
    if region < config.railway.region_count {
        let units = times32(
            regional_number(&*ledger, region, comptime!(RailCapacityField::Level)),
            config.railway.units_per_level,
        );
        let available = regional_value(&*ledger, region, comptime!(RailCapacityField::Available));
        let demand = regional_value(&*ledger, region, comptime!(RailCapacityField::Demand));
        regional_set(
            ledger,
            region,
            maximum(demand, minus(units, available)),
            comptime!(RailCapacityField::Demand),
        );
    }
}

#[cube]
pub(super) fn record_use(
    ledger: &mut Ledger<'_>,
    control: &[Atomic<u32>],
    actor: u32,
    units: Ticks,
) {
    ledger.increase_at(
        control,
        USED_FIELDS as u32 + comptime!(Good::Railway as u32) - 1,
        actor,
        units,
    );
    ledger.increase(control, Field::Used, actor, units);
}

#[cube]
pub(super) fn close(ledger: &mut Ledger<'_>, control: &[Atomic<u32>]) {
    let config = ledger.config;
    let company = ABSOLUTE_POS as u32;
    if company < config.railway.company_count {
        let firm = config.railway.companies[company as usize * COMPANY_WORDS];
        let status = ledger.number(Field::EnterpriseStatus, firm);
        if status >= comptime!(crate::domain::enterprise::EnterpriseState::Liquidating as u32) {
            let expired = ledger.value(Field::Railway, firm);
            record_use(ledger, control, firm, expired);
            ledger.set_number(Field::Railway, firm, 0);
            ledger.set_number(Field::RailOperating, firm, 0);
            for region in 0..config.railway.region_count {
                if config.railway.regions[region as usize * REGION_WORDS] == company {
                    regional_set(
                        ledger,
                        region,
                        zero(),
                        comptime!(RailCapacityField::Available),
                    );
                }
            }
        }
    }
}

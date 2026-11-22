use super::*;
use crate::domain::enterprise::{EnterpriseModel, EnterpriseState};
use crate::simulation::configuration::Feature;

#[cube]
pub(super) fn construct_home(ledger: &mut Ledger<'_>, control: &[Atomic<u32>], owner: u32) {
    let config = ledger.config;
    let owned = ledger.number(Field::InvestmentFirm, owner);
    if owned > 0 {
        let firm = owned - 1;
        let status = ledger.number(Field::EnterpriseStatus, firm);
        let constructing = status == comptime!(EnterpriseState::Forming as u32)
            || (status == comptime!(EnterpriseState::Operating as u32)
                && ledger.number(Field::PendingCapacity, firm) > 0);
        let cash = ledger.value(Field::Cash, firm);
        let reserve = ledger.value(Field::OperatingReserve, firm);
        let rent = rent::obligation(&*ledger, control, firm);
        let protected = select(
            status == comptime!(EnterpriseState::Operating as u32),
            maximum(reserve, rent),
            rent,
        );
        let funds = positive(cash, protected);
        let rate = ledger.number(Field::Wage, firm);
        let available = min(
            ledger.number(Field::Time, owner),
            ledger.number(Field::ConstructionRemaining, firm),
        );
        let minutes = minimum(ticks(available, 0), divide(funds, max(rate, 1))).extract(LOW_WORD);
        let wage = times32(minutes, rate);
        if constructing && minutes > 0 {
            let remaining = ledger.number(Field::ConstructionRemaining, firm) - minutes;
            let time = ledger.number(Field::Time, owner) - minutes;
            ledger.set(Field::Cash, firm, minus(cash, wage));
            let mut net = wage;
            let gross = ledger.value(Field::GrossIncome, owner);
            let accrued = add_checked(control, gross, wage);
            ledger.set(Field::GrossIncome, owner, accrued);
            if comptime!(config.payroll == Feature::Enabled) {
                let before = PayrollAccrual::new(gross, 1, config, control);
                let after = PayrollAccrual::new(accrued, 1, config, control);
                net = minus(after.net, before.net);
                ledger.set(Field::PayrollReserve, owner, after.reserve);
                ledger.increase(
                    control,
                    Field::TaxTreasury,
                    owner,
                    minus(after.tax, before.tax),
                );
                ledger.increase(
                    control,
                    Field::ChargeTreasury,
                    owner,
                    minus(after.charge, before.charge),
                );
            }
            ledger.increase(control, Field::Cash, owner, net);
            ledger.set_number(Field::Time, owner, time);
            ledger.set_number(Field::HomeMinutes, owner, minutes);
            ledger.set_number(Field::ConstructionRemaining, firm, remaining);
            let next = select(
                remaining == 0,
                comptime!(EnterpriseState::Operating as u32),
                status,
            );
            ledger.set_number(Field::EnterpriseStatus, firm, next);
        }
    }
}

#[cube]
pub(super) fn maintain(ledger: &mut Ledger<'_>, control: &[Atomic<u32>], firm: u32, packets: u32) {
    let status = ledger.number(Field::EnterpriseStatus, firm);
    let condition = ledger.number(Field::EquipmentCondition, firm);
    let worn = max(condition, min(packets, 1000)) - min(packets, 1000);
    let tools = ledger.value(Field::Tools, firm);
    let maintenance = worn < 700
        && !equal(tools, zero())
        && status == comptime!(EnterpriseState::Operating as u32);
    if maintenance {
        ledger.set(Field::Tools, firm, minus(tools, ticks(1, 0)));
        ledger.increase_at(
            control,
            USED_FIELDS as u32 + comptime!(Good::Tools as u32) - 1,
            firm,
            ticks(1, 0),
        );
        ledger.increase(control, Field::Used, firm, ticks(1, 0));
    }
    ledger.set_number(
        Field::EquipmentCondition,
        firm,
        select(maintenance, 1000, worn),
    );
    let safety = ledger.number(Field::SafetyProvision, firm);
    let spare = ledger.value(Field::Tools, firm);
    let funded = !lt(ledger.value(Field::Cash, firm), ticks(5000, 0));
    if safety == 0 && !equal(spare, zero()) && funded {
        ledger.set(Field::Tools, firm, minus(spare, ticks(1, 0)));
        ledger.increase_at(
            control,
            USED_FIELDS as u32 + comptime!(Good::Tools as u32) - 1,
            firm,
            ticks(1, 0),
        );
        ledger.increase(control, Field::Used, firm, ticks(1, 0));
        ledger.set_number(Field::SafetyProvision, firm, 750);
    }
}

#[cube]
pub(crate) fn mobility(
    economy: &mut EconomyView<'_>,
    spatial: &[u32],
    quotes: &[u32],
    household: u32,
    day: u32,
    #[comptime] quote_mode: crate::simulation::enterprise::EntryQuoteMode,
) {
    let config = economy.ledger.config;
    let needs_food = economy.ledger.number(Field::Shortage, household) > 0
        && economy.ledger.number(Field::FoodAccess, household) == 1;
    let idle = economy.ledger.number(Field::Employed, household) == 0;
    let landless = economy.ledger.number(Field::LandPlots, household) == 0;
    let settled = economy.ledger.number(Field::MoveDue, household) == 0;
    let alive = economy.ledger.number(Field::LivingMembers, household) > 0;
    let mut target = 0u32;
    if comptime!(config.economy == crate::domain::Economy::Industrial) && settled && alive {
        target = crate::simulation::economy::mobility::work_target(
            economy, spatial, quotes, household, day, quote_mode,
        );
    }
    let mut nearest = u32::new(comptime!(i64::from(u32::MAX)));
    if target == 0 && needs_food && idle && landless && settled && alive {
        for sample in 0..16u32 {
            let candidate = (household + day * 17 + sample * 31) % config.actors;
            let accessible =
                household_production::offers(&*economy, candidate, comptime!(Good::Bread as u32));
            let trip = economy.actors.distance(household, candidate);
            let days = max(ceildiv(trip, config.shipment_speed), 1) + config.coverage;
            let food = times32(
                economy.ledger.number(Field::FoodNeed, household),
                economy.ledger.number(Field::Price, candidate),
            );
            let moving_budget = plus(ticks(100, 0), times(food, days).value).value;
            let affordable = !lt(economy.ledger.value(Field::Cash, household), moving_budget);
            if accessible && affordable && candidate != household && trip < nearest {
                target = candidate + 1;
                nearest = trip;
            }
        }
    }
    economy
        .ledger
        .set_number(Field::MoveTarget, household, target);
}

#[cube]
fn retail_viable(ledger: &Ledger<'_>, control: &[Atomic<u32>], firm: u32) -> bool {
    let config = ledger.config;
    let sales = ledger.value(Field::Sales, firm);
    let replacement = multiply(control, sales, ledger.number(Field::ReplacementCost, firm));
    let shifts = ledger.number(Field::Plan, firm) - ledger.number(Field::Slots, firm);
    let payroll = multiply(
        control,
        times32(
            ledger.number(Field::Wage, firm),
            ledger.number(Field::LaborMinutes, firm),
        ),
        shifts,
    );
    let costs = add_checked(
        control,
        add_checked(control, replacement, payroll),
        ticks(config.firm_rent, 0),
    );
    let revenue = ledger.value(Field::BusinessRevenue, firm);
    !equal(sales, zero()) && !lt(revenue, costs)
}

#[cube]
pub(super) fn lifecycle(
    economy: &mut EconomyView<'_>,
    control: &[Atomic<u32>],
    firm: u32,
    day: u32,
) {
    let config = economy.ledger.config;
    let status = economy.ledger.number(Field::EnterpriseStatus, firm);
    let cash = economy.ledger.value(Field::Cash, firm);
    let payroll = times32(
        economy.ledger.number(Field::Wage, firm),
        economy.ledger.number(Field::LaborMinutes, firm),
    );
    let good = economy.actors.kind(firm).output();
    let crop = farming::farm(&economy.ledger, firm, good);
    let payroll = select(
        crop && !farming::fieldwork(config, day, good),
        zero(),
        payroll,
    );
    let carrying = crop
        && (!equal(economy.ledger.value(Field::CropWork, firm), zero())
            || !equal(
                economy
                    .ledger
                    .value_at(economy.actors.kind(firm).output(), firm),
                zero(),
            ));
    let arrears = economy.ledger.value(Field::RentArrears, firm);
    let current = equal(arrears, zero());
    let funded = current && !lt(cash, plus(payroll, ticks(config.firm_rent, 0)).value);
    let sales = !equal(economy.ledger.value(Field::Sales, firm), zero());
    let viable = match u32::cast_from(
        economy.ledger.number(Field::EnterpriseModel, firm)
            == comptime!(EnterpriseModel::Distributor as u32),
    ) {
        1 => {
            status == comptime!(EnterpriseState::Forming as u32)
                || (good == comptime!(Good::Grain as u32) && funded)
                || retail_viable(&economy.ledger, control, firm)
        }
        _ => sales || funded || carrying,
    };
    let inactive = economy.ledger.number(Field::IdleDays, firm);
    let days = select(viable && current, 0, min(inactive + 1, 365));
    let funded_state = select(
        funded,
        comptime!(EnterpriseState::Operating as u32),
        comptime!(EnterpriseState::Dormant as u32),
    );
    let active_state = select(
        status == comptime!(EnterpriseState::Forming as u32),
        status,
        funded_state,
    );
    let next = select(
        days >= 30,
        comptime!(EnterpriseState::Liquidating as u32),
        active_state,
    );
    let next = select(
        status >= comptime!(EnterpriseState::Liquidating as u32),
        status,
        next,
    );
    economy.ledger.set_number(Field::IdleDays, firm, days);
    economy
        .ledger
        .set_number(Field::EnterpriseStatus, firm, next);
}

#[cube]
fn method_output(
    economy: &EconomyView<'_>,
    control: &[Atomic<u32>],
    firm: u32,
    recipe: u32,
) -> Ticks {
    let config = economy.ledger.config;
    let labor = economy.ledger.number_at(recipe, firm);
    let inputs = economy.ledger.number_at(recipe + 1, firm);
    let output = max(economy.ledger.number_at(recipe + 2, firm), 1);
    let desired = proposed_starts(
        KernelStarts {
            coverage: u32::cast_from(config.coverage),
            demand: economy.ledger.value(Field::Demand, firm),
            finished: economy
                .ledger
                .value_at(economy.actors.kind(firm).output(), firm),
            pipeline: economy.ledger.value(Field::Pipeline, firm),
        },
        control,
    );
    let mut packets = minimum(
        divide(add_checked(control, desired, ticks(output - 1, 0)), output),
        ticks(economy.actors.geographic_capacity(firm), 0),
    );
    #[unroll]
    for input in 0..2u32 {
        let good = economy.ledger.number_at(recipe + 3 + input, firm);
        if good > 0 {
            packets = minimum(
                packets,
                divide(economy.ledger.value_at(good, firm), max(inputs, 1)),
            );
        }
    }
    packets = minimum(
        packets,
        divide(
            economy.ledger.value(Field::Cash, firm),
            max(labor * economy.ledger.number(Field::Wage, firm), 1),
        ),
    );
    minimum(multiply(control, packets, output), desired)
}

#[cube]
pub(super) fn available_method(economy: &mut EconomyView<'_>, control: &[Atomic<u32>], firm: u32) {
    let method = economy.ledger.number(Field::ActiveMethod, firm);
    let mut available = true;
    #[unroll]
    for input in 0..2u32 {
        let good = economy
            .ledger
            .number_at(comptime!(Field::FirstInput as u32) + input, firm);
        if good > 0 {
            available = available
                && !lt(
                    economy.ledger.value_at(good, firm),
                    economy.ledger.value(Field::InputTicks, firm),
                );
        }
    }
    let equipped = !equal(economy.ledger.value(Field::Tools, firm), zero())
        || economy.ledger.number(Field::InstalledCapital, firm) > 0;
    let mut constrained = false;
    if method > 0 {
        let upgraded = method_output(
            &*economy,
            control,
            firm,
            comptime!(Field::LaborMinutes as u32),
        );
        let basic = method_output(
            &*economy,
            control,
            firm,
            comptime!(Field::BasicLaborMinutes as u32),
        );
        constrained = lt(upgraded, basic);
    }
    if method > 0 && (!available || !equipped || constrained) {
        for field in 0..5u32 {
            let basic = economy
                .ledger
                .value_at(comptime!(Field::BasicLaborMinutes as u32) + field, firm);
            economy
                .ledger
                .set_at(comptime!(Field::LaborMinutes as u32) + field, firm, basic);
        }
        economy.ledger.set_number(Field::ActiveMethod, firm, 0);
    }
}

#[cube]
pub(super) fn operating_plan(economy: &mut EconomyView<'_>, control: &[Atomic<u32>], firm: u32) {
    let config = economy.ledger.config;
    let distributor = economy.ledger.number(Field::EnterpriseModel, firm) == 1;
    if comptime!(config.home_mode == Feature::Enabled) || distributor || config.firm_rent > 0 {
        let labor = economy.ledger.number(Field::Wage, firm)
            * economy.ledger.number(Field::LaborMinutes, firm);
        let inputs = economy.ledger.number(Field::InputTicks, firm);
        let mut input_cost = zero();
        for input in 0u32..2 {
            if economy
                .ledger
                .number_at(comptime!(Field::FirstInput as u32) + input, firm)
                > 0
            {
                input_cost = add_checked(
                    control,
                    input_cost,
                    times32(
                        inputs,
                        economy
                            .ledger
                            .number_at(comptime!(Field::FirstInputCost as u32) + input, firm),
                    ),
                );
            }
        }
        let output = max(economy.ledger.number(Field::OutputTicks, firm), 1);
        if distributor {
            input_cost = times32(output, economy.ledger.number(Field::ReplacementCost, firm));
        }
        let demand = economy.ledger.value(Field::Demand, firm);
        let packets = maximum(
            divide(add_checked(control, demand, ticks(output - 1, 0)), output),
            ticks(1, 0),
        );
        let coverage = select(
            distributor,
            economy.ledger.number(Field::StockCoverage, firm),
            config.coverage,
        ) + config.ring;
        let replacement = multiply(control, input_cost, coverage);
        let payroll_days = select(
            farming::farm(&economy.ledger, firm, economy.actors.kind(firm).output()),
            max(
                coverage,
                farming::paid_days(config, economy.actors.kind(firm).output()),
            ),
            coverage,
        );
        let reserve = multiply(
            control,
            add_checked(control, times32(labor, payroll_days), replacement),
            packets.extract(LOW_WORD),
        );
        let construction = times32(
            economy.ledger.number(Field::ConstructionRemaining, firm),
            select(
                comptime!(config.property_mode == Feature::Enabled),
                comptime!(Good::Construction.initial_price()),
                economy.ledger.number(Field::Wage, firm),
            ),
        );
        let reserve = add_checked(
            control,
            add_checked(control, reserve, construction),
            rent::obligation(&economy.ledger, control, firm),
        );
        economy.ledger.set(Field::OperatingReserve, firm, reserve);
    }
}

#[cube]
pub(super) fn purchasing_power(
    economy: &EconomyView<'_>,
    control: &[Atomic<u32>],
    actor: u32,
) -> Ticks {
    let config = economy.ledger.config;
    let cash = economy.ledger.value(Field::Cash, actor);
    let distributor = economy.ledger.number(Field::EnterpriseModel, actor) == 1;
    let mut available = cash;
    if !economy
        .actors
        .kind(actor)
        .is(comptime!(ActorKind::Household))
        && (comptime!(config.home_mode == Feature::Enabled) || distributor)
    {
        let labor = times32(
            economy.ledger.number(Field::Wage, actor),
            economy.ledger.number(Field::LaborMinutes, actor),
        );
        let output = max(economy.ledger.number(Field::OutputTicks, actor), 1);
        let mut replacement = zero();
        #[unroll]
        for input in 0..2u32 {
            if economy
                .ledger
                .number_at(comptime!(Field::FirstInput as u32) + input, actor)
                > 0
            {
                replacement = add_checked(
                    control,
                    replacement,
                    times32(
                        economy.ledger.number(Field::InputTicks, actor),
                        economy
                            .ledger
                            .number_at(comptime!(Field::FirstInputCost as u32) + input, actor),
                    ),
                );
            }
        }
        if distributor {
            replacement = times32(output, economy.ledger.number(Field::ReplacementCost, actor));
        }
        let packet_cost = add_checked(control, labor, replacement);
        if packet_cost.extract(HIGH_WORD) > 0 {
            KernelFaults::new(comptime!(FaultFlags::ARITHMETIC_OVERFLOW)).record(control);
        }
        let demand = economy.ledger.value(Field::Demand, actor);
        let forecast = maximum(
            divide(add_checked(control, demand, ticks(output - 1, 0)), output),
            ticks(1, 0),
        );
        let funded = maximum(
            divide(cash, max(packet_cost.extract(LOW_WORD), 1)),
            ticks(1, 0),
        );
        let capacity = ticks(economy.actors.capacity(actor), 0);
        let packets = minimum(minimum(forecast, funded), capacity);
        available = positive(cash, multiply(control, labor, packets.extract(LOW_WORD)));
    }
    if !economy
        .actors
        .kind(actor)
        .is(comptime!(ActorKind::Household))
    {
        available = positive(available, rent::obligation(&economy.ledger, control, actor));
    }
    available
}

#[cube]
pub(super) fn household_quote(ledger: &mut Ledger<'_>, control: &[Atomic<u32>], household: u32) {
    let config = ledger.config;
    let reserve = ticks(
        ledger.number(Field::FoodNeed, household) * (config.coverage + config.ring),
        0,
    );
    let excess = positive(ledger.value(Field::Bread, household), reserve);
    let sales = ledger.value(Field::Sales, household);
    let multiplier = select(
        lt(excess, sales),
        PRICE_SCALE + PRICE_SCALE / 20,
        select(
            lt(sales, excess),
            PRICE_SCALE - PRICE_SCALE / 20,
            PRICE_SCALE,
        ),
    );
    pricing::adjust(ledger, control, household, multiplier);
}

#[cube]
pub(super) fn expand(economy: &mut EconomyView<'_>, control: &[Atomic<u32>], owner: u32) {
    let config = economy.ledger.config;
    let owned = economy.ledger.number(Field::InvestmentFirm, owner);
    if owned > 0 {
        let firm = owned - 1;
        let capacity = economy.actors.capacity(firm);
        let status = economy.ledger.number(Field::EnterpriseStatus, firm);
        let operating = status == comptime!(EnterpriseState::Operating as u32);
        let added_workers = comptime!(match config.property_mode {
            Feature::Disabled => 1u32,
            Feature::Enabled => 0u32,
        }) + economy.ledger.number(Field::WorkersPerPlot, firm);
        let busy = economy.ledger.number(Field::Slots, firm) == 0
            && economy.ledger.number(Field::Plan, firm) >= capacity;
        let target = times32(capacity, economy.ledger.number(Field::OutputTicks, firm));
        let demand = economy.ledger.value(Field::Demand, firm);
        let tools = economy.ledger.value(Field::Tools, firm);
        let wood = economy.ledger.value(Field::Wood, firm);
        let funded = !lt(
            economy.ledger.value(Field::Cash, firm),
            add_checked(
                control,
                economy.ledger.value(Field::OperatingReserve, firm),
                times32(
                    comptime!(match config.property_mode {
                        Feature::Disabled => 720u32,
                        Feature::Enabled =>
                            crate::domain::property::CONSTRUCTION_WORK_PER_PLOT
                                * Good::Construction.initial_price(),
                    }),
                    select(
                        comptime!(config.property_mode == Feature::Enabled),
                        1,
                        economy.ledger.number(Field::Wage, firm),
                    ),
                ),
            ),
        );
        let interested = expands(&*economy, firm);
        if operating
            && interested
            && economy.actors.kind(firm).bits() != comptime!(Good::Housing as u32)
            && economy.actors.kind(firm).bits() != comptime!(Good::Railway as u32)
            && busy
            && lt(target, demand)
            && funded
            && capacity + added_workers <= 10_000
            && (comptime!(config.property_mode == Feature::Enabled)
                || (!lt(tools, ticks(2, 0)) && !lt(wood, ticks(5, 0))))
            && property::authorized(&economy.ledger, firm, owner)
            && economy.ledger.number(Field::ConstructionRemaining, firm) == 0
            && economy.ledger.number(Field::PendingCapacity, firm) == 0
        {
            if comptime!(config.property_mode == Feature::Disabled) {
                economy
                    .ledger
                    .set(Field::Tools, firm, minus(tools, ticks(2, 0)));
                economy
                    .ledger
                    .set(Field::Wood, firm, minus(wood, ticks(5, 0)));
                economy.ledger.increase_at(
                    control,
                    USED_FIELDS as u32 + comptime!(Good::Tools as u32) - 1,
                    firm,
                    ticks(2, 0),
                );
                economy.ledger.increase_at(
                    control,
                    USED_FIELDS as u32 + comptime!(Good::Wood as u32) - 1,
                    firm,
                    ticks(5, 0),
                );
                economy
                    .ledger
                    .increase(control, Field::Used, firm, ticks(7, 0));
                economy
                    .ledger
                    .increase(control, Field::InstalledCapital, firm, ticks(7, 0));
            }
            if comptime!(config.property_mode == Feature::Enabled) {
                economy.ledger.set_number(Field::RequestedPlots, firm, 1);
                economy.ledger.set_number(
                    Field::ProjectStatus,
                    firm,
                    comptime!(crate::domain::property::ProjectStatus::Requested as u32),
                );
            }
            economy.ledger.set_number(
                Field::ConstructionRemaining,
                firm,
                comptime!(match config.property_mode {
                    Feature::Disabled => 720u32,
                    Feature::Enabled => crate::domain::property::CONSTRUCTION_WORK_PER_PLOT,
                }),
            );
            economy.ledger.set_number(
                Field::PendingCapacity,
                firm,
                comptime!(match config.property_mode {
                    Feature::Disabled => 1u32,
                    Feature::Enabled => 0u32,
                }),
            );
        }
    }
}

#[cube]
pub(super) fn expands(economy: &EconomyView<'_>, firm: u32) -> bool {
    let config = economy.ledger.config;
    let workers = economy.actors.capacity(firm);
    let homes = times32(
        economy.ledger.number(Field::OccupiedPlots, firm),
        economy.ledger.number(Field::OutputTicks, firm),
    );
    let capacity = select(
        economy.actors.kind(firm).bits() == comptime!(Good::Housing as u32),
        homes,
        ticks(workers, 0),
    );
    let draw = economy.actors.expansion_draw(firm);
    draw < select(
        !lt(capacity, ticks(20, 0)),
        config.industrial_expansion,
        config.small_business_expansion,
    )
}

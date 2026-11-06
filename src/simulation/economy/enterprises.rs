use super::*;
use crate::domain::enterprise::EnterpriseState;
use crate::simulation::configuration::{EconomyConfig, Feature};

#[cube]
pub(super) fn construct_home(
    state: &mut [u32],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
    owner: u32,
) {
    let owned = number(
        state,
        config,
        comptime!(Field::InvestmentFirm as u32),
        owner,
    );
    if owned > 0 {
        let firm = owned - 1;
        let status = number(
            state,
            config,
            comptime!(Field::EnterpriseStatus as u32),
            firm,
        );
        let constructing = status == comptime!(EnterpriseState::Forming as u32)
            || (status == comptime!(EnterpriseState::Operating as u32)
                && number(
                    state,
                    config,
                    comptime!(Field::PendingCapacity as u32),
                    firm,
                ) > 0);
        let cash = value(state, config, comptime!(Field::Cash as u32), firm);
        let reserve = value(
            state,
            config,
            comptime!(Field::OperatingReserve as u32),
            firm,
        );
        let rent = rent::obligation(state, config, control, firm);
        let protected = select(
            status == comptime!(EnterpriseState::Operating as u32),
            maximum(reserve, rent),
            rent,
        );
        let funds = positive(cash, protected);
        let rate = number(state, config, comptime!(Field::Wage as u32), firm);
        let available = min(
            number(state, config, comptime!(Field::Time as u32), owner),
            number(
                state,
                config,
                comptime!(Field::ConstructionRemaining as u32),
                firm,
            ),
        );
        let minutes = minimum(ticks(available, 0), divide(funds, max(rate, 1))).extract(LOW_WORD);
        let wage = times32(minutes, rate);
        if constructing && minutes > 0 {
            let remaining = number(
                state,
                config,
                comptime!(Field::ConstructionRemaining as u32),
                firm,
            ) - minutes;
            let time = number(state, config, comptime!(Field::Time as u32), owner) - minutes;
            set(
                state,
                config,
                comptime!(Field::Cash as u32),
                firm,
                minus(cash, wage),
            );
            let mut net = wage;
            let gross = value(state, config, comptime!(Field::GrossIncome as u32), owner);
            let accrued = add_checked(control, gross, wage);
            set(
                state,
                config,
                comptime!(Field::GrossIncome as u32),
                owner,
                accrued,
            );
            if comptime!(config.payroll == Feature::Enabled) {
                let before = PayrollAccrual::new(gross, config, control);
                let after = PayrollAccrual::new(accrued, config, control);
                net = minus(after.net, before.net);
                set(
                    state,
                    config,
                    comptime!(Field::PayrollReserve as u32),
                    owner,
                    after.reserve,
                );
                increase(
                    state,
                    config,
                    control,
                    comptime!(Field::TaxTreasury as u32),
                    owner,
                    minus(after.tax, before.tax),
                );
                increase(
                    state,
                    config,
                    control,
                    comptime!(Field::ChargeTreasury as u32),
                    owner,
                    minus(after.charge, before.charge),
                );
            }
            increase(
                state,
                config,
                control,
                comptime!(Field::Cash as u32),
                owner,
                net,
            );
            set32(state, config, comptime!(Field::Time as u32), owner, time);
            set32(
                state,
                config,
                comptime!(Field::HomeMinutes as u32),
                owner,
                minutes,
            );
            set32(
                state,
                config,
                comptime!(Field::ConstructionRemaining as u32),
                firm,
                remaining,
            );
            let next = select(
                remaining == 0,
                comptime!(EnterpriseState::Operating as u32),
                status,
            );
            set32(
                state,
                config,
                comptime!(Field::EnterpriseStatus as u32),
                firm,
                next,
            );
        }
    }
}

#[cube]
pub(super) fn maintain(
    state: &mut [u32],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
    firm: u32,
    packets: u32,
) {
    let status = number(
        state,
        config,
        comptime!(Field::EnterpriseStatus as u32),
        firm,
    );
    let condition = number(
        state,
        config,
        comptime!(Field::EquipmentCondition as u32),
        firm,
    );
    let worn = max(condition, min(packets, 1000)) - min(packets, 1000);
    let tools = value(state, config, comptime!(Field::Tools as u32), firm);
    let maintenance = worn < 700
        && !equal(tools, zero())
        && status == comptime!(EnterpriseState::Operating as u32);
    if maintenance {
        set(
            state,
            config,
            comptime!(Field::Tools as u32),
            firm,
            minus(tools, ticks(1, 0)),
        );
        increase(
            state,
            config,
            control,
            USED_FIELDS as u32 + comptime!(Good::Tools as u32) - 1,
            firm,
            ticks(1, 0),
        );
        increase(
            state,
            config,
            control,
            comptime!(Field::Used as u32),
            firm,
            ticks(1, 0),
        );
    }
    set32(
        state,
        config,
        comptime!(Field::EquipmentCondition as u32),
        firm,
        select(maintenance, 1000, worn),
    );
    let safety = number(
        state,
        config,
        comptime!(Field::SafetyProvision as u32),
        firm,
    );
    let spare = value(state, config, comptime!(Field::Tools as u32), firm);
    let funded = !lt(
        value(state, config, comptime!(Field::Cash as u32), firm),
        ticks(5000, 0),
    );
    if safety == 0 && !equal(spare, zero()) && funded {
        set(
            state,
            config,
            comptime!(Field::Tools as u32),
            firm,
            minus(spare, ticks(1, 0)),
        );
        increase(
            state,
            config,
            control,
            USED_FIELDS as u32 + comptime!(Good::Tools as u32) - 1,
            firm,
            ticks(1, 0),
        );
        increase(
            state,
            config,
            control,
            comptime!(Field::Used as u32),
            firm,
            ticks(1, 0),
        );
        set32(
            state,
            config,
            comptime!(Field::SafetyProvision as u32),
            firm,
            750,
        );
    }
}

#[cube]
pub(crate) fn mobility(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    household: u32,
    day: u32,
) {
    let needs_food = number(state, config, comptime!(Field::Shortage as u32), household) > 0
        && number(
            state,
            config,
            comptime!(Field::FoodAccess as u32),
            household,
        ) == 1;
    let idle = number(state, config, comptime!(Field::Employed as u32), household) == 0;
    let landless = number(state, config, comptime!(Field::LandPlots as u32), household) == 0;
    let settled = number(state, config, comptime!(Field::MoveDue as u32), household) == 0;
    let alive = number(
        state,
        config,
        comptime!(Field::LivingMembers as u32),
        household,
    ) > 0;
    let mut target = 0u32;
    let mut nearest = u32::new(comptime!(i64::from(u32::MAX)));
    if needs_food && idle && landless && settled && alive {
        for sample in 0..16u32 {
            let candidate = (household + day * 17 + sample * 31) % config.actors;
            let accessible = household_production::offers(
                state,
                actors,
                config,
                candidate,
                comptime!(Good::Bread as u32),
            );
            let trip = distance(actors, household, candidate);
            let days = max(ceildiv(trip, config.shipment_speed), 1) + config.coverage;
            let food = times32(
                number(state, config, comptime!(Field::FoodNeed as u32), household),
                number(state, config, comptime!(Field::Price as u32), candidate),
            );
            let moving_budget = plus(ticks(100, 0), times(food, days).value).value;
            let affordable = !lt(
                value(state, config, comptime!(Field::Cash as u32), household),
                moving_budget,
            );
            if accessible && affordable && candidate != household && trip < nearest {
                target = candidate + 1;
                nearest = trip;
            }
        }
    }
    set32(
        state,
        config,
        comptime!(Field::MoveTarget as u32),
        household,
        target,
    );
}

#[cube]
pub(super) fn lifecycle(state: &mut [u32], config: &EconomyConfig, firm: u32) {
    let status = number(
        state,
        config,
        comptime!(Field::EnterpriseStatus as u32),
        firm,
    );
    let cash = value(state, config, comptime!(Field::Cash as u32), firm);
    let payroll = times32(
        number(state, config, comptime!(Field::Wage as u32), firm),
        number(state, config, comptime!(Field::LaborMinutes as u32), firm),
    );
    let arrears = value(state, config, comptime!(Field::RentArrears as u32), firm);
    let current = equal(arrears, zero());
    let funded = current && !lt(cash, plus(payroll, ticks(config.firm_rent, 0)).value);
    let sales = !equal(
        value(state, config, comptime!(Field::Sales as u32), firm),
        zero(),
    );
    let inactive = number(state, config, comptime!(Field::IdleDays as u32), firm);
    let days = select((sales || funded) && current, 0, min(inactive + 1, 365));
    let mut next = status;
    if status == comptime!(EnterpriseState::Operating as u32) && !funded {
        next = comptime!(EnterpriseState::Dormant as u32);
    }
    if status == comptime!(EnterpriseState::Dormant as u32) && funded {
        next = comptime!(EnterpriseState::Operating as u32);
    }
    if days >= 90 && status != comptime!(EnterpriseState::Closed as u32) {
        next = comptime!(EnterpriseState::Liquidating as u32);
    }
    set32(state, config, comptime!(Field::IdleDays as u32), firm, days);
    set32(
        state,
        config,
        comptime!(Field::EnterpriseStatus as u32),
        firm,
        next,
    );
}

#[cube]
fn method_output(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
    firm: u32,
    recipe: u32,
) -> Ticks {
    let labor = number(state, config, recipe, firm);
    let inputs = number(state, config, recipe + 1, firm);
    let output = max(number(state, config, recipe + 2, firm), 1);
    let desired = proposed_starts(
        KernelStarts {
            coverage: config.coverage,
            demand: value(state, config, comptime!(Field::Demand as u32), firm),
            finished: value(state, config, kind(actors, firm).output(), firm),
            pipeline: value(state, config, comptime!(Field::Pipeline as u32), firm),
        },
        control,
    );
    let mut packets = minimum(
        divide(add_checked(control, desired, ticks(output - 1, 0)), output),
        ticks(
            actors[ActorLayout::GEOGRAPHIC_CAPACITY * (actors.len() / ActorLayout::WORDS)
                + firm as usize],
            0,
        ),
    );
    #[unroll]
    for input in 0..2u32 {
        let good = number(state, config, recipe + 3 + input, firm);
        if good > 0 {
            packets = minimum(
                packets,
                divide(value(state, config, good, firm), max(inputs, 1)),
            );
        }
    }
    packets = minimum(
        packets,
        divide(
            value(state, config, comptime!(Field::Cash as u32), firm),
            max(
                labor * number(state, config, comptime!(Field::Wage as u32), firm),
                1,
            ),
        ),
    );
    minimum(multiply(control, packets, output), desired)
}

#[cube]
pub(super) fn available_method(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
    firm: u32,
) {
    let method = number(state, config, comptime!(Field::ActiveMethod as u32), firm);
    let mut available = true;
    #[unroll]
    for input in 0..2u32 {
        let good = number(
            state,
            config,
            comptime!(Field::FirstInput as u32) + input,
            firm,
        );
        if good > 0 {
            available = available
                && !lt(
                    value(state, config, good, firm),
                    value(state, config, comptime!(Field::InputTicks as u32), firm),
                );
        }
    }
    let equipped = !equal(
        value(state, config, comptime!(Field::Tools as u32), firm),
        zero(),
    ) || number(
        state,
        config,
        comptime!(Field::InstalledCapital as u32),
        firm,
    ) > 0;
    let mut constrained = false;
    if method > 0 {
        let upgraded = method_output(
            state,
            actors,
            config,
            control,
            firm,
            comptime!(Field::LaborMinutes as u32),
        );
        let basic = method_output(
            state,
            actors,
            config,
            control,
            firm,
            comptime!(Field::BasicLaborMinutes as u32),
        );
        constrained = lt(upgraded, basic);
    }
    if method > 0 && (!available || !equipped || constrained) {
        for field in 0..5u32 {
            let basic = value(
                state,
                config,
                comptime!(Field::BasicLaborMinutes as u32) + field,
                firm,
            );
            set(
                state,
                config,
                comptime!(Field::LaborMinutes as u32) + field,
                firm,
                basic,
            );
        }
        set32(
            state,
            config,
            comptime!(Field::ActiveMethod as u32),
            firm,
            0,
        );
    }
}

#[cube]
pub(super) fn operating_plan(
    state: &mut [u32],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
    firm: u32,
) {
    let distributor = number(
        state,
        config,
        comptime!(Field::EnterpriseModel as u32),
        firm,
    ) == 1;
    if comptime!(config.home_mode == Feature::Enabled) || distributor || config.firm_rent > 0 {
        let labor = number(state, config, comptime!(Field::Wage as u32), firm)
            * number(state, config, comptime!(Field::LaborMinutes as u32), firm);
        let inputs = number(state, config, comptime!(Field::InputTicks as u32), firm);
        let mut input_cost = zero();
        for input in 0u32..2 {
            if number(
                state,
                config,
                comptime!(Field::FirstInput as u32) + input,
                firm,
            ) > 0
            {
                input_cost = add_checked(
                    control,
                    input_cost,
                    times32(
                        inputs,
                        number(
                            state,
                            config,
                            comptime!(Field::FirstInputCost as u32) + input,
                            firm,
                        ),
                    ),
                );
            }
        }
        let output = max(
            number(state, config, comptime!(Field::OutputTicks as u32), firm),
            1,
        );
        if distributor {
            input_cost = times32(
                output,
                number(
                    state,
                    config,
                    comptime!(Field::ReplacementCost as u32),
                    firm,
                ),
            );
        }
        let overhead = ceildiv(
            config.firm_rent,
            max(
                number(state, config, comptime!(Field::Plan as u32), firm),
                1,
            ),
        );
        let packet_cost = add_checked(control, ticks(labor + overhead, 0), input_cost);
        let unit_cost = divide(
            add_checked(control, packet_cost, ticks(output - 1, 0)),
            output,
        );
        let quote = maximum(
            value(state, config, comptime!(Field::Price as u32), firm),
            unit_cost,
        );
        set(
            state,
            config,
            comptime!(Field::Price as u32),
            firm,
            minimum(quote, ticks(config.maximum_price, 0)),
        );
        let demand = value(state, config, comptime!(Field::Demand as u32), firm);
        let packets = maximum(
            divide(add_checked(control, demand, ticks(output - 1, 0)), output),
            ticks(1, 0),
        );
        let replacement = multiply(
            control,
            input_cost,
            select(
                distributor,
                number(state, config, comptime!(Field::StockCoverage as u32), firm),
                config.coverage,
            ) + config.ring,
        );
        let reserve = multiply(
            control,
            add_checked(control, ticks(labor, 0), replacement),
            packets.extract(LOW_WORD),
        );
        let reserve = add_checked(
            control,
            reserve,
            rent::obligation(state, config, control, firm),
        );
        set(
            state,
            config,
            comptime!(Field::OperatingReserve as u32),
            firm,
            reserve,
        );
    }
}

#[cube]
pub(super) fn purchasing_power(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
    actor: u32,
) -> Ticks {
    let cash = value(state, config, comptime!(Field::Cash as u32), actor);
    let distributor = number(
        state,
        config,
        comptime!(Field::EnterpriseModel as u32),
        actor,
    ) == 1;
    let mut available = cash;
    if !kind(actors, actor).is(comptime!(ActorKind::Household))
        && (comptime!(config.home_mode == Feature::Enabled) || distributor)
    {
        let labor = times32(
            number(state, config, comptime!(Field::Wage as u32), actor),
            number(state, config, comptime!(Field::LaborMinutes as u32), actor),
        );
        let output = max(
            number(state, config, comptime!(Field::OutputTicks as u32), actor),
            1,
        );
        let mut replacement = zero();
        #[unroll]
        for input in 0..2u32 {
            if number(
                state,
                config,
                comptime!(Field::FirstInput as u32) + input,
                actor,
            ) > 0
            {
                replacement = add_checked(
                    control,
                    replacement,
                    times32(
                        number(state, config, comptime!(Field::InputTicks as u32), actor),
                        number(
                            state,
                            config,
                            comptime!(Field::FirstInputCost as u32) + input,
                            actor,
                        ),
                    ),
                );
            }
        }
        if distributor {
            replacement = times32(
                output,
                number(
                    state,
                    config,
                    comptime!(Field::ReplacementCost as u32),
                    actor,
                ),
            );
        }
        let packet_cost = add_checked(control, labor, replacement);
        if packet_cost.extract(HIGH_WORD) > 0 {
            KernelFaults::new(comptime!(FaultFlags::ARITHMETIC_OVERFLOW)).record(control);
        }
        let demand = value(state, config, comptime!(Field::Demand as u32), actor);
        let forecast = maximum(
            divide(add_checked(control, demand, ticks(output - 1, 0)), output),
            ticks(1, 0),
        );
        let funded = maximum(
            divide(cash, max(packet_cost.extract(LOW_WORD), 1)),
            ticks(1, 0),
        );
        let capacity = ticks(
            actors[ActorLayout::CAPACITY * (actors.len() / ActorLayout::WORDS) + actor as usize],
            0,
        );
        let packets = minimum(minimum(forecast, funded), capacity);
        available = positive(cash, multiply(control, labor, packets.extract(LOW_WORD)));
    }
    if !kind(actors, actor).is(comptime!(ActorKind::Household)) {
        available = positive(available, rent::obligation(state, config, control, actor));
    }
    available
}

#[cube]
pub(super) fn household_quote(state: &mut [u32], config: &EconomyConfig, household: u32) {
    let reserve = ticks(
        number(state, config, comptime!(Field::FoodNeed as u32), household)
            * (config.coverage + config.ring),
        0,
    );
    let excess = positive(
        value(state, config, comptime!(Field::Bread as u32), household),
        reserve,
    );
    let sales = value(state, config, comptime!(Field::Sales as u32), household);
    let quote = number(state, config, comptime!(Field::Price as u32), household);
    let adjustment = max(quote / 20, 1);
    let mut next = quote;
    if lt(excess, sales) {
        next = min(quote + adjustment, config.maximum_price);
    }
    if lt(sales, excess) {
        next = max(max(quote, adjustment) - adjustment, config.minimum_price);
    }
    set32(
        state,
        config,
        comptime!(Field::Price as u32),
        household,
        next,
    );
}

#[cube]
pub(super) fn expand(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
    owner: u32,
) {
    let owned = number(
        state,
        config,
        comptime!(Field::InvestmentFirm as u32),
        owner,
    );
    if owned > 0 {
        let firm = owned - 1;
        let capacity =
            actors[ActorLayout::CAPACITY * (actors.len() / ActorLayout::WORDS) + firm as usize];
        let status = number(
            state,
            config,
            comptime!(Field::EnterpriseStatus as u32),
            firm,
        );
        let operating = status == comptime!(EnterpriseState::Operating as u32)
            && number(
                state,
                config,
                comptime!(Field::EnterpriseModel as u32),
                firm,
            ) == 0;
        let busy = number(state, config, comptime!(Field::Slots as u32), firm) == 0
            && number(state, config, comptime!(Field::Plan as u32), firm) >= capacity;
        let target = times32(
            capacity,
            number(state, config, comptime!(Field::OutputTicks as u32), firm),
        );
        let demand = value(state, config, comptime!(Field::Demand as u32), firm);
        let tools = value(state, config, comptime!(Field::Tools as u32), firm);
        let wood = value(state, config, comptime!(Field::Wood as u32), firm);
        let funded = !lt(
            value(state, config, comptime!(Field::Cash as u32), firm),
            add_checked(
                control,
                value(
                    state,
                    config,
                    comptime!(Field::OperatingReserve as u32),
                    firm,
                ),
                times32(
                    comptime!(match config.property_mode {
                        Feature::Disabled => 720u32,
                        Feature::Enabled =>
                            crate::domain::property::CONSTRUCTION_WORK_PER_SPACE
                                * Good::Construction.initial_price(),
                    }),
                    select(
                        comptime!(config.property_mode == Feature::Enabled),
                        1,
                        number(state, config, comptime!(Field::Wage as u32), firm),
                    ),
                ),
            ),
        );
        if operating
            && busy
            && lt(target, demand)
            && funded
            && capacity < 10_000
            && (comptime!(config.property_mode == Feature::Enabled)
                || (!lt(tools, ticks(2, 0)) && !lt(wood, ticks(5, 0))))
            && property::authorized(state, config, firm, owner)
            && number(
                state,
                config,
                comptime!(Field::ConstructionRemaining as u32),
                firm,
            ) == 0
            && number(
                state,
                config,
                comptime!(Field::PendingCapacity as u32),
                firm,
            ) == 0
        {
            if comptime!(config.property_mode == Feature::Disabled) {
                set(
                    state,
                    config,
                    comptime!(Field::Tools as u32),
                    firm,
                    minus(tools, ticks(2, 0)),
                );
                set(
                    state,
                    config,
                    comptime!(Field::Wood as u32),
                    firm,
                    minus(wood, ticks(5, 0)),
                );
                increase(
                    state,
                    config,
                    control,
                    USED_FIELDS as u32 + comptime!(Good::Tools as u32) - 1,
                    firm,
                    ticks(2, 0),
                );
                increase(
                    state,
                    config,
                    control,
                    USED_FIELDS as u32 + comptime!(Good::Wood as u32) - 1,
                    firm,
                    ticks(5, 0),
                );
                increase(
                    state,
                    config,
                    control,
                    comptime!(Field::Used as u32),
                    firm,
                    ticks(7, 0),
                );
                increase(
                    state,
                    config,
                    control,
                    comptime!(Field::InstalledCapital as u32),
                    firm,
                    ticks(7, 0),
                );
            }
            if comptime!(config.property_mode == Feature::Enabled) {
                set32(
                    state,
                    config,
                    comptime!(Field::RequestedSpace as u32),
                    firm,
                    1,
                );
                set32(
                    state,
                    config,
                    comptime!(Field::ProjectStatus as u32),
                    firm,
                    comptime!(crate::domain::property::ProjectStatus::Requested as u32),
                );
            }
            set32(
                state,
                config,
                comptime!(Field::ConstructionRemaining as u32),
                firm,
                comptime!(match config.property_mode {
                    Feature::Disabled => 720u32,
                    Feature::Enabled => crate::domain::property::CONSTRUCTION_WORK_PER_SPACE,
                }),
            );
            set32(
                state,
                config,
                comptime!(Field::PendingCapacity as u32),
                firm,
                comptime!(match config.property_mode {
                    Feature::Disabled => 1u32,
                    Feature::Enabled => 0u32,
                }),
            );
        }
    }
}

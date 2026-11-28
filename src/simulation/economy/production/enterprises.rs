use crate::domain::enterprise::FoodAccess;
use crate::domain::{
    enterprise::{BusinessScale, EnterpriseModel, EnterpriseState},
    property::CONSTRUCTION_WORK_PER_PLOT,
};
use crate::simulation::economy::configuration::Feature;
use crate::simulation::economy::*;

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
        && economy.ledger.number(Field::FoodAccess, household)
            == comptime!(FoodAccess::MissingSupply as u32);
    let idle = economy.ledger.number(Field::Employed, household) == 0;
    let landless = economy.ledger.number(Field::LandPlots, household) == 0;
    let settled = economy.ledger.number(Field::MoveDue, household) == 0;
    let alive = economy.ledger.number(Field::LivingMembers, household) > 0;
    let mut target = 0u32;
    if comptime!(config.economy == crate::domain::Economy::Industrial) && settled && alive {
        target = crate::simulation::economy::households::mobility::work_target(
            economy, spatial, quotes, household, day, quote_mode,
        );
    }
    let mut nearest = u32::new(comptime!(i64::from(u32::MAX)));
    if target == 0 && needs_food && idle && landless && settled && alive {
        for sample in 0..16u32 {
            let candidate = (household + day * 17 + sample * 31) % config.actors;
            let receiving = !economy
                .actors
                .kind(candidate)
                .is(comptime!(ActorKind::Household))
                || economy.ledger.number(Field::LivingMembers, candidate) > 0;
            let accessible = receiving
                && household_production::offers(
                    &*economy,
                    candidate,
                    comptime!(Good::Bread as u32),
                );
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
pub(in crate::simulation::economy) fn household_quote(
    ledger: &mut Ledger<'_>,
    control: &[Atomic<u32>],
    household: u32,
) {
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
    let quote = pricing::PriceQuote::load(&*ledger, household);
    quote.adjust(ledger, control, multiplier);
}

#[cube]
pub(in crate::simulation::economy) fn expand(
    economy: &mut EconomyView<'_>,
    control: &[Atomic<u32>],
    owner: u32,
) {
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
        let demand = if pricing::distributor(&economy.ledger, firm) {
            market::distribution::handling_demand(&*economy, control, firm)
        } else {
            economy.ledger.value(Field::Demand, firm)
        };
        let funded = !lt(
            economy.ledger.value(Field::Cash, firm),
            plus(
                economy.ledger.value(Field::OperatingReserve, firm),
                ticks(
                    comptime!(CONSTRUCTION_WORK_PER_PLOT * Good::Construction.initial_price()),
                    0,
                ),
            )
            .record(control),
        );
        let interested = expands(&*economy, firm);
        let site = property::PropertySite::new(firm);
        if operating
            && interested
            && economy.actors.kind(firm).bits() != comptime!(Good::Housing as u32)
            && economy.actors.kind(firm).bits() != comptime!(Good::Railway as u32)
            && busy
            && lt(target, demand)
            && funded
            && capacity + added_workers <= 10_000
            && site.authorized(&economy.ledger, owner)
            && !site.has_construction_project(&economy.ledger)
            && economy.ledger.number(Field::PendingCapacity, firm) == 0
        {
            site.request_construction(economy);
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
pub(in crate::simulation::economy) fn expands(economy: &EconomyView<'_>, firm: u32) -> bool {
    let config = economy.ledger.config;
    let kind = economy.actors.kind(firm);
    let industrial_goods = comptime!(
        Good::ALL
            .into_iter()
            .filter(|&good| BusinessScale::industrial_producer(good))
            .fold(0u32, |mask, good| mask | (1 << good as u32))
    );
    let industrial_producer = industrial_goods & (1u32 << kind.output()) != 0
        && economy.ledger.number(Field::EnterpriseModel, firm)
            == comptime!(EnterpriseModel::Producer as u32);
    let workers = economy.actors.capacity(firm);
    let homes = times32(
        economy.ledger.number(Field::OccupiedPlots, firm),
        economy.ledger.number(Field::OutputTicks, firm),
    );
    let capacity = select(
        kind.bits() == comptime!(Good::Housing as u32),
        homes,
        ticks(workers, 0),
    );
    let at_industrial_capacity = !lt(
        capacity,
        ticks(comptime!(BusinessScale::INDUSTRIAL_CAPACITY), 0),
    );
    let draw = economy.actors.expansion_draw(firm);
    draw < select(
        industrial_producer || at_industrial_capacity,
        config.industrial_expansion,
        config.small_business_expansion,
    )
}

#[derive(Clone, Copy)]
enum ProductionRecipe {
    Basic,
    Active,
}

#[derive(CubeType)]
pub(in crate::simulation::economy) struct FirmPolicy<'a> {
    firm: u32,
    control: &'a [Atomic<u32>],
}

#[cube]
impl<'a> FirmPolicy<'a> {
    pub fn new(firm: u32, control: &'a [Atomic<u32>]) -> Self {
        FirmPolicy::<'a> { firm, control }
    }

    fn replacement_cost(&self, ledger: &Ledger<'_>, output: u32) -> Ticks {
        let firm = self.firm;
        let control = self.control;
        let inputs = ledger.number(Field::InputTicks, firm);
        let mut cost = zero();
        #[unroll]
        for input in 0..2u32 {
            if ledger.number_at(comptime!(Field::FirstInput as u32) + input, firm) > 0 {
                cost = plus(
                    cost,
                    times32(
                        inputs,
                        ledger.number_at(comptime!(Field::FirstInputCost as u32) + input, firm),
                    ),
                )
                .record(control);
            }
        }
        if ledger.number(Field::EnterpriseModel, firm) == 1 {
            cost = times32(output, ledger.number(Field::ReplacementCost, firm));
        }
        cost
    }

    pub fn maintain(&self, ledger: &mut Ledger<'_>, packets: u32) {
        let control = self.control;
        let firm = self.firm;
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

    fn retail_viable(&self, ledger: &Ledger<'_>) -> bool {
        let control = self.control;
        let firm = self.firm;
        let config = ledger.config;
        let sales = ledger.value(Field::Sales, firm);
        let replacement = ledger.value_at(
            config.fields - comptime!(crate::domain::RetailAccount::REPLACEMENT_EXPENSE as u32),
            firm,
        );
        let shifts = ledger.number(Field::Plan, firm) - ledger.number(Field::Slots, firm);
        let payroll = times(
            times32(
                ledger.number(Field::Wage, firm),
                ledger.number(Field::LaborMinutes, firm),
            ),
            shifts,
        )
        .record(control);
        let costs = plus(
            plus(replacement, payroll).record(control),
            ticks(config.firm_rent, 0),
        )
        .record(control);
        let revenue = ledger.value(Field::BusinessRevenue, firm);
        !equal(sales, zero()) && !lt(revenue, costs)
    }

    pub fn advance_lifecycle(&self, economy: &mut EconomyView<'_>, day: u32) {
        let firm = self.firm;
        let config = economy.ledger.config;
        let status = economy.ledger.number(Field::EnterpriseStatus, firm);
        let cash = economy.ledger.value(Field::Cash, firm);
        let payroll = times32(
            economy.ledger.number(Field::Wage, firm),
            economy.ledger.number(Field::LaborMinutes, firm),
        );
        let good = economy.actors.kind(firm).output();
        let crop = farming::FarmSite::new(firm, good).is_farm(&economy.ledger);
        let payroll = select(
            crop && !farming::CropCalendar::new(config).on(day).fieldwork(good),
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
                    || self.retail_viable(&economy.ledger)
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

    fn method_output(
        &self,
        economy: &EconomyView<'_>,
        #[comptime] recipe: ProductionRecipe,
    ) -> Ticks {
        let control = self.control;
        let firm = self.firm;
        let recipe = comptime!(match recipe {
            ProductionRecipe::Basic => Field::BasicLaborMinutes as u32,
            ProductionRecipe::Active => Field::LaborMinutes as u32,
        });
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
            divide(plus(desired, ticks(output - 1, 0)).record(control), output),
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
        minimum(times(packets, output).record(control), desired)
    }

    pub fn select_method(&self, economy: &mut EconomyView<'_>) {
        let firm = self.firm;
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
            let upgraded = self.method_output(&*economy, comptime!(ProductionRecipe::Active));
            let basic = self.method_output(&*economy, comptime!(ProductionRecipe::Basic));
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

    pub fn reserve_operating_costs(&self, economy: &mut EconomyView<'_>) {
        let control = self.control;
        let firm = self.firm;
        let config = economy.ledger.config;
        let distributor = economy.ledger.number(Field::EnterpriseModel, firm) == 1;
        if comptime!(config.home_mode == Feature::Enabled) || distributor || config.firm_rent > 0 {
            let labor = economy.ledger.number(Field::Wage, firm)
                * economy.ledger.number(Field::LaborMinutes, firm);
            let output = max(economy.ledger.number(Field::OutputTicks, firm), 1);
            let input_cost = self.replacement_cost(&economy.ledger, output);
            let demand = economy.ledger.value(Field::Demand, firm);
            let forecast_packets = maximum(
                divide(plus(demand, ticks(output - 1, 0)).record(control), output),
                ticks(1, 0),
            );
            let capacity = ticks(economy.actors.capacity(firm), 0);
            let packets = minimum(forecast_packets, capacity);
            let coverage = select(
                distributor,
                economy.ledger.number(Field::StockCoverage, firm),
                config.coverage,
            ) + config.ring;
            let replacement = times(input_cost, coverage).record(control);
            let payroll_days = select(
                farming::FarmSite::new(firm, economy.actors.kind(firm).output())
                    .is_farm(&economy.ledger),
                max(
                    coverage,
                    farming::CropCalendar::new(config)
                        .paid_days(economy.actors.kind(firm).output()),
                ),
                coverage,
            );
            let reserve = times(
                plus(times32(labor, payroll_days), replacement).record(control),
                packets.extract(LOW_WORD),
            )
            .record(control);
            let construction = times32(
                economy.ledger.number(Field::ConstructionRemaining, firm),
                comptime!(Good::Construction.initial_price()),
            );
            let reserve = plus(
                plus(reserve, construction).record(control),
                rent::obligation(&economy.ledger, control, firm),
            )
            .record(control);
            economy.ledger.set(Field::OperatingReserve, firm, reserve);
        }
    }

    pub fn purchasing_power(&self, economy: &EconomyView<'_>) -> Ticks {
        let control = self.control;
        let firm = self.firm;
        let config = economy.ledger.config;
        let cash = economy.ledger.value(Field::Cash, firm);
        let distributor = economy.ledger.number(Field::EnterpriseModel, firm) == 1;
        let mut available = cash;
        if !economy
            .actors
            .kind(firm)
            .is(comptime!(ActorKind::Household))
            && (comptime!(config.home_mode == Feature::Enabled) || distributor)
        {
            let labor = times32(
                economy.ledger.number(Field::Wage, firm),
                economy.ledger.number(Field::LaborMinutes, firm),
            );
            let output = max(economy.ledger.number(Field::OutputTicks, firm), 1);
            let replacement = self.replacement_cost(&economy.ledger, output);
            let packet_cost = plus(labor, replacement).record(control);
            if packet_cost.extract(HIGH_WORD) > 0 {
                KernelFaults::new(comptime!(FaultFlags::ARITHMETIC_OVERFLOW)).record(control);
            }
            let demand = economy.ledger.value(Field::Demand, firm);
            let forecast = maximum(
                divide(plus(demand, ticks(output - 1, 0)).record(control), output),
                ticks(1, 0),
            );
            let funded = maximum(
                divide(cash, max(packet_cost.extract(LOW_WORD), 1)),
                ticks(1, 0),
            );
            let capacity = ticks(economy.actors.capacity(firm), 0);
            let packets = minimum(minimum(forecast, funded), capacity);
            let mut payroll = times(labor, packets.extract(LOW_WORD)).record(control);
            if distributor {
                let days = economy.ledger.number(Field::StockCoverage, firm) + config.ring;
                let stock = economy
                    .ledger
                    .stock_with_transit(economy.actors.kind(firm).output(), firm)
                    .record(control);
                let handling = CheckedTicks::ceiling_ratio(stock, 1, output).record(control);
                let shifts = maximum(handling, times(packets, days).record(control));
                payroll = plus(
                    times(shifts, labor.extract(LOW_WORD)).record(control),
                    times32(days - 1, config.firm_rent),
                )
                .record(control);
            }
            available = positive(cash, payroll);
        }
        if !economy
            .actors
            .kind(firm)
            .is(comptime!(ActorKind::Household))
        {
            available = positive(available, rent::obligation(&economy.ledger, control, firm));
        }
        available
    }
}

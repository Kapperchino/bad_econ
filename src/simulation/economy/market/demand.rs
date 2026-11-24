use crate::domain::enterprise::FoodAccess;
use crate::simulation::economy::*;

#[derive(CubeType)]
pub(super) struct FoodSearchResult {
    pub seller: u32,
    pub seen_supply: bool,
    pub used_fallback: bool,
}

#[cube]
impl FoodSearchResult {
    pub(super) fn record(&self, ledger: &mut Ledger<'_>, buyer: u32, wanted: Ticks, budget: Ticks) {
        let access = if self.seller == NO_ACTOR && !equal(wanted, zero()) {
            select(
                self.seen_supply || equal(budget, zero()),
                comptime!(FoodAccess::Unaffordable as u32),
                comptime!(FoodAccess::MissingSupply as u32),
            )
        } else {
            select(
                self.used_fallback,
                comptime!(FoodAccess::MissingSupply as u32),
                comptime!(FoodAccess::Satisfied as u32),
            )
        };
        ledger.set_number(Field::FoodAccess, buyer, access);
    }
}

#[derive(CubeType)]
pub(in crate::simulation::economy) struct PurchaseRequest {
    pub search: bool,
    pub wanted: Ticks,
    pub purchasing_power: Ticks,
}

#[cube]
impl PurchaseRequest {
    pub fn new(
        economy: &mut EconomyView<'_>,
        spatial: &[u32],
        control: &[Atomic<u32>],
        parameters: &EconomyParameters,
        id: u32,
    ) -> Self {
        let config = economy.ledger.config;
        let kind = economy.actors.kind(id);
        let good = parameters.traded_good();
        let cash = economy.ledger.value(Field::Cash, id);
        let mut order = PurchaseRequest {
            search: false,
            wanted: zero(),
            purchasing_power: cash,
        };
        match comptime!(parameters.market) {
            MarketFlag::Input(InputMarket::Construction) => {
                let ledger = &economy.ledger;
                let project = property::PropertySite::new(id).construction_due(ledger);
                let maintenance = select(
                    kind.output() == comptime!(Good::Housing as u32),
                    landlords::maintenance_due(ledger, id),
                    zero(),
                );
                order.wanted = plus(project, maintenance).record(control);
                order.search = !equal(order.wanted, zero());
                order.purchasing_power =
                    enterprises::FirmPolicy::new(id, control).purchasing_power(&*economy);
            }
            MarketFlag::Input(_) => {
                order.prepare_input(economy, control, id, &kind, good, parameters.day);
            }
            MarketFlag::Consumer(_) => {
                let purchase = consumption::HouseholdPurchase::new(
                    &*economy, spatial, control, parameters, id,
                );
                purchase.prepare(&mut order, good);
            }
            MarketFlag::Household => {
                order.search = kind.is(comptime!(ActorKind::Household));
                let ledger = &economy.ledger;
                let ration = food::HouseholdRation::new(ledger, id);
                let days = config.coverage + config.ring - 1;
                order.wanted = times32(ration.purchase(config, parameters), days);
                if food::HomeKitchen::available(ledger, id) {
                    let flour = ledger
                        .stock_with_transit(comptime!(Good::Flour as u32), id)
                        .record(control);
                    let usable = minimum(
                        minimum(order.wanted, flour),
                        food::HomeKitchen::flour_capacity(ledger, id, days),
                    );
                    let bakeable = food::HomeKitchen::new(ledger, id).bread_from_flour(usable);
                    order.wanted =
                        maximum(ticks(ration.minimum(), 0), positive(order.wanted, bakeable));
                }
            }
            _ => (),
        }
        if good > 0 {
            let held = economy.ledger.stock_with_transit(good, id).record(control);
            let gap = positive(order.wanted, held);
            if comptime!(
                config.food_income == crate::domain::food::FoodIncome::PurchasingPower
                    && matches!(
                        parameters.market,
                        MarketFlag::Input(_) | MarketFlag::Household
                    )
            ) && kind.is(comptime!(ActorKind::Household))
                && (good == Good::Flour as u32 || good == Good::Bread as u32)
            {
                let daily =
                    food::HouseholdRation::new(&economy.ledger, id).purchase(config, parameters);
                let cycle = times32(daily, config.coverage);
                let reorder_gap = if good == Good::Flour as u32 {
                    food::HomeKitchen::new(&economy.ledger, id).flour_for_bread(control, cycle)
                } else {
                    cycle
                };
                order.search = order.search && (lt(reorder_gap, gap) || equal(held, zero()));
            }
            order.wanted = gap;
            if comptime!(parameters.market == MarketFlag::Input(InputMarket::Grain)) {
                let dealer = pricing::distributor(&economy.ledger, id);
                let mill = comptime!(config.economy == crate::domain::Economy::Industrial)
                    && kind.output() == comptime!(Good::Flour as u32)
                    && !dealer
                    && economy.ledger.number(Field::FirstInput, id) == good;
                let units = select(
                    dealer,
                    economy.ledger.number(Field::OutputTicks, id),
                    economy.ledger.number(Field::InputTicks, id),
                );
                let replenishment_days = select(mill, 2u32, 1u32);
                let capacity = times32(economy.actors.capacity(id), units);
                let replenishment = times(capacity, replenishment_days).record(control);
                let rounds = config.grain_rounds;
                let lot = divide(
                    plus(
                        replenishment,
                        ticks(rounds - 1 - parameters.matching_round, 0),
                    )
                    .record(control),
                    rounds,
                );
                order.wanted = select(dealer || mill, minimum(order.wanted, lot), order.wanted);
            }
            order.purchasing_power = positive(
                order.purchasing_power,
                economy.ledger.value(Field::CartagePayable, id),
            );
            order.search = order.search
                && !equal(order.wanted, zero())
                && !equal(order.purchasing_power, zero());
        }
        order
    }

    fn prepare_input(
        &mut self,
        economy: &mut EconomyView<'_>,
        control: &[Atomic<u32>],
        id: u32,
        k: &KernelActorKind,
        good: u32,
        day: u32,
    ) {
        self.purchasing_power =
            enterprises::FirmPolicy::new(id, control).purchasing_power(&*economy);
        let config = economy.ledger.config;
        let pricing = pricing::FirmPricing::new(id, k.output(), control);
        let mut search = !k.is(comptime!(ActorKind::Household))
            && (economy.ledger.number(Field::FirstInput, id) == good
                || economy.ledger.number(Field::SecondInput, id) == good);
        let mut wanted = ticks(
            economy.actors.capacity(id)
                * economy.ledger.number(Field::InputTicks, id)
                * consumption::input_coverage(config, k.bits()),
            0u32,
        );
        if economy.ledger.number(Field::EnterpriseModel, id) == 1 {
            let secondary = economy.ledger.value_at(
                pricing::retail_field(
                    config,
                    good,
                    comptime!(crate::domain::RetailAccount::Demand),
                ),
                id,
            );
            search = good == k.output() || !equal(secondary, zero());
            let forecast = maximum(economy.ledger.value(Field::Demand, id), ticks(1, 0));
            wanted = if good == k.output() {
                pricing.distributor_target(&*economy, forecast, day)
            } else {
                times(
                    secondary,
                    min(
                        config.coverage,
                        economy.ledger.number(Field::StockCoverage, id),
                    ) + config.ring,
                )
                .record(control)
            };
        }
        if !k.is(comptime!(ActorKind::Household))
            && economy.ledger.number(Field::EnterpriseModel, id) == 0
            && comptime!(config.home_mode == Feature::Enabled)
        {
            let horizon = config.coverage + config.ring;
            let packets = divide(
                plus(
                    times(economy.ledger.value(Field::Demand, id), horizon).record(control),
                    ticks(economy.ledger.number(Field::OutputTicks, id) - 1, 0),
                )
                .record(control),
                max(economy.ledger.number(Field::OutputTicks, id), 1),
            );
            let capacity = economy.actors.capacity(id);
            let profitable = comptime!(config.economy != crate::domain::Economy::Industrial)
                || config.price_response == 0
                || pricing.covers_batch(&economy.ledger);
            wanted = times(
                minimum(
                    maximum(
                        select(profitable, packets, zero()),
                        trial_packet(&economy.ledger, id, k.output()),
                    ),
                    times32(capacity, horizon),
                ),
                economy.ledger.number(Field::InputTicks, id),
            )
            .record(control);
        }
        if k.is(comptime!(ActorKind::Household))
            && (good == comptime!(Good::Grain as u32) || good == comptime!(Good::Flour as u32))
            && food::HomeKitchen::available(&economy.ledger, id)
            && economy.ledger.number(Field::LivingMembers, id) > 0
        {
            search = true;
            wanted = if good == comptime!(Good::Grain as u32) {
                household_production::seed_reserve(&economy.ledger, control, id)
            } else {
                let days = config.coverage + config.ring - 1;
                let bread = economy
                    .ledger
                    .stock_with_transit(comptime!(Good::Bread as u32), id)
                    .record(control);
                let needed = positive(
                    times32(economy.ledger.number(Field::FoodNeed, id), days),
                    bread,
                );
                let flour =
                    food::HomeKitchen::new(&economy.ledger, id).flour_for_bread(control, needed);
                minimum(
                    flour,
                    food::HomeKitchen::flour_capacity(&economy.ledger, id, days),
                )
            };
        }
        if !k.is(comptime!(ActorKind::Household))
            && comptime!(
                config.home_mode == Feature::Enabled && config.property_mode == Feature::Disabled
            )
            && good == comptime!(Good::Wood as u32)
            && economy.ledger.number(Field::Slots, id) == 0
            && economy.ledger.number(Field::Plan, id) > 0
            && !lt(economy.ledger.value(Field::Cash, id), ticks(10_000, 0))
        {
            wanted = select(search, maximum(wanted, ticks(5, 0)), ticks(5, 0));
            search = true;
        }
        if !k.is(comptime!(ActorKind::Household))
            && k.bits() != comptime!(Good::Railway as u32)
            && good == comptime!(Good::Tools as u32)
            && comptime!(config.home_mode == Feature::Enabled)
        {
            wanted = select(search, maximum(wanted, ticks(2, 0)), ticks(2, 0));
            search = true;
        }
        let rail = railway::demand(economy, id, good);
        self.search = search || !equal(rail, zero());
        self.wanted = maximum(wanted, rail);
    }
}

#[cube]
pub(in crate::simulation::economy) fn trial_packet(
    ledger: &Ledger<'_>,
    firm: u32,
    good: u32,
) -> Ticks {
    let config = ledger.config;
    let empty = comptime!(
        config.economy == crate::domain::Economy::Industrial
            && config.home_mode == Feature::Enabled
    ) && ledger.number(Field::EnterpriseModel, firm) == 0
        && ledger.number(Field::EnterpriseStatus, firm)
            == comptime!(crate::domain::enterprise::EnterpriseState::Operating as u32)
        && !farming::FarmSite::new(firm, good).is_farm(ledger)
        && equal(ledger.value_at(good, firm), zero())
        && equal(ledger.value(Field::Pipeline, firm), zero());
    ticks(u32::cast_from(empty), 0)
}

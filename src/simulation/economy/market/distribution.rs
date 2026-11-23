use crate::domain::RetailAccount;
use crate::simulation::economy::*;

#[cube]
pub(in crate::simulation::economy) fn open(
    economy: &mut EconomyView<'_>,
    control: &[Atomic<u32>],
    firm: u32,
) {
    let config = economy.ledger.config;
    let output = economy.actors.kind(firm).output();
    let ledger = &mut economy.ledger;
    for field in [
        RetailAccount::PRIOR_SALES as u32,
        RetailAccount::PRIOR_UNFILLED as u32,
        RetailAccount::REPLACEMENT_EXPENSE as u32,
    ] {
        ledger.set_number_at(config.fields - field, firm, 0);
    }
    ledger.set_number(Field::BusinessRevenue, firm, 0);
    let units = max(ledger.number(Field::OutputTicks, firm), 1);
    let handling = CheckedTicks::ceiling_ratio(
        times32(
            ledger.number(Field::Wage, firm),
            ledger.number(Field::LaborMinutes, firm),
        ),
        1,
        units,
    )
    .record(control);
    for good in 1..Good::COUNT as u32 + 1 {
        let orders = pricing::retail_field(config, good, comptime!(RetailAccount::Orders));
        ledger.set_number_at(orders, firm, 0);
        if good != output {
            let cost = pricing::retail_field(config, good, comptime!(RetailAccount::Cost));
            let paid = ledger.number_at(cost, firm);
            let replacement = select(paid == 0, config.opening_prices[good as usize - 1], paid);
            let basis = plus(ticks(replacement, 0), handling).record(control);
            let quote = CheckedTicks::ceiling_ratio(
                basis,
                1000 + ledger.number(Field::RetailMargin, firm),
                1000,
            )
            .record(control);
            let field = pricing::retail_field(config, good, comptime!(RetailAccount::Quote));
            ledger.set_at(
                field,
                firm,
                maximum(
                    ticks(config.minimum_price, 0),
                    minimum(quote, ticks(config.maximum_price, 0)),
                ),
            );
        }
    }
}

#[cube]
pub(in crate::simulation::economy) fn record(
    economy: &mut EconomyView<'_>,
    feedback: &[Atomic<u32>],
    good: u32,
    firm: u32,
) {
    let config = economy.ledger.config;
    if firm < config.actors && pricing::distributor(&economy.ledger, firm) {
        let output = economy.actors.kind(firm).output();
        let terms = household_production::TradeFields::new(&economy.actors, config, firm, good);
        let ledger = &mut economy.ledger;
        let sales = ledger.value(Field::Sales, firm);
        let prior_sales = config.fields - comptime!(RetailAccount::PRIOR_SALES as u32);
        let units = minus(sales, ledger.value_at(prior_sales, firm));
        ledger.set_at(prior_sales, firm, sales);
        let base = CONTROL_WORDS as u32 + firm * TICK_WORDS as u32;
        let unfilled = ticks(
            feedback[base as usize].load(),
            feedback[(base + 1) as usize].load(),
        );
        let prior_unfilled = config.fields - comptime!(RetailAccount::PRIOR_UNFILLED as u32);
        let rejected = minus(unfilled, ledger.value_at(prior_unfilled, firm));
        ledger.set_at(prior_unfilled, firm, unfilled);
        let orders = pricing::retail_field(config, good, comptime!(RetailAccount::Orders));
        ledger.increase_at(
            feedback,
            orders,
            firm,
            plus(units, rejected).record(feedback),
        );
        let price = ledger.number_at(terms.price, firm);
        let cost = ledger.number_at(pricing::cost_field(config, output, good), firm);
        ledger.increase(
            feedback,
            Field::BusinessRevenue,
            firm,
            times(units, price).record(feedback),
        );
        ledger.increase_at(
            feedback,
            config.fields - comptime!(RetailAccount::REPLACEMENT_EXPENSE as u32),
            firm,
            times(units, cost).record(feedback),
        );
    }
}

#[cube]
pub(in crate::simulation::economy) fn handling_demand(
    economy: &EconomyView<'_>,
    control: &[Atomic<u32>],
    firm: u32,
    primary: Ticks,
) -> Ticks {
    let ledger = &economy.ledger;
    let mut total = primary;
    for good in 1..Good::COUNT as u32 + 1 {
        if good != economy.actors.kind(firm).output() {
            let forecast = ledger.value_at(
                pricing::retail_field(ledger.config, good, comptime!(RetailAccount::Demand)),
                firm,
            );
            let stock = ledger.value_at(good, firm);
            total = plus(total, minimum(stock, maximum(forecast, ticks(1, 0)))).record(control);
        }
    }
    total
}

#[cube]
pub(in crate::simulation::economy) fn close(
    economy: &mut EconomyView<'_>,
    control: &[Atomic<u32>],
    firm: u32,
) {
    let config = economy.ledger.config;
    let capacity = times32(
        economy.actors.capacity(firm),
        economy.ledger.number(Field::OutputTicks, firm),
    );
    for good in 1..Good::COUNT as u32 + 1 {
        let field = pricing::retail_field(config, good, comptime!(RetailAccount::Demand));
        let requested = economy.ledger.value_at(
            pricing::retail_field(config, good, comptime!(RetailAccount::Orders)),
            firm,
        );
        let observed = minimum(requested, capacity);
        let prior = economy.ledger.value_at(field, firm);
        let forecast = KernelObservation {
            kind: comptime!(ObservationKind::Observed as u32),
            value: observed,
        }
        .blend(prior, max(config.forecast_weight, 1), control);
        economy
            .ledger
            .set_at(field, firm, maximum(observed, forecast));
    }
}

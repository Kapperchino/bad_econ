use crate::domain::RetailAccount;
use crate::simulation::economy::*;

#[repr(u32)]
enum RetailQuoteRule {
    Carry,
    Discount,
    CoverCost,
}

#[derive(CubeType)]
struct RetailTerms {
    firm: u32,
    output: u32,
    handling: Ticks,
    margin: u32,
}

#[cube]
impl RetailTerms {
    fn new(economy: &EconomyView<'_>, control: &[Atomic<u32>], firm: u32) -> Self {
        let ledger = &economy.ledger;
        let payroll = times32(
            ledger.number(Field::Wage, firm),
            ledger.number(Field::LaborMinutes, firm),
        );
        RetailTerms {
            firm,
            output: economy.actors.kind(firm).output(),
            handling: CheckedTicks::ceiling_ratio(
                payroll,
                1,
                max(ledger.number(Field::OutputTicks, firm), 1),
            )
            .record(control),
            margin: ledger.number(Field::RetailMargin, firm),
        }
    }

    fn asking_price(&self, ledger: &Ledger<'_>, control: &[Atomic<u32>], good: u32) -> u32 {
        let config = ledger.config;
        let cost = ledger.number_at(pricing::cost_field(config, self.output, good), self.firm);
        let replacement = select(cost == 0, config.opening_price(good - 1), cost);
        let basis = plus(ticks(replacement, 0), self.handling).record(control);
        let quote = CheckedTicks::ceiling_ratio(basis, POLICY_SCALE + self.margin, POLICY_SCALE)
            .record(control);
        minimum(
            maximum(ticks(config.minimum_price, 0), quote),
            ticks(config.maximum_price, 0),
        )
        .extract(LOW_WORD)
    }

    fn reprice(
        &self,
        ledger: &mut Ledger<'_>,
        control: &[Atomic<u32>],
        good: u32,
        orders: Ticks,
        target: Ticks,
        demand: Ticks,
    ) {
        let config = ledger.config;
        let firm = self.firm;
        let stock = ledger.value_at(good, firm);
        let field = pricing::retail_field(config, good, comptime!(RetailAccount::Quote));
        let price = ledger.number_at(field, firm);
        if config.price_response > 0 && price > 0 {
            let stale = lt(target, stock) && lt(demand, ledger.value(Field::OutputTicks, firm));
            let rule = select(
                equal(orders, zero()),
                select(
                    stale,
                    comptime!(RetailQuoteRule::Discount as u32),
                    comptime!(RetailQuoteRule::Carry as u32),
                ),
                comptime!(RetailQuoteRule::CoverCost as u32),
            );
            let next = match rule {
                1 => {
                    let step = ceildiv(price * config.price_response, 20 * POLICY_SCALE);
                    max(price, config.minimum_price + step) - step
                }
                2 => {
                    let goal = self.asking_price(&*ledger, control, good);
                    let movement = ceildiv(
                        (max(goal, price) - min(goal, price)) * config.price_response,
                        POLICY_SCALE,
                    );
                    let adjusted = select(goal > price, price + movement, price - movement);
                    let cost = ledger.number_at(
                        pricing::retail_field(config, good, comptime!(RetailAccount::Cost)),
                        firm,
                    );
                    max(adjusted, min(cost, config.maximum_price))
                }
                _ => price,
            };
            ledger.set_number_at(field, firm, next);
        }
    }
}

#[cube]
pub(in crate::simulation::economy) fn open(
    economy: &mut EconomyView<'_>,
    control: &[Atomic<u32>],
    firm: u32,
) {
    let terms = RetailTerms::new(&*economy, control, firm);
    let ledger = &mut economy.ledger;
    let config = ledger.config;
    for field in [
        RetailAccount::PRIOR_SALES as u32,
        RetailAccount::PRIOR_UNFILLED as u32,
        RetailAccount::REPLACEMENT_EXPENSE as u32,
    ] {
        ledger.set_number_at(config.fields - field, firm, 0);
    }
    ledger.set_number(Field::BusinessRevenue, firm, 0);
    ledger.set_number(Field::PrimaryRetailUnfilled, firm, 0);
    for good in 1..Good::COUNT as u32 + 1 {
        let orders = pricing::retail_field(config, good, comptime!(RetailAccount::Orders));
        ledger.set_number_at(orders, firm, 0);
        let field = pricing::retail_field(config, good, comptime!(RetailAccount::Quote));
        if good != terms.output && ledger.number_at(field, firm) == 0 {
            let initial = terms.asking_price(&*ledger, control, good);
            ledger.set_number_at(field, firm, initial);
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
        if good == output {
            ledger.increase(feedback, Field::PrimaryRetailUnfilled, firm, rejected);
        }
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
) -> Ticks {
    let ledger = &economy.ledger;
    let primary = economy.actors.kind(firm).output();
    let stock = ledger.value_at(primary, firm);
    let mut demand = maximum(ledger.value(Field::Demand, firm), ticks(1, 0));
    if primary == comptime!(Good::Grain as u32) {
        let days = max(
            min(
                ledger.number(Field::StockCoverage, firm),
                ledger.config.coverage,
            ),
            1,
        );
        let daily_stock = CheckedTicks::ceiling_ratio(stock, 1, days).record(control);
        demand = maximum(demand, daily_stock);
    }
    let mut total = minimum(stock, demand);
    for good in 1..Good::COUNT as u32 + 1 {
        if good != primary {
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
    let terms = RetailTerms::new(&*economy, control, firm);
    let coverage = min(
        config.coverage,
        economy.ledger.number(Field::StockCoverage, firm),
    ) + config.ring;
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
        let demand = maximum(observed, forecast);
        economy.ledger.set_at(field, firm, demand);
        if good != terms.output {
            let target = times(demand, coverage).record(control);
            terms.reprice(
                &mut economy.ledger,
                control,
                good,
                requested,
                target,
                demand,
            );
        }
    }
}

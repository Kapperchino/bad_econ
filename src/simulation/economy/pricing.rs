use super::*;
use crate::domain::enterprise::{EnterpriseModel, EnterpriseState};

#[repr(u32)]
enum InventoryPolicy {
    Producer,
    Distributor,
    Seasonal,
}

#[repr(u32)]
enum QuoteRule {
    Inventory,
    DiscountToCost,
    CostPlusMargin,
    Scarcity,
}

#[derive(CubeType)]
struct PacketCost {
    cost: Ticks,
}

#[cube]
impl PacketCost {
    fn new(ledger: &Ledger<'_>, control: &[Atomic<u32>], firm: u32, good: u32) -> Self {
        let config = ledger.config;
        let payroll = times32(
            ledger.number(Field::Wage, firm),
            ledger.number(Field::LaborMinutes, firm),
        );
        let mut cost = select(
            farming::farm(ledger, firm, good) && config.home.harvest_days < 365,
            divide(
                multiply(control, payroll, farming::paid_days(config, good)),
                365,
            ),
            payroll,
        );
        for input in 0..2u32 {
            if ledger.number_at(comptime!(Field::FirstInput as u32) + input, firm) > 0 {
                cost = add_checked(
                    control,
                    cost,
                    times32(
                        ledger.number(Field::InputTicks, firm),
                        ledger.number_at(comptime!(Field::FirstInputCost as u32) + input, firm),
                    ),
                );
            }
        }
        if ledger.number(Field::EnterpriseModel, firm)
            == comptime!(EnterpriseModel::Distributor as u32)
        {
            cost = add_checked(
                control,
                cost,
                times32(
                    ledger.number(Field::OutputTicks, firm),
                    ledger.number(Field::ReplacementCost, firm),
                ),
            );
        }
        PacketCost { cost }
    }
}

#[cube]
pub(super) fn covers_batch(
    ledger: &Ledger<'_>,
    control: &[Atomic<u32>],
    firm: u32,
    good: u32,
) -> bool {
    let revenue = times32(
        ledger.number(Field::Price, firm),
        ledger.number(Field::OutputTicks, firm),
    );
    !lt(revenue, PacketCost::new(ledger, control, firm, good).cost)
}

#[cube]
fn quote_rule(ledger: &Ledger<'_>, firm: u32, excess: Ticks) -> u32 {
    if equal(excess, zero()) {
        let no_orders = equal(ledger.value(Field::Sales, firm), zero())
            && equal(ledger.value(Field::Unfilled, firm), zero());
        select(
            no_orders,
            comptime!(QuoteRule::CostPlusMargin as u32),
            comptime!(QuoteRule::Scarcity as u32),
        )
    } else {
        let manufacturer = ledger.number(Field::FirstInput, firm) > 0
            && ledger.number(Field::EnterpriseStatus, firm)
                == comptime!(EnterpriseState::Operating as u32);
        select(
            manufacturer,
            comptime!(QuoteRule::DiscountToCost as u32),
            comptime!(QuoteRule::Inventory as u32),
        )
    }
}

#[cube]
pub(super) fn response(
    ledger: &Ledger<'_>,
    control: &[Atomic<u32>],
    firm: u32,
    good: u32,
    excess: Ticks,
    inventory_response: u32,
) -> u32 {
    let config = ledger.config;
    let mut response = inventory_response;
    let repricing = quote_rule(ledger, firm, excess);
    if comptime!(config.economy == crate::domain::Economy::Industrial)
        && comptime!(config.home_mode == Feature::Enabled)
        && config.price_response > 0
        && repricing != comptime!(QuoteRule::Inventory as u32)
    {
        let cost = PacketCost::new(ledger, control, firm, good).cost;
        let target = select(
            repricing == comptime!(QuoteRule::DiscountToCost as u32),
            cost,
            divide(multiply(control, cost, 5), 4),
        );
        let output = max(ledger.number(Field::OutputTicks, firm), 1);
        let quote = minimum(
            divide(add_checked(control, target, ticks(output - 1, 0)), output),
            ticks(config.maximum_price, 0),
        );
        let price = max(ledger.number(Field::Price, firm), 1);
        let next = add_checked(
            control,
            times32(price, POLICY_SCALE - config.price_response),
            multiply(control, quote, config.price_response),
        );
        let correction = minimum(
            divide(multiply(control, next, PRICE_SCALE / POLICY_SCALE), price),
            ticks(u32::MAX, 0),
        )
        .extract(LOW_WORD);
        response = match repricing {
            1 => max(inventory_response, min(correction, PRICE_SCALE)),
            2 => correction,
            _ => max(inventory_response, correction),
        };
    }
    response
}

#[cube]
pub(super) fn adjust(
    ledger: &mut Ledger<'_>,
    control: &[Atomic<u32>],
    actor: u32,
    multiplier: u32,
) {
    let config = ledger.config;
    let price = ledger.value(Field::Price, actor);
    let previous = ledger.value(Field::PriceQuote, actor);
    let rounded = divide(
        add_checked(control, previous, ticks(PRICE_SCALE / 2, 0)),
        PRICE_SCALE,
    );
    let quote = select(
        equal(rounded, price),
        previous,
        multiply(control, price, PRICE_SCALE),
    );
    let next = divide(multiply(control, quote, multiplier), PRICE_SCALE);
    let next = minimum(
        maximum(next, times32(config.minimum_price, PRICE_SCALE)),
        times32(config.maximum_price, PRICE_SCALE),
    );
    ledger.set(Field::PriceQuote, actor, next);
    ledger.set(
        Field::Price,
        actor,
        divide(
            add_checked(control, next, ticks(PRICE_SCALE / 2, 0)),
            PRICE_SCALE,
        ),
    );
}

#[cube]
pub(super) fn distributor_stock_target(
    economy: &EconomyView<'_>,
    control: &[Atomic<u32>],
    actor: u32,
    demand: Ticks,
) -> Ticks {
    let config = economy.ledger.config;
    let mut daily = demand;
    if economy.actors.kind(actor).output() == comptime!(Good::Grain as u32) {
        let shifts = economy.actors.capacity(actor);
        let units = economy.ledger.number(Field::OutputTicks, actor);
        daily = minimum(demand, times32(shifts, units));
    }
    multiply(
        control,
        daily,
        economy.ledger.number(Field::StockCoverage, actor) + config.ring,
    )
}

#[cube]
pub(super) fn inventory_target(
    economy: &EconomyView<'_>,
    control: &[Atomic<u32>],
    firm: u32,
    good: u32,
    demand: Ticks,
    day: u32,
) -> Ticks {
    let config = economy.ledger.config;
    let policy = select(
        farming::farm(&economy.ledger, firm, good),
        comptime!(InventoryPolicy::Seasonal as u32),
        select(
            economy.ledger.number(Field::EnterpriseModel, firm)
                == comptime!(EnterpriseModel::Distributor as u32),
            comptime!(InventoryPolicy::Distributor as u32),
            comptime!(InventoryPolicy::Producer as u32),
        ),
    );
    match policy {
        0 => {
            let daily = select(
                good <= comptime!(Good::Bread as u32),
                farming::production_target(control, demand),
                demand,
            );
            let batch = economy.ledger.number(Field::OutputTicks, firm);
            add_checked(
                control,
                multiply(control, daily, config.coverage),
                ticks(max(batch, 1) - 1, 0),
            )
        }
        1 => distributor_stock_target(economy, control, firm, demand),
        _ => farming::inventory_target(config, control, demand, day),
    }
}

#[cube]
pub(super) fn stocked(ledger: &Ledger<'_>, actor: u32, good: u32) -> bool {
    good == comptime!(Good::Housing as u32)
        || good == comptime!(Good::Railway as u32)
        || !equal(ledger.value_at(good, actor), zero())
}

#[cube]
pub(super) fn refresh(
    economy: &mut EconomyView<'_>,
    spatial: &mut [u32],
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
    buyer: u32,
) {
    let config = economy.ledger.config;
    if buyer < config.actors {
        let active = !economy
            .actors
            .kind(buyer)
            .is(comptime!(ActorKind::Household))
            && economy.ledger.number(Field::EnterpriseStatus, buyer)
                < comptime!(EnterpriseState::Liquidating as u32);
        if active {
            for input in 0..3u32 {
                let field = match input {
                    0 | 1 => comptime!(Field::FirstInputCost as u32) + input,
                    _ => u32::new(comptime!(Field::ReplacementCost as i64)),
                };
                let good = match input {
                    0 | 1 => economy
                        .ledger
                        .number_at(comptime!(Field::FirstInput as u32) + input, buyer),
                    _ => {
                        let distributor = economy.ledger.number(Field::EnterpriseModel, buyer)
                            == comptime!(EnterpriseModel::Distributor as u32);
                        select(distributor, economy.actors.kind(buyer).output(), 0u32)
                    }
                };
                if good > 0 {
                    let stock = add_checked(
                        control,
                        economy.ledger.value_at(good, buyer),
                        economy
                            .ledger
                            .value_at(TRANSIT_FIELDS as u32 + good - 1, buyer),
                    );
                    let raw_material = input < 2
                        && comptime!(config.economy == crate::domain::Economy::Industrial)
                        && !lt(
                            stock,
                            ticks(economy.ledger.number(Field::InputTicks, buyer), 0),
                        )
                        && economy.ledger.number_at(field, buyer) > 0;
                    if !raw_material {
                        let current = replacement_quote(
                            &*economy, &*spatial, control, parameters, buyer, good,
                        );
                        if current > 0 {
                            economy.ledger.set_number_at(field, buyer, current);
                        }
                    }
                }
            }
        }
    }
}

#[cube]
pub(super) fn replacement_quote(
    economy: &EconomyView<'_>,
    spatial: &[u32],
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
    buyer: u32,
    good: u32,
) -> u32 {
    let config = economy.ledger.config;
    let side = config.spatial_side;
    let cells = side * side;
    let grain = good == comptime!(Good::Grain as u32);
    let household_grain = comptime!(config.home_mode == Feature::Enabled)
        && grain
        && economy.ledger.number(Field::EnterpriseModel, buyer)
            == comptime!(EnterpriseModel::Distributor as u32);
    let first_segment = select(household_grain, 1u32, 2u32);
    let x = economy.actors.x(buyer);
    let y = economy.actors.y(buyer);
    let cx = x / config.search_radius;
    let cy = y / config.search_radius;
    let mut quote = 0u32;
    let mut best = ticks(u32::MAX, u32::MAX);
    let mut row = max(cy, 1) - 1;
    while row <= min(cy + 1, side - 1) {
        let mut column = max(cx, 1) - 1;
        while column <= min(cx + 1, side - 1) {
            let cell = row * side + column;
            for group in first_segment..3u32 {
                let segment = group * (cells + 1 + config.actors);
                let begin = spatial[(segment + cell) as usize];
                let length = spatial[(segment + cell + 1) as usize] - begin;
                let samples = min(length, HOUSEHOLD_SEARCH_SAMPLES);
                for sample in 0..samples {
                    let offset =
                        sample * (length / samples) + sample * (length % samples) / samples;
                    let rotated = (offset + parameters.day + buyer) % length;
                    let seller = spatial[(segment + cells + 1 + begin + rotated) as usize];
                    let k = economy.actors.kind(seller);
                    let relevant = seller != buyer
                        && select(
                            group == 1,
                            k.is(comptime!(ActorKind::Household)),
                            k.output() == good,
                        );
                    if relevant {
                        let status = economy.ledger.number(Field::EnterpriseStatus, seller);
                        let trip = economy.actors.distance(buyer, seller);
                        let terms =
                            household_production::TradeFields::new(&economy.actors, seller, good);
                        let price = economy.ledger.number_at(terms.price, seller);
                        let supplier = if grain {
                            household_production::grain_counterparty(economy, buyer, seller)
                        } else {
                            economy.ledger.number(Field::EnterpriseModel, seller)
                                == comptime!(EnterpriseModel::Producer as u32)
                        };
                        let available = status < comptime!(EnterpriseState::Liquidating as u32)
                            && supplier
                            && household_production::offers(economy, seller, good)
                            && trip <= config.search_radius
                            && price > 0;
                        if available {
                            let days = max(ceildiv(trip, config.shipment_speed), 1);
                            let score = add_checked(
                                control,
                                ticks(price, 0),
                                times32(days, config.delivery_penalty),
                            );
                            if lt(score, best) {
                                quote = price;
                                best = score;
                            }
                        }
                    }
                }
            }
            column += 1;
        }
        row += 1;
    }
    quote
}

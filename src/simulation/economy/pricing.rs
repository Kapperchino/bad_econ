use super::*;
use crate::domain::enterprise::{EnterpriseModel, EnterpriseState};

#[repr(u32)]
enum InventoryPolicy {
    Producer,
    Distributor,
    Seasonal,
}

#[derive(CubeType)]
struct PacketCost {
    cost: Ticks,
}

#[cube]
impl PacketCost {
    fn new(
        state: &mut [u32],
        config: &EconomyConfig,
        control: &[Atomic<u32>],
        firm: u32,
        good: u32,
    ) -> Self {
        let payroll = times32(
            number(state, config, comptime!(Field::Wage as u32), firm),
            number(state, config, comptime!(Field::LaborMinutes as u32), firm),
        );
        let mut cost = select(
            farming::farm(state, config, firm, good) && config.home.harvest_days < 365,
            divide(
                multiply(control, payroll, farming::growing_days(config)),
                365,
            ),
            payroll,
        );
        for input in 0..2u32 {
            if number(
                state,
                config,
                comptime!(Field::FirstInput as u32) + input,
                firm,
            ) > 0
            {
                cost = add_checked(
                    control,
                    cost,
                    times32(
                        number(state, config, comptime!(Field::InputTicks as u32), firm),
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
        if number(
            state,
            config,
            comptime!(Field::EnterpriseModel as u32),
            firm,
        ) == comptime!(EnterpriseModel::Distributor as u32)
        {
            cost = add_checked(
                control,
                cost,
                times32(
                    number(state, config, comptime!(Field::OutputTicks as u32), firm),
                    number(
                        state,
                        config,
                        comptime!(Field::ReplacementCost as u32),
                        firm,
                    ),
                ),
            );
        }
        PacketCost { cost }
    }
}

#[cube]
pub(super) fn response(
    state: &mut [u32],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
    firm: u32,
    good: u32,
    excess: Ticks,
    inventory_response: u32,
) -> u32 {
    let mut response = inventory_response;
    if comptime!(config.economy == crate::domain::Economy::Industrial)
        && comptime!(config.home_mode == Feature::Enabled)
        && equal(excess, zero())
        && config.price_response > 0
    {
        let margin = PacketCost::new(state, config, control, firm, good);
        let target = divide(multiply(control, margin.cost, 5), 4);
        let output = max(
            number(state, config, comptime!(Field::OutputTicks as u32), firm),
            1,
        );
        let quote = minimum(
            divide(add_checked(control, target, ticks(output - 1, 0)), output),
            ticks(config.maximum_price, 0),
        );
        let price = max(
            number(state, config, comptime!(Field::Price as u32), firm),
            1,
        );
        let next = add_checked(
            control,
            ticks(price, 0),
            divide(
                multiply(
                    control,
                    positive(quote, ticks(price, 0)),
                    config.price_response,
                ),
                POLICY_SCALE,
            ),
        );
        let correction = minimum(
            divide(multiply(control, next, PRICE_SCALE), price),
            ticks(u32::MAX, 0),
        )
        .extract(LOW_WORD);
        response = max(response, correction);
    }
    response
}

#[cube]
pub(super) fn adjust(
    state: &mut [u32],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
    actor: u32,
    multiplier: u32,
) {
    let price = value(state, config, comptime!(Field::Price as u32), actor);
    let previous = value(state, config, comptime!(Field::PriceQuote as u32), actor);
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
    set(
        state,
        config,
        comptime!(Field::PriceQuote as u32),
        actor,
        next,
    );
    set(
        state,
        config,
        comptime!(Field::Price as u32),
        actor,
        divide(
            add_checked(control, next, ticks(PRICE_SCALE / 2, 0)),
            PRICE_SCALE,
        ),
    );
}

#[cube]
pub(super) fn inventory_target(
    state: &mut [u32],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
    firm: u32,
    good: u32,
    demand: Ticks,
    day: u32,
) -> Ticks {
    let policy = select(
        farming::farm(state, config, firm, good),
        comptime!(InventoryPolicy::Seasonal as u32),
        select(
            number(
                state,
                config,
                comptime!(Field::EnterpriseModel as u32),
                firm,
            ) == comptime!(EnterpriseModel::Distributor as u32),
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
            let batch = number(state, config, comptime!(Field::OutputTicks as u32), firm);
            add_checked(
                control,
                multiply(control, daily, config.coverage),
                ticks(max(batch, 1) - 1, 0),
            )
        }
        1 => multiply(
            control,
            demand,
            number(state, config, comptime!(Field::StockCoverage as u32), firm) + config.ring,
        ),
        _ => farming::inventory_target(config, control, demand, day),
    }
}

#[cube]
pub(super) fn stocked(state: &mut [u32], config: &EconomyConfig, actor: u32, good: u32) -> bool {
    good == comptime!(Good::Housing as u32)
        || good == comptime!(Good::Railway as u32)
        || !equal(value(state, config, good, actor), zero())
}

#[cube]
pub(super) fn refresh(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    spatial: &mut [u32],
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
    buyer: u32,
) {
    if buyer < config.actors {
        let active = !kind(actors, buyer).is(comptime!(ActorKind::Household))
            && number(
                state,
                config,
                comptime!(Field::EnterpriseStatus as u32),
                buyer,
            ) < comptime!(EnterpriseState::Liquidating as u32);
        if active {
            for input in 0..3u32 {
                let field = match input {
                    0 | 1 => comptime!(Field::FirstInputCost as u32) + input,
                    _ => u32::new(comptime!(Field::ReplacementCost as i64)),
                };
                let good = match input {
                    0 | 1 => number(
                        state,
                        config,
                        comptime!(Field::FirstInput as u32) + input,
                        buyer,
                    ),
                    _ => {
                        let distributor = number(
                            state,
                            config,
                            comptime!(Field::EnterpriseModel as u32),
                            buyer,
                        ) == comptime!(EnterpriseModel::Distributor as u32);
                        select(distributor, kind(actors, buyer).output(), 0u32)
                    }
                };
                if good > 0 {
                    let current = replacement_quote(
                        state, actors, config, &*spatial, control, parameters, buyer, good,
                    );
                    if current > 0 {
                        set32(state, config, field, buyer, current);
                    }
                }
            }
        }
    }
}

#[cube]
pub(super) fn replacement_quote(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    spatial: &[u32],
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
    buyer: u32,
    good: u32,
) -> u32 {
    let side = config.spatial_side;
    let cells = side * side;
    let segment = 2 * (cells + 1 + config.actors);
    let x = actors[ActorLayout::X * config.actors as usize + buyer as usize];
    let y = actors[ActorLayout::Y * config.actors as usize + buyer as usize];
    let cx = x / config.search_radius;
    let cy = y / config.search_radius;
    let mut quote = 0u32;
    let mut best = ticks(u32::MAX, u32::MAX);
    let mut row = max(cy, 1) - 1;
    while row <= min(cy + 1, side - 1) {
        let mut column = max(cx, 1) - 1;
        while column <= min(cx + 1, side - 1) {
            let cell = row * side + column;
            let begin = spatial[(segment + cell) as usize];
            let length = spatial[(segment + cell + 1) as usize] - begin;
            let samples = min(length, HOUSEHOLD_SEARCH_SAMPLES);
            for sample in 0..samples {
                let offset = sample * (length / samples) + sample * (length % samples) / samples;
                let rotated = (offset + parameters.day + buyer) % length;
                let seller = spatial[(segment + cells + 1 + begin + rotated) as usize];
                let relevant = seller != buyer && kind(actors, seller).output() == good;
                if relevant {
                    let status = number(
                        state,
                        config,
                        comptime!(Field::EnterpriseStatus as u32),
                        seller,
                    );
                    let trip = distance(actors, buyer, seller);
                    let price = number(state, config, comptime!(Field::Price as u32), seller);
                    let available = status < comptime!(EnterpriseState::Liquidating as u32)
                        && number(
                            state,
                            config,
                            comptime!(Field::EnterpriseModel as u32),
                            seller,
                        ) == comptime!(EnterpriseModel::Producer as u32)
                        && !equal(value(state, config, good, seller), zero())
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
            column += 1;
        }
        row += 1;
    }
    quote
}

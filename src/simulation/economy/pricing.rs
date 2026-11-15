use super::*;
use crate::domain::enterprise::{EnterpriseModel, EnterpriseState};

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
                        state, actors, config, spatial, control, parameters, buyer, good,
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
fn replacement_quote(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    spatial: &mut [u32],
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
                let status = number(
                    state,
                    config,
                    comptime!(Field::EnterpriseStatus as u32),
                    seller,
                );
                let trip = distance(actors, buyer, seller);
                let price = number(state, config, comptime!(Field::Price as u32), seller);
                let available = seller != buyer
                    && kind(actors, seller).output() == good
                    && status < comptime!(EnterpriseState::Liquidating as u32)
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
            column += 1;
        }
        row += 1;
    }
    quote
}

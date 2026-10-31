use super::words::*;
use cubecl::prelude::*;

#[cube]
fn at(config: &[u32], field: u32, actor: u32) -> u32 {
    field * config[0] + actor
}

#[cube]
fn value(state: &mut [u32], config: &[u32], field: u32, actor: u32) -> Ticks {
    load(state, at(config, field, actor))
}

#[cube]
fn number(state: &mut [u32], config: &[u32], field: u32, actor: u32) -> u32 {
    value(state, config, field, actor).extract(0usize)
}

#[cube]
fn set(state: &mut [u32], config: &[u32], field: u32, actor: u32, number: Ticks) {
    store(state, at(config, field, actor), number);
}

#[cube]
fn set32(state: &mut [u32], config: &[u32], field: u32, actor: u32, number: u32) {
    set(state, config, field, actor, ticks(number, 0u32));
}

#[cube]
fn add_checked(flags: &[Atomic<u32>], a: Ticks, b: Ticks) -> Ticks {
    let result = plus(a, b);
    if result.overflow > 0u32 {
        flags[1].fetch_max(1u32);
    }
    result.value
}

#[cube]
fn multiply(flags: &[Atomic<u32>], a: Ticks, b: u32) -> Ticks {
    let result = times(a, b);
    if result.overflow > 0u32 {
        flags[1].fetch_max(1u32);
    }
    result.value
}

#[cube]
fn increase(
    state: &mut [u32],
    config: &[u32],
    flags: &[Atomic<u32>],
    field: u32,
    actor: u32,
    amount: Ticks,
) {
    let next = add_checked(flags, value(state, config, field, actor), amount);
    set(state, config, field, actor, next);
}

#[cube]
fn kind(actors: &[u32], actor: u32) -> u32 {
    actors[(actor * 5u32 + 2u32) as usize]
}

#[cube]
fn distance(actors: &[u32], a: u32, b: u32) -> u32 {
    let ax = actors[(a * 5u32) as usize];
    let ay = actors[(a * 5u32 + 1u32) as usize];
    let bx = actors[(b * 5u32) as usize];
    let by = actors[(b * 5u32 + 1u32) as usize];
    max(ax, bx) - min(ax, bx) + max(ay, by) - min(ay, by)
}

#[allow(
    clippy::collapsible_match,
    clippy::manual_is_multiple_of,
    reason = "CubeCL uses explicit branches and integer remainder"
)]
#[cube]
fn local(
    state: &mut [u32],
    actors: &[u32],
    config: &[u32],
    flags: &[Atomic<u32>],
    parameters: &[u32],
    id: u32,
) {
    if id < config[0] {
        let k = kind(actors, id);
        let phase = parameters[1];
        match phase {
            0u32 => {
                let slot = 26u32 + parameters[0] % config[4];
                let delivery = value(state, config, slot, id);
                match k {
                    0u32 => {
                        increase(state, config, flags, 3u32, id, delivery);
                    }
                    2u32 | 3u32 => {
                        increase(state, config, flags, k - 1u32, id, delivery);
                    }
                    _ => {}
                }
                if lt(value(state, config, 16u32, id), delivery) {
                    flags[1].fetch_max(2u32);
                }
                let remaining = minus(value(state, config, 16u32, id), delivery);
                set(state, config, 16u32, id, remaining);
                set32(state, config, slot, id, 0u32);
                set32(state, config, 15u32, id, 0u32);
                set32(state, config, 13u32, id, 0u32);
                set32(state, config, 17u32, id, 0u32);
                set32(state, config, 18u32, id, 0u32);
                set32(state, config, 22u32, id, 0u32);
                set32(state, config, 24u32, id, 0u32);
                set32(state, config, 25u32, id, 0u32);
            }
            1u32 => {
                if k == 0u32 {
                    let capacity = divide(
                        times32(
                            config[13] * number(state, config, 11u32, id),
                            1000u32 - number(state, config, 12u32, id),
                        ),
                        1000000u32,
                    );
                    set32(
                        state,
                        config,
                        10u32,
                        id,
                        min(capacity.extract(0usize), 1440u32 - config[14]),
                    );
                }
            }
            2u32 => {
                if k > 0u32 {
                    let desired = positive(
                        multiply(flags, value(state, config, 8u32, id), config[16]),
                        add_checked(
                            flags,
                            value(state, config, k, id),
                            value(state, config, 9u32, id),
                        ),
                    );
                    let quotient = divide(
                        add_checked(flags, desired, ticks(config[11] - 1u32, 0u32)),
                        config[11],
                    );
                    let mut planned =
                        minimum(quotient, ticks(actors[(id * 5u32 + 4u32) as usize], 0u32));
                    if k > 1u32 {
                        planned = minimum(
                            planned,
                            divide(value(state, config, k - 1u32, id), config[11]),
                        );
                    }
                    planned = minimum(
                        planned,
                        divide(
                            value(state, config, 0u32, id),
                            number(state, config, 6u32, id) * config[12],
                        ),
                    );
                    set(state, config, 5u32, id, planned);
                    set(state, config, 23u32, id, planned);
                }
            }
            6u32 => match k {
                0u32 => {
                    let consumed = minimum(value(state, config, 3u32, id), ticks(config[9], 0u32));
                    let remaining = minus(value(state, config, 3u32, id), consumed);
                    set(state, config, 3u32, id, remaining);
                    increase(state, config, flags, 21u32, id, consumed);
                    let shortage = config[9] - consumed.extract(0usize);
                    set32(state, config, 22u32, id, shortage);
                    let mut health = min(number(state, config, 11u32, id) + 5u32, 1000u32);
                    if shortage > 0u32 {
                        health = max(number(state, config, 11u32, id), 20u32) - 20u32;
                    }
                    let mut fatigue = min(number(state, config, 12u32, id) + 10u32, 1000u32);
                    if number(state, config, 15u32, id) == 0u32 {
                        fatigue = max(number(state, config, 12u32, id), 20u32) - 20u32;
                    }
                    set32(state, config, 11u32, id, health);
                    set32(state, config, 12u32, id, fatigue);
                }
                _ => {
                    let output = value(state, config, 9u32, id);
                    increase(state, config, flags, k, id, output);
                    increase(state, config, flags, 19u32, id, output);
                    set32(state, config, 9u32, id, 0u32);
                }
            },
            7u32 => {
                if k > 0u32 {
                    let old = value(state, config, 8, id);
                    let signal = add_checked(
                        flags,
                        value(state, config, 17, id),
                        value(state, config, 18, id),
                    );
                    let increasing = !lt(signal, old);
                    let difference = select(increasing, minus(signal, old), minus(old, signal));
                    let part = scaled(flags, difference, config[15], 1000);
                    let remainder = minus(difference, times(divide(difference, 1000), 1000).value);
                    let adjustment =
                        u32::cast_from((remainder.extract(0usize) * config[15]) % 1000 != 0);
                    let demand = match u32::cast_from(increasing) {
                        1 => add_checked(flags, old, part),
                        _ => minus(old, add_checked(flags, part, ticks(adjustment, 0))),
                    };
                    let excess = positive(value(state, config, k, id), demand);
                    let unfilled = value(state, config, 18u32, id);
                    let positive_sign = !lt(unfilled, excess);
                    let magnitude = select(
                        positive_sign,
                        minus(unfilled, excess),
                        minus(excess, unfilled),
                    );
                    let denominator = maximum(demand, ticks(1, 0));
                    let scaled = fraction_index(magnitude, denominator);
                    let table_index = select(positive_sign, 32u32 + scaled, 32u32 - scaled);
                    let candidate = divide(
                        add_checked(
                            flags,
                            multiply(
                                flags,
                                value(state, config, 7u32, id),
                                config[(32u32 + table_index) as usize],
                            ),
                            ticks(999999u32, 0u32),
                        ),
                        1000000u32,
                    );
                    set32(
                        state,
                        config,
                        7u32,
                        id,
                        min(max(candidate.extract(0usize), config[18]), config[19]),
                    );
                    set(state, config, 8u32, id, demand);
                    let cap = divide(
                        divide(value(state, config, 0u32, id), config[12]),
                        max(number(state, config, 23u32, id), 1u32),
                    );
                    let offered = number(state, config, 6u32, id)
                        + select(number(state, config, 5u32, id) > 0u32, 1u32, 0u32);
                    set32(
                        state,
                        config,
                        6u32,
                        id,
                        min(
                            max(minimum(ticks(offered, 0u32), cap).extract(0usize), 1u32),
                            10000u32,
                        ),
                    );
                }
            }
            _ => {}
        }
    }
}

#[allow(
    clippy::collapsible_match,
    reason = "CubeCL uses explicit runtime branches"
)]
#[cube]
fn propose(
    state: &mut [u32],
    actors: &[u32],
    config: &[u32],
    spatial: &mut [u32],
    proposals: &mut [u32],
    flags: &[Atomic<u32>],
    parameters: &[u32],
    id: u32,
) {
    if id < config[0] {
        let market = parameters[2];
        let k = kind(actors, id);
        let mut chosen = 4294967295u32;
        let mut quantity = ticks(0u32, 0u32);
        let mut trip = 0u32;
        let mut search = false;
        let mut good = 0u32;
        let mut wanted = ticks(0u32, 0u32);
        match market {
            0u32 => {
                search = k == 0u32 && number(state, config, 15u32, id) == 0u32;
            }
            1u32 | 2u32 => {
                good = market;
                search = k == market + 1u32;
                wanted = ticks(
                    actors[(id * 5u32 + 4u32) as usize] * config[11] * config[16],
                    0u32,
                );
            }
            4u32 => {
                good = 3u32;
                search = k == 0u32;
                wanted = ticks(config[9] * (config[16] + config[4] - 1u32), 0u32);
            }
            5u32 => {
                quantity = value(state, config, 14u32, id);
                if !equal(quantity, ticks(0u32, 0u32)) {
                    chosen = number(state, config, 13u32, id) - 1u32;
                }
            }
            6u32 => {
                if k > 0u32 {
                    quantity = scaled(
                        flags,
                        positive(value(state, config, 0, id), ticks(config[21], config[22])),
                        config[17],
                        1000,
                    );
                    if !equal(quantity, ticks(0u32, 0u32)) {
                        chosen = actors[(id * 5u32 + 3u32) as usize];
                    }
                }
            }
            _ => {}
        }
        if good > 0u32 {
            wanted = positive(
                wanted,
                add_checked(
                    flags,
                    value(state, config, good, id),
                    value(state, config, 16u32, id),
                ),
            );
        }
        search = search && number(state, config, 10, id) >= select(market == 0, config[12], 0);
        if good > 0 {
            search =
                search && !equal(wanted, zero()) && !equal(value(state, config, 0, id), zero());
        }
        if search {
            let cx = actors[(id * 5u32) as usize] / config[5];
            let cy = actors[(id * 5u32 + 1u32) as usize] / config[5];
            let side = config[23];
            let mut best = ticks(4294967295u32, 4294967295u32);
            if market == 0u32 {
                best = ticks(0u32, 0u32);
            }
            let mut considered = 0u32;
            let mut y = max(cy, 1u32) - 1u32;
            while y <= min(cy + 1u32, side - 1u32) && considered < 8 {
                let mut x = max(cx, 1u32) - 1u32;
                while x <= min(cx + 1u32, side - 1u32) && considered < 8 {
                    let cell = y * side + x;
                    let mut j = spatial[cell as usize];
                    while j < spatial[(cell + 1) as usize] && considered < 8 {
                        let firm = spatial[(side * side + 1u32 + j) as usize];
                        let dist = distance(actors, id, firm);
                        let wage = number(state, config, 6u32, firm);
                        let quote = number(state, config, 7u32, firm);
                        let mut eligible = dist <= config[5];
                        let mut score = ticks(
                            quote + max(ceildiv(dist, config[8]), 1u32) * config[20],
                            0u32,
                        );
                        let mut amount = ticks(1u32, 0u32);
                        if market == 0u32 {
                            let duration = config[12] + ceildiv(dist, config[7]) * 2u32;
                            eligible = eligible
                                && number(state, config, 5u32, firm) > 0u32
                                && number(state, config, 10u32, id) >= duration;
                            score = divide(times32(wage * config[12], 1000u32), duration);
                        } else {
                            amount = minimum(
                                minimum(wanted, value(state, config, good, firm)),
                                divide(value(state, config, 0u32, id), max(quote, 1u32)),
                            );
                            eligible = eligible
                                && kind(actors, firm) == good
                                && quote > 0u32
                                && !equal(amount, ticks(0u32, 0u32));
                        }
                        let mut improves = lt(score, best);
                        if market == 0u32 {
                            improves = lt(best, score);
                        }
                        improves = improves || (equal(score, best) && firm < chosen);
                        considered += u32::cast_from(eligible);
                        if eligible && improves {
                            chosen = firm;
                            quantity = amount;
                            trip = dist;
                            best = score;
                        }
                        j += 1;
                    }
                    x += 1;
                }
                y += 1;
            }
        }
        proposals[(id * 4u32) as usize] = chosen;
        proposals[(id * 4u32 + 1u32) as usize] = quantity.extract(0usize);
        proposals[(id * 4u32 + 2u32) as usize] = quantity.extract(1usize);
        proposals[(id * 4u32 + 3u32) as usize] = trip;
    }
}

#[cube]
fn leg(
    config: &[u32],
    bundles: &mut [u32],
    event: u32,
    slot: u32,
    field: u32,
    actor: u32,
    debit: Ticks,
    credit: Ticks,
    mode: u32,
    guard: Ticks,
) {
    let base = event * 96u32 + slot * 8u32;
    bundles[(base) as usize] = at(config, field, actor);
    bundles[(base + 1u32) as usize] = debit.extract(0usize);
    bundles[(base + 2u32) as usize] = debit.extract(1usize);
    bundles[(base + 3u32) as usize] = credit.extract(0usize);
    bundles[(base + 4u32) as usize] = credit.extract(1usize);
    bundles[(base + 5u32) as usize] = mode;
    bundles[(base + 6u32) as usize] = guard.extract(0usize);
    bundles[(base + 7u32) as usize] = guard.extract(1usize);
}

#[cube]
fn transfer(
    config: &[u32],
    bundles: &mut [u32],
    event: u32,
    slot: u32,
    field: u32,
    actor: u32,
    debit: Ticks,
    credit: Ticks,
) {
    leg(
        config,
        bundles,
        event,
        slot,
        field,
        actor,
        debit,
        credit,
        0u32,
        ticks(0u32, 0u32),
    );
}

#[cube]
fn construct(
    state: &mut [u32],
    actors: &[u32],
    config: &[u32],
    proposals: &mut [u32],
    bundles: &mut [u32],
    flags: &[Atomic<u32>],
    parameters: &[u32],
    id: u32,
) {
    if id < parameters[4] {
        for slot in 0u32..12u32 {
            for word in 0u32..8u32 {
                bundles[(id * 96u32 + slot * 8u32 + word) as usize] =
                    select(word == 0u32, 4294967295u32, 0u32);
            }
        }
        let actor = id + parameters[3];
        let other = proposals[(actor * 4u32) as usize];
        let quantity = ticks(
            proposals[(actor * 4u32 + 1u32) as usize],
            proposals[(actor * 4u32 + 2u32) as usize],
        );
        let trip = proposals[(actor * 4u32 + 3u32) as usize];
        let zero = ticks(0u32, 0u32);
        if other != 4294967295u32 {
            flags[3].fetch_add(1u32);
            match parameters[2] {
                0u32 => {
                    let wage = times32(number(state, config, 6u32, other), config[12]);
                    transfer(config, bundles, id, 0u32, 0u32, other, wage, zero);
                    transfer(config, bundles, id, 1u32, 4u32, other, zero, wage);
                    transfer(
                        config,
                        bundles,
                        id,
                        2u32,
                        5u32,
                        other,
                        ticks(1u32, 0u32),
                        zero,
                    );
                    transfer(
                        config,
                        bundles,
                        id,
                        3u32,
                        10u32,
                        actor,
                        ticks(config[12] + ceildiv(trip, config[7]) * 2u32, 0u32),
                        zero,
                    );
                    transfer(
                        config,
                        bundles,
                        id,
                        4u32,
                        13u32,
                        actor,
                        zero,
                        ticks(other + 1u32, 0u32),
                    );
                    transfer(config, bundles, id, 5u32, 14u32, actor, zero, wage);
                    leg(
                        config,
                        bundles,
                        id,
                        6u32,
                        15u32,
                        actor,
                        zero,
                        ticks(1u32, 0u32),
                        1u32,
                        zero,
                    );
                    transfer(
                        config,
                        bundles,
                        id,
                        7u32,
                        25u32,
                        actor,
                        zero,
                        ticks(config[12], 0u32),
                    );
                    transfer(
                        config,
                        bundles,
                        id,
                        8u32,
                        9u32,
                        other,
                        zero,
                        ticks(config[11], 0u32),
                    );
                    leg(
                        config,
                        bundles,
                        id,
                        9u32,
                        6u32,
                        other,
                        zero,
                        zero,
                        1u32,
                        value(state, config, 6u32, other),
                    );
                    if kind(actors, other) > 1u32 {
                        transfer(
                            config,
                            bundles,
                            id,
                            10u32,
                            kind(actors, other) - 1u32,
                            other,
                            ticks(config[11], 0u32),
                            zero,
                        );
                        transfer(
                            config,
                            bundles,
                            id,
                            11u32,
                            20u32,
                            other,
                            zero,
                            ticks(config[11], 0u32),
                        );
                    }
                }
                5u32 => {
                    transfer(config, bundles, id, 0u32, 4u32, other, quantity, zero);
                    transfer(config, bundles, id, 1u32, 14u32, actor, quantity, zero);
                    transfer(config, bundles, id, 2u32, 0u32, actor, zero, quantity);
                    leg(
                        config,
                        bundles,
                        id,
                        3u32,
                        13u32,
                        actor,
                        zero,
                        zero,
                        1u32,
                        ticks(other + 1u32, 0u32),
                    );
                }
                6u32 => {
                    transfer(config, bundles, id, 0u32, 0u32, actor, quantity, zero);
                    transfer(config, bundles, id, 1u32, 0u32, other, zero, quantity);
                }
                _ => {
                    let good = select(parameters[2] == 4u32, 3u32, parameters[2]);
                    let cost = multiply(flags, quantity, number(state, config, 7u32, other));
                    let slot =
                        26u32 + (parameters[0] + max(ceildiv(trip, config[8]), 1u32)) % config[4];
                    transfer(config, bundles, id, 0u32, 0u32, actor, cost, zero);
                    transfer(config, bundles, id, 1u32, 0u32, other, zero, cost);
                    transfer(config, bundles, id, 2u32, good, other, quantity, zero);
                    transfer(config, bundles, id, 3u32, slot, actor, zero, quantity);
                    transfer(config, bundles, id, 4u32, 16u32, actor, zero, quantity);
                    transfer(config, bundles, id, 5u32, 17u32, other, zero, quantity);
                    transfer(config, bundles, id, 6u32, 24u32, actor, zero, quantity);
                    leg(
                        config,
                        bundles,
                        id,
                        7u32,
                        7u32,
                        other,
                        zero,
                        zero,
                        1u32,
                        value(state, config, 7u32, other),
                    );
                }
            }
        }
    }
}

#[cube]
fn summarize(
    state: &mut [u32],
    actors: &[u32],
    config: &[u32],
    bundles: &mut [u32],
    flags: &[Atomic<u32>],
    id: u32,
) {
    let groups = ceildiv(config[0], 256u32);
    if id < groups {
        let mut totals = Array::<Ticks>::new(24usize);
        for metric in 0..24usize {
            totals[metric] = zero();
        }
        totals[16] = ticks(4294967295u32, 4294967295u32);
        totals[18] = ticks(4294967295u32, 4294967295u32);
        for actor in id * 256u32..min((id + 1u32) * 256u32, config[0]) {
            totals[0] = add_checked(
                flags,
                totals[0],
                add_checked(
                    flags,
                    value(state, config, 0u32, actor),
                    value(state, config, 4u32, actor),
                ),
            );
            for g in 0u32..3u32 {
                totals[(3u32 + g) as usize] = add_checked(
                    flags,
                    totals[(3u32 + g) as usize],
                    value(state, config, g + 1u32, actor),
                );
            }
            if !equal(value(state, config, 4, actor), zero())
                || !equal(value(state, config, 9, actor), zero())
                || !equal(value(state, config, 14, actor), zero())
            {
                flags[1].fetch_max(4);
            }
            let k = kind(actors, actor);
            let mut input = 0u32;
            match k {
                0u32 => {
                    input = 3u32;
                    totals[1] = add_checked(flags, totals[1], value(state, config, 15u32, actor));
                    totals[2] = add_checked(flags, totals[2], value(state, config, 22u32, actor));
                    totals[15] = add_checked(flags, totals[15], value(state, config, 21u32, actor));
                    let cash = value(state, config, 0u32, actor);
                    let mut bin = 32u32 - cash.extract(0usize).leading_zeros();
                    if cash.extract(1usize) > 0u32 {
                        bin = 64u32 - cash.extract(1usize).leading_zeros();
                    }
                    flags[(4u32 + min(bin, 63u32)) as usize].fetch_add(1u32);
                    let x = actors[(actor * 5u32) as usize] * 16u32 / config[6];
                    let y = actors[(actor * 5u32 + 1u32) as usize] * 16u32 / config[6];
                    flags[(68u32 + y * 16u32 + x) as usize]
                        .fetch_add(number(state, config, 22u32, actor));
                }
                _ => {
                    if k > 1u32 {
                        input = k - 1u32;
                    }
                    totals[(3u32 + k - 1u32) as usize] = add_checked(
                        flags,
                        totals[(3u32 + k - 1u32) as usize],
                        value(state, config, 9u32, actor),
                    );
                    totals[(9u32 + k - 1u32) as usize] = add_checked(
                        flags,
                        totals[(9u32 + k - 1u32) as usize],
                        value(state, config, 19u32, actor),
                    );
                    totals[16] = minimum(totals[16], value(state, config, 6u32, actor));
                    totals[17] = select(
                        lt(totals[17], value(state, config, 6u32, actor)),
                        value(state, config, 6u32, actor),
                        totals[17],
                    );
                    totals[18] = minimum(totals[18], value(state, config, 7u32, actor));
                    totals[19] = select(
                        lt(totals[19], value(state, config, 7u32, actor)),
                        value(state, config, 7u32, actor),
                        totals[19],
                    );
                }
            }
            if input > 0u32 {
                totals[(6u32 + input - 1u32) as usize] = add_checked(
                    flags,
                    totals[(6u32 + input - 1u32) as usize],
                    value(state, config, 16u32, actor),
                );
                totals[(12u32 + input - 1u32) as usize] = add_checked(
                    flags,
                    totals[(12u32 + input - 1u32) as usize],
                    value(state, config, 20u32, actor),
                );
            }
            let mut transit = ticks(0u32, 0u32);
            for slot in 0u32..config[4] {
                transit = add_checked(flags, transit, value(state, config, 26u32 + slot, actor));
            }
            if !equal(transit, value(state, config, 16u32, actor)) {
                flags[1].fetch_max(3u32);
            }
        }
        for metric in 0u32..24u32 {
            let base = (id * 24u32 + metric) * 2u32;
            bundles[(base) as usize] = totals[(metric) as usize].extract(0usize);
            bundles[(base + 1u32) as usize] = totals[(metric) as usize].extract(1usize);
        }
    }
}

#[cube]
fn reduce(
    config: &[u32],
    proposals: &mut [u32],
    bundles: &mut [u32],
    flags: &[Atomic<u32>],
    id: u32,
) {
    if id < 24u32 {
        let mut total = ticks(0u32, 0u32);
        if id == 16u32 || id == 18u32 {
            total = ticks(4294967295u32, 4294967295u32);
        }
        for group in 0u32..ceildiv(config[0], 256u32) {
            let base = (group * 24u32 + id) * 2u32;
            let part = ticks(bundles[(base) as usize], bundles[(base + 1u32) as usize]);
            match id {
                16u32 | 18u32 => {
                    total = minimum(total, part);
                }
                17u32 | 19u32 => {
                    total = select(lt(total, part), part, total);
                }
                _ => {
                    total = add_checked(flags, total, part);
                }
            }
        }
        proposals[(id * 2u32) as usize] = total.extract(0usize);
        proposals[(id * 2u32 + 1u32) as usize] = total.extract(1usize);
    }
}

#[cube]
fn inspect(state: &mut [u32], config: &[u32], proposals: &mut [u32], parameters: &[u32], id: u32) {
    if id < config[3] {
        let item = value(state, config, id, parameters[3]);
        proposals[(id * 2u32) as usize] = item.extract(0usize);
        proposals[(id * 2u32 + 1u32) as usize] = item.extract(1usize);
    }
}

#[cube]
fn copy_state(state: &mut [u32], proposals: &mut [u32], id: u32) {
    if id < (state.len() as u32 / 2) {
        proposals[(id * 2u32) as usize] = state[(id * 2) as usize];
        proposals[(id * 2u32 + 1u32) as usize] = state[(id * 2 + 1) as usize];
    }
}

#[derive(CubeType, Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum EconomyStage {
    ClearFeedback,
    RecordFeedback,
    ApplyFeedback,
    IndexCount,
    IndexPrefix,
    IndexFill,
    Local,
    Propose,
    Construct,
    Summarize,
    Reduce,
    Inspect,
    CopyState,
}

#[cube(launch)]
pub fn economy(
    state: &mut [u32],
    actors: &[u32],
    config: &[u32],
    spatial: &mut [u32],
    proposals: &mut [u32],
    bundles: &mut [u32],
    flags: &[Atomic<u32>],
    parameters: &[u32],
    #[comptime] stage: EconomyStage,
) {
    let id = ABSOLUTE_POS as u32;
    match stage {
        EconomyStage::ClearFeedback => {
            if id < flags.len() as u32 {
                flags[id as usize].store(0);
            }
        }
        EconomyStage::RecordFeedback => {
            record_feedback(config, proposals, bundles, flags, parameters, id)
        }
        EconomyStage::ApplyFeedback => apply_feedback(state, config, flags, id),
        EconomyStage::IndexCount => index_count(state, actors, config, spatial, parameters, id),
        EconomyStage::IndexPrefix => index_prefix(config, spatial, id),
        EconomyStage::IndexFill => index_fill(state, actors, config, spatial, parameters, id),
        EconomyStage::Local => local(state, actors, config, flags, parameters, id),
        EconomyStage::Propose => propose(
            state, actors, config, spatial, proposals, flags, parameters, id,
        ),
        EconomyStage::Construct => construct(
            state, actors, config, proposals, bundles, flags, parameters, id,
        ),
        EconomyStage::Summarize => summarize(state, actors, config, bundles, flags, id),
        EconomyStage::Reduce => reduce(config, proposals, bundles, flags, id),
        EconomyStage::Inspect => inspect(state, config, proposals, parameters, id),
        EconomyStage::CopyState => copy_state(state, proposals, id),
    }
}

#[cube]
fn discoverable(
    state: &mut [u32],
    actors: &[u32],
    config: &[u32],
    parameters: &[u32],
    firm: u32,
) -> bool {
    let market = parameters[2];
    let good = select(market == 4, 3, market);
    match market {
        0 => number(state, config, 5, firm) > 0,
        1 | 2 | 4 => kind(actors, firm) == good && !equal(value(state, config, good, firm), zero()),
        _ => false,
    }
}
#[cube]
fn firm_cell(actors: &[u32], config: &[u32], firm: u32) -> u32 {
    actors[(firm * 5 + 1) as usize] / config[5] * config[23]
        + actors[(firm * 5) as usize] / config[5]
}
#[cube]
fn index_count(
    state: &mut [u32],
    actors: &[u32],
    config: &[u32],
    spatial: &mut [u32],
    parameters: &[u32],
    cell: u32,
) {
    if cell < config[23] * config[23] {
        let mut count = 0u32;
        for firm in config[1]..config[0] {
            if firm_cell(actors, config, firm) == cell {
                count += u32::cast_from(discoverable(state, actors, config, parameters, firm));
            }
        }
        spatial[cell as usize] = count;
    }
}
#[cube]
fn index_prefix(config: &[u32], spatial: &mut [u32], id: u32) {
    if id == 0 {
        let mut offset = 0u32;
        for cell in 0..config[23] * config[23] {
            let count = spatial[cell as usize];
            spatial[cell as usize] = offset;
            offset += count;
        }
        spatial[(config[23] * config[23]) as usize] = offset;
    }
}
#[cube]
fn index_fill(
    state: &mut [u32],
    actors: &[u32],
    config: &[u32],
    spatial: &mut [u32],
    parameters: &[u32],
    cell: u32,
) {
    let cells = config[23] * config[23];
    if cell < cells {
        let mut offset = spatial[cell as usize];
        for firm in config[1]..config[0] {
            if firm_cell(actors, config, firm) == cell
                && discoverable(state, actors, config, parameters, firm)
            {
                spatial[(cells + 1 + offset) as usize] = firm;
                offset += 1;
            }
        }
    }
}

#[cube]
fn scaled(flags: &[Atomic<u32>], value: Ticks, numerator: u32, denominator: u32) -> Ticks {
    let result = ratio(value, numerator, denominator);
    if result.overflow != 0 {
        flags[1].fetch_max(1);
    }
    result.value
}
#[allow(
    clippy::collapsible_if,
    reason = "Guard the storage read before indexing a partial workgroup"
)]
#[cube]
fn record_feedback(
    config: &[u32],
    proposals: &mut [u32],
    outcomes: &mut [u32],
    feedback: &[Atomic<u32>],
    parameters: &[u32],
    id: u32,
) {
    if id < parameters[4] {
        if outcomes[(parameters[5] + id) as usize] == 3 {
            let actor = parameters[3] + id;
            let firm = proposals[(actor * 4) as usize] - config[1];
            let base = 4 + firm * 2;
            let quantity = proposals[(actor * 4 + 1) as usize];
            let old = feedback[base as usize].fetch_add(quantity);
            let carry = u32::cast_from(quantity > u32::MAX - old);
            let high = proposals[(actor * 4 + 2) as usize] + carry;
            let prior = feedback[(base + 1) as usize].fetch_add(high);
            if high > u32::MAX - prior {
                feedback[1].fetch_max(1);
            }
        }
    }
}
#[cube]
fn apply_feedback(state: &mut [u32], config: &[u32], feedback: &[Atomic<u32>], id: u32) {
    if id < config[2] {
        let base = 4 + id * 2;
        set(
            state,
            config,
            18,
            config[1] + id,
            ticks(
                feedback[base as usize].load(),
                feedback[(base + 1) as usize].load(),
            ),
        );
    }
}

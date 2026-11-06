use crate::domain::{Field, Good, enterprise::EnterpriseState};
use crate::simulation::configuration::EconomyConfig;
use crate::simulation::{layout::ActorLayout as A, words::*};
use cubecl::prelude::*;

pub(in crate::simulation) const PRODUCER_ENTRY: u32 = 8 << 24;
pub(in crate::simulation) const DISTRIBUTOR_ENTRY: u32 = 9 << 24;

pub(in crate::simulation) struct EntryPlan {
    pub owner: usize,
    pub good: Good,
    pub position: crate::domain::Position,
    pub model: crate::domain::enterprise::EnterpriseModel,
    pub recipe: crate::domain::industry::Recipe,
    pub price: u64,
    pub first_cost: u64,
    pub second_cost: u64,
}

impl EntryPlan {
    pub fn new(
        state: &crate::domain::State,
        scenario: &crate::domain::Scenario,
        owner: usize,
        request: u32,
    ) -> anyhow::Result<Self> {
        use crate::domain::{Technique, enterprise::EnterpriseModel, industry::Recipe};
        use anyhow::{Context, anyhow};
        let config = scenario.config();
        let first_cost = state.get(Field::ReplacementCost, owner);
        let plan = match request >> 24 {
            0 => {
                let good = Good::from_index(request)?;
                Self {
                    owner,
                    good,
                    position: state.actors.at(owner).position,
                    model: EnterpriseModel::Producer,
                    recipe: Recipe {
                        technique: Technique::new(240, 20, 20)?,
                        inputs: good.inputs(),
                    },
                    price: state
                        .get(Field::Price, owner)
                        .max(u64::from(config.prices[2])),
                    first_cost,
                    second_cost: 0,
                }
            }
            8 | 9 => {
                let source = (request & 0x00ffffff)
                    .checked_sub(1)
                    .context("missing enterprise reference")? as usize;
                let actor = state
                    .actors
                    .get(source)
                    .context("unknown enterprise reference")?;
                let good = Good::from_index(actor.kind)?;
                let model = match request >> 24 {
                    8 => EnterpriseModel::Producer,
                    _ => EnterpriseModel::Distributor,
                };
                let recipe = match model {
                    EnterpriseModel::Distributor => Recipe {
                        technique: Technique::new(60, 0, 24)?,
                        inputs: [None, None],
                    },
                    EnterpriseModel::Producer => Recipe {
                        technique: Technique::new(
                            state.get(Field::BasicLaborMinutes, source) as u32,
                            state.get(Field::BasicInputTicks, source) as u32,
                            state.get(Field::BasicOutputTicks, source) as u32,
                        )?,
                        inputs: [Field::BasicFirstInput, Field::BasicSecondInput]
                            .map(|field| match state.get(field, source) {
                                0 => Ok(None),
                                index => Good::from_index(index as u32).map(Some),
                            })
                            .into_iter()
                            .collect::<anyhow::Result<Vec<_>>>()?
                            .try_into()
                            .expect("two recipe inputs"),
                    },
                };
                Self {
                    owner,
                    good,
                    position: match model {
                        EnterpriseModel::Producer => actor.position,
                        EnterpriseModel::Distributor => state.actors.at(owner).position,
                    },
                    model,
                    recipe,
                    price: match model {
                        EnterpriseModel::Producer => state.get(Field::Price, source),
                        EnterpriseModel::Distributor => {
                            first_cost * 1250 / 1000
                                + u64::from(
                                    60 * config.wage
                                        + config.rent.map_or(0, |rent| rent.firm_daily),
                                )
                                .div_ceil(24)
                        }
                    },
                    first_cost,
                    second_cost: state.get(Field::SecondInputCost, owner),
                }
            }
            _ => Err(anyhow!("unsupported enterprise entry request"))?,
        };
        match state.actors.at(owner).kind == 0
            && state.get(Field::Cash, owner) >= 10_000
            && state.get(Field::RentArrears, owner) == 0
            && state.get(Field::Tools, owner) >= 2
            && state.get(Field::Wood, owner) >= 5
            && (u64::from(config.minimum_price)..=u64::from(config.maximum_price))
                .contains(&plan.price)
        {
            true => Ok(plan),
            false => Err(anyhow!(
                "enterprise entry lacks invested funds, materials, or a valid quote"
            )),
        }
    }
}

#[cube]
fn number(state: &mut [u32], config: &EconomyConfig, actor: u32, field: u32) -> u32 {
    load(state, field * config.actors + actor).extract(0usize)
}

#[cube]
fn distance(actors: &[u32], first: u32, second: u32) -> u32 {
    let count = actors.len() / A::WORDS;
    let ax = actors[A::X * count + first as usize];
    let ay = actors[A::Y * count + first as usize];
    let bx = actors[A::X * count + second as usize];
    let by = actors[A::Y * count + second as usize];
    max(ax, bx) - min(ax, bx) + max(ay, by) - min(ay, by)
}

#[cube]
fn firms(spatial: &[u32], config: &EconomyConfig) -> u32 {
    let cells = config.spatial_side * config.spatial_side;
    spatial[(2 * (cells + 1 + config.actors) + cells) as usize]
}

#[cube]
fn firm_at(spatial: &[u32], config: &EconomyConfig, offset: u32) -> u32 {
    let cells = config.spatial_side * config.spatial_side;
    spatial[(2 * (cells + 1 + config.actors) + cells + 1 + offset % max(firms(spatial, config), 1))
        as usize]
}

#[cube]
fn input_price(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    spatial: &[u32],
    site: u32,
    good: u32,
    quantity: u32,
    day: u32,
) -> u32 {
    let mut price = u32::new(comptime!(i64::from(u32::MAX)));
    if good == 0 {
        price = 0;
    }
    if good > 0 {
        for sample in 0..min(32u32, firms(spatial, config)) {
            let supplier = firm_at(spatial, config, site + day * 17 + sample);
            let kind = actors[A::KIND * (actors.len() / A::WORDS) + supplier as usize];
            let stock = load(state, good * config.actors + supplier);
            if kind == good
                && number(
                    state,
                    config,
                    supplier,
                    comptime!(Field::EnterpriseModel as u32),
                ) == 0
                && number(
                    state,
                    config,
                    supplier,
                    comptime!(Field::EnterpriseStatus as u32),
                ) == comptime!(EnterpriseState::Operating as u32)
                && !lt(stock, ticks(quantity, 0))
                && distance(actors, site, supplier) <= config.search_radius
            {
                price = min(
                    price,
                    number(state, config, supplier, comptime!(Field::Price as u32)),
                );
            }
        }
    }
    price
}

#[cube]
pub(in crate::simulation) fn discover(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    spatial: &[u32],
    owner: u32,
    day: u32,
) -> u32 {
    let count = config.actors;
    let cash = load(state, comptime!(Field::Cash as u32) * count + owner);
    let investment = divide(cash, 2);
    let wage = config.starting_wage;
    let mut selected = 0;
    let mut best = zero();
    let mut first_cost = 0;
    let mut second_cost = 0;
    for sample in 0..min(32u32, firms(spatial, config)) {
        let source = firm_at(spatial, config, owner + day * 17 + sample);
        let kind = actors[A::KIND * (actors.len() / A::WORDS) + source as usize];
        let active = number(
            state,
            config,
            source,
            comptime!(Field::EnterpriseStatus as u32),
        ) == comptime!(EnterpriseState::Operating as u32);
        let producer = number(
            state,
            config,
            source,
            comptime!(Field::EnterpriseModel as u32),
        ) == 0;
        if kind > 0 && active && producer && distance(actors, owner, source) <= config.search_radius
        {
            let price = number(state, config, source, comptime!(Field::Price as u32));
            let unfilled = load(state, comptime!(Field::Unfilled as u32) * count + source);
            let sales = load(state, comptime!(Field::Sales as u32) * count + source);
            if !equal(unfilled, zero())
                && !equal(sales, zero())
                && actors[A::GEOGRAPHIC_CAPACITY * (actors.len() / A::WORDS) + source as usize] > 0
            {
                let inputs = number(
                    state,
                    config,
                    source,
                    comptime!(Field::BasicInputTicks as u32),
                );
                let first_input = number(
                    state,
                    config,
                    source,
                    comptime!(Field::BasicFirstInput as u32),
                );
                let second_input = number(
                    state,
                    config,
                    source,
                    comptime!(Field::BasicSecondInput as u32),
                );
                let first = input_price(
                    state,
                    actors,
                    config,
                    spatial,
                    source,
                    first_input,
                    inputs,
                    day,
                );
                let second = input_price(
                    state,
                    actors,
                    config,
                    spatial,
                    source,
                    second_input,
                    inputs,
                    day,
                );
                if first < u32::MAX && second < u32::MAX {
                    let labor = number(
                        state,
                        config,
                        source,
                        comptime!(Field::BasicLaborMinutes as u32),
                    );
                    let output = number(
                        state,
                        config,
                        source,
                        comptime!(Field::BasicOutputTicks as u32),
                    );
                    let cost = plus(
                        plus(times32(labor, wage), times32(inputs, first + second)).value,
                        ticks(config.firm_rent, 0),
                    )
                    .value;
                    let revenue = times32(output, price);
                    let margin = minus(revenue, minimum(revenue, cost));
                    let needed = plus(
                        times(cost, config.coverage + config.ring).value,
                        times32(720, wage),
                    )
                    .value;
                    if lt(best, margin) && !lt(investment, needed) {
                        selected = comptime!(PRODUCER_ENTRY) + source + 1;
                        best = margin;
                        first_cost = first;
                        second_cost = second;
                    }
                }
            }
            let stock = load(state, kind * count + source);
            let consumer = kind == comptime!(Good::Bread as u32)
                || (comptime!(config.economy == crate::domain::Economy::Industrial)
                    && config.consumption_intervals[kind as usize - 1] > 1);
            if consumer && !lt(stock, ticks(24, 0)) {
                let quote = price * 1250 / 1000 + ceildiv(60 * wage + config.firm_rent, 24);
                let mut buyers = 0;
                for buyer_sample in 0..min(32u32, count) {
                    let buyer = (owner + day * 23 + buyer_sample) % count;
                    let household =
                        actors[A::KIND * (actors.len() / A::WORDS) + buyer as usize] == 0;
                    let buyer_cash = load(state, comptime!(Field::Cash as u32) * count + buyer);
                    let need = select(
                        kind == comptime!(Good::Bread as u32),
                        number(state, config, buyer, comptime!(Field::FoodNeed as u32)),
                        1,
                    );
                    if household
                        && number(state, config, buyer, comptime!(Field::LivingMembers as u32)) > 0
                        && distance(actors, owner, buyer) <= config.search_radius
                        && distance(actors, source, buyer) > config.search_radius
                        && !lt(buyer_cash, times32(need, quote))
                    {
                        buyers += need;
                    }
                }
                let units = min(buyers, 24);
                let revenue = times32(units, quote);
                let cost = plus(
                    plus(times32(units, price), times32(60, wage)).value,
                    ticks(config.firm_rent, 0),
                )
                .value;
                let margin = minus(revenue, minimum(revenue, cost));
                let required = plus(
                    plus(times32(24 * price, 3 + config.ring), times32(780, wage)).value,
                    times32(config.firm_rent, 3 + config.ring),
                )
                .value;
                if lt(best, margin) && !lt(investment, required) && quote <= config.maximum_price {
                    selected = comptime!(DISTRIBUTOR_ENTRY) + source + 1;
                    best = margin;
                    first_cost = price;
                    second_cost = 0;
                }
            }
        }
    }
    if selected > 0 {
        store(
            state,
            comptime!(Field::ReplacementCost as u32) * count + owner,
            ticks(first_cost, 0),
        );
        store(
            state,
            comptime!(Field::SecondInputCost as u32) * count + owner,
            ticks(second_cost, 0),
        );
    }
    selected
}

use crate::domain::{Field, Good, enterprise::EnterpriseState};
use crate::simulation::configuration::EconomyConfig;
use crate::simulation::{layout::ActorLayout as A, words::*};
use cubecl::prelude::*;

pub(in crate::simulation) const PRODUCER_ENTRY: u32 = 8 << 24;
pub(in crate::simulation) const DISTRIBUTOR_ENTRY: u32 = 9 << 24;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(in crate::simulation) enum EntryQuoteMode {
    Cached,
    #[cfg_attr(not(test), allow(dead_code))]
    Sampled,
}

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
                    EnterpriseModel::Producer => {
                        let input = |field| match state.get(field, source) {
                            0 => Ok(None),
                            index => Good::from_index(index as u32).map(Some),
                        };
                        Recipe {
                            technique: Technique::new(
                                state.get(Field::BasicLaborMinutes, source) as u32,
                                state.get(Field::BasicInputTicks, source) as u32,
                                state.get(Field::BasicOutputTicks, source) as u32,
                            )?,
                            inputs: [
                                input(Field::BasicFirstInput)?,
                                input(Field::BasicSecondInput)?,
                            ],
                        }
                    }
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
            && (config.property.is_some()
                || (state.get(Field::Tools, owner) >= 2 && state.get(Field::Wood, owner) >= 5))
            && config.property.as_ref().is_none_or(|property| {
                property.region(plan.position).is_ok_and(|region| {
                    property.regions[region].landowner == state.actors.at(owner).identity
                })
            })
            && (u64::from(config.minimum_price)..=u64::from(config.maximum_price))
                .contains(&plan.price)
        {
            true => Ok(plan),
            false => Err(anyhow!(
                "enterprise entry lacks invested funds, materials, or a valid quote: owner {owner}, cash {}, tools {}, wood {}, arrears {}, quote {}",
                state.get(Field::Cash, owner),
                state.get(Field::Tools, owner),
                state.get(Field::Wood, owner),
                state.get(Field::RentArrears, owner),
                plan.price,
            )),
        }
    }
}

#[cube]
fn number(state: &mut [u32], config: &EconomyConfig, actor: u32, field: u32) -> u32 {
    load(state, field * config.state_stride + actor).extract(0usize)
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
            let stock = load(state, good * config.state_stride + supplier);
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
pub(super) fn prepare_quotes(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    spatial: &[u32],
    quotes: &mut [u32],
    day: u32,
) {
    let member = ABSOLUTE_POS as u32;
    if member < firms(spatial, config) {
        let source = firm_at(spatial, config, member);
        let inputs = number(
            state,
            config,
            source,
            comptime!(Field::BasicInputTicks as u32),
        );
        #[unroll]
        for slot in 0..2u32 {
            let good = number(
                state,
                config,
                source,
                comptime!(Field::BasicFirstInput as u32) + slot,
            );
            quotes[(2 * config.actors + 2 * source + slot) as usize] =
                input_price(state, actors, config, spatial, source, good, inputs, day);
        }
    }
}

#[cube]
pub(in crate::simulation) fn discover(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    spatial: &[u32],
    quotes: &[u32],
    owner: u32,
    day: u32,
    #[comptime] quote_mode: EntryQuoteMode,
) -> u32 {
    let cash = load(
        state,
        comptime!(Field::Cash as u32) * config.state_stride + owner,
    );
    let members = crate::simulation::enterprise::founder_shares(state, actors, config, owner);
    let investment = divide(divide(cash, members), 2);
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
            if actors[A::GEOGRAPHIC_CAPACITY * (actors.len() / A::WORDS) + source as usize] > 0 {
                let inputs = number(
                    state,
                    config,
                    source,
                    comptime!(Field::BasicInputTicks as u32),
                );
                let first = entry_quote(
                    state, actors, config, spatial, quotes, source, inputs, day, 0, quote_mode,
                );
                let second = entry_quote(
                    state, actors, config, spatial, quotes, source, inputs, day, 1, quote_mode,
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
                    let seasonal = config.home.harvest_days < 365
                        && (kind == comptime!(Good::Grain as u32)
                            || kind == comptime!(Good::Flax as u32)
                            || kind == comptime!(Good::Cotton as u32)
                            || kind == comptime!(Good::Dye as u32));
                    let work_days = select(
                        seasonal,
                        crate::simulation::economy::farming::paid_days(config, kind),
                        365u32,
                    );
                    let payroll = divide(times32(labor * wage, work_days), 365);
                    let cost = plus(
                        plus(payroll, times32(inputs, first + second)).value,
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
            let stock = load(state, kind * config.state_stride + source);
            let consumer = kind == comptime!(Good::Bread as u32)
                || kind == comptime!(Good::Wood as u32)
                || kind == comptime!(Good::Coal as u32)
                || (comptime!(config.economy == crate::domain::Economy::Industrial)
                    && config.consumption_intervals[kind as usize - 1] > 1);
            if consumer && !lt(stock, ticks(24, 0)) {
                let quote = price * 1250 / 1000 + ceildiv(60 * wage + config.firm_rent, 24);
                let revenue = times32(24, quote);
                let cost = plus(
                    plus(times32(24, price), times32(60, wage)).value,
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
            comptime!(Field::ReplacementCost as u32) * config.state_stride + owner,
            ticks(first_cost, 0),
        );
        store(
            state,
            comptime!(Field::SecondInputCost as u32) * config.state_stride + owner,
            ticks(second_cost, 0),
        );
    }
    selected
}

#[cube]
fn entry_quote(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    spatial: &[u32],
    quotes: &[u32],
    source: u32,
    quantity: u32,
    day: u32,
    slot: u32,
    #[comptime] mode: EntryQuoteMode,
) -> u32 {
    match comptime!(mode) {
        EntryQuoteMode::Cached => quotes[(2 * config.actors + 2 * source + slot) as usize],
        EntryQuoteMode::Sampled => {
            let good = number(
                state,
                config,
                source,
                comptime!(Field::BasicFirstInput as u32) + slot,
            );
            input_price(state, actors, config, spatial, source, good, quantity, day)
        }
    }
}

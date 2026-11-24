use super::acquisitions::PURCHASE;
use crate::domain::{Field, Good, enterprise::EnterpriseState};
use crate::simulation::economy::configuration::EconomyConfig;
use crate::simulation::storage::state::*;
use crate::simulation::storage::words::*;
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
                "enterprise entry lacks invested funds or a valid quote: owner {owner}, cash {}, arrears {}, quote {}",
                state.get(Field::Cash, owner),
                state.get(Field::RentArrears, owner),
                plan.price,
            )),
        }
    }
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
pub(super) fn purchase(
    economy: &EconomyView<'_>,
    spatial: &[u32],
    buyer: u32,
    day: u32,
    available: Ticks,
) -> u32 {
    let config = economy.ledger.config;
    let owner = economy.actors.owner(buyer);
    let draw = economy.actors.expansion_draw(buyer);
    let product = economy.actors.kind(buyer).bits();
    let model = economy.ledger.number(Field::EnterpriseModel, buyer);
    let mut related = 0u32;
    let mut fallback = 0u32;
    let start = buyer + draw + (day / 30) * 17;
    for sample in 0..min(32u32, firms(spatial, config)) {
        let site = firm_at(spatial, config, start + sample);
        let status = economy.ledger.number(Field::EnterpriseStatus, site);
        let cash = economy.ledger.value(Field::Cash, site);
        if related == 0
            && site < 0x00ff_ffff
            && economy.actors.owner(site) != owner
            && (status == comptime!(EnterpriseState::Operating as u32)
                || status == comptime!(EnterpriseState::Dormant as u32))
            && !lt(available, maximum(ticks(1, 0), divide(cash, 2)))
        {
            let request = comptime!(PURCHASE << 24) + site + 1;
            let similar = economy.actors.kind(site).bits() == product
                && economy.ledger.number(Field::EnterpriseModel, site) == model;
            related = select(similar, request, related);
            fallback = select(fallback == 0, request, fallback);
        }
    }
    select(related > 0, related, fallback)
}

#[cube]
fn input_price(
    economy: &EconomyView<'_>,
    spatial: &[u32],
    site: u32,
    good: u32,
    quantity: u32,
    day: u32,
) -> u32 {
    let config = economy.ledger.config;
    let mut price = u32::new(comptime!(i64::from(u32::MAX)));
    if good == 0 {
        price = 0;
    }
    if good > 0 {
        for sample in 0..min(32u32, firms(spatial, config)) {
            let supplier = firm_at(spatial, config, site + day * 17 + sample);
            let kind = economy.actors.kind(supplier).bits();
            let stock = economy.ledger.value_at(good, supplier);
            if kind == good
                && economy.ledger.number(Field::EnterpriseModel, supplier) == 0
                && economy.ledger.number(Field::EnterpriseStatus, supplier)
                    == comptime!(EnterpriseState::Operating as u32)
                && !lt(stock, ticks(quantity, 0))
                && economy.actors.distance(site, supplier) <= config.search_radius
            {
                price = min(price, economy.ledger.number(Field::Price, supplier));
            }
        }
    }
    price
}

#[cube]
pub(super) fn prepare_quotes(
    economy: &EconomyView<'_>,
    spatial: &[u32],
    quotes: &mut [u32],
    day: u32,
) {
    let config = economy.ledger.config;
    let member = ABSOLUTE_POS as u32;
    if member < firms(spatial, config) {
        let source = firm_at(spatial, config, member);
        let inputs = economy.ledger.number(Field::BasicInputTicks, source);
        #[unroll]
        for slot in 0..2u32 {
            let good = economy
                .ledger
                .number_at(comptime!(Field::BasicFirstInput as u32) + slot, source);
            quotes[(2 * config.actors + 2 * source + slot) as usize] =
                input_price(economy, spatial, source, good, inputs, day);
        }
    }
}

#[cube]
pub(in crate::simulation) fn discover(
    economy: &mut EconomyView<'_>,
    spatial: &[u32],
    quotes: &[u32],
    owner: u32,
    day: u32,
    #[comptime] quote_mode: EntryQuoteMode,
) -> u32 {
    let config = economy.ledger.config;
    let cash = economy.ledger.value(Field::Cash, owner);
    let members = crate::simulation::enterprise::founder_shares(&*economy, owner);
    let investment = divide(divide(cash, members), 2);
    let wage = config.starting_wage;
    let construction = ticks(
        comptime!(
            crate::domain::property::CONSTRUCTION_WORK_PER_PLOT
                * Good::Construction.initial_price()
        ),
        0,
    );
    let mut selected = 0;
    let mut best = zero();
    let mut first_cost = 0;
    let mut second_cost = 0;
    for sample in 0..min(32u32, firms(spatial, config)) {
        let source = firm_at(spatial, config, owner + day * 17 + sample);
        let kind = economy.actors.kind(source).bits();
        let active = economy.ledger.number(Field::EnterpriseStatus, source)
            == comptime!(EnterpriseState::Operating as u32);
        let producer = economy.ledger.number(Field::EnterpriseModel, source) == 0;
        if kind > 0
            && active
            && producer
            && economy.actors.distance(owner, source) <= config.search_radius
        {
            let price = economy.ledger.number(Field::Price, source);
            if economy.actors.geographic_capacity(source) > 0 {
                let inputs = economy.ledger.number(Field::BasicInputTicks, source);
                let first = entry_quote(
                    &*economy, spatial, quotes, source, inputs, day, 0, quote_mode,
                );
                let second = entry_quote(
                    &*economy, spatial, quotes, source, inputs, day, 1, quote_mode,
                );
                if first < u32::MAX && second < u32::MAX {
                    let labor = economy.ledger.number(Field::BasicLaborMinutes, source);
                    let output = economy.ledger.number(Field::BasicOutputTicks, source);
                    let seasonal = config.home.harvest_days < 365
                        && (kind == comptime!(Good::Grain as u32)
                            || kind == comptime!(Good::Flax as u32)
                            || kind == comptime!(Good::Cotton as u32)
                            || kind == comptime!(Good::Dye as u32));
                    let work_days = select(
                        seasonal,
                        crate::simulation::economy::production::farming::CropCalendar::new(config)
                            .paid_days(kind),
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
                        construction,
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
            let stock = economy.ledger.value_at(kind, source);
            let traded = kind == comptime!(Good::Grain as u32)
                || kind == comptime!(Good::Bread as u32)
                || kind == comptime!(Good::Wood as u32)
                || kind == comptime!(Good::Coal as u32)
                || (comptime!(config.economy == crate::domain::Economy::Industrial)
                    && config.market_table[kind as usize - 1] > 1);
            if traded && !lt(stock, ticks(24, 0)) {
                let quote = price * 1250 / 1000 + ceildiv(60 * wage + config.firm_rent, 24);
                let revenue = times32(24, quote);
                let cost = plus(
                    plus(times32(24, price), times32(60, wage)).value,
                    ticks(config.firm_rent, 0),
                )
                .value;
                let margin = minus(revenue, minimum(revenue, cost));
                let required = plus(
                    plus(
                        times32(24 * price, 3 + config.ring),
                        plus(times32(60, wage), construction).value,
                    )
                    .value,
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
        economy
            .ledger
            .set(Field::ReplacementCost, owner, ticks(first_cost, 0));
        economy
            .ledger
            .set(Field::SecondInputCost, owner, ticks(second_cost, 0));
    }
    selected
}

#[cube]
fn entry_quote(
    economy: &EconomyView<'_>,
    spatial: &[u32],
    quotes: &[u32],
    source: u32,
    quantity: u32,
    day: u32,
    slot: u32,
    #[comptime] mode: EntryQuoteMode,
) -> u32 {
    let config = economy.ledger.config;
    match comptime!(mode) {
        EntryQuoteMode::Cached => quotes[(2 * config.actors + 2 * source + slot) as usize],
        EntryQuoteMode::Sampled => {
            let good = economy
                .ledger
                .number_at(comptime!(Field::BasicFirstInput as u32) + slot, source);
            input_price(economy, spatial, source, good, quantity, day)
        }
    }
}

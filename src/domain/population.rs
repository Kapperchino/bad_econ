use super::{Actor, Field, Good, Position, Scenario, ScenarioConfig, State};
use crate::arithmetic::add;
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Technique {
    pub labor_minutes: u32,
    pub input_ticks: u32,
    pub output_ticks: u32,
}
impl Technique {
    pub fn new(labor_minutes: u32, input_ticks: u32, output_ticks: u32) -> Result<Self> {
        match (1..=1440).contains(&labor_minutes)
            && input_ticks <= 100
            && (1..=100).contains(&output_ticks)
        {
            true => Ok(Self {
                labor_minutes,
                input_ticks,
                output_ticks,
            }),
            false => Err(anyhow!("technique exceeds packet bounds")),
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct BehaviorPolicy {
    pub work_fatigue: u32,
    pub rest_recovery: u32,
    pub wage_increment: u32,
    pub price_response: u32,
}
impl Default for BehaviorPolicy {
    fn default() -> Self {
        Self {
            work_fatigue: 10,
            rest_recovery: 20,
            wage_increment: 1,
            price_response: 1000,
        }
    }
}
impl BehaviorPolicy {
    pub fn new(policy: Self) -> Result<Self> {
        let scaled = [
            policy.work_fatigue,
            policy.rest_recovery,
            policy.price_response,
        ];
        match scaled.into_iter().all(|value| value <= 1000) && policy.wage_increment <= 10_000 {
            true => Ok(policy),
            false => Err(anyhow!("behavior policy exceeds fixed-point bounds")),
        }
    }
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct HouseholdSeed {
    pub identity: u64,
    pub position: Position,
    pub cash: u64,
    #[serde(deserialize_with = "super::industry::goods_array")]
    pub inventories: [u64; Good::COUNT],
    pub fatigue: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct FirmSeed {
    pub identity: u64,
    pub position: Position,
    pub owner: u64,
    pub good: Good,
    pub cash: u64,
    #[serde(deserialize_with = "super::industry::goods_array")]
    pub inventories: [u64; Good::COUNT],
    pub capacity: u32,
    pub technique: Technique,
    pub wage: u32,
    pub price: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Population {
    pub households: Vec<HouseholdSeed>,
    pub firms: Vec<FirmSeed>,
}
impl Population {
    pub fn new(mut population: Self, config: &ScenarioConfig) -> Result<Self> {
        population.households.sort_by_key(|actor| actor.identity);
        population.firms.sort_by_key(|actor| actor.identity);
        let identities = population
            .households
            .iter()
            .map(|actor| actor.identity)
            .chain(population.firms.iter().map(|actor| actor.identity))
            .collect::<BTreeSet<_>>();
        let position = |p: Position| p.x < config.map_size && p.y < config.map_size;
        let households = population
            .households
            .iter()
            .all(|actor| position(actor.position) && actor.fatigue <= 1000);
        let firms = population.firms.iter().all(|actor| {
            let input = match actor.good.inputs()[0] {
                None => actor.technique.input_ticks == 0,
                Some(_) => actor.technique.input_ticks > 0,
            };
            position(actor.position)
                && population
                    .households
                    .binary_search_by_key(&actor.owner, |owner| owner.identity)
                    .is_ok()
                && (1..=10_000).contains(&actor.capacity)
                && (1..=10_000).contains(&actor.wage)
                && (config.minimum_price..=config.maximum_price).contains(&actor.price)
                && actor.technique.labor_minutes <= config.work_cap
                && Technique::new(
                    actor.technique.labor_minutes,
                    actor.technique.input_ticks,
                    actor.technique.output_ticks,
                )
                .is_ok()
                && input
        });
        let bounds = [
            population.households.len() == config.households as usize,
            population.firms.len() == config.firms as usize,
            identities.len() == population.households.len() + population.firms.len(),
            identities.iter().all(|id| (1..u64::MAX).contains(id)),
            households,
            firms,
        ];
        match bounds.into_iter().all(std::convert::identity) {
            true => Ok(population),
            false => Err(anyhow!(
                "invalid population identities, ownership, positions, or resources"
            )),
        }
    }
    pub fn index_of(&self, identity: u64) -> Option<usize> {
        self.households
            .binary_search_by_key(&identity, |actor| actor.identity)
            .ok()
            .or_else(|| {
                self.firms
                    .binary_search_by_key(&identity, |actor| actor.identity)
                    .ok()
                    .map(|index| self.households.len() + index)
            })
    }
    pub fn identity(&self, index: usize) -> Option<u64> {
        match index < self.households.len() {
            true => self.households.get(index).map(|actor| actor.identity),
            false => self
                .firms
                .get(index - self.households.len())
                .map(|actor| actor.identity),
        }
    }
    pub fn next_identity(&self) -> u64 {
        self.households
            .last()
            .map(|actor| actor.identity)
            .unwrap_or(0)
            .max(self.firms.last().map(|actor| actor.identity).unwrap_or(0))
            + 1
    }
    pub fn actor(&self, index: usize) -> Actor {
        match index < self.households.len() {
            true => {
                let seed = &self.households[index];
                Actor {
                    identity: seed.identity,
                    position: seed.position,
                    kind: 0,
                    owner: index as u32,
                    capacity: 1,
                }
            }
            false => {
                let seed = &self.firms[index - self.households.len()];
                Actor {
                    identity: seed.identity,
                    position: seed.position,
                    kind: seed.good as u32,
                    owner: self
                        .households
                        .binary_search_by_key(&seed.owner, |actor| actor.identity)
                        .expect("validated household owner") as u32,
                    capacity: seed.capacity,
                }
            }
        }
    }
    pub fn initial(&self, scenario: &Scenario) -> Result<State> {
        let count = scenario.actors();
        let mut state = State {
            day: 0,
            next_identity: self.next_identity(),
            actors: (0..count).map(|index| self.actor(index)).collect(),
            values: vec![0; scenario.fields() * count],
            initial_cash: 0,
            initial_goods: [0; Good::COUNT],
            research: None,
            people: Vec::new(),
            initial_population: 0,
            ownership: Vec::new(),
        };
        self.households
            .iter()
            .enumerate()
            .for_each(|(index, seed)| {
                state.set(Field::Cash, index, seed.cash);
                state.set(Field::Fatigue, index, seed.fatigue.into());
                Good::ALL
                    .map(Field::inventory)
                    .into_iter()
                    .zip(seed.inventories)
                    .for_each(|(field, value)| state.set(field, index, value));
            });
        self.firms.iter().enumerate().for_each(|(offset, seed)| {
            let index = offset + self.households.len();
            state.set(Field::Cash, index, seed.cash);
            state.set(Field::Wage, index, seed.wage.into());
            state.set(Field::Price, index, seed.price.into());
            state.set(
                Field::Demand,
                index,
                u64::from(seed.capacity) * u64::from(seed.technique.output_ticks),
            );
            Good::ALL
                .map(Field::inventory)
                .into_iter()
                .zip(seed.inventories)
                .for_each(|(field, value)| state.set(field, index, value));
        });
        let totals = self.totals()?;
        state.initial_cash = totals.cash;
        state.initial_goods = totals.goods;
        Ok(state)
    }
    pub fn totals(&self) -> Result<InitialTotals> {
        self.households
            .iter()
            .map(|actor| InitialTotals {
                cash: actor.cash,
                goods: actor.inventories,
            })
            .chain(self.firms.iter().map(|actor| InitialTotals {
                cash: actor.cash,
                goods: actor.inventories,
            }))
            .try_fold(InitialTotals::default(), |total, next| {
                Ok(InitialTotals {
                    cash: add(total.cash, next.cash)?,
                    goods: total
                        .goods
                        .into_iter()
                        .zip(next.goods)
                        .map(|(a, b)| add(a, b))
                        .collect::<Result<Vec<_>>>()?
                        .try_into()
                        .expect("goods count"),
                })
            })
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct InitialTotals {
    pub cash: u64,
    pub goods: [u64; Good::COUNT],
}

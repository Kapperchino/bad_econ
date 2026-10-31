use crate::arithmetic::add;
use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const MAX_LEGS: usize = 12;
pub const LEG_WORDS: usize = 8;
pub const EMPTY: u32 = u32::MAX;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Coordinate(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Authorization {
    Any,
    Equal(u64),
    AtLeast(u64),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Leg {
    pub coordinate: Coordinate,
    pub debit: u64,
    pub credit: u64,
    pub authorization: Authorization,
}
impl Leg {
    pub fn transfer(coordinate: Coordinate, debit: u64, credit: u64) -> Self {
        Self {
            coordinate,
            debit,
            credit,
            authorization: Authorization::Any,
        }
    }
    pub fn read(coordinate: Coordinate, value: u64) -> Self {
        Self {
            coordinate,
            debit: 0,
            credit: 0,
            authorization: Authorization::Equal(value),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct EventKey {
    pub day: u64,
    pub phase: u32,
    pub subphase: u32,
    pub priority: u32,
    pub identity: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Bundle {
    key: EventKey,
    legs: Vec<Leg>,
}
impl Bundle {
    pub fn new(key: EventKey, legs: impl IntoIterator<Item = Leg>) -> Result<Self> {
        let grouped = legs.into_iter().try_fold(
            BTreeMap::<Coordinate, Leg>::new(),
            |mut map, leg| -> Result<_> {
                let prior =
                    map.entry(leg.coordinate)
                        .or_insert(Leg::transfer(leg.coordinate, 0, 0));
                prior.debit = add(prior.debit, leg.debit)?;
                prior.credit = add(prior.credit, leg.credit)?;
                prior.authorization = match (prior.authorization, leg.authorization) {
                    (Authorization::Any, next) | (next, Authorization::Any) => Ok(next),
                    (a, b) if a == b => Ok(a),
                    _ => Err(anyhow!("incompatible authorization predicates")),
                }?;
                Ok(map)
            },
        )?;
        match grouped.len() <= MAX_LEGS {
            true => Ok(Self {
                key,
                legs: grouped.into_values().collect(),
            }),
            false => Err(anyhow!("bundle exceeds {MAX_LEGS} coordinates")),
        }
    }
    pub fn key(&self) -> EventKey {
        self.key
    }
    pub fn legs(&self) -> &[Leg] {
        &self.legs
    }
    pub fn apply(&self, state: &mut [u64]) -> Result<Decision> {
        let proposals = self
            .legs
            .iter()
            .map(|leg| -> Result<_> {
                let value = *state
                    .get(leg.coordinate.0 as usize)
                    .context("invalid ledger coordinate")?;
                let authorized = match leg.authorization {
                    Authorization::Any => true,
                    Authorization::Equal(n) => value == n,
                    Authorization::AtLeast(n) => value >= n,
                };
                let after = match value.checked_sub(leg.debit) {
                    Some(remainder) => Some(add(remainder, leg.credit)?),
                    None => None,
                };
                Ok(Posting {
                    coordinate: leg.coordinate,
                    value: after,
                    authorized,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let decision = match proposals.iter().all(|p| p.authorized && p.value.is_some()) {
            true => {
                proposals.iter().for_each(|p| {
                    state[p.coordinate.0 as usize] = p.value.expect("admitted posting")
                });
                Decision::Accepted
            }
            false => Decision::Rejected,
        };
        Ok(decision)
    }
    pub fn encode(&self) -> Vec<u32> {
        let mut words = vec![0; MAX_LEGS * LEG_WORDS];
        words
            .chunks_exact_mut(LEG_WORDS)
            .for_each(|leg| leg[0] = EMPTY);
        self.legs
            .iter()
            .zip(words.chunks_exact_mut(LEG_WORDS))
            .for_each(|(leg, out)| {
                let (mode, guard) = match leg.authorization {
                    Authorization::Any => (0, 0),
                    Authorization::Equal(n) => (1, n),
                    Authorization::AtLeast(n) => (2, n),
                };
                out.copy_from_slice(&[
                    leg.coordinate.0,
                    leg.debit as u32,
                    (leg.debit >> 32) as u32,
                    leg.credit as u32,
                    (leg.credit >> 32) as u32,
                    mode,
                    guard as u32,
                    (guard >> 32) as u32,
                ]);
            });
        words
    }
}
struct Posting {
    coordinate: Coordinate,
    value: Option<u64>,
    authorized: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Decision {
    Accepted,
    Rejected,
}

#[derive(Clone, Debug)]
pub struct Schedule {
    pub frontiers: Vec<Vec<usize>>,
}
impl Schedule {
    pub fn new(events: &[Bundle]) -> Result<Self> {
        match events.windows(2).all(|pair| pair[0].key < pair[1].key) {
            false => Err(anyhow!(
                "event identities must be unique and canonically ordered"
            )),
            true => {
                let mut incidences = events
                    .iter()
                    .enumerate()
                    .flat_map(|(index, bundle)| {
                        bundle.legs.iter().map(move |leg| (leg.coordinate, index))
                    })
                    .collect::<Vec<_>>();
                incidences.sort_unstable();
                let mut predecessors = vec![Vec::<usize>::new(); events.len()];
                incidences
                    .windows(2)
                    .filter(|pair| pair[0].0 == pair[1].0)
                    .for_each(|pair| predecessors[pair[1].1].push(pair[0].1));
                let mut levels = vec![0; events.len()];
                (0..events.len()).for_each(|event| {
                    levels[event] = predecessors[event]
                        .iter()
                        .map(|&p| levels[p] + 1)
                        .max()
                        .unwrap_or(0)
                });
                let mut frontiers = vec![Vec::new(); levels.iter().max().map_or(0, |n| n + 1)];
                levels
                    .iter()
                    .enumerate()
                    .for_each(|(event, &level)| frontiers[level].push(event));
                Ok(Self { frontiers })
            }
        }
    }
}

pub fn serial(state: &mut [u64], events: &[Bundle]) -> Result<Vec<Decision>> {
    let _schedule = Schedule::new(events)?;
    events.iter().map(|event| event.apply(state)).collect()
}

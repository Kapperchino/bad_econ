use crate::arithmetic::add;
use anyhow::{Context, Result, anyhow};
use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, mem::offset_of};

pub const MAX_LEGS: usize = 12;
pub const LEG_WORDS: usize = size_of::<LegLayout>() / size_of::<u32>();
pub const EMPTY: u32 = u32::MAX;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct TickLayout {
    low: u32,
    high: u32,
}
impl TickLayout {
    pub const LOW: usize = offset_of!(Self, low) / size_of::<u32>();
    pub const HIGH: usize = offset_of!(Self, high) / size_of::<u32>();
}
impl From<u64> for TickLayout {
    fn from(value: u64) -> Self {
        Self {
            low: value as u32,
            high: (value >> u32::BITS) as u32,
        }
    }
}
impl From<TickLayout> for u64 {
    fn from(value: TickLayout) -> Self {
        Self::from(value.low) | (Self::from(value.high) << u32::BITS)
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct LegLayout {
    coordinate: u32,
    debit: TickLayout,
    credit: TickLayout,
    authorization: u32,
    guard: TickLayout,
}
impl LegLayout {
    pub const COORDINATE: usize = offset_of!(Self, coordinate) / size_of::<u32>();
    pub const DEBIT_LOW: usize = offset_of!(Self, debit.low) / size_of::<u32>();
    pub const DEBIT_HIGH: usize = offset_of!(Self, debit.high) / size_of::<u32>();
    pub const CREDIT_LOW: usize = offset_of!(Self, credit.low) / size_of::<u32>();
    pub const CREDIT_HIGH: usize = offset_of!(Self, credit.high) / size_of::<u32>();
    pub const AUTHORIZATION: usize = offset_of!(Self, authorization) / size_of::<u32>();
    pub const GUARD_LOW: usize = offset_of!(Self, guard.low) / size_of::<u32>();
    pub const GUARD_HIGH: usize = offset_of!(Self, guard.high) / size_of::<u32>();

    fn new(leg: &Leg) -> Self {
        let authorization = EncodedAuthorization::from(leg.authorization);
        Self {
            coordinate: leg.coordinate.0,
            debit: leg.debit.into(),
            credit: leg.credit.into(),
            authorization: authorization.flags.bits(),
            guard: authorization.guard.into(),
        }
    }

    fn empty() -> Self {
        Self {
            coordinate: EMPTY,
            ..Self::zeroed()
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Coordinate(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Authorization {
    Any,
    Equal(u64),
    AtLeast(u64),
}

bitflags::bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) struct AuthorizationFlags: u32 {
        const EQUAL = 1 << 0;
        const AT_LEAST = 1 << 1;
    }
}

struct EncodedAuthorization {
    flags: AuthorizationFlags,
    guard: u64,
}
impl From<Authorization> for EncodedAuthorization {
    fn from(authorization: Authorization) -> Self {
        match authorization {
            Authorization::Any => Self {
                flags: AuthorizationFlags::empty(),
                guard: 0,
            },
            Authorization::Equal(guard) => Self {
                flags: AuthorizationFlags::EQUAL,
                guard,
            },
            Authorization::AtLeast(guard) => Self {
                flags: AuthorizationFlags::AT_LEAST,
                guard,
            },
        }
    }
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
#[serde(try_from = "BundleDefinition", into = "BundleDefinition")]
pub struct Bundle {
    key: EventKey,
    legs: Vec<Leg>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct BundleDefinition {
    key: EventKey,
    legs: Vec<Leg>,
}

impl TryFrom<BundleDefinition> for Bundle {
    type Error = anyhow::Error;

    fn try_from(value: BundleDefinition) -> Result<Self> {
        Self::new(value.key, value.legs)
    }
}

impl From<Bundle> for BundleDefinition {
    fn from(value: Bundle) -> Self {
        Self {
            key: value.key,
            legs: value.legs,
        }
    }
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
        let legs: [LegLayout; MAX_LEGS] = std::array::from_fn(|index| {
            self.legs
                .get(index)
                .map_or_else(LegLayout::empty, LegLayout::new)
        });
        bytemuck::cast_slice(&legs).to_vec()
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

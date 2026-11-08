use super::{ActorRows, Field, Good, Position, State};
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RailMaterials {
    pub coal: u32,
    pub steel: u32,
    pub locomotives: u32,
}

impl RailMaterials {
    pub fn quantities(self) -> [u32; 3] {
        [self.coal, self.steel, self.locomotives]
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RailwayCompany {
    pub firm: u64,
    pub freight_rate: u32,
    pub access_fee: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RailSegment {
    pub company: u64,
    pub from: Position,
    pub to: Position,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct RailwayConfig {
    pub speed_multiplier: u32,
    pub length_unit: u32,
    pub construction: RailMaterials,
    pub maintenance: RailMaterials,
    pub companies: Vec<RailwayCompany>,
    pub segments: Vec<RailSegment>,
}

impl Default for RailwayConfig {
    fn default() -> Self {
        Self {
            speed_multiplier: 8,
            length_unit: 100,
            construction: RailMaterials {
                coal: 20,
                steel: 40,
                locomotives: 5,
            },
            maintenance: RailMaterials {
                coal: 2,
                steel: 1,
                locomotives: 1,
            },
            companies: Vec::new(),
            segments: Vec::new(),
        }
    }
}

impl RailwayConfig {
    pub fn generated(actors: &ActorRows, map_size: u32) -> Self {
        let mut config = Self {
            length_unit: (map_size / 10).max(1),
            ..Self::default()
        };
        let sites = actors
            .iter()
            .filter(|actor| actor.kind > 0)
            .collect::<Vec<_>>();
        let firms = sites
            .iter()
            .filter(|actor| actor.kind == Good::Railway as u32)
            .collect::<Vec<_>>();
        let station = |position: Position| Position {
            x: (position.x * 8 / map_size * map_size / 8 + map_size / 16).min(map_size - 1),
            y: (position.y * 8 / map_size * map_size / 8 + map_size / 16).min(map_size - 1),
        };
        config.companies = firms
            .iter()
            .map(|firm| RailwayCompany {
                firm: firm.identity,
                freight_rate: 1,
                access_fee: 1,
            })
            .collect();
        for (index, firm) in firms.iter().enumerate() {
            let origin = station(firm.position);
            let mut neighbors = sites
                .iter()
                .filter(|actor| {
                    actor.kind > 0
                        && actor.identity != firm.identity
                        && actor.position != firm.position
                })
                .collect::<Vec<_>>();
            neighbors.sort_by_key(|actor| (firm.position.distance(actor.position), actor.identity));
            let destinations = neighbors
                .into_iter()
                .take(6)
                .map(|actor| actor.position)
                .chain(firms.get(index + 1).map(|actor| actor.position))
                .map(station)
                .filter(|position| *position != origin)
                .map(|position| (position.x, position.y))
                .collect::<BTreeSet<_>>();
            config
                .segments
                .extend(destinations.into_iter().map(|(x, y)| RailSegment {
                    company: firm.identity,
                    from: origin,
                    to: Position { x, y },
                }));
        }
        config
    }

    pub fn new(config: Self, actors: &ActorRows, map_size: u32) -> Result<Self> {
        let identities = config
            .companies
            .iter()
            .map(|company| company.firm)
            .collect::<BTreeSet<_>>();
        let rates = config.companies.iter().all(|company| {
            (1..=1000).contains(&company.freight_rate)
                && (1..=1000).contains(&company.access_fee)
                && actors.iter().any(|actor| {
                    actor.identity == company.firm && actor.kind == Good::Railway as u32
                })
        });
        let mut edges = BTreeSet::new();
        let segments = config.segments.iter().all(|segment| {
            let from = (segment.from.x, segment.from.y);
            let to = (segment.to.x, segment.to.y);
            identities.contains(&segment.company)
                && from != to
                && segment.from.x < map_size
                && segment.from.y < map_size
                && segment.to.x < map_size
                && segment.to.y < map_size
                && edges.insert((segment.company, from.min(to), from.max(to)))
        });
        let materials = config
            .construction
            .quantities()
            .into_iter()
            .zip(config.maintenance.quantities())
            .all(|(build, upkeep)| (1..build).contains(&upkeep) && build <= 1000);
        let units = config
            .segments
            .iter()
            .map(|segment| {
                (u64::from(segment.from.x.abs_diff(segment.to.x))
                    + u64::from(segment.from.y.abs_diff(segment.to.y)))
                .div_ceil(u64::from(config.length_unit.max(1)))
            })
            .sum::<u64>();
        match identities.len() == config.companies.len()
            && rates
            && segments
            && materials
            && (2..=100).contains(&config.speed_multiplier)
            && config.length_unit > 0
            && units <= 100_000
            && config.stations().len() <= 256
        {
            true => Ok(config),
            false => Err(anyhow!(
                "invalid railway companies, track endpoints, rates, materials, or network bounds"
            )),
        }
    }

    pub fn stations(&self) -> Vec<Position> {
        self.segments
            .iter()
            .flat_map(|segment| [segment.from, segment.to])
            .map(|position| (position.x, position.y))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .map(|(x, y)| Position { x, y })
            .collect()
    }

    pub fn audit(&self, state: &State) -> Result<()> {
        let mut counts = BTreeMap::<u64, u64>::new();
        for segment in &self.segments {
            *counts.entry(segment.company).or_default() += 1;
        }
        for (index, actor) in state.actors.iter().enumerate() {
            let segments = counts.get(&actor.identity).copied().unwrap_or(0);
            let built = state.get(Field::RailBuilt, index);
            let operating = state.get(Field::RailOperating, index);
            match built <= segments
                && (operating == 0 || operating == built)
                && state.get(Field::FreightEscrow, index) == 0
            {
                true => (),
                false => Err(anyhow!(
                    "invalid railway construction, operation, or unpaid freight"
                ))?,
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum RailStatus {
    Planned,
    Operating,
    Suspended,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct RailwayBusiness {
    pub built_segments: u64,
    pub operating_segments: u64,
    pub daily_receipts: u64,
}

impl RailwayBusiness {
    pub fn status(&self) -> RailStatus {
        match (self.built_segments, self.operating_segments) {
            (0, _) => RailStatus::Planned,
            (_, 0) => RailStatus::Suspended,
            _ => RailStatus::Operating,
        }
    }
}

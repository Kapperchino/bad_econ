use super::{ActorRows, Field, Good, Position, Scenario, State, property::PropertyConfig};
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

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub enum RailSharing {
    #[default]
    Shared,
    Private,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RailAgreement {
    pub owner: u64,
    pub carrier: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RailConnection {
    pub firm: u64,
    pub company: u64,
    pub station: Position,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RailwayCompany {
    pub firm: u64,
    pub freight_rate: u32,
    pub access_fee: u32,
    #[serde(default)]
    pub sharing: RailSharing,
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
    pub units_per_level: u32,
    pub construction: RailMaterials,
    pub maintenance: RailMaterials,
    pub companies: Vec<RailwayCompany>,
    pub segments: Vec<RailSegment>,
    pub connections: Vec<RailConnection>,
    pub sharing_agreements: Vec<RailAgreement>,
}

impl Default for RailwayConfig {
    fn default() -> Self {
        Self {
            speed_multiplier: 8,
            length_unit: 100,
            units_per_level: 100,
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
            connections: Vec::new(),
            sharing_agreements: Vec::new(),
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
                sharing: RailSharing::Shared,
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
        let stations = config.stations();
        config.connections = sites
            .iter()
            .filter(|firm| firm.capacity >= 20 && firm.kind != Good::Railway as u32)
            .filter(|firm| !stations.contains(&firm.position))
            .take(256usize.saturating_sub(stations.len()))
            .filter_map(|firm| {
                stations
                    .iter()
                    .min_by_key(|station| firm.position.distance(**station))
                    .and_then(|station| {
                        config
                            .segments
                            .iter()
                            .find(|segment| segment.from == *station || segment.to == *station)
                            .map(|segment| RailConnection {
                                firm: firm.identity,
                                company: segment.company,
                                station: *station,
                            })
                    })
            })
            .collect();
        config
    }

    pub fn new(mut config: Self, actors: &ActorRows, map_size: u32) -> Result<Self> {
        let identities = config
            .companies
            .iter()
            .map(|company| company.firm)
            .collect::<BTreeSet<_>>();
        let stations = config.stations();
        let mut connections = BTreeSet::new();
        for connection in &config.connections {
            let firm = actors
                .iter()
                .find(|actor| actor.identity == connection.firm && actor.kind > 0)
                .ok_or_else(|| anyhow!("rail connection requires an existing firm"))?;
            match identities.contains(&connection.company)
                && stations.contains(&connection.station)
                && connections.insert(connection.firm)
            {
                true => (),
                false => Err(anyhow!(
                    "rail connection requires a known operator, a network station, and a unique firm"
                ))?,
            }
            let existing = config.segments.iter().any(|segment| {
                segment.company == connection.company
                    && ((segment.from == firm.position && segment.to == connection.station)
                        || (segment.to == firm.position && segment.from == connection.station))
            });
            if firm.position != connection.station && !existing {
                config.segments.push(RailSegment {
                    company: connection.company,
                    from: firm.position,
                    to: connection.station,
                });
            }
        }
        let mut agreements = BTreeSet::new();
        let sharing = config.sharing_agreements.iter().all(|agreement| {
            identities.contains(&agreement.owner)
                && identities.contains(&agreement.carrier)
                && agreement.owner != agreement.carrier
                && agreements.insert((agreement.owner, agreement.carrier))
        });
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
            && sharing
            && segments
            && materials
            && (2..=100).contains(&config.speed_multiplier)
            && config.length_unit > 0
            && (1..=100_000).contains(&config.units_per_level)
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
    pub daily_capacity: u64,
    pub available_units: u64,
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

#[repr(u32)]
#[derive(Clone, Copy, Debug)]
pub enum RailCapacityField {
    Level,
    Available,
    Demand,
}

impl RailCapacityField {
    pub const COUNT: usize = 3;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RailRegion {
    pub company: usize,
    pub region: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RailGeography {
    pub regions: Vec<RailRegion>,
    pub segments: Vec<Vec<usize>>,
}

impl RailGeography {
    pub fn new(config: &RailwayConfig, property: Option<&PropertyConfig>) -> Result<Self> {
        let mut geography = Self::default();
        let mut regions = BTreeMap::new();
        let mut records = BTreeMap::new();
        for segment in &config.segments {
            let company = config
                .companies
                .iter()
                .position(|company| company.firm == segment.company)
                .ok_or_else(|| anyhow!("unknown railway company"))?;
            let corner = Position {
                x: segment.to.x,
                y: segment.from.y,
            };
            let mut positions = Vec::new();
            for (from, to) in [(segment.from, corner), (corner, segment.to)] {
                let horizontal = from.y == to.y;
                let coordinate = |position: Position| match horizontal {
                    true => position.x,
                    false => position.y,
                };
                let low = coordinate(from).min(coordinate(to));
                let high = coordinate(from).max(coordinate(to));
                let boundaries = match property {
                    Some(property) => property
                        .regions
                        .iter()
                        .flat_map(|region| [coordinate(region.lower), coordinate(region.upper)])
                        .collect::<BTreeSet<_>>(),
                    None => ((low / config.length_unit + 1)..=high / config.length_unit)
                        .map(|cell| cell * config.length_unit)
                        .collect(),
                };
                let mut points = boundaries
                    .into_iter()
                    .filter(|boundary| low < *boundary && *boundary <= high)
                    .flat_map(|boundary| [boundary - 1, boundary])
                    .chain([low, high])
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect::<Vec<_>>();
                if coordinate(from) > coordinate(to) {
                    points.reverse();
                }
                positions.extend(points.into_iter().map(|point| match horizontal {
                    true => Position {
                        x: point,
                        y: from.y,
                    },
                    false => Position {
                        x: from.x,
                        y: point,
                    },
                }));
            }
            let mut path = Vec::new();
            for position in positions {
                let key = match property {
                    Some(property) => (property.region(position)? as u32, 0),
                    None => (
                        position.x / config.length_unit,
                        position.y / config.length_unit,
                    ),
                };
                let next_region = regions.len();
                let region = *regions.entry(key).or_insert(next_region);
                let next_record = geography.regions.len();
                let record = *records.entry((company, region)).or_insert_with(|| {
                    geography.regions.push(RailRegion { company, region });
                    next_record
                });
                if path.last() != Some(&record) {
                    path.push(record);
                }
            }
            geography.segments.push(path);
        }
        match geography.regions.len() <= 100_000
            && geography.segments.iter().map(Vec::len).sum::<usize>() <= 200_000
        {
            true => Ok(geography),
            false => Err(anyhow!("railway network exceeds regional capacity bounds")),
        }
    }

    pub fn blocks(&self, initial_actors: usize) -> usize {
        self.regions.len().div_ceil(initial_actors)
    }

    pub fn coordinate(
        &self,
        scenario: &Scenario,
        state: &State,
        field: RailCapacityField,
        region: usize,
    ) -> usize {
        let stride = scenario.actors();
        let base = super::BASE_FIELDS + scenario.ring() * Good::COUNT;
        (base + field as usize * self.blocks(stride) + region / stride) * state.actors.len()
            + region % stride
    }

    pub fn audit(&self, scenario: &Scenario, state: &State) -> Result<()> {
        let mut stocks = vec![0u64; scenario.config().railway.companies.len()];
        for (region, record) in self.regions.iter().enumerate() {
            let level =
                state.values[self.coordinate(scenario, state, RailCapacityField::Level, region)];
            let stock = state.values
                [self.coordinate(scenario, state, RailCapacityField::Available, region)];
            match level <= 1000
                && stock <= level * u64::from(scenario.config().railway.units_per_level)
            {
                true => stocks[record.company] += stock,
                false => Err(anyhow!("invalid regional railway capacity"))?,
            }
        }
        for (company, stock) in scenario.config().railway.companies.iter().zip(stocks) {
            let firm = state
                .actors
                .iter()
                .position(|actor| actor.identity == company.firm)
                .ok_or_else(|| anyhow!("unknown railway operator"))?;
            match state.get(Field::Railway, firm) == stock {
                true => (),
                false => Err(anyhow!(
                    "regional rail supply disagrees with railway inventory"
                ))?,
            }
        }
        Ok(())
    }
}

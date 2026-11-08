use super::{
    Field, Good, GoodKind, Position, Scenario, State,
    enterprise::{EnterpriseModel, EnterpriseState},
};
use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const CONSTRUCTION_WORK_PER_PLOT: u32 = 20;

#[repr(u32)]
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum PropertyUse {
    Housing = 1,
    Farm,
    Factory,
    Store,
}

impl PropertyUse {
    pub fn for_site(state: &State, site: usize) -> Result<Self> {
        match (
            state.actors.at(site).kind,
            state.get(Field::EnterpriseModel, site),
        ) {
            (0, _) => Ok(Self::Housing),
            (kind, _) if kind == Good::Housing as u32 => Ok(Self::Housing),
            (_, model) if model == EnterpriseModel::Distributor as u64 => Ok(Self::Store),
            (kind, _) => match Good::from_index(kind)?.kind() {
                GoodKind::Natural => Ok(Self::Farm),
                GoodKind::Manufactured => Ok(Self::Factory),
            },
        }
    }
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub enum ProjectStatus {
    #[default]
    None,
    Requested,
    Building,
    Completed,
    Cancelled,
}

impl ProjectStatus {
    pub fn from_index(index: u64) -> Result<Self> {
        match index {
            0 => Ok(Self::None),
            1 => Ok(Self::Requested),
            2 => Ok(Self::Building),
            3 => Ok(Self::Completed),
            4 => Ok(Self::Cancelled),
            _ => Err(anyhow!("unknown construction project state")),
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum Development {
    NewBuilding,
    Extension,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PropertyRegion {
    pub lower: Position,
    pub upper: Position,
    #[serde(alias = "space")]
    pub plots: u32,
    pub landowner: u64,
}

impl PropertyRegion {
    pub fn contains(&self, position: Position) -> bool {
        (self.lower.x..self.upper.x).contains(&position.x)
            && (self.lower.y..self.upper.y).contains(&position.y)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Building {
    pub site: u64,
    #[serde(alias = "space")]
    pub plots: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ConstructionOrder {
    pub landowner: u64,
    pub site: u64,
    #[serde(alias = "space")]
    pub plots: u32,
    pub work: u32,
    pub development: Development,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PropertyConfig {
    pub regions: Vec<PropertyRegion>,
    #[serde(default)]
    pub buildings: Vec<Building>,
    #[serde(default)]
    pub orders: Vec<ConstructionOrder>,
}

impl PropertyConfig {
    pub fn new(config: Self, map_size: u32) -> Result<Self> {
        let rectangles = config.regions.iter().all(|region| {
            region.lower.x < region.upper.x
                && region.lower.y < region.upper.y
                && region.upper.x <= map_size
                && region.upper.y <= map_size
                && region.plots > 0
                && region.landowner > 0
        });
        let separate = config.regions.iter().enumerate().all(|(index, region)| {
            config.regions[..index].iter().all(|other| {
                region.upper.x <= other.lower.x
                    || other.upper.x <= region.lower.x
                    || region.upper.y <= other.lower.y
                    || other.upper.y <= region.lower.y
            })
        });
        let area = config
            .regions
            .iter()
            .map(|region| {
                u128::from(region.upper.x.saturating_sub(region.lower.x))
                    * u128::from(region.upper.y.saturating_sub(region.lower.y))
            })
            .sum::<u128>();
        let unique_buildings = config
            .buildings
            .iter()
            .map(|building| building.site)
            .collect::<BTreeSet<_>>();
        let unique_orders = config
            .orders
            .iter()
            .map(|order| order.site)
            .collect::<BTreeSet<_>>();
        let orders = config
            .orders
            .iter()
            .all(|order| order.plots > 0 && order.work > 0 && order.landowner > 0);
        match rectangles
            && separate
            && area == u128::from(map_size).pow(2)
            && unique_buildings.len() == config.buildings.len()
            && unique_orders.len() == config.orders.len()
            && orders
        {
            true => Ok(config),
            false => Err(anyhow!(
                "property regions must partition the map with positive plots; building sites and valid construction orders must be unique"
            )),
        }
    }

    pub fn region(&self, position: Position) -> Result<usize> {
        self.regions
            .iter()
            .position(|region| region.contains(position))
            .context("property site is outside its regions")
    }

    pub fn initialize(&self, state: &mut State) -> Result<()> {
        let rental_regions = state
            .actors
            .iter()
            .filter(|actor| actor.kind == Good::Housing as u32)
            .map(|actor| self.region(actor.position))
            .collect::<Result<BTreeSet<_>>>()?;
        for region in &self.regions {
            match state
                .actors
                .iter()
                .any(|actor| actor.identity == region.landowner && actor.kind == 0)
            {
                true => (),
                false => Err(anyhow!(
                    "property region references a missing landowner household"
                ))?,
            }
        }
        for index in 0..state.actors.len() {
            let actor = state.actors.at(index);
            state.set(
                Field::PropertyRegion,
                index,
                self.region(actor.position)? as u64 + 1,
            );
            state.set(
                Field::PropertyUse,
                index,
                PropertyUse::for_site(state, index)? as u64,
            );
            let plots = match (
                actor.kind,
                rental_regions.contains(&self.region(actor.position)?),
                state.get(Field::LandPlots, index),
            ) {
                (0, true, 0) => 0,
                _ => 1,
            };
            state.set(Field::OccupiedPlots, index, plots);
            state.set(
                Field::WorkersPerPlot,
                index,
                match actor.kind {
                    0 => 0,
                    _ => u64::from(actor.capacity),
                },
            );
        }
        for building in &self.buildings {
            let site = state
                .actors
                .iter()
                .position(|actor| actor.identity == building.site)
                .context("building references a missing site")?;
            state.set(Field::OccupiedPlots, site, building.plots.into());
        }
        for order in &self.orders {
            let site = state
                .actors
                .iter()
                .position(|actor| actor.identity == order.site)
                .context("construction order references a missing site")?;
            let actor = state.actors.at(site);
            let region = &self.regions[self.region(actor.position)?];
            let owner = match actor.kind {
                0 => actor.identity,
                _ => state.actors.at(actor.owner as usize).identity,
            };
            match region.landowner == order.landowner && owner == order.landowner {
                true => (),
                false => Err(anyhow!(
                    "construction requires the region landowner to own its site"
                ))?,
            }
            let plots = match order.development {
                Development::NewBuilding => u64::from(order.plots),
                Development::Extension => {
                    state.get(Field::OccupiedPlots, site) + u64::from(order.plots)
                }
            };
            match plots * state.get(Field::WorkersPerPlot, site) <= 10_000 {
                true => (),
                false => Err(anyhow!("construction extension exceeds firm capacity"))?,
            }
            match order.development {
                Development::NewBuilding => {
                    state.set(Field::OccupiedPlots, site, 0);
                    if actor.kind > 0 {
                        state.set(
                            Field::EnterpriseStatus,
                            site,
                            EnterpriseState::Forming as u64,
                        );
                    }
                }
                Development::Extension => match state.get(Field::OccupiedPlots, site) > 0 {
                    true => (),
                    false => Err(anyhow!("an extension requires an existing building"))?,
                },
            }
            state.set(Field::RequestedPlots, site, order.plots.into());
            state.set(Field::ConstructionRemaining, site, order.work.into());
            state.set(Field::ProjectStatus, site, ProjectStatus::Requested as u64);
        }
        for site in 0..state.actors.len() {
            let mut actor = state.actors.at(site);
            let capacity = match actor.kind {
                0 => u64::from(actor.capacity),
                _ => {
                    state.get(Field::WorkersPerPlot, site)
                        * state.get(Field::OccupiedPlots, site).max(1)
                }
            };
            match capacity <= 10_000
                && (capacity > 0
                    || matches!(actor.kind, kind if kind == Good::Housing as u32 || kind == Good::Railway as u32))
            {
                true => actor.capacity = capacity as u32,
                false => Err(anyhow!("building plots exceed staffing capacity"))?,
            }
            state.actors.set(site, actor);
        }
        for (region_index, region) in self.regions.iter().enumerate() {
            let occupied = (0..state.actors.len())
                .filter(|&actor| state.get(Field::PropertyRegion, actor) == region_index as u64 + 1)
                .map(|actor| state.get(Field::OccupiedPlots, actor))
                .sum::<u64>();
            let mut free = u64::from(region.plots)
                .checked_sub(occupied)
                .context("initial buildings exceed regional plots")?;
            for actor in 0..state.actors.len() {
                if state.actors.at(actor).kind == 0
                    && region.contains(state.actors.at(actor).position)
                {
                    let plots = state.get(Field::LandPlots, actor);
                    let retained = match plots <= free {
                        true => plots,
                        false => 0,
                    };
                    state.set(Field::LandPlots, actor, retained);
                    state.set(Field::DisplacedPlots, actor, plots - retained);
                    free -= retained;
                    let work = match retained {
                        0 => 0,
                        _ => state.get(Field::CropWork, actor),
                    };
                    state.set(Field::CropWork, actor, work);
                }
            }
        }
        PropertySnapshot::new(self, state)?;
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct RegionAllocation {
    pub plots: u64,
    pub housing: u64,
    pub farms: u64,
    pub factories: u64,
    pub stores: u64,
    pub reserved: u64,
    pub subsistence: u64,
    pub free: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct PropertySnapshot {
    pub regions: Vec<RegionAllocation>,
    pub displaced_plots: u64,
}

impl PropertySnapshot {
    pub fn new(config: &PropertyConfig, state: &State) -> Result<Self> {
        let mut regions = config
            .regions
            .iter()
            .map(|region| RegionAllocation {
                plots: region.plots.into(),
                housing: 0,
                farms: 0,
                factories: 0,
                stores: 0,
                reserved: 0,
                subsistence: 0,
                free: 0,
            })
            .collect::<Vec<_>>();
        let mut displaced_plots = 0;
        for index in 0..state.actors.len() {
            let actor = state.actors.at(index);
            match state.get(Field::PropertyUse, index)
                == PropertyUse::for_site(state, index)? as u64
            {
                true => (),
                false => Err(anyhow!("property use does not match its site"))?,
            }
            let number = state.get(Field::PropertyRegion, index);
            let plots = state.get(Field::OccupiedPlots, index);
            let reserved = state.get(Field::ReservedPlots, index);
            let requested = state.get(Field::RequestedPlots, index);
            let remaining = state.get(Field::ConstructionRemaining, index);
            let status = state.get(Field::ProjectStatus, index);
            let project = match status {
                0 | 3 | 4 => requested == 0 && reserved == 0 && remaining == 0,
                1 => requested > 0 && reserved == 0 && remaining > 0,
                2 => requested == 0 && reserved > 0 && remaining > 0,
                _ => false,
            };
            match project {
                true => (),
                false => Err(anyhow!("invalid property project state"))?,
            }
            let built = actor.kind == 0
                || plots > 0
                || state.get(Field::EnterpriseStatus, index) != EnterpriseState::Operating as u64;
            match built {
                true => (),
                false => Err(anyhow!("an operating firm requires a completed building"))?,
            }
            let workers = state.get(Field::WorkersPerPlot, index);
            let expected = u128::from(workers) * u128::from(plots.max(1));
            let capacity = u64::from(actor.capacity) + state.get(Field::PendingCapacity, index);
            let staffing = match actor.kind {
                0 => workers == 0,
                kind if kind == Good::Housing as u32 || kind == Good::Railway as u32 => {
                    workers == 0 && capacity == 0
                }
                _ => workers > 0 && expected <= 10_000 && u128::from(capacity) == expected,
            };
            match staffing {
                true => (),
                false => Err(anyhow!(
                    "facility staffing does not match its developed plots"
                ))?,
            }
            let produced = u128::from(state.get(Field::Produced, index));
            let unscaled = u128::from(state.get(Field::UnscaledProduced, index));
            let scaled = state.get(Field::ScaleRemainder, index) < 1000
                && produced >= unscaled
                && produced * 1000 <= unscaled * 1150;
            match scaled {
                true => (),
                false => Err(anyhow!("invalid scale production accounting"))?,
            }
            let region = number
                .checked_sub(1)
                .and_then(|number| regions.get_mut(number as usize))
                .context("invalid property region")?;
            match state.get(Field::PropertyUse, index) {
                1 => region.housing = super::add(region.housing, plots)?,
                2 => region.farms = super::add(region.farms, plots)?,
                3 => region.factories = super::add(region.factories, plots)?,
                4 => region.stores = super::add(region.stores, plots)?,
                _ => Err(anyhow!("invalid property use"))?,
            }
            region.reserved = super::add(region.reserved, reserved)?;
            let current = config.region(state.actors.at(index).position)?;
            regions[current].subsistence = super::add(
                regions[current].subsistence,
                state.get(Field::LandPlots, index),
            )?;
            displaced_plots = super::add(displaced_plots, state.get(Field::DisplacedPlots, index))?;
        }
        for region in &mut regions {
            let occupied = u128::from(region.housing)
                + u128::from(region.farms)
                + u128::from(region.factories)
                + u128::from(region.stores)
                + u128::from(region.reserved)
                + u128::from(region.subsistence);
            region.free = u64::try_from(
                u128::from(region.plots)
                    .checked_sub(occupied)
                    .context("regional land capacity exceeded")?,
            )?;
        }
        Ok(Self {
            regions,
            displaced_plots,
        })
    }

    pub fn from_state(scenario: &Scenario, state: &State) -> Result<Option<Self>> {
        scenario
            .config()
            .property
            .as_ref()
            .map(|config| Self::new(config, state))
            .transpose()
    }
}

use super::{Experiment, Good, GoodKind, Position, ScenarioConfig};
use crate::arithmetic::DrawKey;
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub enum ResourceMap {
    #[default]
    Generated,
    Explicit {
        regions: Vec<ResourceRegion>,
    },
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ResourceRegion {
    good: Good,
    lower: Position,
    upper: Position,
}

impl ResourceRegion {
    pub fn new(good: Good, lower: Position, upper: Position, map_size: u32) -> Result<Self> {
        match good.kind() {
            GoodKind::Manufactured => {
                Err(anyhow!("resource regions require a natural good: {good:?}"))
            }
            GoodKind::Natural => match lower.x < upper.x
                && lower.y < upper.y
                && upper.x <= map_size
                && upper.y <= map_size
            {
                true => Ok(Self { good, lower, upper }),
                false => Err(anyhow!(
                    "{good:?} resource region must be a nonempty rectangle within map size {map_size}"
                )),
            },
        }
    }

    pub fn good(&self) -> Good {
        self.good
    }

    pub fn lower(&self) -> Position {
        self.lower
    }

    pub fn upper(&self) -> Position {
        self.upper
    }

    pub fn contains(&self, position: Position) -> bool {
        (self.lower.x..self.upper.x).contains(&position.x)
            && (self.lower.y..self.upper.y).contains(&position.y)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProductionSite {
    UnrestrictedManufacturing,
    ResourceAvailable,
    ResourceMissing,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ResourceGeography {
    regions: Vec<ResourceRegion>,
}

impl ResourceGeography {
    pub fn new(config: &ScenarioConfig) -> Result<Self> {
        let regions = match &config.resources {
            ResourceMap::Explicit { regions } => regions
                .iter()
                .map(|region| {
                    ResourceRegion::new(region.good, region.lower, region.upper, config.map_size)
                })
                .collect::<Result<Vec<_>>>()?,
            ResourceMap::Generated => {
                let side = (config.map_size / 4).max(1);
                Good::ALL
                    .into_iter()
                    .filter(|good| good.kind() == GoodKind::Natural)
                    .flat_map(|good| (0..4).map(move |patch| (good, patch)))
                    .map(|(good, patch)| {
                        let draw = DrawKey {
                            seed: config.seed,
                            day: 0,
                            phase: 0,
                            actor: good as u64,
                            purpose: 1,
                            index: patch,
                        }
                        .draw();
                        let lower = match (config.experiment, patch) {
                            (Experiment::Concentrated, 0) => Position {
                                x: config.map_size / 2 - side / 2,
                                y: config.map_size / 2 - side / 2,
                            },
                            _ => Position {
                                x: draw as u32 % (config.map_size - side + 1),
                                y: (draw >> 32) as u32 % (config.map_size - side + 1),
                            },
                        };
                        ResourceRegion::new(
                            good,
                            lower,
                            Position {
                                x: lower.x + side,
                                y: lower.y + side,
                            },
                            config.map_size,
                        )
                    })
                    .collect::<Result<Vec<_>>>()?
            }
        };
        let geography = Self { regions };
        if config.population.is_none() {
            let count = match config.economy {
                super::Economy::Agrarian => 3,
                super::Economy::Industrial => Good::COUNT,
            };
            for good in Good::ALL
                .into_iter()
                .take(count.min(config.firms as usize))
                .filter(|good| good.kind() == GoodKind::Natural)
            {
                match geography.regions.iter().any(|region| region.good == good) {
                    true => Ok(()),
                    false => Err(anyhow!(
                        "automatically generated {good:?} producer has no matching resource region"
                    )),
                }?;
            }
        }
        Ok(geography)
    }

    pub fn regions(&self) -> &[ResourceRegion] {
        &self.regions
    }

    pub fn has_resource(&self, good: Good, position: Position) -> bool {
        self.regions
            .iter()
            .any(|region| region.good == good && region.contains(position))
    }

    pub fn production_site(&self, good: Good, position: Position) -> ProductionSite {
        match good.kind() {
            GoodKind::Manufactured => ProductionSite::UnrestrictedManufacturing,
            GoodKind::Natural => match self.has_resource(good, position) {
                true => ProductionSite::ResourceAvailable,
                false => ProductionSite::ResourceMissing,
            },
        }
    }

    pub fn place(
        &self,
        good: Good,
        identity: u64,
        config: &ScenarioConfig,
        original: Position,
    ) -> Position {
        match (self.production_site(good, original), config.experiment) {
            (ProductionSite::UnrestrictedManufacturing, _)
            | (ProductionSite::ResourceAvailable, Experiment::Concentrated) => original,
            _ => {
                let mut regions = self.regions.iter().filter(|region| region.good == good);
                let key = DrawKey {
                    seed: config.seed,
                    day: 0,
                    phase: 0,
                    actor: identity,
                    purpose: 2,
                    index: 0,
                };
                let selection = (key.draw() % regions.clone().count() as u64) as usize;
                let region = regions.nth(selection).expect("validated producer resource");
                let draw = DrawKey { purpose: 3, ..key }.draw();
                Position {
                    x: region.lower.x + draw as u32 % (region.upper.x - region.lower.x),
                    y: region.lower.y + (draw >> 32) as u32 % (region.upper.y - region.lower.y),
                }
            }
        }
    }
}

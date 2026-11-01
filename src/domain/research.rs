use super::{ActorRows, Good};
use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const RESEARCH_SCALE: u32 = 1000;
pub const RESEARCH_PROFILE: &str = "ResearchFixedV2";
pub const MAX_RESEARCH_RECORDS: usize = 100_000;

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub enum AdoptionPolicy {
    #[default]
    Manual,
    MasterAndAdoptPermitted,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ResearchConsent {
    Granted,
    Withheld,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ResearchProject {
    pub identity: u64,
    pub prerequisites: u32,
    pub outcomes: u32,
    pub inputs: [u32; 3],
    pub labor_minutes: u32,
    pub minimum_effort: u32,
    pub pioneer_effort: u32,
    pub capacity: u32,
    pub relatedness: u32,
    pub shape: u32,
    pub exposure: u32,
    pub exposure_signal: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ResearchLab {
    pub owner: u64,
    pub researcher: u64,
    pub discovered: u32,
    pub mastered: u32,
    pub permitted: u32,
    pub adopted: u32,
    pub consent: ResearchConsent,
    #[serde(default)]
    pub production_requirements: u32,
    #[serde(default)]
    pub adoption: AdoptionPolicy,
    pub projects: Vec<ResearchProject>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Equipment {
    pub firm: u64,
    pub remaining_service: u64,
    pub condition: u32,
    pub nameplate: u32,
    pub downtime: u32,
    pub wear_per_packet: u32,
    pub maintenance_good: Good,
    pub maintenance_ticks: u32,
    pub restoration_per_tick: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct NaturalStock {
    pub firm: u64,
    pub stock: u32,
    pub capacity: u32,
    pub renewal_rate: u32,
    pub extraction_per_packet: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct StrikePolicy {
    pub household: u64,
    pub grievance: u32,
    pub solidarity: u32,
    pub trust: u32,
    pub efficacy: u32,
    pub peers: u32,
    pub cost: u32,
    pub risk: u32,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ResearchConfig {
    #[serde(default)]
    pub labs: Vec<ResearchLab>,
    #[serde(default)]
    pub equipment: Vec<Equipment>,
    #[serde(default)]
    pub natural_stocks: Vec<NaturalStock>,
    #[serde(default)]
    pub strikes: Vec<StrikePolicy>,
}

impl ResearchConfig {
    pub fn memory_estimate(&self) -> u64 {
        let projects = self
            .labs
            .iter()
            .map(|lab| lab.projects.len() as u64)
            .sum::<u64>();
        (self.labs.len() as u64
            + projects
            + self.equipment.len() as u64
            + self.natural_stocks.len() as u64
            + self.strikes.len() as u64)
            * 512
            + 4096
    }

    pub fn new(mut config: Self, actors: &ActorRows) -> Result<Self> {
        config.labs.sort_by_key(|lab| lab.owner);
        config.equipment.sort_by_key(|machine| machine.firm);
        config.natural_stocks.sort_by_key(|stock| stock.firm);
        config.strikes.sort_by_key(|strike| strike.household);
        let identities = actors
            .iter()
            .map(|actor| (actor.identity, actor))
            .collect::<BTreeMap<_, _>>();
        let actor = |identity| {
            identities
                .get(&identity)
                .cloned()
                .context("institution references an unknown actor")
        };
        let mut owners = BTreeSet::new();
        let mut researchers = BTreeSet::new();
        let mut projects = BTreeSet::new();
        for lab in &mut config.labs {
            let owner = actor(lab.owner)?;
            let researcher = actor(lab.researcher)?;
            let admitted = owner.kind != 0
                && researcher.kind == 0
                && owners.insert(lab.owner)
                && researchers.insert(lab.researcher)
                && lab.mastered & !lab.discovered == 0
                && lab.adopted & !(lab.mastered & lab.permitted) == 0;
            match admitted {
                true => (),
                false => Err(anyhow!(
                    "labs require distinct firms/researchers and coherent access states"
                ))?,
            }
            lab.projects.sort_by_key(|project| project.identity);
            for project in &lab.projects {
                let admitted = project.identity > 0
                    && projects.insert(project.identity)
                    && project.labor_minutes <= 1440
                    && (1..=1_000_000).contains(&project.minimum_effort)
                    && (project.minimum_effort..=1_000_000).contains(&project.pioneer_effort)
                    && project.capacity <= RESEARCH_SCALE
                    && project.relatedness <= RESEARCH_SCALE
                    && (1..=16).contains(&project.shape)
                    && project.exposure <= RESEARCH_SCALE
                    && project.exposure_signal <= 16 * RESEARCH_SCALE
                    && project.inputs.iter().all(|&input| input <= 1_000_000);
                match admitted {
                    true => (),
                    false => Err(anyhow!(
                        "research project exceeds identity or fixed-point policy bounds"
                    ))?,
                }
            }
        }
        let mut machines = BTreeSet::new();
        for machine in &config.equipment {
            let admitted = actor(machine.firm)?.kind != 0
                && machines.insert(machine.firm)
                && machine.condition <= RESEARCH_SCALE
                && machine.nameplate <= 10_000
                && machine.downtime <= machine.nameplate
                && machine.wear_per_packet <= RESEARCH_SCALE
                && machine.maintenance_ticks <= 1_000_000
                && machine.restoration_per_tick <= RESEARCH_SCALE;
            match admitted {
                true => (),
                false => Err(anyhow!("equipment exceeds actor or service bounds"))?,
            }
        }
        let mut stocks = BTreeSet::new();
        for stock in &config.natural_stocks {
            let admitted = actor(stock.firm)?.kind != 0
                && stocks.insert(stock.firm)
                && (1..=1_000_000).contains(&stock.capacity)
                && stock.stock <= stock.capacity
                && stock.renewal_rate <= RESEARCH_SCALE
                && (1..=1_000_000).contains(&stock.extraction_per_packet);
            match admitted {
                true => (),
                false => Err(anyhow!(
                    "natural stock exceeds actor or renewable-stock bounds"
                ))?,
            }
        }
        let mut strikers = BTreeSet::new();
        for strike in &config.strikes {
            let fractions = [
                strike.grievance,
                strike.solidarity,
                strike.trust,
                strike.efficacy,
                strike.peers,
                strike.cost,
                strike.risk,
            ];
            let admitted = actor(strike.household)?.kind == 0
                && strikers.insert(strike.household)
                && fractions.into_iter().all(|value| value <= RESEARCH_SCALE);
            match admitted {
                true => (),
                false => Err(anyhow!(
                    "strike policy exceeds household or fraction bounds"
                ))?,
            }
        }
        let count = config.labs.len()
            + projects.len()
            + config.equipment.len()
            + config.natural_stocks.len()
            + config.strikes.len();
        match count <= MAX_RESEARCH_RECORDS {
            true => Ok(config),
            false => Err(anyhow!("research institution exceeds record capacity")),
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ResearchSnapshot {
    pub profile: String,
    pub table_hash: String,
    pub configuration_hash: String,
    pub words: Vec<u32>,
    pub summary: ResearchSummary,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ResearchSummary {
    pub consumed: [u64; 3],
    pub attempts: u64,
    pub discoveries: u64,
    pub remaining_service: u64,
    pub natural_stock: u64,
    pub striking: u64,
}

pub const RESEARCH_TABLE_HASH: &str =
    "4f4ac0e6515edaa9d5847f686c9958c3107e4b0626595faf84751e31fc146f35";

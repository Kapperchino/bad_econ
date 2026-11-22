use super::{
    BASE_FIELDS, Field, Good, ObservationSnapshot, Scenario, ScenarioConfig, State, Technique,
    USED_FIELDS, demography, railway,
};
use crate::arithmetic::{PROFILE, hex, table_hash};
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const CHECKPOINT_VERSION: u32 = 28;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(try_from = "CheckpointDocument", deny_unknown_fields)]
pub struct Checkpoint {
    version: u32,
    profile: String,
    table_hash: String,
    scenario: ScenarioConfig,
    state: State,
    checksum: String,
}
impl Checkpoint {
    pub fn new(scenario: &Scenario, state: &State) -> Result<Self> {
        let _ = ObservationSnapshot::from_state(scenario, state)?;
        let checksum = Self::checksum(scenario.config(), state)?;
        Ok(Self {
            version: CHECKPOINT_VERSION,
            profile: PROFILE.into(),
            table_hash: table_hash(),
            scenario: scenario.config().clone(),
            state: state.clone(),
            checksum,
        })
    }
    fn checksum(config: &ScenarioConfig, state: &State) -> Result<String> {
        Ok(hex(&Sha256::digest(serde_json::to_vec(&(config, state))?)))
    }
    pub fn restore(&self) -> Result<Restored> {
        let scenario = Scenario::new(self.scenario.clone())?;
        let valid = [
            self.version == CHECKPOINT_VERSION,
            self.profile == PROFILE,
            self.table_hash == table_hash(),
            self.state.values.len() == scenario.state_len_for(self.state.actors.len()),
            self.state.actors.len() >= scenario.actors(),
            self.checksum == Self::checksum(&self.scenario, &self.state)?,
        ];
        match valid.into_iter().all(std::convert::identity) {
            true => {
                let _ = ObservationSnapshot::from_state(&scenario, &self.state)?;
                Ok(Restored {
                    scenario,
                    state: self.state.clone(),
                })
            }
            false => Err(anyhow!(
                "checkpoint version, shape, profile, table hash, or checksum mismatch"
            )),
        }
    }
}
pub struct Restored {
    pub scenario: Scenario,
    pub state: State,
}

#[derive(Deserialize)]
#[serde(transparent)]
struct CheckpointDocument(serde_json::Value);

impl TryFrom<CheckpointDocument> for Checkpoint {
    type Error = anyhow::Error;
    fn try_from(document: CheckpointDocument) -> Result<Self> {
        match document
            .0
            .get("version")
            .and_then(serde_json::Value::as_u64)
        {
            Some(23..=27) => (),
            Some(version) if version == u64::from(CHECKPOINT_VERSION) => (),
            _ => Err(anyhow!(
                "unsupported checkpoint format: expected 23 through {CHECKPOINT_VERSION}"
            ))?,
        }
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Current {
            version: u32,
            profile: String,
            table_hash: String,
            scenario: ScenarioConfig,
            state: State,
            checksum: String,
        }
        let mut current: Current = serde_json::from_value(document.0)?;
        if current.version < CHECKPOINT_VERSION {
            let scenario = Scenario::new(current.scenario.clone())?;
            let count = current.state.actors.len();
            let expected = scenario.state_len_for(count);
            let grain = Field::GrainReserve as usize;
            let freight = Field::RoadFreight as usize;
            let missing = match current.version {
                23 => Field::JobDaysRemaining as usize..USED_FIELDS,
                24 => Field::EligibleWorkers as usize..USED_FIELDS,
                25 => Field::Infants as usize..USED_FIELDS,
                26 => grain..USED_FIELDS,
                27 if current.state.values.len() == expected - (freight - grain) * count => {
                    grain..freight
                }
                27 => freight..USED_FIELDS,
                _ => unreachable!(),
            };
            let add_freight = missing.end == USED_FIELDS;
            let shipping = Good::COUNT * scenario.ring() * count;
            let added = missing.len() * count + usize::from(add_freight) * shipping;
            match (current.version >= 25 || current.scenario.people.is_none())
                && count >= scenario.actors()
                && current.state.values.len() == expected - added
                && current.checksum == Self::checksum(&current.scenario, &current.state)?
            {
                true => (),
                false => Err(anyhow!("invalid legacy checkpoint shape or checksum"))?,
            }
            let offset = missing.start * count;
            current.state.values.splice(
                offset..offset,
                std::iter::repeat_n(0, missing.len() * count),
            );
            if add_freight {
                let offset = BASE_FIELDS * count + shipping;
                current
                    .state
                    .values
                    .splice(offset..offset, std::iter::repeat_n(0, shipping));
                for row in 0..count {
                    if current.state.actors.at(row).kind == Good::Railway as u32 {
                        let mut actor = current.state.actors.at(row);
                        actor.capacity = railway::DEFAULT_CREW_CAPACITY;
                        current.state.actors.set(row, actor);
                        let minutes = Technique::railway()
                            .labor_minutes
                            .min(current.scenario.work_cap);
                        current.state.set(Field::LaborMinutes, row, minutes.into());
                        current
                            .state
                            .set(Field::BasicLaborMinutes, row, minutes.into());
                    }
                }
            }
            if current.version == 25 && current.scenario.people.is_some() {
                demography::CohortPopulation::from_individuals(
                    &current.scenario,
                    &mut current.state,
                )?;
            }
            current.version = CHECKPOINT_VERSION;
            current.checksum = Self::checksum(&current.scenario, &current.state)?;
        }
        Ok(Self {
            version: current.version,
            profile: current.profile,
            table_hash: current.table_hash,
            scenario: current.scenario,
            state: current.state,
            checksum: current.checksum,
        })
    }
}

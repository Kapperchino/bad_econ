use super::{Good, State};
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const DAYS_PER_YEAR: u32 = 365;
pub const PROBABILITY_SCALE: u64 = 1_000_000_000_000;
pub const EXPOSURE_SCALE: u32 = 1000;

#[repr(u32)]
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum DeathCause {
    Background = 1,
    Occupational = 2,
    Famine = 3,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum LivingState {
    Living,
    Deceased { day: u64, cause: DeathCause },
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum Sex {
    Female,
    Male,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PersonSeed {
    pub identity: u64,
    pub household: u64,
    pub age_days: u32,
    pub sex: Sex,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Person {
    pub identity: u64,
    pub household: u64,
    pub age_days: u32,
    pub sex: Sex,
    pub state: LivingState,
    pub employer: Option<u64>,
    pub work_minutes: u32,
    pub fatigue: u32,
    pub famine_exposure: u32,
    pub birth_cooldown: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MortalityProfile {
    pub version: u32,
    pub name: String,
    pub sources: Vec<String>,
    pub background_annual_ppm: [u32; 5],
    pub occupational_annual_ppm: [u32; Good::COUNT],
    pub reference_work_minutes_per_year: u32,
    pub famine_daily_ppm_at_thirty_days: u32,
    pub famine_vulnerability_per_mille: [u32; 5],
    pub exposure_recovery_per_day: u32,
}

impl Default for MortalityProfile {
    fn default() -> Self {
        Self {
            version: 1,
            name: "Generic industrial transition v1 (estimated)".into(),
            sources: vec![
                "https://api.parliament.uk/historic-hansard/lords/1900/jun/28/railways-prevention-of-accidents-bill".into(),
                "https://www.populationspast.org/about/".into(),
            ],
            background_annual_ppm: [150_000, 15_000, 6_000, 15_000, 80_000],
            occupational_annual_ppm: [300, 200, 200, 960, 1280, 200, 300, 100, 100, 100, 500, 200, 200, 300, 300, 200],
            reference_work_minutes_per_year: 300 * 480,
            famine_daily_ppm_at_thirty_days: 20_000,
            famine_vulnerability_per_mille: [2000, 1500, 1000, 1200, 2000],
            exposure_recovery_per_day: 500,
        }
    }
}

impl MortalityProfile {
    pub fn new(profile: Self) -> Result<Self> {
        let admitted = [
            profile.version == 1,
            !profile.name.is_empty(),
            !profile.sources.is_empty(),
            profile
                .background_annual_ppm
                .iter()
                .all(|&rate| rate <= 1_000_000),
            profile
                .occupational_annual_ppm
                .iter()
                .all(|&rate| rate <= 1_000_000),
            (1..=525_600).contains(&profile.reference_work_minutes_per_year),
            profile.famine_daily_ppm_at_thirty_days <= 100_000,
            profile
                .famine_vulnerability_per_mille
                .iter()
                .all(|&rate| rate <= 4000),
            profile.exposure_recovery_per_day <= 30_000,
        ];
        match admitted.into_iter().all(std::convert::identity) {
            true => Ok(profile),
            false => Err(anyhow!(
                "invalid mortality profile version, units, or rates"
            )),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct DemographicPolicy {
    pub initial_household_members: u32,
    pub minimum_work_age_years: u32,
    pub retirement_age_years: u32,
    pub annual_births_per_million_fertile_women: u32,
    pub birth_spacing_days: u32,
    pub food_requirement_per_mille: [u32; 5],
    pub work_food_per_mille: u32,
    pub mortality: MortalityProfile,
    #[serde(default)]
    pub people: Vec<PersonSeed>,
}

impl Default for DemographicPolicy {
    fn default() -> Self {
        Self {
            initial_household_members: 1,
            minimum_work_age_years: 15,
            retirement_age_years: 65,
            annual_births_per_million_fertile_women: 120_000,
            birth_spacing_days: 365,
            food_requirement_per_mille: [400, 700, 1000, 1000, 900],
            work_food_per_mille: 250,
            mortality: MortalityProfile::default(),
            people: Vec::new(),
        }
    }
}

impl DemographicPolicy {
    pub fn new(mut policy: Self) -> Result<Self> {
        policy.mortality = MortalityProfile::new(policy.mortality)?;
        let valid = [
            (1..=8).contains(&policy.initial_household_members),
            (10..=30).contains(&policy.minimum_work_age_years),
            (40..=90).contains(&policy.retirement_age_years),
            policy.annual_births_per_million_fertile_women <= 1_000_000,
            (270..=3650).contains(&policy.birth_spacing_days),
            policy
                .food_requirement_per_mille
                .iter()
                .all(|&need| (1..=2000).contains(&need)),
            policy.work_food_per_mille <= 1000,
        ];
        match valid.into_iter().all(std::convert::identity) {
            true => Ok(policy),
            false => Err(anyhow!(
                "invalid demographic ages, fertility, or food requirements"
            )),
        }
    }
    pub fn initial(&self, state: &State) -> Result<Vec<Person>> {
        let households = state
            .actors
            .iter()
            .filter(|actor| actor.kind == 0)
            .map(|actor| actor.identity)
            .collect::<BTreeSet<_>>();
        let fatigue = state
            .actors
            .iter()
            .enumerate()
            .filter(|(_, actor)| actor.kind == 0)
            .map(|(index, actor)| {
                (
                    actor.identity,
                    state.get(super::Field::Fatigue, index) as u32,
                )
            })
            .collect::<std::collections::BTreeMap<_, _>>();
        let seeds = match self.people.is_empty() {
            true => households
                .iter()
                .enumerate()
                .flat_map(|(index, &household)| {
                    (0..self.initial_household_members).map(move |member| PersonSeed {
                        identity: state.next_identity
                            + index as u64 * u64::from(self.initial_household_members)
                            + u64::from(member),
                        household,
                        age_days: match member {
                            0..=1 => (20 + index as u32 % 25) * DAYS_PER_YEAR,
                            _ => (3 + (index as u32 + member * 3) % 12) * DAYS_PER_YEAR,
                        },
                        sex: match (index + member as usize) % 2 {
                            0 => Sex::Female,
                            _ => Sex::Male,
                        },
                    })
                })
                .collect(),
            false => self.people.clone(),
        };
        let identities = seeds
            .iter()
            .map(|person| person.identity)
            .collect::<BTreeSet<_>>();
        let actors = state
            .actors
            .iter()
            .map(|actor| actor.identity)
            .collect::<BTreeSet<_>>();
        let valid = seeds.iter().all(|person| {
            households.contains(&person.household)
                && person.age_days <= 150 * DAYS_PER_YEAR
                && (1..u64::MAX).contains(&person.identity)
        }) && identities.len() == seeds.len()
            && identities.is_disjoint(&actors);
        match valid {
            true => Ok(seeds
                .into_iter()
                .map(|seed| Person {
                    identity: seed.identity,
                    household: seed.household,
                    age_days: seed.age_days,
                    sex: seed.sex,
                    state: LivingState::Living,
                    employer: None,
                    work_minutes: 0,
                    fatigue: fatigue[&seed.household],
                    famine_exposure: 0,
                    birth_cooldown: 0,
                })
                .collect()),
            false => Err(anyhow!(
                "person identities, ages, or household membership are invalid"
            )),
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct PopulationObservation {
    pub living: u64,
    pub dependents: u64,
    pub births: u64,
    pub deaths: [u64; 3],
    pub work_minutes: u64,
    pub famine_exposure: u64,
}

impl PopulationObservation {
    pub fn from_state(state: &State, policy: &DemographicPolicy) -> Self {
        state.people.iter().fold(
            Self {
                births: state.people.len() as u64 - state.initial_population,
                ..Self::default()
            },
            |mut observation, person| {
                match person.state {
                    LivingState::Living => {
                        observation.living += 1;
                        observation.dependents += u64::from(
                            person.age_days < policy.minimum_work_age_years * DAYS_PER_YEAR
                                || person.age_days >= policy.retirement_age_years * DAYS_PER_YEAR,
                        );
                    }
                    LivingState::Deceased { cause, .. } => {
                        observation.deaths[cause as usize - 1] += 1
                    }
                }
                observation.work_minutes += u64::from(person.work_minutes);
                observation.famine_exposure += u64::from(person.famine_exposure);
                observation
            },
        )
    }
}

pub struct PersonRegistry<'a> {
    pub people: &'a [Person],
}
impl<'a> PersonRegistry<'a> {
    pub fn new(state: &'a State) -> Result<Self> {
        let actors = state
            .actors
            .iter()
            .map(|actor| (actor.identity, actor.kind))
            .collect::<std::collections::HashMap<_, _>>();
        let mut identities = actors
            .keys()
            .copied()
            .collect::<std::collections::HashSet<_>>();
        let valid = state.people.iter().all(|person| {
            let living_state = match person.state {
                LivingState::Living => true,
                LivingState::Deceased { day, .. } => (1..=state.day).contains(&day),
            };
            identities.insert(person.identity)
                && person.identity > 0
                && person.identity < state.next_identity
                && actors.get(&person.household) == Some(&0)
                && person.age_days <= 150 * DAYS_PER_YEAR
                && person.fatigue <= 1000
                && person.work_minutes <= 1440
                && person.famine_exposure <= 365_000
                && person
                    .employer
                    .is_none_or(|identity| actors.get(&identity).is_some_and(|&kind| kind > 0))
                && living_state
        });
        match valid
            && state.initial_population <= state.people.len() as u64
            && actors.len() == state.actors.len()
        {
            true => Ok(Self {
                people: &state.people,
            }),
            false => Err(anyhow!(
                "invalid person registry identity, membership, mortality, or time budget"
            )),
        }
    }
}

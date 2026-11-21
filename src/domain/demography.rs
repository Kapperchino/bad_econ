use super::{Good, State};
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub use super::calendar::DAYS_PER_YEAR;
pub const PROBABILITY_SCALE: u64 = 1_000_000_000_000;
pub const EXPOSURE_SCALE: u32 = 1000;

#[repr(u32)]
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum DeathCause {
    Background = 1,
    Occupational = 2,
    Famine = 3,
    Cold = 4,
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
    #[serde(default)]
    pub cold_exposure: u32,
    pub birth_cooldown: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MortalityProfile {
    pub version: u32,
    pub name: String,
    pub sources: Vec<String>,
    pub background_annual_ppm: [u32; 5],
    #[serde(deserialize_with = "super::industry::goods_array")]
    pub occupational_annual_ppm: [u32; Good::COUNT],
    pub reference_work_minutes_per_year: u32,
    pub famine_daily_ppm_at_thirty_days: u32,
    pub famine_vulnerability_per_mille: [u32; 5],
    pub exposure_recovery_per_day: u32,
    #[serde(default = "default_cold_daily_ppm")]
    pub cold_daily_ppm_at_seven_days: u32,
    #[serde(default = "default_cold_vulnerability")]
    pub cold_vulnerability_per_mille: [u32; 5],
}

fn default_cold_daily_ppm() -> u32 {
    1000
}

fn default_cold_vulnerability() -> [u32; 5] {
    [2000, 1500, 1000, 1200, 2000]
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
            occupational_annual_ppm: [300, 200, 200, 960, 1280, 200, 300, 100, 100, 100, 500, 200, 200, 300, 300, 200, 600, 0, 200, 960, 100, 200],
            reference_work_minutes_per_year: 300 * 480,
            famine_daily_ppm_at_thirty_days: 20_000,
            famine_vulnerability_per_mille: [2000, 1500, 1000, 1200, 2000],
            exposure_recovery_per_day: 500,
            cold_daily_ppm_at_seven_days: default_cold_daily_ppm(),
            cold_vulnerability_per_mille: default_cold_vulnerability(),
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
            profile.cold_daily_ppm_at_seven_days <= 100_000,
            profile
                .cold_vulnerability_per_mille
                .iter()
                .all(|&rate| rate <= 4000),
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
    pub(super) fn cohort_band(&self, age_days: u32) -> usize {
        let years = age_days / DAYS_PER_YEAR;
        match years {
            0 => 0,
            age if age < self.minimum_work_age_years => 1,
            age if age >= self.retirement_age_years => 4,
            0..45 => 2,
            _ => 3,
        }
    }
    pub fn initial(&self, state: &State) -> Result<Vec<Person>> {
        let seeds = match self.people.is_empty() {
            true => state
                .actors
                .iter()
                .filter(|actor| actor.kind == 0)
                .map(|actor| actor.identity)
                .collect::<BTreeSet<_>>()
                .into_iter()
                .enumerate()
                .flat_map(|(index, household)| {
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
        self.from_seeds(state, seeds)
    }
    pub(super) fn from_seeds(&self, state: &State, seeds: Vec<PersonSeed>) -> Result<Vec<Person>> {
        let households = state
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
            households.contains_key(&person.household)
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
                    fatigue: households[&seed.household],
                    famine_exposure: 0,
                    cold_exposure: 0,
                    birth_cooldown: 0,
                })
                .collect()),
            false => Err(anyhow!(
                "person identities, ages, or household membership are invalid"
            )),
        }
    }
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Occupation {
    Farmer,
    WageWorker,
    Owner,
    Unemployed,
    Dependent,
    OtherAdult,
}
impl Occupation {
    pub const ALL: [Self; 6] = [
        Self::Farmer,
        Self::WageWorker,
        Self::Owner,
        Self::Unemployed,
        Self::Dependent,
        Self::OtherAdult,
    ];
    pub fn label(self) -> &'static str {
        match self {
            Self::Farmer => "Farmers",
            Self::WageWorker => "Wage workers",
            Self::Owner => "Owners",
            Self::Unemployed => "Unemployed",
            Self::Dependent => "Dependents",
            Self::OtherAdult => "Other adults",
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct PopulationObservation {
    #[serde(default)]
    pub occupations: [u64; 6],
    pub living: u64,
    pub dependents: u64,
    pub births: u64,
    #[serde(deserialize_with = "death_counts")]
    pub deaths: [u64; 4],
    pub work_minutes: u64,
    pub famine_exposure: u64,
    #[serde(default)]
    pub cold_exposure: u64,
}

fn death_counts<'de, D>(deserializer: D) -> std::result::Result<[u64; 4], D::Error>
where
    D: serde::Deserializer<'de>,
{
    let counts = Vec::<u64>::deserialize(deserializer)?;
    match counts.as_slice() {
        [background, occupational, famine] => Ok([*background, *occupational, *famine, 0]),
        [background, occupational, famine, cold] => {
            Ok([*background, *occupational, *famine, *cold])
        }
        _ => Err(serde::de::Error::custom(
            "expected three or four death causes",
        )),
    }
}

impl PopulationObservation {
    pub fn unemployment_rate(&self) -> Option<f64> {
        let unemployed = self.occupations[Occupation::Unemployed as usize];
        let workforce = unemployed + self.occupations[Occupation::WageWorker as usize];
        match workforce {
            0 => None,
            count => Some(100.0 * unemployed as f64 / count as f64),
        }
    }

    pub fn from_state(state: &State, policy: &DemographicPolicy) -> Self {
        Self::observe(state, policy, |row| {
            state.actors.at(row).capacity > 1 || state.get(super::Field::EligibleWorkers, row) > 0
        })
    }

    pub(super) fn for_scenario(state: &State, scenario: &super::Scenario) -> Self {
        Self::observe(state, &scenario.config().demography, |_| {
            scenario.config().people.is_some()
        })
    }

    fn observe(
        state: &State,
        policy: &DemographicPolicy,
        is_cohort: impl Fn(usize) -> bool,
    ) -> Self {
        match state.people.is_empty() {
            true => CohortPopulation::observation(state),
            false => Self::individual_observation(state, policy, is_cohort),
        }
    }

    fn individual_observation(
        state: &State,
        policy: &DemographicPolicy,
        is_cohort: impl Fn(usize) -> bool,
    ) -> Self {
        let households = state
            .actors
            .iter()
            .enumerate()
            .map(|(index, actor)| (actor.identity, index))
            .collect::<std::collections::HashMap<_, _>>();
        state.people.iter().enumerate().fold(
            Self {
                births: state.people.len() as u64 - state.initial_population,
                ..Self::default()
            },
            |mut observation, (index, person)| {
                match person.state {
                    LivingState::Living => {
                        observation.living += 1;
                        let household = households[&person.household];
                        let selected = state.get(super::Field::Worker, household) == index as u64;
                        let participant = selected || is_cohort(household);
                        let owner = state
                            .get(super::Field::InvestmentFirm, household)
                            .checked_sub(1)
                            .filter(|&firm| {
                                state.get(super::Field::EnterpriseStatus, firm as usize)
                                    < super::enterprise::EnterpriseState::Closed as u64
                            })
                            .is_some();
                        let category = match () {
                            _ if person.age_days
                                < policy.minimum_work_age_years * DAYS_PER_YEAR
                                || person.age_days
                                    >= policy.retirement_age_years * DAYS_PER_YEAR =>
                            {
                                Occupation::Dependent
                            }
                            _ if selected && owner => Occupation::Owner,
                            _ if participant
                                && state.get(super::Field::LandPlots, household) > 0 =>
                            {
                                Occupation::Farmer
                            }
                            _ if person.employer.is_some() => Occupation::WageWorker,
                            _ if participant => Occupation::Unemployed,
                            _ => Occupation::OtherAdult,
                        };
                        observation.occupations[category as usize] += 1;
                        observation.dependents += u64::from(category == Occupation::Dependent);
                    }
                    LivingState::Deceased { cause, .. } => {
                        observation.deaths[cause as usize - 1] += 1
                    }
                }
                observation.work_minutes += u64::from(person.work_minutes);
                observation.famine_exposure += u64::from(person.famine_exposure);
                observation.cold_exposure += u64::from(person.cold_exposure);
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
                && person.cold_exposure <= 365_000
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

pub(super) const AGE_FIELDS: [super::Field; 5] = {
    use super::Field::*;
    [Infants, Children, YoungAdults, OlderAdults, Seniors]
};

pub(super) const FEMALE_FIELDS: [super::Field; 5] = {
    use super::Field::*;
    [
        FemaleInfants,
        FemaleChildren,
        FemaleYoungAdults,
        FemaleOlderAdults,
        FemaleSeniors,
    ]
};

pub(super) struct CohortPopulation;

impl CohortPopulation {
    pub fn from_individuals(config: &super::ScenarioConfig, state: &mut State) -> Result<Self> {
        use super::Field::*;
        PersonRegistry::new(state)?;
        let households = state
            .actors
            .identities()
            .iter()
            .enumerate()
            .map(|(row, &id)| (id, row))
            .collect::<std::collections::HashMap<_, _>>();
        let seeds = config
            .demography
            .people
            .iter()
            .map(|seed| seed.identity)
            .collect::<BTreeSet<_>>();
        let first = config.population.as_ref().map_or_else(
            || u64::from(config.population_size() + config.firms) + 1,
            super::Population::next_identity,
        );
        let end = first
            .checked_add(state.initial_population)
            .ok_or_else(|| anyhow!("initial identity range overflow"))?;
        for person in std::mem::take(&mut state.people) {
            let row = households[&person.household];
            let born = match seeds.is_empty() {
                true => !(first..end).contains(&person.identity),
                false => !seeds.contains(&person.identity),
            };
            state.set(
                CohortBirths,
                row,
                state.get(CohortBirths, row) + u64::from(born),
            );
            match person.state {
                LivingState::Living => {
                    let band = config.demography.cohort_band(person.age_days);
                    let field = AGE_FIELDS[band];
                    state.set(field, row, state.get(field, row) + 1);
                    if person.sex == Sex::Female {
                        let field = FEMALE_FIELDS[band];
                        state.set(field, row, state.get(field, row) + 1);
                        if band == 2 && person.birth_cooldown > 0 {
                            state.set(
                                RecoveringMothers,
                                row,
                                state.get(RecoveringMothers, row) + 1,
                            );
                        }
                    }
                    for (field, value) in [
                        (CohortFamineExposure, person.famine_exposure),
                        (CohortColdExposure, person.cold_exposure),
                    ] {
                        state.set(field, row, state.get(field, row) + u64::from(value));
                    }
                }
                LivingState::Deceased { cause, .. } => {
                    let field = [
                        BackgroundDeaths,
                        OccupationalDeaths,
                        FamineDeaths,
                        ColdDeaths,
                    ][cause as usize - 1];
                    state.set(field, row, state.get(field, row) + 1);
                }
            }
        }
        for row in 0..state.actors.len() {
            let living = AGE_FIELDS
                .iter()
                .map(|&field| state.get(field, row))
                .sum::<u64>();
            state.set(LivingMembers, row, living);
            for field in [CohortFamineExposure, CohortColdExposure] {
                state.set(field, row, state.get(field, row) / living.max(1));
            }
        }
        Self::new(state)
    }

    pub fn new(state: &State) -> Result<Self> {
        use super::Field::*;
        let mut registry = 0u128;
        let mut births = 0u128;
        for (row, actor) in state.actors.iter().enumerate() {
            let living = [Infants, Children, YoungAdults, OlderAdults, Seniors]
                .into_iter()
                .map(|field| u128::from(state.get(field, row)))
                .sum::<u128>();
            let deaths = [
                BackgroundDeaths,
                OccupationalDeaths,
                FamineDeaths,
                ColdDeaths,
            ]
            .into_iter()
            .map(|field| u128::from(state.get(field, row)))
            .sum::<u128>();
            let born = u128::from(state.get(CohortBirths, row));
            let valid = state.people.is_empty()
                && living == u128::from(state.get(LivingMembers, row))
                && AGE_FIELDS
                    .into_iter()
                    .zip(FEMALE_FIELDS)
                    .all(|(age, female)| state.get(female, row) <= state.get(age, row))
                && state.get(RecoveringMothers, row) <= state.get(FemaleYoungAdults, row)
                && (actor.kind == 0 || living + deaths + born == 0)
                && state.get(CohortColdExposure, row) <= 365_000
                && state.get(CohortFamineExposure, row) <= 365_000;
            if !valid {
                Err(anyhow!(
                    "invalid demographic counts in cohort {}",
                    actor.identity
                ))?;
            }
            registry += living + deaths;
            births += born;
        }
        let identities = state
            .actors
            .identities()
            .iter()
            .collect::<std::collections::HashSet<_>>();
        match registry == u128::from(state.initial_population) + births
            && identities.len() == state.actors.len()
        {
            true => Ok(Self),
            false => Err(anyhow!(
                "cohort births, deaths and living population do not balance"
            )),
        }
    }

    fn observation(state: &State) -> PopulationObservation {
        use super::Field::*;
        let mut total = PopulationObservation::default();
        for (row, _) in state
            .actors
            .iter()
            .enumerate()
            .filter(|(_, actor)| actor.kind == 0)
        {
            let value = |field| state.get(field, row);
            let living = value(LivingMembers);
            let adults = value(YoungAdults) + value(OlderAdults);
            let dependents = living - adults;
            let owner = u64::from(
                adults > 0
                    && value(InvestmentFirm).checked_sub(1).is_some_and(|firm| {
                        state.get(EnterpriseStatus, firm as usize)
                            < super::enterprise::EnterpriseState::Closed as u64
                    }),
            );
            let farmers = if value(LandPlots) > 0 {
                adults - owner
            } else {
                0
            };
            let paid = value(Employed).min(adults - owner - farmers);
            let counts = [
                farmers,
                paid,
                owner,
                adults - owner - farmers - paid,
                dependents,
                0,
            ];
            for (sum, count) in total.occupations.iter_mut().zip(counts) {
                *sum += count;
            }
            total.living += living;
            total.dependents += dependents;
            total.births += value(CohortBirths);
            for (sum, field) in total.deaths.iter_mut().zip([
                BackgroundDeaths,
                OccupationalDeaths,
                FamineDeaths,
                ColdDeaths,
            ]) {
                *sum += value(field);
            }
            total.work_minutes += value(Hours) + value(HomeMinutes) + value(CartageMinutes);
            total.famine_exposure += living * value(CohortFamineExposure);
            total.cold_exposure += living * value(CohortColdExposure);
        }
        total
    }
}

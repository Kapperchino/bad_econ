use anyhow::Result;
use bad_econ::{
    domain::*,
    simulation::{Device, Simulation},
};
use clap::{Parser, ValueEnum};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

#[derive(Clone, Copy, ValueEnum)]
enum ResearchPolicy {
    Enabled,
    Disabled,
}

#[derive(Parser)]
struct Arguments {
    #[arg(long)]
    scenario: Option<PathBuf>,
    #[arg(long)]
    checkpoint: Option<PathBuf>,
    #[arg(long, default_value_t = 1000)]
    households: u32,
    #[arg(long, default_value_t = 365)]
    days: u32,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, default_value_t = 128)]
    workgroup: u32,
    #[arg(long, value_enum, default_value = "enabled")]
    research: ResearchPolicy,
}

#[derive(Clone, Copy, Serialize)]
#[repr(usize)]
enum HouseholdClass {
    Landless,
    LandlessOwner,
    Farming,
    FarmingOwner,
    Formed,
}

impl HouseholdClass {
    const ALL: [Self; 5] = [
        Self::Landless,
        Self::LandlessOwner,
        Self::Farming,
        Self::FarmingOwner,
        Self::Formed,
    ];

    fn new(state: &State, index: usize, owners: &BTreeSet<u64>) -> Self {
        match (
            state.get(Field::LandPlots, index),
            owners.contains(&state.actors.at(index).identity),
        ) {
            (0, false) => Self::Landless,
            (0, true) => Self::LandlessOwner,
            (_, false) => Self::Farming,
            (_, true) => Self::FarmingOwner,
        }
    }
}

fn shareholders(state: &State) -> BTreeSet<u64> {
    let active = state
        .actors
        .iter()
        .enumerate()
        .filter(|(index, actor)| {
            actor.kind > 0
                && state.get(Field::EnterpriseStatus, *index)
                    != enterprise::EnterpriseState::Closed as u64
        })
        .map(|(_, actor)| actor.identity)
        .collect::<BTreeSet<_>>();
    state
        .ownership
        .iter()
        .filter(|firm| active.contains(&firm.firm))
        .flat_map(|firm| firm.shares.iter().map(|share| share.household))
        .collect()
}

#[derive(Default, Serialize)]
struct Accounts {
    count: usize,
    living_accounts: usize,
    members: u64,
    cash: u64,
    cash_median: u64,
    cash_p90: u64,
    paid: u64,
    food_need: u64,
    deficit: u64,
    grain: u64,
    bread: u64,
    minutes: u64,
    home_minutes: u64,
    budget: u64,
}

#[derive(Serialize)]
struct HouseholdSummary {
    class: HouseholdClass,
    accounts: Accounts,
}

fn household_summaries(
    state: &State,
    classify: impl Fn(usize, u64) -> HouseholdClass,
) -> Vec<HouseholdSummary> {
    let members = state
        .people
        .iter()
        .filter(|person| matches!(person.state, demography::LivingState::Living))
        .fold(BTreeMap::<u64, u64>::new(), |mut counts, person| {
            *counts.entry(person.household).or_default() += 1;
            counts
        });
    let mut households = std::array::from_fn::<_, 5, _>(|_| Accounts::default());
    let mut balances = std::array::from_fn::<_, 5, _>(|_| Vec::new());
    for (index, actor) in state
        .actors
        .iter()
        .enumerate()
        .filter(|(_, actor)| actor.kind == 0)
    {
        let group = classify(index, actor.identity) as usize;
        let summary = &mut households[group];
        summary.count += 1;
        let living = members.get(&actor.identity).copied().unwrap_or(0);
        summary.members += living;
        if living > 0 {
            summary.living_accounts += 1;
            balances[group].push(state.get(Field::Cash, index));
        }
        summary.cash += state.get(Field::Cash, index);
        summary.paid += state.get(Field::Employed, index);
        summary.food_need += state.get(Field::FoodNeed, index);
        summary.deficit += state.get(Field::Shortage, index);
        summary.grain += state.get(Field::Grain, index);
        summary.bread += state.get(Field::Bread, index);
        summary.minutes += state.get(Field::Hours, index);
        summary.home_minutes += state.get(Field::HomeMinutes, index);
        summary.budget += state.get(Field::HouseholdBudget, index);
    }
    for (summary, balances) in households.iter_mut().zip(&mut balances) {
        balances.sort_unstable();
        summary.cash_median = balances.get(balances.len() / 2).copied().unwrap_or(0);
        summary.cash_p90 = balances.get(balances.len() * 9 / 10).copied().unwrap_or(0);
    }
    HouseholdClass::ALL
        .into_iter()
        .zip(households)
        .map(|(class, accounts)| HouseholdSummary { class, accounts })
        .collect()
}

struct InitialCohorts {
    households: BTreeMap<u64, HouseholdClass>,
    people: BTreeMap<u64, HouseholdClass>,
}

#[derive(Serialize)]
struct PeopleCohort {
    class: HouseholdClass,
    initial_people: u64,
    living_initial_people: u64,
    deaths: [u64; 3],
}

impl InitialCohorts {
    fn new(state: &State) -> Self {
        let owners = shareholders(state);
        let households = state
            .actors
            .iter()
            .enumerate()
            .filter(|(_, actor)| actor.kind == 0)
            .map(|(index, actor)| (actor.identity, HouseholdClass::new(state, index, &owners)))
            .collect::<BTreeMap<_, _>>();
        let people = state
            .people
            .iter()
            .map(|person| (person.identity, households[&person.household]))
            .collect();
        Self { households, people }
    }

    fn population(&self, state: &State) -> [PeopleCohort; 5] {
        let mut cohorts = HouseholdClass::ALL.map(|class| PeopleCohort {
            class,
            initial_people: 0,
            living_initial_people: 0,
            deaths: [0; 3],
        });
        for person in &state.people {
            if let Some(&group) = self.people.get(&person.identity) {
                let cohort = &mut cohorts[group as usize];
                cohort.initial_people += 1;
                match person.state {
                    demography::LivingState::Living => cohort.living_initial_people += 1,
                    demography::LivingState::Deceased { cause, .. } => {
                        cohort.deaths[cause as usize - 1] += 1
                    }
                }
            }
        }
        cohorts
    }
}

#[derive(Serialize)]
struct Industry {
    good: Good,
    firms: usize,
    cash: u64,
    capacity: u64,
    plan: u64,
    vacancies: u64,
    demand: u64,
    sales: u64,
    unfilled: u64,
    stock: u64,
    produced: u64,
    states: [usize; 5],
    upgraded: usize,
    price: u64,
    wage: u64,
}

fn main() -> Result<()> {
    let arguments = Arguments::parse();
    let mut config = match arguments.scenario {
        Some(path) => serde_json::from_slice(&std::fs::read(path)?)?,
        None => Scenario::industrial(arguments.households, arguments.seed, Experiment::Baseline)?
            .config()
            .clone(),
    };
    if let ResearchPolicy::Disabled = arguments.research {
        config.research = None;
    }
    let mut simulation = Simulation::new(
        Scenario::new(config)?,
        Device::headless(arguments.workgroup)?,
    )?;
    let cohorts = InitialCohorts::new(&simulation.download()?);
    for day in 1..=arguments.days {
        simulation.advance()?;
        if [1, 4, 8, 16, 32, 60, 90, 180, 270].contains(&day)
            || day.is_multiple_of(365)
            || day == arguments.days
        {
            let state = simulation.download()?;
            let owners = shareholders(&state);
            let households = household_summaries(&state, |index, _| {
                HouseholdClass::new(&state, index, &owners)
            });
            let initial_households = household_summaries(&state, |_, identity| {
                cohorts
                    .households
                    .get(&identity)
                    .copied()
                    .unwrap_or(HouseholdClass::Formed)
            });
            let initial_people = cohorts.population(&state);
            let industries = Good::ALL
                .into_iter()
                .map(|good| {
                    let firms = state
                        .actors
                        .iter()
                        .enumerate()
                        .filter(|(index, actor)| {
                            actor.kind == good as u32
                                && state.get(Field::EnterpriseModel, *index) == 0
                        })
                        .map(|(index, _)| index)
                        .collect::<Vec<_>>();
                    let sum = |field| firms.iter().map(|&index| state.get(field, index)).sum();
                    Industry {
                        good,
                        firms: firms.len(),
                        cash: sum(Field::Cash),
                        capacity: firms
                            .iter()
                            .map(|&index| u64::from(state.actors.at(index).capacity))
                            .sum(),
                        plan: sum(Field::Plan),
                        vacancies: sum(Field::Slots),
                        demand: sum(Field::Demand),
                        sales: sum(Field::Sales),
                        unfilled: sum(Field::Unfilled),
                        stock: sum(Field::inventory(good)),
                        produced: sum(Field::Produced),
                        states: std::array::from_fn(|status| {
                            firms
                                .iter()
                                .filter(|&&index| {
                                    state.get(Field::EnterpriseStatus, index) == status as u64
                                })
                                .count()
                        }),
                        upgraded: firms
                            .iter()
                            .filter(|&&index| state.get(Field::ActiveMethod, index) > 0)
                            .count(),
                        price: sum(Field::Price),
                        wage: sum(Field::Wage),
                    }
                })
                .collect::<Vec<_>>();
            println!(
                "{}",
                serde_json::to_string(
                    &serde_json::json!({"day":day,"snapshot":simulation.snapshot,"households":households,"initial_households":initial_households,"initial_people":initial_people,"industries":industries})
                )?
            );
        }
    }
    if let Some(path) = arguments.checkpoint {
        std::fs::write(path, serde_json::to_vec(&simulation.checkpoint()?)?)?;
    }
    Ok(())
}

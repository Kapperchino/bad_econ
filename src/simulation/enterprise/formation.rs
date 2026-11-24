use super::{SEPARATE_OWNER, opportunities};
use crate::domain::{
    Actor, Field, Good, Scenario, State,
    enterprise::{EnterpriseModel, EnterpriseState, Ownership, OwnershipShare},
    property::{CONSTRUCTION_WORK_PER_PLOT, ProjectStatus, PropertyUse},
};
use anyhow::{Context, Result};
use std::ops::Range;

fn resize_values(state: &mut State, scenario: &Scenario, actors: usize) {
    let before = state.actors.len();
    if actors > before {
        let fields = match (before, &scenario.config().finance) {
            (1.., None) => state.values.len() / before,
            _ => scenario.fields(),
        };
        let mut values = vec![0; actors * fields];
        for field in 0..fields {
            values[field * actors..field * actors + before]
                .copy_from_slice(&state.values[field * before..(field + 1) * before]);
        }
        values.extend_from_slice(&state.values[before * fields..]);
        state.values = values;
    }
}

struct NewActorSlot {
    offset: usize,
    identity: u64,
}

fn append_actors(
    state: &mut State,
    scenario: &Scenario,
    count: usize,
    exhausted: &'static str,
    actor: impl Fn(&State, NewActorSlot) -> Actor,
) -> Result<Range<usize>> {
    let before = state.actors.len();
    let after = before + count;
    let next_identity = state
        .next_identity
        .checked_add(count as u64)
        .context(exhausted)?;
    let additions = (state.next_identity..next_identity)
        .enumerate()
        .map(|(offset, identity)| actor(state, NewActorSlot { offset, identity }))
        .collect::<Vec<_>>();
    resize_values(state, scenario, after);
    state.actors.extend(additions);
    state.next_identity = next_identity;
    Ok(before..after)
}

pub(in crate::simulation) fn enter(
    state: &mut State,
    scenario: &Scenario,
    requests: &[u32],
) -> Result<usize> {
    let entries = requests
        .iter()
        .enumerate()
        .filter(|(_, request)| {
            (1..=Good::COUNT as u32).contains(*request) || matches!(**request >> 24, 8 | 9)
        })
        .map(|(owner, &request)| opportunities::EntryPlan::new(state, scenario, owner, request))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let rows = append_actors(
        state,
        scenario,
        entries.len(),
        "enterprise identity capacity exhausted",
        |_, slot| {
            let entry = &entries[slot.offset];
            Actor {
                identity: slot.identity,
                position: entry.position,
                kind: entry.good as u32,
                owner: entry.owner as u32,
                capacity: 1,
            }
        },
    )?;
    for (firm, entry) in rows.zip(&entries) {
        let owner = entry.owner;
        let cash = state.get(Field::Cash, owner);
        let invested = cash / 2;
        state.set(Field::Cash, owner, cash - invested);
        state.set(Field::Cash, firm, invested);
        state.set(Field::EnterpriseModel, firm, entry.model as u64);
        let usage = PropertyUse::for_site(state, firm)?;
        let project = match &scenario.config().property {
            Some(property) => {
                state.set(
                    Field::PropertyRegion,
                    firm,
                    property.region(entry.position)? as u64 + 1,
                );
                state.set(Field::PropertyUse, firm, usage as u64);
                state.set(Field::RequestedPlots, firm, 1);
                state.set(Field::WorkersPerPlot, firm, 1);
                ProjectStatus::Requested
            }
            None => ProjectStatus::Building,
        };
        state.set(Field::ProjectStatus, firm, project as u64);
        state.set(
            Field::ConstructionRemaining,
            firm,
            CONSTRUCTION_WORK_PER_PLOT.into(),
        );
        state.set(
            Field::ConstructionDaysRemaining,
            firm,
            usage.construction_days().into(),
        );
        state.set(
            Field::EnterpriseStatus,
            firm,
            EnterpriseState::Forming as u64,
        );
        state.set(Field::EquipmentCondition, firm, 1000);
        state.set(Field::Wage, firm, scenario.config().wage.into());
        state.set(Field::Price, firm, entry.price);
        state.set(
            Field::Demand,
            firm,
            entry.recipe.technique.output_ticks.into(),
        );
        state.set(Field::InvestmentFirm, owner, firm as u64 + 1);
        state.set(Field::InvestmentGood, owner, 0);
        state.set_recipe(firm, entry.recipe);
        state.set(Field::FirstInputCost, firm, entry.first_cost);
        state.set(Field::SecondInputCost, firm, entry.second_cost);
        if entry.model == EnterpriseModel::Distributor {
            state.set(Field::ReplacementCost, firm, entry.first_cost);
            state.set(Field::StockCoverage, firm, 3);
            state.set(Field::RetailMargin, firm, 250);
        }
        state.retain_basic_recipe(firm);
        state.ownership.push(Ownership::new(Ownership {
            firm: state.actors.at(firm).identity,
            shares: vec![OwnershipShare {
                household: state.actors.at(owner).identity,
                units: 1_000_000,
            }],
        })?);
    }
    Ok(entries.len())
}

pub(in crate::simulation) struct HouseholdFormation {
    pub parent: usize,
    pub household: usize,
    pub person: u64,
}

pub(in crate::simulation) fn form_households(
    state: &mut State,
    scenario: &Scenario,
    requests: &[u32],
) -> Result<Vec<HouseholdFormation>> {
    let splits = requests
        .iter()
        .enumerate()
        .filter(|(_, action)| **action >> 24 == 1)
        .map(|(household, _)| household)
        .collect::<Vec<_>>();
    let mut formations = Vec::with_capacity(splits.len());
    let before = state.actors.len();
    let rows = append_actors(
        state,
        scenario,
        splits.len(),
        "household identity capacity exhausted",
        |state, slot| Actor {
            identity: slot.identity,
            position: state.actors.at(splits[slot.offset]).position,
            kind: 0,
            owner: (before + slot.offset) as u32,
            capacity: 1,
        },
    )?;
    for (household, parent) in rows.zip(splits) {
        let person_identity = state.get(Field::SplitPerson, parent);
        formations.push(HouseholdFormation {
            parent,
            household,
            person: person_identity,
        });
        let living = state.get(Field::LivingMembers, parent);
        for field in std::iter::once(Field::Cash)
            .chain(Good::ALL.map(Field::inventory))
            .chain([Field::LandPlots, Field::CropWork])
        {
            let assets = state.get(field, parent);
            let share = assets / living.max(2);
            state.set(field, parent, assets - share);
            state.set(field, household, share);
        }
        state.set(
            Field::ParentHousehold,
            household,
            state.actors.at(parent).identity,
        );
        state.set(Field::LivingMembers, household, 1);
        if let Some(band) = [Field::YoungAdults, Field::OlderAdults]
            .into_iter()
            .find(|&field| state.get(field, parent) > 0)
        {
            let women = match band {
                Field::YoungAdults => Field::FemaleYoungAdults,
                _ => Field::FemaleOlderAdults,
            };
            let female = u64::from(state.get(women, parent) * 2 >= state.get(band, parent));
            let retained = state.get(women, parent) - female;
            let mother = u64::from(
                band == Field::YoungAdults
                    && state.get(Field::RecoveringMothers, parent) > retained,
            );
            for (field, amount) in [
                (band, 1),
                (women, female),
                (Field::RecoveringMothers, mother),
            ] {
                state.set(field, parent, state.get(field, parent) - amount);
                state.set(field, household, amount);
            }
            state.set(Field::EligibleWorkers, household, 1);
            for field in [
                Field::CohortFamineExposure,
                Field::CohortColdExposure,
                Field::Fatigue,
            ] {
                state.set(field, household, state.get(field, parent));
            }
        }
        if requests[parent] == SEPARATE_OWNER {
            state.set(
                Field::InvestmentGood,
                household,
                state.get(Field::InvestmentGood, parent),
            );
            state.set(Field::InvestmentGood, parent, 0);
        }
        let remaining = living - 1;
        state.set(Field::LivingMembers, parent, remaining);
        state.set(
            Field::HousingAvailable,
            parent,
            state.get(Field::HousingAvailable, parent).min(remaining),
        );
        state.set(Field::Price, household, state.get(Field::Price, parent));
        state.set(
            Field::FoodPrice,
            household,
            state.get(Field::FoodPrice, parent),
        );
        state.set(Field::EquipmentCondition, household, 1000);
        state.set(Field::SplitPerson, parent, 0);
        if let Some(property) = &scenario.config().property {
            state.set(
                Field::PropertyRegion,
                household,
                property.region(state.actors.at(household).position)? as u64 + 1,
            );
            state.set(Field::PropertyUse, household, PropertyUse::Housing as u64);
        }
    }
    Ok(formations)
}

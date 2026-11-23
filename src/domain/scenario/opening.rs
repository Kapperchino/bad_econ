use super::Scenario;
use crate::{
    arithmetic::add,
    domain::{
        Field, Good, HiringTerm, State, Technique, company, enterprise, food, industry, landlords,
    },
};
use anyhow::{Context, Result, anyhow};

impl Scenario {
    pub(in crate::domain) fn empty_opening_state(&self, next_identity: u64) -> State {
        State {
            day: 0,
            next_identity,
            actors: self.initial_actors.as_ref().clone(),
            values: vec![0; self.fields() * self.actors()],
            initial_cash: 0,
            initial_goods: [0; Good::COUNT],
            research: None,
            people: Vec::new(),
            initial_population: 0,
            ownership: Vec::new(),
            companies: Vec::new(),
        }
    }

    pub fn initial(&self) -> Result<State> {
        let mut state = match &self.config.population {
            Some(population) => population.initial(self),
            None => self.generated(),
        }?;
        if let Some(finance) = &self.config.finance {
            state.values.extend(finance.initial_values());
            state.initial_cash = add(state.initial_cash, finance.initial_cash()?)?;
        }
        for actor in self.population_rows()..self.actors() {
            state.set_recipe(actor, self.production_recipe(actor, None)?);
        }
        state.ownership = state
            .actors
            .firm_actors()
            .map(|actor| enterprise::Ownership {
                firm: actor.identity,
                shares: vec![enterprise::OwnershipShare {
                    household: state.actors.at(actor.owner as usize).identity,
                    units: 1_000_000,
                }],
            })
            .collect();
        for ownership in &self.config.ownership {
            let index = state
                .ownership
                .iter()
                .position(|entry| entry.firm == ownership.firm)
                .context("unknown ownership firm")?;
            for share in &ownership.shares {
                match self
                    .index_of(share.household)
                    .is_some_and(|index| state.actors.at(index).kind == 0)
                {
                    true => (),
                    false => Err(anyhow!("ownership references a missing household"))?,
                }
            }
            state.ownership[index] = ownership.clone();
        }
        state.companies = self.config.companies.clone();
        for company in &self.config.companies {
            company.initialize(&mut state)?;
        }
        company::CompanyRegistry::new(&state)?;
        self.config.railway.audit(&state)?;
        for actor in self.population_rows()..self.actors() {
            state.set(
                Field::EnterpriseStatus,
                actor,
                enterprise::EnterpriseState::Operating as u64,
            );
            let owner = state.actors.at(actor).owner as usize;
            state.set(Field::InvestmentFirm, owner, actor as u64 + 1);
        }
        for distributor in &self.config.distributors {
            let firm = self
                .index_of(distributor.firm)
                .context("unknown distributor identity")?;
            match state.actors.at(firm).kind == distributor.good as u32 {
                true => (),
                false => Err(anyhow!("distributor inventory good must match its firm"))?,
            }
            state.set(
                Field::EnterpriseModel,
                firm,
                enterprise::EnterpriseModel::Distributor as u64,
            );
            state.set(Field::StockCoverage, firm, distributor.reserve_days.into());
            state.set(
                Field::RetailMargin,
                firm,
                distributor.margin_per_mille.into(),
            );
            state.set(
                Field::ReplacementCost,
                firm,
                self.initial_price(distributor.good).into(),
            );
            state.set(
                Field::Demand,
                firm,
                match &self.config.population {
                    None => self.production.daily[distributor.good as usize - 1],
                    Some(_) => {
                        u64::from(state.actors.at(firm).capacity)
                            * u64::from(distributor.units_per_shift)
                    }
                },
            );
            state.set_recipe(
                firm,
                industry::Recipe {
                    technique: Technique::new(
                        distributor.handling_minutes,
                        0,
                        distributor.units_per_shift,
                    )?,
                    inputs: [None, None],
                },
            );
        }
        for actor in self.population_rows()..self.actors() {
            state.retain_basic_recipe(actor);
            for (field, input) in [
                (Field::FirstInputCost, Field::FirstInput),
                (Field::SecondInputCost, Field::SecondInput),
            ] {
                let good = state.get(input, actor);
                if good > 0 {
                    let price = self.initial_price(Good::from_index(good as u32)?);
                    state.set(field, actor, price.into());
                }
            }
        }
        if let Some(land) = &self.config.land_access {
            for index in 0..state.actors.len() {
                state.set(Field::LandPlots, index, 0);
            }
            for grant in &land.allocations {
                let index = self
                    .member_row(grant.household)
                    .filter(|&index| state.actors.at(index).kind == 0)
                    .context("land grant references a missing household")?;
                state.set(
                    Field::LandPlots,
                    index,
                    add(state.get(Field::LandPlots, index), grant.plots.into())?,
                );
            }
        }
        match &self.cohorts {
            Some(layout) => layout.initialize(self, &mut state)?,
            None => {
                state.people = self.config.demography.initial(&state)?;
                state.initial_population = state.people.len() as u64;
            }
        }
        for actor in 0..state.actors.len() {
            state.set(Field::EquipmentCondition, actor, 1000);
            if let Some(firm) = self
                .initial_employment
                .employer(self.source(actor))
                .map(|source| self.row(source))
            {
                let employer = state.actors.at(firm);
                let term = HiringTerm::at_firm(&state, firm, state.get(Field::LandPlots, actor))?;
                state.set(Field::Job, actor, firm as u64 + 1);
                state.set(
                    Field::JobDaysRemaining,
                    actor,
                    term.opening_days(self.config.seed, employer.identity),
                );
            }
            if state.actors.at(actor).kind == 0 {
                state.set(Field::Price, actor, self.config.prices[2].into());
                state.set(Field::FoodPrice, actor, self.config.prices[2].into());
            }
        }
        for setting in &self.config.working_conditions {
            let firm = self
                .index_of(setting.firm)
                .filter(|&index| state.actors.at(index).kind > 0)
                .context("working conditions reference a missing firm")?;
            state.set(
                Field::EquipmentCondition,
                firm,
                setting.conditions.equipment_condition.into(),
            );
            state.set(
                Field::SafetyProvision,
                firm,
                setting.conditions.safety_provision.into(),
            );
        }
        state.next_identity = state
            .people
            .iter()
            .map(|person| person.identity)
            .max()
            .unwrap_or(0)
            .max(state.next_identity - 1)
            .checked_add(1)
            .context("person identity capacity exhausted")?;
        if let Some(property) = &self.config.property {
            property.initialize(&mut state)?;
        }
        landlords::initialize(self, &mut state)?;
        if let (Some(food), None) = (self.config.food, &self.config.property) {
            let mut land = food::FarmLand::new(self.config.land_access.as_ref());
            let farms = (self.population_rows()..state.actors.len())
                .filter(|&actor| {
                    state.actors.at(actor).kind == Good::Grain as u32
                        && state.get(Field::EnterpriseModel, actor) == 0
                })
                .collect::<Vec<_>>();
            for actor in farms {
                let hectares = land.allocate(
                    food.hectares(state.actors.at(actor).capacity, self.technique(actor)),
                );
                state.set(Field::FarmHectares, actor, hectares);
            }
        }
        Ok(state)
    }
}

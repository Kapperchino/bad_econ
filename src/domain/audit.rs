use super::{
    CONSUMPTION_FIELDS, FIRM_RENT_GRACE_DAYS, Field, Good, HiringTerm, HousingState,
    ProductionMethod, Scenario, State, Technique, USED_FIELDS, company, demography, employment,
    enterprise, finance, industry, landlords, property, research,
};
use anyhow::{Result, anyhow};

pub(crate) struct AuditedState<'a> {
    pub(super) state: &'a State,
}
impl<'a> AuditedState<'a> {
    pub(crate) fn new(scenario: &Scenario, state: &'a State) -> Result<Self> {
        let c = scenario.config();
        let _ = enterprise::OwnershipRegistry::new(state)?;
        let _ = company::CompanyRegistry::new(state)?;
        let shape = state.actors.len() >= scenario.actors()
            && state.values.len()
                == state.actors.len() * scenario.fields()
                    + c.finance
                        .as_ref()
                        .map_or(0, finance::FinanceConfig::state_len);
        match shape {
            true => Ok(()),
            false => Err(anyhow!("state shape does not match scenario")),
        }?;
        match scenario.aggregate_demography() {
            true => {
                demography::CohortPopulation::new(state)?;
            }
            _ => {
                demography::PersonRegistry::new(state)?;
            }
        }
        c.railway.audit(state)?;
        scenario.rail_geography().audit(scenario, state)?;
        property::PropertySnapshot::from_state(scenario, state)?;
        let ownership = state
            .ownership
            .iter()
            .map(|entry| (entry.firm, entry))
            .collect::<std::collections::HashMap<_, _>>();
        let identities = state.actors.iter().enumerate().find_map(|(index, actor)| {
            let original = match index < scenario.actors() {
                true => {
                    actor.identity == scenario.actor(index).identity
                        && actor.kind == scenario.actor(index).kind
                }
                false => true,
            };
            let capacity = index >= scenario.actors()
                || actor.capacity == scenario.actor(index).capacity
                || (c.property.is_some()
                    && u128::from(actor.capacity)
                        == u128::from(state.get(Field::WorkersPerPlot, index))
                            * u128::from(state.get(Field::OccupiedPlots, index).max(1)))
                || state.get(Field::InstalledCapital, index) > 0
                || state.get(Field::EnterpriseStatus, index)
                    == enterprise::EnterpriseState::Closed as u64;
            let position = index >= scenario.actors()
                || actor.position == scenario.actor(index).position
                || (u64::from(actor.position.x) == state.get(Field::MoveX, index)
                    && u64::from(actor.position.y) == state.get(Field::MoveY, index))
                || (state.get(Field::MoveDue, index) > 0
                    && u64::from(actor.position.x) == state.get(Field::PreviousX, index)
                    && u64::from(actor.position.y) == state.get(Field::PreviousY, index));
            let owner = actor.kind == 0
                || ownership.get(&actor.identity).is_some_and(|ownership| {
                    ownership.shares.iter().any(|share| {
                        state
                            .actors
                            .get(actor.owner as usize)
                            .is_some_and(|owner| owner.identity == share.household)
                    })
                });
            let admitted = original
                && capacity
                && position
                && owner
                && actor.kind <= Good::COUNT as u32
                && actor.owner < state.actors.len() as u32
                && actor.position.x < c.map_size
                && actor.position.y < c.map_size;
            (!admitted).then_some((actor, original, capacity, position, owner))
        });
        match identities {
            None => Ok(()),
            Some((actor, original, capacity, position, owner)) => Err(anyhow!(
                "invalid actor registry: {actor:?}; identity={original}, capacity={capacity}, position={position}, ownership={owner}"
            )),
        }?;
        let resources = (0..state.actors.len()).all(|i| {
            let remaining = state.get(Field::JobDaysRemaining, i);
            let job = match state.get(Field::Job, i) {
                0 => remaining == 0,
                employer => {
                    let firm = (employer - 1) as usize;
                    state.actors.at(i).kind == 0
                        && state.actors.get(firm).is_some_and(|actor| actor.kind > 0)
                        && (remaining <= 31
                            || HiringTerm::at_firm(state, firm, 0)
                                .is_ok_and(|term| remaining <= term as u64))
                }
            };
            let limits = state.get(Field::HousingStatus, i) <= HousingState::Homeless as u64
                && state.get(Field::Landlord, i) <= state.actors.len() as u64
                && state.get(Field::RentPending, i) <= state.get(Field::RentTreasury, i)
                && state.get(Field::HousingAvailable, i)
                    <= state
                        .get(Field::LivingMembers, i)
                        .max(u64::from(state.actors.at(i).capacity))
                        .max(1)
                && state.get(Field::MaintenanceDays, i) <= u64::from(landlords::MAINTENANCE_DAYS)
                && state.get(Field::RentOverdueDays, i) <= u64::from(FIRM_RENT_GRACE_DAYS)
                && state.get(Field::EquipmentCondition, i) <= 1000
                && state.get(Field::SafetyProvision, i) <= 900
                && state.get(Field::EnterpriseStatus, i)
                    <= enterprise::EnterpriseState::Closed as u64
                && state.get(Field::EnterpriseModel, i)
                    <= enterprise::EnterpriseModel::Distributor as u64
                && state.get(Field::FoodAccess, i)
                    <= enterprise::FoodAccess::MatchingFailure as u64
                && state.get(Field::MillingRemainder, i) < c.food.map_or(1, |_| 5)
                && state.get(Field::BakingRemainder, i) < c.food.map_or(1, |_| 25)
                && state.get(Field::Fatigue, i) <= 1000
                && state.get(Field::Time, i) <= state.get(Field::HouseholdBudget, i).max(1440)
                && state.get(Field::Hours, i) <= state.get(Field::HouseholdBudget, i).max(1440)
                && state.get(Field::ResearchMinutes, i) <= 1440
                && state.get(Field::Employed, i) <= state.get(Field::EligibleWorkers, i).max(1)
                && state.get(Field::Employer, i) <= state.actors.len() as u64
                && job;
            let closed = [
                Field::Escrow,
                Field::Due,
                Field::Pipeline,
                Field::FreightEscrow,
            ]
            .into_iter()
            .all(|field| state.get(field, i) == 0);
            let actor = &state.actors.at(i);
            let recipe = match (actor.kind, state.get(Field::EnterpriseModel, i)) {
                (_, 1) => {
                    state.get(Field::FirstInput, i) == 0
                        && state.get(Field::SecondInput, i) == 0
                        && Technique::new(
                            state.get(Field::LaborMinutes, i) as u32,
                            0,
                            state.get(Field::OutputTicks, i) as u32,
                        )
                        .is_ok()
                }
                (kind, _) if kind == Good::Railway as u32 => {
                    (1..=1440).contains(&state.get(Field::LaborMinutes, i))
                        && state.get(Field::FirstInput, i) == 0
                        && state.get(Field::SecondInput, i) == 0
                        && state.get(Field::InputTicks, i) == 0
                        && state.get(Field::OutputTicks, i) == 1
                }
                (kind, _) if kind == Good::Housing as u32 => {
                    actor.capacity == 0
                        && state.get(Field::LaborMinutes, i) == 0
                        && state.get(Field::FirstInput, i) == Good::Construction as u64
                        && state.get(Field::SecondInput, i) == 0
                        && (1..=100).contains(&state.get(Field::OutputTicks, i))
                        && (1..=100).contains(&state.get(Field::InputTicks, i))
                }
                (0, _) => [
                    Field::LaborMinutes,
                    Field::InputTicks,
                    Field::OutputTicks,
                    Field::FirstInput,
                    Field::SecondInput,
                    Field::ActiveMethod,
                ]
                .into_iter()
                .all(|field| state.get(field, i) == 0),
                _ => ProductionMethod::from_index(state.get(Field::ActiveMethod, i))
                    .and_then(|method| match i < scenario.actors() {
                        true => scenario.production_recipe(i, method),
                        false => Ok(industry::Recipe {
                            technique: Technique::new(
                                state.get(Field::LaborMinutes, i) as u32,
                                state.get(Field::InputTicks, i) as u32,
                                state.get(Field::OutputTicks, i) as u32,
                            )?,
                            inputs: Good::from_index(actor.kind)?.inputs(),
                        }),
                    })
                    .is_ok_and(|recipe| {
                        state.get(Field::LaborMinutes, i)
                            == u64::from(recipe.technique.labor_minutes)
                            && state.get(Field::InputTicks, i)
                                == u64::from(recipe.technique.input_ticks)
                            && state.get(Field::OutputTicks, i)
                                == u64::from(recipe.technique.output_ticks)
                            && state.get(Field::FirstInput, i)
                                == recipe.inputs[0].map_or(0, |good| good as u64)
                            && state.get(Field::SecondInput, i)
                                == recipe.inputs[1].map_or(0, |good| good as u64)
                    }),
            };
            let used = (0..Good::COUNT)
                .map(|good| u128::from(state.values[(USED_FIELDS + good) * state.actors.len() + i]))
                .sum::<u128>();
            let accounting = used == u128::from(state.get(Field::Used, i))
                && state.values
                    [(CONSUMPTION_FIELDS + Good::Bread as usize - 1) * state.actors.len() + i]
                    == state.get(Field::Consumed, i);
            let daily_budget = u128::from(state.get(Field::HouseholdBudget, i));
            let labor = match (
                actor.kind,
                state.get(Field::Employed, i),
                state.get(Field::Employer, i),
            ) {
                (0, 0, 0) => {
                    state.get(Field::Hours, i) == 0
                        && u128::from(state.get(Field::Time, i))
                            + u128::from(state.get(Field::CartageMinutes, i))
                            + u128::from(state.get(Field::HomeMinutes, i))
                            + u128::from(state.get(Field::ResearchMinutes, i))
                            <= daily_budget
                }
                (0, workers, employer)
                    if workers > 0 && (1..=state.actors.len() as u64).contains(&employer) =>
                {
                    let employer = (employer - 1) as usize;
                    let employer_capacity = scenario.operating_capacity(state, employer);
                    let labor_minutes = state.get(Field::LaborMinutes, employer);
                    let commute = actor
                        .position
                        .distance(state.actors.at(employer).position)
                        .div_ceil(c.commute_speed)
                        * 2;
                    let hours = state.get(Field::Hours, i);
                    let travel = match hours {
                        0 => 0,
                        _ => commute,
                    };
                    state.actors.at(employer).kind > 0
                        && (hours == 0
                            || (employer_capacity > 0
                                && u128::from(hours)
                                    == u128::from(labor_minutes) * u128::from(workers)))
                        && u128::from(hours)
                            + u128::from(state.get(Field::Time, i))
                            + u128::from(state.get(Field::CartageMinutes, i))
                            + u128::from(state.get(Field::HomeMinutes, i))
                            + u128::from(state.get(Field::ResearchMinutes, i))
                            + u128::from(travel) * u128::from(workers)
                            <= daily_budget
                }
                (kind, 0, 0) if (1..=Good::COUNT as u32).contains(&kind) => {
                    state.get(Field::Hours, i) == 0 && state.get(Field::Time, i) == 0
                }
                _ => false,
            };
            let capacity = scenario.operating_capacity(state, i);
            let planning = state.get(Field::Slots, i) <= state.get(Field::Plan, i)
                && state.get(Field::Plan, i) <= u64::from(capacity);
            let production = match (actor.kind, capacity) {
                (kind, _) if kind == Good::Housing as u32 || kind == Good::Railway as u32 => true,
                (_, 0) => state.get(Field::Produced, i) == 0,
                _ => true,
            };
            let role = match actor.kind {
                0 => [Field::Slots, Field::Plan, Field::Produced]
                    .into_iter()
                    .all(|field| state.get(field, i) == 0),
                _ => [Field::Consumed, Field::Shortage]
                    .into_iter()
                    .all(|field| state.get(field, i) == 0),
            };
            let cleared_slot = ((state.day % scenario.ring() as u64 + scenario.ring() as u64 - 1)
                % scenario.ring() as u64) as usize;
            let delivery_window = Good::ALL.into_iter().all(|good| {
                state.values[state
                    .shipment_coordinate(good, cleared_slot, scenario.ring(), i)
                    .0 as usize]
                    == 0
            });
            let quotes = match actor.kind {
                0 => state.get(Field::GrainQuote, i) <= u64::from(c.maximum_price),
                _ => {
                    (1..=10_000).contains(&state.get(Field::Wage, i))
                        && (u64::from(c.minimum_price)..=u64::from(c.maximum_price))
                            .contains(&state.get(Field::Price, i))
                }
            };
            let payroll = match (&c.payroll, actor.kind) {
                (Some(_), 0) => state.get(Field::PayrollReserve, i) <= 2,
                (None, 0) => [
                    Field::PayrollReserve,
                    Field::TaxTreasury,
                    Field::ChargeTreasury,
                ]
                .into_iter()
                .all(|field| state.get(field, i) == 0),
                _ => [
                    Field::GrossIncome,
                    Field::PayrollReserve,
                    Field::TaxTreasury,
                    Field::ChargeTreasury,
                ]
                .into_iter()
                .all(|field| state.get(field, i) == 0),
            };
            limits
                && closed
                && labor
                && planning
                && production
                && role
                && delivery_window
                && quotes
                && payroll
                && recipe
                && accounting
        });
        let commercial_land = c.food.map_or(0, |food| {
            (scenario.population_rows()..scenario.actors())
                .filter(|&index| {
                    scenario.actor(index).kind == Good::Grain as u32
                        && scenario
                            .distributor(scenario.actor(index).identity)
                            .is_none()
                })
                .map(|index| {
                    food.hectares(scenario.actor(index).capacity, scenario.technique(index))
                })
                .sum::<u64>()
        });
        let land_limit = match &c.land_access {
            Some(land) => u64::from(land.total_plots),
            None => {
                u64::from(employment::InitialSettlement::new(c).farmers(c.population_size()))
                    * u64::from(c.demography.initial_household_members.div_ceil(2))
                    + commercial_land
            }
        };
        let land = (0..state.actors.len())
            .map(|index| {
                let consolidated = match (&c.property, state.actors.at(index).kind) {
                    (None, kind) if kind == Good::Grain as u32 => {
                        state.get(Field::OccupiedPlots, index)
                            + state.get(Field::FarmHectares, index)
                    }
                    _ => 0,
                };
                u128::from(state.get(Field::LandPlots, index)) + u128::from(consolidated)
            })
            .sum::<u128>();
        let valid = resources
            && land <= u128::from(land_limit)
            && state.initial_population
                == match c.demography.people.is_empty() {
                    true => {
                        u64::from(c.population_size())
                            * u64::from(c.demography.initial_household_members)
                    }
                    false => c.demography.people.len() as u64,
                }
            && match (&c.research, &state.research) {
                (None, None) => true,
                (Some(_), None) => state.day == 0,
                (Some(_), Some(snapshot)) => {
                    snapshot.profile == research::RESEARCH_PROFILE && !snapshot.words.is_empty()
                }
                (None, Some(_)) => false,
            }
            && state
                .actors
                .iter()
                .map(|actor| actor.identity)
                .chain(state.people.iter().map(|person| person.identity))
                .all(|identity| identity < state.next_identity)
            && state.day <= u64::from(u32::MAX) - scenario.ring() as u64
            && state.initial_cash == scenario.opening_totals().cash
            && state.initial_goods == scenario.opening_totals().goods;
        match valid {
            true => Ok(Self { state }),
            false => Err(anyhow!(
                "invalid identity, policy bounds, opening totals, date, or unsettled day"
            )),
        }
    }
}

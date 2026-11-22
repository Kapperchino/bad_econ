use super::{acquisitions, formation::form_households, settlement::Settlement};
use crate::{
    arithmetic::add,
    domain::{Economy, Field, Position, Scenario, State, demography::LivingState},
};
use anyhow::{Context, Result, anyhow};

pub(in crate::simulation) fn transitions(
    state: &mut State,
    scenario: &Scenario,
    requests: &[u32],
) -> Result<()> {
    transitions_in_rows(state, scenario, requests, |row| row)
}

pub(in crate::simulation) fn transitions_in_rows(
    state: &mut State,
    scenario: &Scenario,
    requests: &[u32],
    global_row: impl Fn(usize) -> usize,
) -> Result<()> {
    let formations = form_households(state, scenario, requests)?;
    if !state.people.is_empty() {
        for formation in formations {
            let person = state
                .people
                .iter_mut()
                .find(|person| person.identity == formation.person)
                .context("unknown household formation person")?;
            match person.state {
                LivingState::Living => (),
                _ => Err(anyhow!("deceased person cannot form a household"))?,
            }
            person.household = state.actors.at(formation.household).identity;
        }
    }
    let settlement = Settlement::new(scenario, state);
    for (actor, &action) in requests.iter().enumerate() {
        match action >> 24 {
            2 => {
                let destination = (action & 0x00ffffff)
                    .checked_sub(1)
                    .context("missing migration destination")?
                    as usize;
                let position = state
                    .actors
                    .get(destination)
                    .context("unknown migration destination")?
                    .position;
                let cash = state
                    .get(Field::Cash, actor)
                    .checked_sub(100)
                    .context("unfunded migration")?;
                state.set(Field::Cash, actor, cash);
                state.set(
                    Field::Cash,
                    destination,
                    add(state.get(Field::Cash, destination), 100)?,
                );
                let previous = state.actors.at(actor).position;
                state.set(Field::PreviousX, actor, previous.x.into());
                state.set(Field::PreviousY, actor, previous.y.into());
                state.set(Field::MoveX, actor, position.x.into());
                state.set(Field::MoveY, actor, position.y.into());
                let duration = previous
                    .distance(position)
                    .div_ceil(scenario.config().shipment_speed)
                    .max(1);
                state.set(Field::MoveDue, actor, state.day + u64::from(duration));
            }
            3 => {
                let mut household = state.actors.at(actor);
                household.position = Position {
                    x: state.get(Field::MoveX, actor) as u32,
                    y: state.get(Field::MoveY, actor) as u32,
                };
                state.actors.set(actor, household);
                state.set(Field::MoveDue, actor, 0);
                state.set(Field::MoveTarget, actor, 0);
                if scenario.config().property.is_some()
                    || scenario.config().economy == Economy::Industrial
                {
                    state.set(Field::LandPlots, actor, 0);
                    state.set(Field::CropWork, actor, 0);
                }
            }
            10 => {
                state.set(Field::MoveTarget, actor, 0);
                state.set(Field::LandPlots, actor, 0);
                state.set(Field::CropWork, actor, 0);
            }
            6 => {
                let mut firm = state.actors.at(actor);
                firm.capacity = firm
                    .capacity
                    .checked_add(state.get(Field::PendingCapacity, actor) as u32)
                    .context("firm capacity exhausted")?;
                state.actors.set(actor, firm);
                state.set(Field::PendingCapacity, actor, 0);
            }
            7 => settlement.invest(state, actor, (action & 0x00ff_ffff) as usize - 1)?,
            4 => settlement.close_firm(state, actor)?,
            5 => settlement.inherit(state, actor)?,
            _ => (),
        }
    }
    acquisitions::settle(state, scenario, requests, global_row)
}

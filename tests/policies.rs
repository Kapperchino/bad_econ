#![cfg(feature = "compute")]
mod support;
use anyhow::Result;
use bad_econ::{
    arithmetic,
    domain::*,
    simulation::{Simulation, Trace},
};

fn planning_scenario() -> Result<Scenario> {
    let mut config = Scenario::baseline(12, 81, Experiment::Baseline)?
        .config()
        .clone();
    config.firms = 6;
    config.distribution_rate = 0;
    let position = Position { x: 0, y: 0 };
    config.population = Some(Population {
        households: (1..=12)
            .map(|identity| HouseholdSeed {
                identity,
                position,
                cash: 2000,
                inventories: Good::food([0, 0, 2]),
                health: 1000,
                fatigue: 0,
            })
            .collect(),
        firms: vec![
            FirmSeed {
                identity: 100,
                position,
                owner: 1,
                good: Good::Grain,
                cash: 120,
                inventories: Good::food([0, 0, 0]),
                capacity: 7,
                technique: Technique::new(60, 0, 3)?,
                wage: 2,
                price: 100,
            },
            FirmSeed {
                identity: 200,
                position,
                owner: 2,
                good: Good::Grain,
                cash: 100_000,
                inventories: Good::food([0, 0, 0]),
                capacity: 2,
                technique: Technique::new(90, 0, 5)?,
                wage: 2,
                price: 105,
            },
            FirmSeed {
                identity: 300,
                position,
                owner: 3,
                good: Good::Flour,
                cash: 100_000,
                inventories: Good::food([1, 0, 0]),
                capacity: 20,
                technique: Technique::new(120, 2, 3)?,
                wage: 3,
                price: 200,
            },
            FirmSeed {
                identity: 400,
                position,
                owner: 4,
                good: Good::Flour,
                cash: 100_000,
                inventories: Good::food([20, 115, 0]),
                capacity: 20,
                technique: Technique::new(180, 2, 3)?,
                wage: 3,
                price: 205,
            },
            FirmSeed {
                identity: 500,
                position,
                owner: 5,
                good: Good::Bread,
                cash: 100_000,
                inventories: Good::food([0, 15, 0]),
                capacity: 20,
                technique: Technique::new(240, 3, 2)?,
                wage: 4,
                price: 300,
            },
            FirmSeed {
                identity: 600,
                position,
                owner: 6,
                good: Good::Bread,
                cash: 100_000,
                inventories: Good::food([0, 15, 1000]),
                capacity: 20,
                technique: Technique::new(300, 3, 2)?,
                wage: 4,
                price: 305,
            },
        ],
    });
    Scenario::new(config)
}

fn planned_packets(scenario: &Scenario, state: &State, index: usize) -> Result<u64> {
    let actor = &state.actors.at(index);
    let technique = scenario.technique(index);
    let output = Good::from_index(actor.kind)?;
    let demand =
        u128::from(state.get(Field::Demand, index)) * u128::from(scenario.config().coverage);
    let available = u128::from(state.get(Field::inventory(output), index))
        + u128::from(state.get(Field::Pipeline, index));
    let requested = demand
        .saturating_sub(available)
        .div_ceil(u128::from(technique.output_ticks));
    let inputs = output
        .inputs()
        .into_iter()
        .flatten()
        .map(|good| {
            u128::from(state.get(Field::inventory(good), index)) / u128::from(technique.input_ticks)
        })
        .min()
        .unwrap_or(u128::MAX);
    let funding = u128::from(state.get(Field::Cash, index))
        / (u128::from(state.get(Field::Wage, index)) * u128::from(technique.labor_minutes));
    Ok(u64::try_from(
        requested
            .min(inputs)
            .min(funding)
            .min(u128::from(actor.capacity)),
    )?)
}

#[test]
fn proposal_and_decision_kernels_match_independent_policy_formulas() -> Result<()> {
    let scenario = planning_scenario()?;
    let mut simulation = Simulation::new(scenario.clone(), support::device(64)?)?;
    for day in 0..6 {
        simulation.phase(Phase::Opening, Trace::Record)?;
        simulation.phase(Phase::Observation, Trace::Record)?;
        let observed = simulation.download()?;
        simulation.phase(Phase::Proposal, Trace::Record)?;
        let proposed = simulation.download()?;
        for firm in scenario.config().households as usize..scenario.actors() {
            let expected = planned_packets(&scenario, &observed, firm)?;
            assert_eq!(proposed.get(Field::Plan, firm), expected);
            assert_eq!(proposed.get(Field::Slots, firm), expected);
        }
        if day == 0 {
            let first_plans = (12..18)
                .map(|firm| proposed.get(Field::Plan, firm))
                .collect::<Vec<_>>();
            assert_eq!(first_plans, [1, 2, 0, 2, 5, 0]);
        }
        for phase in [
            Phase::Commitment,
            Phase::WorkPayment,
            Phase::ProvisionPurchase,
            Phase::PhysicalClose,
        ] {
            simulation.phase(phase, Trace::Record)?;
        }
        let before = simulation.download()?;
        simulation.phase(Phase::AccountsDecisions, Trace::Record)?;
        let after = simulation.download()?;
        for firm in scenario.config().households as usize..scenario.actors() {
            let demand = arithmetic::forecast(
                before.get(Field::Demand, firm),
                before.get(Field::Sales, firm),
                before.get(Field::Unfilled, firm),
                u64::from(scenario.config().forecast_weight),
            )?;
            let good = Good::from_index(before.actors.at(firm).kind)?;
            let excess = before
                .get(Field::inventory(good), firm)
                .saturating_sub(demand);
            let price = arithmetic::price(
                before.get(Field::Price, firm),
                before.get(Field::Unfilled, firm),
                excess,
                demand,
                u64::from(scenario.config().minimum_price),
                u64::from(scenario.config().maximum_price),
            )?;
            let increment = match before.get(Field::Slots, firm) {
                0 => 0,
                _ => u64::from(scenario.config().behavior.wage_increment),
            };
            let offered = before.get(Field::Wage, firm) + increment;
            let funded = before.get(Field::Cash, firm)
                / u64::from(scenario.technique(firm).labor_minutes)
                / before.get(Field::Plan, firm).max(1);
            assert_eq!(after.get(Field::Demand, firm), demand);
            assert_eq!(after.get(Field::Price, firm), price);
            assert_eq!(
                after.get(Field::Wage, firm),
                offered.min(funded).clamp(1, 10_000)
            );
        }
        simulation.phase(Phase::Audit, Trace::Record)?;
    }
    Ok(())
}

#[test]
fn commuting_rounds_up_and_can_exclude_a_higher_wage() -> Result<()> {
    let mut config = planning_scenario()?.config().clone();
    config.households = 3;
    config.firms = 3;
    config.commute_speed = 10;
    config.population = Some(Population {
        households: (1..=3)
            .map(|identity| HouseholdSeed {
                identity,
                position: Position { x: 0, y: 0 },
                cash: 2000,
                inventories: Good::food([0, 0, 2]),
                health: 1000,
                fatigue: 0,
            })
            .collect(),
        firms: vec![
            FirmSeed {
                identity: 100,
                position: Position { x: 11, y: 0 },
                owner: 1,
                good: Good::Grain,
                cash: 100_000,
                inventories: Good::food([0, 0, 0]),
                capacity: 3,
                technique: Technique::new(460, 0, 3)?,
                wage: 1,
                price: 100,
            },
            FirmSeed {
                identity: 200,
                position: Position { x: 101, y: 0 },
                owner: 2,
                good: Good::Flour,
                cash: 15_000_000,
                inventories: Good::food([9, 0, 0]),
                capacity: 3,
                technique: Technique::new(460, 3, 3)?,
                wage: 10_000,
                price: 200,
            },
            FirmSeed {
                identity: 300,
                position: Position { x: 0, y: 0 },
                owner: 3,
                good: Good::Bread,
                cash: 100_000,
                inventories: Good::food([0, 0, 0]),
                capacity: 3,
                technique: Technique::new(480, 3, 3)?,
                wage: 2,
                price: 300,
            },
        ],
    });
    let scenario = Scenario::new(config)?;
    let mut simulation = Simulation::new(scenario.clone(), support::device(64)?)?;
    for phase in [
        Phase::Opening,
        Phase::Observation,
        Phase::Proposal,
        Phase::Commitment,
    ] {
        simulation.phase(phase, Trace::Record)?;
    }
    let state = simulation.download()?;
    for household in 0..3 {
        assert_eq!(state.get(Field::Employer, household), 4);
        assert_eq!(state.get(Field::Hours, household), 460);
        assert_eq!(state.get(Field::Time, household), 16);
        assert_eq!(state.get(Field::Due, household), 460);
        assert_eq!(state.get(Field::Employed, household), 1);
    }
    assert_eq!(state.get(Field::Slots, 3), 0);
    assert_eq!(state.get(Field::Slots, 4), 3);
    assert_eq!(state.get(Field::Escrow, 3), 1380);
    assert_eq!(state.get(Field::Cash, 4), 15_000_000);
    for phase in [
        Phase::WorkPayment,
        Phase::ProvisionPurchase,
        Phase::PhysicalClose,
        Phase::AccountsDecisions,
        Phase::Audit,
    ] {
        simulation.phase(phase, Trace::Record)?;
    }
    assert_eq!(simulation.snapshot.employed, 3);
    assert!(simulation.checkpoint()?.restore().is_ok());
    Ok(())
}

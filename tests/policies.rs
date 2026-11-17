use super::support;
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
    config.resources = ResourceMap::Explicit {
        regions: vec![ResourceRegion::new(
            Good::Wood,
            Position { x: 0, y: 0 },
            Position {
                x: config.map_size,
                y: config.map_size,
            },
            config.map_size,
        )?],
    };
    config.population = Some(Population {
        households: (1..=12)
            .map(|identity| HouseholdSeed {
                identity,
                position,
                cash: 2000,
                inventories: Good::food([0, 0, 2]),
                fatigue: 0,
            })
            .collect(),
        firms: vec![
            FirmSeed {
                identity: 100,
                position,
                owner: 1,
                good: Good::Wood,
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
                good: Good::Wood,
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
    let observed = u128::from(state.get(Field::Demand, index));
    let daily = match output {
        Good::Grain | Good::Flour | Good::Bread => (observed * 5).div_ceil(4),
        _ => observed,
    };
    let demand = daily * u128::from(scenario.config().coverage);
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
            assert_eq!(first_plans, [1, 2, 0, 10, 5, 0]);
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
            let daily = match good {
                Good::Grain | Good::Flour | Good::Bread => (u128::from(demand) * 5).div_ceil(4),
                _ => u128::from(demand),
            };
            let target = daily * u128::from(scenario.config().coverage)
                + u128::from(scenario.technique(firm).output_ticks - 1);
            let excess = u64::try_from(
                u128::from(before.get(Field::inventory(good), firm)).saturating_sub(target),
            )?;
            let scale = 1_000_000u128;
            let previous = u128::from(before.get(Field::PriceQuote, firm));
            let quote = u128::from(before.get(Field::Price, firm));
            let precise = match (previous + scale / 2) / scale == quote {
                true => previous,
                false => quote * scale,
            };
            let imbalance = (i128::from(before.get(Field::Unfilled, firm)) - i128::from(excess))
                * 32
                / i128::from(demand.max(1));
            let multiplier = arithmetic::PRICE_TABLE[(32 + imbalance.clamp(-32, 32)) as usize];
            let precise = (precise * u128::from(multiplier) / scale).clamp(
                u128::from(scenario.config().minimum_price) * scale,
                u128::from(scenario.config().maximum_price) * scale,
            );
            let price = u64::try_from((precise + scale / 2) / scale)?;
            let increment = u64::from(scenario.config().behavior.wage_increment);
            let offered = match before.get(Field::Slots, firm) {
                0 => before
                    .get(Field::Wage, firm)
                    .saturating_sub(increment)
                    .max(1),
                _ => before.get(Field::Wage, firm) + increment,
            };
            let funded = before.get(Field::Cash, firm)
                / u64::from(scenario.technique(firm).labor_minutes)
                / before.get(Field::Plan, firm).max(1);
            assert_eq!(after.get(Field::Demand, firm), demand);
            assert_eq!(after.get(Field::Price, firm), price);
            assert_eq!(u128::from(after.get(Field::PriceQuote, firm)), precise);
            assert_eq!(
                after.get(Field::Wage, firm),
                offered.min(funded).clamp(1, 10_000)
            );
        }
        simulation.phase(Phase::Audit, Trace::Record)?;
    }
    Ok(())
}

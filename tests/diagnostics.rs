#![cfg(feature = "compute")]
use crate::support;
use anyhow::Result;
use bad_econ::{
    domain::{demography::LivingState, *},
    simulation::{Simulation, Trace},
};

fn economy() -> Result<Scenario> {
    let mut config = Scenario::industrial(100, 42, Experiment::Concentrated)?
        .config()
        .clone();
    config.entrepreneurship.interested_per_mille = 0;
    config.expansion = enterprise::ExpansionPolicy {
        small_business_per_mille: 0,
        industrial_per_mille: 0,
    };
    config.demography.annual_births_per_million_fertile_women = 0;
    config.demography.mortality.background_annual_ppm = [0; 5];
    config.demography.mortality.occupational_annual_ppm = [0; Good::COUNT];
    Scenario::new(config)
}

#[test]
fn business_accounts_measure_trading_cash_before_price_changes_and_payouts_on_gpu() -> Result<()> {
    let scenario = economy()?;
    let mut simulation = Simulation::new(scenario.clone(), support::device(64)?)?;
    for phase in [Phase::Opening, Phase::Observation, Phase::Proposal] {
        simulation.phase(phase, Trace::Discard)?;
    }
    let opening = simulation.download()?;
    for phase in [
        Phase::Commitment,
        Phase::WorkPayment,
        Phase::ProvisionPurchase,
        Phase::PhysicalClose,
    ] {
        simulation.phase(phase, Trace::Discard)?;
    }
    let trading = simulation.download()?;
    for phase in [Phase::AccountsDecisions, Phase::Audit] {
        simulation.phase(phase, Trace::Discard)?;
    }
    let closing = simulation.download()?;
    let accounts = &simulation.snapshot.economy.accounts;
    assert!(accounts.iter().any(|account| account.profit() < 0));
    assert!(accounts.iter().any(|account| account.revenue > 0));
    for (index, actor) in closing
        .actors
        .iter()
        .enumerate()
        .filter(|(_, actor)| actor.kind > 0)
    {
        let rent_received =
            closing.get(Field::RentalIncome, index) - opening.get(Field::RentalIncome, index);
        let revenue = match actor.kind {
            kind if kind == Good::Housing as u32 => rent_received,
            _ => trading.get(Field::Sales, index) * trading.get(Field::Price, index),
        };
        assert_eq!(
            closing.get(Field::BusinessOpeningCash, index),
            opening.get(Field::Cash, index)
        );
        assert_eq!(
            closing.get(Field::BusinessClosingCash, index),
            trading.get(Field::Cash, index) + rent_received
        );
        assert_eq!(closing.get(Field::BusinessRevenue, index), revenue);
        let inspection = simulation.inspect(InspectionRequest::new(actor.identity, &scenario)?)?;
        let account = inspection.business_account.unwrap();
        assert_eq!(account.revenue, revenue);
        assert!(account.outlays() >= 0);
        assert_eq!(account.profit() + account.outlays(), i128::from(revenue));
    }
    assert_eq!(
        simulation.snapshot,
        ObservationSnapshot::from_state(&scenario, &closing)?
    );
    assert_eq!(
        simulation
            .snapshot
            .economy
            .active_by_good
            .iter()
            .sum::<u64>(),
        closing
            .actors
            .iter()
            .enumerate()
            .filter(|(index, actor)| actor.kind > 0
                && closing.get(Field::EnterpriseStatus, *index)
                    < enterprise::EnterpriseState::Closed as u64)
            .count() as u64
    );
    let checkpoint = simulation.checkpoint()?;
    let mut replay = Simulation::from_checkpoint(checkpoint, support::device(64)?)?;
    assert_eq!(simulation.snapshot, replay.snapshot);
    simulation.advance()?;
    replay.advance()?;
    assert_eq!(simulation.snapshot, replay.snapshot);
    Ok(())
}

#[test]
fn occupations_partition_living_people_on_gpu() -> Result<()> {
    let scenario = economy()?;
    let mut simulation = Simulation::new(scenario.clone(), support::device(64)?)?;
    for _ in 0..3 {
        simulation.advance()?;
        let state = simulation.download()?;
        let population = &simulation.snapshot.population;
        assert_eq!(
            population.occupations.iter().sum::<u64>(),
            population.living
        );
        assert_eq!(population.occupations[4], population.dependents);
        assert!(population.occupations[0] > 0);
        assert!(population.occupations[2] > 0);
        assert_eq!(
            *population,
            demography::PopulationObservation::from_state(&state, &scenario.config().demography)
        );
        assert_eq!(
            population.living,
            state
                .people
                .iter()
                .filter(|person| person.state == LivingState::Living)
                .count() as u64
        );
        let economy = &simulation.snapshot.economy;
        assert_eq!(
            economy.workforce + economy.subsistence + economy.inactive,
            simulation.snapshot.households
        );
        assert_eq!(
            economy.business_types.iter().sum::<u64>(),
            economy.active_by_good.iter().sum::<u64>()
        );
        assert_eq!(
            economy.business_states.iter().sum::<u64>(),
            economy.active_by_good.iter().sum::<u64>()
        );
    }
    Ok(())
}

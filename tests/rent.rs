use super::support;
use anyhow::Result;
use bad_econ::{
    domain::{enterprise::HouseholdProduction, *},
    simulation::{Simulation, Trace},
};

#[test]
fn daily_rent_conserves_cash_and_replays_arrears_on_gpu() -> Result<()> {
    let mut config = Scenario::baseline(6, 42, Experiment::Concentrated)?
        .config()
        .clone();
    config.rent = Some(RentPolicy {
        household_daily: 60,
        firm_daily: 60,
    });
    config.household_production = HouseholdProduction::industrial();
    config.household_cash = 30;
    config.firm_cash = 600;
    config.distribution_rate = 0;
    config.demography.annual_births_per_million_fertile_women = 0;
    config.demography.mortality.background_annual_ppm = [0; 5];
    config.demography.mortality.occupational_annual_ppm = [0; Good::COUNT];
    config.demography.mortality.famine_daily_ppm_at_thirty_days = 0;
    let scenario = Scenario::new(config)?;
    let mut initial = scenario.initial()?;
    for actor in 6..initial.actors.len() {
        initial.set(Field::Demand, actor, 0);
    }
    let checkpoint = Checkpoint::new(&scenario, &initial)?;
    let mut simulation = Simulation::from_checkpoint(checkpoint, support::device(64)?)?;
    for phase in [
        Phase::Opening,
        Phase::Observation,
        Phase::Proposal,
        Phase::Commitment,
    ] {
        simulation.phase(phase, Trace::Record)?;
    }
    let before = simulation.download()?;
    assert_eq!(before.get(Field::RentTreasury, 1), 0);
    simulation.phase(Phase::WorkPayment, Trace::Record)?;
    let paid = simulation.download()?;
    for household in 0..6 {
        assert_eq!(paid.get(Field::Cash, household), 30);
        assert_eq!(paid.get(Field::RentTreasury, household), 0);
        assert_eq!(paid.get(Field::RentArrears, household), 0);
    }
    for firm in
        (6..paid.actors.len()).filter(|&firm| paid.actors.at(firm).kind != Good::Housing as u32)
    {
        assert_eq!(
            paid.get(Field::Cash, firm),
            before.get(Field::Cash, firm) - 60
        );
        assert_eq!(paid.get(Field::RentTreasury, firm), 60);
        assert_eq!(paid.get(Field::RentArrears, firm), 0);
    }
    simulation.phase(Phase::ProvisionPurchase, Trace::Record)?;
    let purchased = simulation.download()?;
    for household in 0..6 {
        match initial.get(Field::LandPlots, household) {
            0 => {
                assert_eq!(purchased.get(Field::Cash, household), 0);
                assert_eq!(purchased.get(Field::RentTreasury, household), 30);
                assert_eq!(purchased.get(Field::RentArrears, household), 30);
            }
            _ => {
                assert_eq!(purchased.get(Field::Cash, household), 30);
                assert_eq!(purchased.get(Field::RentTreasury, household), 0);
                assert_eq!(purchased.get(Field::RentArrears, household), 0);
            }
        }
    }
    for phase in [Phase::PhysicalClose, Phase::AccountsDecisions, Phase::Audit] {
        simulation.phase(phase, Trace::Record)?;
    }
    assert_eq!(simulation.snapshot.rent_collected, 270);
    assert_eq!(simulation.snapshot.rent_arrears, 90);
    assert_eq!(simulation.snapshot.cash, initial.initial_cash);
    let state = simulation.download()?;
    assert_eq!(
        simulation.snapshot,
        ObservationSnapshot::from_state(&scenario, &state)?
    );
    let request = InspectionRequest::new(2, &scenario)?;
    assert_eq!(
        simulation.inspect(request.clone())?,
        Inspection::from_state(&scenario, &state, &request)?
    );
    let encoded = serde_json::to_string(&simulation.checkpoint()?)?;
    let mut resumed =
        Simulation::from_checkpoint(serde_json::from_str(&encoded)?, support::device(64)?)?;
    for day in 2..=4 {
        simulation.advance()?;
        resumed.advance()?;
        assert_eq!(simulation.download()?, resumed.download()?);
        assert_eq!(simulation.snapshot, resumed.snapshot);
        assert_eq!(
            simulation.snapshot.rent_collected + simulation.snapshot.rent_arrears,
            360 + 180 * (day - 1)
        );
        assert_eq!(simulation.snapshot.cash, initial.initial_cash);
    }
    Ok(())
}

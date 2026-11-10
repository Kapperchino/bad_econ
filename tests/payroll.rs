#![cfg(feature = "compute")]
use anyhow::Result;
use bad_econ::domain::*;

fn policy() -> PayrollPolicy {
    PayrollPolicy {
        base_rate: 100,
        employee_charge: 50,
        brackets: vec![TaxBracket {
            threshold: 100,
            marginal_rate: 300,
        }],
    }
}

#[cfg(feature = "compute")]
use super::support;

#[cfg(feature = "compute")]
fn scenario(minutes: u32) -> Result<Scenario> {
    let mut config = Scenario::baseline(3, 1, Experiment::Concentrated)?
        .config()
        .clone();
    config.payroll = Some(policy());
    config.distribution_rate = 0;
    config.forecast_weight = 0;
    config.coverage = 30;
    config.behavior.work_fatigue = 0;
    config.behavior.wage_increment = 0;
    config.resources = ResourceMap::Explicit {
        regions: vec![ResourceRegion::new(
            Good::Grain,
            Position { x: 0, y: 0 },
            Position {
                x: config.map_size,
                y: config.map_size,
            },
            config.map_size,
        )?],
    };
    config.population = Some(Population {
        households: (1..=3)
            .map(|identity| HouseholdSeed {
                identity,
                position: Position { x: 0, y: 0 },
                cash: 0,
                inventories: Good::food([0; 3]),
                fatigue: 0,
            })
            .collect(),
        firms: (4..=6)
            .map(|identity| FirmSeed {
                identity,
                position: Position { x: 0, y: 0 },
                owner: 1,
                good: Good::Grain,
                cash: 100_000,
                inventories: Good::food([0; 3]),
                capacity: 1,
                technique: Technique {
                    labor_minutes: minutes,
                    input_ticks: 0,
                    output_ticks: 1,
                },
                wage: 1,
                price: 1,
            })
            .collect(),
    });
    Scenario::new(config)
}

#[cfg(feature = "compute")]
#[test]
fn untaxed_wages_remain_observed_without_implying_entrepreneurial_interest() -> Result<()> {
    use bad_econ::{domain::enterprise::HouseholdProduction, simulation::Simulation};
    let mut config = scenario(80)?.config().clone();
    config.payroll = None;
    config.entrepreneurship.interested_per_mille = 0;
    config.household_production = HouseholdProduction::industrial();
    for household in &mut config.population.as_mut().unwrap().households {
        household.cash = 20_000;
    }
    let mut untaxed = Simulation::new(Scenario::new(config.clone())?, support::device(64)?)?;
    config.payroll = Some(PayrollPolicy::default());
    let mut zero_rate = Simulation::new(Scenario::new(config)?, support::device(64)?)?;
    untaxed.advance()?;
    zero_rate.advance()?;
    let state = untaxed.download()?;
    assert_eq!(state, zero_rate.download()?);
    for household in 1..3 {
        assert_eq!(state.get(Field::GrossIncome, household), 80);
        assert_eq!(state.get(Field::Sales, household), 0);
        assert_eq!(state.get(Field::InvestmentGood, household), 0);
    }
    for _ in 0..3 {
        untaxed.advance()?;
        zero_rate.advance()?;
        assert_eq!(untaxed.download()?, zero_rate.download()?);
    }
    untaxed.checkpoint()?.restore()?;
    Ok(())
}

#[cfg(feature = "compute")]
#[test]
#[ignore = "extended scenario coverage; run cargo gpu-extended"]
fn payroll_specializations_and_runtime_rates_share_a_gpu_client() -> Result<()> {
    use bad_econ::simulation::Simulation;

    struct Case {
        policy: Option<PayrollPolicy>,
        minutes: u32,
        cash: u64,
        tax: u64,
        charge: u64,
    }

    let device = support::device(64)?;
    let cases = [
        Case {
            policy: None,
            minutes: 80,
            cash: 80,
            tax: 0,
            charge: 0,
        },
        Case {
            policy: Some(policy()),
            minutes: 80,
            cash: 68,
            tax: 8,
            charge: 4,
        },
        Case {
            policy: Some(PayrollPolicy {
                brackets: vec![
                    TaxBracket {
                        threshold: 100,
                        marginal_rate: 300,
                    },
                    TaxBracket {
                        threshold: 140,
                        marginal_rate: 500,
                    },
                ],
                ..policy()
            }),
            minutes: 160,
            cash: 120,
            tax: 32,
            charge: 8,
        },
        Case {
            policy: Some(PayrollPolicy {
                base_rate: 200,
                brackets: vec![TaxBracket {
                    threshold: 40,
                    marginal_rate: 400,
                }],
                ..policy()
            }),
            minutes: 80,
            cash: 52,
            tax: 24,
            charge: 4,
        },
        Case {
            policy: Some(PayrollPolicy::default()),
            minutes: 80,
            cash: 80,
            tax: 0,
            charge: 0,
        },
        Case {
            policy: None,
            minutes: 80,
            cash: 80,
            tax: 0,
            charge: 0,
        },
    ];
    for case in cases {
        let mut config = scenario(case.minutes)?.config().clone();
        config.payroll = case.policy;
        let mut simulation = Simulation::new(Scenario::new(config)?, device.clone())?;
        simulation.advance()?;
        let state = simulation.download()?;
        for household in 0..3 {
            assert_eq!(state.get(Field::Cash, household), case.cash);
            assert_eq!(
                state.get(Field::GrossIncome, household),
                u64::from(case.minutes)
            );
            assert_eq!(state.get(Field::TaxTreasury, household), case.tax);
            assert_eq!(state.get(Field::ChargeTreasury, household), case.charge);
            assert_eq!(state.get(Field::PayrollReserve, household), 0);
        }
    }
    Ok(())
}

#[cfg(feature = "compute")]
#[test]
fn payroll_is_atomic_device_accounting_and_survives_checkpoints() -> Result<()> {
    use bad_econ::{runtime::SimulationCommand, simulation::Simulation};
    let mut simulation = Simulation::new(scenario(80)?, support::device(64)?)?;
    let opening = simulation.snapshot.cash;
    simulation.advance()?;
    let state = simulation.download()?;
    for household in 0..3 {
        assert_eq!(state.get(Field::Cash, household), 68);
        assert_eq!(state.get(Field::GrossIncome, household), 80);
        assert_eq!(state.get(Field::TaxTreasury, household), 8);
        assert_eq!(state.get(Field::ChargeTreasury, household), 4);
        assert_eq!(state.get(Field::PayrollReserve, household), 0);
    }
    assert_eq!(simulation.snapshot.income_tax, 24);
    assert_eq!(simulation.snapshot.employee_charges, 12);
    assert_eq!(simulation.snapshot.cash, opening);
    let checkpoint = simulation.checkpoint()?;
    (0..4).try_for_each(|_| simulation.advance())?;
    let expected = simulation.download()?;
    for household in 0..3 {
        assert_eq!(expected.get(Field::GrossIncome, household), 400);
        assert_eq!(expected.get(Field::TaxTreasury, household), 100);
        assert_eq!(expected.get(Field::ChargeTreasury, household), 20);
        assert_eq!(expected.get(Field::Cash, household), 280);
    }
    simulation.command(SimulationCommand::Restore(checkpoint.into()))?;
    (0..4).try_for_each(|_| simulation.advance())?;
    assert_eq!(simulation.download()?, expected);
    assert_eq!(simulation.snapshot.cash, opening);
    Ok(())
}

#[cfg(feature = "compute")]
#[test]
fn fractional_payroll_claims_remain_funded_across_small_payments() -> Result<()> {
    use bad_econ::simulation::Simulation;
    let mut simulation = Simulation::new(scenario(1)?, support::device(64)?)?;
    for gross in 1..=20u64 {
        simulation.advance()?;
        let state = simulation.download()?;
        for household in 0..3 {
            let net = gross * 850 / 1000;
            let tax = gross * 100 / 1000;
            let charge = gross * 50 / 1000;
            let reserve = gross - net - tax - charge;
            assert_eq!(state.get(Field::GrossIncome, household), gross);
            assert_eq!(state.get(Field::Cash, household), net);
            assert_eq!(state.get(Field::TaxTreasury, household), tax);
            assert_eq!(state.get(Field::ChargeTreasury, household), charge);
            assert_eq!(state.get(Field::PayrollReserve, household), reserve);
        }
        assert_eq!(
            simulation.snapshot,
            ObservationSnapshot::from_state(&simulation.scenario, &state)?
        );
    }
    Ok(())
}

#[cfg(feature = "compute")]
#[test]
#[ignore = "extended scenario coverage; run cargo gpu-extended"]
fn changing_employers_does_not_restart_tax_allowances() -> Result<()> {
    use bad_econ::simulation::Simulation;
    let mut config = scenario(80)?.config().clone();
    let firms = &mut config.population.as_mut().unwrap().firms;
    firms[0].cash = 80;
    firms[1].technique.labor_minutes = 31;
    firms[2].technique.labor_minutes = 31;
    let mut simulation = Simulation::new(Scenario::new(config)?, support::device(64)?)?;
    simulation.advance()?;
    let first = simulation.download()?;
    let worker = (0..3)
        .find(|&household| first.get(Field::Employer, household) == 4)
        .expect("the first firm paid a worker");
    simulation.advance()?;
    for _ in 0..6 {
        if simulation.download()?.get(Field::Employer, worker) == 0 {
            simulation.advance()?;
        }
    }
    let state = simulation.download()?;
    assert!((5..=6).contains(&state.get(Field::Employer, worker)));
    assert_eq!(state.get(Field::GrossIncome, worker), 111);
    assert_eq!(state.get(Field::Cash, worker), 92);
    assert_eq!(state.get(Field::TaxTreasury, worker), 13);
    assert_eq!(state.get(Field::ChargeTreasury, worker), 5);
    assert_eq!(state.get(Field::PayrollReserve, worker), 1);
    Ok(())
}

#[cfg(feature = "compute")]
#[test]
#[ignore = "extended scenario coverage; run cargo gpu-extended"]
fn household_fatigue_uses_executed_work() -> Result<()> {
    use bad_econ::simulation::Simulation;
    let mut config = scenario(80)?.config().clone();
    config.household_need = 2;
    config.household_policy = Some(HouseholdPolicy {
        fatigue_response: 100,
        rest_minutes: 480,
    });
    for household in &mut config.population.as_mut().unwrap().households {
        household.fatigue = 200;
        household.inventories[2] = 1;
    }
    let mut simulation = Simulation::new(Scenario::new(config)?, support::device(64)?)?;
    simulation.advance()?;
    let state = simulation.download()?;
    for household in 0..3 {
        assert_eq!(state.get(Field::Hours, household), 80);
        assert_eq!(state.get(Field::Shortage, household), 1);
        assert_eq!(state.get(Field::Fatigue, household), 116);
    }
    Ok(())
}

#[cfg(feature = "compute")]
fn bargaining_policy() -> bargaining::BargainingPolicy {
    bargaining::BargainingPolicy {
        legal_minimum: 1,
        quantum: 1,
        daily_cash_cost: 80,
        funded_other: 0,
        participating_adults: 1,
        normal_day_minutes: 80,
        work_cost_per_minute: 0,
        desired_wage: 0,
        mandate: 1000,
        withdrawal: 1000,
        alternatives: 1000,
        concentration: 0,
        replacements: 0,
    }
}

#[cfg(feature = "compute")]
#[test]
fn negotiated_wages_set_real_escrow_and_payroll_amounts() -> Result<()> {
    use bad_econ::simulation::Simulation;
    let mut config = scenario(80)?.config().clone();
    config.bargaining = Some(bargaining::BargainingPolicy {
        desired_wage: 5,
        ..bargaining_policy()
    });
    let mut simulation = Simulation::new(Scenario::new(config)?, support::device(64)?)?;
    simulation.advance()?;
    let state = simulation.download()?;
    for household in 0..3 {
        assert_eq!(state.get(Field::GrossIncome, household), 320);
        assert_eq!(state.get(Field::Cash, household), 228);
        assert_eq!(state.get(Field::TaxTreasury, household), 76);
        assert_eq!(state.get(Field::ChargeTreasury, household), 16);
    }
    Ok(())
}

#[cfg(feature = "compute")]
#[test]
#[ignore = "extended scenario coverage; run cargo gpu-extended"]
fn reservation_pay_is_not_rounded_before_payable_minutes() -> Result<()> {
    use bad_econ::simulation::Simulation;
    let mut config = scenario(80)?.config().clone();
    config.bargaining = Some(bargaining_policy());
    let mut simulation = Simulation::new(Scenario::new(config)?, support::device(64)?)?;
    simulation.advance()?;
    let state = simulation.download()?;
    for household in 0..3 {
        assert_eq!(state.get(Field::GrossIncome, household), 80);
        assert_eq!(state.get(Field::Cash, household), 68);
    }
    Ok(())
}

#[cfg(feature = "compute")]
#[test]
#[ignore = "extended scenario coverage; run cargo gpu-extended"]
fn empty_funded_wage_intervals_prevent_labor_commitments() -> Result<()> {
    use bad_econ::simulation::Simulation;
    let mut config = scenario(80)?.config().clone();
    config.bargaining = Some(bargaining::BargainingPolicy {
        legal_minimum: 2,
        ..bargaining_policy()
    });
    for firm in &mut config.population.as_mut().unwrap().firms {
        firm.cash = 80;
    }
    let mut simulation = Simulation::new(Scenario::new(config)?, support::device(64)?)?;
    simulation.advance()?;
    assert_eq!(simulation.snapshot.employed, 0);
    assert_eq!(simulation.snapshot.produced, Good::food([0; 3]));
    Ok(())
}

#[cfg(feature = "compute")]
#[test]
#[ignore = "extended scenario coverage; run cargo gpu-extended"]
fn bargaining_plans_affordable_slots_below_the_posted_offer() -> Result<()> {
    use bad_econ::simulation::Simulation;
    let mut config = scenario(80)?.config().clone();
    config.bargaining = Some(bargaining_policy());
    let firms = &mut config.population.as_mut().unwrap().firms;
    firms[0].cash = 240;
    firms[0].capacity = 3;
    firms[0].wage = 3;
    firms[1].cash = 0;
    firms[2].cash = 0;
    let mut simulation = Simulation::new(Scenario::new(config)?, support::device(64)?)?;
    simulation.advance()?;
    let state = simulation.download()?;
    assert_eq!(simulation.snapshot.employed, 3);
    assert_eq!(state.get(Field::Plan, 3), 3);
    for household in 0..3 {
        assert_eq!(state.get(Field::GrossIncome, household), 80);
    }
    Ok(())
}

#[cfg(feature = "compute")]
#[test]
#[ignore = "extended scenario coverage; run cargo gpu-extended"]
fn later_matching_rounds_fund_only_remaining_negotiable_slots() -> Result<()> {
    use bad_econ::simulation::Simulation;
    let mut config = scenario(80)?.config().clone();
    config.bargaining = Some(bargaining_policy());
    config.search_radius = 100;
    let population = config.population.as_mut().unwrap();
    population.households[2].position.x = 200;
    population.firms[0].cash = 160;
    population.firms[0].wage = 2;
    population.firms[1].cash = 160;
    population.firms[1].capacity = 2;
    population.firms[1].position.x = 100;
    population.firms[2].cash = 0;
    let mut simulation = Simulation::new(Scenario::new(config)?, support::device(64)?)?;
    simulation.advance()?;
    let state = simulation.download()?;
    assert_eq!(simulation.snapshot.employed, 3);
    assert_eq!(state.get(Field::GrossIncome, 0), 160);
    assert_eq!(state.get(Field::GrossIncome, 1), 80);
    assert_eq!(state.get(Field::GrossIncome, 2), 80);
    assert_eq!(state.get(Field::Slots, 4), 0);
    Ok(())
}

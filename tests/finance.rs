#![cfg(feature = "compute")]
use super::support;
use anyhow::Result;
use bad_econ::{domain::finance::*, ledger::Coordinate, simulation::finance::*};

fn actors(count: usize) -> Vec<FinancialActor> {
    (0..count)
        .map(|index| FinancialActor {
            identity: index as u64 + 1,
            cash: Coordinate(index as u32),
            kind: FinancialActorKind::Household,
        })
        .collect()
}

fn words(values: &[u64]) -> Vec<u32> {
    values
        .iter()
        .flat_map(|&value| [value as u32, (value >> 32) as u32])
        .collect()
}

fn ticks(values: Vec<u32>) -> Vec<u64> {
    values
        .chunks_exact(2)
        .map(|value| u64::from(value[0]) | (u64::from(value[1]) << 32))
        .collect()
}

fn bank() -> BankSeed {
    BankSeed {
        identity: 10,
        reserves: 100,
        minimum_capital: 100,
        risk_weight: 1000,
    }
}

fn loan(identity: u64, borrower: u64, requested: u64) -> LoanContract {
    LoanContract {
        identity,
        bank: 10,
        borrower,
        requested,
        target_cash: 60,
        first_day: 0,
        term: 2,
        daily_rate: 100,
        surplus: 100,
        debt_service: 0,
        service_fraction: 1000,
        advance: 1000,
        collateral: 1000,
        secured: 0,
        consent: Consent::Granted,
    }
}

#[test]
fn funded_loan_creation_accrual_repayment_and_balance_sheet_run_on_device() -> Result<()> {
    let device = support::device(64)?;
    let config = FinanceConfig::new(
        FinanceConfig {
            banks: vec![bank()],
            loans: vec![loan(1, 1, 40)],
            support: None,
        },
        &[1],
        &[1],
    )?;
    let layout = config.layout(1)?;
    let runtime = FinanceRuntime::new(&device, &config, layout, &actors(1))?;
    let initial = std::iter::once(20)
        .chain(config.initial_values())
        .collect::<Vec<_>>();
    let state = device.tensor(&words(&initial));
    runtime.phase(&device, &state, 0, FinanceStage::Lending)?;
    let issued = ticks(device.read(&state)?);
    assert_eq!(issued[0], 60);
    assert_eq!(issued[layout.banks as usize], 60);
    assert_eq!(issued[layout.loans as usize], 40);
    assert_eq!(
        FinanceStateView::new(&config, layout, &issued, 0)?.cash + issued[0],
        120
    );
    runtime.phase(&device, &state, 1, FinanceStage::Opening)?;
    let day_one = runtime.observe(&device, &state, 1)?;
    assert_eq!(day_one.principal, 40);
    assert_eq!(day_one.interest_due, 4);
    runtime.phase(&device, &state, 1, FinanceStage::Opening)?;
    assert_eq!(runtime.observe(&device, &state, 1)?, day_one);
    runtime.phase(&device, &state, 2, FinanceStage::Opening)?;
    let settled = ticks(device.read(&state)?);
    assert_eq!(settled[0], 12);
    assert_eq!(settled[layout.banks as usize], 108);
    assert_eq!(settled[layout.bank(BankField::Equity, 0)], 108);
    assert_eq!(
        settled[layout.loan(LoanField::Status, 0)],
        LoanStatus::Repaid as u64
    );
    assert_eq!(runtime.observe(&device, &state, 2)?.principal, 0);
    assert_eq!(
        FinanceStateView::new(&config, layout, &settled, 2)?.cash + settled[0],
        120
    );
    Ok(())
}

#[test]
fn support_rounding_preserves_treasury_and_individual_bounds_at_u64_width() -> Result<()> {
    let device = support::device(64)?;
    let needs = [1, (1_u64 << 62) - 1, 1_u64 << 62, (1_u64 << 63) - 1];
    let capacity = u64::MAX - 17;
    let config = FinanceConfig::new(
        FinanceConfig {
            banks: vec![],
            loans: vec![],
            support: Some(SupportPolicy {
                treasury: capacity,
                recipients: needs
                    .into_iter()
                    .enumerate()
                    .map(|(index, target_cash)| SupportRecipient {
                        household: index as u64 + 1,
                        target_cash,
                        consent: Consent::Granted,
                    })
                    .collect(),
            }),
        },
        &[1, 2, 3, 4],
        &[1, 2, 3, 4],
    )?;
    let layout = config.layout(4)?;
    let runtime = FinanceRuntime::new(&device, &config, layout, &actors(4))?;
    let state = device.tensor(&words(&[0, 0, 0, 0, capacity]));
    runtime.support(&device, &state)?;
    let result = ticks(device.read(&state)?);
    let total = needs.iter().map(|&need| u128::from(need)).sum::<u128>();
    let mut prefix = 0_u128;
    let mut paid = 0_u128;
    for (index, need) in needs.into_iter().enumerate() {
        prefix += u128::from(need);
        let due = u128::from(capacity) * prefix / total;
        assert_eq!(u128::from(result[index]), due - paid);
        assert!(result[index] <= need);
        paid = due;
    }
    assert_eq!(paid, u128::from(capacity));
    assert_eq!(result[4], 0);
    Ok(())
}

#[test]
fn lean_cross_bank_payments_conserve_deposits_and_reserves_separately() -> Result<()> {
    let device = support::device(64)?;
    let payer = BankAccount {
        deposit: Coordinate(0),
        reserve: Coordinate(1),
    };
    let payee = BankAccount {
        deposit: Coordinate(3),
        reserve: Coordinate(2),
    };
    let result = bank_payments(
        &device,
        &[12, 7, 3, 1],
        &[BankPayment {
            identity: 1,
            payer,
            payee,
            amount: 5,
        }],
    )?;
    assert_eq!(result.state, [7, 2, 8, 6]);
    assert_eq!(result.decisions, [bad_econ::ledger::Decision::Accepted]);
    let result = bank_payments(
        &device,
        &[12, 2, 3, 1],
        &[
            BankPayment {
                identity: 1,
                payer,
                payee,
                amount: 5,
            },
            BankPayment {
                identity: 2,
                payer,
                payee,
                amount: 2,
            },
        ],
    )?;
    assert_eq!(result.state, [10, 0, 5, 3]);
    assert_eq!(
        result.decisions,
        [
            bad_econ::ledger::Decision::Rejected,
            bad_econ::ledger::Decision::Accepted
        ]
    );
    let same_bank = BankAccount {
        deposit: Coordinate(3),
        reserve: Coordinate(1),
    };
    let result = bank_payments(
        &device,
        &[12, 0, 3, 1],
        &[BankPayment {
            identity: 1,
            payer,
            payee: same_bank,
            amount: 5,
        }],
    )?;
    assert_eq!(result.state, [7, 0, 3, 6]);
    Ok(())
}

#[test]
fn loan_arithmetic_fault_restores_the_simulations_entire_audited_boundary() -> Result<()> {
    use bad_econ::{
        domain::{Experiment, Scenario},
        runtime::RuntimeState,
        simulation::Simulation,
    };
    let mut scenario = Scenario::baseline(3, 42, Experiment::Baseline)?
        .config()
        .clone();
    let mut contract = loan(1, 4, 1);
    contract.surplus = u64::MAX;
    contract.target_cash = u64::MAX;
    scenario.finance = Some(FinanceConfig {
        banks: vec![bank()],
        loans: vec![contract],
        support: None,
    });
    let mut simulation = Simulation::new(Scenario::new(scenario)?, support::device(64)?)?;
    let audited = simulation.download()?;
    assert!(simulation.advance().is_err());
    assert!(matches!(
        simulation.status,
        RuntimeState::Fault {
            last_audited_day: 0,
            ..
        }
    ));
    assert_eq!(simulation.download()?, audited);
    assert_eq!(simulation.checkpoint()?.restore()?.state, audited);
    Ok(())
}

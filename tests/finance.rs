#![cfg(feature = "compute")]
use crate::support;
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
fn unfunded_maturity_retains_claim_and_other_borrowers_release_no_resources() -> Result<()> {
    let device = support::device(64)?;
    let mut first = loan(1, 1, 80);
    first.target_cash = 80;
    let mut second = loan(2, 2, 80);
    second.target_cash = 80;
    let config = FinanceConfig::new(
        FinanceConfig {
            banks: vec![bank()],
            loans: vec![first, second],
            support: None,
        },
        &[1, 2],
        &[1, 2],
    )?;
    let layout = config.layout(2)?;
    let runtime = FinanceRuntime::new(&device, &config, layout, &actors(2))?;
    let initial = [0, 0]
        .into_iter()
        .chain(config.initial_values())
        .collect::<Vec<_>>();
    let state = device.tensor(&words(&initial));
    runtime.phase(&device, &state, 0, FinanceStage::Lending)?;
    let issued = ticks(device.read(&state)?);
    assert_eq!(&issued[..2], &[80, 20]);
    runtime.phase(&device, &state, 2, FinanceStage::Opening)?;
    let overdue = runtime.observe(&device, &state, 2)?;
    assert_eq!(overdue.principal, 100);
    assert_eq!(overdue.overdue_loans, 2);
    assert_eq!(overdue.interest_due, 20);
    assert_eq!(&ticks(device.read(&state)?)[..2], &[80, 20]);
    Ok(())
}

#[test]
fn lean_proportional_household_support_uses_parallel_prefix_allocation() -> Result<()> {
    let device = support::device(64)?;
    let config = FinanceConfig::new(
        FinanceConfig {
            banks: vec![],
            loans: vec![],
            support: Some(SupportPolicy {
                treasury: 6,
                recipients: [2, 4, 6]
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
        &[1, 2, 3],
        &[1, 2, 3],
    )?;
    let layout = config.layout(3)?;
    let runtime = FinanceRuntime::new(&device, &config, layout, &actors(3))?;
    let state = device.tensor(&words(&[0, 0, 0, 6]));
    runtime.support(&device, &state)?;
    assert_eq!(ticks(device.read(&state)?), [1, 2, 3, 0]);
    assert_eq!(runtime.last_event_count(), 3);
    assert_eq!(
        runtime.support_decisions(),
        vec![bad_econ::ledger::Decision::Accepted; 3]
    );
    runtime.support(&device, &state)?;
    assert_eq!(runtime.last_event_count(), 0);
    assert!(runtime.support_decisions().is_empty());
    assert_eq!(runtime.observe(&device, &state, 0)?.support_treasury, 0);
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
fn lean_dated_funding_and_divisible_pool_examples_run_in_shared_kernels() -> Result<()> {
    let device = support::device(64)?;
    let plans = [0, 3]
        .into_iter()
        .map(|cushion| FundingPlan {
            opening_cash: 5,
            opening_cushion: cushion,
            flows: vec![
                CashFlow {
                    day: 1,
                    outflow: 8,
                    receipt: 0,
                    cushion,
                },
                CashFlow {
                    day: 2,
                    outflow: 0,
                    receipt: 10,
                    cushion,
                },
            ],
        })
        .collect::<Vec<_>>();
    assert_eq!(
        FinancePolicies::funding(&device, &plans)?,
        [
            FundingResult {
                opening_need: 8,
                gap: 3
            },
            FundingResult {
                opening_need: 11,
                gap: 6
            }
        ]
    );
    assert_eq!(
        FinancePolicies::pools(
            &device,
            &[
                DivisiblePool {
                    capacity: 5,
                    demands: vec![2, 4, 1]
                },
                DivisiblePool {
                    capacity: 5,
                    demands: vec![6, 4]
                }
            ]
        )?,
        [
            PoolResult {
                fills: vec![2, 3, 0],
                remaining: 0
            },
            PoolResult {
                fills: vec![5, 0],
                remaining: 0
            },
        ]
    );
    Ok(())
}

#[test]
fn lending_caps_reject_invalid_domains_and_respect_each_constraint() -> Result<()> {
    let device = support::device(64)?;
    let input = LendingInput {
        requested: 100,
        term: 10,
        rate: 0,
        surplus: 100,
        debt_service: 0,
        service_fraction: 500,
        advance: 500,
        collateral: 1000,
        secured: 0,
        equity: 100,
        minimum_capital: 100,
        risk_assets: 0,
        risk_weight: 1000,
        liquidity: 1000,
    };
    let mut liquidity = input.clone();
    liquidity.liquidity = 7;
    let mut collateral = input.clone();
    collateral.secured = 490;
    let mut service = input.clone();
    service.debt_service = 48;
    let mut capital = input.clone();
    capital.risk_assets = 980;
    assert_eq!(
        FinancePolicies::lending(
            &device,
            &[input.clone(), liquidity, collateral, service, capital]
        )?
        .into_iter()
        .map(|result| result.amount)
        .collect::<Vec<_>>(),
        [100, 7, 10, 20, 20]
    );
    let mut invalid = input;
    invalid.minimum_capital = 0;
    assert!(FinancePolicies::lending(&device, &[invalid]).is_err());
    Ok(())
}

#[test]
fn repayment_retains_fractional_interest_without_accruing_after_settlement() -> Result<()> {
    let device = support::device(64)?;
    let mut contract = loan(1, 1, 1);
    contract.term = 1;
    contract.daily_rate = 333;
    contract.target_cash = 2;
    let config = FinanceConfig::new(
        FinanceConfig {
            banks: vec![bank()],
            loans: vec![contract],
            support: None,
        },
        &[1],
        &[1],
    )?;
    let layout = config.layout(1)?;
    let runtime = FinanceRuntime::new(&device, &config, layout, &actors(1))?;
    let initial = std::iter::once(1)
        .chain(config.initial_values())
        .collect::<Vec<_>>();
    let state = device.tensor(&words(&initial));
    runtime.phase(&device, &state, 0, FinanceStage::Lending)?;
    assert_eq!(
        runtime.decisions(&device)?,
        [bad_econ::ledger::Decision::Accepted]
    );
    runtime.phase(&device, &state, 1, FinanceStage::Opening)?;
    assert_eq!(
        runtime.decisions(&device)?,
        [bad_econ::ledger::Decision::Accepted]
    );
    let settled = ticks(device.read(&state)?);
    assert_eq!(settled[layout.loan(LoanField::InterestRemainder, 0)], 333);
    assert_eq!(
        settled[layout.loan(LoanField::Status, 0)],
        LoanStatus::Repaid as u64
    );
    runtime.phase(&device, &state, 9, FinanceStage::Opening)?;
    assert_eq!(ticks(device.read(&state)?), settled);
    assert!(runtime.decisions(&device)?.is_empty());
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
fn replacement_rent_and_discount_bounds_use_device_arithmetic() -> Result<()> {
    let device = support::device(64)?;
    assert_eq!(
        FinancePolicies::budgets(
            &device,
            &[BudgetPolicy {
                replacement_target: 10,
                earmarked: 3,
                free_cash: 4,
                rent_fraction: 500,
                before_rent: 10,
                retained: 2,
                rent_budget: 6,
            }]
        )?,
        [BudgetResult {
            earmark: 4,
            rent: 3
        }]
    );
    let exact = FinancePolicies::discounted_flows(
        &device,
        &[DiscountPlan {
            daily_rate: 1000,
            flows: vec![
                DiscountFlow {
                    receipt: 4,
                    outflow: 0,
                },
                DiscountFlow {
                    receipt: 4,
                    outflow: 0,
                },
                DiscountFlow {
                    receipt: 0,
                    outflow: 4,
                },
            ],
        }],
    )?;
    assert_eq!(
        exact,
        [DiscountedValue {
            lower: SignedAmount::Positive(5),
            upper: SignedAmount::Positive(5)
        }]
    );
    let inexact = FinancePolicies::discounted_flows(
        &device,
        &[DiscountPlan {
            daily_rate: 100,
            flows: vec![
                DiscountFlow {
                    receipt: 0,
                    outflow: 0,
                },
                DiscountFlow {
                    receipt: 1,
                    outflow: 0,
                },
            ],
        }],
    )?;
    assert_eq!(
        inexact,
        [DiscountedValue {
            lower: SignedAmount::Zero,
            upper: SignedAmount::Positive(1)
        }]
    );
    Ok(())
}

#[test]
fn withheld_consent_empty_institutions_and_empty_support_do_not_move_cash() -> Result<()> {
    let device = support::device(64)?;
    let mut contract = loan(1, 1, 40);
    contract.consent = Consent::Withheld;
    let config = FinanceConfig::new(
        FinanceConfig {
            banks: vec![bank()],
            loans: vec![contract],
            support: Some(SupportPolicy {
                treasury: 6,
                recipients: vec![SupportRecipient {
                    household: 1,
                    target_cash: 100,
                    consent: Consent::Withheld,
                }],
            }),
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
    assert!(runtime.decisions(&device)?.is_empty());
    assert_eq!(runtime.last_event_count(), 0);
    runtime.support(&device, &state)?;
    assert_eq!(ticks(device.read(&state)?), initial);
    for support in [
        None,
        Some(SupportPolicy {
            treasury: 6,
            recipients: vec![],
        }),
    ] {
        let config = FinanceConfig::new(
            FinanceConfig {
                banks: vec![],
                loans: vec![],
                support,
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
        runtime.phase(&device, &state, 0, FinanceStage::Opening)?;
        runtime.support(&device, &state)?;
        assert_eq!(ticks(device.read(&state)?), initial);
        assert_eq!(runtime.observe(&device, &state, 0)?.principal, 0);
    }
    Ok(())
}

#[test]
fn financial_admission_rejects_duplicate_borrowers_and_unknown_recipients() -> Result<()> {
    let duplicated = FinanceConfig {
        banks: vec![bank()],
        loans: vec![loan(1, 1, 1), loan(2, 1, 1)],
        support: None,
    };
    assert!(FinanceConfig::new(duplicated, &[1], &[1]).is_err());
    let mut invalid_bank = bank();
    invalid_bank.minimum_capital = 0;
    assert!(
        FinanceConfig::new(
            FinanceConfig {
                banks: vec![invalid_bank],
                loans: vec![],
                support: None
            },
            &[1],
            &[1]
        )
        .is_err()
    );
    let support = SupportPolicy {
        treasury: 6,
        recipients: vec![SupportRecipient {
            household: 2,
            target_cash: 10,
            consent: Consent::Granted,
        }],
    };
    assert!(
        FinanceConfig::new(
            FinanceConfig {
                banks: vec![],
                loans: vec![],
                support: Some(support)
            },
            &[1, 2],
            &[1]
        )
        .is_err()
    );
    let mut invalid_loan = loan(1, 1, 1);
    invalid_loan.bank = 11;
    assert!(
        FinanceConfig::new(
            FinanceConfig {
                banks: vec![bank()],
                loans: vec![invalid_loan],
                support: None
            },
            &[1],
            &[1]
        )
        .is_err()
    );
    Ok(())
}

#[test]
fn support_overflow_faults_before_any_transfer_is_posted() -> Result<()> {
    let device = support::device(64)?;
    let config = FinanceConfig::new(
        FinanceConfig {
            banks: vec![],
            loans: vec![],
            support: Some(SupportPolicy {
                treasury: 1,
                recipients: [1, 2]
                    .into_iter()
                    .map(|household| SupportRecipient {
                        household,
                        target_cash: u64::MAX,
                        consent: Consent::Granted,
                    })
                    .collect(),
            }),
        },
        &[1, 2],
        &[1, 2],
    )?;
    let runtime = FinanceRuntime::new(&device, &config, config.layout(2)?, &actors(2))?;
    let state = device.tensor(&words(&[0, 0, 1]));
    assert!(runtime.support(&device, &state).is_err());
    assert_eq!(ticks(device.read(&state)?), [0, 0, 1]);
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

#[test]
fn financial_audit_rejects_forged_risk_assets_even_when_bank_totals_match() -> Result<()> {
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
    let state = device.tensor(&words(
        &std::iter::once(20)
            .chain(config.initial_values())
            .collect::<Vec<_>>(),
    ));
    runtime.phase(&device, &state, 0, FinanceStage::Lending)?;
    let mut forged = ticks(device.read(&state)?);
    forged[layout.bank(BankField::RiskAssets, 0)] = 1;
    forged[layout.loan(LoanField::RiskAssets, 0)] = 1;
    assert!(FinanceStateView::new(&config, layout, &forged, 0).is_err());
    device.write(&state, &words(&forged));
    assert!(runtime.observe(&device, &state, 0).is_err());
    Ok(())
}

#[test]
fn financial_rows_keep_distinct_banks_loans_and_support_recipients() -> Result<()> {
    let device = support::device(64)?;
    let config = FinanceConfig::new(
        FinanceConfig {
            banks: vec![
                BankSeed {
                    reserves: 1 << 40,
                    ..bank()
                },
                BankSeed {
                    identity: 20,
                    reserves: 1 << 41,
                    risk_weight: 500,
                    ..bank()
                },
            ],
            loans: (0..3)
                .map(|index| LoanContract {
                    bank: match index {
                        2 => 20,
                        _ => 10,
                    },
                    target_cash: 30 + index * 10,
                    daily_rate: 100 + index as u32 * 100,
                    ..loan(index + 1, index + 1, 30 + index * 10)
                })
                .collect(),
            support: Some(SupportPolicy {
                treasury: 11,
                recipients: vec![
                    SupportRecipient {
                        household: 1,
                        target_cash: 32,
                        consent: Consent::Granted,
                    },
                    SupportRecipient {
                        household: 2,
                        target_cash: 49,
                        consent: Consent::Granted,
                    },
                ],
            }),
        },
        &[1, 2, 3],
        &[1, 2, 3],
    )?;
    let layout = config.layout(3)?;
    let initial = [0, 0, 0]
        .into_iter()
        .chain(config.initial_values())
        .collect::<Vec<_>>();
    assert_eq!(&initial[3..9], &[1 << 40, 1 << 41, 1 << 40, 1 << 41, 0, 0]);
    let state = device.tensor(&words(&initial));
    let runtime = FinanceRuntime::new(&device, &config, layout, &actors(3))?;
    runtime.phase(&device, &state, 0, FinanceStage::Lending)?;
    runtime.support(&device, &state)?;
    runtime.phase(&device, &state, 1, FinanceStage::Opening)?;
    let actual = ticks(device.read(&state)?);
    assert_eq!(&actual[..3], &[32, 49, 50]);
    assert_eq!(actual[layout.bank(BankField::Reserve, 0)], (1 << 40) - 70);
    assert_eq!(actual[layout.bank(BankField::Reserve, 1)], (1 << 41) - 50);
    assert_eq!(actual[layout.bank(BankField::RiskAssets, 0)], 70);
    assert_eq!(actual[layout.bank(BankField::RiskAssets, 1)], 25);
    for index in 0..3 {
        assert_eq!(
            actual[layout.loan(LoanField::Principal, index)],
            30 + index as u64 * 10
        );
        assert_eq!(
            actual[layout.loan(LoanField::Interest, index)],
            [3, 8, 15][index]
        );
        assert_eq!(
            actual[layout.loan(LoanField::Status, index)],
            LoanStatus::Active as u64
        );
    }
    let observed = runtime.observe(&device, &state, 1)?;
    let audited = FinanceStateView::new(&config, layout, &actual, 1)?;
    assert_eq!(observed.principal, 120);
    assert_eq!(observed.interest_due, 26);
    assert_eq!(observed.active_loans, 3);
    assert_eq!(observed.cash()?, audited.cash);
    assert_eq!(observed.principal, audited.principal);
    assert_eq!(observed.interest_due, audited.interest_due);
    Ok(())
}

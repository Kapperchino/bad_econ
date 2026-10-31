use super::*;
use crate::simulation::{
    flags::KernelFaults,
    layout::{HIGH_WORD, LOW_WORD, TICK_BITS, WORD_BITS, WORD_TOP_BIT},
};

#[cube]
fn read(words: &[u32], base: usize) -> Ticks {
    ticks(words[base], words[base + 1])
}

#[cube]
fn add(a: Ticks, b: Ticks, faults: &mut KernelFaults) -> Ticks {
    let result = plus(a, b);
    faults.insert(result.faults);
    result.value
}

#[cube]
fn multiply(a: Ticks, b: u32, faults: &mut KernelFaults) -> Ticks {
    let result = times(a, b);
    faults.insert(result.faults);
    result.value
}

#[cube]
fn scaled(a: Ticks, n: u32, d: u32, faults: &mut KernelFaults) -> Ticks {
    let result = ratio(a, n, d);
    faults.insert(result.faults);
    result.value
}

#[cube]
fn ceil_scaled(a: Ticks, n: u32, d: u32, faults: &mut KernelFaults) -> Ticks {
    let result = scaled(a, n, d, faults);
    let quotient = divide(a, d);
    let remainder = minus(a, times(quotient, d).value);
    let extra = times32(remainder.extract(LOW_WORD), n);
    let fraction = minus(extra, times(divide(extra, d), d).value);
    add(
        result,
        ticks(u32::cast_from(!equal(fraction, zero())), 0),
        faults,
    )
}

#[cube]
pub(super) fn lending_limit(
    requested: Ticks,
    term: u32,
    rate: u32,
    surplus: Ticks,
    debt_service: Ticks,
    service_fraction: u32,
    advance: u32,
    collateral: Ticks,
    secured: Ticks,
    equity: Ticks,
    minimum_capital: u32,
    risk_assets: Ticks,
    risk_weight: u32,
    liquidity: Ticks,
    faults: &mut KernelFaults,
) -> Ticks {
    let service = positive(
        multiply(surplus, service_fraction, faults),
        multiply(debt_service, FINANCE_SCALE, faults),
    );
    let service_cap = scaled(service, term, FINANCE_SCALE + rate * term, faults);
    let collateral_cap = divide(
        positive(
            multiply(collateral, advance, faults),
            multiply(secured, FINANCE_SCALE, faults),
        ),
        FINANCE_SCALE,
    );
    let total_risk_capacity = scaled(equity, FINANCE_SCALE, minimum_capital, faults);
    let capital_cap = scaled(
        positive(total_risk_capacity, risk_assets),
        FINANCE_SCALE,
        risk_weight,
        faults,
    );
    minimum(
        minimum(
            minimum(minimum(requested, service_cap), collateral_cap),
            capital_cap,
        ),
        liquidity,
    )
}

#[cube]
fn issue(
    state: &mut [u32],
    banks: &[u32],
    loans: &[u32],
    bank: usize,
    loan: usize,
    day: u32,
    control: &[Atomic<u32>],
) -> u32 {
    let mut decision = 0u32;
    let b = banks[bank + BankMeta::STATE];
    let l = loans[loan + LoanMeta::STATE];
    let cash_coordinate = loans[loan + LoanMeta::CASH];
    let status = load(state, l + LoanField::Status as u32);
    let cash = load(state, cash_coordinate);
    let requested = minimum(
        read(loans, loan + LoanMeta::REQUESTED),
        positive(read(loans, loan + LoanMeta::TARGET_CASH), cash),
    );
    let eligible = equal(status, ticks(LoanStatus::Dormant as u32, 0))
        && day >= loans[loan + LoanMeta::FIRST_DAY]
        && loans[loan + LoanMeta::CONSENT] == 1
        && !equal(requested, zero());
    if eligible {
        decision = 2u32;
        let mut faults = KernelFaults::none();
        let reserve = load(state, b + BankField::Reserve as u32);
        let risk = load(state, b + BankField::RiskAssets as u32);
        let amount = lending_limit(
            requested,
            loans[loan + LoanMeta::TERM],
            loans[loan + LoanMeta::DAILY_RATE],
            read(loans, loan + LoanMeta::SURPLUS),
            read(loans, loan + LoanMeta::DEBT_SERVICE),
            loans[loan + LoanMeta::SERVICE_FRACTION],
            loans[loan + LoanMeta::ADVANCE],
            read(loans, loan + LoanMeta::COLLATERAL),
            read(loans, loan + LoanMeta::SECURED),
            load(state, b + BankField::Equity as u32),
            banks[bank + BankMeta::MINIMUM_CAPITAL],
            risk,
            banks[bank + BankMeta::RISK_WEIGHT],
            reserve,
            &mut faults,
        );
        let risk_debit = ceil_scaled(
            amount,
            banks[bank + BankMeta::RISK_WEIGHT],
            FINANCE_SCALE,
            &mut faults,
        );
        let next_risk = add(risk, risk_debit, &mut faults);
        let next_cash = add(cash, amount, &mut faults);
        let due = day + loans[loan + LoanMeta::TERM];
        if due < day {
            faults.insert(KernelFaults::new(comptime!(
                FaultFlags::ARITHMETIC_OVERFLOW
            )));
        }
        if faults.bits() == 0 && !equal(amount, zero()) {
            decision = 1u32;
            store(state, b + BankField::Reserve as u32, minus(reserve, amount));
            store(state, b + BankField::RiskAssets as u32, next_risk);
            store(state, cash_coordinate, next_cash);
            store(state, l + LoanField::Principal as u32, amount);
            store(state, l + LoanField::Due as u32, ticks(due, 0));
            store(
                state,
                l + LoanField::Status as u32,
                ticks(LoanStatus::Active as u32, 0),
            );
            store(state, l + LoanField::RiskAssets as u32, risk_debit);
            store(state, l + LoanField::LastAccrual as u32, ticks(day, 0));
        }
        faults.record(control);
    }
    if decision != 0 {
        control[comptime!(crate::simulation::flags::CONSTRUCTED_EVENTS_WORD)].fetch_add(1);
    }
    decision
}

#[cube]
fn opening(
    state: &mut [u32],
    banks: &[u32],
    loans: &[u32],
    bank: usize,
    loan: usize,
    day: u32,
    control: &[Atomic<u32>],
) -> u32 {
    let mut decision = 0u32;
    let b = banks[bank + BankMeta::STATE];
    let l = loans[loan + LoanMeta::STATE];
    let status = load(state, l + LoanField::Status as u32).extract(LOW_WORD);
    if status == LoanStatus::Active as u32 || status == LoanStatus::Overdue as u32 {
        let mut faults = KernelFaults::none();
        let principal = load(state, l + LoanField::Principal as u32);
        let last = load(state, l + LoanField::LastAccrual as u32).extract(LOW_WORD);
        let elapsed = day - min(day, last);
        let daily = scaled(
            principal,
            loans[loan + LoanMeta::DAILY_RATE],
            FINANCE_SCALE,
            &mut faults,
        );
        let remainder = minus(
            principal,
            times(divide(principal, FINANCE_SCALE), FINANCE_SCALE).value,
        );
        let daily_remainder =
            remainder.extract(LOW_WORD) * loans[loan + LoanMeta::DAILY_RATE] % FINANCE_SCALE;
        let accumulated = add(
            times32(daily_remainder, elapsed),
            load(state, l + LoanField::InterestRemainder as u32),
            &mut faults,
        );
        let carried = divide(accumulated, FINANCE_SCALE);
        let new_remainder = minus(accumulated, times(carried, FINANCE_SCALE).value);
        let interest = add(
            load(state, l + LoanField::Interest as u32),
            add(multiply(daily, elapsed, &mut faults), carried, &mut faults),
            &mut faults,
        );
        let total = add(principal, interest, &mut faults);
        let cash_coordinate = loans[loan + LoanMeta::CASH];
        let cash = load(state, cash_coordinate);
        let due = load(state, l + LoanField::Due as u32).extract(LOW_WORD);
        let repayment = day >= due && !lt(cash, total);
        let reserve = add(
            load(state, b + BankField::Reserve as u32),
            select(repayment, total, zero()),
            &mut faults,
        );
        let equity = add(
            load(state, b + BankField::Equity as u32),
            select(repayment, interest, zero()),
            &mut faults,
        );
        let risk = load(state, b + BankField::RiskAssets as u32);
        let loan_risk = load(state, l + LoanField::RiskAssets as u32);
        if last > day || lt(risk, loan_risk) {
            faults.insert(KernelFaults::new(comptime!(FaultFlags::INVALID_COORDINATE)));
        }
        if faults.bits() == 0 {
            store(state, l + LoanField::Interest as u32, interest);
            store(
                state,
                l + LoanField::InterestRemainder as u32,
                new_remainder,
            );
            store(state, l + LoanField::LastAccrual as u32, ticks(day, 0));
            if day >= due {
                decision = 2u32;
                store(
                    state,
                    l + LoanField::Status as u32,
                    ticks(LoanStatus::Overdue as u32, 0),
                );
            }
            if repayment {
                decision = 1u32;
                store(state, cash_coordinate, minus(cash, total));
                store(state, b + BankField::Reserve as u32, reserve);
                store(state, b + BankField::Equity as u32, equity);
                store(
                    state,
                    b + BankField::RiskAssets as u32,
                    minus(risk, loan_risk),
                );
                store(state, l + LoanField::Principal as u32, zero());
                store(state, l + LoanField::Interest as u32, zero());
                store(state, l + LoanField::RiskAssets as u32, zero());
                store(
                    state,
                    l + LoanField::Status as u32,
                    ticks(LoanStatus::Repaid as u32, 0),
                );
            }
        }
        faults.record(control);
    }
    if decision != 0 {
        control[comptime!(crate::simulation::flags::CONSTRUCTED_EVENTS_WORD)].fetch_add(1);
    }
    decision
}

#[cube(launch)]
pub(super) fn banks(
    state: &mut [u32],
    banks: &[u32],
    loans: &[u32],
    output: &mut [u32],
    decisions: &mut [u32],
    control: &[Atomic<u32>],
    day: u32,
    count: u32,
    #[comptime] stage: FinanceStage,
) {
    let id = ABSOLUTE_POS as u32;
    if id < count {
        let base = id as usize * BankMeta::WORDS;
        let first = banks[base + BankMeta::FIRST_LOAN];
        let last = first + banks[base + BankMeta::LOAN_COUNT];
        let mut principal = zero();
        let mut interest = zero();
        let mut risk = zero();
        let mut active = 0u32;
        let mut overdue = 0u32;
        let mut faults = KernelFaults::none();
        for index in first..last {
            let loan = index as usize * LoanMeta::WORDS;
            match stage {
                FinanceStage::Opening => {
                    decisions[index as usize] =
                        opening(state, banks, loans, base, loan, day, control);
                }
                FinanceStage::Lending => {
                    decisions[index as usize] =
                        issue(state, banks, loans, base, loan, day, control);
                }
                FinanceStage::Audit => {
                    let state_base = loans[loan + LoanMeta::STATE];
                    principal = add(
                        principal,
                        load(state, state_base + LoanField::Principal as u32),
                        &mut faults,
                    );
                    interest = add(
                        interest,
                        load(state, state_base + LoanField::Interest as u32),
                        &mut faults,
                    );
                    risk = add(
                        risk,
                        load(state, state_base + LoanField::RiskAssets as u32),
                        &mut faults,
                    );
                    let status =
                        load(state, state_base + LoanField::Status as u32).extract(LOW_WORD);
                    let loan_principal = load(state, state_base + LoanField::Principal as u32);
                    let loan_interest = load(state, state_base + LoanField::Interest as u32);
                    let remainder = load(state, state_base + LoanField::InterestRemainder as u32);
                    let due = load(state, state_base + LoanField::Due as u32);
                    let last_accrual = load(state, state_base + LoanField::LastAccrual as u32);
                    let loan_risk = load(state, state_base + LoanField::RiskAssets as u32);
                    let lifecycle = match status {
                        0 => {
                            equal(loan_principal, zero())
                                && equal(loan_interest, zero())
                                && equal(remainder, zero())
                                && equal(due, zero())
                                && equal(last_accrual, zero())
                                && equal(loan_risk, zero())
                        }
                        1 => {
                            !equal(loan_principal, zero())
                                && lt(last_accrual, due)
                                && !equal(loan_risk, zero())
                        }
                        2 => {
                            !equal(loan_principal, zero())
                                && !equal(due, zero())
                                && !lt(last_accrual, due)
                                && !equal(loan_risk, zero())
                        }
                        3 => {
                            equal(loan_principal, zero())
                                && equal(loan_interest, zero())
                                && equal(loan_risk, zero())
                                && !equal(due, zero())
                                && !lt(last_accrual, due)
                        }
                        _ => false,
                    };
                    let expected_risk = ceil_scaled(
                        load(state, state_base + LoanField::Principal as u32),
                        banks[base + BankMeta::RISK_WEIGHT],
                        FINANCE_SCALE,
                        &mut faults,
                    );
                    active += u32::cast_from(status == LoanStatus::Active as u32);
                    overdue += u32::cast_from(status == LoanStatus::Overdue as u32);
                    if status > LoanStatus::Repaid as u32
                        || !lifecycle
                        || load(state, state_base + LoanField::Status as u32).extract(HIGH_WORD)
                            != 0
                        || due.extract(HIGH_WORD) != 0
                        || lt(ticks(day, 0), last_accrual)
                        || lt(
                            minimum(
                                read(loans, loan + LoanMeta::REQUESTED),
                                read(loans, loan + LoanMeta::TARGET_CASH),
                            ),
                            loan_principal,
                        )
                        || (loans[loan + LoanMeta::CONSENT] != 1
                            && status != LoanStatus::Dormant as u32)
                        || (status != LoanStatus::Dormant as u32
                            && lt(
                                due,
                                ticks(
                                    loans[loan + LoanMeta::FIRST_DAY]
                                        + loans[loan + LoanMeta::TERM],
                                    0,
                                ),
                            ))
                        || !equal(
                            expected_risk,
                            load(state, state_base + LoanField::RiskAssets as u32),
                        )
                        || !lt(
                            load(state, state_base + LoanField::InterestRemainder as u32),
                            ticks(FINANCE_SCALE, 0),
                        )
                    {
                        faults.insert(KernelFaults::new(comptime!(FaultFlags::INVALID_COORDINATE)));
                    }
                }
            }
        }
        if comptime!(matches!(stage, FinanceStage::Audit)) {
            let b = banks[base + BankMeta::STATE];
            let reserve = load(state, b + BankField::Reserve as u32);
            let assets = add(reserve, principal, &mut faults);
            if !equal(assets, load(state, b + BankField::Equity as u32))
                || !equal(risk, load(state, b + BankField::RiskAssets as u32))
            {
                faults.insert(KernelFaults::new(comptime!(FaultFlags::INVALID_COORDINATE)));
            }
            store(output, id * 5, reserve);
            store(output, id * 5 + 1, principal);
            store(output, id * 5 + 2, interest);
            store(output, id * 5 + 3, ticks(active, 0));
            store(output, id * 5 + 4, ticks(overdue, 0));
        }
        faults.record(control);
    }
}

#[cube(launch)]
pub(super) fn support_needs(state: &mut [u32], recipients: &[u32], output: &mut [u32], count: u32) {
    let id = ABSOLUTE_POS as u32;
    if id < count {
        let base = id as usize * SupportMeta::WORDS;
        let need = positive(
            read(recipients, base + SupportMeta::TARGET),
            load(state, recipients[base + SupportMeta::CASH]),
        );
        store(
            output,
            id,
            select(recipients[base + SupportMeta::CONSENT] == 1, need, zero()),
        );
    }
}

#[cube(launch)]
pub(super) fn scan(
    input: &mut [u32],
    output: &mut [u32],
    control: &[Atomic<u32>],
    count: u32,
    channels: u32,
    offset: u32,
) {
    let id = ABSOLUTE_POS as u32;
    if id < count * channels {
        let mut sum = load(input, id);
        if id % count >= offset {
            let addition = plus(sum, load(input, id - offset));
            addition.faults.record(control);
            sum = addition.value;
        }
        store(output, id, sum);
    }
}

#[cube]
pub(super) fn bounded_product_ratio(a: Ticks, b: Ticks, denominator: Ticks) -> Ticks {
    let mut quotient = zero();
    let mut remainder = zero();
    for index in 0..TICK_BITS {
        let bit = TICK_BITS - 1 - index;
        let digit = (select(bit >= WORD_BITS, b.extract(HIGH_WORD), b.extract(LOW_WORD))
            >> (bit & WORD_TOP_BIT))
            & 1;
        let carry = remainder.extract(HIGH_WORD) >> WORD_TOP_BIT;
        remainder = ticks(
            remainder.extract(LOW_WORD) << 1,
            (remainder.extract(HIGH_WORD) << 1) | (remainder.extract(LOW_WORD) >> WORD_TOP_BIT),
        );
        let reduce = carry != 0 || !lt(remainder, denominator);
        remainder = select(reduce, minus(remainder, denominator), remainder);
        let added = plus(remainder, select(digit == 1, a, zero()));
        let reduce_again = added.faults.bits() != 0 || !lt(added.value, denominator);
        remainder = select(reduce_again, minus(added.value, denominator), added.value);
        quotient = ticks(
            quotient.extract(LOW_WORD) << 1,
            (quotient.extract(HIGH_WORD) << 1) | (quotient.extract(LOW_WORD) >> WORD_TOP_BIT),
        );
        quotient = plus(
            quotient,
            ticks(u32::cast_from(reduce) + u32::cast_from(reduce_again), 0),
        )
        .value;
    }
    quotient
}

#[cube(launch)]
pub(super) fn support_plan(
    state: &mut [u32],
    recipients: &[u32],
    scanned: &mut [u32],
    output: &mut [u32],
    control: &[Atomic<u32>],
    count: u32,
    treasury: u32,
) {
    let id = ABSOLUTE_POS as u32;
    let available = load(state, treasury);
    let mut total = zero();
    if count > 0 {
        total = load(scanned, count - 1);
    }
    let capacity = minimum(available, total);
    if id < count {
        let cumulative = load(scanned, id);
        let mut previous = zero();
        if id > 0 {
            previous = load(scanned, id - 1);
        }
        let divisor = maximum(total, ticks(1, 0));
        let grant = minus(
            bounded_product_ratio(capacity, cumulative, divisor),
            bounded_product_ratio(capacity, previous, divisor),
        );
        let cash = load(
            state,
            recipients[id as usize * SupportMeta::WORDS + SupportMeta::CASH],
        );
        let new_cash = plus(cash, grant);
        new_cash.faults.record(control);
        if !equal(grant, zero()) {
            control[comptime!(crate::simulation::flags::CONSTRUCTED_EVENTS_WORD)].fetch_add(1);
        }
        store(output, id, new_cash.value);
    }
    if id == count {
        store(output, count, minus(available, capacity));
    }
}

#[cube(launch)]
pub(super) fn support_post(
    state: &mut [u32],
    recipients: &[u32],
    output: &mut [u32],
    count: u32,
    treasury: u32,
) {
    let id = ABSOLUTE_POS as u32;
    if id < count {
        store(
            state,
            recipients[id as usize * SupportMeta::WORDS + SupportMeta::CASH],
            load(output, id),
        );
    }
    if id == count {
        store(state, treasury, load(output, count));
    }
}

#[cube(launch)]
pub(super) fn audit_columns(input: &mut [u32], output: &mut [u32], count: u32) {
    let id = ABSOLUTE_POS as u32;
    if id < count * 5 {
        let bank = id % count;
        let field = id / count;
        store(output, id, load(input, bank * 5 + field));
    }
}

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
    bank: &RowIndex,
    loan: &RowIndex,
    day: u32,
    control: &[Atomic<u32>],
) -> u32 {
    let mut decision = 0u32;
    let bank_count = ((banks.len() - 1) / BankMeta::WORDS) as u32;
    let loan_count = ((loans.len() - 1) / LoanMeta::WORDS) as u32;
    let b = banks[bank.field(BankMeta::STATE, 1usize)];
    let l = loans[loan.field(LoanMeta::STATE, 1usize)];
    let cash_coordinate = loans[loan.field(LoanMeta::CASH, 1usize)];
    let status = load(state, l + LoanField::Status as u32 * loan_count);
    let cash = load(state, cash_coordinate);
    let requested = minimum(
        read(loans, loan.field(LoanMeta::REQUESTED, 2usize)),
        positive(read(loans, loan.field(LoanMeta::TARGET_CASH, 2usize)), cash),
    );
    let eligible = equal(status, ticks(LoanStatus::Dormant as u32, 0))
        && day >= loans[loan.field(LoanMeta::FIRST_DAY, 1usize)]
        && loans[loan.field(LoanMeta::CONSENT, 1usize)] == 1
        && !equal(requested, zero());
    if eligible {
        decision = 2u32;
        let mut faults = KernelFaults::none();
        let reserve = load(state, b + BankField::Reserve as u32 * bank_count);
        let risk = load(state, b + BankField::RiskAssets as u32 * bank_count);
        let amount = lending_limit(
            requested,
            loans[loan.field(LoanMeta::TERM, 1usize)],
            loans[loan.field(LoanMeta::DAILY_RATE, 1usize)],
            read(loans, loan.field(LoanMeta::SURPLUS, 2usize)),
            read(loans, loan.field(LoanMeta::DEBT_SERVICE, 2usize)),
            loans[loan.field(LoanMeta::SERVICE_FRACTION, 1usize)],
            loans[loan.field(LoanMeta::ADVANCE, 1usize)],
            read(loans, loan.field(LoanMeta::COLLATERAL, 2usize)),
            read(loans, loan.field(LoanMeta::SECURED, 2usize)),
            load(state, b + BankField::Equity as u32 * bank_count),
            banks[bank.field(BankMeta::MINIMUM_CAPITAL, 1usize)],
            risk,
            banks[bank.field(BankMeta::RISK_WEIGHT, 1usize)],
            reserve,
            &mut faults,
        );
        let risk_debit = ceil_scaled(
            amount,
            banks[bank.field(BankMeta::RISK_WEIGHT, 1usize)],
            FINANCE_SCALE,
            &mut faults,
        );
        let next_risk = add(risk, risk_debit, &mut faults);
        let next_cash = add(cash, amount, &mut faults);
        let due = day + loans[loan.field(LoanMeta::TERM, 1usize)];
        if due < day {
            faults.insert(KernelFaults::new(comptime!(
                FaultFlags::ARITHMETIC_OVERFLOW
            )));
        }
        if faults.bits() == 0 && !equal(amount, zero()) {
            decision = 1u32;
            store(
                state,
                b + BankField::Reserve as u32 * bank_count,
                minus(reserve, amount),
            );
            store(
                state,
                b + BankField::RiskAssets as u32 * bank_count,
                next_risk,
            );
            store(state, cash_coordinate, next_cash);
            store(state, l + LoanField::Principal as u32 * loan_count, amount);
            store(state, l + LoanField::Due as u32 * loan_count, ticks(due, 0));
            store(
                state,
                l + LoanField::Status as u32 * loan_count,
                ticks(LoanStatus::Active as u32, 0),
            );
            store(
                state,
                l + LoanField::RiskAssets as u32 * loan_count,
                risk_debit,
            );
            store(
                state,
                l + LoanField::LastAccrual as u32 * loan_count,
                ticks(day, 0),
            );
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
    bank: &RowIndex,
    loan: &RowIndex,
    day: u32,
    control: &[Atomic<u32>],
) -> u32 {
    let mut decision = 0u32;
    let bank_count = ((banks.len() - 1) / BankMeta::WORDS) as u32;
    let loan_count = ((loans.len() - 1) / LoanMeta::WORDS) as u32;
    let b = banks[bank.field(BankMeta::STATE, 1usize)];
    let l = loans[loan.field(LoanMeta::STATE, 1usize)];
    let status = load(state, l + LoanField::Status as u32 * loan_count).extract(LOW_WORD);
    if status == LoanStatus::Active as u32 || status == LoanStatus::Overdue as u32 {
        let mut faults = KernelFaults::none();
        let principal = load(state, l + LoanField::Principal as u32 * loan_count);
        let last = load(state, l + LoanField::LastAccrual as u32 * loan_count).extract(LOW_WORD);
        let elapsed = day - min(day, last);
        let daily = scaled(
            principal,
            loans[loan.field(LoanMeta::DAILY_RATE, 1usize)],
            FINANCE_SCALE,
            &mut faults,
        );
        let remainder = minus(
            principal,
            times(divide(principal, FINANCE_SCALE), FINANCE_SCALE).value,
        );
        let daily_remainder = remainder.extract(LOW_WORD)
            * loans[loan.field(LoanMeta::DAILY_RATE, 1usize)]
            % FINANCE_SCALE;
        let accumulated = add(
            times32(daily_remainder, elapsed),
            load(state, l + LoanField::InterestRemainder as u32 * loan_count),
            &mut faults,
        );
        let carried = divide(accumulated, FINANCE_SCALE);
        let new_remainder = minus(accumulated, times(carried, FINANCE_SCALE).value);
        let interest = add(
            load(state, l + LoanField::Interest as u32 * loan_count),
            add(multiply(daily, elapsed, &mut faults), carried, &mut faults),
            &mut faults,
        );
        let total = add(principal, interest, &mut faults);
        let cash_coordinate = loans[loan.field(LoanMeta::CASH, 1usize)];
        let cash = load(state, cash_coordinate);
        let due = load(state, l + LoanField::Due as u32 * loan_count).extract(LOW_WORD);
        let repayment = day >= due && !lt(cash, total);
        let reserve = add(
            load(state, b + BankField::Reserve as u32 * bank_count),
            select(repayment, total, zero()),
            &mut faults,
        );
        let equity = add(
            load(state, b + BankField::Equity as u32 * bank_count),
            select(repayment, interest, zero()),
            &mut faults,
        );
        let risk = load(state, b + BankField::RiskAssets as u32 * bank_count);
        let loan_risk = load(state, l + LoanField::RiskAssets as u32 * loan_count);
        if last > day || lt(risk, loan_risk) {
            faults.insert(KernelFaults::new(comptime!(FaultFlags::INVALID_COORDINATE)));
        }
        if faults.bits() == 0 {
            store(state, l + LoanField::Interest as u32 * loan_count, interest);
            store(
                state,
                l + LoanField::InterestRemainder as u32 * loan_count,
                new_remainder,
            );
            store(
                state,
                l + LoanField::LastAccrual as u32 * loan_count,
                ticks(day, 0),
            );
            if day >= due {
                decision = 2u32;
                store(
                    state,
                    l + LoanField::Status as u32 * loan_count,
                    ticks(LoanStatus::Overdue as u32, 0),
                );
            }
            if repayment {
                decision = 1u32;
                store(state, cash_coordinate, minus(cash, total));
                store(state, b + BankField::Reserve as u32 * bank_count, reserve);
                store(state, b + BankField::Equity as u32 * bank_count, equity);
                store(
                    state,
                    b + BankField::RiskAssets as u32 * bank_count,
                    minus(risk, loan_risk),
                );
                store(state, l + LoanField::Principal as u32 * loan_count, zero());
                store(state, l + LoanField::Interest as u32 * loan_count, zero());
                store(state, l + LoanField::RiskAssets as u32 * loan_count, zero());
                store(
                    state,
                    l + LoanField::Status as u32 * loan_count,
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
        let bank_count = count;
        let loan_count = ((loans.len() - 1) / LoanMeta::WORDS) as u32;
        let base = RowIndex::new(0, count as usize, id as usize);
        let first = banks[base.field(BankMeta::FIRST_LOAN, 1usize)];
        let last = first + banks[base.field(BankMeta::LOAN_COUNT, 1usize)];
        let mut principal = zero();
        let mut interest = zero();
        let mut risk = zero();
        let mut active = 0u32;
        let mut overdue = 0u32;
        let mut faults = KernelFaults::none();
        for index in first..last {
            let loan = RowIndex::new(0, (loans.len() - 1) / LoanMeta::WORDS, index as usize);
            match stage {
                FinanceStage::Opening => {
                    decisions[index as usize] =
                        opening(state, banks, loans, &base, &loan, day, control);
                }
                FinanceStage::Lending => {
                    decisions[index as usize] =
                        issue(state, banks, loans, &base, &loan, day, control);
                }
                FinanceStage::Audit => {
                    let state_base = loans[loan.field(LoanMeta::STATE, 1usize)];
                    principal = add(
                        principal,
                        load(state, state_base + LoanField::Principal as u32 * loan_count),
                        &mut faults,
                    );
                    interest = add(
                        interest,
                        load(state, state_base + LoanField::Interest as u32 * loan_count),
                        &mut faults,
                    );
                    risk = add(
                        risk,
                        load(
                            state,
                            state_base + LoanField::RiskAssets as u32 * loan_count,
                        ),
                        &mut faults,
                    );
                    let status = load(state, state_base + LoanField::Status as u32 * loan_count)
                        .extract(LOW_WORD);
                    let loan_principal =
                        load(state, state_base + LoanField::Principal as u32 * loan_count);
                    let loan_interest =
                        load(state, state_base + LoanField::Interest as u32 * loan_count);
                    let remainder = load(
                        state,
                        state_base + LoanField::InterestRemainder as u32 * loan_count,
                    );
                    let due = load(state, state_base + LoanField::Due as u32 * loan_count);
                    let last_accrual = load(
                        state,
                        state_base + LoanField::LastAccrual as u32 * loan_count,
                    );
                    let loan_risk = load(
                        state,
                        state_base + LoanField::RiskAssets as u32 * loan_count,
                    );
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
                        load(state, state_base + LoanField::Principal as u32 * loan_count),
                        banks[base.field(BankMeta::RISK_WEIGHT, 1usize)],
                        FINANCE_SCALE,
                        &mut faults,
                    );
                    active += u32::cast_from(status == LoanStatus::Active as u32);
                    overdue += u32::cast_from(status == LoanStatus::Overdue as u32);
                    if status > LoanStatus::Repaid as u32
                        || !lifecycle
                        || load(state, state_base + LoanField::Status as u32 * loan_count)
                            .extract(HIGH_WORD)
                            != 0
                        || due.extract(HIGH_WORD) != 0
                        || lt(ticks(day, 0), last_accrual)
                        || lt(
                            minimum(
                                read(loans, loan.field(LoanMeta::REQUESTED, 2usize)),
                                read(loans, loan.field(LoanMeta::TARGET_CASH, 2usize)),
                            ),
                            loan_principal,
                        )
                        || (loans[loan.field(LoanMeta::CONSENT, 1usize)] != 1
                            && status != LoanStatus::Dormant as u32)
                        || (status != LoanStatus::Dormant as u32
                            && lt(
                                due,
                                ticks(
                                    loans[loan.field(LoanMeta::FIRST_DAY, 1usize)]
                                        + loans[loan.field(LoanMeta::TERM, 1usize)],
                                    0,
                                ),
                            ))
                        || !equal(
                            expected_risk,
                            load(
                                state,
                                state_base + LoanField::RiskAssets as u32 * loan_count,
                            ),
                        )
                        || !lt(
                            load(
                                state,
                                state_base + LoanField::InterestRemainder as u32 * loan_count,
                            ),
                            ticks(FINANCE_SCALE, 0),
                        )
                    {
                        faults.insert(KernelFaults::new(comptime!(FaultFlags::INVALID_COORDINATE)));
                    }
                }
            }
        }
        if comptime!(matches!(stage, FinanceStage::Audit)) {
            let b = banks[base.field(BankMeta::STATE, 1usize)];
            let reserve = load(state, b + BankField::Reserve as u32 * bank_count);
            let assets = add(reserve, principal, &mut faults);
            if !equal(
                assets,
                load(state, b + BankField::Equity as u32 * bank_count),
            ) || !equal(
                risk,
                load(state, b + BankField::RiskAssets as u32 * bank_count),
            ) {
                faults.insert(KernelFaults::new(comptime!(FaultFlags::INVALID_COORDINATE)));
            }
            store(output, id, reserve);
            store(output, count + id, principal);
            store(output, 2 * count + id, interest);
            store(output, 3 * count + id, ticks(active, 0));
            store(output, 4 * count + id, ticks(overdue, 0));
        }
        faults.record(control);
    }
}

#[cube(launch)]
pub(super) fn support_needs(state: &mut [u32], recipients: &[u32], output: &mut [u32], count: u32) {
    let id = ABSOLUTE_POS as u32;
    if id < count {
        let base = RowIndex::new(0, count as usize, id as usize);
        let need = positive(
            read(recipients, base.field(SupportMeta::TARGET, 2usize)),
            load(state, recipients[base.field(SupportMeta::CASH, 1usize)]),
        );
        store(
            output,
            id,
            select(
                recipients[base.field(SupportMeta::CONSENT, 1usize)] == 1,
                need,
                zero(),
            ),
        );
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
            recipients
                [RowIndex::new(0, count as usize, id as usize).field(SupportMeta::CASH, 1usize)],
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
            recipients
                [RowIndex::new(0, count as usize, id as usize).field(SupportMeta::CASH, 1usize)],
            load(output, id),
        );
    }
    if id == count {
        store(state, treasury, load(output, count));
    }
}

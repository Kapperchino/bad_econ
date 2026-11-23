use super::{
    credit::{CreditCapacity, CreditTerms},
    *,
};
use crate::simulation::storage::{
    flags::{CONSTRUCTED_EVENTS_WORD, KernelFaults},
    layout::{HIGH_WORD, LOW_WORD},
    state::BalanceBuffer,
};

#[derive(CubeType, Clone, Copy)]
struct BankAccountRow {
    base: u32,
    stride: u32,
    first_loan: u32,
    end_loan: u32,
    minimum_capital: u32,
    risk_weight: u32,
}

#[cube]
impl BankAccountRow {
    fn new(banks: &[u32], count: u32, id: u32) -> Self {
        let row = RowIndex::new(0, count as usize, id as usize);
        let first_loan = banks[row.field(BankMeta::FIRST_LOAN, 1usize)];
        BankAccountRow {
            base: banks[row.field(BankMeta::STATE, 1usize)],
            stride: count,
            first_loan,
            end_loan: first_loan + banks[row.field(BankMeta::LOAN_COUNT, 1usize)],
            minimum_capital: banks[row.field(BankMeta::MINIMUM_CAPITAL, 1usize)],
            risk_weight: banks[row.field(BankMeta::RISK_WEIGHT, 1usize)],
        }
    }

    fn coordinate(&self, #[comptime] field: BankField) -> u32 {
        self.base + comptime!(field as u32) * self.stride
    }
}

#[derive(CubeType)]
struct LoanAccount<'a> {
    records: &'a [u32],
    row: RowIndex,
    base: u32,
    stride: u32,
}

#[cube]
impl<'a> LoanAccount<'a> {
    fn new(records: &'a [u32], count: u32, id: u32) -> Self {
        let row = RowIndex::new(0, count as usize, id as usize);
        let base = records[row.field(LoanMeta::STATE, 1usize)];
        LoanAccount::<'a> {
            records,
            row,
            base,
            stride: count,
        }
    }

    fn coordinate(&self, #[comptime] field: LoanField) -> u32 {
        self.base + comptime!(field as u32) * self.stride
    }

    fn number(&self, #[comptime] field: usize) -> u32 {
        self.records[self.row.field(field, 1usize)]
    }

    fn amount(&self, #[comptime] field: usize) -> Ticks {
        read_tick_words(self.records, self.row.field(field, TICK_WORDS))
    }

    fn credit_terms(&self, requested: Ticks) -> CreditTerms {
        CreditTerms {
            requested,
            term: self.number(LoanMeta::TERM),
            rate: self.number(LoanMeta::DAILY_RATE),
            surplus: self.amount(LoanMeta::SURPLUS),
            debt_service: self.amount(LoanMeta::DEBT_SERVICE),
            service_fraction: self.number(LoanMeta::SERVICE_FRACTION),
            advance: self.number(LoanMeta::ADVANCE),
            collateral: self.amount(LoanMeta::COLLATERAL),
            secured: self.amount(LoanMeta::SECURED),
        }
    }
}

#[derive(CubeType)]
struct BankLedger<'a> {
    balances: BalanceBuffer<'a>,
    bank: BankAccountRow,
    control: &'a [Atomic<u32>],
    day: u32,
}

#[cube]
impl<'a> BankLedger<'a> {
    fn value(&self, #[comptime] field: BankField) -> Ticks {
        self.balances.value(self.bank.coordinate(field))
    }

    fn set(&mut self, #[comptime] field: BankField, amount: Ticks) {
        self.balances.set(self.bank.coordinate(field), amount);
    }

    fn loan_value(&self, loan: &LoanAccount, #[comptime] field: LoanField) -> Ticks {
        self.balances.value(loan.coordinate(field))
    }

    fn set_loan(&mut self, loan: &LoanAccount, #[comptime] field: LoanField, amount: Ticks) {
        self.balances.set(loan.coordinate(field), amount);
    }

    fn credit_capacity(&self) -> CreditCapacity {
        CreditCapacity {
            equity: self.value(BankField::Equity),
            minimum_capital: self.bank.minimum_capital,
            risk_assets: self.value(BankField::RiskAssets),
            risk_weight: self.bank.risk_weight,
            liquidity: self.value(BankField::Reserve),
        }
    }

    fn issue(&mut self, loan: &LoanAccount) -> u32 {
        let mut decision = u32::new(comptime!(LoanDecision::NoEvent as i64));
        let cash_coordinate = loan.number(LoanMeta::CASH);
        let status = self.loan_value(loan, LoanField::Status);
        let cash = self.balances.value(cash_coordinate);
        let requested = minimum(
            loan.amount(LoanMeta::REQUESTED),
            positive(loan.amount(LoanMeta::TARGET_CASH), cash),
        );
        let eligible = equal(status, ticks(LoanStatus::Dormant as u32, 0))
            && self.day >= loan.number(LoanMeta::FIRST_DAY)
            && loan.number(LoanMeta::CONSENT) == 1
            && !equal(requested, zero());
        if eligible {
            decision = comptime!(LoanDecision::Rejected as u32);
            let mut faults = KernelFaults::none();
            let capacity = self.credit_capacity();
            let terms = loan.credit_terms(requested);
            let amount = terms.limit(&capacity, &mut faults);
            let risk_debit =
                CheckedTicks::ceiling_ratio(amount, capacity.risk_weight, FINANCE_SCALE)
                    .accumulate(&mut faults);
            let next_risk = plus(capacity.risk_assets, risk_debit).accumulate(&mut faults);
            let next_cash = plus(cash, amount).accumulate(&mut faults);
            let due = self.day + terms.term;
            if due < self.day {
                faults.insert(KernelFaults::new(comptime!(
                    FaultFlags::ARITHMETIC_OVERFLOW
                )));
            }
            if faults.is_empty() && !equal(amount, zero()) {
                decision = comptime!(LoanDecision::Accepted as u32);
                self.set(BankField::Reserve, minus(capacity.liquidity, amount));
                self.set(BankField::RiskAssets, next_risk);
                self.balances.set(cash_coordinate, next_cash);
                self.set_loan(loan, LoanField::Principal, amount);
                self.set_loan(loan, LoanField::Due, ticks(due, 0));
                self.set_loan(loan, LoanField::Status, ticks(LoanStatus::Active as u32, 0));
                self.set_loan(loan, LoanField::RiskAssets, risk_debit);
                self.set_loan(loan, LoanField::LastAccrual, ticks(self.day, 0));
            }
            faults.record(self.control);
        }
        decision
    }

    fn accrue_and_collect(&mut self, loan: &LoanAccount) -> u32 {
        let mut decision = u32::new(comptime!(LoanDecision::NoEvent as i64));
        let status = self.loan_value(loan, LoanField::Status).extract(LOW_WORD);
        if status == LoanStatus::Active as u32 || status == LoanStatus::Overdue as u32 {
            let mut faults = KernelFaults::none();
            let principal = self.loan_value(loan, LoanField::Principal);
            let last = self
                .loan_value(loan, LoanField::LastAccrual)
                .extract(LOW_WORD);
            let elapsed = self.day - min(self.day, last);
            let rate = loan.number(LoanMeta::DAILY_RATE);
            let daily = ratio(principal, rate, FINANCE_SCALE).accumulate(&mut faults);
            let remainder = minus(
                principal,
                times(divide(principal, FINANCE_SCALE), FINANCE_SCALE).value,
            );
            let daily_remainder = remainder.extract(LOW_WORD) * rate % FINANCE_SCALE;
            let accumulated = plus(
                times32(daily_remainder, elapsed),
                self.loan_value(loan, LoanField::InterestRemainder),
            )
            .accumulate(&mut faults);
            let carried = divide(accumulated, FINANCE_SCALE);
            let new_remainder = minus(accumulated, times(carried, FINANCE_SCALE).value);
            let interest = plus(
                self.loan_value(loan, LoanField::Interest),
                plus(times(daily, elapsed).accumulate(&mut faults), carried)
                    .accumulate(&mut faults),
            )
            .accumulate(&mut faults);
            let total = plus(principal, interest).accumulate(&mut faults);
            let cash_coordinate = loan.number(LoanMeta::CASH);
            let cash = self.balances.value(cash_coordinate);
            let due = self.loan_value(loan, LoanField::Due).extract(LOW_WORD);
            let repayment = self.day >= due && !lt(cash, total);
            let reserve = plus(
                self.value(BankField::Reserve),
                select(repayment, total, zero()),
            )
            .accumulate(&mut faults);
            let equity = plus(
                self.value(BankField::Equity),
                select(repayment, interest, zero()),
            )
            .accumulate(&mut faults);
            let risk = self.value(BankField::RiskAssets);
            let loan_risk = self.loan_value(loan, LoanField::RiskAssets);
            if last > self.day || lt(risk, loan_risk) {
                faults.insert(KernelFaults::new(comptime!(FaultFlags::INVALID_COORDINATE)));
            }
            if faults.is_empty() {
                self.set_loan(loan, LoanField::Interest, interest);
                self.set_loan(loan, LoanField::InterestRemainder, new_remainder);
                self.set_loan(loan, LoanField::LastAccrual, ticks(self.day, 0));
                if self.day >= due {
                    decision = comptime!(LoanDecision::Rejected as u32);
                    self.set_loan(
                        loan,
                        LoanField::Status,
                        ticks(LoanStatus::Overdue as u32, 0),
                    );
                }
                if repayment {
                    decision = comptime!(LoanDecision::Accepted as u32);
                    self.balances.set(cash_coordinate, minus(cash, total));
                    self.set(BankField::Reserve, reserve);
                    self.set(BankField::Equity, equity);
                    self.set(BankField::RiskAssets, minus(risk, loan_risk));
                    self.set_loan(loan, LoanField::Principal, zero());
                    self.set_loan(loan, LoanField::Interest, zero());
                    self.set_loan(loan, LoanField::RiskAssets, zero());
                    self.set_loan(loan, LoanField::Status, ticks(LoanStatus::Repaid as u32, 0));
                }
            }
            faults.record(self.control);
        }
        decision
    }
}

#[derive(CubeType)]
struct BankAudit {
    principal: Ticks,
    interest: Ticks,
    risk: Ticks,
    active: u32,
    overdue: u32,
    faults: KernelFaults,
}

#[cube]
impl BankAudit {
    fn new() -> Self {
        BankAudit {
            principal: zero(),
            interest: zero(),
            risk: zero(),
            active: 0,
            overdue: 0,
            faults: KernelFaults::none(),
        }
    }

    fn include(&mut self, book: &BankLedger, loan: &LoanAccount) {
        let principal = book.loan_value(loan, LoanField::Principal);
        let interest = book.loan_value(loan, LoanField::Interest);
        let risk = book.loan_value(loan, LoanField::RiskAssets);
        let remainder = book.loan_value(loan, LoanField::InterestRemainder);
        let due = book.loan_value(loan, LoanField::Due);
        let last_accrual = book.loan_value(loan, LoanField::LastAccrual);
        let status = book.loan_value(loan, LoanField::Status);
        let lifecycle = match status.extract(LOW_WORD) {
            0 => {
                equal(principal, zero())
                    && equal(interest, zero())
                    && equal(remainder, zero())
                    && equal(due, zero())
                    && equal(last_accrual, zero())
                    && equal(risk, zero())
            }
            1 => !equal(principal, zero()) && lt(last_accrual, due) && !equal(risk, zero()),
            2 => {
                !equal(principal, zero())
                    && !equal(due, zero())
                    && !lt(last_accrual, due)
                    && !equal(risk, zero())
            }
            3 => {
                equal(principal, zero())
                    && equal(interest, zero())
                    && equal(risk, zero())
                    && !equal(due, zero())
                    && !lt(last_accrual, due)
            }
            _ => false,
        };
        self.principal = plus(self.principal, principal).accumulate(&mut self.faults);
        self.interest = plus(self.interest, interest).accumulate(&mut self.faults);
        self.risk = plus(self.risk, risk).accumulate(&mut self.faults);
        let expected_risk =
            CheckedTicks::ceiling_ratio(principal, book.bank.risk_weight, FINANCE_SCALE)
                .accumulate(&mut self.faults);
        self.active += u32::cast_from(status.extract(LOW_WORD) == LoanStatus::Active as u32);
        self.overdue += u32::cast_from(status.extract(LOW_WORD) == LoanStatus::Overdue as u32);
        let contracted = minimum(
            loan.amount(LoanMeta::REQUESTED),
            loan.amount(LoanMeta::TARGET_CASH),
        );
        let dormant = status.extract(LOW_WORD) == LoanStatus::Dormant as u32;
        let earliest_due = loan.number(LoanMeta::FIRST_DAY) + loan.number(LoanMeta::TERM);
        if !lifecycle
            || status.extract(HIGH_WORD) != 0
            || due.extract(HIGH_WORD) != 0
            || lt(ticks(book.day, 0), last_accrual)
            || lt(contracted, principal)
            || (!dormant
                && (loan.number(LoanMeta::CONSENT) != 1 || lt(due, ticks(earliest_due, 0))))
            || !equal(expected_risk, risk)
            || !lt(remainder, ticks(FINANCE_SCALE, 0))
        {
            self.faults
                .insert(KernelFaults::new(comptime!(FaultFlags::INVALID_COORDINATE)));
        }
    }

    fn report(&mut self, book: &BankLedger, output: &mut BalanceBuffer, id: u32) {
        let count = book.bank.stride;
        let reserve = book.value(BankField::Reserve);
        let assets = plus(reserve, self.principal).accumulate(&mut self.faults);
        if !equal(assets, book.value(BankField::Equity))
            || !equal(self.risk, book.value(BankField::RiskAssets))
        {
            self.faults
                .insert(KernelFaults::new(comptime!(FaultFlags::INVALID_COORDINATE)));
        }
        output.set(id, reserve);
        output.set(count + id, self.principal);
        output.set(2 * count + id, self.interest);
        output.set(3 * count + id, ticks(self.active, 0));
        output.set(4 * count + id, ticks(self.overdue, 0));
        self.faults.record(book.control);
    }
}

#[cube]
pub(super) fn account<'a>(input: &'a mut BankInput, #[comptime] stage: FinanceStage) {
    let id = ABSOLUTE_POS as u32;
    if id < input.count {
        let bank = BankAccountRow::new(&input.banks, input.count, id);
        let mut book = BankLedger::<'a> {
            balances: BalanceBuffer::new(&mut input.state),
            bank,
            control: &input.control,
            day: input.day,
        };
        let loan_count = ((input.loans.len() - 1) / LoanMeta::WORDS) as u32;
        let mut audit = BankAudit::new();
        for index in book.bank.first_loan..book.bank.end_loan {
            let loan = LoanAccount::new(&input.loans, loan_count, index);
            match stage {
                FinanceStage::Opening => {
                    let decision = book.accrue_and_collect(&loan);
                    post_decision(&mut input.decisions, &input.control, index, decision);
                }
                FinanceStage::Lending => {
                    let decision = book.issue(&loan);
                    post_decision(&mut input.decisions, &input.control, index, decision);
                }
                FinanceStage::Audit => audit.include(&book, &loan),
            }
        }
        if comptime!(matches!(stage, FinanceStage::Audit)) {
            audit.report(&book, &mut BalanceBuffer::new(&mut input.output), id);
        }
    }
}

#[cube]
fn post_decision(decisions: &mut [u32], control: &[Atomic<u32>], index: u32, decision: u32) {
    decisions[index as usize] = decision;
    if decision != comptime!(LoanDecision::NoEvent as u32) {
        control[CONSTRUCTED_EVENTS_WORD].fetch_add(1);
    }
}

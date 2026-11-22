use super::*;
use crate::simulation::{flags::KernelFaults, layout::LOW_WORD};

#[derive(CubeType, Clone, Copy)]
pub(super) struct CreditTerms {
    pub requested: Ticks,
    pub term: u32,
    pub rate: u32,
    pub surplus: Ticks,
    pub debt_service: Ticks,
    pub service_fraction: u32,
    pub advance: u32,
    pub collateral: Ticks,
    pub secured: Ticks,
}

#[derive(CubeType, Clone, Copy)]
pub(super) struct CreditCapacity {
    pub equity: Ticks,
    pub minimum_capital: u32,
    pub risk_assets: Ticks,
    pub risk_weight: u32,
    pub liquidity: Ticks,
}

#[cube]
pub(super) fn add(a: Ticks, b: Ticks, faults: &mut KernelFaults) -> Ticks {
    let result = plus(a, b);
    faults.insert(result.faults);
    result.value
}

#[cube]
pub(super) fn multiply(a: Ticks, b: u32, faults: &mut KernelFaults) -> Ticks {
    let result = times(a, b);
    faults.insert(result.faults);
    result.value
}

#[cube]
pub(super) fn scaled(a: Ticks, n: u32, d: u32, faults: &mut KernelFaults) -> Ticks {
    let result = ratio(a, n, d);
    faults.insert(result.faults);
    result.value
}

#[cube]
pub(super) fn ceil_scaled(a: Ticks, n: u32, d: u32, faults: &mut KernelFaults) -> Ticks {
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
    terms: &CreditTerms,
    capacity: &CreditCapacity,
    faults: &mut KernelFaults,
) -> Ticks {
    let service = positive(
        multiply(terms.surplus, terms.service_fraction, faults),
        multiply(terms.debt_service, FINANCE_SCALE, faults),
    );
    let service_cap = scaled(
        service,
        terms.term,
        FINANCE_SCALE + terms.rate * terms.term,
        faults,
    );
    let collateral_cap = divide(
        positive(
            multiply(terms.collateral, terms.advance, faults),
            multiply(terms.secured, FINANCE_SCALE, faults),
        ),
        FINANCE_SCALE,
    );
    let total_risk_capacity = scaled(
        capacity.equity,
        FINANCE_SCALE,
        capacity.minimum_capital,
        faults,
    );
    let capital_cap = scaled(
        positive(total_risk_capacity, capacity.risk_assets),
        FINANCE_SCALE,
        capacity.risk_weight,
        faults,
    );
    minimum(
        minimum(
            minimum(minimum(terms.requested, service_cap), collateral_cap),
            capital_cap,
        ),
        capacity.liquidity,
    )
}

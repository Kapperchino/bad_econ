use super::*;
use crate::simulation::flags::KernelFaults;

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
impl CreditTerms {
    pub(super) fn limit(&self, capacity: &CreditCapacity, faults: &mut KernelFaults) -> Ticks {
        let service = positive(
            times(self.surplus, self.service_fraction).accumulate(faults),
            times(self.debt_service, FINANCE_SCALE).accumulate(faults),
        );
        let service_cap =
            ratio(service, self.term, FINANCE_SCALE + self.rate * self.term).accumulate(faults);
        let collateral_cap = divide(
            positive(
                times(self.collateral, self.advance).accumulate(faults),
                times(self.secured, FINANCE_SCALE).accumulate(faults),
            ),
            FINANCE_SCALE,
        );
        let total_risk_capacity =
            ratio(capacity.equity, FINANCE_SCALE, capacity.minimum_capital).accumulate(faults);
        let capital_cap = ratio(
            positive(total_risk_capacity, capacity.risk_assets),
            FINANCE_SCALE,
            capacity.risk_weight,
        )
        .accumulate(faults);
        minimum(
            minimum(
                minimum(minimum(self.requested, service_cap), collateral_cap),
                capital_cap,
            ),
            capacity.liquidity,
        )
    }
}

use super::support;

use anyhow::Result;
use bad_econ::simulation::bargaining::*;

fn teaching_request() -> BargainingRequest {
    BargainingRequest {
        income: NetIncomeBracket {
            gross_threshold: 10_000,
            net_at_threshold: 8500,
            net_slope: 650,
            target_net: 9200,
            prior_gross: 8000,
        },
        wage: WageTerms {
            legal: 0,
            funded: 4000,
            payable_hours: 5,
            employer_charge: 100,
            quantum: 1,
        },
        employer: EmployerBudget {
            receipts: 5000,
            nonlabor: 700,
            contingency: 300,
            budget: 4500,
        },
        household: HouseholdPosition {
            cash_cost: 8000,
            funded_other: 1600,
            participating_adults: 2,
            normal_day: 8,
            cover_millidays: 0,
            work_cost: 100,
        },
        factors: BargainingFactors {
            mandate: 1000,
            withdrawal: 0,
            alternatives: 0,
            concentration: 0,
            replacements: 0,
        },
        first_offer: 616,
        desired_offer: 727,
    }
}

#[test]
fn lean_tax_inversion_directional_wages_and_household_bargaining() -> Result<()> {
    let device = support::device(64)?;
    let mut funded = teaching_request();
    funded.household.cover_millidays = 30_000;
    funded.household.funded_other = 0;
    let mut impossible = teaching_request();
    impossible.wage.funded = 100;
    let mut extreme = teaching_request();
    extreme.income.prior_gross = 100_000;
    extreme.wage.payable_hours = 1;
    extreme.wage.funded = u64::MAX;
    extreme.wage.employer_charge = 0;
    extreme.first_offer = u64::MAX;
    extreme.desired_offer = 0;
    let result = evaluate(&device, &[teaching_request(), funded, impossible, extreme])?;
    assert_eq!(result.len(), 4);
    assert_eq!(result[1].required_net_rate, 700);
    assert_eq!(result[2].second_offer, None);
    assert_eq!(
        result[0],
        BargainingResult {
            gross: 11077,
            required_gross: 3077,
            minimum_wage: 616,
            maximum_wage: 727,
            payroll_ceiling: 4000,
            anchor: 400,
            required_net_rate: 340,
            weight: 600,
            second_offer: Some(682),
        }
    );
    let mut invalid = teaching_request();
    invalid.income.net_slope = 0;
    assert!(evaluate(&device, &[invalid]).is_err());
    let mut invalid = teaching_request();
    invalid.income.target_net = 1;
    assert!(evaluate(&device, &[invalid]).is_err());
    let mut invalid = teaching_request();
    invalid.household.participating_adults = 0;
    assert!(evaluate(&device, &[invalid]).is_err());
    let mut overflow = teaching_request();
    overflow.income.gross_threshold = u64::MAX;
    assert!(evaluate(&device, &[overflow]).is_err());
    Ok(())
}

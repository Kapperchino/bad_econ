use super::{Device, flags::FaultFlags, layout::WORD_BYTES};
use crate::ledger::TickLayout;
use anyhow::{Result, anyhow};
use burn_wgpu::CubeTensor;
use bytemuck::{Pod, Zeroable};
use cubecl::CubeDim;
use std::mem::offset_of;

mod kernels;
pub use kernels::{
    HouseholdQuote, KernelBargainingFactors, KernelHousehold, KernelWageBounds, bargaining_weight,
    employer_ceiling, household_quote, inverse_bracket, second_offer, wage_bounds,
};

pub const BARGAINING_SCALE: u32 = 1000;

#[derive(Clone, Debug)]
pub struct NetIncomeBracket {
    pub gross_threshold: u64,
    pub net_at_threshold: u64,
    pub net_slope: u32,
    pub target_net: u64,
    pub prior_gross: u64,
}

#[derive(Clone, Debug)]
pub struct WageTerms {
    pub legal: u64,
    pub funded: u64,
    pub payable_hours: u32,
    pub employer_charge: u32,
    pub quantum: u32,
}

#[derive(Clone, Debug)]
pub struct EmployerBudget {
    pub receipts: u64,
    pub nonlabor: u64,
    pub contingency: u64,
    pub budget: u64,
}

#[derive(Clone, Debug)]
pub struct HouseholdPosition {
    pub cash_cost: u64,
    pub funded_other: u64,
    pub participating_adults: u32,
    pub normal_day: u32,
    pub cover_millidays: u32,
    pub work_cost: u64,
}

#[derive(Clone, Debug)]
pub struct BargainingFactors {
    pub mandate: u32,
    pub withdrawal: u32,
    pub alternatives: u32,
    pub concentration: u32,
    pub replacements: u32,
}

#[derive(Clone, Debug)]
pub struct BargainingRequest {
    pub income: NetIncomeBracket,
    pub wage: WageTerms,
    pub employer: EmployerBudget,
    pub household: HouseholdPosition,
    pub factors: BargainingFactors,
    pub first_offer: u64,
    pub desired_offer: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RequestLayout {
    threshold: TickLayout,
    net_at_threshold: TickLayout,
    target: TickLayout,
    prior: TickLayout,
    legal: TickLayout,
    funded: TickLayout,
    receipts: TickLayout,
    nonlabor: TickLayout,
    contingency: TickLayout,
    budget: TickLayout,
    cash_cost: TickLayout,
    funded_other: TickLayout,
    work_cost: TickLayout,
    first: TickLayout,
    desired: TickLayout,
    slope: u32,
    hours: u32,
    employer_charge: u32,
    quantum: u32,
    adults: u32,
    normal_day: u32,
    cover_millidays: u32,
    mandate: u32,
    withdrawal: u32,
    alternatives: u32,
    concentration: u32,
    replacements: u32,
}

impl RequestLayout {
    const WORDS: usize = size_of::<Self>() / WORD_BYTES;
    const THRESHOLD: usize = offset_of!(Self, threshold) / WORD_BYTES;
    const NET_THRESHOLD: usize = offset_of!(Self, net_at_threshold) / WORD_BYTES;
    const TARGET: usize = offset_of!(Self, target) / WORD_BYTES;
    const PRIOR: usize = offset_of!(Self, prior) / WORD_BYTES;
    const LEGAL: usize = offset_of!(Self, legal) / WORD_BYTES;
    const FUNDED: usize = offset_of!(Self, funded) / WORD_BYTES;
    const RECEIPTS: usize = offset_of!(Self, receipts) / WORD_BYTES;
    const NONLABOR: usize = offset_of!(Self, nonlabor) / WORD_BYTES;
    const CONTINGENCY: usize = offset_of!(Self, contingency) / WORD_BYTES;
    const BUDGET: usize = offset_of!(Self, budget) / WORD_BYTES;
    const CASH_COST: usize = offset_of!(Self, cash_cost) / WORD_BYTES;
    const FUNDED_OTHER: usize = offset_of!(Self, funded_other) / WORD_BYTES;
    const WORK_COST: usize = offset_of!(Self, work_cost) / WORD_BYTES;
    const FIRST: usize = offset_of!(Self, first) / WORD_BYTES;
    const DESIRED: usize = offset_of!(Self, desired) / WORD_BYTES;
    const SLOPE: usize = offset_of!(Self, slope) / WORD_BYTES;
    const HOURS: usize = offset_of!(Self, hours) / WORD_BYTES;
    const EMPLOYER_CHARGE: usize = offset_of!(Self, employer_charge) / WORD_BYTES;
    const QUANTUM: usize = offset_of!(Self, quantum) / WORD_BYTES;
    const ADULTS: usize = offset_of!(Self, adults) / WORD_BYTES;
    const NORMAL_DAY: usize = offset_of!(Self, normal_day) / WORD_BYTES;
    const COVER: usize = offset_of!(Self, cover_millidays) / WORD_BYTES;
    const MANDATE: usize = offset_of!(Self, mandate) / WORD_BYTES;
    const WITHDRAWAL: usize = offset_of!(Self, withdrawal) / WORD_BYTES;
    const ALTERNATIVES: usize = offset_of!(Self, alternatives) / WORD_BYTES;
    const CONCENTRATION: usize = offset_of!(Self, concentration) / WORD_BYTES;
    const REPLACEMENTS: usize = offset_of!(Self, replacements) / WORD_BYTES;

    fn new(request: &BargainingRequest) -> Result<Self> {
        let factors = [
            &request.factors.mandate,
            &request.factors.withdrawal,
            &request.factors.alternatives,
            &request.factors.concentration,
            &request.factors.replacements,
        ];
        let hours = request
            .household
            .participating_adults
            .checked_mul(request.household.normal_day)
            .and_then(|hours| hours.checked_mul(35_000));
        let payable = BARGAINING_SCALE
            .checked_add(request.wage.employer_charge)
            .and_then(|charge| charge.checked_mul(request.wage.payable_hours));
        let admitted = (1..=BARGAINING_SCALE).contains(&request.income.net_slope)
            && request.income.target_net >= request.income.net_at_threshold
            && request.wage.quantum > 0
            && request.wage.payable_hours > 0
            && hours.is_some_and(|hours| hours > 0)
            && payable.is_some_and(|payable| payable > 0)
            && factors
                .into_iter()
                .all(|&factor| factor <= BARGAINING_SCALE);
        match admitted {
            false => Err(anyhow!(
                "invalid bargaining domain, selected income bracket, or fixed-width denominator"
            )),
            true => Ok(Self {
                threshold: request.income.gross_threshold.into(),
                net_at_threshold: request.income.net_at_threshold.into(),
                target: request.income.target_net.into(),
                prior: request.income.prior_gross.into(),
                legal: request.wage.legal.into(),
                funded: request.wage.funded.into(),
                receipts: request.employer.receipts.into(),
                nonlabor: request.employer.nonlabor.into(),
                contingency: request.employer.contingency.into(),
                budget: request.employer.budget.into(),
                cash_cost: request.household.cash_cost.into(),
                funded_other: request.household.funded_other.into(),
                work_cost: request.household.work_cost.into(),
                first: request.first_offer.into(),
                desired: request.desired_offer.into(),
                slope: request.income.net_slope,
                hours: request.wage.payable_hours,
                employer_charge: request.wage.employer_charge,
                quantum: request.wage.quantum,
                adults: request.household.participating_adults,
                normal_day: request.household.normal_day,
                cover_millidays: request.household.cover_millidays,
                mandate: request.factors.mandate,
                withdrawal: request.factors.withdrawal,
                alternatives: request.factors.alternatives,
                concentration: request.factors.concentration,
                replacements: request.factors.replacements,
            }),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ResultLayout {
    gross: TickLayout,
    required_gross: TickLayout,
    minimum_wage: TickLayout,
    maximum_wage: TickLayout,
    payroll_ceiling: TickLayout,
    anchor: TickLayout,
    required_net: TickLayout,
    second: TickLayout,
    weight: u32,
    quote_available: u32,
}
impl ResultLayout {
    const WORDS: usize = size_of::<Self>() / WORD_BYTES;
    const GROSS: usize = offset_of!(Self, gross) / WORD_BYTES;
    const REQUIRED_GROSS: usize = offset_of!(Self, required_gross) / WORD_BYTES;
    const MINIMUM_WAGE: usize = offset_of!(Self, minimum_wage) / WORD_BYTES;
    const MAXIMUM_WAGE: usize = offset_of!(Self, maximum_wage) / WORD_BYTES;
    const PAYROLL_CEILING: usize = offset_of!(Self, payroll_ceiling) / WORD_BYTES;
    const ANCHOR: usize = offset_of!(Self, anchor) / WORD_BYTES;
    const REQUIRED_NET: usize = offset_of!(Self, required_net) / WORD_BYTES;
    const SECOND: usize = offset_of!(Self, second) / WORD_BYTES;
    const WEIGHT: usize = offset_of!(Self, weight) / WORD_BYTES;
    const AVAILABLE: usize = offset_of!(Self, quote_available) / WORD_BYTES;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BargainingResult {
    pub gross: u64,
    pub required_gross: u64,
    pub minimum_wage: u64,
    pub maximum_wage: u64,
    pub payroll_ceiling: u64,
    pub anchor: u64,
    pub required_net_rate: u64,
    pub weight: u32,
    pub second_offer: Option<u64>,
}

pub fn evaluate(device: &Device, requests: &[BargainingRequest]) -> Result<Vec<BargainingResult>> {
    let mut encoded = requests
        .iter()
        .map(RequestLayout::new)
        .collect::<Result<Vec<_>>>()?;
    encoded.resize(encoded.len().max(1), RequestLayout::zeroed());
    let input = device.tensor(bytemuck::cast_slice(&encoded));
    let output = device.empty(requests.len().max(1) * ResultLayout::WORDS);
    let control = device.tensor(&[0; 4]);
    let argument = |tensor: &CubeTensor| tensor.clone().into_buffer_arg();
    kernels::evaluate::launch(
        &device.client,
        device.count(requests.len()),
        CubeDim::new_1d(device.workgroup),
        argument(&input),
        argument(&output),
        argument(&control),
        requests.len() as u32,
    );
    FaultFlags::read(&device.read(&control)?)?.into_result()?;
    let words = device.read(&output)?;
    Ok(bytemuck::cast_slice::<u32, ResultLayout>(&words)
        .iter()
        .take(requests.len())
        .map(|result| BargainingResult {
            gross: result.gross.into(),
            required_gross: result.required_gross.into(),
            minimum_wage: result.minimum_wage.into(),
            maximum_wage: result.maximum_wage.into(),
            payroll_ceiling: result.payroll_ceiling.into(),
            anchor: result.anchor.into(),
            required_net_rate: result.required_net.into(),
            weight: result.weight,
            second_offer: match result.quote_available {
                0 => None,
                _ => Some(result.second.into()),
            },
        })
        .collect())
}

use super::*;
use crate::simulation::flags::KernelFaults;

#[derive(Clone, Debug)]
pub struct LendingInput {
    pub requested: u64,
    pub term: u32,
    pub rate: u32,
    pub surplus: u64,
    pub debt_service: u64,
    pub service_fraction: u32,
    pub advance: u32,
    pub collateral: u64,
    pub secured: u64,
    pub equity: u64,
    pub minimum_capital: u32,
    pub risk_assets: u64,
    pub risk_weight: u32,
    pub liquidity: u64,
}
impl LendingInput {
    pub fn new(input: Self) -> Result<Self> {
        match (1..=36_500).contains(&input.term)
            && input.rate <= FINANCE_SCALE
            && input.service_fraction <= FINANCE_SCALE
            && input.advance <= FINANCE_SCALE
            && (1..=FINANCE_SCALE).contains(&input.minimum_capital)
            && (1..=10_000).contains(&input.risk_weight)
        {
            true => Ok(input),
            false => Err(anyhow!("invalid lending coefficient or denominator")),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LendingResult {
    pub amount: u64,
}

#[derive(Clone, Debug)]
pub struct CashFlow {
    pub day: u32,
    pub outflow: u64,
    pub receipt: u64,
    pub cushion: u64,
}

#[derive(Clone, Debug)]
pub struct FundingPlan {
    pub opening_cash: u64,
    pub opening_cushion: u64,
    pub flows: Vec<CashFlow>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FundingResult {
    pub opening_need: u64,
    pub gap: u64,
}

#[derive(Clone, Debug)]
pub struct DivisiblePool {
    pub capacity: u64,
    pub demands: Vec<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PoolResult {
    pub fills: Vec<u64>,
    pub remaining: u64,
}

pub struct FinancePolicies;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LendingRecord {
    requested: crate::ledger::TickLayout,
    term: u32,
    rate: u32,
    surplus: crate::ledger::TickLayout,
    debt_service: crate::ledger::TickLayout,
    service_fraction: u32,
    advance: u32,
    collateral: crate::ledger::TickLayout,
    secured: crate::ledger::TickLayout,
    equity: crate::ledger::TickLayout,
    minimum_capital: u32,
    risk_assets: crate::ledger::TickLayout,
    risk_weight: u32,
    liquidity: crate::ledger::TickLayout,
}
impl LendingRecord {
    const WORDS: usize = size_of::<Self>() / WORD_BYTES;
    const REQUESTED: usize = offset_of!(Self, requested) / WORD_BYTES;
    const TERM: usize = offset_of!(Self, term) / WORD_BYTES;
    const RATE: usize = offset_of!(Self, rate) / WORD_BYTES;
    const SURPLUS: usize = offset_of!(Self, surplus) / WORD_BYTES;
    const DEBT_SERVICE: usize = offset_of!(Self, debt_service) / WORD_BYTES;
    const SERVICE_FRACTION: usize = offset_of!(Self, service_fraction) / WORD_BYTES;
    const ADVANCE: usize = offset_of!(Self, advance) / WORD_BYTES;
    const COLLATERAL: usize = offset_of!(Self, collateral) / WORD_BYTES;
    const SECURED: usize = offset_of!(Self, secured) / WORD_BYTES;
    const EQUITY: usize = offset_of!(Self, equity) / WORD_BYTES;
    const MINIMUM_CAPITAL: usize = offset_of!(Self, minimum_capital) / WORD_BYTES;
    const RISK_ASSETS: usize = offset_of!(Self, risk_assets) / WORD_BYTES;
    const RISK_WEIGHT: usize = offset_of!(Self, risk_weight) / WORD_BYTES;
    const LIQUIDITY: usize = offset_of!(Self, liquidity) / WORD_BYTES;
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PlanRecord {
    first: u32,
    count: u32,
    cash: crate::ledger::TickLayout,
    cushion: crate::ledger::TickLayout,
}
impl PlanRecord {
    const WORDS: usize = size_of::<Self>() / WORD_BYTES;
    const FIRST: usize = offset_of!(Self, first) / WORD_BYTES;
    const COUNT: usize = offset_of!(Self, count) / WORD_BYTES;
    const CASH: usize = offset_of!(Self, cash) / WORD_BYTES;
    const CUSHION: usize = offset_of!(Self, cushion) / WORD_BYTES;
}

#[cube]
fn read(words: &[u32], base: usize) -> Ticks {
    ticks(words[base], words[base + 1])
}

#[cube(launch)]
fn lending(input: &[u32], output: &mut [u32], control: &[Atomic<u32>], count: u32) {
    let id = ABSOLUTE_POS as u32;
    if id < count {
        let base = id as usize * LendingRecord::WORDS;
        let mut faults = KernelFaults::none();
        let value = super::kernels::lending_limit(
            read(input, base + LendingRecord::REQUESTED),
            input[base + LendingRecord::TERM],
            input[base + LendingRecord::RATE],
            read(input, base + LendingRecord::SURPLUS),
            read(input, base + LendingRecord::DEBT_SERVICE),
            input[base + LendingRecord::SERVICE_FRACTION],
            input[base + LendingRecord::ADVANCE],
            read(input, base + LendingRecord::COLLATERAL),
            read(input, base + LendingRecord::SECURED),
            read(input, base + LendingRecord::EQUITY),
            input[base + LendingRecord::MINIMUM_CAPITAL],
            read(input, base + LendingRecord::RISK_ASSETS),
            input[base + LendingRecord::RISK_WEIGHT],
            read(input, base + LendingRecord::LIQUIDITY),
            &mut faults,
        );
        faults.record(control);
        store(output, id, value);
    }
}

#[cube(launch)]
fn funding(
    plans: &[u32],
    flows: &mut [u32],
    output: &mut [u32],
    control: &[Atomic<u32>],
    count: u32,
) {
    let id = ABSOLUTE_POS as u32;
    if id < count {
        let base = id as usize * PlanRecord::WORDS;
        let first = plans[base + PlanRecord::FIRST];
        let end = first + plans[base + PlanRecord::COUNT];
        let mut outflows = zero();
        let mut receipts = zero();
        let mut need = read(plans, base + PlanRecord::CUSHION);
        for index in first..end {
            let outgoing = plus(outflows, load(flows, index * 3));
            let incoming = plus(receipts, load(flows, index * 3 + 1));
            outgoing.faults.union(incoming.faults).record(control);
            outflows = outgoing.value;
            receipts = incoming.value;
            let required = plus(outflows, load(flows, index * 3 + 2));
            required.faults.record(control);
            need = maximum(need, positive(required.value, receipts));
        }
        store(output, id * 2, need);
        store(
            output,
            id * 2 + 1,
            positive(need, read(plans, base + PlanRecord::CASH)),
        );
    }
}

#[cube(launch)]
fn pools(
    plans: &[u32],
    demands: &mut [u32],
    output: &mut [u32],
    remaining: &mut [u32],
    count: u32,
) {
    let id = ABSOLUTE_POS as u32;
    if id < count {
        let base = id as usize * PlanRecord::WORDS;
        let first = plans[base + PlanRecord::FIRST];
        let end = first + plans[base + PlanRecord::COUNT];
        let mut capacity = read(plans, base + PlanRecord::CASH);
        for index in first..end {
            let fill = minimum(capacity, load(demands, index));
            store(output, index, fill);
            capacity = minus(capacity, fill);
        }
        store(remaining, id, capacity);
    }
}

impl FinancePolicies {
    pub fn lending(device: &Device, inputs: &[LendingInput]) -> Result<Vec<LendingResult>> {
        let admitted = inputs
            .iter()
            .cloned()
            .map(LendingInput::new)
            .collect::<Result<Vec<_>>>()?;
        let records = admitted
            .iter()
            .map(|input| LendingRecord {
                requested: input.requested.into(),
                term: input.term,
                rate: input.rate,
                surplus: input.surplus.into(),
                debt_service: input.debt_service.into(),
                service_fraction: input.service_fraction,
                advance: input.advance,
                collateral: input.collateral.into(),
                secured: input.secured.into(),
                equity: input.equity.into(),
                minimum_capital: input.minimum_capital,
                risk_assets: input.risk_assets.into(),
                risk_weight: input.risk_weight,
                liquidity: input.liquidity.into(),
            })
            .collect::<Vec<_>>();
        let input = padded(device, bytemuck::cast_slice(&records));
        let output = device.empty(inputs.len().max(1) * TICK_WORDS);
        let control = device.tensor(&[0; CONTROL_WORDS]);
        lending::launch(
            &device.client,
            count(device, inputs.len()),
            CubeDim::new_1d(device.workgroup),
            argument(&input),
            argument(&output),
            argument(&control),
            inputs.len() as u32,
        );
        FaultFlags::read(&device.read(&control)?)?.into_result()?;
        Ok(read_ticks(device, &output, inputs.len())?
            .into_iter()
            .map(|amount| LendingResult { amount })
            .collect())
    }

    pub fn funding(device: &Device, inputs: &[FundingPlan]) -> Result<Vec<FundingResult>> {
        let mut records = Vec::new();
        let mut flows = Vec::<crate::ledger::TickLayout>::new();
        for plan in inputs {
            match plan.flows.windows(2).all(|pair| pair[0].day <= pair[1].day) {
                true => (),
                false => Err(anyhow!("funding flows must be dated in execution order"))?,
            }
            records.push(PlanRecord {
                first: (flows.len() / 3) as u32,
                count: plan.flows.len() as u32,
                cash: plan.opening_cash.into(),
                cushion: plan.opening_cushion.into(),
            });
            flows.extend(plan.flows.iter().flat_map(|flow| {
                [flow.outflow, flow.receipt, flow.cushion].map(crate::ledger::TickLayout::from)
            }));
        }
        let records = padded(device, bytemuck::cast_slice(&records));
        let flows = padded(device, bytemuck::cast_slice(&flows));
        let output = device.empty(inputs.len().max(1) * 2 * TICK_WORDS);
        let control = device.tensor(&[0; CONTROL_WORDS]);
        funding::launch(
            &device.client,
            count(device, inputs.len()),
            CubeDim::new_1d(device.workgroup),
            argument(&records),
            argument(&flows),
            argument(&output),
            argument(&control),
            inputs.len() as u32,
        );
        FaultFlags::read(&device.read(&control)?)?.into_result()?;
        Ok(read_ticks(device, &output, inputs.len() * 2)?
            .chunks_exact(2)
            .map(|row| FundingResult {
                opening_need: row[0],
                gap: row[1],
            })
            .collect())
    }

    pub fn pools(device: &Device, inputs: &[DivisiblePool]) -> Result<Vec<PoolResult>> {
        let mut records = Vec::new();
        let mut demands = Vec::<crate::ledger::TickLayout>::new();
        for pool in inputs {
            records.push(PlanRecord {
                first: demands.len() as u32,
                count: pool.demands.len() as u32,
                cash: pool.capacity.into(),
                cushion: 0.into(),
            });
            demands.extend(
                pool.demands
                    .iter()
                    .copied()
                    .map(crate::ledger::TickLayout::from),
            );
        }
        let metadata = padded(device, bytemuck::cast_slice(&records));
        let input = padded(device, bytemuck::cast_slice(&demands));
        let output = device.empty(demands.len().max(1) * TICK_WORDS);
        let remaining = device.empty(inputs.len().max(1) * TICK_WORDS);
        pools::launch(
            &device.client,
            count(device, inputs.len()),
            CubeDim::new_1d(device.workgroup),
            argument(&metadata),
            argument(&input),
            argument(&output),
            argument(&remaining),
            inputs.len() as u32,
        );
        let filled = read_ticks(device, &output, demands.len())?;
        let remainder = read_ticks(device, &remaining, inputs.len())?;
        Ok(records
            .iter()
            .zip(remainder)
            .map(|(record, remaining)| PoolResult {
                fills: filled[record.first as usize..(record.first + record.count) as usize]
                    .to_vec(),
                remaining,
            })
            .collect())
    }
}

fn padded(device: &Device, words: &[u32]) -> CubeTensor {
    device.tensor(
        &words
            .iter()
            .copied()
            .chain(std::iter::once(0))
            .collect::<Vec<_>>(),
    )
}

fn read_ticks(device: &Device, tensor: &CubeTensor, length: usize) -> Result<Vec<u64>> {
    match length {
        0 => Ok(Vec::new()),
        _ => Ok(device
            .read_range(tensor, 0, length * TICK_WORDS)?
            .chunks_exact(TICK_WORDS)
            .map(|words| u64::from(words[0]) | (u64::from(words[1]) << 32))
            .collect()),
    }
}

#[derive(Clone, Debug)]
pub struct BudgetPolicy {
    pub replacement_target: u64,
    pub earmarked: u64,
    pub free_cash: u64,
    pub rent_fraction: u32,
    pub before_rent: u64,
    pub retained: u64,
    pub rent_budget: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BudgetResult {
    pub earmark: u64,
    pub rent: u64,
}

#[derive(Clone, Debug)]
pub struct DiscountFlow {
    pub receipt: u64,
    pub outflow: u64,
}

#[derive(Clone, Debug)]
pub struct DiscountPlan {
    pub daily_rate: u32,
    pub flows: Vec<DiscountFlow>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SignedAmount {
    Negative(u64),
    Zero,
    Positive(u64),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DiscountedValue {
    pub lower: SignedAmount,
    pub upper: SignedAmount,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BudgetRecord {
    target: crate::ledger::TickLayout,
    earmarked: crate::ledger::TickLayout,
    free_cash: crate::ledger::TickLayout,
    fraction: u32,
    before_rent: crate::ledger::TickLayout,
    retained: crate::ledger::TickLayout,
    budget: crate::ledger::TickLayout,
}
impl BudgetRecord {
    const WORDS: usize = size_of::<Self>() / WORD_BYTES;
    const TARGET: usize = offset_of!(Self, target) / WORD_BYTES;
    const EARMARKED: usize = offset_of!(Self, earmarked) / WORD_BYTES;
    const FREE_CASH: usize = offset_of!(Self, free_cash) / WORD_BYTES;
    const FRACTION: usize = offset_of!(Self, fraction) / WORD_BYTES;
    const BEFORE_RENT: usize = offset_of!(Self, before_rent) / WORD_BYTES;
    const RETAINED: usize = offset_of!(Self, retained) / WORD_BYTES;
    const BUDGET: usize = offset_of!(Self, budget) / WORD_BYTES;
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DiscountRecord {
    first: u32,
    count: u32,
    rate: u32,
}
impl DiscountRecord {
    const WORDS: usize = size_of::<Self>() / WORD_BYTES;
    const FIRST: usize = offset_of!(Self, first) / WORD_BYTES;
    const COUNT: usize = offset_of!(Self, count) / WORD_BYTES;
    const RATE: usize = offset_of!(Self, rate) / WORD_BYTES;
}

const DISCOUNT_SCALE: u32 = 1_000_000;

#[cube(launch)]
fn budgets(input: &[u32], output: &mut [u32], control: &[Atomic<u32>], count: u32) {
    let id = ABSOLUTE_POS as u32;
    if id < count {
        let base = id as usize * BudgetRecord::WORDS;
        let earmark = minimum(
            positive(
                read(input, base + BudgetRecord::TARGET),
                read(input, base + BudgetRecord::EARMARKED),
            ),
            read(input, base + BudgetRecord::FREE_CASH),
        );
        let rent_base = minimum(
            positive(
                read(input, base + BudgetRecord::BEFORE_RENT),
                read(input, base + BudgetRecord::RETAINED),
            ),
            read(input, base + BudgetRecord::BUDGET),
        );
        let rent = ratio(
            rent_base,
            input[base + BudgetRecord::FRACTION],
            FINANCE_SCALE,
        );
        rent.faults.record(control);
        store(output, id * 2, earmark);
        store(output, id * 2 + 1, rent.value);
    }
}

#[cube]
fn discount_ceil(
    amount: Ticks,
    numerator: u32,
    denominator: u32,
    control: &[Atomic<u32>],
) -> Ticks {
    let quotient = ratio(amount, numerator, denominator);
    let remaining = minus(
        amount,
        times(divide(amount, denominator), denominator).value,
    );
    let product = times32(remaining.extract(0usize), numerator);
    let fractional = minus(
        product,
        times(divide(product, denominator), denominator).value,
    );
    let result = plus(
        quotient.value,
        ticks(u32::cast_from(!equal(fractional, zero())), 0),
    );
    quotient.faults.union(result.faults).record(control);
    result.value
}

#[cube]
fn signed_difference(output: &mut [u32], base: u32, credit: Ticks, debit: Ticks) {
    let negative = lt(credit, debit);
    let amount = select(negative, minus(debit, credit), minus(credit, debit));
    store(output, base, ticks(u32::cast_from(negative), 0));
    store(output, base + 1, amount);
}

#[cube(launch)]
fn discounts(
    plans: &[u32],
    flows: &mut [u32],
    output: &mut [u32],
    control: &[Atomic<u32>],
    count: u32,
) {
    let id = ABSOLUTE_POS as u32;
    if id < count {
        let base = id as usize * DiscountRecord::WORDS;
        let first = plans[base + DiscountRecord::FIRST];
        let end = first + plans[base + DiscountRecord::COUNT];
        let denominator = FINANCE_SCALE + plans[base + DiscountRecord::RATE];
        let mut lower_weight = u32::new(comptime!(i64::from(DISCOUNT_SCALE)));
        let mut upper_weight = u32::new(comptime!(i64::from(DISCOUNT_SCALE)));
        let mut receipt_lower = zero();
        let mut receipt_upper = zero();
        let mut outflow_lower = zero();
        let mut outflow_upper = zero();
        for index in first..end {
            if index > first {
                lower_weight =
                    divide(times32(lower_weight, FINANCE_SCALE), denominator).extract(0usize);
                upper_weight =
                    discount_ceil(ticks(upper_weight, 0), FINANCE_SCALE, denominator, control)
                        .extract(0usize);
            }
            let receipt = load(flows, index * 2);
            let outflow = load(flows, index * 2 + 1);
            let r_lower = ratio(receipt, lower_weight, DISCOUNT_SCALE);
            let o_lower = ratio(outflow, lower_weight, DISCOUNT_SCALE);
            let r_upper = discount_ceil(receipt, upper_weight, DISCOUNT_SCALE, control);
            let o_upper = discount_ceil(outflow, upper_weight, DISCOUNT_SCALE, control);
            let next_r_lower = plus(receipt_lower, r_lower.value);
            let next_r_upper = plus(receipt_upper, r_upper);
            let next_o_lower = plus(outflow_lower, o_lower.value);
            let next_o_upper = plus(outflow_upper, o_upper);
            r_lower
                .faults
                .union(o_lower.faults)
                .union(next_r_lower.faults)
                .union(next_r_upper.faults)
                .union(next_o_lower.faults)
                .union(next_o_upper.faults)
                .record(control);
            receipt_lower = next_r_lower.value;
            receipt_upper = next_r_upper.value;
            outflow_lower = next_o_lower.value;
            outflow_upper = next_o_upper.value;
        }
        signed_difference(output, id * 4, receipt_lower, outflow_upper);
        signed_difference(output, id * 4 + 2, receipt_upper, outflow_lower);
    }
}

impl FinancePolicies {
    pub fn budgets(device: &Device, inputs: &[BudgetPolicy]) -> Result<Vec<BudgetResult>> {
        match inputs
            .iter()
            .all(|policy| policy.rent_fraction <= FINANCE_SCALE)
        {
            true => Ok(()),
            false => Err(anyhow!("rent fraction exceeds one")),
        }?;
        let records = inputs
            .iter()
            .map(|policy| BudgetRecord {
                target: policy.replacement_target.into(),
                earmarked: policy.earmarked.into(),
                free_cash: policy.free_cash.into(),
                fraction: policy.rent_fraction,
                before_rent: policy.before_rent.into(),
                retained: policy.retained.into(),
                budget: policy.rent_budget.into(),
            })
            .collect::<Vec<_>>();
        let input = padded(device, bytemuck::cast_slice(&records));
        let output = device.empty(inputs.len().max(1) * 2 * TICK_WORDS);
        let control = device.tensor(&[0; CONTROL_WORDS]);
        budgets::launch(
            &device.client,
            count(device, inputs.len()),
            CubeDim::new_1d(device.workgroup),
            argument(&input),
            argument(&output),
            argument(&control),
            inputs.len() as u32,
        );
        FaultFlags::read(&device.read(&control)?)?.into_result()?;
        Ok(read_ticks(device, &output, inputs.len() * 2)?
            .chunks_exact(2)
            .map(|row| BudgetResult {
                earmark: row[0],
                rent: row[1],
            })
            .collect())
    }

    pub fn discounted_flows(
        device: &Device,
        inputs: &[DiscountPlan],
    ) -> Result<Vec<DiscountedValue>> {
        match inputs.iter().all(|plan| plan.daily_rate <= FINANCE_SCALE) {
            true => Ok(()),
            false => Err(anyhow!("discount rate exceeds fixed-point domain")),
        }?;
        let mut records = Vec::new();
        let mut flows = Vec::<crate::ledger::TickLayout>::new();
        for plan in inputs {
            records.push(DiscountRecord {
                first: (flows.len() / 2) as u32,
                count: plan.flows.len() as u32,
                rate: plan.daily_rate,
            });
            flows.extend(plan.flows.iter().flat_map(|flow| {
                [flow.receipt, flow.outflow].map(crate::ledger::TickLayout::from)
            }));
        }
        let records = padded(device, bytemuck::cast_slice(&records));
        let flows = padded(device, bytemuck::cast_slice(&flows));
        let output = device.empty(inputs.len().max(1) * 4 * TICK_WORDS);
        let control = device.tensor(&[0; CONTROL_WORDS]);
        discounts::launch(
            &device.client,
            count(device, inputs.len()),
            CubeDim::new_1d(device.workgroup),
            argument(&records),
            argument(&flows),
            argument(&output),
            argument(&control),
            inputs.len() as u32,
        );
        FaultFlags::read(&device.read(&control)?)?.into_result()?;
        let signed = |sign, magnitude| match (sign, magnitude) {
            (_, 0) => SignedAmount::Zero,
            (0, amount) => SignedAmount::Positive(amount),
            (_, amount) => SignedAmount::Negative(amount),
        };
        Ok(read_ticks(device, &output, inputs.len() * 4)?
            .chunks_exact(4)
            .map(|row| DiscountedValue {
                lower: signed(row[0], row[1]),
                upper: signed(row[2], row[3]),
            })
            .collect())
    }
}

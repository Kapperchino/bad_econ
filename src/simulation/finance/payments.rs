use super::*;
use crate::{
    ledger::{Decision, EMPTY, LEG_WORDS, LegLayout, MAX_LEGS, TickLayout},
    simulation::{
        Trace,
        flags::ControlLayout,
        layout::{EVENT_WORDS, LinkLayout, MIN_BUFFER_WORDS, ScratchLayout},
        ops::SchedulerBuffers,
    },
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BankAccount {
    pub deposit: Coordinate,
    pub reserve: Coordinate,
}

#[derive(Clone, Copy, Debug)]
pub struct BankPayment {
    pub identity: u64,
    pub payer: BankAccount,
    pub payee: BankAccount,
    pub amount: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PaymentRecord {
    payer: u32,
    payee: u32,
    payer_reserve: u32,
    payee_reserve: u32,
    amount: TickLayout,
}
impl PaymentRecord {
    const PAYER: usize = offset_of!(Self, payer) / WORD_BYTES;
    const PAYEE: usize = offset_of!(Self, payee) / WORD_BYTES;
    const PAYER_RESERVE: usize = offset_of!(Self, payer_reserve) / WORD_BYTES;
    const PAYEE_RESERVE: usize = offset_of!(Self, payee_reserve) / WORD_BYTES;
    const AMOUNT: usize = offset_of!(Self, amount) / WORD_BYTES;
    const WORDS: usize = size_of::<Self>() / WORD_BYTES;
}

#[derive(CubeType, Clone, Copy)]
struct PaymentLeg {
    coordinate: u32,
    debit: Ticks,
    credit: Ticks,
}

#[cube]
fn post_leg(output: &mut [u32], base: usize, leg: PaymentLeg) {
    output[base + LegLayout::COORDINATE] = leg.coordinate;
    output[base + LegLayout::DEBIT_LOW] = leg.debit.extract(0usize);
    output[base + LegLayout::DEBIT_HIGH] = leg.debit.extract(1usize);
    output[base + LegLayout::CREDIT_LOW] = leg.credit.extract(0usize);
    output[base + LegLayout::CREDIT_HIGH] = leg.credit.extract(1usize);
}

#[cube(launch)]
fn construct(input: &[u32], output: &mut [u32], count: u32) {
    let id = ABSOLUTE_POS as u32;
    if id < count {
        let base = id as usize * PaymentRecord::WORDS;
        let event = id as usize * EVENT_WORDS;
        for leg in 0..MAX_LEGS {
            for word in 0..LEG_WORDS {
                output[event + leg * LEG_WORDS + word] =
                    select(word == LegLayout::COORDINATE, EMPTY, 0);
            }
        }
        let payer = input[base + PaymentRecord::PAYER];
        let payee = input[base + PaymentRecord::PAYEE];
        let sender = input[base + PaymentRecord::PAYER_RESERVE];
        let receiver = input[base + PaymentRecord::PAYEE_RESERVE];
        let amount = ticks(
            input[base + PaymentRecord::AMOUNT],
            input[base + PaymentRecord::AMOUNT + 1],
        );
        post_leg(
            output,
            event,
            PaymentLeg {
                coordinate: payer,
                debit: amount,
                credit: select(payer == payee, amount, zero()),
            },
        );
        if payer != payee {
            post_leg(
                output,
                event + LEG_WORDS,
                PaymentLeg {
                    coordinate: payee,
                    debit: zero(),
                    credit: amount,
                },
            );
        }
        if sender != receiver {
            post_leg(
                output,
                event + 2 * LEG_WORDS,
                PaymentLeg {
                    coordinate: sender,
                    debit: amount,
                    credit: zero(),
                },
            );
            post_leg(
                output,
                event + 3 * LEG_WORDS,
                PaymentLeg {
                    coordinate: receiver,
                    debit: zero(),
                    credit: amount,
                },
            );
        }
    }
}

pub fn bank_payments(
    device: &Device,
    balances: &[u64],
    payments: &[BankPayment],
) -> Result<crate::simulation::LedgerResult> {
    let values = balances
        .iter()
        .copied()
        .map(TickLayout::from)
        .collect::<Vec<_>>();
    let state = device.tensor(bytemuck::cast_slice(&values));
    let decisions = post_bank_payments(device, &state, payments)?;
    let updated = device
        .read(&state)?
        .chunks_exact(TICK_WORDS)
        .map(|words| u64::from(words[0]) | (u64::from(words[1]) << 32))
        .collect();
    Ok(crate::simulation::LedgerResult {
        state: updated,
        decisions: decisions.decisions,
        frontier_depth: decisions.frontier_depth,
    })
}

pub struct BankPaymentResult {
    pub decisions: Vec<Decision>,
    pub frontier_depth: usize,
    pub readback_bytes: u64,
}

pub fn post_bank_payments(
    device: &Device,
    state: &CubeTensor,
    payments: &[BankPayment],
) -> Result<BankPaymentResult> {
    let mut accounts = BTreeMap::new();
    for account in payments
        .iter()
        .flat_map(|payment| [payment.payer, payment.payee])
    {
        match accounts.insert(account.deposit, account.reserve) {
            Some(prior) if prior != account.reserve => {
                Err(anyhow!("deposit account belongs to conflicting banks"))?
            }
            _ => (),
        }
    }
    let distinct_instruments = accounts
        .values()
        .all(|reserve| !accounts.contains_key(reserve));
    let bounded = accounts.iter().all(|(deposit, reserve)| {
        [deposit, reserve]
            .into_iter()
            .all(|coordinate| (coordinate.0 as usize) < state.meta.shape()[0] / TICK_WORDS)
    });
    let ordered = payments
        .windows(2)
        .all(|pair| pair[0].identity < pair[1].identity)
        && payments.iter().all(|payment| payment.identity > 0);
    match distinct_instruments && bounded && ordered {
        true => Ok(()),
        false => Err(anyhow!(
            "bank payments require disjoint deposit/reserve instruments and canonical unique identities"
        )),
    }?;
    let records = payments
        .iter()
        .map(|payment| PaymentRecord {
            payer: payment.payer.deposit.0,
            payee: payment.payee.deposit.0,
            payer_reserve: payment.payer.reserve.0,
            payee_reserve: payment.payee.reserve.0,
            amount: payment.amount.into(),
        })
        .collect::<Vec<_>>();
    let input = device.tensor(
        &bytemuck::cast_slice(&records)
            .iter()
            .copied()
            .chain(std::iter::once(0))
            .collect::<Vec<_>>(),
    );
    let events = device.empty((payments.len() * EVENT_WORDS).max(MIN_BUFFER_WORDS));
    let scratch =
        device.empty((payments.len() * ScratchLayout::WORDS_PER_EVENT).max(MIN_BUFFER_WORDS));
    let links = device.empty((payments.len() * LinkLayout::WORDS_PER_EVENT).max(MIN_BUFFER_WORDS));
    let control = device.tensor(bytemuck::cast_slice(&[ControlLayout::zeroed()]));
    construct::launch(
        &device.client,
        count(device, payments.len()),
        CubeDim::new_1d(device.workgroup),
        argument(&input),
        argument(&events),
        payments.len() as u32,
    );
    let result = crate::simulation::execute(
        device,
        SchedulerBuffers {
            state,
            events: &events,
            scratch: &scratch,
            control: &control,
            links: &links,
        },
        payments.len(),
        Trace::Record,
    )?;
    Ok(BankPaymentResult {
        decisions: result.decisions,
        frontier_depth: result.depth,
        readback_bytes: result.readback_bytes,
    })
}

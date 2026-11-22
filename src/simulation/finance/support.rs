use super::*;
use crate::simulation::{
    flags::CONSTRUCTED_EVENTS_WORD,
    layout::{HIGH_WORD, LOW_WORD, TICK_BITS, WORD_BITS, WORD_TOP_BIT},
    state::BalanceBuffer,
};

#[derive(CubeType)]
struct SupportRecipients<'a> {
    records: &'a [u32],
    count: u32,
}

#[cube]
impl<'a> SupportRecipients<'a> {
    fn new(records: &'a [u32], count: u32) -> Self {
        SupportRecipients::<'a> { records, count }
    }

    fn row(&self, index: u32) -> RowIndex {
        RowIndex::new(0, self.count as usize, index as usize)
    }

    fn cash(&self, id: u32) -> u32 {
        self.records[self.row(id).field(SupportMeta::CASH, 1usize)]
    }

    fn need(&self, balances: &BalanceBuffer, id: u32) -> Ticks {
        let row = self.row(id);
        let need = positive(
            read_tick_words(self.records, row.field(SupportMeta::TARGET, TICK_WORDS)),
            balances.value(self.cash(id)),
        );
        select(
            self.records[row.field(SupportMeta::CONSENT, 1usize)] == 1,
            need,
            zero(),
        )
    }
}

#[cube]
pub(super) fn needs(input: &mut SupportNeedsInput) {
    let id = ABSOLUTE_POS as u32;
    if id < input.count {
        let recipients = SupportRecipients::new(&input.recipients, input.count);
        let balances = BalanceBuffer::new(&mut input.state);
        store(&mut input.output, id, recipients.need(&balances, id));
    }
}

#[cube]
fn bounded_product_ratio(a: Ticks, b: Ticks, denominator: Ticks) -> Ticks {
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

#[cube]
pub(super) fn plan(input: &mut SupportPlanInput) {
    let id = ABSOLUTE_POS as u32;
    let balances = BalanceBuffer::new(&mut input.state);
    let available = balances.value(input.treasury);
    let mut total = zero();
    if input.count > 0 {
        total = read_ticks(&input.scanned, input.count - 1);
    }
    let capacity = minimum(available, total);
    if id < input.count {
        let cumulative = read_ticks(&input.scanned, id);
        let mut previous = zero();
        if id > 0 {
            previous = read_ticks(&input.scanned, id - 1);
        }
        let divisor = maximum(total, ticks(1, 0));
        let grant = minus(
            bounded_product_ratio(capacity, cumulative, divisor),
            bounded_product_ratio(capacity, previous, divisor),
        );
        let recipients = SupportRecipients::new(&input.recipients, input.count);
        let new_cash = plus(balances.value(recipients.cash(id)), grant);
        new_cash.faults.record(&input.control);
        if !equal(grant, zero()) {
            input.control[CONSTRUCTED_EVENTS_WORD].fetch_add(1);
        }
        store(&mut input.output, id, new_cash.value);
    }
    if id == input.count {
        store(&mut input.output, input.count, minus(available, capacity));
    }
}

#[cube]
pub(super) fn post(input: &mut SupportPostInput) {
    let id = ABSOLUTE_POS as u32;
    let mut balances = BalanceBuffer::new(&mut input.state);
    if id < input.count {
        let recipients = SupportRecipients::new(&input.recipients, input.count);
        balances.set(recipients.cash(id), read_ticks(&input.output, id));
    }
    if id == input.count {
        balances.set(input.treasury, read_ticks(&input.output, input.count));
    }
}

use super::{kernels::ResearchInput, layout::*};
use crate::{
    domain::{Field, Good},
    simulation::{
        flags::{FaultFlags, KernelFaults},
        layout::TICK_WORDS,
        rows::RowIndex,
        state::{ActorBalances, ReadLedger, ReadLedgerExpand},
        words::*,
    },
};
use cubecl::prelude::*;

#[derive(CubeType)]
pub(super) struct ResearchView<'a> {
    pub main: ActorBalances<'a>,
    pub state: ResearchState<'a>,
    pub parameters: &'a ResearchParameters,
    pub records: ResearchRecords<'a>,
    pub table: &'a [u32],
    pub samples: &'a [f32],
    pub output: &'a mut [u32],
    pub control: &'a [Atomic<u32>],
}

#[cube]
impl<'a> ResearchView<'a> {
    pub fn new(input: &'a mut ResearchInput) -> Self {
        ResearchView::<'a> {
            main: ActorBalances::new(&mut input.main, input.config.parameters.state_stride),
            state: ResearchState::new(&mut input.state),
            parameters: &input.config.parameters,
            records: ResearchRecords::new(&input.config.records),
            table: &input.table,
            samples: &input.samples,
            output: &mut input.output,
            control: &input.control,
        }
    }

    pub fn cap_plan(&mut self, firm: u32, capacity: Ticks) {
        let capped = minimum(self.main.value(Field::Plan, firm), capacity);
        self.main.set(Field::Plan, firm, capped);
        self.main.set(Field::Slots, firm, capped);
    }

    pub fn executed_packets(&self, firm: u32) -> Ticks {
        let plan = self.main.value(Field::Plan, firm);
        let slots = self.main.value(Field::Slots, firm);
        if lt(plan, slots) {
            self.record_mismatch();
        }
        positive(plan, slots)
    }

    pub fn record_mismatch(&self) {
        KernelFaults::new(comptime!(FaultFlags::RESEARCH_MISMATCH)).record(self.control);
    }
}

#[derive(CubeType)]
pub(super) struct ResearchRow {
    pub definition: RowIndex,
    pub saved: RowIndex,
}

#[cube]
impl ResearchRow {
    fn new(definition: u32, state: u32, count: u32, id: usize) -> Self {
        ResearchRow {
            definition: RowIndex::new(definition as usize, count as usize, id),
            saved: RowIndex::new(state as usize, count as usize, id),
        }
    }
}

#[cube]
impl ResearchParameters {
    pub fn lab(&self, id: usize) -> ResearchRow {
        ResearchRow::new(self.labs, self.lab_state, self.lab_count, id)
    }

    pub fn project(&self, id: usize) -> ResearchRow {
        ResearchRow::new(self.projects, self.project_state, self.project_count, id)
    }

    pub fn machine(&self, id: usize) -> ResearchRow {
        ResearchRow::new(self.machines, self.machine_state, self.machine_count, id)
    }

    pub fn stock(&self, id: usize) -> ResearchRow {
        ResearchRow::new(self.stocks, self.stock_state, self.stock_count, id)
    }

    pub fn strike(&self, id: usize) -> ResearchRow {
        ResearchRow::new(self.strikes, self.strike_state, self.strike_count, id)
    }

    pub fn method(&self, id: u32) -> RowIndex {
        RowIndex::new(
            self.methods as usize,
            self.method_count as usize,
            id as usize,
        )
    }
}

#[derive(CubeType)]
pub(super) struct ResearchRecords<'a> {
    words: &'a [u32],
}

#[cube]
impl<'a> ResearchRecords<'a> {
    pub fn new(words: &'a [u32]) -> Self {
        ResearchRecords::<'a> { words }
    }

    pub fn number(&self, row: &RowIndex, #[comptime] field: usize) -> u32 {
        self.words[row.field(field, 1usize)]
    }

    pub fn amount(&self, row: &RowIndex, #[comptime] field: usize) -> Ticks {
        read_tick_words(self.words, row.field(field, TICK_WORDS))
    }

    pub fn project_input(&self, project: &RowIndex, good: usize) -> u32 {
        self.words[project.field(ProjectLayout::INPUTS, Good::COUNT) + good]
    }
}

#[derive(CubeType)]
pub(super) struct ResearchState<'a> {
    words: &'a mut [u32],
}

#[cube]
impl<'a> ResearchState<'a> {
    pub fn new(words: &'a mut [u32]) -> Self {
        ResearchState::<'a> { words }
    }

    pub fn number(&self, row: &RowIndex, #[comptime] field: usize) -> u32 {
        self.words[row.field(field, 1usize)]
    }

    pub fn set_number(&mut self, row: &RowIndex, #[comptime] field: usize, value: u32) {
        self.words[row.field(field, 1usize)] = value;
    }

    pub fn amount(&self, row: &RowIndex, #[comptime] field: usize) -> Ticks {
        read_tick_words(&*self.words, row.field(field, TICK_WORDS))
    }

    pub fn set_amount(&mut self, row: &RowIndex, #[comptime] field: usize, value: Ticks) {
        write_tick_words(self.words, row.field(field, TICK_WORDS), value);
    }

    pub fn increase(
        &mut self,
        row: &RowIndex,
        #[comptime] field: usize,
        value: Ticks,
        control: &[Atomic<u32>],
    ) {
        let sum = plus(self.amount(row, field), value).record(control);
        self.set_amount(row, field, sum);
    }

    pub fn lab_consumed(&self, lab: &RowIndex, good: usize) -> Ticks {
        let word = lab.field(LabState::CONSUMED, Good::COUNT * TICK_WORDS) + good * TICK_WORDS;
        read_tick_words(&*self.words, word)
    }

    pub fn consume(&mut self, lab: &RowIndex, good: usize, used: Ticks, control: &[Atomic<u32>]) {
        let word = lab.field(LabState::CONSUMED, Good::COUNT * TICK_WORDS) + good * TICK_WORDS;
        let sum = plus(read_tick_words(&*self.words, word), used).record(control);
        write_tick_words(self.words, word, sum);
    }

    pub fn lab_access(&self, lab: &RowIndex) -> u32 {
        self.number(lab, LabState::DISCOVERED)
            & self.number(lab, LabState::MASTERED)
            & self.number(lab, LabState::PERMITTED)
            & self.number(lab, LabState::ADOPTED)
    }

    pub fn adopt(&mut self, lab: &RowIndex) {
        let mastered =
            self.number(lab, LabState::MASTERED) | self.number(lab, LabState::DISCOVERED);
        let adopted = self.number(lab, LabState::ADOPTED)
            | (mastered & self.number(lab, LabState::PERMITTED));
        self.set_number(lab, LabState::MASTERED, mastered);
        self.set_number(lab, LabState::ADOPTED, adopted);
    }
}

use super::{
    configuration::EconomyConfig,
    flags::KernelActorKind,
    layout::{ActorLayout, LOW_WORD},
    words::*,
};
use crate::domain::Field;
use cubecl::prelude::*;

#[derive(CubeType)]
pub(super) struct BalanceBuffer<'a> {
    words: &'a mut [u32],
}

#[cube]
impl<'a> BalanceBuffer<'a> {
    pub fn new(words: &'a mut [u32]) -> Self {
        BalanceBuffer::<'a> { words }
    }

    pub fn value(&self, coordinate: u32) -> Ticks {
        read_ticks(&*self.words, coordinate)
    }

    pub fn set(&mut self, coordinate: u32, value: Ticks) {
        store(self.words, coordinate, value);
    }
}

#[derive(CubeType)]
pub(super) struct ActorBalances<'a> {
    balances: BalanceBuffer<'a>,
    stride: u32,
}

#[cube]
impl<'a> ActorBalances<'a> {
    pub fn new(words: &'a mut [u32], stride: u32) -> Self {
        ActorBalances::<'a> {
            balances: BalanceBuffer::new(words),
            stride,
        }
    }

    pub fn set_at(&mut self, field: u32, actor: u32, value: Ticks) {
        self.balances.set(field * self.stride + actor, value);
    }

    pub fn set(&mut self, #[comptime] field: Field, actor: u32, value: Ticks) {
        self.set_at(comptime!(field as u32), actor, value);
    }

    pub fn set_number(&mut self, #[comptime] field: Field, actor: u32, value: u32) {
        self.set(field, actor, ticks(value, 0));
    }

    pub fn increase(
        &mut self,
        control: &[Atomic<u32>],
        #[comptime] field: Field,
        actor: u32,
        amount: Ticks,
    ) {
        let total = plus(self.value(field, actor), amount).record(control);
        self.set(field, actor, total);
    }
}

#[cube]
impl<'a> ReadLedger for ActorBalances<'a> {
    fn value_at(&self, field: u32, actor: u32) -> Ticks {
        self.balances.value(field * self.stride + actor)
    }

    fn number_at(&self, field: u32, actor: u32) -> u32 {
        self.value_at(field, actor).extract(LOW_WORD)
    }

    fn value(&self, #[comptime] field: Field, actor: u32) -> Ticks {
        self.value_at(comptime!(field as u32), actor)
    }

    fn number(&self, #[comptime] field: Field, actor: u32) -> u32 {
        self.value(field, actor).extract(LOW_WORD)
    }
}

#[derive(CubeType)]
pub(super) struct EconomyView<'a> {
    pub ledger: Ledger<'a>,
    pub actors: ActorTable<'a>,
}

#[cube]
impl<'a> EconomyView<'a> {
    pub fn new(words: &'a mut [u32], actors: &'a [u32], config: &'a EconomyConfig) -> Self {
        EconomyView::<'a> {
            ledger: Ledger::new(words, config),
            actors: ActorTable::new(actors),
        }
    }
}

#[derive(CubeType)]
pub(super) struct Ledger<'a> {
    pub words: &'a mut [u32],
    pub config: &'a EconomyConfig,
}

#[cube]
impl<'a> Ledger<'a> {
    pub fn new(words: &'a mut [u32], config: &'a EconomyConfig) -> Self {
        Ledger::<'a> { words, config }
    }

    pub fn set_at(&mut self, field: u32, actor: u32, amount: Ticks) {
        store(self.words, self.config.coordinate(field, actor), amount);
    }

    pub fn set(&mut self, #[comptime] field: Field, actor: u32, amount: Ticks) {
        self.set_at(comptime!(field as u32), actor, amount);
    }

    pub fn set_number_at(&mut self, field: u32, actor: u32, amount: u32) {
        self.set_at(field, actor, ticks(amount, 0));
    }

    pub fn set_number(&mut self, #[comptime] field: Field, actor: u32, amount: u32) {
        self.set_number_at(comptime!(field as u32), actor, amount);
    }

    pub fn increase_at(&mut self, control: &[Atomic<u32>], field: u32, actor: u32, amount: Ticks) {
        let total = plus(self.value_at(field, actor), amount).record(control);
        self.set_at(field, actor, total);
    }

    pub fn increase(
        &mut self,
        control: &[Atomic<u32>],
        #[comptime] field: Field,
        actor: u32,
        amount: Ticks,
    ) {
        self.increase_at(control, comptime!(field as u32), actor, amount);
    }
}

#[derive(CubeType)]
pub(super) struct LedgerColumn<'a> {
    words: &'a [u32],
    start: u32,
}

#[cube]
impl<'a> LedgerColumn<'a> {
    pub fn new(words: &'a [u32], config: &EconomyConfig, #[comptime] field: Field) -> Self {
        LedgerColumn::<'a> {
            words,
            start: config.coordinate(comptime!(field as u32), 0),
        }
    }

    pub fn number(&self, actor: u32) -> u32 {
        read_ticks(self.words, self.start + actor).extract(LOW_WORD)
    }
}

#[cube]
pub(super) trait ReadLedger: CubeType {
    fn value_at(&self, field: u32, actor: u32) -> Ticks;
    fn number_at(&self, field: u32, actor: u32) -> u32;
    fn value(&self, #[comptime] field: Field, actor: u32) -> Ticks;
    fn number(&self, #[comptime] field: Field, actor: u32) -> u32;
}

#[cube]
impl<'a> ReadLedger for Ledger<'a> {
    fn value_at(&self, field: u32, actor: u32) -> Ticks {
        read_ticks(&*self.words, self.config.coordinate(field, actor))
    }

    fn number_at(&self, field: u32, actor: u32) -> u32 {
        self.value_at(field, actor).extract(LOW_WORD)
    }

    fn value(&self, #[comptime] field: Field, actor: u32) -> Ticks {
        self.value_at(comptime!(field as u32), actor)
    }

    fn number(&self, #[comptime] field: Field, actor: u32) -> u32 {
        self.number_at(comptime!(field as u32), actor)
    }
}

#[derive(CubeType)]
pub(super) struct TickLedger<'a> {
    balances: &'a [Ticks],
    pub config: &'a EconomyConfig,
}

#[cube]
impl<'a> TickLedger<'a> {
    pub fn new(balances: &'a [Ticks], config: &'a EconomyConfig) -> Self {
        TickLedger::<'a> { balances, config }
    }
}

#[cube]
impl<'a> ReadLedger for TickLedger<'a> {
    fn value_at(&self, field: u32, actor: u32) -> Ticks {
        self.balances[self.config.coordinate(field, actor) as usize]
    }

    fn number_at(&self, field: u32, actor: u32) -> u32 {
        self.value_at(field, actor).extract(LOW_WORD)
    }

    fn value(&self, #[comptime] field: Field, actor: u32) -> Ticks {
        self.value_at(comptime!(field as u32), actor)
    }

    fn number(&self, #[comptime] field: Field, actor: u32) -> u32 {
        self.number_at(comptime!(field as u32), actor)
    }
}

#[derive(CubeType)]
pub(super) struct ActorTable<'a> {
    words: &'a [u32],
    stride: usize,
}

#[cube]
impl<'a> ActorTable<'a> {
    pub fn new(words: &'a [u32]) -> Self {
        Self::with_stride(words, words.len() / ActorLayout::WORDS)
    }

    pub fn with_stride(words: &'a [u32], stride: usize) -> Self {
        ActorTable::<'a> { words, stride }
    }

    fn column(&self, #[comptime] field: usize, actor: u32) -> u32 {
        self.words[field * self.stride + actor as usize]
    }
}

#[cube]
pub(super) trait ActorMetadata: CubeType {
    fn kind(&self, actor: u32) -> KernelActorKind;
    fn x(&self, actor: u32) -> u32;
    fn y(&self, actor: u32) -> u32;
    fn owner(&self, actor: u32) -> u32;
    fn capacity(&self, actor: u32) -> u32;
    fn geographic_capacity(&self, actor: u32) -> u32;
    fn expansion_draw(&self, actor: u32) -> u32;
    fn entrepreneurial_interest(&self, actor: u32) -> u32;
    fn distance(&self, a: u32, b: u32) -> u32;
}

#[cube]
impl<'a> ActorMetadata for ActorTable<'a> {
    fn kind(&self, actor: u32) -> KernelActorKind {
        KernelActorKind::load(self.column(comptime!(ActorLayout::KIND), actor))
    }

    fn x(&self, actor: u32) -> u32 {
        self.column(comptime!(ActorLayout::X), actor)
    }

    fn y(&self, actor: u32) -> u32 {
        self.column(comptime!(ActorLayout::Y), actor)
    }

    fn owner(&self, actor: u32) -> u32 {
        self.column(comptime!(ActorLayout::OWNER), actor)
    }

    fn capacity(&self, actor: u32) -> u32 {
        self.column(comptime!(ActorLayout::CAPACITY), actor)
    }

    fn geographic_capacity(&self, actor: u32) -> u32 {
        self.column(comptime!(ActorLayout::GEOGRAPHIC_CAPACITY), actor)
    }

    fn expansion_draw(&self, actor: u32) -> u32 {
        self.column(comptime!(ActorLayout::EXPANSION_DRAW), actor)
    }

    fn entrepreneurial_interest(&self, actor: u32) -> u32 {
        self.column(comptime!(ActorLayout::ENTREPRENEURIAL_INTEREST), actor)
    }

    fn distance(&self, a: u32, b: u32) -> u32 {
        let ax = self.x(a);
        let ay = self.y(a);
        let bx = self.x(b);
        let by = self.y(b);
        max(ax, bx) - min(ax, bx) + max(ay, by) - min(ay, by)
    }
}

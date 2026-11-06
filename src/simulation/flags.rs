use super::words::{Ticks, equal, lt};
use crate::domain::{Good, Market};
use crate::ledger::{AuthorizationFlags, Decision};
use anyhow::{Context, Result, anyhow};
use bytemuck::{Pod, Zeroable};
use cubecl::prelude::*;
use enumflags2::BitFlags;
use std::mem::offset_of;

pub const CONTROL_WORDS: usize = size_of::<ControlLayout>() / size_of::<u32>();
pub const PROGRESS_WORD: usize = offset_of!(ControlLayout, progress) / size_of::<u32>();
pub const FAULT_WORD: usize = offset_of!(ControlLayout, faults) / size_of::<u32>();
pub const COMPLETED_WORD: usize = offset_of!(ControlLayout, completed) / size_of::<u32>();
pub const DEPENDENCY_FLAGS_WORD: usize = offset_of!(ControlLayout, phase_data) / size_of::<u32>();
pub const CONSTRUCTED_EVENTS_WORD: usize = offset_of!(ControlLayout, phase_data) / size_of::<u32>();

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ControlLayout {
    pub progress: u32,
    pub faults: u32,
    pub completed: u32,
    pub phase_data: u32,
}
impl ControlLayout {
    pub fn new(words: &[u32]) -> Result<&Self> {
        let record = words
            .get(..CONTROL_WORDS)
            .context("incomplete scheduler control record")?;
        Ok(bytemuck::from_bytes(bytemuck::cast_slice(record)))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MarketFlag {
    Labor,
    Input(Good),
    Consumer(Good),
    Household,
    Payroll,
    Distribution,
}
impl From<Market> for MarketFlag {
    fn from(market: Market) -> Self {
        match market {
            Market::Labor => Self::Labor,
            Market::Input(good) => Self::Input(good),
            Market::Consumer(good) => Self::Consumer(good),
            Market::Household => Self::Household,
            Market::Payroll => Self::Payroll,
            Market::Distribution => Self::Distribution,
        }
    }
}
impl MarketFlag {
    pub fn good(self) -> Option<Good> {
        match self {
            Self::Input(good) | Self::Consumer(good) => Some(good),
            Self::Household => Some(Good::Bread),
            _ => None,
        }
    }
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActorKind {
    Household = 0,
}
impl ActorKind {
    pub const HOUSEHOLD_BITS: u32 = 0;
}
#[derive(CubeType, Clone, Copy)]
pub struct KernelActorKind {
    bits: u32,
}
#[cube]
impl KernelActorKind {
    pub fn load(bits: u32) -> KernelActorKind {
        KernelActorKind { bits }
    }
    pub fn bits(&self) -> u32 {
        self.bits
    }
    pub fn is(&self, #[comptime] kind: ActorKind) -> bool {
        self.bits == comptime!(kind as u32)
    }
    pub fn output(&self) -> u32 {
        self.bits
    }
}

bitflags::bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct FaultFlags: u32 {
        const ARITHMETIC_OVERFLOW = 1 << 0;
        const INVALID_COORDINATE = 1 << 1;
        const TRANSIT_UNDERFLOW = 1 << 2;
        const TRANSIT_MISMATCH = 1 << 3;
        const UNSETTLED_COMMITMENTS = 1 << 4;
        const DEMAND_OVERFLOW = 1 << 5;
        const PAYROLL_MISMATCH = 1 << 6;
        const RESEARCH_MISMATCH = 1 << 7;
        const PROPERTY_CAPACITY = 1 << 8;
    }
}
impl FaultFlags {
    pub fn new(bits: u32) -> Result<Self> {
        Self::from_bits(bits).with_context(|| format!("unknown compute fault bits: {bits:#x}"))
    }
    pub fn read(words: &[u32]) -> Result<Self> {
        Self::new(
            *words
                .get(FAULT_WORD)
                .context("missing compute fault word")?,
        )
    }
    pub fn into_result(self) -> Result<()> {
        match self.is_empty() {
            true => Ok(()),
            false => Err(anyhow!("compute fault: {self:?}")),
        }
    }
}

#[enumflags2::bitflags]
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DependencyFlag {
    MultiplePredecessors,
    MultipleSuccessors,
}

#[derive(Debug, PartialEq, Eq)]
pub enum DependencyShape {
    Chains,
    Frontiers,
}

#[derive(Debug)]
pub struct SchedulerControl {
    pub progress: usize,
    pub faults: FaultFlags,
    pub completed: usize,
    pub dependencies: BitFlags<DependencyFlag>,
}
impl SchedulerControl {
    pub fn new(words: &[u32]) -> Result<Self> {
        let record = ControlLayout::new(words)?;
        Ok(Self {
            progress: record.progress as usize,
            faults: FaultFlags::new(record.faults)?,
            completed: record.completed as usize,
            dependencies: BitFlags::from_bits(record.phase_data)
                .context("unknown dependency flags")?,
        })
    }
    pub fn shape(&self) -> DependencyShape {
        match self.dependencies.is_empty() {
            true => DependencyShape::Chains,
            false => DependencyShape::Frontiers,
        }
    }
}

#[enumflags2::bitflags]
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EventStatus {
    Inactive,
    Pending,
    Accepted,
    Rejected,
    Faulted,
}
impl TryFrom<u32> for EventStatus {
    type Error = anyhow::Error;
    fn try_from(word: u32) -> Result<Self> {
        let flags = BitFlags::<Self>::from_bits(word).context("unknown event status bits")?;
        match flags.len() {
            1 => flags.iter().next().context("missing event status"),
            _ => Err(anyhow!("expected exactly one event status: {flags:?}")),
        }
    }
}
impl EventStatus {
    pub fn decision(self) -> Result<Option<Decision>> {
        match self {
            Self::Inactive => Ok(None),
            Self::Accepted => Ok(Some(Decision::Accepted)),
            Self::Rejected => Ok(Some(Decision::Rejected)),
            Self::Pending | Self::Faulted => Err(anyhow!("unfinished event: {self:?}")),
        }
    }
}

#[derive(CubeType, Clone, Copy)]
pub struct KernelEventStatus {
    bits: u32,
}
#[cube]
impl KernelEventStatus {
    pub fn new(#[comptime] status: EventStatus) -> KernelEventStatus {
        KernelEventStatus {
            bits: comptime!(BitFlags::from_flag(status).bits()),
        }
    }
    pub fn load(statuses: &mut [u32], index: u32) -> KernelEventStatus {
        KernelEventStatus {
            bits: statuses[index as usize],
        }
    }
    pub fn store(&self, statuses: &mut [u32], index: u32) {
        statuses[index as usize] = self.bits;
    }
    pub fn contains(&self, #[comptime] status: EventStatus) -> bool {
        self.bits & comptime!(BitFlags::from_flag(status).bits()) != 0
    }
    pub fn set(&mut self, #[comptime] status: EventStatus) {
        self.bits = comptime!(BitFlags::from_flag(status).bits());
    }
    pub fn replace(&mut self, status: KernelEventStatus) {
        self.bits = status.bits;
    }
}

#[derive(CubeType, Clone, Copy)]
pub struct KernelFaults {
    bits: u32,
}
#[cube]
impl KernelFaults {
    pub fn none() -> KernelFaults {
        KernelFaults {
            bits: comptime!(FaultFlags::empty().bits()),
        }
    }
    pub fn new(#[comptime] flags: FaultFlags) -> KernelFaults {
        KernelFaults {
            bits: comptime!(flags.bits()),
        }
    }
    pub fn union(&self, other: KernelFaults) -> KernelFaults {
        KernelFaults {
            bits: self.bits | other.bits,
        }
    }
    pub fn insert(&mut self, other: KernelFaults) {
        self.bits |= other.bits;
    }
    pub fn is_empty(&self) -> bool {
        self.bits == 0
    }
    pub fn bits(&self) -> u32 {
        self.bits
    }
    pub fn record(&self, control: &[Atomic<u32>]) {
        if !self.is_empty() {
            control[FAULT_WORD].fetch_or(self.bits());
        }
    }
}

#[derive(CubeType, Clone, Copy)]
pub struct KernelAuthorization {
    bits: u32,
}
#[cube]
impl KernelAuthorization {
    pub fn new(#[comptime] flags: AuthorizationFlags) -> KernelAuthorization {
        KernelAuthorization {
            bits: comptime!(flags.bits()),
        }
    }
    pub fn load(bits: u32) -> KernelAuthorization {
        KernelAuthorization { bits }
    }
    pub fn bits(&self) -> u32 {
        self.bits
    }
    pub fn permits(&self, balance: Ticks, guard: Ticks) -> bool {
        let known = self.bits & comptime!(!AuthorizationFlags::all().bits()) == 0;
        let exact = self.bits & comptime!(AuthorizationFlags::EQUAL.bits()) != 0;
        let minimum = self.bits & comptime!(AuthorizationFlags::AT_LEAST.bits()) != 0;
        known && (!exact || equal(balance, guard)) && (!minimum || !lt(balance, guard))
    }
}

#[cube]
pub fn record_dependency(control: &[Atomic<u32>], #[comptime] flag: DependencyFlag) {
    control[DEPENDENCY_FLAGS_WORD].fetch_or(comptime!(BitFlags::from_flag(flag).bits()));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::{Device, ops::TEST_WORKGROUP};

    #[cube(launch)]
    fn concurrent_flags(control: &[Atomic<u32>]) {
        let id = ABSOLUTE_POS as u32;
        match id % 9 {
            0 => KernelFaults::new(comptime!(FaultFlags::ARITHMETIC_OVERFLOW)).record(control),
            1 => KernelFaults::new(comptime!(FaultFlags::INVALID_COORDINATE)).record(control),
            2 => KernelFaults::new(comptime!(FaultFlags::TRANSIT_UNDERFLOW)).record(control),
            3 => KernelFaults::new(comptime!(FaultFlags::TRANSIT_MISMATCH)).record(control),
            4 => KernelFaults::new(comptime!(FaultFlags::UNSETTLED_COMMITMENTS)).record(control),
            5 => KernelFaults::new(comptime!(FaultFlags::DEMAND_OVERFLOW)).record(control),
            6 => KernelFaults::new(comptime!(FaultFlags::PAYROLL_MISMATCH)).record(control),
            7 => KernelFaults::new(comptime!(FaultFlags::RESEARCH_MISMATCH)).record(control),
            _ => KernelFaults::new(comptime!(FaultFlags::PROPERTY_CAPACITY)).record(control),
        }
        match id % 2 {
            0 => {
                record_dependency(control, comptime!(DependencyFlag::MultiplePredecessors));
            }
            _ => {
                record_dependency(control, comptime!(DependencyFlag::MultipleSuccessors));
            }
        }
    }

    fn check_atomic_flags(device: Device) -> Result<()> {
        let control = device.tensor(bytemuck::cast_slice(&[ControlLayout::zeroed()]));
        concurrent_flags::launch(
            &device.client,
            CubeCount::Static(8, 1, 1),
            CubeDim::new_1d(TEST_WORKGROUP),
            control.clone().into_buffer_arg(),
        );
        let result = SchedulerControl::new(&device.read(&control)?)?;
        assert_eq!(result.faults, FaultFlags::all());
        assert_eq!(result.dependencies, BitFlags::all());
        assert_eq!(result.progress, 0);
        assert_eq!(result.completed, 0);
        Ok(())
    }

    #[test]
    fn concurrent_flag_sets_survive_gpu_readback() -> Result<()> {
        check_atomic_flags(Device::headless(TEST_WORKGROUP)?)
    }
}

use super::flags::MarketFlag;
use crate::domain::Phase;
use cubecl::prelude::*;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum Discovery {
    #[cfg(test)]
    Linear,
    #[cfg(test)]
    Original,
    Indexed,
}

impl Discovery {
    pub fn indexed(self) -> bool {
        match self {
            #[cfg(test)]
            Self::Linear => false,
            _ => true,
        }
    }
}

#[derive(CubeType, CubeLaunch, Clone, Copy)]
pub struct EconomyParameters {
    #[cube(comptime)]
    pub scan: super::parallel::ScanMode,
    pub day: u32,
    pub settlement_rotation: u32,
    #[cube(comptime)]
    pub phase: Phase,
    #[cube(comptime)]
    pub market: MarketFlag,
    #[cube(comptime)]
    pub discovery: Discovery,
    pub offset: u32,
    pub count: u32,
    pub status_start: u32,
    pub sort_source: u32,
    pub sort_width: u32,
    pub sort_count: u32,
}
impl EconomyParameters {
    pub fn launch(self) -> EconomyParametersLaunch {
        EconomyParametersLaunch {
            scan: self.scan,
            day: self.day,
            settlement_rotation: self.settlement_rotation,
            phase: self.phase,
            market: self.market,
            discovery: self.discovery,
            offset: self.offset,
            count: self.count,
            status_start: self.status_start,
            sort_source: self.sort_source,
            sort_width: self.sort_width,
            sort_count: self.sort_count,
        }
    }
}

#[derive(CubeType, CubeLaunch, Clone, Copy, Default)]
pub struct SchedulerParameters {
    pub count: u32,
    pub width: u32,
    pub source: u32,
    pub status_offset: u32,
}
impl SchedulerParameters {
    pub fn launch(self) -> SchedulerParametersLaunch {
        SchedulerParametersLaunch {
            count: self.count,
            width: self.width,
            source: self.source,
            status_offset: self.status_offset,
        }
    }
}

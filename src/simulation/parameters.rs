use super::flags::{InputMarket, MarketFlag};
use crate::domain::{Good, Phase};
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
    pub clock: MarketClock,
    #[cube(comptime)]
    pub scan: super::parallel::ScanMode,
    pub day: u32,
    pub settlement_rotation: u32,
    #[cube(comptime)]
    pub phase: Phase,
    #[cube(comptime)]
    pub market: MarketFlag,
    pub good: u32,
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
            clock: self.clock,
            scan: self.scan,
            day: self.day,
            settlement_rotation: self.settlement_rotation,
            phase: self.phase,
            market: self.market,
            good: self.good,
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

#[cube]
impl EconomyParameters {
    pub fn refresh(&mut self, config: &mut super::configuration::EconomyConfig) {
        if comptime!(self.clock == MarketClock::Device) {
            self.day = config.consumption_intervals[crate::domain::Good::COUNT + 0usize];
            self.settlement_rotation =
                config.consumption_intervals[crate::domain::Good::COUNT + 1usize];
            self.good = config.consumption_intervals[crate::domain::Good::COUNT + 2usize];
            config.heating_per_person =
                config.consumption_intervals[crate::domain::Good::COUNT + 3usize];
        }
    }

    pub fn traded_good(&self) -> u32 {
        match comptime!(self.market) {
            MarketFlag::Input(InputMarket::Grain) => u32::new(Good::Grain as i64),
            MarketFlag::Input(InputMarket::Other) | MarketFlag::Consumer => self.good,
            MarketFlag::Household => u32::new(Good::Bread as i64),
            _ => u32::new(0),
        }
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum MarketClock {
    Scalars,
    Device,
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

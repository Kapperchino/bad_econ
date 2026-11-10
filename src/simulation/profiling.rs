use super::{Device, profiling_day};
use crate::domain::Market;
use cubecl::{client::ProfileWindow, profile::ProfileDuration};
use std::{cell::RefCell, fmt::Debug};

pub(super) enum MarketProfile {
    Disabled,
    Enabled(RefCell<Vec<Measurement>>),
}

pub(super) struct Measurement {
    family: &'static str,
    stage: String,
    duration: ProfileDuration,
}

impl MarketProfile {
    pub(super) fn new(day: u64) -> Self {
        match day == profiling_day() && std::env::var_os("BAD_ECON_PROFILE_DEVICE").is_some() {
            true => Self::Enabled(RefCell::default()),
            false => Self::Disabled,
        }
    }

    pub(super) fn start(&self, device: &Device) -> Option<ProfileWindow> {
        match self {
            Self::Disabled => None,
            Self::Enabled(_) => Some(
                device
                    .client
                    .profile_start()
                    .expect("device stage profiling start"),
            ),
        }
    }

    pub(super) fn end(
        &self,
        device: &Device,
        window: Option<ProfileWindow>,
        family: &'static str,
        stage: impl Debug,
    ) {
        if let (Self::Enabled(measurements), Some(window)) = (self, window) {
            measurements.borrow_mut().push(Measurement {
                family,
                stage: format!("{stage:?}"),
                duration: device
                    .client
                    .profile_end(window)
                    .expect("device stage profiling end"),
            });
        }
    }

    pub(super) fn resolve(self, market: Market, round: usize) {
        if let Self::Enabled(measurements) = self {
            for measurement in measurements.into_inner() {
                let method = measurement.duration.timing_method();
                let elapsed = pollster::block_on(measurement.duration.resolve());
                match elapsed {
                    Some(elapsed) => eprintln!(
                        "GPU {} {market:?} {round} {} {method} {:.3}",
                        measurement.family,
                        measurement.stage,
                        elapsed.duration().as_secs_f64() * 1000.0,
                    ),
                    None => eprintln!(
                        "GPU {} {market:?} {round} {} unmeasured",
                        measurement.family, measurement.stage,
                    ),
                }
            }
        }
    }
}

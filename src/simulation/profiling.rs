use super::{
    Device,
    configuration::Configuration,
    settlement::{Stage, Wave, WaveBookings},
};
use crate::domain::Market;
use anyhow::Result;
use burn_wgpu::CubeTensor;
use cubecl::{client::ProfileWindow, profile::ProfileDuration};
use std::{cell::RefCell, fmt::Debug, time::Instant};

pub(super) enum MarketProfile {
    Disabled,
    Enabled(RefCell<Vec<Measurement>>),
    Freight(RefCell<FreightProfile>),
}

pub(super) struct Measurement {
    family: &'static str,
    stage: String,
    duration: ProfileDuration,
}

pub(super) struct FreightProfile {
    device: Device,
    window: Option<ProfileWindow>,
    dispatches: usize,
    completed: Vec<WaveMeasurement>,
}

struct WaveMeasurement {
    wave: Wave,
    dispatches: usize,
    duration: ProfileDuration,
}

impl Drop for FreightProfile {
    fn drop(&mut self) {
        if let Some(window) = self.window.take() {
            self.device.client.profile_abandon(window);
        }
    }
}

impl FreightProfile {
    fn report(&mut self, market: Market, round: usize, bookings: WaveBookings) {
        for measurement in self.completed.drain(..) {
            let wave = measurement.wave;
            let dispatches = measurement.dispatches;
            let method = measurement.duration.timing_method();
            let elapsed = pollster::block_on(measurement.duration.resolve())
                .map(|ticks| format!("{:.3}", ticks.duration().as_secs_f64() * 1000.0))
                .unwrap_or_else(|| String::from("unmeasured"));
            let counts = match bookings.faults.is_empty() {
                true => match wave {
                    Wave::Initial => {
                        format!("bookings={} remaining={}", bookings.initial, bookings.retry)
                    }
                    Wave::Retry => format!("bookings={} remaining=0", bookings.retry),
                },
                false => format!("faults={:?}", bookings.faults),
            };
            eprintln!(
                "FREIGHT_WAVE {market:?} round={round} wave={wave:?} {counts} dispatches={dispatches} timing={method} ms={elapsed}"
            );
        }
    }
}

impl MarketProfile {
    pub(super) fn new(device: &Device, day: u64) -> Self {
        match day == profiling_day() {
            true if std::env::var_os("BAD_ECON_PROFILE_FREIGHT").is_some() => {
                Self::Freight(RefCell::new(FreightProfile {
                    device: device.clone(),
                    window: None,
                    dispatches: 0,
                    completed: Vec::new(),
                }))
            }
            true if std::env::var_os("BAD_ECON_PROFILE_DEVICE").is_some() => {
                Self::Enabled(RefCell::default())
            }
            _ => Self::Disabled,
        }
    }

    pub(super) fn freight(
        &self,
        device: &Device,
        stage: Stage,
        wave: Wave,
        launch: impl FnOnce() -> usize,
    ) {
        match self {
            Self::Freight(profile) => {
                let mut profile = profile.borrow_mut();
                if matches!(
                    (wave, stage),
                    (Wave::Initial, Stage::Initialize) | (Wave::Retry, Stage::Queue)
                ) {
                    profile.window = Some(
                        device
                            .client
                            .profile_start()
                            .expect("freight wave profiling start"),
                    );
                    profile.dispatches = 0;
                }
                profile.dispatches += launch();
                if stage == Stage::Release {
                    let window = profile.window.take().expect("open freight wave");
                    let dispatches = profile.dispatches;
                    profile.completed.push(WaveMeasurement {
                        wave,
                        dispatches,
                        duration: device
                            .client
                            .profile_end(window)
                            .expect("freight wave profiling end"),
                    });
                }
            }
            _ => {
                let window = self.start(device);
                launch();
                self.end(device, window, "FREIGHT", stage);
            }
        }
    }

    pub(super) fn start(&self, device: &Device) -> Option<ProfileWindow> {
        match self {
            Self::Disabled | Self::Freight(_) => None,
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

    pub(super) fn resolve(
        self,
        market: Market,
        round: usize,
        config: &Configuration,
        control: &CubeTensor,
    ) -> Result<u64> {
        match self {
            Self::Freight(profile) if !profile.borrow().completed.is_empty() => {
                let mut profile = profile.into_inner();
                let bookings = WaveBookings::read(&profile.device, config, control)?;
                profile.report(market, round, bookings);
                Ok(WaveBookings::BYTES as u64)
            }
            Self::Disabled | Self::Freight(_) => Ok(0),
            Self::Enabled(measurements) => {
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
                Ok(0)
            }
        }
    }
}

pub(super) fn audit_checkpoints(device: &Device, day: u64) -> impl FnMut(&str) + use<> {
    let mut profile = (day == profiling_day()
        && std::env::var_os("BAD_ECON_PROFILE_STAGES").is_some())
    .then(|| (device.clone(), Instant::now()));
    move |label| {
        if let Some((device, previous)) = &mut profile {
            device.sync().expect("audit profile synchronization");
            let now = Instant::now();
            eprintln!(
                "AUDIT {day} {label} {:.3}",
                now.duration_since(*previous).as_secs_f64() * 1000.0
            );
            *previous = now;
        }
    }
}

pub(super) fn profiling_day() -> u64 {
    static DAY: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *DAY.get_or_init(|| match std::env::var("BAD_ECON_PROFILE_DAY") {
        Ok(day) => day
            .parse::<std::num::NonZeroU64>()
            .expect("BAD_ECON_PROFILE_DAY must be a positive day number")
            .get(),
        Err(std::env::VarError::NotPresent) => 30,
        Err(error) => panic!("BAD_ECON_PROFILE_DAY: {error}"),
    })
}

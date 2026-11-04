mod kernels;
use super::{Device, flags::FaultFlags, layout::WORD_BYTES};
use crate::ledger::TickLayout;
use anyhow::{Result, anyhow};
use burn_wgpu::CubeTensor;
use bytemuck::{Pod, Zeroable};
use cubecl::CubeDim;
pub use kernels::{
    KernelCapacity, KernelDeficit, KernelFatigue, KernelObservation, available_minutes,
    next_fatigue, observed_blend, service_deficit,
};
use serde::{Deserialize, Serialize};
use std::mem::offset_of;

pub const HOUSEHOLD_SCALE: u32 = 1000;
pub const DAY_MINUTES: u32 = 1440;
pub const NORMAL_WORK_MINUTES: u32 = 480;

#[repr(u32)]
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum Participation {
    Unavailable,
    Available,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum Observation {
    Missing,
    Observed(u64),
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObservationKind {
    Missing,
    Observed,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeliveredService {
    pub need: u64,
    pub delivered: u64,
    pub weight: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HouseholdObservation {
    pub services: Vec<DeliveredService>,
    pub participation: Participation,
    pub capacity_minutes: u32,
    pub fatigue: u32,
    pub reserved_minutes: u32,
    pub fatigue_response: u32,
    pub exertion_minutes: u32,
    pub rest_minutes: u32,
    pub prior_forecast: u64,
    pub observation: Observation,
    pub observation_weight: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RequestLayout {
    forecast: TickLayout,
    signal: TickLayout,
    service_offset: u32,
    service_count: u32,
    participation: u32,
    capacity: u32,
    fatigue: u32,
    reserved: u32,
    fatigue_response: u32,
    exertion: u32,
    rest: u32,
    observation: u32,
    weight: u32,
}

impl RequestLayout {
    const WORDS: usize = size_of::<Self>() / WORD_BYTES;
    const FORECAST: usize = offset_of!(Self, forecast) / WORD_BYTES;
    const SIGNAL: usize = offset_of!(Self, signal) / WORD_BYTES;
    const SERVICE_OFFSET: usize = offset_of!(Self, service_offset) / WORD_BYTES;
    const SERVICE_COUNT: usize = offset_of!(Self, service_count) / WORD_BYTES;
    const PARTICIPATION: usize = offset_of!(Self, participation) / WORD_BYTES;
    const CAPACITY: usize = offset_of!(Self, capacity) / WORD_BYTES;
    const FATIGUE: usize = offset_of!(Self, fatigue) / WORD_BYTES;
    const RESERVED: usize = offset_of!(Self, reserved) / WORD_BYTES;
    const FATIGUE_RESPONSE: usize = offset_of!(Self, fatigue_response) / WORD_BYTES;
    const EXERTION: usize = offset_of!(Self, exertion) / WORD_BYTES;
    const REST: usize = offset_of!(Self, rest) / WORD_BYTES;
    const OBSERVATION: usize = offset_of!(Self, observation) / WORD_BYTES;
    const WEIGHT: usize = offset_of!(Self, weight) / WORD_BYTES;

    fn new(request: &HouseholdObservation, service_offset: u32) -> Result<Self> {
        let fractions = [
            request.fatigue,
            request.fatigue_response,
            request.observation_weight,
        ];
        let weights: u64 = request
            .services
            .iter()
            .filter(|service| service.need > 0)
            .map(|service| u64::from(service.weight))
            .sum();
        let admitted = fractions
            .into_iter()
            .all(|fraction| fraction <= HOUSEHOLD_SCALE)
            && request.services.len() <= u32::MAX as usize
            && weights <= u64::from(HOUSEHOLD_SCALE)
            && request.capacity_minutes <= DAY_MINUTES
            && request.exertion_minutes <= DAY_MINUTES
            && request.rest_minutes <= request.reserved_minutes.min(DAY_MINUTES);
        match admitted {
            false => Err(anyhow!(
                "household observations exceed time, fraction, or active service-weight bounds"
            )),
            true => Ok(Self {
                forecast: request.prior_forecast.into(),
                signal: match request.observation {
                    Observation::Missing => 0,
                    Observation::Observed(signal) => signal,
                }
                .into(),
                service_offset,
                service_count: request.services.len() as u32,
                participation: request.participation as u32,
                capacity: request.capacity_minutes,
                fatigue: request.fatigue,
                reserved: request.reserved_minutes,
                fatigue_response: request.fatigue_response,
                exertion: request.exertion_minutes,
                rest: request.rest_minutes,
                observation: match request.observation {
                    Observation::Missing => ObservationKind::Missing,
                    Observation::Observed(_) => ObservationKind::Observed,
                } as u32,
                weight: request.observation_weight,
            }),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ServiceLayout {
    need: TickLayout,
    delivered: TickLayout,
    weight: u32,
}
impl ServiceLayout {
    const WORDS: usize = size_of::<Self>() / WORD_BYTES;
    const NEED: usize = offset_of!(Self, need) / WORD_BYTES;
    const DELIVERED: usize = offset_of!(Self, delivered) / WORD_BYTES;
    const WEIGHT: usize = offset_of!(Self, weight) / WORD_BYTES;
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ResultLayout {
    forecast: TickLayout,
    deficit_lower: u32,
    deficit_upper: u32,
    available: u32,
    fatigue: u32,
}
impl ResultLayout {
    const WORDS: usize = size_of::<Self>() / WORD_BYTES;
    const FORECAST: usize = offset_of!(Self, forecast) / WORD_BYTES;
    const DEFICIT_LOWER: usize = offset_of!(Self, deficit_lower) / WORD_BYTES;
    const DEFICIT_UPPER: usize = offset_of!(Self, deficit_upper) / WORD_BYTES;
    const AVAILABLE: usize = offset_of!(Self, available) / WORD_BYTES;
    const FATIGUE: usize = offset_of!(Self, fatigue) / WORD_BYTES;
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct HouseholdPolicyResult {
    pub deficit_lower: u32,
    pub deficit_upper: u32,
    pub available_minutes: u32,
    pub fatigue: u32,
    pub forecast: u64,
}

pub fn evaluate(
    device: &Device,
    requests: &[HouseholdObservation],
) -> Result<Vec<HouseholdPolicyResult>> {
    let mut services = Vec::new();
    let mut encoded = Vec::new();
    for request in requests {
        let offset = u32::try_from(services.len())?;
        encoded.push(RequestLayout::new(request, offset)?);
        services.extend(request.services.iter().map(|service| ServiceLayout {
            need: service.need.into(),
            delivered: service.delivered.into(),
            weight: service.weight,
        }));
    }
    let count = u32::try_from(requests.len())?;
    let _ = u32::try_from(services.len())?;
    encoded.resize(encoded.len().max(1), RequestLayout::zeroed());
    services.resize(services.len().max(1), ServiceLayout::zeroed());
    let input = device.tensor(bytemuck::cast_slice(&encoded));
    let services = device.tensor(bytemuck::cast_slice(&services));
    let output = device.empty(requests.len().max(1) * ResultLayout::WORDS);
    let control = device.tensor(&[0; 4]);
    let argument = |tensor: &CubeTensor| tensor.clone().into_buffer_arg();
    let cube_count = device.count(count as usize);
    kernels::evaluate::launch(
        &device.client,
        cube_count,
        CubeDim::new_1d(device.workgroup),
        argument(&input),
        argument(&services),
        argument(&output),
        argument(&control),
        count,
    );
    FaultFlags::read(&device.read(&control)?)?.into_result()?;
    let output = device.read(&output)?;
    Ok(bytemuck::cast_slice::<u32, ResultLayout>(&output)
        .iter()
        .take(requests.len())
        .map(|result| HouseholdPolicyResult {
            deficit_lower: result.deficit_lower,
            deficit_upper: result.deficit_upper,
            available_minutes: result.available,
            fatigue: result.fatigue,
            forecast: result.forecast.into(),
        })
        .collect())
}

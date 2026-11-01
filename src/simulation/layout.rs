use super::flags::ControlLayout;
use super::negotiation::PolicyLayout;
use super::rows::row_layout;
use crate::{
    arithmetic::PRICE_TABLE,
    domain::{Actor, Good, MAX_TAX_BRACKETS, Scenario, State},
    ledger::{LEG_WORDS, MAX_LEGS, TickLayout},
};
use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable};
use std::mem::offset_of;

pub const WORD_BYTES: usize = size_of::<u32>();
pub const TICK_BYTES: usize = size_of::<u64>();
pub const TICK_WORDS: usize = TICK_BYTES / WORD_BYTES;
pub const WORD_BITS: u32 = u32::BITS;
pub const TICK_BITS: u32 = u64::BITS;
pub const WORD_TOP_BIT: u32 = WORD_BITS - 1;
pub const HALF_WORD_BITS: u32 = WORD_BITS / 2;
pub const HALF_WORD_MASK: u32 = u16::MAX as u32;
pub const LOW_WORD: usize = TickLayout::LOW;
pub const HIGH_WORD: usize = TickLayout::HIGH;
pub const MIN_BUFFER_WORDS: usize = 4;
pub const EVENT_WORDS: usize = MAX_LEGS * LEG_WORDS;
pub const NO_ACTOR: u32 = u32::MAX;
pub const NO_PREDECESSOR: u32 = u32::MAX;
pub const NO_SUCCESSOR: u32 = u32::MAX;

pub struct SpatialIndex {
    pub words: Vec<u32>,
}
impl SpatialIndex {
    pub fn new(scenario: &Scenario, state: &State) -> Self {
        let config = scenario.config();
        let side = config.map_size.div_ceil(config.search_radius);
        let cells = (side * side) as usize;
        let segment = cells + 1 + config.firms as usize;
        let cell = |index: usize| {
            let position = state.actors.at(index).position;
            (position.y / config.search_radius * side + position.x / config.search_radius) as usize
        };
        let mut firms = (config.households as usize..scenario.actors()).collect::<Vec<_>>();
        firms.sort_by_key(|&index| cell(index));
        let mut words = vec![0; 2 * segment];
        for (slot, &firm) in firms.iter().enumerate() {
            words[segment + cell(firm) + 1] += 1;
            words[segment + cells + 1 + slot] = firm as u32;
        }
        for index in 1..=cells {
            words[segment + index] += words[segment + index - 1];
        }
        Self { words }
    }
}

row_layout!(ActorMetadataRows, ActorLayout {
    x: u32 => X,
    y: u32 => Y,
    kind: u32 => KIND,
    owner: u32 => OWNER,
    capacity: u32 => CAPACITY,
});
impl ActorLayout {
    pub fn new(actor: &Actor) -> Self {
        Self {
            x: actor.position.x,
            y: actor.position.y,
            kind: actor.kind,
            owner: actor.owner,
            capacity: actor.capacity,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ConfigLayout {
    actors: u32,
    households: u32,
    firms: u32,
    fields: u32,
    ring: u32,
    search_radius: u32,
    map_size: u32,
    commute_speed: u32,
    shipment_speed: u32,
    household_need: u32,
    industrial: u32,
    consumption_intervals: [u32; Good::COUNT],
    capacity: u32,
    packet_output: u32,
    packet_minutes: u32,
    work_cap: u32,
    reserved_minutes: u32,
    forecast_weight: u32,
    coverage: u32,
    distribution_rate: u32,
    minimum_price: u32,
    maximum_price: u32,
    delivery_penalty: u32,
    reserve_low: u32,
    reserve_high: u32,
    pub spatial_side: u32,
    health_recovery: u32,
    hunger_damage: u32,
    work_fatigue: u32,
    rest_recovery: u32,
    wage_increment: u32,
    price_response: u32,
    payroll_enabled: u32,
    tax_base_rate: u32,
    employee_charge: u32,
    tax_bracket_count: u32,
    tax_thresholds: [TickLayout; MAX_TAX_BRACKETS],
    tax_rates: [u32; MAX_TAX_BRACKETS],
    household_policy_enabled: u32,
    household_recovery: u32,
    deficit_damage: u32,
    fatigue_damage: u32,
    fatigue_response: u32,
    rest_minutes: u32,
    bargaining: PolicyLayout,
    reserved: [u32; 2],
    price_table: [u32; PRICE_TABLE.len()],
}
impl ConfigLayout {
    pub const ACTORS: usize = offset_of!(Self, actors) / WORD_BYTES;
    pub const HOUSEHOLDS: usize = offset_of!(Self, households) / WORD_BYTES;
    pub const FIRMS: usize = offset_of!(Self, firms) / WORD_BYTES;
    pub const FIELDS: usize = offset_of!(Self, fields) / WORD_BYTES;
    pub const RING: usize = offset_of!(Self, ring) / WORD_BYTES;
    pub const SEARCH_RADIUS: usize = offset_of!(Self, search_radius) / WORD_BYTES;
    pub const MAP_SIZE: usize = offset_of!(Self, map_size) / WORD_BYTES;
    pub const COMMUTE_SPEED: usize = offset_of!(Self, commute_speed) / WORD_BYTES;
    pub const SHIPMENT_SPEED: usize = offset_of!(Self, shipment_speed) / WORD_BYTES;
    pub const CONSUMPTION_INTERVALS: usize = offset_of!(Self, consumption_intervals) / WORD_BYTES;
    pub const INDUSTRIAL: usize = offset_of!(Self, industrial) / WORD_BYTES;
    pub const HOUSEHOLD_NEED: usize = offset_of!(Self, household_need) / WORD_BYTES;
    pub const WORK_CAP: usize = offset_of!(Self, work_cap) / WORD_BYTES;
    pub const RESERVED_MINUTES: usize = offset_of!(Self, reserved_minutes) / WORD_BYTES;
    pub const FORECAST_WEIGHT: usize = offset_of!(Self, forecast_weight) / WORD_BYTES;
    pub const COVERAGE: usize = offset_of!(Self, coverage) / WORD_BYTES;
    pub const DISTRIBUTION_RATE: usize = offset_of!(Self, distribution_rate) / WORD_BYTES;
    pub const MINIMUM_PRICE: usize = offset_of!(Self, minimum_price) / WORD_BYTES;
    pub const MAXIMUM_PRICE: usize = offset_of!(Self, maximum_price) / WORD_BYTES;
    pub const DELIVERY_PENALTY: usize = offset_of!(Self, delivery_penalty) / WORD_BYTES;
    pub const RESERVE_LOW: usize = offset_of!(Self, reserve_low) / WORD_BYTES;
    pub const RESERVE_HIGH: usize = offset_of!(Self, reserve_high) / WORD_BYTES;
    pub const SPATIAL_SIDE: usize = offset_of!(Self, spatial_side) / WORD_BYTES;
    pub const HEALTH_RECOVERY: usize = offset_of!(Self, health_recovery) / WORD_BYTES;
    pub const HUNGER_DAMAGE: usize = offset_of!(Self, hunger_damage) / WORD_BYTES;
    pub const WORK_FATIGUE: usize = offset_of!(Self, work_fatigue) / WORD_BYTES;
    pub const REST_RECOVERY: usize = offset_of!(Self, rest_recovery) / WORD_BYTES;
    pub const WAGE_INCREMENT: usize = offset_of!(Self, wage_increment) / WORD_BYTES;
    pub const PRICE_RESPONSE: usize = offset_of!(Self, price_response) / WORD_BYTES;
    pub const PAYROLL_ENABLED: usize = offset_of!(Self, payroll_enabled) / WORD_BYTES;
    pub const TAX_BASE_RATE: usize = offset_of!(Self, tax_base_rate) / WORD_BYTES;
    pub const EMPLOYEE_CHARGE: usize = offset_of!(Self, employee_charge) / WORD_BYTES;
    pub const TAX_BRACKET_COUNT: usize = offset_of!(Self, tax_bracket_count) / WORD_BYTES;
    pub const TAX_THRESHOLDS: usize = offset_of!(Self, tax_thresholds) / WORD_BYTES;
    pub const TAX_RATES: usize = offset_of!(Self, tax_rates) / WORD_BYTES;
    pub const HOUSEHOLD_POLICY_ENABLED: usize =
        offset_of!(Self, household_policy_enabled) / WORD_BYTES;
    pub const HOUSEHOLD_RECOVERY: usize = offset_of!(Self, household_recovery) / WORD_BYTES;
    pub const DEFICIT_DAMAGE: usize = offset_of!(Self, deficit_damage) / WORD_BYTES;
    pub const FATIGUE_DAMAGE: usize = offset_of!(Self, fatigue_damage) / WORD_BYTES;
    pub const FATIGUE_RESPONSE: usize = offset_of!(Self, fatigue_response) / WORD_BYTES;
    pub const REST_MINUTES: usize = offset_of!(Self, rest_minutes) / WORD_BYTES;
    pub const BARGAINING: usize = offset_of!(Self, bargaining) / WORD_BYTES;
    pub const PRICE_TABLE: usize = offset_of!(Self, price_table) / WORD_BYTES;

    pub fn new(scenario: &Scenario) -> Self {
        let config = scenario.config();
        Self {
            actors: scenario.actors() as u32,
            households: config.households,
            firms: config.firms,
            fields: scenario.fields() as u32,
            ring: scenario.ring() as u32,
            search_radius: config.search_radius,
            map_size: config.map_size,
            commute_speed: config.commute_speed,
            shipment_speed: config.shipment_speed,
            household_need: config.household_need,
            industrial: u32::from(config.economy == crate::domain::Economy::Industrial),
            consumption_intervals: Good::ALL.map(Good::consumption_interval),
            capacity: config.capacity,
            packet_output: config.packet_output,
            packet_minutes: config.packet_minutes,
            work_cap: config.work_cap,
            reserved_minutes: config.reserved_minutes,
            forecast_weight: config.forecast_weight,
            coverage: config.coverage,
            distribution_rate: config.distribution_rate,
            minimum_price: config.minimum_price,
            maximum_price: config.maximum_price,
            delivery_penalty: config.delivery_penalty,
            reserve_low: config.reserve as u32,
            reserve_high: (config.reserve >> WORD_BITS) as u32,
            spatial_side: config.map_size.div_ceil(config.search_radius),
            health_recovery: config.behavior.health_recovery,
            hunger_damage: config.behavior.hunger_damage,
            work_fatigue: config.behavior.work_fatigue,
            rest_recovery: config.behavior.rest_recovery,
            wage_increment: config.behavior.wage_increment,
            price_response: config.behavior.price_response,
            household_policy_enabled: u32::from(config.household_policy.is_some()),
            household_recovery: config
                .household_policy
                .as_ref()
                .map_or(0, |policy| policy.recovery),
            deficit_damage: config
                .household_policy
                .as_ref()
                .map_or(0, |policy| policy.deficit_damage),
            fatigue_damage: config
                .household_policy
                .as_ref()
                .map_or(0, |policy| policy.fatigue_damage),
            fatigue_response: config
                .household_policy
                .as_ref()
                .map_or(0, |policy| policy.fatigue_response),
            rest_minutes: config
                .household_policy
                .as_ref()
                .map_or(0, |policy| policy.rest_minutes),
            bargaining: PolicyLayout::new(config.bargaining.as_ref()),
            payroll_enabled: u32::from(config.payroll.is_some()),
            tax_base_rate: config.payroll.as_ref().map_or(0, |policy| policy.base_rate),
            employee_charge: config
                .payroll
                .as_ref()
                .map_or(0, |policy| policy.employee_charge),
            tax_bracket_count: config
                .payroll
                .as_ref()
                .map_or(0, |policy| policy.brackets.len() as u32),
            tax_thresholds: std::array::from_fn(|index| {
                config
                    .payroll
                    .as_ref()
                    .and_then(|policy| policy.brackets.get(index))
                    .map_or(0, |bracket| bracket.threshold)
                    .into()
            }),
            tax_rates: std::array::from_fn(|index| {
                config
                    .payroll
                    .as_ref()
                    .and_then(|policy| policy.brackets.get(index))
                    .map_or(0, |bracket| bracket.marginal_rate)
            }),
            reserved: [0; 2],
            price_table: PRICE_TABLE,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ProposalLayout {
    actor: u32,
    quantity: TickLayout,
    distance: u32,
}
impl ProposalLayout {
    pub const ACTOR: usize = offset_of!(Self, actor) / WORD_BYTES;
    pub const QUANTITY_LOW: usize = offset_of!(Self, quantity) / WORD_BYTES + TickLayout::LOW;
    pub const QUANTITY_HIGH: usize = offset_of!(Self, quantity) / WORD_BYTES + TickLayout::HIGH;
    pub const DISTANCE: usize = offset_of!(Self, distance) / WORD_BYTES;
    pub const WORDS: usize = size_of::<Self>() / WORD_BYTES;
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ScratchLayout {
    sort_first: [u32; MAX_LEGS],
    sort_second: [u32; MAX_LEGS],
    predecessors: [u32; MAX_LEGS],
    statuses: [u32; 2],
}
impl ScratchLayout {
    pub const PREDECESSORS: usize = offset_of!(Self, predecessors) / size_of::<[u32; MAX_LEGS]>();
    pub const STATUSES: usize = offset_of!(Self, statuses) / size_of::<[u32; MAX_LEGS]>();
    pub const WORDS_PER_EVENT: usize = size_of::<Self>() / WORD_BYTES;
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct LinkLayout {
    first_predecessor: u32,
    last_predecessor: u32,
    successor: u32,
}
impl LinkLayout {
    pub const FIRST_PREDECESSOR: usize = offset_of!(Self, first_predecessor) / WORD_BYTES;
    pub const LAST_PREDECESSOR: usize = offset_of!(Self, last_predecessor) / WORD_BYTES;
    pub const SUCCESSOR: usize = offset_of!(Self, successor) / WORD_BYTES;
    pub const WORDS_PER_EVENT: usize = size_of::<Self>() / WORD_BYTES;
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct SummaryLayout {
    pub cash: TickLayout,
    pub payroll_reserves: TickLayout,
    pub income_tax: TickLayout,
    pub employee_charges: TickLayout,
    pub employed: TickLayout,
    pub shortages: TickLayout,
    pub inventories: [TickLayout; Good::COUNT],
    pub transit: [TickLayout; Good::COUNT],
    pub produced: [TickLayout; Good::COUNT],
    pub used: [TickLayout; Good::COUNT],
    pub consumed: TickLayout,
    pub consumption: [TickLayout; Good::COUNT],
    pub minimum_wage: TickLayout,
    pub maximum_wage: TickLayout,
    pub minimum_price: TickLayout,
    pub maximum_price: TickLayout,
    pub firms_by_good: [TickLayout; Good::COUNT],
    pub wage_minima: [TickLayout; Good::COUNT],
    pub wage_maxima: [TickLayout; Good::COUNT],
    pub wage_totals: [TickLayout; Good::COUNT],
    pub wage_squares: [TickLayout; Good::COUNT],
    pub price_minima: [TickLayout; Good::COUNT],
    pub price_maxima: [TickLayout; Good::COUNT],
    pub price_totals: [TickLayout; Good::COUNT],
    pub price_squares: [TickLayout; Good::COUNT],
}
impl SummaryLayout {
    pub const CASH: u32 = (offset_of!(Self, cash) / TICK_BYTES) as u32;
    pub const PAYROLL_RESERVES: u32 = (offset_of!(Self, payroll_reserves) / TICK_BYTES) as u32;
    pub const INCOME_TAX: u32 = (offset_of!(Self, income_tax) / TICK_BYTES) as u32;
    pub const EMPLOYEE_CHARGES: u32 = (offset_of!(Self, employee_charges) / TICK_BYTES) as u32;
    pub const EMPLOYED: u32 = (offset_of!(Self, employed) / TICK_BYTES) as u32;
    pub const SHORTAGES: u32 = (offset_of!(Self, shortages) / TICK_BYTES) as u32;
    pub const INVENTORIES: u32 = (offset_of!(Self, inventories) / TICK_BYTES) as u32;
    pub const TRANSIT: u32 = (offset_of!(Self, transit) / TICK_BYTES) as u32;
    pub const PRODUCED: u32 = (offset_of!(Self, produced) / TICK_BYTES) as u32;
    pub const USED: u32 = (offset_of!(Self, used) / TICK_BYTES) as u32;
    pub const CONSUMPTION: u32 = (offset_of!(Self, consumption) / TICK_BYTES) as u32;
    pub const CONSUMED: u32 = (offset_of!(Self, consumed) / TICK_BYTES) as u32;
    pub const MINIMUM_WAGE: u32 = (offset_of!(Self, minimum_wage) / TICK_BYTES) as u32;
    pub const MAXIMUM_WAGE: u32 = (offset_of!(Self, maximum_wage) / TICK_BYTES) as u32;
    pub const MINIMUM_PRICE: u32 = (offset_of!(Self, minimum_price) / TICK_BYTES) as u32;
    pub const MAXIMUM_PRICE: u32 = (offset_of!(Self, maximum_price) / TICK_BYTES) as u32;
    pub const FIRMS_BY_GOOD: u32 = (offset_of!(Self, firms_by_good) / TICK_BYTES) as u32;
    pub const WAGE_MINIMA: u32 = (offset_of!(Self, wage_minima) / TICK_BYTES) as u32;
    pub const WAGE_MAXIMA: u32 = (offset_of!(Self, wage_maxima) / TICK_BYTES) as u32;
    pub const WAGE_TOTALS: u32 = (offset_of!(Self, wage_totals) / TICK_BYTES) as u32;
    pub const WAGE_SQUARES: u32 = (offset_of!(Self, wage_squares) / TICK_BYTES) as u32;
    pub const PRICE_MINIMA: u32 = (offset_of!(Self, price_minima) / TICK_BYTES) as u32;
    pub const PRICE_MAXIMA: u32 = (offset_of!(Self, price_maxima) / TICK_BYTES) as u32;
    pub const PRICE_TOTALS: u32 = (offset_of!(Self, price_totals) / TICK_BYTES) as u32;
    pub const PRICE_SQUARES: u32 = (offset_of!(Self, price_squares) / TICK_BYTES) as u32;
    pub const METRICS: usize = size_of::<Self>() / TICK_BYTES;
    pub const WORDS: usize = size_of::<Self>() / WORD_BYTES;
    pub const ACTORS_PER_GROUP: u32 = 256;
    pub const WEALTH_BINS: usize = u64::BITS as usize;
    pub const REGION_SIDE: usize = 16;
    pub const REGIONS: usize = Self::REGION_SIDE * Self::REGION_SIDE;
    pub const WEALTH_START: usize = offset_of!(ObservationControlLayout, wealth_bins) / WORD_BYTES;
    pub const REGIONS_START: usize = offset_of!(ObservationControlLayout, regions) / WORD_BYTES;
    pub const CONTROL_WORDS: usize = size_of::<ObservationControlLayout>() / WORD_BYTES;

    pub fn new(words: &[u32]) -> Result<&Self> {
        let record = words
            .get(..Self::WORDS)
            .context("incomplete summary record")?;
        Ok(bytemuck::from_bytes(bytemuck::cast_slice(record)))
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ObservationControlLayout {
    pub scheduler: ControlLayout,
    pub wealth_bins: [u32; SummaryLayout::WEALTH_BINS],
    pub regions: [u32; SummaryLayout::REGIONS],
}
impl ObservationControlLayout {
    pub fn new(words: &[u32]) -> Result<&Self> {
        let record = words
            .get(..SummaryLayout::CONTROL_WORDS)
            .context("incomplete observation control record")?;
        Ok(bytemuck::from_bytes(bytemuck::cast_slice(record)))
    }
}

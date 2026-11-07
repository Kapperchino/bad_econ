use super::flags::ControlLayout;

use super::rows::row_layout;
use crate::{
    domain::{Actor, ActorRows, Good, Scenario, State},
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
pub const DISCOVERY_BLOCK: u32 = 128;

pub struct SpatialIndex {
    pub words: Vec<u32>,
}
impl SpatialIndex {
    pub fn new(scenario: &Scenario, state: &State) -> Self {
        Self::from_actors(scenario, &state.actors)
    }
    pub(super) fn from_actors(scenario: &Scenario, actors: &ActorRows) -> Self {
        let config = scenario.config();
        let side = config.map_size.div_ceil(config.search_radius);
        let cells = (side * side) as usize;
        let segment = cells + 1 + actors.len();
        let cell = |index: usize| {
            let position = actors.at(index).position;
            (position.y / config.search_radius * side + position.x / config.search_radius) as usize
        };
        let mut starts = vec![0; cells + 1];
        for actor in 0..actors.len() {
            starts[cell(actor) + 1] += 1;
        }
        for cell in 1..starts.len() {
            starts[cell] += starts[cell - 1];
        }
        let mut members = vec![0; actors.len()];
        for actor in 0..actors.len() {
            let offset = &mut starts[cell(actor)];
            members[*offset] = actor;
            *offset += 1;
        }
        let firms = members
            .iter()
            .copied()
            .filter(|&index| actors.at(index).kind > 0)
            .collect::<Vec<_>>();
        let mut words =
            vec![
                0;
                3 * segment + actors.len().div_ceil(DISCOVERY_BLOCK as usize) + 1 + actors.len()
            ];
        for (base, members) in [(segment, &members), (2 * segment, &firms)] {
            for (slot, &actor) in members.iter().enumerate() {
                words[base + cell(actor) + 1] += 1;
                words[base + cells + 1 + slot] = actor as u32;
            }
            for index in 1..=cells {
                words[base + index] += words[base + index - 1];
            }
        }
        let tile = config.map_size.div_ceil(128);
        let query_side = config.map_size.div_ceil(tile);
        let query_cell = |index: usize| {
            let position = actors.at(index).position;
            (position.y / tile * query_side + position.x / tile) as usize
        };
        let mut offsets = vec![0; (query_side * query_side) as usize + 1];
        for actor in 0..actors.len() {
            offsets[query_cell(actor) + 1] += 1;
        }
        for cell in 1..offsets.len() {
            offsets[cell] += offsets[cell - 1];
        }
        let query_base = 3 * segment + actors.len().div_ceil(DISCOVERY_BLOCK as usize) + 1;
        for actor in 0..actors.len() {
            let offset = &mut offsets[query_cell(actor)];
            words[query_base + *offset] = actor as u32;
            *offset += 1;
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
    geographic_capacity: u32 => GEOGRAPHIC_CAPACITY,
    entrepreneurial_interest: u32 => ENTREPRENEURIAL_INTEREST,
    expansion_draw: u32 => EXPANSION_DRAW,
});
impl ActorLayout {
    pub fn new(actor: &Actor, scenario: &Scenario) -> Self {
        Self {
            x: actor.position.x,
            y: actor.position.y,
            kind: actor.kind,
            owner: actor.owner,
            capacity: actor.capacity,
            geographic_capacity: scenario.geographic_capacity(actor),
            expansion_draw: crate::domain::enterprise::ExpansionPolicy::draw(
                scenario.config().seed,
                u64::from(actor.owner),
            ),
            entrepreneurial_interest: scenario
                .config()
                .entrepreneurship
                .interest(scenario.config().seed, actor.identity)
                as u32,
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
    pub rent_collected: TickLayout,
    pub rent_arrears: TickLayout,
    pub homeless_households: TickLayout,
    pub employed: TickLayout,
    pub shortages: TickLayout,
    pub inventories: [TickLayout; Good::COUNT],
    pub transit: [TickLayout; Good::COUNT],
    pub produced: [TickLayout; Good::COUNT],
    pub used: [TickLayout; Good::COUNT],
    pub consumed: TickLayout,
    pub household_production: [TickLayout; Good::COUNT],
    pub distributors: TickLayout,
    pub closures: TickLayout,
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
    pub const RENT_COLLECTED: u32 = (offset_of!(Self, rent_collected) / TICK_BYTES) as u32;
    pub const RENT_ARREARS: u32 = (offset_of!(Self, rent_arrears) / TICK_BYTES) as u32;
    pub const HOMELESS_HOUSEHOLDS: u32 =
        (offset_of!(Self, homeless_households) / TICK_BYTES) as u32;
    pub const EMPLOYED: u32 = (offset_of!(Self, employed) / TICK_BYTES) as u32;
    pub const SHORTAGES: u32 = (offset_of!(Self, shortages) / TICK_BYTES) as u32;
    pub const INVENTORIES: u32 = (offset_of!(Self, inventories) / TICK_BYTES) as u32;
    pub const TRANSIT: u32 = (offset_of!(Self, transit) / TICK_BYTES) as u32;
    pub const PRODUCED: u32 = (offset_of!(Self, produced) / TICK_BYTES) as u32;
    pub const USED: u32 = (offset_of!(Self, used) / TICK_BYTES) as u32;
    pub const CONSUMPTION: u32 = (offset_of!(Self, consumption) / TICK_BYTES) as u32;
    pub const HOUSEHOLD_PRODUCTION: u32 =
        (offset_of!(Self, household_production) / TICK_BYTES) as u32;
    pub const DISTRIBUTORS: u32 = (offset_of!(Self, distributors) / TICK_BYTES) as u32;
    pub const CLOSURES: u32 = (offset_of!(Self, closures) / TICK_BYTES) as u32;
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
    pub const ACTORS_PER_GROUP: u32 = 64;
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

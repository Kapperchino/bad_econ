use super::flags::CONTROL_WORDS;
use crate::ledger::{LEG_WORDS, MAX_LEGS};

pub const WORD_BYTES: usize = size_of::<u32>();
pub const TICK_BYTES: usize = size_of::<u64>();
pub const TICK_WORDS: usize = TICK_BYTES / WORD_BYTES;
pub const WORD_BITS: u32 = u32::BITS;
pub const TICK_BITS: u32 = u64::BITS;
pub const WORD_TOP_BIT: u32 = WORD_BITS - 1;
pub const HALF_WORD_BITS: u32 = WORD_BITS / 2;
pub const HALF_WORD_MASK: u32 = u16::MAX as u32;
pub const LOW_WORD: usize = 0;
pub const HIGH_WORD: usize = 1;
pub const MIN_BUFFER_WORDS: usize = 4;
pub const EVENT_WORDS: usize = MAX_LEGS * LEG_WORDS;
pub const NO_ACTOR: u32 = u32::MAX;
pub const NO_PREDECESSOR: u32 = u32::MAX;
pub const NO_SUCCESSOR: u32 = u32::MAX;

pub struct ActorLayout;
impl ActorLayout {
    pub const X: usize = 0;
    pub const Y: usize = 1;
    pub const KIND: usize = 2;
    pub const OWNER: usize = 3;
    pub const CAPACITY: usize = 4;
    pub const WORDS: usize = Self::CAPACITY + 1;
}

pub struct ConfigLayout;
impl ConfigLayout {
    pub const ACTORS: usize = 0;
    pub const HOUSEHOLDS: usize = 1;
    pub const FIRMS: usize = 2;
    pub const FIELDS: usize = 3;
    pub const RING: usize = 4;
    pub const SEARCH_RADIUS: usize = 5;
    pub const MAP_SIZE: usize = 6;
    pub const COMMUTE_SPEED: usize = 7;
    pub const SHIPMENT_SPEED: usize = 8;
    pub const HOUSEHOLD_NEED: usize = 9;
    pub const CAPACITY: usize = 10;
    pub const PACKET_OUTPUT: usize = 11;
    pub const PACKET_MINUTES: usize = 12;
    pub const WORK_CAP: usize = 13;
    pub const RESERVED_MINUTES: usize = 14;
    pub const FORECAST_WEIGHT: usize = 15;
    pub const COVERAGE: usize = 16;
    pub const DISTRIBUTION_RATE: usize = 17;
    pub const MINIMUM_PRICE: usize = 18;
    pub const MAXIMUM_PRICE: usize = 19;
    pub const DELIVERY_PENALTY: usize = 20;
    pub const RESERVE_LOW: usize = 21;
    pub const RESERVE_HIGH: usize = 22;
    pub const SPATIAL_SIDE: usize = 23;
    pub const PRICE_TABLE: usize = 32;
}

pub struct ProposalLayout;
impl ProposalLayout {
    pub const ACTOR: usize = 0;
    pub const QUANTITY_LOW: usize = 1;
    pub const QUANTITY_HIGH: usize = 2;
    pub const DISTANCE: usize = 3;
    pub const WORDS: usize = Self::DISTANCE + 1;
}

pub struct ScratchLayout;
impl ScratchLayout {
    pub const SORT_BUFFERS: usize = 2;
    pub const PREDECESSORS: usize = Self::SORT_BUFFERS;
    pub const STATUSES: usize = Self::PREDECESSORS + 1;
    pub const STATUS_BUFFERS: usize = 2;
    pub const WORDS_PER_EVENT: usize = Self::STATUSES * MAX_LEGS + Self::STATUS_BUFFERS;
}

pub struct LinkLayout;
impl LinkLayout {
    pub const FIRST_PREDECESSOR: usize = 0;
    pub const LAST_PREDECESSOR: usize = 1;
    pub const SUCCESSOR: usize = 2;
    pub const WORDS_PER_EVENT: usize = Self::SUCCESSOR + 1;
}

pub struct SummaryLayout;
impl SummaryLayout {
    pub const CASH: u32 = 0;
    pub const EMPLOYED: u32 = 1;
    pub const SHORTAGES: u32 = 2;
    pub const INVENTORIES: u32 = 3;
    pub const TRANSIT: u32 = 6;
    pub const PRODUCED: u32 = 9;
    pub const USED: u32 = 12;
    pub const CONSUMED: u32 = 15;
    pub const MINIMUM_WAGE: u32 = 16;
    pub const MAXIMUM_WAGE: u32 = 17;
    pub const MINIMUM_PRICE: u32 = 18;
    pub const MAXIMUM_PRICE: u32 = 19;
    pub const METRICS: usize = 24;
    pub const WORDS: usize = Self::METRICS * TICK_WORDS;
    pub const ACTORS_PER_GROUP: u32 = 256;
    pub const WEALTH_BINS: usize = u64::BITS as usize;
    pub const REGION_SIDE: usize = 16;
    pub const REGIONS: usize = Self::REGION_SIDE * Self::REGION_SIDE;
    pub const WEALTH_START: usize = CONTROL_WORDS;
    pub const REGIONS_START: usize = Self::WEALTH_START + Self::WEALTH_BINS;
    pub const CONTROL_WORDS: usize = Self::REGIONS_START + Self::REGIONS;
}

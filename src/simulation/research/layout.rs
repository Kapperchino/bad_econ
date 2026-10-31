use crate::ledger::TickLayout;
use bytemuck::{Pod, Zeroable};
use std::mem::offset_of;

macro_rules! layout {
    ($name:ident { $($field:ident : $ty:ty => $constant:ident),* $(,)? }) => {
        #[repr(C)]
        #[derive(Clone, Copy, Pod, Zeroable)]
        pub struct $name { $(pub $field: $ty),* }
        impl $name {
            pub const WORDS: usize = size_of::<Self>() / size_of::<u32>();
            $(pub const $constant: usize = offset_of!(Self, $field) / size_of::<u32>();)*
        }
    };
}

layout!(Header {
    actors: u32 => ACTORS,
    lab_count: u32 => LAB_COUNT,
    project_count: u32 => PROJECT_COUNT,
    machine_count: u32 => MACHINE_COUNT,
    stock_count: u32 => STOCK_COUNT,
    strike_count: u32 => STRIKE_COUNT,
    labs: u32 => LABS,
    projects: u32 => PROJECTS,
    machines: u32 => MACHINES,
    stocks: u32 => STOCKS,
    strikes: u32 => STRIKES,
    lab_state: u32 => LAB_STATE,
    project_state: u32 => PROJECT_STATE,
    machine_state: u32 => MACHINE_STATE,
    stock_state: u32 => STOCK_STATE,
    strike_state: u32 => STRIKE_STATE,
    seed: TickLayout => SEED,
});

layout!(LabLayout {
    owner: u32 => OWNER,
    researcher: u32 => RESEARCHER,
    first_project: u32 => FIRST_PROJECT,
    project_count: u32 => PROJECT_COUNT,
    adoption: u32 => ADOPTION,
    consent: u32 => CONSENT,
    production_requirements: u32 => PRODUCTION_REQUIREMENTS,
    discovered: u32 => DISCOVERED,
    mastered: u32 => MASTERED,
    permitted: u32 => PERMITTED,
    adopted: u32 => ADOPTED,
});

layout!(ProjectLayout {
    identity: TickLayout => IDENTITY,
    prerequisites: u32 => PREREQUISITES,
    outcomes: u32 => OUTCOMES,
    inputs: [u32; 3] => INPUTS,
    labor_minutes: u32 => LABOR_MINUTES,
    minimum_effort: u32 => MINIMUM_EFFORT,
    pioneer_effort: u32 => PIONEER_EFFORT,
    capacity: u32 => CAPACITY,
    relatedness: u32 => RELATEDNESS,
    shape: u32 => SHAPE,
    signal: u32 => SIGNAL,
});

layout!(MachineLayout {
    firm: u32 => FIRM,
    output_ticks: u32 => OUTPUT_TICKS,
    initial_service: TickLayout => INITIAL_SERVICE,
    maximum_packets: u32 => MAXIMUM_PACKETS,
    nameplate: u32 => NAMEPLATE,
    downtime: u32 => DOWNTIME,
    wear: u32 => WEAR,
    maintenance_good: u32 => MAINTENANCE_GOOD,
    maintenance_ticks: u32 => MAINTENANCE_TICKS,
    restoration: u32 => RESTORATION,
});

layout!(StockLayout {
    firm: u32 => FIRM,
    output_ticks: u32 => OUTPUT_TICKS,
    maximum_packets: u32 => MAXIMUM_PACKETS,
    capacity: u32 => CAPACITY,
    renewal: u32 => RENEWAL,
    extraction: u32 => EXTRACTION,
});

layout!(StrikeLayout {
    household: u32 => HOUSEHOLD,
    identity: TickLayout => IDENTITY,
    grievance: u32 => GRIEVANCE,
    solidarity: u32 => SOLIDARITY,
    trust: u32 => TRUST,
    efficacy: u32 => EFFICACY,
    peers: u32 => PEERS,
    cost: u32 => COST,
    risk: u32 => RISK,
});

layout!(LabState {
    discovered: u32 => DISCOVERED,
    mastered: u32 => MASTERED,
    permitted: u32 => PERMITTED,
    adopted: u32 => ADOPTED,
    consumed: [TickLayout; 3] => CONSUMED,
});

layout!(ProjectState {
    attempts: TickLayout => ATTEMPTS,
    discoveries: TickLayout => DISCOVERIES,
    exposure: u32 => EXPOSURE,
    outcome: u32 => OUTCOME,
    probability: u32 => PROBABILITY,
});

layout!(MachineState {
    remaining: TickLayout => REMAINING,
    condition: u32 => CONDITION,
    produced_before: TickLayout => PRODUCED_BEFORE,
    consumed: TickLayout => CONSUMED,
    maintenance_restoration: u32 => MAINTENANCE_RESTORATION,
});

layout!(StockState {
    stock: u32 => STOCK,
    produced_before: TickLayout => PRODUCED_BEFORE,
    extracted: TickLayout => EXTRACTED,
});

layout!(StrikeState {
    score: u32 => SCORE,
    probability: u32 => PROBABILITY,
    participation: u32 => PARTICIPATION,
});

layout!(SummaryLayout {
    consumed: [TickLayout; 3] => CONSUMED,
    attempts: TickLayout => ATTEMPTS,
    discoveries: TickLayout => DISCOVERIES,
    remaining_service: TickLayout => REMAINING_SERVICE,
    natural_stock: TickLayout => NATURAL_STOCK,
    striking: TickLayout => STRIKING,
});

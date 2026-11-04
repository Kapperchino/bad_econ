use crate::domain::Good;
use crate::ledger::TickLayout;
use crate::simulation::rows::row_layout;
use bytemuck::{Pod, Zeroable};
use cubecl::prelude::*;
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

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Header {
    pub actors: u32,
    pub lab_count: u32,
    pub project_count: u32,
    pub machine_count: u32,
    pub stock_count: u32,
    pub strike_count: u32,
    pub labs: u32,
    pub methods: u32,
    pub method_count: u32,
    pub projects: u32,
    pub machines: u32,
    pub stocks: u32,
    pub strikes: u32,
    pub lab_state: u32,
    pub project_state: u32,
    pub machine_state: u32,
    pub stock_state: u32,
    pub strike_state: u32,
    pub seed: TickLayout,
}
impl Header {
    pub const WORDS: usize = size_of::<Self>() / size_of::<u32>();
}

#[derive(CubeType, CubeLaunch)]
pub struct ResearchParameters {
    pub actors: u32,
    pub lab_count: u32,
    pub project_count: u32,
    pub machine_count: u32,
    pub stock_count: u32,
    pub strike_count: u32,
    pub labs: u32,
    pub methods: u32,
    pub method_count: u32,
    pub projects: u32,
    pub machines: u32,
    pub stocks: u32,
    pub strikes: u32,
    pub lab_state: u32,
    pub project_state: u32,
    pub machine_state: u32,
    pub stock_state: u32,
    pub strike_state: u32,
}
impl Header {
    pub fn launch(self) -> ResearchParametersLaunch {
        ResearchParametersLaunch {
            actors: self.actors,
            lab_count: self.lab_count,
            project_count: self.project_count,
            machine_count: self.machine_count,
            stock_count: self.stock_count,
            strike_count: self.strike_count,
            labs: self.labs - Self::WORDS as u32,
            methods: self.methods - Self::WORDS as u32,
            method_count: self.method_count,
            projects: self.projects - Self::WORDS as u32,
            machines: self.machines - Self::WORDS as u32,
            stocks: self.stocks - Self::WORDS as u32,
            strikes: self.strikes - Self::WORDS as u32,
            lab_state: self.lab_state,
            project_state: self.project_state,
            machine_state: self.machine_state,
            stock_state: self.stock_state,
            strike_state: self.strike_state,
        }
    }
}
#[derive(CubeType, CubeLaunch)]
pub struct KernelResearchConfig {
    pub parameters: ResearchParameters,
    pub records: Box<[u32]>,
}

row_layout!(LabLayoutRows, LabLayout {
    owner: u32 => OWNER,
    first_method: u32 => FIRST_METHOD,
    method_count: u32 => METHOD_COUNT,
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

row_layout!(ProjectLayoutRows, ProjectLayout {
    prerequisites: u32 => PREREQUISITES,
    outcomes: u32 => OUTCOMES,
    inputs: [u32; Good::COUNT] => INPUTS,
    labor_minutes: u32 => LABOR_MINUTES,
    minimum_effort: u32 => MINIMUM_EFFORT,
    pioneer_effort: u32 => PIONEER_EFFORT,
    capacity: u32 => CAPACITY,
    relatedness: u32 => RELATEDNESS,
    shape: u32 => SHAPE,
    signal: u32 => SIGNAL,
});

row_layout!(MachineLayoutRows, MachineLayout {
    firm: u32 => FIRM,
    initial_service: TickLayout => INITIAL_SERVICE,
    maximum_packets: u32 => MAXIMUM_PACKETS,
    nameplate: u32 => NAMEPLATE,
    downtime: u32 => DOWNTIME,
    wear: u32 => WEAR,
    maintenance_good: u32 => MAINTENANCE_GOOD,
    maintenance_ticks: u32 => MAINTENANCE_TICKS,
    restoration: u32 => RESTORATION,
});

row_layout!(StockLayoutRows, StockLayout {
    firm: u32 => FIRM,
    maximum_packets: u32 => MAXIMUM_PACKETS,
    capacity: u32 => CAPACITY,
    renewal: u32 => RENEWAL,
    extraction: u32 => EXTRACTION,
});

row_layout!(StrikeLayoutRows, StrikeLayout {
    household: u32 => HOUSEHOLD,
    grievance: u32 => GRIEVANCE,
    solidarity: u32 => SOLIDARITY,
    trust: u32 => TRUST,
    efficacy: u32 => EFFICACY,
    peers: u32 => PEERS,
    cost: u32 => COST,
    risk: u32 => RISK,
});

row_layout!(LabStateRows, LabState {
    discovered: u32 => DISCOVERED,
    mastered: u32 => MASTERED,
    permitted: u32 => PERMITTED,
    adopted: u32 => ADOPTED,
    consumed: [TickLayout; Good::COUNT] => CONSUMED,
});

row_layout!(ProjectStateRows, ProjectState {
    attempts: TickLayout => ATTEMPTS,
    discoveries: TickLayout => DISCOVERIES,
    exposure: u32 => EXPOSURE,
    outcome: u32 => OUTCOME,
    probability: u32 => PROBABILITY,
});

row_layout!(MachineStateRows, MachineState {
    remaining: TickLayout => REMAINING,
    condition: u32 => CONDITION,
    produced_before: TickLayout => PRODUCED_BEFORE,
    consumed: TickLayout => CONSUMED,
    maintenance_restoration: u32 => MAINTENANCE_RESTORATION,
});

row_layout!(StockStateRows, StockState {
    stock: u32 => STOCK,
    produced_before: TickLayout => PRODUCED_BEFORE,
    extracted: TickLayout => EXTRACTED,
});

row_layout!(StrikeStateRows, StrikeState {
    score: u32 => SCORE,
    probability: u32 => PROBABILITY,
    participation: u32 => PARTICIPATION,
});

layout!(SummaryLayout {
    consumed: [TickLayout; Good::COUNT] => CONSUMED,
    attempts: TickLayout => ATTEMPTS,
    discoveries: TickLayout => DISCOVERIES,
    remaining_service: TickLayout => REMAINING_SERVICE,
    natural_stock: TickLayout => NATURAL_STOCK,
    striking: TickLayout => STRIKING,
});

impl LabState {
    pub fn from_rows(
        words: &[u32],
        offset: usize,
        count: usize,
        index: usize,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            discovered: read_row(words, offset + Self::DISCOVERED * count, index)?,
            mastered: read_row(words, offset + Self::MASTERED * count, index)?,
            permitted: read_row(words, offset + Self::PERMITTED * count, index)?,
            adopted: read_row(words, offset + Self::ADOPTED * count, index)?,
            consumed: read_row(words, offset + Self::CONSUMED * count, index)?,
        })
    }
}

impl ProjectState {
    pub fn from_rows(
        words: &[u32],
        offset: usize,
        count: usize,
        index: usize,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            attempts: read_row(words, offset + Self::ATTEMPTS * count, index)?,
            discoveries: read_row(words, offset + Self::DISCOVERIES * count, index)?,
            exposure: read_row(words, offset + Self::EXPOSURE * count, index)?,
            outcome: read_row(words, offset + Self::OUTCOME * count, index)?,
            probability: read_row(words, offset + Self::PROBABILITY * count, index)?,
        })
    }
}

impl MachineState {
    pub fn from_rows(
        words: &[u32],
        offset: usize,
        count: usize,
        index: usize,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            remaining: read_row(words, offset + Self::REMAINING * count, index)?,
            condition: read_row(words, offset + Self::CONDITION * count, index)?,
            produced_before: read_row(words, offset + Self::PRODUCED_BEFORE * count, index)?,
            consumed: read_row(words, offset + Self::CONSUMED * count, index)?,
            maintenance_restoration: read_row(
                words,
                offset + Self::MAINTENANCE_RESTORATION * count,
                index,
            )?,
        })
    }
}

impl StockState {
    pub fn from_rows(
        words: &[u32],
        offset: usize,
        count: usize,
        index: usize,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            stock: read_row(words, offset + Self::STOCK * count, index)?,
            produced_before: read_row(words, offset + Self::PRODUCED_BEFORE * count, index)?,
            extracted: read_row(words, offset + Self::EXTRACTED * count, index)?,
        })
    }
}

impl StrikeState {
    pub fn from_rows(
        words: &[u32],
        offset: usize,
        count: usize,
        index: usize,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            score: read_row(words, offset + Self::SCORE * count, index)?,
            probability: read_row(words, offset + Self::PROBABILITY * count, index)?,
            participation: read_row(words, offset + Self::PARTICIPATION * count, index)?,
        })
    }
}

fn read_row<T: Pod>(words: &[u32], offset: usize, index: usize) -> anyhow::Result<T> {
    use anyhow::Context;
    let width = size_of::<T>() / size_of::<u32>();
    let start = offset + index * width;
    let value = words
        .get(start..start + width)
        .context("incomplete research observation")?;
    Ok(bytemuck::pod_read_unaligned(bytemuck::cast_slice(value)))
}

row_layout!(MethodLayoutRows, MethodLayout {
    identity: u32 => IDENTITY,
    requirements: u32 => REQUIREMENTS,
    labor_minutes: u32 => LABOR_MINUTES,
    input_ticks: u32 => INPUT_TICKS,
    output_ticks: u32 => OUTPUT_TICKS,
    first_input: u32 => FIRST_INPUT,
    second_input: u32 => SECOND_INPUT,
});

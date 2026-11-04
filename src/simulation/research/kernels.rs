use super::super::{
    flags::{FaultFlags, KernelFaults},
    pass::GpuPass,
    system::{System, gpu_passes},
    words::*,
};
use super::{layout::*, nonlinear::*};
use crate::domain::{Field, Good, research::RESEARCH_SCALE};
use crate::simulation::rows::RowIndex;
use cubecl::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ResearchStage {
    Opening,
    ProductionReadiness,
    MachineCapacity,
    StockCapacity,
    Strike,
    Maintenance,
    Attempt,
    MachineClose,
    StockClose,
    Summary,
    Audit,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttemptOutcome {
    Idle,
    Ineligible,
    MissingResources,
    NoTime,
    Failed,
    Discovered,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Participation {
    Working,
    Striking,
}

#[cube]
fn read(state: &mut [u32], position: usize) -> Ticks {
    ticks(state[position], state[position + 1])
}

#[cube]
fn write(state: &mut [u32], position: usize, value: Ticks) {
    state[position] = value.extract(0usize);
    state[position + 1] = value.extract(1usize);
}

#[cube]
fn main_coordinate(config: &KernelResearchConfig, field: u32, actor: u32) -> usize {
    ((field * config.parameters.actors + actor) * 2) as usize
}

#[cube]
fn add(control: &[Atomic<u32>], first: Ticks, second: Ticks) -> Ticks {
    let sum = plus(first, second);
    sum.faults.record(control);
    sum.value
}

#[cube]
fn increment(state: &mut [u32], position: usize, amount: Ticks, control: &[Atomic<u32>]) {
    let total = add(control, read(state, position), amount);
    write(state, position, total);
}

#[cube]
fn cap_plan(main: &mut [u32], config: &KernelResearchConfig, firm: u32, capacity: Ticks) {
    let plan = main_coordinate(config, comptime!(Field::Plan as u32), firm);
    let slots = main_coordinate(config, comptime!(Field::Slots as u32), firm);
    let capped = minimum(read(main, plan), capacity);
    write(main, plan, capped);
    write(main, slots, capped);
}

#[cube]
fn access(state: &mut [u32], base: &RowIndex) -> u32 {
    state[base.field(LabState::DISCOVERED, 1usize)]
        & state[base.field(LabState::MASTERED, 1usize)]
        & state[base.field(LabState::PERMITTED, 1usize)]
        & state[base.field(LabState::ADOPTED, 1usize)]
}

#[cube]
fn attempt(
    main: &mut [u32],
    state: &mut [u32],
    config: &KernelResearchConfig,
    table: &[u32],
    samples: &[f32],
    control: &[Atomic<u32>],
    id: usize,
) {
    let base = RowIndex::new(
        config.parameters.labs as usize,
        config.parameters.lab_count as usize,
        id,
    );
    let lab_state = RowIndex::new(
        config.parameters.lab_state as usize,
        config.parameters.lab_count as usize,
        id,
    );
    let owner = config.records[base.field(LabLayout::OWNER, 1usize)];
    let researcher = config.records[base.field(LabLayout::RESEARCHER, 1usize)];
    let time = main_coordinate(config, comptime!(Field::Time as u32), researcher);
    let research_minutes =
        main_coordinate(config, comptime!(Field::ResearchMinutes as u32), researcher);
    let start = config.records[base.field(LabLayout::FIRST_PROJECT, 1usize)];
    let end = start + config.records[base.field(LabLayout::PROJECT_COUNT, 1usize)];
    for project in start..end {
        let item = RowIndex::new(
            config.parameters.projects as usize,
            config.parameters.project_count as usize,
            project as usize,
        );
        let saved = RowIndex::new(
            config.parameters.project_state as usize,
            config.parameters.project_count as usize,
            project as usize,
        );
        let exposure = state[saved.field(ProjectState::EXPOSURE, 1usize)];
        let correlated = exposure
            * config.records[item.field(ProjectLayout::CAPACITY, 1usize)]
            * config.records[item.field(ProjectLayout::RELATEDNESS, 1usize)]
            / (RESEARCH_SCALE * RESEARCH_SCALE);
        let mut power = u32::new(comptime!(RESEARCH_SCALE as i64));
        for _step in 0..config.records[item.field(ProjectLayout::SHAPE, 1usize)] {
            power = power * (RESEARCH_SCALE - correlated) / RESEARCH_SCALE;
        }
        let scale = config.records[item.field(ProjectLayout::MINIMUM_EFFORT, 1usize)]
            + (config.records[item.field(ProjectLayout::PIONEER_EFFORT, 1usize)]
                - config.records[item.field(ProjectLayout::MINIMUM_EFFORT, 1usize)])
                * power
                / RESEARCH_SCALE;
        let effort = config.records[item.field(ProjectLayout::LABOR_MINUTES, 1usize)];
        let probability = hazard(
            min(effort * RESEARCH_SCALE / scale, 16 * RESEARCH_SCALE),
            table,
        );
        state[saved.field(ProjectState::PROBABILITY, 1usize)] = probability;
        let mut outcome = u32::new(comptime!(AttemptOutcome::Failed as i64));
        let required = config.records[item.field(ProjectLayout::PREREQUISITES, 1usize)];
        let outcomes = config.records[item.field(ProjectLayout::OUTCOMES, 1usize)];
        if (access(state, &lab_state) & required) != required
            || state[lab_state.field(LabState::DISCOVERED, 1usize)] & outcomes == outcomes
            || config.records[base.field(LabLayout::CONSENT, 1usize)] == 0
        {
            outcome = comptime!(AttemptOutcome::Ineligible as u32);
        }
        for good in 0..Good::COUNT {
            let available = read(main, main_coordinate(config, good as u32 + 1, owner));
            if outcome == comptime!(AttemptOutcome::Failed as u32)
                && lt(
                    available,
                    ticks(
                        config.records[item.field(ProjectLayout::INPUTS, Good::COUNT) + good],
                        0,
                    ),
                )
            {
                outcome = comptime!(AttemptOutcome::MissingResources as u32);
            }
        }
        if outcome == comptime!(AttemptOutcome::Failed as u32)
            && lt(read(main, time), ticks(effort, 0))
        {
            outcome = comptime!(AttemptOutcome::NoTime as u32);
        }
        if outcome == comptime!(AttemptOutcome::Failed as u32) {
            let random = samples[project as usize];
            for good in 0..Good::COUNT {
                let coordinate = main_coordinate(config, good as u32 + 1, owner);
                let used = ticks(
                    config.records[item.field(ProjectLayout::INPUTS, Good::COUNT) + good],
                    0,
                );
                let remaining = minus(read(main, coordinate), used);
                write(main, coordinate, remaining);
                increment(
                    state,
                    lab_state.field(LabState::CONSUMED, Good::COUNT * 2) + good * 2,
                    used,
                    control,
                );
            }
            let remaining = minus(read(main, time), ticks(effort, 0));
            write(main, time, remaining);
            increment(main, research_minutes, ticks(effort, 0), control);
            increment(
                state,
                saved.field(ProjectState::ATTEMPTS, 2usize),
                ticks(1, 0),
                control,
            );
            if random < probability as f32 / PROBABILITY_SCALE as f32 {
                outcome = comptime!(AttemptOutcome::Discovered as u32);
                increment(
                    state,
                    saved.field(ProjectState::DISCOVERIES, 2usize),
                    ticks(1, 0),
                    control,
                );
                state[lab_state.field(LabState::DISCOVERED, 1usize)] |=
                    config.records[item.field(ProjectLayout::OUTCOMES, 1usize)];
            }
        }
        state[saved.field(ProjectState::OUTCOME, 1usize)] = outcome;
        let signal = hazard(
            config.records[item.field(ProjectLayout::SIGNAL, 1usize)],
            table,
        );
        state[saved.field(ProjectState::EXPOSURE, 1usize)] =
            exposure + (RESEARCH_SCALE - exposure) * signal / PROBABILITY_SCALE;
        if config.records[base.field(LabLayout::ADOPTION, 1usize)] != 0 {
            state[lab_state.field(LabState::MASTERED, 1usize)] |=
                state[lab_state.field(LabState::DISCOVERED, 1usize)];
            state[lab_state.field(LabState::ADOPTED, 1usize)] |= state
                [lab_state.field(LabState::MASTERED, 1usize)]
                & state[lab_state.field(LabState::PERMITTED, 1usize)];
        }
    }
    if config.records[base.field(LabLayout::ADOPTION, 1usize)] != 0 {
        state[lab_state.field(LabState::MASTERED, 1usize)] |=
            state[lab_state.field(LabState::DISCOVERED, 1usize)];
        state[lab_state.field(LabState::ADOPTED, 1usize)] |= state
            [lab_state.field(LabState::MASTERED, 1usize)]
            & state[lab_state.field(LabState::PERMITTED, 1usize)];
    }
}

#[derive(CubeType, CubeLaunch)]
pub(super) struct ResearchInput {
    pub main: Box<[u32]>,
    pub state: Box<[u32]>,
    pub config: KernelResearchConfig,
    pub table: Box<[u32]>,
    pub samples: Box<[f32]>,
    pub output: Box<[u32]>,
    pub control: Box<[Atomic<u32>]>,
    #[cube(comptime)]
    pub stage: ResearchStage,
}

pub(super) struct ResearchSystem;
pub(super) struct EvaluateResearch;

impl System for ResearchSystem {
    type Input = ResearchInput;
    type Passes = gpu_passes![EvaluateResearch];
}

#[cube]
impl GpuPass<ResearchInput> for EvaluateResearch {
    type Output = ();

    fn execute(mut input: ResearchInput) {
        research(
            &mut input.main,
            &mut input.state,
            &input.config,
            &input.table,
            &input.samples,
            &mut input.output,
            &input.control,
            comptime!(input.stage),
        );
    }
}

#[cube]
fn research(
    main: &mut [u32],
    state: &mut [u32],
    config: &KernelResearchConfig,
    table: &[u32],
    samples: &[f32],
    output: &mut [u32],
    control: &[Atomic<u32>],
    #[comptime] stage: ResearchStage,
) {
    let id = ABSOLUTE_POS;
    match stage {
        ResearchStage::Opening => {
            if id < config.parameters.lab_count as usize {
                let lab = RowIndex::new(
                    config.parameters.labs as usize,
                    config.parameters.lab_count as usize,
                    id,
                );
                let saved = RowIndex::new(
                    config.parameters.lab_state as usize,
                    config.parameters.lab_count as usize,
                    id,
                );
                let owner = config.records[lab.field(LabLayout::OWNER, 1usize)];
                let first = config.records[lab.field(LabLayout::FIRST_METHOD, 1usize)];
                let end = first + config.records[lab.field(LabLayout::METHOD_COUNT, 1usize)];
                for index in first..end {
                    let method = RowIndex::new(
                        config.parameters.methods as usize,
                        config.parameters.method_count as usize,
                        index as usize,
                    );
                    let required = config.records[method.field(MethodLayout::REQUIREMENTS, 1usize)];
                    if access(state, &saved) & required == required {
                        write(
                            main,
                            main_coordinate(config, comptime!(Field::ActiveMethod as u32), owner),
                            ticks(
                                config.records[method.field(MethodLayout::IDENTITY, 1usize)],
                                0,
                            ),
                        );
                        write(
                            main,
                            main_coordinate(config, comptime!(Field::LaborMinutes as u32), owner),
                            ticks(
                                config.records[method.field(MethodLayout::LABOR_MINUTES, 1usize)],
                                0,
                            ),
                        );
                        write(
                            main,
                            main_coordinate(config, comptime!(Field::InputTicks as u32), owner),
                            ticks(
                                config.records[method.field(MethodLayout::INPUT_TICKS, 1usize)],
                                0,
                            ),
                        );
                        write(
                            main,
                            main_coordinate(config, comptime!(Field::OutputTicks as u32), owner),
                            ticks(
                                config.records[method.field(MethodLayout::OUTPUT_TICKS, 1usize)],
                                0,
                            ),
                        );
                        write(
                            main,
                            main_coordinate(config, comptime!(Field::FirstInput as u32), owner),
                            ticks(
                                config.records[method.field(MethodLayout::FIRST_INPUT, 1usize)],
                                0,
                            ),
                        );
                        write(
                            main,
                            main_coordinate(config, comptime!(Field::SecondInput as u32), owner),
                            ticks(
                                config.records[method.field(MethodLayout::SECOND_INPUT, 1usize)],
                                0,
                            ),
                        );
                    }
                }
            }
            if id < config.parameters.machine_count as usize {
                let item = RowIndex::new(
                    config.parameters.machines as usize,
                    config.parameters.machine_count as usize,
                    id,
                );
                let saved = RowIndex::new(
                    config.parameters.machine_state as usize,
                    config.parameters.machine_count as usize,
                    id,
                );
                let produced = read(
                    main,
                    main_coordinate(
                        config,
                        comptime!(Field::Produced as u32),
                        config.records[item.field(MachineLayout::FIRM, 1usize)],
                    ),
                );
                write(
                    state,
                    saved.field(MachineState::PRODUCED_BEFORE, 2usize),
                    produced,
                );
                state[saved.field(MachineState::MAINTENANCE_RESTORATION, 1usize)] = 0;
            }
            if id < config.parameters.stock_count as usize {
                let item = RowIndex::new(
                    config.parameters.stocks as usize,
                    config.parameters.stock_count as usize,
                    id,
                );
                let saved = RowIndex::new(
                    config.parameters.stock_state as usize,
                    config.parameters.stock_count as usize,
                    id,
                );
                let produced = read(
                    main,
                    main_coordinate(
                        config,
                        comptime!(Field::Produced as u32),
                        config.records[item.field(StockLayout::FIRM, 1usize)],
                    ),
                );
                write(
                    state,
                    saved.field(StockState::PRODUCED_BEFORE, 2usize),
                    produced,
                );
            }
            if id < config.parameters.project_count as usize {
                let saved = RowIndex::new(
                    config.parameters.project_state as usize,
                    config.parameters.project_count as usize,
                    id,
                );
                state[saved.field(ProjectState::OUTCOME, 1usize)] =
                    comptime!(AttemptOutcome::Idle as u32);
            }
        }
        ResearchStage::ProductionReadiness => {
            if id < config.parameters.lab_count as usize {
                let item = RowIndex::new(
                    config.parameters.labs as usize,
                    config.parameters.lab_count as usize,
                    id,
                );
                let saved = RowIndex::new(
                    config.parameters.lab_state as usize,
                    config.parameters.lab_count as usize,
                    id,
                );
                let required =
                    config.records[item.field(LabLayout::PRODUCTION_REQUIREMENTS, 1usize)];
                if (access(state, &saved) & required) != required {
                    cap_plan(
                        main,
                        config,
                        config.records[item.field(LabLayout::OWNER, 1usize)],
                        zero(),
                    );
                }
            }
        }
        ResearchStage::MachineCapacity => {
            if id < config.parameters.machine_count as usize {
                let item = RowIndex::new(
                    config.parameters.machines as usize,
                    config.parameters.machine_count as usize,
                    id,
                );
                let saved = RowIndex::new(
                    config.parameters.machine_state as usize,
                    config.parameters.machine_count as usize,
                    id,
                );
                let nominal = config.records[item.field(MachineLayout::NAMEPLATE, 1usize)]
                    - config.records[item.field(MachineLayout::DOWNTIME, 1usize)];
                let capacity = minimum(
                    read(state, saved.field(MachineState::REMAINING, 2usize)),
                    ticks(
                        state[saved.field(MachineState::CONDITION, 1usize)] * nominal
                            / RESEARCH_SCALE,
                        0,
                    ),
                );
                cap_plan(
                    main,
                    config,
                    config.records[item.field(MachineLayout::FIRM, 1usize)],
                    capacity,
                );
            }
        }
        ResearchStage::StockCapacity => {
            if id < config.parameters.stock_count as usize {
                let item = RowIndex::new(
                    config.parameters.stocks as usize,
                    config.parameters.stock_count as usize,
                    id,
                );
                let saved = RowIndex::new(
                    config.parameters.stock_state as usize,
                    config.parameters.stock_count as usize,
                    id,
                );
                cap_plan(
                    main,
                    config,
                    config.records[item.field(StockLayout::FIRM, 1usize)],
                    ticks(
                        state[saved.field(StockState::STOCK, 1usize)]
                            / config.records[item.field(StockLayout::EXTRACTION, 1usize)],
                        0,
                    ),
                );
            }
        }
        ResearchStage::Strike => {
            if id < config.parameters.strike_count as usize {
                let item = RowIndex::new(
                    config.parameters.strikes as usize,
                    config.parameters.strike_count as usize,
                    id,
                );
                let saved = RowIndex::new(
                    config.parameters.strike_state as usize,
                    config.parameters.strike_count as usize,
                    id,
                );
                let positive = 2 * config.records[item.field(StrikeLayout::GRIEVANCE, 1usize)]
                    + config.records[item.field(StrikeLayout::SOLIDARITY, 1usize)]
                    + config.records[item.field(StrikeLayout::TRUST, 1usize)]
                    + config.records[item.field(StrikeLayout::EFFICACY, 1usize)]
                    + config.records[item.field(StrikeLayout::PEERS, 1usize)];
                let negative = 2 * config.records[item.field(StrikeLayout::COST, 1usize)]
                    + config.records[item.field(StrikeLayout::RISK, 1usize)]
                    + 1500;
                let score = i32::cast_from(positive) - i32::cast_from(negative);
                let probability = sigmoid(score, table);
                let random = samples[id];
                let participation = select(
                    random < probability as f32 / PROBABILITY_SCALE as f32,
                    comptime!(Participation::Striking as u32),
                    comptime!(Participation::Working as u32),
                );
                state[saved.field(StrikeState::SCORE, 1usize)] = u32::cast_from(score + 4500);
                state[saved.field(StrikeState::PROBABILITY, 1usize)] = probability;
                state[saved.field(StrikeState::PARTICIPATION, 1usize)] = participation;
                if participation == comptime!(Participation::Striking as u32) {
                    write(
                        main,
                        main_coordinate(
                            config,
                            comptime!(Field::Time as u32),
                            config.records[item.field(StrikeLayout::HOUSEHOLD, 1usize)],
                        ),
                        zero(),
                    );
                }
            }
        }
        ResearchStage::Maintenance => {
            if id < config.parameters.machine_count as usize {
                let item = RowIndex::new(
                    config.parameters.machines as usize,
                    config.parameters.machine_count as usize,
                    id,
                );
                let saved = RowIndex::new(
                    config.parameters.machine_state as usize,
                    config.parameters.machine_count as usize,
                    id,
                );
                let coordinate = main_coordinate(
                    config,
                    config.records[item.field(MachineLayout::MAINTENANCE_GOOD, 1usize)],
                    config.records[item.field(MachineLayout::FIRM, 1usize)],
                );
                let required = ticks(
                    config.records[item.field(MachineLayout::MAINTENANCE_TICKS, 1usize)],
                    0,
                );
                let available = read(main, coordinate);
                if !lt(available, required) {
                    write(main, coordinate, minus(available, required));
                    increment(
                        state,
                        saved.field(MachineState::CONSUMED, 2usize),
                        required,
                        control,
                    );
                    state[saved.field(MachineState::MAINTENANCE_RESTORATION, 1usize)] = config
                        .records[item.field(MachineLayout::MAINTENANCE_TICKS, 1usize)]
                        * config.records[item.field(MachineLayout::RESTORATION, 1usize)];
                }
            }
        }
        ResearchStage::Attempt => {
            if id < config.parameters.lab_count as usize {
                attempt(main, state, config, table, samples, control, id);
            }
        }
        ResearchStage::MachineClose => {
            if id < config.parameters.machine_count as usize {
                let item = RowIndex::new(
                    config.parameters.machines as usize,
                    config.parameters.machine_count as usize,
                    id,
                );
                let saved = RowIndex::new(
                    config.parameters.machine_state as usize,
                    config.parameters.machine_count as usize,
                    id,
                );
                let produced = read(
                    main,
                    main_coordinate(
                        config,
                        comptime!(Field::Produced as u32),
                        config.records[item.field(MachineLayout::FIRM, 1usize)],
                    ),
                );
                let before = read(state, saved.field(MachineState::PRODUCED_BEFORE, 2usize));
                let used = divide(
                    positive(produced, before),
                    read(
                        main,
                        main_coordinate(
                            config,
                            comptime!(Field::OutputTicks as u32),
                            config.records[item.field(MachineLayout::FIRM, 1usize)],
                        ),
                    )
                    .extract(0usize),
                );
                let remaining = read(state, saved.field(MachineState::REMAINING, 2usize));
                let reproduced = times(
                    used,
                    read(
                        main,
                        main_coordinate(
                            config,
                            comptime!(Field::OutputTicks as u32),
                            config.records[item.field(MachineLayout::FIRM, 1usize)],
                        ),
                    )
                    .extract(0usize),
                );
                reproduced.faults.record(control);
                if lt(remaining, used)
                    || lt(produced, before)
                    || lt(
                        ticks(
                            config.records[item.field(MachineLayout::MAXIMUM_PACKETS, 1usize)],
                            0,
                        ),
                        used,
                    )
                    || !equal(reproduced.value, positive(produced, before))
                {
                    KernelFaults::new(comptime!(FaultFlags::RESEARCH_MISMATCH)).record(control);
                }
                write(
                    state,
                    saved.field(MachineState::REMAINING, 2usize),
                    positive(remaining, used),
                );
                let wear =
                    used.extract(0usize) * config.records[item.field(MachineLayout::WEAR, 1usize)];
                let repaired = state[saved.field(MachineState::CONDITION, 1usize)]
                    + state[saved.field(MachineState::MAINTENANCE_RESTORATION, 1usize)];
                state[saved.field(MachineState::CONDITION, 1usize)] =
                    min(RESEARCH_SCALE, repaired - min(repaired, wear));
                state[saved.field(MachineState::MAINTENANCE_RESTORATION, 1usize)] = 0;
            }
        }
        ResearchStage::StockClose => {
            if id < config.parameters.stock_count as usize {
                let item = RowIndex::new(
                    config.parameters.stocks as usize,
                    config.parameters.stock_count as usize,
                    id,
                );
                let saved = RowIndex::new(
                    config.parameters.stock_state as usize,
                    config.parameters.stock_count as usize,
                    id,
                );
                let produced = read(
                    main,
                    main_coordinate(
                        config,
                        comptime!(Field::Produced as u32),
                        config.records[item.field(StockLayout::FIRM, 1usize)],
                    ),
                );
                let before = read(state, saved.field(StockState::PRODUCED_BEFORE, 2usize));
                let used = divide(
                    positive(produced, before),
                    read(
                        main,
                        main_coordinate(
                            config,
                            comptime!(Field::OutputTicks as u32),
                            config.records[item.field(StockLayout::FIRM, 1usize)],
                        ),
                    )
                    .extract(0usize),
                );
                let extraction = times(
                    used,
                    config.records[item.field(StockLayout::EXTRACTION, 1usize)],
                );
                extraction.faults.record(control);
                let stock = state[saved.field(StockState::STOCK, 1usize)];
                let reproduced = times(
                    used,
                    read(
                        main,
                        main_coordinate(
                            config,
                            comptime!(Field::OutputTicks as u32),
                            config.records[item.field(StockLayout::FIRM, 1usize)],
                        ),
                    )
                    .extract(0usize),
                );
                reproduced.faults.record(control);
                if lt(ticks(stock, 0), extraction.value)
                    || lt(produced, before)
                    || lt(
                        ticks(
                            config.records[item.field(StockLayout::MAXIMUM_PACKETS, 1usize)],
                            0,
                        ),
                        used,
                    )
                    || !equal(reproduced.value, positive(produced, before))
                {
                    KernelFaults::new(comptime!(FaultFlags::RESEARCH_MISMATCH)).record(control);
                }
                let remaining = positive(ticks(stock, 0), extraction.value).extract(0usize);
                let capacity = config.records[item.field(StockLayout::CAPACITY, 1usize)];
                let growth = ratio(
                    ticks(remaining, 0),
                    config.records[item.field(StockLayout::RENEWAL, 1usize)]
                        * (capacity - remaining),
                    capacity * RESEARCH_SCALE,
                );
                growth.faults.record(control);
                state[saved.field(StockState::STOCK, 1usize)] =
                    remaining + growth.value.extract(0usize);
                increment(
                    state,
                    saved.field(StockState::EXTRACTED, 2usize),
                    extraction.value,
                    control,
                );
            }
        }
        ResearchStage::Summary => {
            let metrics = comptime!(SummaryLayout::WORDS / 2);
            let shards = output.len() / SummaryLayout::WORDS;
            let shard = id / metrics;
            let metric = id % metrics;
            if shard < shards {
                let mut total = zero();
                let count = config.parameters.lab_count as usize;
                let chunk = ceildiv(count as u32, shards as u32) as usize;
                for lab in shard * chunk..min((shard + 1) * chunk, count) {
                    if metric < SummaryLayout::CONSUMED / 2 + Good::COUNT {
                        total = add(
                            control,
                            total,
                            read(
                                state,
                                RowIndex::new(
                                    config.parameters.lab_state as usize,
                                    config.parameters.lab_count as usize,
                                    lab,
                                )
                                .field(LabState::CONSUMED, Good::COUNT * 2)
                                    + metric * 2,
                            ),
                        );
                    }
                }
                let count = config.parameters.project_count as usize;
                let chunk = ceildiv(count as u32, shards as u32) as usize;
                for project in shard * chunk..min((shard + 1) * chunk, count) {
                    let saved = RowIndex::new(
                        config.parameters.project_state as usize,
                        config.parameters.project_count as usize,
                        project,
                    );
                    if metric == SummaryLayout::ATTEMPTS / 2 {
                        total = add(
                            control,
                            total,
                            read(state, saved.field(ProjectState::ATTEMPTS, 2usize)),
                        );
                    }
                    if metric == SummaryLayout::DISCOVERIES / 2 {
                        total = add(
                            control,
                            total,
                            read(state, saved.field(ProjectState::DISCOVERIES, 2usize)),
                        );
                    }
                }
                let count = config.parameters.machine_count as usize;
                let chunk = ceildiv(count as u32, shards as u32) as usize;
                for machine in shard * chunk..min((shard + 1) * chunk, count) {
                    let saved = RowIndex::new(
                        config.parameters.machine_state as usize,
                        config.parameters.machine_count as usize,
                        machine,
                    );
                    let item = RowIndex::new(
                        config.parameters.machines as usize,
                        config.parameters.machine_count as usize,
                        machine,
                    );
                    if metric < SummaryLayout::CONSUMED / 2 + Good::COUNT
                        && config.records[item.field(MachineLayout::MAINTENANCE_GOOD, 1usize)]
                            == metric as u32 + 1
                    {
                        total = add(
                            control,
                            total,
                            read(state, saved.field(MachineState::CONSUMED, 2usize)),
                        );
                    }
                    if metric == SummaryLayout::REMAINING_SERVICE / 2 {
                        total = add(
                            control,
                            total,
                            read(state, saved.field(MachineState::REMAINING, 2usize)),
                        );
                    }
                }
                let count = config.parameters.stock_count as usize;
                let chunk = ceildiv(count as u32, shards as u32) as usize;
                for stock in shard * chunk..min((shard + 1) * chunk, count) {
                    if metric == SummaryLayout::NATURAL_STOCK / 2 {
                        total = add(
                            control,
                            total,
                            ticks(
                                state[RowIndex::new(
                                    config.parameters.stock_state as usize,
                                    config.parameters.stock_count as usize,
                                    stock,
                                )
                                .field(StockState::STOCK, 1usize)],
                                0,
                            ),
                        );
                    }
                }
                let count = config.parameters.strike_count as usize;
                let chunk = ceildiv(count as u32, shards as u32) as usize;
                for strike in shard * chunk..min((shard + 1) * chunk, count) {
                    if metric == SummaryLayout::STRIKING / 2 {
                        total = add(
                            control,
                            total,
                            ticks(
                                state[RowIndex::new(
                                    config.parameters.strike_state as usize,
                                    config.parameters.strike_count as usize,
                                    strike,
                                )
                                .field(StrikeState::PARTICIPATION, 1usize)],
                                0,
                            ),
                        );
                    }
                }
                write(output, id * 2, total);
            }
        }
        ResearchStage::Audit => {
            if id < config.parameters.lab_count as usize {
                let saved = RowIndex::new(
                    config.parameters.lab_state as usize,
                    config.parameters.lab_count as usize,
                    id,
                );
                let item = RowIndex::new(
                    config.parameters.labs as usize,
                    config.parameters.lab_count as usize,
                    id,
                );
                if state[saved.field(LabState::MASTERED, 1usize)]
                    & !state[saved.field(LabState::DISCOVERED, 1usize)]
                    != 0
                    || state[saved.field(LabState::ADOPTED, 1usize)]
                        & !(state[saved.field(LabState::MASTERED, 1usize)]
                            & state[saved.field(LabState::PERMITTED, 1usize)])
                        != 0
                    || state[saved.field(LabState::PERMITTED, 1usize)]
                        != config.records[item.field(LabLayout::PERMITTED, 1usize)]
                    || config.records[item.field(LabLayout::DISCOVERED, 1usize)]
                        & !state[saved.field(LabState::DISCOVERED, 1usize)]
                        != 0
                    || config.records[item.field(LabLayout::MASTERED, 1usize)]
                        & !state[saved.field(LabState::MASTERED, 1usize)]
                        != 0
                    || config.records[item.field(LabLayout::ADOPTED, 1usize)]
                        & !state[saved.field(LabState::ADOPTED, 1usize)]
                        != 0
                {
                    KernelFaults::new(comptime!(FaultFlags::RESEARCH_MISMATCH)).record(control);
                }
            }
            if id < config.parameters.project_count as usize {
                let saved = RowIndex::new(
                    config.parameters.project_state as usize,
                    config.parameters.project_count as usize,
                    id,
                );
                if state[saved.field(ProjectState::EXPOSURE, 1usize)] > RESEARCH_SCALE
                    || state[saved.field(ProjectState::OUTCOME, 1usize)]
                        > comptime!(AttemptOutcome::Discovered as u32)
                    || state[saved.field(ProjectState::PROBABILITY, 1usize)] > PROBABILITY_SCALE
                    || lt(
                        read(state, saved.field(ProjectState::ATTEMPTS, 2usize)),
                        read(state, saved.field(ProjectState::DISCOVERIES, 2usize)),
                    )
                {
                    KernelFaults::new(comptime!(FaultFlags::RESEARCH_MISMATCH)).record(control);
                }
            }
            if id < config.parameters.machine_count as usize {
                let saved = RowIndex::new(
                    config.parameters.machine_state as usize,
                    config.parameters.machine_count as usize,
                    id,
                );
                let item = RowIndex::new(
                    config.parameters.machines as usize,
                    config.parameters.machine_count as usize,
                    id,
                );
                let initial = ticks(
                    config.records[item.field(MachineLayout::INITIAL_SERVICE, 2usize)],
                    config.records[item.field(MachineLayout::INITIAL_SERVICE, 2usize) + 1],
                );
                if state[saved.field(MachineState::CONDITION, 1usize)] > RESEARCH_SCALE
                    || state[saved.field(MachineState::MAINTENANCE_RESTORATION, 1usize)] != 0
                    || lt(
                        initial,
                        read(state, saved.field(MachineState::REMAINING, 2usize)),
                    )
                {
                    KernelFaults::new(comptime!(FaultFlags::RESEARCH_MISMATCH)).record(control);
                }
            }
            if id < config.parameters.stock_count as usize {
                let saved = RowIndex::new(
                    config.parameters.stock_state as usize,
                    config.parameters.stock_count as usize,
                    id,
                );
                let item = RowIndex::new(
                    config.parameters.stocks as usize,
                    config.parameters.stock_count as usize,
                    id,
                );
                if state[saved.field(StockState::STOCK, 1usize)]
                    > config.records[item.field(StockLayout::CAPACITY, 1usize)]
                {
                    KernelFaults::new(comptime!(FaultFlags::RESEARCH_MISMATCH)).record(control);
                }
            }
            if id < config.parameters.strike_count as usize {
                let saved = RowIndex::new(
                    config.parameters.strike_state as usize,
                    config.parameters.strike_count as usize,
                    id,
                );
                if state[saved.field(StrikeState::SCORE, 1usize)] > 9000
                    || state[saved.field(StrikeState::PROBABILITY, 1usize)] > PROBABILITY_SCALE
                    || state[saved.field(StrikeState::PARTICIPATION, 1usize)]
                        > comptime!(Participation::Striking as u32)
                {
                    KernelFaults::new(comptime!(FaultFlags::RESEARCH_MISMATCH)).record(control);
                }
            }
        }
    }
}

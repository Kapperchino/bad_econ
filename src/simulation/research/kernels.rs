use super::super::{
    flags::{FaultFlags, KernelFaults},
    words::*,
};
use super::{layout::*, nonlinear::*, random::*};
use crate::domain::{Field, Phase, research::RESEARCH_SCALE};
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
    Copy,
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
fn main_coordinate(config: &[u32], field: u32, actor: u32) -> usize {
    ((field * config[Header::ACTORS] + actor) * 2) as usize
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
fn cap_plan(main: &mut [u32], config: &[u32], firm: u32, capacity: Ticks) {
    let plan = main_coordinate(config, comptime!(Field::Plan as u32), firm);
    let slots = main_coordinate(config, comptime!(Field::Slots as u32), firm);
    let capped = minimum(read(main, plan), capacity);
    write(main, plan, capped);
    write(main, slots, capped);
}

#[cube]
fn access(state: &mut [u32], base: usize) -> u32 {
    state[base + LabState::DISCOVERED]
        & state[base + LabState::MASTERED]
        & state[base + LabState::PERMITTED]
        & state[base + LabState::ADOPTED]
}

#[cube]
fn seed(config: &[u32]) -> Ticks {
    ticks(config[Header::SEED], config[Header::SEED + 1])
}

#[cube]
fn attempt(
    main: &mut [u32],
    state: &mut [u32],
    config: &[u32],
    table: &[u32],
    constants: &[u32],
    control: &[Atomic<u32>],
    day: u32,
    id: usize,
) {
    let base = config[Header::LABS] as usize + id * LabLayout::WORDS;
    let lab_state = config[Header::LAB_STATE] as usize + id * LabState::WORDS;
    let owner = config[base + LabLayout::OWNER];
    let researcher = config[base + LabLayout::RESEARCHER];
    let time = main_coordinate(config, comptime!(Field::Time as u32), researcher);
    let research_minutes =
        main_coordinate(config, comptime!(Field::ResearchMinutes as u32), researcher);
    let start = config[base + LabLayout::FIRST_PROJECT];
    let end = start + config[base + LabLayout::PROJECT_COUNT];
    for project in start..end {
        let item = config[Header::PROJECTS] as usize + project as usize * ProjectLayout::WORDS;
        let saved = config[Header::PROJECT_STATE] as usize + project as usize * ProjectState::WORDS;
        let exposure = state[saved + ProjectState::EXPOSURE];
        let correlated = exposure
            * config[item + ProjectLayout::CAPACITY]
            * config[item + ProjectLayout::RELATEDNESS]
            / (RESEARCH_SCALE * RESEARCH_SCALE);
        let mut power = u32::new(comptime!(RESEARCH_SCALE as i64));
        for _step in 0..config[item + ProjectLayout::SHAPE] {
            power = power * (RESEARCH_SCALE - correlated) / RESEARCH_SCALE;
        }
        let scale = config[item + ProjectLayout::MINIMUM_EFFORT]
            + (config[item + ProjectLayout::PIONEER_EFFORT]
                - config[item + ProjectLayout::MINIMUM_EFFORT])
                * power
                / RESEARCH_SCALE;
        let effort = config[item + ProjectLayout::LABOR_MINUTES];
        let probability = hazard(
            min(effort * RESEARCH_SCALE / scale, 16 * RESEARCH_SCALE),
            table,
        );
        state[saved + ProjectState::PROBABILITY] = probability;
        let mut outcome = u32::new(comptime!(AttemptOutcome::Failed as i64));
        let required = config[item + ProjectLayout::PREREQUISITES];
        if (access(state, lab_state) & required) != required
            || config[base + LabLayout::CONSENT] == 0
        {
            outcome = comptime!(AttemptOutcome::Ineligible as u32);
        }
        for good in 0..3 {
            let available = read(main, main_coordinate(config, good as u32 + 1, owner));
            if outcome == comptime!(AttemptOutcome::Failed as u32)
                && lt(
                    available,
                    ticks(config[item + ProjectLayout::INPUTS + good], 0),
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
            let attempts = read(state, saved + ProjectState::ATTEMPTS);
            let random = draw(
                RandomKey {
                    seed: seed(config),
                    day: ticks(day, 0),
                    phase: comptime!(Phase::PhysicalClose as u32),
                    actor: ticks(
                        config[item + ProjectLayout::IDENTITY],
                        config[item + ProjectLayout::IDENTITY + 1],
                    ),
                    purpose: 0x52535243,
                    index: attempts,
                },
                constants,
            );
            for good in 0..3 {
                let coordinate = main_coordinate(config, good as u32 + 1, owner);
                let used = ticks(config[item + ProjectLayout::INPUTS + good], 0);
                let remaining = minus(read(main, coordinate), used);
                write(main, coordinate, remaining);
                increment(
                    state,
                    lab_state + LabState::CONSUMED + good * 2,
                    used,
                    control,
                );
            }
            let remaining = minus(read(main, time), ticks(effort, 0));
            write(main, time, remaining);
            increment(main, research_minutes, ticks(effort, 0), control);
            increment(state, saved + ProjectState::ATTEMPTS, ticks(1, 0), control);
            if succeeds(probability, random) {
                outcome = comptime!(AttemptOutcome::Discovered as u32);
                increment(
                    state,
                    saved + ProjectState::DISCOVERIES,
                    ticks(1, 0),
                    control,
                );
                state[lab_state + LabState::DISCOVERED] |= config[item + ProjectLayout::OUTCOMES];
            }
        }
        state[saved + ProjectState::OUTCOME] = outcome;
        let signal = hazard(config[item + ProjectLayout::SIGNAL], table);
        state[saved + ProjectState::EXPOSURE] =
            exposure + (RESEARCH_SCALE - exposure) * signal / PROBABILITY_SCALE;
        if config[base + LabLayout::ADOPTION] != 0 {
            state[lab_state + LabState::MASTERED] |= state[lab_state + LabState::DISCOVERED];
            state[lab_state + LabState::ADOPTED] |=
                state[lab_state + LabState::MASTERED] & state[lab_state + LabState::PERMITTED];
        }
    }
    if config[base + LabLayout::ADOPTION] != 0 {
        state[lab_state + LabState::MASTERED] |= state[lab_state + LabState::DISCOVERED];
        state[lab_state + LabState::ADOPTED] |=
            state[lab_state + LabState::MASTERED] & state[lab_state + LabState::PERMITTED];
    }
}

#[cube(launch)]
pub fn research(
    main: &mut [u32],
    state: &mut [u32],
    config: &[u32],
    table: &[u32],
    constants: &[u32],
    output: &mut [u32],
    control: &[Atomic<u32>],
    day: u32,
    #[comptime] stage: ResearchStage,
) {
    let id = ABSOLUTE_POS;
    match stage {
        ResearchStage::Copy => {
            if id < state.len() {
                output[id] = state[id];
            }
        }
        ResearchStage::Opening => {
            if id < config[Header::MACHINE_COUNT] as usize {
                let item = config[Header::MACHINES] as usize + id * MachineLayout::WORDS;
                let saved = config[Header::MACHINE_STATE] as usize + id * MachineState::WORDS;
                let produced = read(
                    main,
                    main_coordinate(
                        config,
                        comptime!(Field::Produced as u32),
                        config[item + MachineLayout::FIRM],
                    ),
                );
                write(state, saved + MachineState::PRODUCED_BEFORE, produced);
                state[saved + MachineState::MAINTENANCE_RESTORATION] = 0;
            }
            if id < config[Header::STOCK_COUNT] as usize {
                let item = config[Header::STOCKS] as usize + id * StockLayout::WORDS;
                let saved = config[Header::STOCK_STATE] as usize + id * StockState::WORDS;
                let produced = read(
                    main,
                    main_coordinate(
                        config,
                        comptime!(Field::Produced as u32),
                        config[item + StockLayout::FIRM],
                    ),
                );
                write(state, saved + StockState::PRODUCED_BEFORE, produced);
            }
            if id < config[Header::PROJECT_COUNT] as usize {
                let saved = config[Header::PROJECT_STATE] as usize + id * ProjectState::WORDS;
                state[saved + ProjectState::OUTCOME] = comptime!(AttemptOutcome::Idle as u32);
            }
        }
        ResearchStage::ProductionReadiness => {
            if id < config[Header::LAB_COUNT] as usize {
                let item = config[Header::LABS] as usize + id * LabLayout::WORDS;
                let saved = config[Header::LAB_STATE] as usize + id * LabState::WORDS;
                let required = config[item + LabLayout::PRODUCTION_REQUIREMENTS];
                if (access(state, saved) & required) != required {
                    cap_plan(main, config, config[item + LabLayout::OWNER], zero());
                }
            }
        }
        ResearchStage::MachineCapacity => {
            if id < config[Header::MACHINE_COUNT] as usize {
                let item = config[Header::MACHINES] as usize + id * MachineLayout::WORDS;
                let saved = config[Header::MACHINE_STATE] as usize + id * MachineState::WORDS;
                let nominal = config[item + MachineLayout::NAMEPLATE]
                    - config[item + MachineLayout::DOWNTIME];
                let capacity = minimum(
                    read(state, saved + MachineState::REMAINING),
                    ticks(
                        state[saved + MachineState::CONDITION] * nominal / RESEARCH_SCALE,
                        0,
                    ),
                );
                cap_plan(main, config, config[item + MachineLayout::FIRM], capacity);
            }
        }
        ResearchStage::StockCapacity => {
            if id < config[Header::STOCK_COUNT] as usize {
                let item = config[Header::STOCKS] as usize + id * StockLayout::WORDS;
                let saved = config[Header::STOCK_STATE] as usize + id * StockState::WORDS;
                cap_plan(
                    main,
                    config,
                    config[item + StockLayout::FIRM],
                    ticks(
                        state[saved + StockState::STOCK] / config[item + StockLayout::EXTRACTION],
                        0,
                    ),
                );
            }
        }
        ResearchStage::Strike => {
            if id < config[Header::STRIKE_COUNT] as usize {
                let item = config[Header::STRIKES] as usize + id * StrikeLayout::WORDS;
                let saved = config[Header::STRIKE_STATE] as usize + id * StrikeState::WORDS;
                let positive = 2 * config[item + StrikeLayout::GRIEVANCE]
                    + config[item + StrikeLayout::SOLIDARITY]
                    + config[item + StrikeLayout::TRUST]
                    + config[item + StrikeLayout::EFFICACY]
                    + config[item + StrikeLayout::PEERS];
                let negative = 2 * config[item + StrikeLayout::COST]
                    + config[item + StrikeLayout::RISK]
                    + 1500;
                let score = i32::cast_from(positive) - i32::cast_from(negative);
                let probability = sigmoid(score, table);
                let random = draw(
                    RandomKey {
                        seed: seed(config),
                        day: ticks(day, 0),
                        phase: comptime!(Phase::Observation as u32),
                        actor: ticks(
                            config[item + StrikeLayout::IDENTITY],
                            config[item + StrikeLayout::IDENTITY + 1],
                        ),
                        purpose: 0x5354524b,
                        index: zero(),
                    },
                    constants,
                );
                let participation = select(
                    succeeds(probability, random),
                    comptime!(Participation::Striking as u32),
                    comptime!(Participation::Working as u32),
                );
                state[saved + StrikeState::SCORE] = u32::cast_from(score + 4500);
                state[saved + StrikeState::PROBABILITY] = probability;
                state[saved + StrikeState::PARTICIPATION] = participation;
                if participation == comptime!(Participation::Striking as u32) {
                    write(
                        main,
                        main_coordinate(
                            config,
                            comptime!(Field::Time as u32),
                            config[item + StrikeLayout::HOUSEHOLD],
                        ),
                        zero(),
                    );
                }
            }
        }
        ResearchStage::Maintenance => {
            if id < config[Header::MACHINE_COUNT] as usize {
                let item = config[Header::MACHINES] as usize + id * MachineLayout::WORDS;
                let saved = config[Header::MACHINE_STATE] as usize + id * MachineState::WORDS;
                let coordinate = main_coordinate(
                    config,
                    config[item + MachineLayout::MAINTENANCE_GOOD],
                    config[item + MachineLayout::FIRM],
                );
                let required = ticks(config[item + MachineLayout::MAINTENANCE_TICKS], 0);
                let available = read(main, coordinate);
                if !lt(available, required) {
                    write(main, coordinate, minus(available, required));
                    increment(state, saved + MachineState::CONSUMED, required, control);
                    state[saved + MachineState::MAINTENANCE_RESTORATION] = config
                        [item + MachineLayout::MAINTENANCE_TICKS]
                        * config[item + MachineLayout::RESTORATION];
                }
            }
        }
        ResearchStage::Attempt => {
            if id < config[Header::LAB_COUNT] as usize {
                attempt(main, state, config, table, constants, control, day, id);
            }
        }
        ResearchStage::MachineClose => {
            if id < config[Header::MACHINE_COUNT] as usize {
                let item = config[Header::MACHINES] as usize + id * MachineLayout::WORDS;
                let saved = config[Header::MACHINE_STATE] as usize + id * MachineState::WORDS;
                let produced = read(
                    main,
                    main_coordinate(
                        config,
                        comptime!(Field::Produced as u32),
                        config[item + MachineLayout::FIRM],
                    ),
                );
                let before = read(state, saved + MachineState::PRODUCED_BEFORE);
                let used = divide(
                    positive(produced, before),
                    config[item + MachineLayout::OUTPUT_TICKS],
                );
                let remaining = read(state, saved + MachineState::REMAINING);
                let reproduced = times(used, config[item + MachineLayout::OUTPUT_TICKS]);
                reproduced.faults.record(control);
                if lt(remaining, used)
                    || lt(produced, before)
                    || lt(
                        ticks(config[item + MachineLayout::MAXIMUM_PACKETS], 0),
                        used,
                    )
                    || !equal(reproduced.value, positive(produced, before))
                {
                    KernelFaults::new(comptime!(FaultFlags::RESEARCH_MISMATCH)).record(control);
                }
                write(
                    state,
                    saved + MachineState::REMAINING,
                    positive(remaining, used),
                );
                let wear = used.extract(0usize) * config[item + MachineLayout::WEAR];
                let repaired = state[saved + MachineState::CONDITION]
                    + state[saved + MachineState::MAINTENANCE_RESTORATION];
                state[saved + MachineState::CONDITION] =
                    min(RESEARCH_SCALE, repaired - min(repaired, wear));
                state[saved + MachineState::MAINTENANCE_RESTORATION] = 0;
            }
        }
        ResearchStage::StockClose => {
            if id < config[Header::STOCK_COUNT] as usize {
                let item = config[Header::STOCKS] as usize + id * StockLayout::WORDS;
                let saved = config[Header::STOCK_STATE] as usize + id * StockState::WORDS;
                let produced = read(
                    main,
                    main_coordinate(
                        config,
                        comptime!(Field::Produced as u32),
                        config[item + StockLayout::FIRM],
                    ),
                );
                let before = read(state, saved + StockState::PRODUCED_BEFORE);
                let used = divide(
                    positive(produced, before),
                    config[item + StockLayout::OUTPUT_TICKS],
                );
                let extraction = times(used, config[item + StockLayout::EXTRACTION]);
                extraction.faults.record(control);
                let stock = state[saved + StockState::STOCK];
                let reproduced = times(used, config[item + StockLayout::OUTPUT_TICKS]);
                reproduced.faults.record(control);
                if lt(ticks(stock, 0), extraction.value)
                    || lt(produced, before)
                    || lt(ticks(config[item + StockLayout::MAXIMUM_PACKETS], 0), used)
                    || !equal(reproduced.value, positive(produced, before))
                {
                    KernelFaults::new(comptime!(FaultFlags::RESEARCH_MISMATCH)).record(control);
                }
                let remaining = positive(ticks(stock, 0), extraction.value).extract(0usize);
                let capacity = config[item + StockLayout::CAPACITY];
                let growth = ratio(
                    ticks(remaining, 0),
                    config[item + StockLayout::RENEWAL] * (capacity - remaining),
                    capacity * RESEARCH_SCALE,
                );
                growth.faults.record(control);
                state[saved + StockState::STOCK] = remaining + growth.value.extract(0usize);
                increment(
                    state,
                    saved + StockState::EXTRACTED,
                    extraction.value,
                    control,
                );
            }
        }
        ResearchStage::Summary => {
            if id < SummaryLayout::WORDS / 2 {
                let mut total = zero();
                for lab in 0..config[Header::LAB_COUNT] as usize {
                    if id < SummaryLayout::CONSUMED / 2 + 3 {
                        total = add(
                            control,
                            total,
                            read(
                                state,
                                config[Header::LAB_STATE] as usize
                                    + lab * LabState::WORDS
                                    + LabState::CONSUMED
                                    + id * 2,
                            ),
                        );
                    }
                }
                for project in 0..config[Header::PROJECT_COUNT] as usize {
                    let saved =
                        config[Header::PROJECT_STATE] as usize + project * ProjectState::WORDS;
                    if id == SummaryLayout::ATTEMPTS / 2 {
                        total = add(control, total, read(state, saved + ProjectState::ATTEMPTS));
                    }
                    if id == SummaryLayout::DISCOVERIES / 2 {
                        total = add(
                            control,
                            total,
                            read(state, saved + ProjectState::DISCOVERIES),
                        );
                    }
                }
                for machine in 0..config[Header::MACHINE_COUNT] as usize {
                    let saved =
                        config[Header::MACHINE_STATE] as usize + machine * MachineState::WORDS;
                    let item = config[Header::MACHINES] as usize + machine * MachineLayout::WORDS;
                    if id < SummaryLayout::CONSUMED / 2 + 3
                        && config[item + MachineLayout::MAINTENANCE_GOOD] == id as u32 + 1
                    {
                        total = add(control, total, read(state, saved + MachineState::CONSUMED));
                    }
                    if id == SummaryLayout::REMAINING_SERVICE / 2 {
                        total = add(control, total, read(state, saved + MachineState::REMAINING));
                    }
                }
                for stock in 0..config[Header::STOCK_COUNT] as usize {
                    if id == SummaryLayout::NATURAL_STOCK / 2 {
                        total = add(
                            control,
                            total,
                            ticks(
                                state[config[Header::STOCK_STATE] as usize
                                    + stock * StockState::WORDS
                                    + StockState::STOCK],
                                0,
                            ),
                        );
                    }
                }
                for strike in 0..config[Header::STRIKE_COUNT] as usize {
                    if id == SummaryLayout::STRIKING / 2 {
                        total = add(
                            control,
                            total,
                            ticks(
                                state[config[Header::STRIKE_STATE] as usize
                                    + strike * StrikeState::WORDS
                                    + StrikeState::PARTICIPATION],
                                0,
                            ),
                        );
                    }
                }
                write(output, id * 2, total);
            }
        }
        ResearchStage::Audit => {
            if id < config[Header::LAB_COUNT] as usize {
                let saved = config[Header::LAB_STATE] as usize + id * LabState::WORDS;
                let item = config[Header::LABS] as usize + id * LabLayout::WORDS;
                if state[saved + LabState::MASTERED] & !state[saved + LabState::DISCOVERED] != 0
                    || state[saved + LabState::ADOPTED]
                        & !(state[saved + LabState::MASTERED] & state[saved + LabState::PERMITTED])
                        != 0
                    || state[saved + LabState::PERMITTED] != config[item + LabLayout::PERMITTED]
                    || config[item + LabLayout::DISCOVERED] & !state[saved + LabState::DISCOVERED]
                        != 0
                    || config[item + LabLayout::MASTERED] & !state[saved + LabState::MASTERED] != 0
                    || config[item + LabLayout::ADOPTED] & !state[saved + LabState::ADOPTED] != 0
                {
                    KernelFaults::new(comptime!(FaultFlags::RESEARCH_MISMATCH)).record(control);
                }
            }
            if id < config[Header::PROJECT_COUNT] as usize {
                let saved = config[Header::PROJECT_STATE] as usize + id * ProjectState::WORDS;
                if state[saved + ProjectState::EXPOSURE] > RESEARCH_SCALE
                    || state[saved + ProjectState::OUTCOME]
                        > comptime!(AttemptOutcome::Discovered as u32)
                    || state[saved + ProjectState::PROBABILITY] > PROBABILITY_SCALE
                    || lt(
                        read(state, saved + ProjectState::ATTEMPTS),
                        read(state, saved + ProjectState::DISCOVERIES),
                    )
                {
                    KernelFaults::new(comptime!(FaultFlags::RESEARCH_MISMATCH)).record(control);
                }
            }
            if id < config[Header::MACHINE_COUNT] as usize {
                let saved = config[Header::MACHINE_STATE] as usize + id * MachineState::WORDS;
                let item = config[Header::MACHINES] as usize + id * MachineLayout::WORDS;
                let initial = ticks(
                    config[item + MachineLayout::INITIAL_SERVICE],
                    config[item + MachineLayout::INITIAL_SERVICE + 1],
                );
                if state[saved + MachineState::CONDITION] > RESEARCH_SCALE
                    || state[saved + MachineState::MAINTENANCE_RESTORATION] != 0
                    || lt(initial, read(state, saved + MachineState::REMAINING))
                {
                    KernelFaults::new(comptime!(FaultFlags::RESEARCH_MISMATCH)).record(control);
                }
            }
            if id < config[Header::STOCK_COUNT] as usize {
                let saved = config[Header::STOCK_STATE] as usize + id * StockState::WORDS;
                let item = config[Header::STOCKS] as usize + id * StockLayout::WORDS;
                if state[saved + StockState::STOCK] > config[item + StockLayout::CAPACITY] {
                    KernelFaults::new(comptime!(FaultFlags::RESEARCH_MISMATCH)).record(control);
                }
            }
            if id < config[Header::STRIKE_COUNT] as usize {
                let saved = config[Header::STRIKE_STATE] as usize + id * StrikeState::WORDS;
                if state[saved + StrikeState::SCORE] > 9000
                    || state[saved + StrikeState::PROBABILITY] > PROBABILITY_SCALE
                    || state[saved + StrikeState::PARTICIPATION]
                        > comptime!(Participation::Striking as u32)
                {
                    KernelFaults::new(comptime!(FaultFlags::RESEARCH_MISMATCH)).record(control);
                }
            }
        }
    }
}

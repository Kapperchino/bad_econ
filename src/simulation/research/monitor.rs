use super::{
    access::{ResearchView, ResearchViewExpand},
    kernels::{AttemptOutcome, Participation},
    layout::*,
    nonlinear::PROBABILITY_SCALE,
};
use crate::{
    domain::{Good, research::RESEARCH_SCALE},
    simulation::storage::{layout::TICK_WORDS, words::*},
};
use cubecl::prelude::*;

#[cube]
impl<'a> ResearchView<'a> {
    pub fn summarize(&mut self, id: usize) {
        let metrics = comptime!(SummaryLayout::WORDS / TICK_WORDS);
        let shards = self.output.len() / SummaryLayout::WORDS;
        let shard = id / metrics;
        let metric = id % metrics;
        if shard < shards {
            let mut total = zero();
            let count = self.parameters.lab_count as usize;
            let chunk = ceildiv(count as u32, shards as u32) as usize;
            if metric < SummaryLayout::CONSUMED / TICK_WORDS + Good::COUNT {
                for index in shard * chunk..min((shard + 1) * chunk, count) {
                    let lab = self.parameters.lab(index);
                    total = plus(total, self.state.lab_consumed(&lab.saved, metric))
                        .record(self.control);
                }
            }
            let count = self.parameters.project_count as usize;
            let chunk = ceildiv(count as u32, shards as u32) as usize;
            for index in shard * chunk..min((shard + 1) * chunk, count) {
                let project = self.parameters.project(index);
                if metric == SummaryLayout::ATTEMPTS / TICK_WORDS {
                    total = plus(
                        total,
                        self.state.amount(&project.saved, ProjectState::ATTEMPTS),
                    )
                    .record(self.control);
                }
                if metric == SummaryLayout::DISCOVERIES / TICK_WORDS {
                    total = plus(
                        total,
                        self.state.amount(&project.saved, ProjectState::DISCOVERIES),
                    )
                    .record(self.control);
                }
            }
            let count = self.parameters.machine_count as usize;
            let chunk = ceildiv(count as u32, shards as u32) as usize;
            for index in shard * chunk..min((shard + 1) * chunk, count) {
                let machine = self.parameters.machine(index);
                if metric < SummaryLayout::CONSUMED / TICK_WORDS + Good::COUNT
                    && self
                        .records
                        .number(&machine.definition, MachineLayout::MAINTENANCE_GOOD)
                        == metric as u32 + 1
                {
                    total = plus(
                        total,
                        self.state.amount(&machine.saved, MachineState::CONSUMED),
                    )
                    .record(self.control);
                }
                if metric == SummaryLayout::REMAINING_SERVICE / TICK_WORDS {
                    total = plus(
                        total,
                        self.state.amount(&machine.saved, MachineState::REMAINING),
                    )
                    .record(self.control);
                }
            }
            let count = self.parameters.stock_count as usize;
            let chunk = ceildiv(count as u32, shards as u32) as usize;
            if metric == SummaryLayout::NATURAL_STOCK / TICK_WORDS {
                for index in shard * chunk..min((shard + 1) * chunk, count) {
                    let stock = self.parameters.stock(index);
                    total = plus(
                        total,
                        ticks(self.state.number(&stock.saved, StockState::STOCK), 0),
                    )
                    .record(self.control);
                }
            }
            let count = self.parameters.strike_count as usize;
            let chunk = ceildiv(count as u32, shards as u32) as usize;
            if metric == SummaryLayout::STRIKING / TICK_WORDS {
                for index in shard * chunk..min((shard + 1) * chunk, count) {
                    let strike = self.parameters.strike(index);
                    total = plus(
                        total,
                        ticks(
                            self.state.number(&strike.saved, StrikeState::PARTICIPATION),
                            0,
                        ),
                    )
                    .record(self.control);
                }
            }
            store(self.output, id as u32, total);
        }
    }

    pub fn audit(&mut self, id: usize) {
        if id < self.parameters.lab_count as usize {
            let lab = self.parameters.lab(id);
            let discovered = self.state.number(&lab.saved, LabState::DISCOVERED);
            let mastered = self.state.number(&lab.saved, LabState::MASTERED);
            let permitted = self.state.number(&lab.saved, LabState::PERMITTED);
            let adopted = self.state.number(&lab.saved, LabState::ADOPTED);
            if mastered & !discovered != 0
                || adopted & !(mastered & permitted) != 0
                || permitted != self.records.number(&lab.definition, LabLayout::PERMITTED)
                || self.records.number(&lab.definition, LabLayout::DISCOVERED) & !discovered != 0
                || self.records.number(&lab.definition, LabLayout::MASTERED) & !mastered != 0
                || self.records.number(&lab.definition, LabLayout::ADOPTED) & !adopted != 0
            {
                self.record_mismatch();
            }
        }
        if id < self.parameters.project_count as usize {
            let saved = self.parameters.project(id).saved;
            if self.state.number(&saved, ProjectState::EXPOSURE) > RESEARCH_SCALE
                || self.state.number(&saved, ProjectState::OUTCOME)
                    > comptime!(AttemptOutcome::Discovered as u32)
                || self.state.number(&saved, ProjectState::PROBABILITY) > PROBABILITY_SCALE
                || lt(
                    self.state.amount(&saved, ProjectState::ATTEMPTS),
                    self.state.amount(&saved, ProjectState::DISCOVERIES),
                )
            {
                self.record_mismatch();
            }
        }
        if id < self.parameters.machine_count as usize {
            let machine = self.parameters.machine(id);
            let initial = self
                .records
                .amount(&machine.definition, MachineLayout::INITIAL_SERVICE);
            if self.state.number(&machine.saved, MachineState::CONDITION) > RESEARCH_SCALE
                || self
                    .state
                    .number(&machine.saved, MachineState::MAINTENANCE_RESTORATION)
                    != 0
                || lt(
                    initial,
                    self.state.amount(&machine.saved, MachineState::REMAINING),
                )
            {
                self.record_mismatch();
            }
        }
        if id < self.parameters.stock_count as usize {
            let stock = self.parameters.stock(id);
            if self.state.number(&stock.saved, StockState::STOCK)
                > self
                    .records
                    .number(&stock.definition, StockLayout::CAPACITY)
            {
                self.record_mismatch();
            }
        }
        if id < self.parameters.strike_count as usize {
            let saved = self.parameters.strike(id).saved;
            if self.state.number(&saved, StrikeState::SCORE) > 9000
                || self.state.number(&saved, StrikeState::PROBABILITY) > PROBABILITY_SCALE
                || self.state.number(&saved, StrikeState::PARTICIPATION)
                    > comptime!(Participation::Striking as u32)
            {
                self.record_mismatch();
            }
        }
    }
}

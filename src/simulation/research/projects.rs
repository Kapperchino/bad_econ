use super::{
    access::{ResearchRow, ResearchView, ResearchViewExpand},
    kernels::{AttemptOutcome, Participation},
    layout::*,
    nonlinear::{PROBABILITY_SCALE, hazard, sigmoid},
};
use crate::{
    domain::{Field, Good, research::RESEARCH_SCALE},
    simulation::storage::{
        rows::RowIndex,
        state::{ReadLedger, ReadLedgerExpand},
        words::*,
    },
};
use cubecl::prelude::*;

#[derive(CubeType, Clone, Copy)]
struct LabParticipants {
    owner: u32,
    researcher: u32,
}

#[cube]
impl<'a> ResearchView<'a> {
    pub fn select_method(&mut self, id: usize) {
        if id < self.parameters.lab_count as usize {
            let lab = self.parameters.lab(id);
            let owner = self.records.number(&lab.definition, LabLayout::OWNER);
            let first = self
                .records
                .number(&lab.definition, LabLayout::FIRST_METHOD);
            let end = first
                + self
                    .records
                    .number(&lab.definition, LabLayout::METHOD_COUNT);
            for index in first..end {
                let method = self.parameters.method(index);
                let required = self.records.number(&method, MethodLayout::REQUIREMENTS);
                if self.state.lab_access(&lab.saved) & required == required {
                    self.main.set_number(
                        Field::ActiveMethod,
                        owner,
                        self.records.number(&method, MethodLayout::IDENTITY),
                    );
                    self.main.set_number(
                        Field::LaborMinutes,
                        owner,
                        self.records.number(&method, MethodLayout::LABOR_MINUTES),
                    );
                    self.main.set_number(
                        Field::InputTicks,
                        owner,
                        self.records.number(&method, MethodLayout::INPUT_TICKS),
                    );
                    self.main.set_number(
                        Field::OutputTicks,
                        owner,
                        self.records.number(&method, MethodLayout::OUTPUT_TICKS),
                    );
                    self.main.set_number(
                        Field::FirstInput,
                        owner,
                        self.records.number(&method, MethodLayout::FIRST_INPUT),
                    );
                    self.main.set_number(
                        Field::SecondInput,
                        owner,
                        self.records.number(&method, MethodLayout::SECOND_INPUT),
                    );
                }
            }
        }
    }

    pub fn open_project(&mut self, id: usize) {
        if id < self.parameters.project_count as usize {
            let saved = self.parameters.project(id).saved;
            self.state.set_number(
                &saved,
                ProjectState::OUTCOME,
                comptime!(AttemptOutcome::Idle as u32),
            );
        }
    }

    pub fn production_readiness(&mut self, id: usize) {
        if id < self.parameters.lab_count as usize {
            let lab = self.parameters.lab(id);
            let required = self
                .records
                .number(&lab.definition, LabLayout::PRODUCTION_REQUIREMENTS);
            if (self.state.lab_access(&lab.saved) & required) != required {
                self.cap_plan(
                    self.records.number(&lab.definition, LabLayout::OWNER),
                    zero(),
                );
            }
        }
    }

    pub fn strike(&mut self, id: usize) {
        if id < self.parameters.strike_count as usize {
            let strike = self.parameters.strike(id);
            let positive = 2 * self
                .records
                .number(&strike.definition, StrikeLayout::GRIEVANCE)
                + self
                    .records
                    .number(&strike.definition, StrikeLayout::SOLIDARITY)
                + self.records.number(&strike.definition, StrikeLayout::TRUST)
                + self
                    .records
                    .number(&strike.definition, StrikeLayout::EFFICACY)
                + self.records.number(&strike.definition, StrikeLayout::PEERS);
            let negative = 2 * self.records.number(&strike.definition, StrikeLayout::COST)
                + self.records.number(&strike.definition, StrikeLayout::RISK)
                + 1500;
            let score = i32::cast_from(positive) - i32::cast_from(negative);
            let probability = sigmoid(score, self.table);
            let random = self.samples[id];
            let participation = select(
                random < probability as f32 / PROBABILITY_SCALE as f32,
                comptime!(Participation::Striking as u32),
                comptime!(Participation::Working as u32),
            );
            self.state.set_number(
                &strike.saved,
                StrikeState::SCORE,
                u32::cast_from(score + 4500),
            );
            self.state
                .set_number(&strike.saved, StrikeState::PROBABILITY, probability);
            self.state
                .set_number(&strike.saved, StrikeState::PARTICIPATION, participation);
            if participation == comptime!(Participation::Striking as u32) {
                self.main.set(
                    Field::Time,
                    self.records
                        .number(&strike.definition, StrikeLayout::HOUSEHOLD),
                    zero(),
                );
            }
        }
    }

    pub fn research(&mut self, id: usize) {
        if id < self.parameters.lab_count as usize {
            let lab = self.parameters.lab(id);
            let participants = LabParticipants {
                owner: self.records.number(&lab.definition, LabLayout::OWNER),
                researcher: self.records.number(&lab.definition, LabLayout::RESEARCHER),
            };
            let first = self
                .records
                .number(&lab.definition, LabLayout::FIRST_PROJECT);
            let end = first
                + self
                    .records
                    .number(&lab.definition, LabLayout::PROJECT_COUNT);
            let adoption = self.records.number(&lab.definition, LabLayout::ADOPTION);
            for project in first..end {
                self.attempt(&lab, &participants, project as usize);
                if adoption != 0 {
                    self.state.adopt(&lab.saved);
                }
            }
            if adoption != 0 {
                self.state.adopt(&lab.saved);
            }
        }
    }

    fn attempt(&mut self, lab: &ResearchRow, participants: &LabParticipants, id: usize) {
        let project = self.parameters.project(id);
        let exposure = self.state.number(&project.saved, ProjectState::EXPOSURE);
        let effort = self
            .records
            .number(&project.definition, ProjectLayout::LABOR_MINUTES);
        let probability = self.discovery_probability(&project.definition, exposure, effort);
        self.state
            .set_number(&project.saved, ProjectState::PROBABILITY, probability);

        let mut outcome = self.check_resources(lab, &project.definition, participants, effort);
        if outcome == comptime!(AttemptOutcome::Failed as u32) {
            for good in 0..Good::COUNT {
                let field = good as u32 + 1;
                let used = ticks(self.records.project_input(&project.definition, good), 0);
                let remaining = minus(self.main.value_at(field, participants.owner), used);
                self.main.set_at(field, participants.owner, remaining);
                self.state.consume(&lab.saved, good, used, self.control);
            }
            let time = self.main.value(Field::Time, participants.researcher);
            self.main.set(
                Field::Time,
                participants.researcher,
                minus(time, ticks(effort, 0)),
            );
            self.main.increase(
                self.control,
                Field::ResearchMinutes,
                participants.researcher,
                ticks(effort, 0),
            );
            self.state.increase(
                &project.saved,
                ProjectState::ATTEMPTS,
                ticks(1, 0),
                self.control,
            );
            if self.samples[id] < probability as f32 / PROBABILITY_SCALE as f32 {
                outcome = comptime!(AttemptOutcome::Discovered as u32);
                self.state.increase(
                    &project.saved,
                    ProjectState::DISCOVERIES,
                    ticks(1, 0),
                    self.control,
                );
                let discovered = self.state.number(&lab.saved, LabState::DISCOVERED)
                    | self
                        .records
                        .number(&project.definition, ProjectLayout::OUTCOMES);
                self.state
                    .set_number(&lab.saved, LabState::DISCOVERED, discovered);
            }
        }
        self.state
            .set_number(&project.saved, ProjectState::OUTCOME, outcome);
        let signal = hazard(
            self.records
                .number(&project.definition, ProjectLayout::SIGNAL),
            self.table,
        );
        self.state.set_number(
            &project.saved,
            ProjectState::EXPOSURE,
            exposure + (RESEARCH_SCALE - exposure) * signal / PROBABILITY_SCALE,
        );
    }

    fn discovery_probability(&self, project: &RowIndex, exposure: u32, effort: u32) -> u32 {
        let correlated = exposure
            * self.records.number(project, ProjectLayout::CAPACITY)
            * self.records.number(project, ProjectLayout::RELATEDNESS)
            / (RESEARCH_SCALE * RESEARCH_SCALE);
        let mut power = u32::new(comptime!(RESEARCH_SCALE as i64));
        for _step in 0..self.records.number(project, ProjectLayout::SHAPE) {
            power = power * (RESEARCH_SCALE - correlated) / RESEARCH_SCALE;
        }
        let minimum = self.records.number(project, ProjectLayout::MINIMUM_EFFORT);
        let pioneer = self.records.number(project, ProjectLayout::PIONEER_EFFORT);
        let scale = minimum + (pioneer - minimum) * power / RESEARCH_SCALE;
        hazard(
            min(effort * RESEARCH_SCALE / scale, 16 * RESEARCH_SCALE),
            self.table,
        )
    }

    fn check_resources(
        &self,
        lab: &ResearchRow,
        project: &RowIndex,
        participants: &LabParticipants,
        effort: u32,
    ) -> u32 {
        let required = self.records.number(project, ProjectLayout::PREREQUISITES);
        let outcomes = self.records.number(project, ProjectLayout::OUTCOMES);
        let mut outcome = u32::new(comptime!(AttemptOutcome::Failed as i64));
        if self.state.lab_access(&lab.saved) & required != required
            || self.state.number(&lab.saved, LabState::DISCOVERED) & outcomes == outcomes
            || self.records.number(&lab.definition, LabLayout::CONSENT) == 0
        {
            outcome = comptime!(AttemptOutcome::Ineligible as u32);
        }
        for good in 0..Good::COUNT {
            let available = self.main.value_at(good as u32 + 1, participants.owner);
            if outcome == comptime!(AttemptOutcome::Failed as u32)
                && lt(
                    available,
                    ticks(self.records.project_input(project, good), 0),
                )
            {
                outcome = comptime!(AttemptOutcome::MissingResources as u32);
            }
        }
        if outcome == comptime!(AttemptOutcome::Failed as u32)
            && lt(
                self.main.value(Field::Time, participants.researcher),
                ticks(effort, 0),
            )
        {
            outcome = comptime!(AttemptOutcome::NoTime as u32);
        }
        outcome
    }
}

use crate::domain::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug)]
pub enum SimulationCommand {
    Pause,
    Resume,
    AdvanceDay,
    Reset(Scenario),
    Restore(Checkpoint),
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RuntimeState {
    Paused,
    Running,
    Advancing(Phase),
    Fault {
        last_audited_day: u64,
        message: String,
    },
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PhaseTiming {
    pub phase: Phase,
    pub milliseconds: f64,
    pub events: usize,
    pub frontier_depth: usize,
}

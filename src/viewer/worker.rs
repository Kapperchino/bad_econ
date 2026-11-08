use super::feedback::Cue;
use super::{
    ComparisonRun, Control, Display, DisplayState, Publication, RunTarget, Speed, WorkerState,
};
use crate::{
    domain::{Actor, InspectionRequest, Scenario, enterprise::EnterpriseState},
    runtime::{RuntimeState, SimulationCommand},
    simulation::Simulation,
};
use anyhow::Result;
use crossbeam_channel::{Receiver, RecvTimeoutError};
use std::{
    collections::BTreeSet,
    sync::{Arc, Mutex},
    time::Instant,
};

#[cfg(test)]
mod tests;

struct Publisher<F> {
    data: Arc<Mutex<Display>>,
    wake: F,
}

impl<F: Fn()> Publisher<F> {
    fn update(&self, update: impl FnOnce(&mut Display)) {
        let mut display = self.data.lock().expect("dashboard lock");
        update(&mut display);
        display.revision = display.revision.wrapping_add(1);
        drop(display);
        (self.wake)();
    }

    fn error(&self, simulation: &Simulation, error: anyhow::Error) {
        self.update(|display| {
            display.error = Some(error.to_string());
            display.status = DisplayState::Active(simulation.status.clone());
            display.cues.push_back(Cue::Error);
        });
    }
}

pub(super) fn run(
    mut simulation: Simulation,
    receiver: Receiver<Control>,
    data: Arc<Mutex<Display>>,
    wake: impl Fn(),
) -> Result<()> {
    let publisher = Publisher { data, wake };
    let mut speed = Speed::Normal;
    let mut workflow = WorkerState::Active;
    let mut next_day = Instant::now();
    let mut target = RunTarget::Open;
    publish(
        &mut simulation,
        &publisher,
        Publication::Restart,
        Some(Cue::Ready),
    )?;
    while matches!(workflow, WorkerState::Active) {
        let control = match simulation.status {
            RuntimeState::Running => {
                receiver.recv_timeout(next_day.saturating_duration_since(Instant::now()))
            }
            _ => receiver.recv().map_err(|_| RecvTimeoutError::Disconnected),
        };
        let result = match control {
            Ok(Control::Simulation(command)) => {
                let cue = match &*command {
                    SimulationCommand::Pause => Cue::Paused,
                    SimulationCommand::Resume => Cue::Running,
                    SimulationCommand::AdvanceDay => Cue::Stepped,
                    SimulationCommand::Reset(_) | SimulationCommand::Restore(_) => Cue::Reset,
                };
                let publication = match &*command {
                    SimulationCommand::Reset(_) | SimulationCommand::Restore(_) => {
                        Publication::Restart
                    }
                    _ => Publication::Snapshot,
                };
                simulation.command(*command).and_then(|()| {
                    if let Publication::Restart = publication {
                        target = RunTarget::Open;
                        publisher.update(|display| display.comparison = None);
                    }
                    publish(&mut simulation, &publisher, publication, Some(cue))
                })
            }
            Ok(Control::Inspect(request)) => inspect(&mut simulation, &publisher, request),
            Ok(Control::Speed(value)) => {
                speed = value;
                publisher.update(|display| display.speed = speed);
                next_day = Instant::now();
                Ok(())
            }
            Ok(Control::Compare) => compare(&mut simulation, &publisher).map(|next| {
                target = next;
                next_day = Instant::now();
            }),
            Err(RecvTimeoutError::Disconnected) => {
                workflow = WorkerState::Stopped;
                Ok(())
            }
            Err(RecvTimeoutError::Timeout) => Ok(()),
        };
        if let Err(error) = result {
            publisher.error(&simulation, error);
        }
        if target.reached(simulation.snapshot.day) {
            simulation.command(SimulationCommand::Pause)?;
            target = RunTarget::Open;
            publish(
                &mut simulation,
                &publisher,
                Publication::Snapshot,
                Some(Cue::Compared),
            )?;
        }
        if let (WorkerState::Active, RuntimeState::Running, true) =
            (&workflow, &simulation.status, Instant::now() >= next_day)
        {
            let result = simulation.advance().and_then(|()| {
                let cue = match target.reached(simulation.snapshot.day) {
                    true => {
                        simulation.command(SimulationCommand::Pause)?;
                        target = RunTarget::Open;
                        Some(Cue::Compared)
                    }
                    false => None,
                };
                publish(&mut simulation, &publisher, Publication::Snapshot, cue)
            });
            if let Err(error) = result {
                publisher.error(&simulation, error);
            }
            next_day = Instant::now() + speed.period();
        }
    }
    Ok(())
}

fn inspect(
    simulation: &mut Simulation,
    publisher: &Publisher<impl Fn()>,
    request: InspectionRequest,
) -> Result<()> {
    let inspection = simulation.inspect(request)?;
    publisher.update(|display| {
        display.inspection = Some(inspection)
            .filter(|inspection| inspection.enterprise_state != Some(EnterpriseState::Closed));
    });
    Ok(())
}

fn compare(simulation: &mut Simulation, publisher: &Publisher<impl Fn()>) -> Result<RunTarget> {
    let mut config = simulation.scenario.config().clone();
    config.seed = config
        .seed
        .checked_add(1)
        .ok_or_else(|| anyhow::anyhow!("seed overflow"))?;
    let scenario = Scenario::new(config)?;
    let horizon = simulation.snapshot.day;
    let reference = ComparisonRun {
        seed: simulation.scenario.config().seed,
        horizon,
        history: publisher
            .data
            .lock()
            .expect("dashboard lock")
            .history
            .clone(),
    };
    simulation.command(SimulationCommand::Reset(scenario.into()))?;
    publisher.update(|display| display.comparison = Some(reference));
    let target = match horizon {
        0 => RunTarget::Open,
        _ => {
            simulation.command(SimulationCommand::Resume)?;
            RunTarget::UntilDay(horizon)
        }
    };
    publish(
        simulation,
        publisher,
        Publication::Restart,
        match target {
            RunTarget::Open => Some(Cue::Compared),
            RunTarget::UntilDay(_) => None,
        },
    )?;
    Ok(target)
}

fn publish(
    simulation: &mut Simulation,
    publisher: &Publisher<impl Fn()>,
    publication: Publication,
    cue: Option<Cue>,
) -> Result<()> {
    let observation = {
        let display = publisher.data.lock().expect("dashboard lock");
        match (
            publication,
            display.history.back().map(|snapshot| snapshot.day),
        ) {
            (Publication::Snapshot, Some(day)) if day == simulation.snapshot.day => None,
            _ => Some(()),
        }
    }
    .map(|()| ObservationUpdate::new(simulation))
    .transpose()?;
    let request = match publication {
        Publication::Restart => None,
        Publication::Snapshot => {
            let display = publisher.data.lock().expect("dashboard lock");
            let identities = observation
                .as_ref()
                .map_or(display.identities.as_slice(), |update| {
                    update.identities.as_slice()
                });
            display
                .inspection
                .as_ref()
                .filter(|inspection| identities.contains(&inspection.actor.identity))
                .filter(|inspection| inspection.day != simulation.snapshot.day)
                .map(|inspection| {
                    InspectionRequest::from_registry(inspection.actor.identity, identities)
                })
                .transpose()?
        }
    };
    let refreshed = request
        .map(|request| simulation.inspect(request))
        .transpose()?;
    publisher.update(|display| {
        if let Some(inspection) = refreshed {
            display.inspection = Some(inspection);
        }
        if let Publication::Restart = publication {
            display.map_size = simulation.scenario.config().map_size;
            display.history.clear();
            display.inspection = None;
            display.map_revision = display.map_revision.wrapping_add(1);
        }
        if let Some(observation) = observation {
            display.inspection = display
                .inspection
                .take()
                .filter(|inspection| observation.identities.contains(&inspection.actor.identity));
            if display.firms != observation.firms {
                display.firms = observation.firms;
                display.map_revision = display.map_revision.wrapping_add(1);
            }
            display.identities = observation.identities;
            display.observation_revision = display.observation_revision.wrapping_add(1);
        }
        let snapshot = crate::domain::ObservationSnapshot {
            firms: display.firms.len() as u64,
            ..simulation.snapshot.clone()
        };
        match display.history.back().map(|s| s.day) {
            Some(day) if day == simulation.snapshot.day => (),
            _ => {
                display.history.push_back(snapshot.clone());
            }
        }
        display.snapshot = snapshot;
        display.status = DisplayState::Active(simulation.status.clone());
        display.seed = simulation.scenario.config().seed;
        display.error = None;
        display.cues.extend(cue);
    });
    Ok(())
}

struct ObservationUpdate {
    firms: Vec<Actor>,
    identities: Vec<u64>,
}

impl ObservationUpdate {
    fn new(simulation: &mut Simulation) -> Result<Self> {
        let firms = simulation.open_firms()?;
        let firm_identities = firms
            .iter()
            .map(|firm| firm.identity)
            .collect::<BTreeSet<_>>();
        let identities = simulation
            .actors()
            .iter()
            .filter(|actor| match actor.kind {
                0 => true,
                _ => firm_identities.contains(&actor.identity),
            })
            .map(|actor| actor.identity)
            .collect();
        Ok(Self { firms, identities })
    }
}

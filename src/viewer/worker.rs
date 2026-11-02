use super::feedback::Cue;
use super::{
    ComparisonRun, Control, Display, DisplayState, Publication, RunTarget, Speed, WorkerState,
};
use crate::{
    domain::{InspectionRequest, Scenario},
    runtime::{RuntimeState, SimulationCommand},
    simulation::Simulation,
};
use anyhow::Result;
use std::{
    sync::{Arc, Mutex, mpsc},
    time::{Duration, Instant},
};

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
    receiver: mpsc::Receiver<Control>,
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
        let timeout = match simulation.status {
            RuntimeState::Running => next_day
                .saturating_duration_since(Instant::now())
                .min(Duration::from_millis(50)),
            _ => Duration::from_millis(50),
        };
        let result = match receiver.recv_timeout(timeout) {
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
            Ok(Control::Inspect(request)) => simulation.inspect(request).map(|inspection| {
                publisher.update(|display| display.inspection = Some(inspection))
            }),
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
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                workflow = WorkerState::Stopped;
                Ok(())
            }
            Err(mpsc::RecvTimeoutError::Timeout) => Ok(()),
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
    let request = match publication {
        Publication::Restart => None,
        Publication::Snapshot => {
            let display = publisher.data.lock().expect("dashboard lock");
            display
                .inspection
                .as_ref()
                .filter(|inspection| inspection.day != simulation.snapshot.day)
                .map(|inspection| {
                    InspectionRequest::from_registry(inspection.actor.identity, &display.identities)
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
        display.firms = simulation
            .actors()
            .iter()
            .filter(|actor| actor.kind > 0)
            .collect();
        match publication {
            Publication::Restart => {
                display.map_size = simulation.scenario.config().map_size;
                display.history.clear();
                display.inspection = None;
                display.identities = simulation
                    .actors()
                    .iter()
                    .map(|actor| actor.identity)
                    .collect();
            }
            Publication::Snapshot => {
                if display.identities.len() != simulation.actors().len() {
                    display.identities = simulation
                        .actors()
                        .iter()
                        .map(|actor| actor.identity)
                        .collect();
                }
            }
        }
        match display.history.back().map(|s| s.day) {
            Some(day) if day == simulation.snapshot.day => (),
            _ => {
                display.history.push_back(simulation.snapshot.clone());
                if display.history.len() > 160 {
                    display.history.pop_front();
                }
            }
        }
        display.snapshot = simulation.snapshot.clone();
        display.status = DisplayState::Active(simulation.status.clone());
        display.seed = simulation.scenario.config().seed;
        display.error = None;
        display.cues.extend(cue);
    });
    Ok(())
}

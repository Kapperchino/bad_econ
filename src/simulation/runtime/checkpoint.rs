use crate::domain::{Checkpoint, State};
use crate::simulation::{Simulation, gpu::tensor, profiling_day, storage::layout::TICK_BYTES};
use anyhow::Result;
use std::time::Instant;

#[derive(Clone, Copy)]
pub(in crate::simulation) enum CopyDirection {
    Save,
    Restore,
}

impl Simulation {
    fn read_state(&mut self, direction: CopyDirection) -> Result<Vec<u64>> {
        let mut values = match direction {
            CopyDirection::Save => {
                let mut ticks = Vec::new();
                self.device.read_ticks_into(&self.state, &mut ticks)?;
                ticks
            }
            CopyDirection::Restore => self.audited.read_ticks(&self.device, &self.state)?,
        };
        self.readback_bytes += (values.len() * TICK_BYTES) as u64;
        let count = self.metadata.actors.len();
        let stride = self.config.state_stride as usize;
        if stride != count {
            for field in 1..self.scenario.fields() {
                values.copy_within(field * stride..field * stride + count, field * count);
            }
            values.truncate(count * self.scenario.fields());
        }
        Ok(values)
    }
    pub(in crate::simulation) fn state_snapshot(
        &mut self,
        direction: CopyDirection,
    ) -> Result<State> {
        let values = self.read_state(direction)?;
        let research = self
            .research
            .as_ref()
            .map(|research| match direction {
                CopyDirection::Save => research.snapshot(&self.device),
                CopyDirection::Restore => research.audited_snapshot(&self.device),
            })
            .transpose()?;
        self.account_research_readbacks();
        Ok(State {
            people: self
                .demography
                .download(&self.device, &self.metadata, direction)?,
            day: match direction {
                CopyDirection::Save => self.day,
                CopyDirection::Restore => self.audited_day,
            },
            values,
            research,
            ..self.metadata.clone()
        })
    }
    pub fn download(&mut self) -> Result<State> {
        self.state_snapshot(CopyDirection::Save)
    }
    pub(in crate::simulation) fn copy_audited(&mut self, direction: CopyDirection) -> Result<()> {
        let started = Instant::now();
        let profile =
            self.day == profiling_day() && std::env::var_os("BAD_ECON_PROFILE_STAGES").is_some();
        if profile {
            eprintln!(
                "COPYSHAPE words={} modulo={}",
                self.state.meta.num_elements(),
                self.state.meta.num_elements() % 4
            );
            self.device.sync()?;
            eprintln!("COPYWAIT {:.3}", started.elapsed().as_secs_f64() * 1000.0);
        }
        match direction {
            CopyDirection::Save => self.snapshot_copy.save_state(
                &self.device,
                &self.state,
                &self.audited,
                self.config.state_stride as usize,
                self.scenario.ring(),
            ),
            CopyDirection::Restore => {
                self.audited.restore(&self.device, &self.state);
                if let Some(mask) = self.snapshot_copy.ring_mask() {
                    tensor::reset_ring_mask(&self.device, mask, tensor::RingMaskState::Unknown);
                }
            }
        }
        if profile {
            self.device.sync()?;
            eprintln!("COPYSTATE {:.3}", started.elapsed().as_secs_f64() * 1000.0);
        }
        let research_started = Instant::now();
        if let Some(research) = &self.research {
            match direction {
                CopyDirection::Save => research.save_audited(&self.device)?,
                CopyDirection::Restore => research.restore_audited(&self.device)?,
            }
        }
        self.account_research_readbacks();
        if profile {
            self.device.sync()?;
            eprintln!(
                "COPYRESEARCH {:.3}",
                research_started.elapsed().as_secs_f64() * 1000.0
            );
        }
        let demography_started = Instant::now();
        self.demography.copy(&self.device, direction);
        self.device.sync()?;
        if profile {
            eprintln!(
                "COPYDEMOGRAPHY {:.3}",
                demography_started.elapsed().as_secs_f64() * 1000.0
            );
            eprintln!(
                "AUDIT {} Copy {:.3}",
                self.day,
                started.elapsed().as_secs_f64() * 1000.0
            );
        }
        Ok(())
    }

    pub fn checkpoint(&mut self) -> Result<Checkpoint> {
        let state = self.state_snapshot(CopyDirection::Restore)?;
        Checkpoint::new(&self.scenario, &state)
    }
}

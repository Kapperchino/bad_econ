use super::{
    Device,
    configuration::Configuration,
    economy::{self, EconomyStage, markets},
    flags::CONTROL_WORDS,
    layout::*,
    parameters::{EconomyParameters, SchedulerParameters},
    scheduler::{self, SchedulerStage},
    system::{AsBufferArg, System},
};
use burn_wgpu::CubeTensor;

#[derive(Clone, Copy)]
pub(super) struct DeviceEconomy<'a> {
    pub device: &'a Device,
    pub state: &'a CubeTensor,
    pub actors: &'a CubeTensor,
    pub config: &'a Configuration,
}

pub(super) struct PhaseBuffers {
    pub proposals: CubeTensor,
    pub events: CubeTensor,
    pub scratch: CubeTensor,
    pub feedback: CubeTensor,
    #[cfg(test)]
    pub links: CubeTensor,
}

struct BufferSizes {
    proposals: usize,
    events: usize,
    scratch: usize,
    feedback: usize,
    #[cfg(test)]
    links: usize,
}

impl BufferSizes {
    fn for_actors(actors: usize, batch_limit: u32) -> Self {
        let batch = actors.min(batch_limit as usize);
        let mut sizes = Self {
            proposals: actors * ProposalLayout::WORDS,
            events: actors.div_ceil(economy::audit::TILE_ACTORS) * SummaryLayout::WORDS,
            scratch: markets::scratch_words(actors),
            feedback: CONTROL_WORDS + actors * TICK_WORDS,
            #[cfg(test)]
            links: batch * LinkLayout::WORDS_PER_EVENT,
        };
        if cfg!(test) {
            sizes.events = sizes.events.max(batch * EVENT_WORDS).max(
                actors.div_ceil(SummaryLayout::ACTORS_PER_GROUP as usize) * SummaryLayout::WORDS,
            );
            sizes.scratch = sizes.scratch.max(batch * ScratchLayout::WORDS_PER_EVENT);
        }
        sizes
    }
}

impl PhaseBuffers {
    pub(super) fn new(device: &Device, actors: usize, batch_limit: u32) -> Self {
        let sizes = BufferSizes::for_actors(actors, batch_limit);
        Self {
            events: device.empty(sizes.events),
            scratch: device.empty(sizes.scratch),
            #[cfg(test)]
            links: device.empty(sizes.links),
            proposals: device.empty(sizes.proposals),
            feedback: device.tensor(&vec![0; sizes.feedback]),
        }
    }

    pub(super) fn resized(&self, device: &Device, actors: usize, batch_limit: u32) -> Self {
        let sizes = BufferSizes::for_actors(actors, batch_limit);
        Self {
            proposals: device.resize_scratch(&self.proposals, sizes.proposals),
            events: device.resize_scratch(&self.events, sizes.events),
            scratch: device.resize_scratch(&self.scratch, sizes.scratch),
            #[cfg(test)]
            links: device.empty(sizes.links),
            feedback: device.resize_scratch(&self.feedback, sizes.feedback),
        }
    }
}

pub(super) struct EconomyBuffers<'a> {
    pub economy: DeviceEconomy<'a>,
    pub spatial: &'a CubeTensor,
    pub proposals: &'a CubeTensor,
    pub events: &'a CubeTensor,
    pub control: &'a CubeTensor,
}

#[derive(Clone, Copy)]
pub(super) struct SchedulerBuffers<'a> {
    pub state: &'a CubeTensor,
    pub events: &'a CubeTensor,
    pub scratch: &'a CubeTensor,
    pub control: &'a CubeTensor,
    pub links: &'a CubeTensor,
}

impl EconomyBuffers<'_> {
    pub(super) fn launch_input(
        &self,
        stage: EconomyStage,
        parameters: &EconomyParameters,
    ) -> economy::EconomyInputLaunch {
        economy::EconomyInputLaunch {
            state: self.economy.state.buffer_arg(),
            actors: self.economy.actors.buffer_arg(),
            config: self.economy.config.launch(),
            spatial: self.spatial.buffer_arg(),
            proposals: self.proposals.buffer_arg(),
            bundles: self.events.buffer_arg(),
            control: self.control.buffer_arg(),
            parameters: parameters.launch(),
            stage,
        }
    }

    pub(super) fn run(&self, stage: EconomyStage, count: usize, parameters: &EconomyParameters) {
        match (stage, parameters.phase) {
            (EconomyStage::Local, crate::domain::Phase::Opening) => economy::opening::launch(
                self.economy.device,
                self.economy.state,
                self.economy.config,
                self.control,
                parameters.day,
            ),
            _ => economy::EconomySystem::launch_elements(
                self.economy.device,
                count,
                self.launch_input(stage, parameters),
            ),
        }
    }
}

impl SchedulerBuffers<'_> {
    pub(super) fn schedule(
        &self,
        device: &Device,
        stage: SchedulerStage,
        count: usize,
        parameters: SchedulerParameters,
    ) {
        scheduler::EventScheduling::launch_elements(
            device,
            count,
            scheduler::ScheduleInputLaunch {
                balances: self.state.buffer_arg(),
                events: self.events.buffer_arg(),
                scratch: self.scratch.buffer_arg(),
                control: self.control.buffer_arg(),
                links: self.links.buffer_arg(),
                params: parameters.launch(),
                stage,
            },
        );
    }
}

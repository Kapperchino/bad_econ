use super::*;
use crate::simulation::{Device, configuration::Configuration};
use burn_wgpu::CubeTensor;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum OpeningStage {
    Actors,
    Goods,
}

pub(in crate::simulation) fn launch(
    device: &Device,
    state: &CubeTensor,
    config: &Configuration,
    control: &CubeTensor,
    day: u32,
) {
    for stage in [OpeningStage::Actors, OpeningStage::Goods] {
        let count = match stage {
            OpeningStage::Actors => config.actors as usize,
            OpeningStage::Goods => config.actors as usize * Good::COUNT,
        };
        opening::launch(
            &device.client,
            device.count(count),
            CubeDim::new_1d(device.workgroup),
            state.clone().into_buffer_arg(),
            config.launch(),
            control.clone().into_buffer_arg(),
            day,
            stage,
        );
    }
}

#[cube]
fn coordinate(config: &EconomyConfig, field: u32, actor: u32) -> usize {
    (field * config.actors + actor) as usize
}

#[cube(launch)]
fn opening(
    state: &mut [Ticks],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
    day: u32,
    #[comptime] stage: OpeningStage,
) {
    let index = ABSOLUTE_POS as u32;
    match stage {
        OpeningStage::Actors => {
            if index < config.actors {
                let transit = coordinate(config, comptime!(Field::Transit as u32), index);
                let mut total = state[transit];
                for good in 0..Good::COUNT as u32 {
                    let slot = BASE_FIELDS as u32 + good * config.ring + day % config.ring;
                    total = minus(total, state[coordinate(config, slot, index)]);
                }
                state[transit] = total;
                for field in [
                    Field::Employed as u32,
                    Field::Employer as u32,
                    Field::Sales as u32,
                    Field::Unfilled as u32,
                    Field::Shortage as u32,
                    Field::Purchased as u32,
                    Field::Hours as u32,
                    Field::HomeMinutes as u32,
                    Field::ResearchMinutes as u32,
                ] {
                    state[coordinate(config, field, index)] = zero();
                }
            }
        }
        OpeningStage::Goods => {
            let actor = index % config.actors;
            let good = index / config.actors;
            if good < Good::COUNT as u32 {
                let slot = coordinate(
                    config,
                    BASE_FIELDS as u32 + good * config.ring + day % config.ring,
                    actor,
                );
                let inventory = coordinate(config, good + 1, actor);
                let transit = coordinate(config, TRANSIT_FIELDS as u32 + good, actor);
                let delivery = state[slot];
                let received = plus(state[inventory], delivery);
                received.faults.record(control);
                state[inventory] = received.value;
                let available = state[transit];
                if lt(available, delivery) {
                    KernelFaults::new(comptime!(FaultFlags::TRANSIT_UNDERFLOW)).record(control);
                }
                state[transit] = minus(available, delivery);
                state[slot] = zero();
            }
        }
    }
}

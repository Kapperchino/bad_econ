use super::{Device, layout::ConfigLayout, words::*};
use crate::domain::{Field, State, enterprise::EnterpriseState};
use burn_wgpu::CubeTensor;
use cubecl::{CubeCount, CubeDim, prelude::*};
use std::collections::BTreeMap;

pub(super) struct EnterpriseManagement {
    groups: u32,
    firms: CubeTensor,
}

impl EnterpriseManagement {
    pub fn new(device: &Device, state: &State) -> Self {
        let mut owners = BTreeMap::<u32, Vec<u32>>::new();
        for (index, actor) in state.actors.iter().enumerate() {
            if actor.kind > 0 {
                owners.entry(actor.owner).or_default().push(index as u32);
            }
        }
        let groups = owners.len() as u32;
        let mut words = vec![0; owners.len() * 3];
        for (index, (owner, firms)) in owners.into_iter().enumerate() {
            words[index * 3] = owner;
            words[index * 3 + 1] = words.len() as u32;
            words.extend(firms);
            words[index * 3 + 2] = words.len() as u32;
        }
        Self {
            groups,
            firms: device.tensor(&words),
        }
    }

    pub fn assign(&self, device: &Device, state: &CubeTensor, config: &CubeTensor, day: u32) {
        if self.groups > 0 {
            let argument = |tensor: &CubeTensor| unsafe {
                BufferArg::from_raw_parts(tensor.handle.clone(), tensor.meta.shape()[0])
            };
            let groups = self.groups.div_ceil(device.workgroup);
            let width = groups.min(u16::MAX.into());
            assign::launch(
                &device.client,
                CubeCount::Static(width, groups.div_ceil(width), 1),
                CubeDim::new_1d(device.workgroup),
                argument(state),
                argument(config),
                argument(&self.firms),
                self.groups,
                day,
            );
        }
    }
}

#[cube(launch)]
fn assign(state: &mut [u32], config: &[u32], firms: &[u32], groups: u32, day: u32) {
    let group = ABSOLUTE_POS as u32;
    if group < groups && config[ConfigLayout::HOME] > 0 {
        let count = config[ConfigLayout::ACTORS];
        let owner = firms[(group * 3) as usize];
        let start = firms[(group * 3 + 1) as usize];
        let end = firms[(group * 3 + 2) as usize];
        let mut active = 0;
        let mut construction = 0;
        for offset in 0..end - start {
            let firm = firms[(start + (offset + day) % (end - start)) as usize];
            let status = load(
                state,
                comptime!(Field::EnterpriseStatus as u32) * count + firm,
            )
            .extract(0usize);
            if status != comptime!(EnterpriseState::Closed as u32) && active == 0 {
                active = firm + 1;
            }
            let cash = load(state, comptime!(Field::Cash as u32) * count + firm);
            let reserve = load(
                state,
                comptime!(Field::OperatingReserve as u32) * count + firm,
            );
            let available = select(
                status == comptime!(EnterpriseState::Operating as u32),
                positive(cash, reserve),
                cash,
            );
            let wage = load(state, comptime!(Field::Wage as u32) * count + firm);
            let remaining = load(
                state,
                comptime!(Field::ConstructionRemaining as u32) * count + firm,
            );
            let building = status == comptime!(EnterpriseState::Forming as u32)
                || status == comptime!(EnterpriseState::Operating as u32);
            if building && !equal(remaining, zero()) && !lt(available, wage) && construction == 0 {
                construction = firm + 1;
            }
        }
        let living = load(
            state,
            comptime!(Field::LivingMembers as u32) * count + owner,
        );
        let selected = select(construction > 0, construction, active);
        store(
            state,
            comptime!(Field::InvestmentFirm as u32) * count + owner,
            ticks(select(equal(living, zero()), 0, selected), 0),
        );
    }
}

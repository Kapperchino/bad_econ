use crate::domain::{Field, State, enterprise::EnterpriseState};
use crate::simulation::economy::configuration::{Configuration, EconomyConfig, Feature};
use crate::simulation::{
    Device,
    pass::GpuPass,
    storage::{
        state::{Ledger, ReadLedger, ReadLedgerExpand},
        words::*,
    },
    system::{AsBufferArg, System, gpu_passes},
};
use burn_wgpu::CubeTensor;
use cubecl::prelude::*;
use std::collections::BTreeMap;

pub(in crate::simulation) struct EnterpriseManagement {
    groups: u32,
    firms: CubeTensor,
}

#[derive(CubeType, CubeLaunch)]
pub(in crate::simulation) struct AssignmentInput {
    state: Box<[u32]>,
    config: EconomyConfig,
    firms: Box<[u32]>,
    groups: u32,
    day: u32,
}

pub(in crate::simulation) struct AssignOwners;

impl System for EnterpriseManagement {
    type Input = AssignmentInput;
    type Passes = gpu_passes![AssignOwners];
}

#[cube]
impl GpuPass<AssignmentInput> for AssignOwners {
    type Output = ();

    fn execute(mut input: AssignmentInput) {
        assign(
            &mut Ledger::new(&mut input.state, &input.config),
            &input.firms,
            input.groups,
            input.day,
        );
    }
}

impl EnterpriseManagement {
    pub fn new(device: &Device, state: &State) -> Self {
        let mut owners = BTreeMap::<u32, Vec<u32>>::new();
        for index in state.actors.firm_indices() {
            let actor = state.actors.at(index);
            owners.entry(actor.owner).or_default().push(index as u32);
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

    pub fn assign(&self, device: &Device, state: &CubeTensor, config: &Configuration, day: u32) {
        if self.groups > 0 {
            Self::launch_elements(
                device,
                self.groups as usize,
                AssignmentInputLaunch {
                    state: state.buffer_arg(),
                    config: config.launch(),
                    firms: self.firms.buffer_arg(),
                    groups: self.groups,
                    day,
                },
            );
        }
    }
}

#[cube]
fn assign(ledger: &mut Ledger<'_>, firms: &[u32], groups: u32, day: u32) {
    let config = ledger.config;
    let group = ABSOLUTE_POS as u32;
    if group < groups && comptime!(config.home_mode == Feature::Enabled) {
        let owner = firms[(group * 3) as usize];
        let start = firms[(group * 3 + 1) as usize];
        let end = firms[(group * 3 + 2) as usize];
        let mut active = 0;
        let mut construction = 0;
        for offset in 0..end - start {
            let firm = firms[(start + (offset + day) % (end - start)) as usize];
            let status = ledger.number(Field::EnterpriseStatus, firm);
            if status != comptime!(EnterpriseState::Closed as u32) && active == 0 {
                active = firm + 1;
            }
            let cash = ledger.value(Field::Cash, firm);
            let reserve = ledger.value(Field::OperatingReserve, firm);
            let available = select(
                status == comptime!(EnterpriseState::Operating as u32),
                positive(cash, reserve),
                cash,
            );
            let wage = ledger.value(Field::Wage, firm);
            let remaining = ledger.value(Field::ConstructionRemaining, firm);
            let building = status == comptime!(EnterpriseState::Forming as u32)
                || status == comptime!(EnterpriseState::Operating as u32);
            if building && !equal(remaining, zero()) && !lt(available, wage) && construction == 0 {
                construction = firm + 1;
            }
        }
        let living = ledger.value(Field::LivingMembers, owner);
        let selected = select(construction > 0, construction, active);
        ledger.set(
            Field::InvestmentFirm,
            owner,
            ticks(select(equal(living, zero()), 0, selected), 0),
        );
    }
}

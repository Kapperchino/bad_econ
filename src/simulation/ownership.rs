use super::{Device, flags::FaultFlags, words::*};
use crate::domain::{Field, State};
use crate::simulation::configuration::{Configuration, EconomyConfig};
use anyhow::{Context, Result};
use burn_wgpu::CubeTensor;
use cubecl::{CubeCount, CubeDim, prelude::*};

pub(super) struct OwnershipRuntime {
    shares: CubeTensor,
    control: CubeTensor,
}
impl OwnershipRuntime {
    pub fn new(device: &Device, state: &State) -> Result<Self> {
        let actors = state
            .actors
            .iter()
            .enumerate()
            .map(|(index, actor)| (actor.identity, index as u32))
            .collect::<std::collections::HashMap<_, _>>();
        let mut words = vec![state.ownership.len() as u32];
        for ownership in &state.ownership {
            words.push(
                *actors
                    .get(&ownership.firm)
                    .context("missing ownership firm")?,
            );
            words.push(ownership.shares.len() as u32);
            for share in &ownership.shares {
                words.push(
                    *actors
                        .get(&share.household)
                        .context("missing shareholder")?,
                );
                words.push(share.units);
            }
        }
        Ok(Self {
            shares: device.tensor(&words),
            control: device.tensor(&[0; 4]),
        })
    }
    pub fn distribute(
        &self,
        device: &Device,
        state: &CubeTensor,
        config: &Configuration,
    ) -> Result<()> {
        let argument = |tensor: &CubeTensor| tensor.clone().into_buffer_arg();
        device.write(&self.control, &[0; 4]);
        distribute::launch(
            &device.client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(device.workgroup),
            argument(state),
            config.launch(),
            argument(&self.shares),
            argument(&self.control),
        );
        FaultFlags::read(&device.read(&self.control)?)?.into_result()
    }
}

#[cube(launch)]
fn distribute(state: &mut [u32], config: &EconomyConfig, shares: &[u32], control: &[Atomic<u32>]) {
    if ABSOLUTE_POS == 0 {
        let actors = config.actors;
        let mut cursor = 1usize;
        for _ in 0..shares[0usize] {
            let firm = shares[cursor];
            let count = shares[cursor + 1];
            cursor += 2;
            let cash = load(state, comptime!(Field::Cash as u32) * actors + firm);
            let labor =
                load(state, comptime!(Field::LaborMinutes as u32) * actors + firm).extract(0usize);
            let wage = load(state, comptime!(Field::Wage as u32) * actors + firm).extract(0usize);
            let plan = load(state, comptime!(Field::Plan as u32) * actors + firm).extract(0usize);
            let payroll = times32(labor * wage, plan);
            let reserve = maximum(
                maximum(
                    payroll,
                    load(
                        state,
                        comptime!(Field::OperatingReserve as u32) * actors + firm,
                    ),
                ),
                ticks(config.reserve_low, config.reserve_high),
            );
            let amount = ratio(positive(cash, reserve), config.distribution_rate, 1000);
            amount.faults.record(control);
            let mut distributed = zero();
            for _ in 0..count {
                let owner = shares[cursor];
                let units = shares[cursor + 1];
                cursor += 2;
                let payment = ratio(amount.value, units, 1_000_000);
                payment.faults.record(control);
                let prior = load(state, comptime!(Field::Cash as u32) * actors + owner);
                let next = plus(prior, payment.value);
                next.faults.record(control);
                store(
                    state,
                    comptime!(Field::Cash as u32) * actors + owner,
                    next.value,
                );
                let accrued = plus(distributed, payment.value);
                accrued.faults.record(control);
                distributed = accrued.value;
            }
            store(
                state,
                comptime!(Field::Cash as u32) * actors + firm,
                minus(cash, distributed),
            );
        }
    }
}

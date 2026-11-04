use crate::domain::{Field, State};
use crate::simulation::configuration::{Configuration, EconomyConfig};
use crate::simulation::{
    Device,
    flags::FaultFlags,
    pass::GpuPass,
    system::{System, gpu_passes},
    words::*,
};
use anyhow::{Context, Result};
use burn_wgpu::CubeTensor;
use cubecl::{CubeDim, prelude::*};
use std::collections::BTreeMap;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DistributionShare {
    firm: u32,
    units: u32,
}

pub(in crate::simulation) struct OwnershipRuntime {
    shares: CubeTensor,
    offsets: CubeTensor,
    entries: CubeTensor,
    payments: CubeTensor,
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
        let mut shares = Vec::new();
        let mut accounts = BTreeMap::<u32, Vec<u32>>::new();
        for ownership in &state.ownership {
            let firm = *actors
                .get(&ownership.firm)
                .context("missing ownership firm")?;
            for share in &ownership.shares {
                let owner = *actors
                    .get(&share.household)
                    .context("missing shareholder")?;
                let index = u32::try_from(shares.len())?;
                shares.push(DistributionShare {
                    firm,
                    units: share.units,
                });
                accounts.entry(firm).or_default().push(index);
                accounts.entry(owner).or_default().push(index);
            }
        }
        let mut entries = Vec::with_capacity(shares.len() * 2);
        let mut offsets = Vec::with_capacity(state.actors.len() + 1);
        offsets.push(0);
        for actor in 0..state.actors.len() as u32 {
            entries.extend(accounts.remove(&actor).unwrap_or_default());
            offsets.push(u32::try_from(entries.len())?);
        }
        Ok(Self {
            payments: device.empty(shares.len().max(1) * 2),
            shares: device.tensor(bytemuck::cast_slice(&shares)),
            offsets: device.tensor(&offsets),
            entries: device.tensor(&entries),
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
        OwnershipPayments::launch(
            device,
            device.count(self.shares.meta.num_elements() / 2),
            CubeDim::new_1d(device.workgroup),
            PaymentInputLaunch {
                state: argument(state),
                config: config.launch(),
                shares: argument(&self.shares),
                payments: argument(&self.payments),
                control: argument(&self.control),
            },
        );
        OwnershipDistribution::launch(
            device,
            device.count(config.actors as usize),
            CubeDim::new_1d(device.workgroup),
            DistributionInputLaunch {
                state: argument(state),
                config: config.launch(),
                shares: argument(&self.shares),
                offsets: argument(&self.offsets),
                entries: argument(&self.entries),
                payments: argument(&self.payments),
                control: argument(&self.control),
            },
        );
        FaultFlags::read(&device.read(&self.control)?)?.into_result()
    }
}

#[derive(CubeType, CubeLaunch)]
struct PaymentInput {
    state: Box<[Ticks]>,
    config: EconomyConfig,
    shares: Box<[Vector<u32, Const<2>>]>,
    payments: Box<[Ticks]>,
    control: Box<[Atomic<u32>]>,
}

#[derive(CubeType, CubeLaunch)]
struct DistributionInput {
    state: Box<[Ticks]>,
    config: EconomyConfig,
    shares: Box<[Vector<u32, Const<2>>]>,
    offsets: Box<[u32]>,
    entries: Box<[u32]>,
    payments: Box<[Ticks]>,
    control: Box<[Atomic<u32>]>,
}

struct OwnershipPayments;
struct OwnershipDistribution;
struct CalculatePayments;
struct DistributePayments;

impl System for OwnershipPayments {
    type Input = PaymentInput;
    type Passes = gpu_passes![CalculatePayments];
}

impl System for OwnershipDistribution {
    type Input = DistributionInput;
    type Passes = gpu_passes![DistributePayments];
}

#[cube]
impl GpuPass<PaymentInput> for CalculatePayments {
    type Output = ();

    fn execute(mut input: PaymentInput) {
        payments(
            &input.state,
            &input.config,
            &input.shares,
            &mut input.payments,
            &input.control,
        );
    }
}

#[cube]
impl GpuPass<DistributionInput> for DistributePayments {
    type Output = ();

    fn execute(mut input: DistributionInput) {
        distribute(
            &mut input.state,
            &input.config,
            &input.shares,
            &input.offsets,
            &input.entries,
            &input.payments,
            &input.control,
        );
    }
}

#[cube]
fn payments(
    state: &[Ticks],
    config: &EconomyConfig,
    shares: &[Vector<u32, Const<2>>],
    payments: &mut [Ticks],
    control: &[Atomic<u32>],
) {
    let index = ABSOLUTE_POS;
    if index < shares.len() {
        let share = shares[index];
        let firm = share.extract(0usize);
        let units = share.extract(1usize);
        let actors = config.actors;
        let cash = state[(comptime!(Field::Cash as u32) * actors + firm) as usize];
        let labor =
            state[(comptime!(Field::LaborMinutes as u32) * actors + firm) as usize].extract(0usize);
        let wage = state[(comptime!(Field::Wage as u32) * actors + firm) as usize].extract(0usize);
        let plan = state[(comptime!(Field::Plan as u32) * actors + firm) as usize].extract(0usize);
        let payroll = times32(labor * wage, plan);
        let reserve = maximum(
            maximum(
                payroll,
                state[(comptime!(Field::OperatingReserve as u32) * actors + firm) as usize],
            ),
            ticks(config.reserve_low, config.reserve_high),
        );
        let amount = ratio(positive(cash, reserve), config.distribution_rate, 1000);
        amount.faults.record(control);
        let payment = ratio(amount.value, units, 1_000_000);
        payment.faults.record(control);
        payments[index] = payment.value;
    }
}

#[cube]
fn distribute(
    state: &mut [Ticks],
    config: &EconomyConfig,
    shares: &[Vector<u32, Const<2>>],
    offsets: &[u32],
    entries: &[u32],
    payments: &[Ticks],
    control: &[Atomic<u32>],
) {
    let actor = ABSOLUTE_POS as u32;
    if actor < config.actors && offsets[actor as usize] < offsets[actor as usize + 1] {
        let start = offsets[actor as usize];
        let end = offsets[actor as usize + 1];
        let mut total = zero();
        for index in start..end {
            let accrued = plus(total, payments[entries[index as usize] as usize]);
            accrued.faults.record(control);
            total = accrued.value;
        }
        let coordinate = comptime!(Field::Cash as u32) * config.actors + actor;
        state[coordinate as usize] = account_balance(
            state[coordinate as usize],
            total,
            shares[entries[start as usize] as usize].extract(0usize),
            actor,
            control,
        );
    }
}

#[cube]
fn account_balance(
    cash: Ticks,
    total: Ticks,
    firm: u32,
    actor: u32,
    control: &[Atomic<u32>],
) -> Ticks {
    let mut balance = minus(cash, total);
    if firm != actor {
        let credited = plus(cash, total);
        credited.faults.record(control);
        balance = credited.value;
    }
    balance
}

#[cfg(test)]
#[cube(launch)]
fn original_distribute(
    state: &mut [u32],
    config: &EconomyConfig,
    shares: &[u32],
    control: &[Atomic<u32>],
) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{
        Experiment, Scenario,
        enterprise::{Ownership, OwnershipShare},
    };
    use crate::simulation::encode;

    #[test]
    fn parallel_dividends_match_serial_rounding_shared_owners_and_overflow_on_gpu() -> Result<()> {
        let mut scenario_config = Scenario::baseline(193, 42, Experiment::Baseline)?
            .config()
            .clone();
        scenario_config.firms = 65;
        let scenario = Scenario::new(scenario_config)?;
        let mut initial = scenario.initial()?;
        initial.ownership = initial
            .actors
            .iter()
            .enumerate()
            .filter(|(_, actor)| actor.kind > 0)
            .map(|(index, actor)| {
                Ownership::new(Ownership {
                    firm: actor.identity,
                    shares: vec![
                        OwnershipShare {
                            household: initial.actors.at(0).identity,
                            units: 333_333,
                        },
                        OwnershipShare {
                            household: initial.actors.at(index % 192 + 1).identity,
                            units: 666_667,
                        },
                    ],
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let actors = initial
            .actors
            .iter()
            .enumerate()
            .map(|(index, actor)| (actor.identity, index as u32))
            .collect::<BTreeMap<_, _>>();
        let mut original = vec![initial.ownership.len() as u32];
        for ownership in &initial.ownership {
            original.extend([actors[&ownership.firm], ownership.shares.len() as u32]);
            original.extend(
                ownership
                    .shares
                    .iter()
                    .flat_map(|share| [actors[&share.household], share.units]),
            );
        }
        for actor in 193..initial.actors.len() {
            initial.set(Field::Cash, actor, (1 << 40) + actor as u64 * 101);
            initial.set(Field::OperatingReserve, actor, (actor as u64 % 3) << 40);
        }
        for workgroup in [32, 64, 128, 256] {
            let device = Device::headless(workgroup)?;
            let original = device.tensor(&original);
            let runtime = OwnershipRuntime::new(&device, &initial)?;
            let mut config = Configuration::new(&device, &scenario, &initial);
            for rate in [0, 333, 1000] {
                config.distribution_rate = rate;
                for cash in [5000, u64::MAX] {
                    initial.set(Field::Cash, 0, cash);
                    let words = encode(&initial.values);
                    let actual = device.tensor(bytemuck::cast_slice(&words));
                    let expected = device.tensor(bytemuck::cast_slice(&words));
                    let control = device.tensor(&[0; 4]);
                    original_distribute::launch(
                        &device.client,
                        CubeCount::Static(1, 1, 1),
                        CubeDim::new_1d(workgroup),
                        expected.clone().into_buffer_arg(),
                        config.launch(),
                        original.clone().into_buffer_arg(),
                        control.clone().into_buffer_arg(),
                    );
                    let result = runtime.distribute(&device, &actual, &config);
                    let expected_faults = FaultFlags::read(&device.read(&control)?)?;
                    assert_eq!(result.is_ok(), expected_faults.is_empty());
                    assert_eq!(
                        FaultFlags::read(&device.read(&runtime.control)?)?,
                        expected_faults
                    );
                    assert_eq!(
                        device.read(&actual)?,
                        device.read(&expected)?,
                        "workgroup {workgroup}, rate {rate}, cash {cash}"
                    );
                }
            }
        }
        Ok(())
    }
}

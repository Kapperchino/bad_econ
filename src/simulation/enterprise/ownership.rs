use crate::domain::{Field, Good, State};
use crate::simulation::configuration::{Configuration, EconomyConfig};
use crate::simulation::layout::ActorLayout;
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
use std::collections::HashMap;

enum IdentityIndex<'a> {
    Ordered(&'a [u64]),
    Unordered(HashMap<u64, usize>),
}

impl<'a> IdentityIndex<'a> {
    fn new(identities: &'a [u64]) -> Self {
        match identities.is_sorted() {
            true => Self::Ordered(identities),
            false => Self::Unordered(
                identities
                    .iter()
                    .enumerate()
                    .map(|(index, &identity)| (identity, index))
                    .collect(),
            ),
        }
    }

    fn actor(&self, identity: u64) -> Option<usize> {
        match self {
            Self::Ordered(identities) => identities.binary_search(&identity).ok(),
            Self::Unordered(identities) => identities.get(&identity).copied(),
        }
    }
}

struct ShareAccount {
    firm: usize,
    household: usize,
    units: u32,
}

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
        let actors = IdentityIndex::new(state.actors.identities());
        let mut accounts = Vec::new();
        for ownership in &state.ownership {
            let firm = actors
                .actor(ownership.firm)
                .context("missing ownership firm")?;
            for share in &ownership.shares {
                let household = actors
                    .actor(share.household)
                    .context("missing shareholder")?;
                accounts.push(ShareAccount {
                    firm,
                    household,
                    units: share.units,
                });
            }
        }
        let count = u32::try_from(
            accounts
                .len()
                .checked_mul(2)
                .context("too many share accounts")?,
        )?;
        let mut offsets = vec![0u32; state.actors.len() + 1];
        for account in &accounts {
            offsets[account.firm + 1] += 1;
            offsets[account.household + 1] += 1;
        }
        for actor in 0..state.actors.len() {
            offsets[actor + 1] += offsets[actor];
        }
        let mut entries = vec![0; count as usize];
        for (index, account) in accounts.iter().enumerate() {
            for actor in [account.firm, account.household] {
                entries[offsets[actor] as usize] = index as u32;
                offsets[actor] += 1;
            }
        }
        offsets.rotate_right(1);
        offsets[0] = 0;
        let shares = accounts
            .iter()
            .map(|account| {
                Ok(DistributionShare {
                    firm: u32::try_from(account.firm)?,
                    units: account.units,
                })
            })
            .collect::<Result<Vec<_>>>()?;
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
        actors: &CubeTensor,
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
                actors: argument(actors),
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
    actors: Box<[u32]>,
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
            &input.actors,
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
    actor_rows: &[u32],
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
        let kind = actor_rows[comptime!(ActorLayout::KIND) * actors as usize + firm as usize];
        let reserve = maximum(
            state[(comptime!(Field::OperatingReserve as u32) * actors + firm) as usize],
            select(
                kind == comptime!(Good::Housing as u32),
                zero(),
                ticks(config.reserve_low, config.reserve_high),
            ),
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
    actor_rows: &[u32],
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
            let kind = actor_rows[comptime!(ActorLayout::KIND) * actors as usize + firm as usize];
            let reserve = maximum(
                load(
                    state,
                    comptime!(Field::OperatingReserve as u32) * actors + firm,
                ),
                select(
                    kind == comptime!(Good::Housing as u32),
                    zero(),
                    ticks(config.reserve_low, config.reserve_high),
                ),
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
    use crate::simulation::{encode, layout::ActorMetadataRows};
    use std::collections::BTreeMap;

    #[test]
    fn parallel_dividends_match_serial_with_ordered_and_unordered_identities_on_gpu() -> Result<()>
    {
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
        let workgroup = 64;
        let mut initial = initial.clone();
        if workgroup >= 128 {
            let reverse = |identity| u64::MAX - identity;
            for index in 0..initial.actors.len() {
                let mut actor = initial.actors.at(index);
                actor.identity = reverse(actor.identity);
                initial.actors.set(index, actor);
            }
            for ownership in &mut initial.ownership {
                ownership.firm = reverse(ownership.firm);
                for share in &mut ownership.shares {
                    share.household = reverse(share.household);
                }
            }
        }
        let device = crate::simulation::tests::device(workgroup)?;
        let original = device.tensor(&original);
        let runtime = OwnershipRuntime::new(&device, &initial)?;
        let rows = initial
            .actors
            .iter()
            .map(|actor| ActorLayout::new(&actor, &scenario))
            .collect::<ActorMetadataRows>();
        let actors = device.tensor(&rows.words());
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
                    actors.clone().into_buffer_arg(),
                    config.launch(),
                    original.clone().into_buffer_arg(),
                    control.clone().into_buffer_arg(),
                );
                let result = runtime.distribute(&device, &actual, &actors, &config);
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
        Ok(())
    }
}

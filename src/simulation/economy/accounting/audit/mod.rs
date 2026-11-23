use crate::simulation::economy::*;
use crate::simulation::parallel::reduction::checked_sum;
use crate::simulation::{Device, DeviceEconomy};
use anyhow::Result;
use burn_wgpu::CubeTensor;
mod metrics;
pub(in crate::simulation) use metrics::AuditBuffers;

pub(in crate::simulation) const TILE_ACTORS: usize = 4096;

fn dispatch(device: &Device, groups: usize, tasks: usize) -> CubeCount {
    let width = groups
        .max(1)
        .min(device.client.properties().hardware.max_cube_count.0 as usize);
    CubeCount::Static(
        width as u32,
        tasks as u32,
        groups.max(1).div_ceil(width) as u32,
    )
}

pub(in crate::simulation) struct AuditInput<'a> {
    pub economy: DeviceEconomy<'a>,
    pub spatial: &'a CubeTensor,
    pub partials: &'a CubeTensor,
    pub summary: &'a CubeTensor,
    pub control: &'a CubeTensor,
    pub ring_mask: Option<&'a CubeTensor>,
    pub rings: RingValidation<'a>,
}

pub(in crate::simulation) enum RingValidation<'a> {
    All,
    Unchanged,
    Selected(&'a CubeTensor),
    Changed {
        entries: &'a CubeTensor,
        settlements: &'a CubeTensor,
    },
}

#[derive(Debug)]
pub(in crate::simulation) enum RegistryObservation {
    Existing,
    Updated,
}

impl RegistryObservation {
    pub(in crate::simulation) fn new(
        audit: crate::simulation::RegistryAudit,
        records: &[crate::simulation::registry::records::PendingRecord],
        settlement: Option<&crate::simulation::registry::settlement::SettlementBatch>,
    ) -> Self {
        match (
            audit,
            settlement,
            records
                .iter()
                .all(|record| matches!(record.request >> 24, 0 | 5 | 6)),
        ) {
            #[cfg(test)]
            (crate::simulation::RegistryAudit::Full, _, _) => Self::Updated,
            (_, None, true) => Self::Existing,
            _ => Self::Updated,
        }
    }
}

impl AuditInput<'_> {
    fn check_actors(&self) {
        let DeviceEconomy {
            device,
            state,
            actors,
            config,
        } = self.economy;
        let count = actors.meta.num_elements() / ActorLayout::WORDS;
        let ring_tracking = match self.ring_mask {
            Some(_) => RingTracking::Observed,
            None => RingTracking::Untracked,
        };
        let hardware = &device.client.properties().hardware;
        let actor_ring_tracking = match (
            &self.rings,
            ring_tracking,
            parallel::ScanMode::new(device),
            config.state_stride % 32,
            hardware.plane_size_min,
            hardware.plane_size_max,
        ) {
            (RingValidation::All, RingTracking::Observed, parallel::ScanMode::Plane, 0, 32, 32) => {
                RingTracking::ObservedPlane
            }
            _ => ring_tracking,
        };
        if let (RingValidation::All, Some(mask)) = (&self.rings, self.ring_mask) {
            crate::simulation::gpu::tensor::reset_ring_mask(
                device,
                mask,
                crate::simulation::gpu::tensor::RingMaskState::Empty,
            );
        }
        let ring_mask = self.ring_mask.unwrap_or(self.control);
        let validation_workgroup = 128;
        ActorValidation::launch(
            device,
            dispatch(
                device,
                count.div_ceil(validation_workgroup as usize),
                match self.rings {
                    RingValidation::All => Good::COUNT + 1,
                    RingValidation::Unchanged
                    | RingValidation::Selected(_)
                    | RingValidation::Changed { .. } => 1,
                },
            ),
            CubeDim::new_1d(validation_workgroup),
            ValidationInputLaunch {
                state: state.buffer_arg(),
                actors: actors.buffer_arg(),
                config: config.launch(),
                control: self.control.buffer_arg(),
                ring_mask: ring_mask.buffer_arg(),
                ring_tracking: actor_ring_tracking,
            },
        );
        let selected = |indices: &CubeTensor| {
            SelectedValidation::launch(
                device,
                dispatch(
                    device,
                    indices
                        .meta
                        .num_elements()
                        .div_ceil(device.workgroup as usize),
                    Good::COUNT,
                ),
                CubeDim::new_1d(device.workgroup),
                SelectedValidationInputLaunch {
                    state: state.buffer_arg(),
                    indices: indices.buffer_arg(),
                    config: config.launch(),
                    control: self.control.buffer_arg(),
                    ring_mask: ring_mask.buffer_arg(),
                    ring_tracking,
                },
            );
        };
        match self.rings {
            RingValidation::Selected(indices) => selected(indices),
            RingValidation::Changed {
                entries,
                settlements,
            } => {
                selected(entries);
                selected(settlements);
            }
            _ => (),
        }
    }
}

#[derive(CubeType, CubeLaunch)]
struct ValidationInput {
    state: Box<[Ticks]>,
    actors: Box<[u32]>,
    config: EconomyConfig,
    control: Box<[Atomic<u32>]>,
    ring_mask: Box<[Atomic<u32>]>,
    #[cube(comptime)]
    ring_tracking: RingTracking,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum RingTracking {
    Untracked,
    Observed,
    ObservedPlane,
}

struct ActorValidation;
struct ValidateActors;

#[derive(CubeType, CubeLaunch)]
struct SelectedValidationInput {
    state: Box<[Ticks]>,
    indices: Box<[u32]>,
    config: EconomyConfig,
    control: Box<[Atomic<u32>]>,
    ring_mask: Box<[Atomic<u32>]>,
    #[cube(comptime)]
    ring_tracking: RingTracking,
}

struct SelectedValidation;
struct ValidateSelected;

impl System for SelectedValidation {
    type Input = SelectedValidationInput;
    type Passes = gpu_passes![ValidateSelected];
}

#[cube]
impl GpuPass<SelectedValidationInput> for ValidateSelected {
    type Output = ();

    fn execute(input: SelectedValidationInput) {
        let index = (CUBE_POS_X + CUBE_POS_Z * CUBE_COUNT_X) * CUBE_DIM + UNIT_POS;
        if index < input.indices.len() as u32 {
            let audit = LedgerAudit::new(
                &input.state,
                &input.config,
                &input.control,
                &input.ring_mask,
                comptime!(input.ring_tracking),
            );
            audit.validate_ring(input.indices[index as usize], CUBE_POS_Y);
        }
    }
}

impl System for ActorValidation {
    type Input = ValidationInput;
    type Passes = gpu_passes![ValidateActors];
}

#[cube]
impl GpuPass<ValidationInput> for ValidateActors {
    type Output = ();

    fn execute(input: ValidationInput) {
        let actors = ActorTable::new(&input.actors);
        let audit = LedgerAudit::new(
            &input.state,
            &input.config,
            &input.control,
            &input.ring_mask,
            comptime!(input.ring_tracking),
        );
        audit.validate_actors(&actors);
    }
}

#[derive(CubeType)]
struct LedgerAudit<'a> {
    ledger: TickLedger<'a>,
    control: &'a [Atomic<u32>],
    ring_mask: &'a [Atomic<u32>],
    #[cube(comptime)]
    tracking: RingTracking,
}

#[cube]
impl<'a> LedgerAudit<'a> {
    fn new(
        state: &'a [Ticks],
        config: &'a EconomyConfig,
        control: &'a [Atomic<u32>],
        ring_mask: &'a [Atomic<u32>],
        #[comptime] tracking: RingTracking,
    ) -> Self {
        LedgerAudit::<'a> {
            ledger: TickLedger::new(state, config),
            control,
            ring_mask,
            tracking,
        }
    }

    fn validate_actors(&self, actors: &ActorTable<'_>) {
        let config = self.ledger.config;
        let actor = (CUBE_POS_X + CUBE_POS_Z * CUBE_COUNT_X) * CUBE_DIM + UNIT_POS;
        let task = CUBE_POS_Y;
        match task {
            0 => {
                self.validate_actor(actors, actor);
                self.household_histograms(actors, actor);
            }
            _ => match comptime!(self.tracking) {
                RingTracking::ObservedPlane => {
                    self.validate_ring(actor, task - 1);
                }
                _ => {
                    if actor < config.actors {
                        self.validate_ring(actor, task - 1);
                    }
                }
            },
        }
    }

    fn validate_ring(&self, actor: u32, good: u32) {
        let config = self.ledger.config;
        let mut held = 0u64;
        if comptime!(self.tracking != RingTracking::ObservedPlane) || actor < config.actors {
            held = native(self.ledger.value_at(TRANSIT_FIELDS as u32 + good, actor));
        }
        let mut in_ring = 0u64;
        for slot in 0u32..config.ring {
            let field = BASE_FIELDS as u32 + good * config.ring + slot;
            let mut waiting = zero();
            let mut amount = zero();
            if comptime!(self.tracking != RingTracking::ObservedPlane) || actor < config.actors {
                amount = self.ledger.value_at(field, actor);
                waiting = self
                    .ledger
                    .value_at(field + Good::COUNT as u32 * config.ring, actor);
            }
            self.mark_transit(actor, field, amount);
            self.mark_transit(actor, field + Good::COUNT as u32 * config.ring, waiting);
            in_ring = checked_sum(
                checked_sum(in_ring, native(amount), self.control),
                native(waiting),
                self.control,
            );
        }
        if in_ring != held {
            KernelFaults::new(comptime!(FaultFlags::TRANSIT_MISMATCH)).record(self.control);
        }
    }

    fn mark_transit(&self, actor: u32, field: u32, amount: Ticks) {
        let config = self.ledger.config;
        let block = (field * config.state_stride + actor) * 2
            / crate::simulation::gpu::tensor::SNAPSHOT_BLOCK_WORDS as u32;
        match comptime!(self.tracking) {
            RingTracking::ObservedPlane => {
                let present = plane_any(!equal(amount, zero()));
                if present && UNIT_POS % 32 == 0 {
                    self.ring_mask[block as usize].store(1);
                }
            }
            RingTracking::Observed => {
                if !equal(amount, zero()) {
                    self.ring_mask[block as usize].store(1);
                }
            }
            RingTracking::Untracked => (),
        }
    }

    fn validate_actor(&self, actors: &ActorTable<'_>, actor: u32) {
        let config = self.ledger.config;
        if actor < config.actors {
            let payroll_reserve = self.ledger.value(Field::PayrollReserve, actor);
            let tax = self.ledger.value(Field::TaxTreasury, actor);
            let charge = self.ledger.value(Field::ChargeTreasury, actor);
            if comptime!(config.payroll == Feature::Enabled) {
                let accrued = PayrollAccrual::new(
                    self.ledger.value(Field::GrossIncome, actor),
                    employment::payroll_members(actors, config, actor),
                    config,
                    self.control,
                );
                if !equal(payroll_reserve, accrued.reserve)
                    || !equal(tax, accrued.tax)
                    || !equal(charge, accrued.charge)
                {
                    KernelFaults::new(comptime!(FaultFlags::PAYROLL_MISMATCH)).record(self.control);
                }
            }
            if !equal(self.ledger.value(Field::Escrow, actor), zero())
                || !equal(self.ledger.value(Field::Pipeline, actor), zero())
                || !equal(self.ledger.value(Field::Due, actor), zero())
            {
                KernelFaults::new(comptime!(FaultFlags::UNSETTLED_COMMITMENTS))
                    .record(self.control);
            }
            let mut transit = 0u64;
            for good in 0u32..Good::COUNT as u32 {
                let held = self.ledger.value_at(TRANSIT_FIELDS as u32 + good, actor);
                transit = checked_sum(transit, native(held), self.control);
            }
            if transit != native(self.ledger.value(Field::Transit, actor)) {
                KernelFaults::new(comptime!(FaultFlags::TRANSIT_MISMATCH)).record(self.control);
            }
        }
    }

    fn household_histograms(&self, actors: &ActorTable<'_>, actor: u32) {
        let config = self.ledger.config;
        let histograms = Shared::<[Atomic<u32>]>::new_slice(comptime!(
            SummaryLayout::WEALTH_BINS + SummaryLayout::REGIONS
        ));
        let mut index = UNIT_POS;
        while index < SummaryLayout::WEALTH_BINS as u32 + SummaryLayout::REGIONS as u32 {
            histograms[index as usize].store(0);
            index += CUBE_DIM;
        }
        sync_cube();
        if actor < config.actors && actors.kind(actor).is(comptime!(ActorKind::Household)) {
            let members = select(
                comptime!(config.cohorts == Feature::Enabled),
                self.ledger.number(Field::LivingMembers, actor),
                1,
            );
            let bin = observation::wealth_bin::<TickLedger>(&self.ledger, actor, members);
            histograms[bin as usize].fetch_add(members);
            let shortage = self.ledger.number(Field::Shortage, actor);
            if shortage > 0 {
                let region = observation::histogram_region(actors, config, actor);
                histograms[(SummaryLayout::WEALTH_BINS as u32 + region) as usize]
                    .fetch_add(shortage);
            }
        }
        sync_cube();
        index = UNIT_POS;
        while index < SummaryLayout::WEALTH_BINS as u32 + SummaryLayout::REGIONS as u32 {
            let count = histograms[index as usize].load();
            let target = select(
                index < SummaryLayout::WEALTH_BINS as u32,
                SummaryLayout::WEALTH_START as u32 + index,
                SummaryLayout::REGIONS_START as u32 + index - SummaryLayout::WEALTH_BINS as u32,
            );
            if count > 0 {
                self.control[target as usize].fetch_add(count);
            }
            index += CUBE_DIM;
        }
    }
}

#[cfg(test)]
mod tests;

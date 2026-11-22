use super::*;
use crate::domain::railway::FREIGHT_CLAIM_SORT_WORDS;

const CLAIM_RECORD_WORDS: usize = FREIGHT_CLAIM_WORDS + FREIGHT_CLAIM_SORT_WORDS;

#[derive(Clone, Copy)]
pub enum ClaimStorage {
    Unallocated,
    Allocated,
}

impl ClaimStorage {
    pub fn new(config: &Configuration) -> Self {
        match config.railway.freight.claim_capacity {
            0 => Self::Unallocated,
            _ => Self::Allocated,
        }
    }

    pub fn executions(self) -> &'static [Execution] {
        match self {
            Self::Unallocated => &[Execution::Buffered(Wave::Initial)],
            Self::Allocated => &[
                Execution::Buffered(Wave::Initial),
                Execution::Buffered(Wave::Retry),
            ],
        }
    }

    pub fn stages(self) -> &'static [Stage] {
        match self {
            Self::Unallocated => &[Stage::AdmitClaims],
            Self::Allocated => &[
                Stage::AdmitClaims,
                Stage::EmitClaims,
                Stage::SortClaims,
                Stage::Resolve,
            ],
        }
    }
}

pub struct Buffers {
    pub(super) sorting: claims::SortMethod,
    pub(super) resources: CubeTensor,
    pub(super) accounts: CubeTensor,
    pub(super) bookings: CubeTensor,
    pub(super) queue: CubeTensor,
    pub(super) claims: CubeTensor,
    pub(super) claim_capacity: usize,
    pub(super) progress: CubeTensor,
    pub(super) dispatches: CubeTensor,
}

impl Buffers {
    fn claim_words(capacity: usize, resources: usize) -> usize {
        capacity * CLAIM_RECORD_WORDS + resources + 1
    }

    pub(in crate::simulation) fn graph_buffers(&self) -> [&CubeTensor; 7] {
        [
            &self.resources,
            &self.accounts,
            &self.bookings,
            &self.queue,
            &self.claims,
            &self.progress,
            &self.dispatches,
        ]
    }

    pub fn new(device: &Device, actors: usize, regions: usize) -> Self {
        let actors = match actors {
            0 => 0,
            _ => crate::simulation::registry::entries::state_capacity(actors),
        };
        let claim_capacity = 0;
        Self {
            sorting: claims::SortMethod::Block2048.for_device(device),
            resources: device.empty((actors + regions) * RESOURCE_WORDS),
            accounts: device.empty(actors * ACCOUNT_WORDS),
            bookings: device.empty(actors * FREIGHT_BOOKING_WORDS),
            queue: device.empty(actors * FREIGHT_QUEUE_WORDS),
            claims: device.empty(Self::claim_words(claim_capacity, actors + regions)),
            claim_capacity,
            progress: device.tensor(&[0; FREIGHT_PROGRESS_WORDS]),
            dispatches: device
                .client
                .memory_dedicated_allocation(FREIGHT_DISPATCH_WORDS, |words| device.empty(words)),
        }
    }

    pub fn updated(&self, device: &Device, actors: usize, regions: usize) -> Self {
        let capacity = self.accounts.meta.num_elements() / ACCOUNT_WORDS;
        match (
            actors <= capacity,
            (actors + regions) * RESOURCE_WORDS <= self.resources.meta.num_elements(),
        ) {
            (true, true) => {
                let required_claim_words = Self::claim_words(self.claim_capacity, actors + regions);
                let claims = match required_claim_words <= self.claims.meta.num_elements() {
                    true => self.claims.clone(),
                    false => {
                        device.empty(Self::claim_words(self.claim_capacity, capacity + regions))
                    }
                };
                Self {
                    sorting: self.sorting,
                    resources: self.resources.clone(),
                    accounts: self.accounts.clone(),
                    bookings: self.bookings.clone(),
                    queue: self.queue.clone(),
                    claims,
                    claim_capacity: self.claim_capacity,
                    progress: self.progress.clone(),
                    dispatches: self.dispatches.clone(),
                }
            }
            _ => {
                let mut updated = Self::new(device, actors, regions);
                updated.sorting = self.sorting;
                if self.claim_capacity > updated.claim_capacity {
                    updated.claim_capacity = self.claim_capacity;
                    updated.claims =
                        device.empty(Self::claim_words(self.claim_capacity, actors + regions));
                }
                updated
            }
        }
    }
}

pub enum ClaimWork {
    Empty,
    Ready,
}

pub struct ClaimPreparation {
    pub claims: ClaimWork,
    pub searches: u32,
}

pub(super) struct ClaimAllocation {
    capacity: usize,
    words: usize,
}

impl ClaimAllocation {
    pub(super) fn new(count: u64, config: &Configuration) -> anyhow::Result<Self> {
        use anyhow::Context;
        let resources = (config.actors + config.railway.region_count) as usize;
        let capacity = usize::try_from(count)?
            .checked_next_power_of_two()
            .context("too many freight claims")?;
        let words = capacity
            .checked_mul(CLAIM_RECORD_WORDS)
            .and_then(|words| words.checked_add(resources + 1))
            .context("freight claim storage overflow")?;
        u32::try_from(words).context("freight claims exceed GPU indexing capacity")?;
        Ok(Self { capacity, words })
    }

    pub(super) fn bytes(&self) -> u64 {
        self.words as u64 * size_of::<u32>() as u64
    }
}

pub(super) fn claim_work(
    device: &Device,
    config: &mut Configuration,
    count: u64,
) -> anyhow::Result<ClaimWork> {
    if count > config.railway.freight.claim_capacity as u64 {
        let allocation = ClaimAllocation::new(count, config)?;
        let reserved = device.reclaim_memory(allocation.bytes())?.bytes_reserved;
        match reserved.saturating_add(allocation.bytes()) <= device.memory_budget.bytes() {
            true => Ok(()),
            false => Err(anyhow::anyhow!(
                "freight claims exceed the device memory budget"
            )),
        }?;
        let buffers = &mut config.railway.freight;
        buffers.claims = device.empty(allocation.words);
        buffers.claim_capacity = allocation.capacity;
    }
    Ok(match count {
        0 => ClaimWork::Empty,
        _ => ClaimWork::Ready,
    })
}

pub fn reserve_active_market(device: &Device, config: &mut Configuration) -> anyhow::Result<()> {
    let records = u64::from(config.actors) / 8;
    if records > config.railway.freight.sorting.tile() as u64
        && config.railway.freight.claim_capacity == 0
    {
        let allocation = ClaimAllocation::new(records, config)?;
        if device
            .memory_usage()
            .bytes_reserved
            .saturating_add(allocation.bytes())
            <= device.memory_budget.bytes()
        {
            claim_work(device, config, records)?;
        }
    }
    Ok(())
}

pub fn reserve_claims_for_replay(
    device: &Device,
    config: &mut Configuration,
    count: u64,
) -> anyhow::Result<()> {
    let reserve = config.railway.freight.sorting.tile() as u64;
    let records = match ClaimAllocation::new(reserve, config) {
        Ok(allocation)
            if (1..reserve).contains(&count)
                && device
                    .memory_usage()
                    .bytes_reserved
                    .saturating_add(allocation.bytes())
                    <= device.memory_budget.bytes() =>
        {
            reserve
        }
        _ => count,
    };
    claim_work(device, config, records).map(|_| ())
}

pub fn reserve_market_claims(
    device: &Device,
    config: &mut Configuration,
    control: &CubeTensor,
) -> anyhow::Result<()> {
    let values = device.client.read(vec![
        word_range(&config.railway.freight.progress, CLAIM_COUNT, 2),
        word_range(control, crate::simulation::flags::FAULT_WORD, 1),
    ])?;
    let count = u64::from_le_bytes(values[0][..8].try_into()?);
    let faults = u32::from_le_bytes(values[1][..4].try_into()?);
    if count > 0 && faults == 0 {
        reserve_claims_for_replay(device, config, count)?;
    }
    Ok(())
}

fn word_range(tensor: &CubeTensor, start: usize, words: usize) -> cubecl::server::Handle {
    let handle = tensor.handle.clone();
    let size = handle.size_in_used();
    let offset = start * size_of::<u32>();
    handle
        .offset_start(offset as u64)
        .offset_end(size - (offset + words * size_of::<u32>()) as u64)
}

pub struct WaveBookings {
    pub initial: u32,
    pub retry: u32,
    pub faults: FaultFlags,
}

impl WaveBookings {
    pub const BYTES: usize = 3 * size_of::<u32>();

    pub fn read(
        device: &Device,
        config: &Configuration,
        control: &CubeTensor,
    ) -> anyhow::Result<Self> {
        let values = device.client.read(vec![
            word_range(&config.railway.freight.progress, Wave::Initial as usize, 2),
            word_range(control, crate::simulation::flags::FAULT_WORD, 1),
        ])?;
        Ok(Self {
            initial: u32::from_le_bytes(values[0][..4].try_into()?),
            retry: u32::from_le_bytes(values[0][4..8].try_into()?),
            faults: FaultFlags::new(u32::from_le_bytes(values[1][..4].try_into()?))?,
        })
    }
}

pub fn prepare_market_claims(
    device: &Device,
    config: &mut Configuration,
    control: &CubeTensor,
) -> anyhow::Result<ClaimPreparation> {
    let values = device.client.read(vec![
        word_range(&config.railway.freight.progress, CLAIM_COUNT, 2),
        word_range(control, crate::simulation::flags::FAULT_WORD, 1),
        word_range(control, markets::SEARCHES_WORD, 1),
    ])?;
    let count = u64::from_le_bytes(values[0][..8].try_into()?);
    let faults = u32::from_le_bytes(values[1][..4].try_into()?);
    FaultFlags::new(faults)?.into_result()?;
    Ok(ClaimPreparation {
        claims: claim_work(device, config, count)?,
        searches: u32::from_le_bytes(values[2][..4].try_into()?),
    })
}

#[cfg(test)]
pub fn prepare_claims(device: &Device, config: &mut Configuration) -> anyhow::Result<ClaimWork> {
    let count = device.read_range(&config.railway.freight.progress, CLAIM_COUNT, 2)?;
    claim_work(
        device,
        config,
        u64::from(count[0]) | (u64::from(count[1]) << 32),
    )
}

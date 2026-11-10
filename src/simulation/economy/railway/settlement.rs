use super::*;

use crate::domain::railway::{
    FREIGHT_ACCOUNT_WORDS as ACCOUNT_WORDS, FREIGHT_BOOKING_WORDS, FREIGHT_CLAIM_WORDS,
    FREIGHT_DISPATCH_WORDS, FREIGHT_QUEUE_WORDS, FREIGHT_RESOURCE_WORDS as RESOURCE_WORDS,
};
const PENDING: usize = 0;
const ACTIVE: usize = 1;
const CLAIM_COUNT: usize = 2;
const CLAIM_CURSOR: usize = 4;
const RESOURCE_CLAIMS: usize = 10;
const REQUESTED: usize = 2;
const CONSUMED: usize = 4;
const PREFERRED: usize = 6;
const REQUEST_OVERFLOW: usize = 8;
const PREFERRED_OVERFLOW: usize = 9;
const DENIED: usize = 8;

mod claims;

#[cfg(test)]
mod tests;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Stage {
    Snapshot,
    Initialize,
    Queue,
    Reset,
    Propose,
    CountClaims,
    EmitClaims,
    SortClaims,
    ClaimOffsets,
    Resolve,
    Commit,
    CommitFinal,
    Release,
    Accounts,
}

#[derive(Clone, Copy)]
#[repr(u32)]
enum BookingState {
    Pending,
    Ready,
    Complete,
    Retrying,
}

#[derive(Clone, Copy)]
#[repr(usize)]
enum Account {
    Cost = 0,
    Revenue = 2,
    Used = 4,
    Sold = 6,
}

pub struct Buffers {
    resources: CubeTensor,
    accounts: CubeTensor,
    bookings: CubeTensor,
    queue: CubeTensor,
    claims: CubeTensor,
    claim_capacity: usize,
    progress: CubeTensor,
    dispatches: CubeTensor,
}

impl Buffers {
    pub fn new(device: &Device, actors: usize, regions: usize) -> Self {
        let claim_capacity = actors * 2;
        Self {
            resources: device.empty((actors + regions) * RESOURCE_WORDS),
            accounts: device.empty(actors * ACCOUNT_WORDS),
            bookings: device.empty(actors * FREIGHT_BOOKING_WORDS),
            queue: device.empty(actors * FREIGHT_QUEUE_WORDS),
            claims: device.empty(
                claim_capacity
                    * (FREIGHT_CLAIM_WORDS + crate::domain::railway::FREIGHT_CLAIM_SORT_WORDS)
                    + actors
                    + regions
                    + 1,
            ),
            claim_capacity,
            progress: device.tensor(&[0, 0, 0, 0, 0]),
            dispatches: device
                .client
                .memory_dedicated_allocation(FREIGHT_DISPATCH_WORDS, |words| device.empty(words)),
        }
    }
}

#[derive(CubeType, CubeLaunch)]
struct DispatchInput {
    control: Box<[Atomic<u32>]>,
    dispatches: Box<[u32]>,
    workgroup: u32,
    resources: u32,
    good: u32,
}

struct FreightDispatch;
struct PrepareDispatch;

impl System for FreightDispatch {
    type Input = DispatchInput;
    type Passes = gpu_passes![PrepareDispatch];
}

#[cube]
impl GpuPass<DispatchInput> for PrepareDispatch {
    type Output = ();

    fn execute(mut input: DispatchInput) {
        let active = input.control[CONSTRUCTED_EVENTS_WORD].load();
        let mut resources = input.resources;
        if input.good == comptime!(Good::Bread as u32) {
            resources += active;
        }
        let mut counts = Array::<u32>::new(3usize);
        counts[0usize] = resources;
        counts[1usize] = active;
        counts[2usize] = max(resources, active);
        for index in 0..3usize {
            let groups = ceildiv(max(counts[index], 1), input.workgroup);
            input.dispatches[index * 3] = min(groups, 65535);
            input.dispatches[index * 3 + 1] = ceildiv(groups, 65535);
            input.dispatches[index * 3 + 2] = 1;
        }
    }
}

#[derive(CubeType, CubeLaunch)]
struct QueueInput {
    progress: Box<[Atomic<u32>]>,
    dispatches: Box<[u32]>,
    workgroup: u32,
}

struct FreightQueue;
struct PrepareQueue;

impl System for FreightQueue {
    type Input = QueueInput;
    type Passes = gpu_passes![PrepareQueue];
}

#[cube]
impl GpuPass<QueueInput> for PrepareQueue {
    type Output = ();

    fn execute(mut input: QueueInput) {
        let count = input.progress[PENDING].load();
        input.progress[ACTIVE].store(count);
        input.progress[PENDING].store(0);
        let groups = ceildiv(max(count, 1), input.workgroup);
        input.dispatches[9usize] = min(groups, 65535);
        input.dispatches[10usize] = ceildiv(groups, 65535);
        input.dispatches[11usize] = 1;
    }
}

#[derive(CubeType, CubeLaunch)]
struct Input {
    state: Box<[u32]>,
    actors: Box<[u32]>,
    spatial: Box<[u32]>,
    config: EconomyConfig,
    proposals: Box<[u32]>,
    scratch: Box<[u32]>,
    control: Box<[Atomic<u32>]>,
    resources: Box<[Atomic<u32>]>,
    accounts: Box<[Atomic<u32>]>,
    bookings: Box<[u32]>,
    queue: Box<[u32]>,
    claims: Box<[u32]>,
    claim_source: u32,
    claim_offsets: u32,
    progress: Box<[Atomic<u32>]>,
    parameters: EconomyParameters,
    #[cube(comptime)]
    stage: Stage,
}

struct FreightSettlement;
struct SettleFreight;

impl System for FreightSettlement {
    type Input = Input;
    type Passes = gpu_passes![SettleFreight];
}

pub fn launch(
    device: &Device,
    state: &CubeTensor,
    actors: &CubeTensor,
    spatial: &CubeTensor,
    config: &Configuration,
    proposals: &CubeTensor,
    scratch: &CubeTensor,
    control: &CubeTensor,
    parameters: &crate::simulation::parameters::EconomyParameters,
    stage: Stage,
) {
    let buffers = &config.railway.freight;
    if let Stage::Snapshot = stage {
        FreightDispatch::launch(
            device,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            DispatchInputLaunch {
                control: control.clone().into_buffer_arg(),
                dispatches: buffers.dispatches.clone().into_buffer_arg(),
                workgroup: device.workgroup,
                resources: config.railway.firms + config.railway.region_count,
                good: parameters.good,
            },
        );
    }
    let dispatch = match stage {
        Stage::Queue => CubeCount::Static(1, 1, 1),
        Stage::Propose | Stage::EmitClaims | Stage::Commit | Stage::CommitFinal => {
            markets::dispatch(&buffers.dispatches, 9)
        }
        Stage::Initialize | Stage::Reset => markets::dispatch(&buffers.dispatches, 6),
        Stage::ClaimOffsets => {
            device.count((config.actors + config.railway.region_count) as usize + 1)
        }
        _ => markets::dispatch(&buffers.dispatches, 0),
    };
    match stage {
        Stage::SortClaims => claims::sort(device, buffers),
        Stage::Queue => FreightQueue::launch(
            device,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            QueueInputLaunch {
                progress: buffers.progress.clone().into_buffer_arg(),
                dispatches: buffers.dispatches.clone().into_buffer_arg(),
                workgroup: device.workgroup,
            },
        ),
        _ => FreightSettlement::launch(
            device,
            dispatch,
            CubeDim::new_1d(device.workgroup),
            InputLaunch {
                state: state.clone().into_buffer_arg(),
                actors: actors.clone().into_buffer_arg(),
                spatial: spatial.clone().into_buffer_arg(),
                config: config.launch(),
                proposals: proposals.clone().into_buffer_arg(),
                scratch: scratch.clone().into_buffer_arg(),
                control: control.clone().into_buffer_arg(),
                resources: buffers.resources.clone().into_buffer_arg(),
                accounts: buffers.accounts.clone().into_buffer_arg(),
                bookings: buffers.bookings.clone().into_buffer_arg(),
                queue: buffers.queue.clone().into_buffer_arg(),
                claims: buffers.claims.clone().into_buffer_arg(),
                claim_source: claims::sorted_source(buffers.claim_capacity),
                claim_offsets: claims::offsets_start(buffers.claim_capacity),
                progress: buffers.progress.clone().into_buffer_arg(),
                parameters: parameters.launch(),
                stage,
            },
        ),
    }
}

#[cube]
fn resource(input: &Input, index: u32) -> u32 {
    let cells = input.config.spatial_side * input.config.spatial_side;
    let base = 2 * (cells + 1 + input.config.actors);
    let firms = input.spatial[(base + cells) as usize];
    let mut sellers = 0u32;
    if input.parameters.traded_good() == comptime!(Good::Bread as u32) {
        sellers = markets::active(&input.control, &input.parameters);
    }
    let mut resource = u32::new(comptime!(i64::from(u32::MAX)));
    if index < firms {
        resource = input.spatial[(base + cells + 1 + index) as usize];
    } else if index < firms + sellers {
        let slot = index - firms;
        let seller = input.scratch[(input.parameters.sort_source + slot) as usize];
        let mut first = true;
        if slot > 0 {
            first = input.scratch[(input.parameters.sort_source + slot - 1) as usize] != seller;
        }
        if first && kind(&input.actors, seller).is(comptime!(ActorKind::Household)) {
            resource = seller;
        }
    } else {
        resource = input.config.actors + index - firms - sellers;
    }
    resource
}

pub struct Progress {
    #[cfg(test)]
    pub pending: u32,
    pub searches: u32,
}

pub enum ClaimWork {
    Empty,
    Ready,
}

#[derive(Clone, Copy)]
pub enum Round {
    Initial,
    Retry,
    Complete,
}

impl Round {
    pub fn commit(self) -> Option<Stage> {
        match self {
            Self::Initial => Some(Stage::Commit),
            Self::Retry => Some(Stage::CommitFinal),
            Self::Complete => None,
        }
    }

    pub fn next(self, claims: ClaimWork) -> Self {
        match (self, claims) {
            (Self::Initial, ClaimWork::Ready) => Self::Retry,
            _ => Self::Complete,
        }
    }
}

struct ClaimAllocation {
    capacity: usize,
    words: usize,
}

impl ClaimAllocation {
    fn new(count: u64, resources: usize) -> anyhow::Result<Self> {
        use anyhow::Context;
        let capacity = usize::try_from(count)?
            .checked_next_power_of_two()
            .context("too many freight claims")?;
        let words = capacity
            .checked_mul(FREIGHT_CLAIM_WORDS + crate::domain::railway::FREIGHT_CLAIM_SORT_WORDS)
            .and_then(|words| words.checked_add(resources + 1))
            .context("freight claim storage overflow")?;
        u32::try_from(words).context("freight claims exceed GPU indexing capacity")?;
        Ok(Self { capacity, words })
    }
}

pub fn prepare_claims(device: &Device, config: &mut Configuration) -> anyhow::Result<ClaimWork> {
    let buffers = &mut config.railway.freight;
    let count = device.read_range(&buffers.progress, CLAIM_COUNT, 2)?;
    let count = u64::from(count[0]) | (u64::from(count[1]) << 32);
    if count > buffers.claim_capacity as u64 {
        let allocation = ClaimAllocation::new(
            count,
            (config.actors + config.railway.region_count) as usize,
        )?;
        let additional = allocation.words as u64 * size_of::<u32>() as u64;
        device.reclaim_memory(additional)?;
        let reserved = device
            .client
            .memory_report(cubecl::MemoryScope::Device)
            .usage()
            .bytes_reserved;
        match reserved.saturating_add(additional) <= device.memory_budget.bytes() {
            true => Ok(()),
            false => Err(anyhow::anyhow!(
                "freight claims exceed the device memory budget"
            )),
        }?;
        buffers.claims = device.empty(allocation.words);
        buffers.claim_capacity = allocation.capacity;
    }
    Ok(match count {
        0 => ClaimWork::Empty,
        _ => ClaimWork::Ready,
    })
}

pub fn progress(
    device: &Device,
    _config: &Configuration,
    control: &CubeTensor,
) -> anyhow::Result<Progress> {
    let word = |index: usize| {
        let handle = control.handle.clone();
        let size = handle.size_in_used();
        let offset = index * size_of::<u32>();
        handle
            .offset_start(offset as u64)
            .offset_end(size - (offset + size_of::<u32>()) as u64)
    };
    let values = device.client.read(vec![
        word(crate::simulation::flags::FAULT_WORD),
        word(markets::SEARCHES_WORD),
        #[cfg(test)]
        _config.railway.freight.progress.handle.clone(),
    ])?;
    let faults = u32::from_le_bytes(values[0][..4].try_into()?);
    FaultFlags::new(faults)?.into_result()?;
    Ok(Progress {
        #[cfg(test)]
        pending: u32::from_le_bytes(values[2][..4].try_into()?),
        searches: u32::from_le_bytes(values[1][..4].try_into()?),
    })
}

#[cube]
fn atomic_ticks(words: &[Atomic<u32>], index: usize) -> Ticks {
    ticks(words[index].load(), words[index + 1].load())
}

#[cube]
fn atomic_store(words: &[Atomic<u32>], index: usize, amount: Ticks) {
    words[index].store(amount.extract(LOW_WORD));
    words[index + 1].store(amount.extract(HIGH_WORD));
}

#[cube]
fn atomic_add(words: &[Atomic<u32>], index: usize, amount: Ticks, control: &[Atomic<u32>]) {
    let low = amount.extract(LOW_WORD);
    let previous = words[index].fetch_add(low);
    let carry = u32::cast_from(previous + low < previous);
    let high = amount.extract(HIGH_WORD);
    let previous_high = words[index + 1].fetch_add(high);
    let before_carry = words[index + 1].fetch_add(carry);
    if previous_high + high < previous_high || before_carry + carry < before_carry {
        KernelFaults::new(comptime!(FaultFlags::ARITHMETIC_OVERFLOW)).record(control);
    }
}

#[cube]
fn budget(input: &Input, resource: u32) -> Ticks {
    atomic_ticks(&input.resources, resource as usize * RESOURCE_WORDS)
}

#[cube]
fn claim(input: &Input, resource: u32, amount: Ticks) {
    let base = resource as usize * RESOURCE_WORDS;
    if !equal(amount, zero()) {
        input.resources[base + RESOURCE_CLAIMS].fetch_add(1);
        claims::add_demand(
            &input.resources,
            base + REQUESTED,
            base + REQUEST_OVERFLOW,
            amount,
        );
    }
}

#[cube]
fn consume_resource(input: &Input, resource: u32, amount: Ticks) {
    atomic_add(
        &input.resources,
        resource as usize * RESOURCE_WORDS + CONSUMED,
        amount,
        &input.control,
    );
}

#[cube]
fn account(input: &Input, actor: u32, amount: Ticks, #[comptime] field: Account) {
    atomic_add(
        &input.accounts,
        actor as usize * ACCOUNT_WORDS + comptime!(field as usize),
        amount,
        &input.control,
    );
}

#[derive(CubeType)]
struct RegionCursor {
    node: u32,
    origin: u32,
    carrier: u32,
    edge: u32,
    entry: u32,
    length: u32,
    steps: u32,
    previous: u32,
    pending: u32,
    first: u32,
    emitted: u32,
    finished: u32,
    reverse: bool,
}

#[cube]
impl RegionCursor {
    fn new(path: &FreightRoute) -> Self {
        RegionCursor {
            node: path.destination,
            origin: path.origin,
            carrier: path.carrier,
            edge: 0,
            entry: 0,
            length: 0,
            steps: 0,
            previous: u32::MAX,
            pending: u32::MAX,
            first: u32::MAX,
            emitted: 0,
            finished: 0,
            reverse: false,
        }
    }

    fn next(&mut self, config: &EconomyConfig) -> u32 {
        let mut result = u32::new(comptime!(i64::from(u32::MAX)));
        while result == u32::MAX && self.finished == 0 {
            let phase = select(
                self.entry < self.length,
                0,
                select(
                    self.node != self.origin && self.steps < config.railway.nodes,
                    1u32,
                    2u32,
                ),
            );
            match phase {
                0 => {
                    let start = config.railway.segments[self.edge as usize * SEGMENT_WORDS + 5];
                    let offset = select(self.reverse, self.length - self.entry - 1, self.entry);
                    let record = config.railway.segment_regions[(start + offset) as usize];
                    let region = config.railway.regions[record as usize * REGION_WORDS + 1];
                    self.first = select(self.first == u32::MAX, record, self.first);
                    if region != self.previous && self.pending < config.railway.region_count {
                        result = self.pending;
                        self.emitted += 1;
                    }
                    self.pending = record;
                    self.previous = region;
                    self.entry += 1;
                }
                1 => {
                    let nodes = config.railway.nodes;
                    self.edge = config.railway.routes[((self.carrier * nodes + self.origin) * nodes
                        + self.node)
                        as usize
                        * ROUTE_WORDS
                        + 1];
                    self.steps += 1;
                    if self.edge < config.railway.segment_count {
                        let base = self.edge as usize * SEGMENT_WORDS;
                        self.reverse = config.railway.segments[base + 1] == self.node;
                        self.node = select(
                            self.reverse,
                            config.railway.segments[base],
                            config.railway.segments[base + 1],
                        );
                        self.length = config.railway.segments[base + 6];
                        self.entry = 0;
                    } else {
                        self.finished = 1;
                    }
                }
                _ => {
                    self.finished = 1;
                    result = select(self.emitted == 0, self.first, u32::MAX);
                }
            }
        }
        result
    }
}

#[cube]
fn visits(config: &EconomyConfig, path: &FreightRoute, target: u32) -> u32 {
    let mut cursor = RegionCursor::new(path);
    let mut record = cursor.next(config);
    let mut total = 0u32;
    while record < config.railway.region_count {
        total += u32::cast_from(record == target);
        record = cursor.next(config);
    }
    total
}

#[cube]
fn fits(input: &Input, path: &FreightRoute, quantity: Ticks, sender: u32) -> bool {
    let mut allowed = !lt(
        budget(input, sender),
        multiply(&input.control, quantity, path.fee),
    );
    let mut cursor = RegionCursor::new(path);
    let mut record = cursor.next(&input.config);
    let upper = multiply(&input.control, quantity, input.config.railway.nodes);
    while record < input.config.railway.region_count && allowed {
        let stock = budget(input, input.config.actors + record);
        if lt(stock, upper) {
            let required = multiply(
                &input.control,
                quantity,
                visits(&input.config, path, record),
            );
            allowed = !lt(stock, required);
        }
        record = cursor.next(&input.config);
    }
    allowed
}

#[cube]
fn quantity(input: &Input, buyer: u32) -> Ticks {
    ticks(
        input.proposals[buyer as usize * ProposalLayout::WORDS + ProposalLayout::QUANTITY_LOW],
        input.proposals[buyer as usize * ProposalLayout::WORDS + ProposalLayout::QUANTITY_HIGH],
    )
}

#[cube]
fn propose(input: &mut Input, invocation: u32) {
    let count = input.parameters.count;
    if invocation < input.progress[ACTIVE].load() {
        let index = input.queue[invocation as usize];
        let id = input.scratch[(input.parameters.sort_source + count + index) as usize];
        let buyer = id + input.parameters.offset;
        let booking = input.bookings[buyer as usize * 2];
        if booking != comptime!(BookingState::Complete as u32) {
            let seller =
                input.proposals[buyer as usize * ProposalLayout::WORDS + ProposalLayout::ACTOR];
            let amount = quantity(&*input, buyer);
            let mut selected = animal_route(&input.actors, &input.config, buyer, seller);
            let preferred = route(&input.actors, &input.config, buyer, seller);
            if preferred.carrier < input.config.railway.company_count {
                let mut cursor = RegionCursor::new(&preferred);
                let mut record = cursor.next(&input.config);
                while record < input.config.railway.region_count {
                    claims::add_demand(
                        &input.resources,
                        (input.config.actors + record) as usize * RESOURCE_WORDS + PREFERRED,
                        (input.config.actors + record) as usize * RESOURCE_WORDS
                            + PREFERRED_OVERFLOW,
                        amount,
                    );
                    record = cursor.next(&input.config);
                }
            }
            let origin = input.config.railway.access[seller as usize * 2];
            let mut start = 0u32;
            let mut end = 0u32;
            if origin < input.config.railway.nodes {
                start = input.config.railway.station_companies[origin as usize];
                end = input.config.railway.station_companies[origin as usize + 1];
            }
            if booking == comptime!(BookingState::Retrying as u32) {
                let previous = candidate(
                    &input.actors,
                    &input.config,
                    buyer,
                    seller,
                    input.bookings[buyer as usize * 2 + 1],
                );
                if fits(&*input, &previous, amount, seller) {
                    selected.replace(&previous);
                    start = end;
                }
            }
            for entry in start..end {
                let company = input.config.railway.station_companies[entry as usize];
                let path = candidate(&input.actors, &input.config, buyer, seller, company);
                if path.carrier < input.config.railway.company_count
                    && path.preferred_to(&selected)
                    && fits(&*input, &path, amount, seller)
                {
                    selected.replace(&path);
                }
            }
            input.bookings[buyer as usize * 2] = comptime!(BookingState::Complete as u32);
            if selected.carrier < input.config.railway.company_count {
                input.bookings[buyer as usize * 2] = comptime!(BookingState::Ready as u32);
                input.bookings[buyer as usize * 2 + 1] = selected.carrier;
                let cost = multiply(&input.control, amount, selected.fee);
                claim(&*input, seller, cost);
                let mut cursor = RegionCursor::new(&selected);
                let mut record = cursor.next(&input.config);
                while record < input.config.railway.region_count {
                    claim(&*input, input.config.actors + record, amount);
                    record = cursor.next(&input.config);
                }
            }
        }
    }
}

#[cube]
fn toll(input: &Input, carrier: u32, owner: u32, units: u32, quantity: Ticks) {
    if owner < input.config.railway.company_count {
        let companies = &input.config.railway.companies;
        if companies[owner as usize * COMPANY_WORDS + 3]
            != companies[carrier as usize * COMPANY_WORDS + 3]
        {
            let fee = max(units, 1) * companies[owner as usize * COMPANY_WORDS + 2];
            account(
                input,
                companies[owner as usize * COMPANY_WORDS],
                multiply(&input.control, quantity, fee),
                comptime!(Account::Revenue),
            );
        }
    }
}

#[cube]
fn credit_tolls(input: &Input, path: &FreightRoute, quantity: Ticks) {
    let config = &input.config;
    let mut node = path.destination;
    let mut owner = u32::new(comptime!(i64::from(u32::MAX)));
    let mut region = u32::new(comptime!(i64::from(u32::MAX)));
    let mut units = 0u32;
    let mut steps = 0u32;
    while node != path.origin && steps < config.railway.nodes {
        let edge = config.railway.routes[((path.carrier * config.railway.nodes + path.origin)
            * config.railway.nodes
            + node) as usize
            * ROUTE_WORDS
            + 1];
        if edge < config.railway.segment_count {
            let base = edge as usize * SEGMENT_WORDS;
            let next_owner = config.railway.segments[base + 2];
            if next_owner != owner {
                toll(input, path.carrier, owner, units, quantity);
                owner = next_owner;
                region = u32::MAX;
                units = 0;
            }
            let reverse = config.railway.segments[base + 1] == node;
            let start = config.railway.segments[base + 5];
            let length = config.railway.segments[base + 6];
            for entry in 0..length {
                let offset = select(reverse, length - entry - 1, entry);
                let record = config.railway.segment_regions[(start + offset) as usize];
                let next_region = config.railway.regions[record as usize * REGION_WORDS + 1];
                units += u32::cast_from(region != u32::MAX && region != next_region);
                region = next_region;
            }
            node = select(
                reverse,
                config.railway.segments[base],
                config.railway.segments[base + 1],
            );
        }
        steps += 1;
    }
    toll(input, path.carrier, owner, units, quantity);
}

#[cube]
fn commit(input: &mut Input, invocation: u32) {
    let count = input.parameters.count;
    if invocation < input.progress[ACTIVE].load() {
        let index = input.queue[invocation as usize];
        let id = input.scratch[(input.parameters.sort_source + count + index) as usize];
        let buyer = id + input.parameters.offset;
        if input.bookings[buyer as usize * 2] == comptime!(BookingState::Ready as u32) {
            let seller =
                input.proposals[buyer as usize * ProposalLayout::WORDS + ProposalLayout::ACTOR];
            let path = candidate(
                &input.actors,
                &input.config,
                buyer,
                seller,
                input.bookings[buyer as usize * 2 + 1],
            );
            let amount = quantity(&*input, buyer);
            let allowed = input.accounts[buyer as usize * ACCOUNT_WORDS + DENIED].load() == 0;
            input.bookings[buyer as usize * 2] = comptime!(BookingState::Retrying as u32);
            if allowed {
                let good = input.parameters.traded_good();
                let animal = animal_route(&input.actors, &input.config, buyer, seller);
                let base = BASE_FIELDS as u32 + (good - 1) * input.config.ring;
                let old_field = base + (input.parameters.day + animal.days) % input.config.ring;
                let new_field = base + (input.parameters.day + path.days) % input.config.ring;
                let old = value(&mut input.state, &input.config, old_field, buyer);
                let new = add_checked(
                    &input.control,
                    value(&mut input.state, &input.config, new_field, buyer),
                    amount,
                );
                set(
                    &mut input.state,
                    &input.config,
                    old_field,
                    buyer,
                    minus(old, amount),
                );
                set(&mut input.state, &input.config, new_field, buyer, new);
                let cost = multiply(&input.control, amount, path.fee);
                consume_resource(&*input, seller, cost);
                account(&*input, seller, cost, comptime!(Account::Cost));
                let mut cursor = RegionCursor::new(&path);
                let mut record = cursor.next(&input.config);
                let mut units = 0u32;
                while record < input.config.railway.region_count {
                    consume_resource(&*input, input.config.actors + record, amount);
                    let owner = input.config.railway.regions[record as usize * REGION_WORDS];
                    let firm = input.config.railway.companies[owner as usize * COMPANY_WORDS];
                    account(&*input, firm, amount, comptime!(Account::Sold));
                    units += 1;
                    record = cursor.next(&input.config);
                }
                let total = multiply(&input.control, amount, units);
                let carrier = input.config.railway.companies[path.carrier as usize * COMPANY_WORDS];
                let rate =
                    input.config.railway.companies[path.carrier as usize * COMPANY_WORDS + 1];
                account(&*input, seller, total, comptime!(Account::Used));
                account(
                    &*input,
                    carrier,
                    multiply(&input.control, total, rate),
                    comptime!(Account::Revenue),
                );
                credit_tolls(&*input, &path, amount);
                input.bookings[buyer as usize * 2] = comptime!(BookingState::Complete as u32);
            } else {
                match comptime!(input.stage) {
                    Stage::Commit => {
                        let next = input.progress[PENDING].fetch_add(1);
                        input.queue[(input.config.actors + next) as usize] = index;
                    }
                    _ => {
                        input.bookings[buyer as usize * FREIGHT_BOOKING_WORDS] =
                            comptime!(BookingState::Complete as u32)
                    }
                }
            }
        }
    }
}

#[cube]
impl GpuPass<Input> for SettleFreight {
    type Output = ();

    fn execute(mut input: Input) {
        let invocation = ABSOLUTE_POS as u32;
        let id = resource(&input, invocation);
        let actors = input.config.actors;
        let resources = actors + input.config.railway.region_count;
        let base = id as usize * RESOURCE_WORDS;
        match comptime!(input.stage) {
            Stage::Snapshot => {
                if invocation == 0 {
                    input.progress[PENDING].store(0);
                }
                if id < actors {
                    atomic_store(
                        &input.resources,
                        base,
                        value(
                            &mut input.state,
                            &input.config,
                            comptime!(Field::Cash as u32),
                            id,
                        ),
                    );
                }
            }
            Stage::Initialize => {
                if id < actors {
                    let cash = value(
                        &mut input.state,
                        &input.config,
                        comptime!(Field::Cash as u32),
                        id,
                    );
                    let available = minimum(cash, budget(&input, id));
                    atomic_store(&input.resources, base, available);
                    for field in 0..ACCOUNT_WORDS {
                        input.accounts[id as usize * ACCOUNT_WORDS + field].store(0);
                    }
                }
                if invocation < markets::active(&input.control, &input.parameters) {
                    let buyer = input.scratch[(input.parameters.sort_source
                        + input.parameters.count
                        + invocation) as usize]
                        + input.parameters.offset;
                    let status = KernelEventStatus::load(
                        &mut input.scratch,
                        4 * input.parameters.count + buyer - input.parameters.offset,
                    );
                    input.bookings[buyer as usize * 2] = select(
                        status.contains(comptime!(EventStatus::Accepted)),
                        comptime!(BookingState::Pending as u32),
                        comptime!(BookingState::Complete as u32),
                    );
                    if status.contains(comptime!(EventStatus::Accepted)) {
                        let next = input.progress[PENDING].fetch_add(1);
                        input.queue[(actors + next) as usize] = invocation;
                    }
                }
                if id >= actors && id < resources {
                    let stock = regional_value(
                        &mut input.state,
                        &input.config,
                        id - actors,
                        comptime!(RailCapacityField::Available),
                    );
                    atomic_store(&input.resources, base, stock);
                }
            }
            Stage::Queue | Stage::SortClaims => (),
            Stage::Reset => {
                if invocation == 0 {
                    input.progress[CLAIM_COUNT].store(0);
                    input.progress[CLAIM_COUNT + 1].store(0);
                    input.progress[CLAIM_CURSOR].store(0);
                }
                if invocation < input.progress[ACTIVE].load() {
                    let priority = input.queue[(actors + invocation) as usize];
                    input.queue[invocation as usize] = priority;
                    let buyer = input.scratch[(input.parameters.sort_source
                        + input.parameters.count
                        + priority) as usize]
                        + input.parameters.offset;
                    input.accounts[buyer as usize * ACCOUNT_WORDS + DENIED].store(0);
                }
                if id < resources {
                    for offset in 2..RESOURCE_WORDS {
                        input.resources[base + offset].store(0);
                    }
                }
            }
            Stage::Propose => propose(&mut input, invocation),
            Stage::CountClaims => {
                if id < resources {
                    let count = select(
                        claims::contested(&input, id),
                        input.resources[base + RESOURCE_CLAIMS].load(),
                        0,
                    );
                    atomic_add(
                        &input.progress,
                        CLAIM_COUNT,
                        ticks(count, 0),
                        &input.control,
                    );
                }
            }
            Stage::EmitClaims => claims::emit(&mut input, invocation),
            Stage::ClaimOffsets => claims::offset(&mut input, invocation),
            Stage::Resolve => {
                if id < resources {
                    claims::resolve(&mut input, id);
                }
            }
            Stage::Commit | Stage::CommitFinal => commit(&mut input, invocation),
            Stage::Release => {
                if id < resources {
                    let available = budget(&input, id);
                    let used = atomic_ticks(&input.resources, base + CONSUMED);
                    atomic_store(&input.resources, base, minus(available, used));
                }
                if id >= actors && id < resources {
                    let region = id - actors;
                    let available = budget(&input, id);
                    regional_set(
                        &mut input.state,
                        &input.config,
                        region,
                        available,
                        comptime!(RailCapacityField::Available),
                    );
                    let level = regional_number(
                        &mut input.state,
                        &input.config,
                        region,
                        comptime!(RailCapacityField::Level),
                    );
                    let before =
                        plus(available, atomic_ticks(&input.resources, base + CONSUMED)).value;
                    let used = minus(times32(level, input.config.railway.units_per_level), before);
                    let wanted = claims::saturated_sum(
                        used,
                        maximum(
                            claims::demand(
                                &input.resources,
                                base + REQUESTED,
                                base + REQUEST_OVERFLOW,
                            ),
                            claims::demand(
                                &input.resources,
                                base + PREFERRED,
                                base + PREFERRED_OVERFLOW,
                            ),
                        ),
                    );
                    let previous = regional_value(
                        &mut input.state,
                        &input.config,
                        region,
                        comptime!(RailCapacityField::Demand),
                    );
                    regional_set(
                        &mut input.state,
                        &input.config,
                        region,
                        maximum(previous, wanted),
                        comptime!(RailCapacityField::Demand),
                    );
                }
            }
            Stage::Accounts => {
                if id < actors {
                    let offset = id as usize * ACCOUNT_WORDS;
                    let cost =
                        atomic_ticks(&input.accounts, offset + comptime!(Account::Cost as usize));
                    let revenue = atomic_ticks(
                        &input.accounts,
                        offset + comptime!(Account::Revenue as usize),
                    );
                    let used =
                        atomic_ticks(&input.accounts, offset + comptime!(Account::Used as usize));
                    let sold =
                        atomic_ticks(&input.accounts, offset + comptime!(Account::Sold as usize));
                    let cash = value(
                        &mut input.state,
                        &input.config,
                        comptime!(Field::Cash as u32),
                        id,
                    );
                    let cash = add_checked(&input.control, minus(cash, cost), revenue);
                    set(
                        &mut input.state,
                        &input.config,
                        comptime!(Field::Cash as u32),
                        id,
                        cash,
                    );
                    increase(
                        &mut input.state,
                        &input.config,
                        &input.control,
                        comptime!(Field::RailRevenue as u32),
                        id,
                        revenue,
                    );
                    record_use(&mut input.state, &input.config, &input.control, id, used);
                    let stock = value(
                        &mut input.state,
                        &input.config,
                        comptime!(Field::Railway as u32),
                        id,
                    );
                    set(
                        &mut input.state,
                        &input.config,
                        comptime!(Field::Railway as u32),
                        id,
                        minus(stock, sold),
                    );
                    increase(
                        &mut input.state,
                        &input.config,
                        &input.control,
                        comptime!(Field::Sales as u32),
                        id,
                        sold,
                    );
                }
            }
        }
    }
}

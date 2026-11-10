use super::*;

use crate::domain::railway::{
    FREIGHT_ACCOUNT_WORDS as ACCOUNT_WORDS, FREIGHT_BOOKING_WORDS,
    FREIGHT_RESOURCE_WORDS as RESOURCE_WORDS,
};
const PENDING: usize = 0;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Stage {
    Snapshot,
    Initialize,
    Reset,
    Propose,
    Commit,
    Release,
    Accounts,
}

#[derive(Clone, Copy)]
#[repr(u32)]
enum BookingState {
    Pending,
    Ready,
    Complete,
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
    progress: CubeTensor,
}

impl Buffers {
    pub fn new(device: &Device, actors: usize, regions: usize) -> Self {
        Self {
            resources: device.empty((actors + regions) * RESOURCE_WORDS),
            accounts: device.empty(actors * ACCOUNT_WORDS),
            bookings: device.empty(actors * FREIGHT_BOOKING_WORDS),
            progress: device.tensor(&[0]),
        }
    }
}

#[derive(CubeType, CubeLaunch)]
struct Input {
    state: Box<[u32]>,
    actors: Box<[u32]>,
    config: EconomyConfig,
    proposals: Box<[u32]>,
    scratch: Box<[u32]>,
    control: Box<[Atomic<u32>]>,
    resources: Box<[Atomic<u32>]>,
    accounts: Box<[Atomic<u32>]>,
    bookings: Box<[u32]>,
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
    config: &Configuration,
    proposals: &CubeTensor,
    scratch: &CubeTensor,
    control: &CubeTensor,
    parameters: &crate::simulation::parameters::EconomyParameters,
    stage: Stage,
) {
    let buffers = &config.railway.freight;
    let count = match stage {
        Stage::Reset | Stage::Initialize | Stage::Release => {
            config.actors + config.railway.region_count
        }
        _ => config.actors,
    };
    FreightSettlement::launch(
        device,
        device.count(count as usize),
        CubeDim::new_1d(device.workgroup),
        InputLaunch {
            state: state.clone().into_buffer_arg(),
            actors: actors.clone().into_buffer_arg(),
            config: config.launch(),
            proposals: proposals.clone().into_buffer_arg(),
            scratch: scratch.clone().into_buffer_arg(),
            control: control.clone().into_buffer_arg(),
            resources: buffers.resources.clone().into_buffer_arg(),
            accounts: buffers.accounts.clone().into_buffer_arg(),
            bookings: buffers.bookings.clone().into_buffer_arg(),
            progress: buffers.progress.clone().into_buffer_arg(),
            parameters: parameters.launch(),
            stage,
        },
    );
}

pub fn pending(device: &Device, config: &Configuration) -> anyhow::Result<u32> {
    Ok(device.read(&config.railway.freight.progress)?[PENDING])
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
fn claim(input: &Input, resource: u32, amount: Ticks, maximum: Ticks, priority: u32) {
    let base = resource as usize * RESOURCE_WORDS;
    atomic_add(&input.resources, base + 2, amount, &input.control);
    input.resources[base + 4].fetch_min(priority);
    input.resources[base + 9].fetch_max(maximum.extract(LOW_WORD));
    input.resources[base + 10].fetch_max(maximum.extract(HIGH_WORD));
}

#[cube]
fn approved(input: &Input, resource: u32, priority: u32) -> bool {
    let base = resource as usize * RESOURCE_WORDS;
    let first = input.resources[base + 4].load();
    let prefix = times(
        atomic_ticks(&input.resources, base + 9),
        priority - first + 1,
    );
    !lt(
        budget(input, resource),
        atomic_ticks(&input.resources, base + 2),
    ) || first == priority
        || prefix.faults.is_empty() && !lt(budget(input, resource), prefix.value)
}

#[cube]
fn consume_resource(input: &Input, resource: u32, amount: Ticks) {
    atomic_add(
        &input.resources,
        resource as usize * RESOURCE_WORDS + 5,
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
fn propose(input: &mut Input, index: u32) {
    let count = input.parameters.count;
    if index < markets::active(&input.control, &input.parameters) {
        let id = input.scratch[(input.parameters.sort_source + count + index) as usize];
        let buyer = id + input.parameters.offset;
        if input.bookings[buyer as usize * 2] == comptime!(BookingState::Pending as u32) {
            let seller =
                input.proposals[buyer as usize * ProposalLayout::WORDS + ProposalLayout::ACTOR];
            let amount = quantity(&*input, buyer);
            let mut selected = animal_route(&input.actors, &input.config, buyer, seller);
            let preferred = route(&input.actors, &input.config, buyer, seller);
            if preferred.carrier < input.config.railway.company_count {
                let mut cursor = RegionCursor::new(&preferred);
                let mut record = cursor.next(&input.config);
                while record < input.config.railway.region_count {
                    atomic_add(
                        &input.resources,
                        (input.config.actors + record) as usize * RESOURCE_WORDS + 7,
                        amount,
                        &input.control,
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
                claim(&*input, seller, cost, cost, index);
                let mut cursor = RegionCursor::new(&selected);
                let mut record = cursor.next(&input.config);
                while record < input.config.railway.region_count {
                    claim(
                        &*input,
                        input.config.actors + record,
                        amount,
                        multiply(&input.control, amount, input.config.railway.nodes),
                        index,
                    );
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
fn commit(input: &mut Input, index: u32) {
    let count = input.parameters.count;
    if index < markets::active(&input.control, &input.parameters) {
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
            let mut allowed = approved(&*input, seller, index);
            let mut cursor = RegionCursor::new(&path);
            let mut record = cursor.next(&input.config);
            while record < input.config.railway.region_count {
                allowed = allowed && approved(&*input, input.config.actors + record, index);
                record = cursor.next(&input.config);
            }
            input.bookings[buyer as usize * 2] = comptime!(BookingState::Pending as u32);
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
                input.progress[PENDING].fetch_add(1);
            }
        }
    }
}

#[cube]
impl GpuPass<Input> for SettleFreight {
    type Output = ();

    fn execute(mut input: Input) {
        let id = ABSOLUTE_POS as u32;
        let actors = input.config.actors;
        let resources = actors + input.config.railway.region_count;
        let base = id as usize * RESOURCE_WORDS;
        match comptime!(input.stage) {
            Stage::Snapshot => {
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
                    let status = KernelEventStatus::load(
                        &mut input.scratch,
                        4 * input.parameters.count + id,
                    );
                    input.bookings[id as usize * 2] = select(
                        status.contains(comptime!(EventStatus::Accepted)),
                        comptime!(BookingState::Pending as u32),
                        comptime!(BookingState::Complete as u32),
                    );
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
            Stage::Reset => {
                if id < resources {
                    for offset in 2..RESOURCE_WORDS {
                        input.resources[base + offset].store(select(offset == 4, u32::MAX, 0));
                    }
                }
                if id == 0 {
                    input.progress[PENDING].store(0);
                }
            }
            Stage::Propose => propose(&mut input, id),
            Stage::Commit => commit(&mut input, id),
            Stage::Release => {
                if id < resources {
                    let available = budget(&input, id);
                    let used = atomic_ticks(&input.resources, base + 5);
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
                    let before = plus(available, atomic_ticks(&input.resources, base + 5)).value;
                    let used = minus(times32(level, input.config.railway.units_per_level), before);
                    let wanted = add_checked(
                        &input.control,
                        used,
                        maximum(
                            atomic_ticks(&input.resources, base + 2),
                            atomic_ticks(&input.resources, base + 7),
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

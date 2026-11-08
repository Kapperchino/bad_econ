use super::*;
use crate::domain::railway::RailwayConfig;
use crate::simulation::{Device, configuration::Configuration};
use burn_wgpu::CubeTensor;

const COMPANY_WORDS: usize = 4;
const SEGMENT_WORDS: usize = 5;
const ROUTE_WORDS: usize = 5;
const UNREACHABLE: u32 = u32::MAX / 4;

#[derive(CubeType, CubeLaunch)]
pub struct RailNetwork {
    #[cube(comptime)]
    pub enabled: Feature,
    pub nodes: u32,
    pub company_count: u32,
    pub segment_count: u32,
    pub speed_multiplier: u32,
    pub length_unit: u32,
    pub stations: Box<[u32]>,
    pub companies: Box<[u32]>,
    pub segments: Box<[u32]>,
    pub materials: Box<[u32]>,
    pub routes: Box<[u32]>,
    pub receipts: Box<[Atomic<u32>]>,
}

pub struct Network {
    nodes: u32,
    company_count: u32,
    segment_count: u32,
    speed_multiplier: u32,
    length_unit: u32,
    stations: CubeTensor,
    companies: CubeTensor,
    segments: CubeTensor,
    materials: CubeTensor,
    routes: CubeTensor,
    receipts: CubeTensor,
}

impl Network {
    pub fn new(device: &Device, config: &RailwayConfig, state: &crate::domain::State) -> Self {
        let stations = config.stations();
        let companies = config
            .companies
            .iter()
            .flat_map(|company| {
                [
                    state
                        .actors
                        .iter()
                        .position(|actor| actor.identity == company.firm)
                        .expect("railway firm") as u32,
                    company.freight_rate,
                    company.access_fee,
                    state
                        .actors
                        .iter()
                        .position(|actor| actor.identity == state.company_identity(company.firm))
                        .expect("company founding site") as u32,
                ]
            })
            .collect::<Vec<_>>();
        let mut ordinals = vec![0; config.companies.len()];
        let segments = config
            .segments
            .iter()
            .flat_map(|segment| {
                let company = config
                    .companies
                    .iter()
                    .position(|company| company.firm == segment.company)
                    .expect("railway company");
                let ordinal = ordinals[company];
                ordinals[company] += 1;
                [
                    stations
                        .iter()
                        .position(|position| *position == segment.from)
                        .expect("station") as u32,
                    stations
                        .iter()
                        .position(|position| *position == segment.to)
                        .expect("station") as u32,
                    company as u32,
                    segment.from.distance(segment.to),
                    ordinal,
                ]
            })
            .collect::<Vec<_>>();
        Self {
            nodes: stations.len() as u32,
            company_count: config.companies.len() as u32,
            segment_count: config.segments.len() as u32,
            speed_multiplier: config.speed_multiplier,
            length_unit: config.length_unit,
            stations: device.tensor(
                &stations
                    .iter()
                    .flat_map(|position| [position.x, position.y])
                    .collect::<Vec<_>>(),
            ),
            companies: device.tensor(&companies),
            segments: device.tensor(&segments),
            materials: device.tensor(
                &config
                    .construction
                    .quantities()
                    .into_iter()
                    .chain(config.maintenance.quantities())
                    .collect::<Vec<_>>(),
            ),
            routes: device.empty((stations.len() * stations.len() * ROUTE_WORDS).max(1)),
            receipts: device.empty((config.companies.len() * 4).max(1)),
        }
    }

    pub fn launch(&self) -> RailNetworkLaunch {
        RailNetworkLaunch {
            enabled: match self.company_count {
                0 => Feature::Disabled,
                _ => Feature::Enabled,
            },
            nodes: self.nodes,
            company_count: self.company_count,
            segment_count: self.segment_count,
            speed_multiplier: self.speed_multiplier,
            length_unit: self.length_unit,
            stations: self.stations.clone().into_buffer_arg(),
            companies: self.companies.clone().into_buffer_arg(),
            segments: self.segments.clone().into_buffer_arg(),
            materials: self.materials.clone().into_buffer_arg(),
            routes: self.routes.clone().into_buffer_arg(),
            receipts: self.receipts.clone().into_buffer_arg(),
        }
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum RailStage {
    Infrastructure,
    Routes,
    ClearFees,
    Fees,
    #[cfg_attr(not(test), allow(dead_code))]
    ReferenceFees,
    PayFees,
}

#[derive(CubeType, CubeLaunch)]
struct RailInput {
    state: Box<[u32]>,
    actors: Box<[u32]>,
    config: EconomyConfig,
    proposals: Box<[u32]>,
    scratch: Box<[u32]>,
    control: Box<[Atomic<u32>]>,
    sort_source: u32,
    #[cube(comptime)]
    stage: RailStage,
}

struct RailSystem;
struct AdvanceRailways;
impl System for RailSystem {
    type Input = RailInput;
    type Passes = gpu_passes![AdvanceRailways];
}

#[cube]
impl GpuPass<RailInput> for AdvanceRailways {
    type Output = ();
    fn execute(mut input: RailInput) {
        match comptime!(input.stage) {
            RailStage::Infrastructure => {
                infrastructure(&mut input.state, &input.config, &input.control)
            }
            RailStage::Routes => routes(&mut input.state, &mut input.config),
            RailStage::ClearFees => clear_fees(&input.config),
            RailStage::PayFees => pay_fees(&mut input.state, &input.config, &input.control),
            RailStage::Fees | RailStage::ReferenceFees => fees(
                &mut input.state,
                &input.actors,
                &input.config,
                &input.proposals,
                &input.scratch,
                &input.control,
                input.sort_source,
                comptime!(input.stage),
            ),
        }
    }
}

pub fn launch(
    device: &Device,
    state: &CubeTensor,
    actors: &CubeTensor,
    config: &Configuration,
    proposals: &CubeTensor,
    scratch: &CubeTensor,
    control: &CubeTensor,
    stage: RailStage,
    sort_source: u32,
) {
    if config.railway.company_count > 0 {
        let stages = match stage {
            RailStage::Fees | RailStage::ReferenceFees => {
                vec![RailStage::ClearFees, stage, RailStage::PayFees]
            }
            _ => vec![stage],
        };
        for stage in stages {
            let count = match stage {
                RailStage::Infrastructure | RailStage::ClearFees | RailStage::PayFees => {
                    config.railway.company_count
                }
                RailStage::Routes => config.railway.nodes,
                RailStage::Fees | RailStage::ReferenceFees => config.actors,
            };
            RailSystem::launch(
                device,
                device.count(count as usize),
                CubeDim::new_1d(device.workgroup),
                RailInputLaunch {
                    state: state.clone().into_buffer_arg(),
                    actors: actors.clone().into_buffer_arg(),
                    config: config.launch(),
                    proposals: proposals.clone().into_buffer_arg(),
                    scratch: scratch.clone().into_buffer_arg(),
                    control: control.clone().into_buffer_arg(),
                    stage,
                    sort_source,
                },
            );
        }
    }
}

#[cube]
fn material(index: u32) -> u32 {
    let mut good = u32::new(comptime!(Good::Coal as i64));
    match index {
        1 => good = comptime!(Good::Steel as u32),
        2 => good = comptime!(Good::Locomotives as u32),
        _ => (),
    }
    good
}

#[cube]
fn supplied(state: &mut [u32], config: &EconomyConfig, firm: u32, units: u32, offset: u32) -> bool {
    let mut funded = true;
    for index in 0..3u32 {
        let required = times32(units, config.railway.materials[(offset + index) as usize]);
        funded = funded && !lt(value(state, config, material(index), firm), required);
    }
    funded
}

#[cube]
fn consume(
    state: &mut [u32],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
    firm: u32,
    units: u32,
    offset: u32,
) {
    for index in 0..3u32 {
        let good = material(index);
        let required = times32(units, config.railway.materials[(offset + index) as usize]);
        let remaining = minus(value(state, config, good, firm), required);
        set(state, config, good, firm, remaining);
        increase(
            state,
            config,
            control,
            USED_FIELDS as u32 + good - 1,
            firm,
            required,
        );
        increase(
            state,
            config,
            control,
            comptime!(Field::Used as u32),
            firm,
            required,
        );
    }
}

#[cube]
fn infrastructure(state: &mut [u32], config: &EconomyConfig, control: &[Atomic<u32>]) {
    let company = ABSOLUTE_POS as u32;
    if company < config.railway.company_count {
        let firm = config.railway.companies[company as usize * COMPANY_WORDS];
        let mut built = number(state, config, comptime!(Field::RailBuilt as u32), firm);
        let mut units = 0u32;
        let mut next = 0u32;
        for edge in 0..config.railway.segment_count {
            let base = edge as usize * SEGMENT_WORDS;
            let own = config.railway.segments[base + 2] == company;
            let ordinal = config.railway.segments[base + 4];
            let length = ceildiv(
                config.railway.segments[base + 3],
                config.railway.length_unit,
            );
            units += select(own && ordinal < built, length, 0);
            next += select(own && ordinal == built, length, 0);
        }
        let active = number(
            state,
            config,
            comptime!(Field::EnterpriseStatus as u32),
            firm,
        ) == comptime!(crate::domain::enterprise::EnterpriseState::Operating as u32);
        let maintained = active && supplied(state, config, firm, units, 3);
        if maintained {
            consume(state, config, control, firm, units, 3);
        }
        let build = maintained && next > 0 && supplied(state, config, firm, next, 0);
        if build {
            consume(state, config, control, firm, next, 0);
            built += 1;
        }
        set32(
            state,
            config,
            comptime!(Field::RailBuilt as u32),
            firm,
            built,
        );
        set32(
            state,
            config,
            comptime!(Field::RailOperating as u32),
            firm,
            select(maintained, built, 0),
        );
        set32(state, config, comptime!(Field::RailRevenue as u32), firm, 0);
    }
}

#[cube]
pub(super) fn demand(state: &mut [u32], config: &EconomyConfig, firm: u32, good: u32) -> Ticks {
    let mut wanted = zero();
    if comptime!(config.railway.enabled == Feature::Enabled) {
        for company in 0..config.railway.company_count {
            if config.railway.companies[company as usize * COMPANY_WORDS] == firm {
                let built = number(state, config, comptime!(Field::RailBuilt as u32), firm);
                let mut units = 0;
                let mut next = 0;
                for edge in 0..config.railway.segment_count {
                    let base = edge as usize * SEGMENT_WORDS;
                    let own = config.railway.segments[base + 2] == company;
                    let ordinal = config.railway.segments[base + 4];
                    let length = ceildiv(
                        config.railway.segments[base + 3],
                        config.railway.length_unit,
                    );
                    units += select(own && ordinal < built, length, 0);
                    next += select(own && ordinal == built, length, 0);
                }
                for index in 0..3u32 {
                    if good == material(index) {
                        let construction = times32(next, config.railway.materials[index as usize]);
                        let upkeep = times(
                            times32(units + next, config.railway.materials[index as usize + 3]),
                            config.coverage + config.ring,
                        )
                        .value;
                        wanted = plus(construction, upkeep).value;
                    }
                }
            }
        }
    }
    wanted
}

#[cube]
fn routes(state: &mut [u32], config: &mut EconomyConfig) {
    let origin = ABSOLUTE_POS as u32;
    let nodes = config.railway.nodes;
    if origin < nodes {
        let row = origin * nodes * ROUTE_WORDS as u32;
        for node in 0..nodes {
            let base = (row + node * ROUTE_WORDS as u32) as usize;
            config.railway.routes[base] = select(node == origin, 0, UNREACHABLE);
            config.railway.routes[base + 1] = u32::MAX;
            config.railway.routes[base + 2] = u32::MAX;
            config.railway.routes[base + 3] = 0;
            config.railway.routes[base + 4] = 0;
        }
        for _ in 0..nodes {
            let mut nearest = nodes;
            let mut shortest = u32::new(comptime!(i64::from(UNREACHABLE)));
            for node in 0..nodes {
                let base = (row + node * ROUTE_WORDS as u32) as usize;
                let improve =
                    config.railway.routes[base + 4] == 0 && config.railway.routes[base] < shortest;
                nearest = select(improve, node, nearest);
                shortest = select(improve, config.railway.routes[base], shortest);
            }
            if nearest < nodes {
                let current = (row + nearest * ROUTE_WORDS as u32) as usize;
                config.railway.routes[current + 4] = 1;
                for edge in 0..config.railway.segment_count {
                    let base = edge as usize * SEGMENT_WORDS;
                    let from = config.railway.segments[base];
                    let to = config.railway.segments[base + 1];
                    let company = config.railway.segments[base + 2];
                    let firm = config.railway.companies[company as usize * COMPANY_WORDS];
                    let operating = load(
                        state,
                        comptime!(Field::RailOperating as u32) * config.actors + firm,
                    )
                    .extract(0usize);
                    let connected = (from == nearest || to == nearest)
                        && config.railway.segments[base + 4] < operating;
                    let target = select(from == nearest, to, from);
                    let destination = (row + target * ROUTE_WORDS as u32) as usize;
                    let distance = shortest + config.railway.segments[base + 3];
                    let improve = connected && distance < config.railway.routes[destination];
                    if improve {
                        config.railway.routes[destination] = distance;
                        config.railway.routes[destination + 1] = edge;
                        config.railway.routes[destination + 2] = select(
                            nearest == origin,
                            company,
                            config.railway.routes[current + 2],
                        );
                    }
                }
            }
        }
        for node in 0..nodes {
            let base = (row + node * ROUTE_WORDS as u32) as usize;
            let carrier = config.railway.routes[base + 2];
            if carrier < config.railway.company_count {
                let mut fee = ceildiv(config.railway.routes[base], config.railway.length_unit)
                    * config.railway.companies[carrier as usize * COMPANY_WORDS + 1];
                let mut cursor = node;
                for _ in 0..nodes {
                    let prior =
                        config.railway.routes[(row + cursor * ROUTE_WORDS as u32 + 1) as usize];
                    if prior < config.railway.segment_count {
                        let edge = prior as usize * SEGMENT_WORDS;
                        let owner = config.railway.segments[edge + 2];
                        let toll = ceildiv(
                            config.railway.segments[edge + 3],
                            config.railway.length_unit,
                        ) * config.railway.companies[owner as usize * COMPANY_WORDS + 2];
                        let foreign = config.railway.companies[owner as usize * COMPANY_WORDS + 3]
                            != config.railway.companies[carrier as usize * COMPANY_WORDS + 3];
                        fee += select(foreign, toll, 0);
                        cursor = select(
                            config.railway.segments[edge] == cursor,
                            config.railway.segments[edge + 1],
                            config.railway.segments[edge],
                        );
                    }
                }
                config.railway.routes[base + 3] = fee;
            }
        }
    }
}

#[derive(CubeType, Clone, Copy)]
pub(super) struct FreightRoute {
    pub days: u32,
    pub fee: u32,
    pub origin: u32,
    pub destination: u32,
    pub carrier: u32,
}

#[cube]
fn station(actors: &[u32], config: &EconomyConfig, actor: u32) -> u32 {
    let stride = actors.len() / ActorLayout::WORDS;
    let x = actors[ActorLayout::X * stride + actor as usize];
    let y = actors[ActorLayout::Y * stride + actor as usize];
    let mut chosen = 0;
    let mut distance = u32::new(comptime!(i64::from(u32::MAX)));
    for node in 0..config.railway.nodes {
        let nx = config.railway.stations[node as usize * 2];
        let ny = config.railway.stations[node as usize * 2 + 1];
        let trip = max(x, nx) - min(x, nx) + max(y, ny) - min(y, ny);
        chosen = select(trip < distance, node, chosen);
        distance = min(distance, trip);
    }
    chosen
}

#[cube]
fn access_distance(actors: &[u32], config: &EconomyConfig, actor: u32, node: u32) -> u32 {
    let stride = actors.len() / ActorLayout::WORDS;
    let x = actors[ActorLayout::X * stride + actor as usize];
    let y = actors[ActorLayout::Y * stride + actor as usize];
    let nx = config.railway.stations[node as usize * 2];
    let ny = config.railway.stations[node as usize * 2 + 1];
    max(x, nx) - min(x, nx) + max(y, ny) - min(y, ny)
}

#[cube]
pub(super) fn animal_route(
    actors: &[u32],
    config: &EconomyConfig,
    buyer: u32,
    seller: u32,
) -> FreightRoute {
    FreightRoute {
        days: max(
            ceildiv(distance(actors, buyer, seller), config.shipment_speed),
            1,
        ),
        fee: 0,
        origin: 0,
        destination: 0,
        carrier: u32::MAX,
    }
}

#[cube]
pub(super) fn route(
    actors: &[u32],
    config: &EconomyConfig,
    buyer: u32,
    seller: u32,
) -> FreightRoute {
    let mut result = animal_route(actors, config, buyer, seller);
    if comptime!(config.railway.enabled == Feature::Enabled) && config.railway.nodes > 0 {
        let origin = station(actors, config, seller);
        let destination = station(actors, config, buyer);
        let base = ((origin * config.railway.nodes + destination) * ROUTE_WORDS as u32) as usize;
        let rail_distance = config.railway.routes[base];
        let carrier = config.railway.routes[base + 2];
        let access = access_distance(actors, config, seller, origin)
            + access_distance(actors, config, buyer, destination);
        let days = ceildiv(access, config.shipment_speed)
            + max(
                ceildiv(
                    rail_distance,
                    config.shipment_speed * config.railway.speed_multiplier,
                ),
                1,
            );
        let faster = carrier < config.railway.company_count && days < result.days;
        result.days = select(faster, days, result.days);
        result.fee = select(faster, config.railway.routes[base + 3], 0);
        result.origin = origin;
        result.destination = destination;
        result.carrier = select(faster, carrier, u32::MAX);
    }
    result
}

#[derive(Clone, Copy)]
enum RailAccount {
    Cash,
    Revenue,
}

#[cube]
fn receipt(
    config: &EconomyConfig,
    control: &[Atomic<u32>],
    company: u32,
    amount: Ticks,
    #[comptime] account: RailAccount,
) {
    let offset = comptime!(match account {
        RailAccount::Cash => 0,
        RailAccount::Revenue => 2,
    });
    let base = company as usize * 4 + offset;
    let low = amount.extract(LOW_WORD);
    let high = amount.extract(HIGH_WORD);
    let previous_low = config.railway.receipts[base].fetch_add(low);
    let carry = u32::cast_from(low > u32::MAX - previous_low);
    let previous_high = config.railway.receipts[base + 1].fetch_add(high);
    let before_carry = config.railway.receipts[base + 1].fetch_add(carry);
    if high > u32::MAX - previous_high || carry > u32::MAX - before_carry {
        KernelFaults::new(comptime!(FaultFlags::ARITHMETIC_OVERFLOW)).record(control);
    }
}

#[cube]
fn clear_fees(config: &EconomyConfig) {
    let company = ABSOLUTE_POS as u32;
    if company < config.railway.company_count {
        for word in 0..4usize {
            config.railway.receipts[company as usize * 4 + word].store(0);
        }
    }
}

#[cube]
fn pay_fees(state: &mut [u32], config: &EconomyConfig, control: &[Atomic<u32>]) {
    let company = ABSOLUTE_POS as u32;
    if company < config.railway.company_count {
        let firm = config.railway.companies[company as usize * COMPANY_WORDS];
        let base = company as usize * 4;
        let cash = ticks(
            config.railway.receipts[base].load(),
            config.railway.receipts[base + 1].load(),
        );
        let revenue = ticks(
            config.railway.receipts[base + 2].load(),
            config.railway.receipts[base + 3].load(),
        );
        increase(
            state,
            config,
            control,
            comptime!(Field::Cash as u32),
            firm,
            cash,
        );
        increase(
            state,
            config,
            control,
            comptime!(Field::RailRevenue as u32),
            firm,
            revenue,
        );
    }
}

#[cube]
fn fees(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    proposals: &[u32],
    scratch: &[u32],
    control: &[Atomic<u32>],
    sort_source: u32,
    #[comptime] stage: RailStage,
) {
    let index = ABSOLUTE_POS as u32;
    let count = match comptime!(stage) {
        RailStage::Fees => control[CONSTRUCTED_EVENTS_WORD].load(),
        _ => config.actors,
    };
    if index < count {
        let buyer = match comptime!(stage) {
            RailStage::Fees => scratch[(sort_source + config.actors + index) as usize],
            _ => index,
        };
        let escrow = value(state, config, comptime!(Field::FreightEscrow as u32), buyer);
        if !equal(escrow, zero()) {
            let seller = proposals[buyer as usize * ProposalLayout::WORDS + ProposalLayout::ACTOR];
            let path = route(actors, config, buyer, seller);
            let quantity = divide(escrow, path.fee);
            receipt(
                config,
                control,
                path.carrier,
                escrow,
                comptime!(RailAccount::Revenue),
            );
            let mut retained = escrow;
            let mut cursor = path.destination;
            for _ in 0..config.railway.nodes {
                let edge = config.railway.routes[((path.origin * config.railway.nodes + cursor)
                    * ROUTE_WORDS as u32
                    + 1) as usize];
                if edge < config.railway.segment_count {
                    let base = edge as usize * SEGMENT_WORDS;
                    let owner = config.railway.segments[base + 2];
                    let toll_rate = ceildiv(
                        config.railway.segments[base + 3],
                        config.railway.length_unit,
                    ) * config.railway.companies
                        [owner as usize * COMPANY_WORDS + 2];
                    let foreign = config.railway.companies[owner as usize * COMPANY_WORDS + 3]
                        != config.railway.companies[path.carrier as usize * COMPANY_WORDS + 3];
                    let toll = multiply(control, quantity, select(foreign, toll_rate, 0));
                    retained = minus(retained, toll);
                    receipt(config, control, owner, toll, comptime!(RailAccount::Cash));
                    receipt(
                        config,
                        control,
                        owner,
                        toll,
                        comptime!(RailAccount::Revenue),
                    );
                    cursor = select(
                        config.railway.segments[base] == cursor,
                        config.railway.segments[base + 1],
                        config.railway.segments[base],
                    );
                }
            }
            receipt(
                config,
                control,
                path.carrier,
                retained,
                comptime!(RailAccount::Cash),
            );
            set32(
                state,
                config,
                comptime!(Field::FreightEscrow as u32),
                buyer,
                0,
            );
        }
    }
}

use super::*;
use crate::domain::railway::RailCapacityField;
use crate::simulation::{Device, configuration::Configuration};
use burn_wgpu::CubeTensor;

const COMPANY_WORDS: usize = 4;
const SEGMENT_WORDS: usize = 7;
const ROUTE_WORDS: usize = 6;
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
    pub regions: Box<[u32]>,
    pub segment_regions: Box<[u32]>,
    pub requests: Box<[Atomic<u32>]>,
    pub access_requests: Box<[Atomic<u32>]>,
    pub region_count: u32,
    pub region_base: u32,
    pub region_blocks: u32,
    pub initial_actors: u32,
    pub units_per_level: u32,
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
    regions: CubeTensor,
    segment_regions: CubeTensor,
    requests: CubeTensor,
    access_requests: CubeTensor,
    region_count: u32,
    region_base: u32,
    region_blocks: u32,
    initial_actors: u32,
    units_per_level: u32,
}

impl Network {
    pub fn new(
        device: &Device,
        scenario: &crate::domain::Scenario,
        state: &crate::domain::State,
    ) -> Self {
        let config = &scenario.config().railway;
        let geography = scenario.rail_geography();
        let mut segment_regions = Vec::new();
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
            .enumerate()
            .flat_map(|(index, segment)| {
                let company = config
                    .companies
                    .iter()
                    .position(|company| company.firm == segment.company)
                    .expect("railway company");
                let ordinal = ordinals[company];
                ordinals[company] += 1;
                let offset = segment_regions.len();
                segment_regions.extend(
                    geography.segments[index]
                        .iter()
                        .map(|region| *region as u32),
                );
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
                    offset as u32,
                    geography.segments[index].len() as u32,
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
            routes: device.empty(
                (config.companies.len() * stations.len() * stations.len() * ROUTE_WORDS).max(1),
            ),
            regions: device.tensor(
                &geography
                    .regions
                    .iter()
                    .flat_map(|region| [region.company as u32, region.region as u32])
                    .collect::<Vec<_>>(),
            ),
            segment_regions: device.tensor(&segment_regions),
            requests: device.empty(geography.regions.len().max(1)),
            access_requests: device.empty(config.companies.len().max(1)),
            region_count: geography.regions.len() as u32,
            region_base: (BASE_FIELDS + scenario.ring() * Good::COUNT) as u32,
            region_blocks: geography.blocks(scenario.actors()) as u32,
            initial_actors: scenario.actors() as u32,
            units_per_level: config.units_per_level,
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
            regions: self.regions.clone().into_buffer_arg(),
            segment_regions: self.segment_regions.clone().into_buffer_arg(),
            requests: self.requests.clone().into_buffer_arg(),
            access_requests: self.access_requests.clone().into_buffer_arg(),
            region_count: self.region_count,
            region_base: self.region_base,
            region_blocks: self.region_blocks,
            initial_actors: self.initial_actors,
            units_per_level: self.units_per_level,
        }
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum RailStage {
    Infrastructure,
    Routes,
    Close,
}

#[derive(CubeType, CubeLaunch)]
struct RailInput {
    state: Box<[u32]>,
    actors: Box<[u32]>,
    config: EconomyConfig,
    control: Box<[Atomic<u32>]>,
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
            RailStage::Infrastructure => infrastructure(
                &mut input.state,
                &input.actors,
                &input.config,
                &input.control,
            ),
            RailStage::Routes => routes(&mut input.state, &mut input.config),
            RailStage::Close => close(&mut input.state, &input.config, &input.control),
        }
    }
}

pub fn launch(
    device: &Device,
    state: &CubeTensor,
    actors: &CubeTensor,
    config: &Configuration,
    control: &CubeTensor,
    stage: RailStage,
) {
    if config.railway.company_count > 0 {
        let count = match stage {
            RailStage::Infrastructure | RailStage::Close => config.railway.company_count,
            RailStage::Routes => config.railway.nodes * config.railway.company_count,
        };
        RailSystem::launch(
            device,
            device.count(count as usize),
            CubeDim::new_1d(device.workgroup),
            RailInputLaunch {
                state: state.clone().into_buffer_arg(),
                actors: actors.clone().into_buffer_arg(),
                config: config.launch(),
                control: control.clone().into_buffer_arg(),
                stage,
            },
        );
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
fn infrastructure(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
) {
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
        for region in 0..config.railway.region_count {
            if config.railway.regions[region as usize * 2] == company {
                let level =
                    regional_number(state, config, region, comptime!(RailCapacityField::Level));
                units += select(level > 0, max(level, 1) - 1, 0);
            }
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
        let expired = value(state, config, comptime!(Field::Railway as u32), firm);
        record_use(state, config, control, firm, expired);
        set32(state, config, comptime!(Field::Railway as u32), firm, 0);
        let expanding = enterprises::expands(state, actors, config, firm);
        let mut capacity = zero();
        let mut production = zero();
        for region in 0..config.railway.region_count {
            if config.railway.regions[region as usize * 2] == company {
                let mut connected = false;
                for edge in 0..config.railway.segment_count {
                    let base = edge as usize * SEGMENT_WORDS;
                    if config.railway.segments[base + 2] == company
                        && config.railway.segments[base + 4] < built
                    {
                        let start = config.railway.segments[base + 5];
                        let length = config.railway.segments[base + 6];
                        for index in 0..length {
                            connected = connected
                                || config.railway.segment_regions[(start + index) as usize]
                                    == region;
                        }
                    }
                }
                let mut level =
                    regional_number(state, config, region, comptime!(RailCapacityField::Level));
                level = max(level, u32::cast_from(connected));
                let demand =
                    regional_value(state, config, region, comptime!(RailCapacityField::Demand));
                let upgrade = maintained
                    && expanding
                    && level > 0
                    && level < 1000
                    && lt(times32(level, config.railway.units_per_level), demand)
                    && supplied(state, config, firm, 1, 0);
                if upgrade {
                    consume(state, config, control, firm, 1, 0);
                    level += 1;
                }
                let units = level * config.railway.units_per_level;
                let supply = select(maintained, units, 0);
                regional_set(
                    state,
                    config,
                    region,
                    ticks(level, 0),
                    comptime!(RailCapacityField::Level),
                );
                regional_set(
                    state,
                    config,
                    region,
                    ticks(supply, 0),
                    comptime!(RailCapacityField::Available),
                );
                capacity = add_checked(control, capacity, ticks(units, 0));
                production = add_checked(control, production, ticks(supply, 0));
            }
        }
        set(
            state,
            config,
            comptime!(Field::RailCapacity as u32),
            firm,
            capacity,
        );
        set(
            state,
            config,
            comptime!(Field::Railway as u32),
            firm,
            production,
        );
        increase(
            state,
            config,
            control,
            comptime!(Field::Produced as u32),
            firm,
            production,
        );
        increase(
            state,
            config,
            control,
            comptime!(Field::UnscaledProduced as u32),
            firm,
            production,
        );
        set32(state, config, comptime!(Field::RailRevenue as u32), firm, 0);
    }
}

#[cube]
pub(super) fn demand(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    firm: u32,
    good: u32,
) -> Ticks {
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
                let expanding = enterprises::expands(state, actors, config, firm);
                for region in 0..config.railway.region_count {
                    if config.railway.regions[region as usize * 2] == company {
                        let level = regional_number(
                            state,
                            config,
                            region,
                            comptime!(RailCapacityField::Level),
                        );
                        let demand = regional_value(
                            state,
                            config,
                            region,
                            comptime!(RailCapacityField::Demand),
                        );
                        units += max(level, 1) - 1;
                        let upgrade = expanding
                            && level > 0
                            && level < 1000
                            && lt(times32(level, config.railway.units_per_level), demand);
                        next += u32::cast_from(upgrade);
                    }
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
    let invocation = ABSOLUTE_POS as u32;
    let nodes = config.railway.nodes;
    if invocation < nodes * config.railway.company_count {
        let origin = invocation % nodes;
        let carrier = invocation / nodes;
        let row = invocation * nodes * ROUTE_WORDS as u32;
        for node in 0..nodes {
            let base = (row + node * ROUTE_WORDS as u32) as usize;
            config.railway.routes[base] = select(node == origin, 0, UNREACHABLE);
            config.railway.routes[base + 1] = u32::MAX;
            config.railway.routes[base + 2] = u32::MAX;
            config.railway.routes[base + 3] = 0;
            config.railway.routes[base + 4] = 0;
            config.railway.routes[base + 5] = select(node == origin, 0, UNREACHABLE);
        }
        for _ in 0..nodes {
            let mut nearest = nodes;
            let mut shortest = u32::new(comptime!(i64::from(UNREACHABLE)));
            let mut least_foreign = u32::new(comptime!(i64::from(UNREACHABLE)));
            for node in 0..nodes {
                let base = (row + node * ROUTE_WORDS as u32) as usize;
                let foreign = config.railway.routes[base + 5];
                let distance = config.railway.routes[base];
                let improve = config.railway.routes[base + 4] == 0
                    && (foreign < least_foreign || foreign == least_foreign && distance < shortest);
                nearest = select(improve, node, nearest);
                shortest = select(improve, distance, shortest);
                least_foreign = select(improve, foreign, least_foreign);
            }
            if nearest < nodes {
                config.railway.routes[(row + nearest * ROUTE_WORDS as u32 + 4) as usize] = 1;
                for edge in 0..config.railway.segment_count {
                    let base = edge as usize * SEGMENT_WORDS;
                    let from = config.railway.segments[base];
                    let to = config.railway.segments[base + 1];
                    let owner = config.railway.segments[base + 2];
                    let firm = config.railway.companies[owner as usize * COMPANY_WORDS];
                    let operating = load(
                        state,
                        comptime!(Field::RailOperating as u32) * config.actors + firm,
                    )
                    .extract(0usize);
                    let connected = (from == nearest || to == nearest)
                        && config.railway.segments[base + 4] < operating
                        && (nearest != origin || owner == carrier);
                    let target = select(from == nearest, to, from);
                    let destination = (row + target * ROUTE_WORDS as u32) as usize;
                    let distance = shortest + config.railway.segments[base + 3];
                    let foreign_owner = config.railway.companies
                        [owner as usize * COMPANY_WORDS + 3]
                        != config.railway.companies[carrier as usize * COMPANY_WORDS + 3];
                    let foreign =
                        least_foreign + select(foreign_owner, config.railway.segments[base + 3], 0);
                    let improve = connected
                        && (foreign < config.railway.routes[destination + 5]
                            || foreign == config.railway.routes[destination + 5]
                                && distance < config.railway.routes[destination]);
                    if improve {
                        config.railway.routes[destination] = distance;
                        config.railway.routes[destination + 1] = edge;
                        config.railway.routes[destination + 2] = carrier;
                        config.railway.routes[destination + 5] = foreign;
                    }
                }
            }
        }
        for node in 0..nodes {
            let base = (row + node * ROUTE_WORDS as u32) as usize;
            if config.railway.routes[base + 2] < config.railway.company_count {
                let fee = traverse(
                    &*config,
                    &FreightRoute {
                        days: 0,
                        fee: 0,
                        origin,
                        destination: node,
                        carrier,
                        foreign_distance: 0,
                    },
                    comptime!(Traversal::Quote),
                );
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
    pub foreign_distance: u32,
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
        foreign_distance: u32::MAX,
    }
}

#[cube]
impl FreightRoute {
    pub fn replace(&mut self, path: &FreightRoute) {
        self.days = path.days;
        self.fee = path.fee;
        self.origin = path.origin;
        self.destination = path.destination;
        self.carrier = path.carrier;
        self.foreign_distance = path.foreign_distance;
    }

    pub fn preferred_to(&self, other: &FreightRoute) -> bool {
        self.foreign_distance < other.foreign_distance
            || self.foreign_distance == other.foreign_distance && self.fee < other.fee
            || self.foreign_distance == other.foreign_distance
                && self.fee == other.fee
                && self.days < other.days
    }
}

#[cube]
fn candidate(
    actors: &[u32],
    config: &EconomyConfig,
    buyer: u32,
    seller: u32,
    company: u32,
) -> FreightRoute {
    let mut result = animal_route(actors, config, buyer, seller);
    if comptime!(config.railway.enabled == Feature::Enabled)
        && config.railway.nodes > 0
        && company < config.railway.company_count
    {
        let origin = station(actors, config, seller);
        let destination = station(actors, config, buyer);
        let base = (((company * config.railway.nodes + origin) * config.railway.nodes
            + destination)
            * ROUTE_WORDS as u32) as usize;
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
        result.foreign_distance = select(faster, config.railway.routes[base + 5], u32::MAX);
    }
    result
}

#[cube]
pub(super) fn route(
    actors: &[u32],
    config: &EconomyConfig,
    buyer: u32,
    seller: u32,
) -> FreightRoute {
    let mut result = animal_route(actors, config, buyer, seller);
    if comptime!(config.railway.enabled == Feature::Enabled) {
        for company in 0..config.railway.company_count {
            let path = candidate(actors, config, buyer, seller, company);
            if path.preferred_to(&result) {
                result.replace(&path);
            }
        }
    }
    result
}

#[cube]
fn regional_cell(config: &EconomyConfig, region: u32, #[comptime] field: RailCapacityField) -> u32 {
    let row = config.railway.region_base
        + comptime!(field as u32) * config.railway.region_blocks
        + region / config.railway.initial_actors;
    row * config.actors + region % config.railway.initial_actors
}

#[cube]
fn regional_value(
    state: &mut [u32],
    config: &EconomyConfig,
    region: u32,
    #[comptime] field: RailCapacityField,
) -> Ticks {
    load(state, regional_cell(config, region, comptime!(field)))
}

#[cube]
fn regional_number(
    state: &mut [u32],
    config: &EconomyConfig,
    region: u32,
    #[comptime] field: RailCapacityField,
) -> u32 {
    regional_value(state, config, region, comptime!(field)).extract(LOW_WORD)
}

#[cube]
fn regional_set(
    state: &mut [u32],
    config: &EconomyConfig,
    region: u32,
    amount: Ticks,
    #[comptime] field: RailCapacityField,
) {
    store(
        state,
        regional_cell(config, region, comptime!(field)),
        amount,
    );
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum Traversal {
    Quote,
    Request,
}

#[cube]
fn access_fare(
    config: &EconomyConfig,
    owner: u32,
    carrier: u32,
    regions: u32,
    #[comptime] traversal: Traversal,
) -> u32 {
    let mut fee = 0;
    if owner < config.railway.company_count {
        let foreign = config.railway.companies[owner as usize * COMPANY_WORDS + 3]
            != config.railway.companies[carrier as usize * COMPANY_WORDS + 3];
        let units = select(foreign, max(regions, 1), 0);
        fee = units * config.railway.companies[owner as usize * COMPANY_WORDS + 2];
        if comptime!(traversal == Traversal::Request) {
            config.railway.access_requests[owner as usize].fetch_add(units);
        }
    }
    fee
}

#[cube]
fn traverse(config: &EconomyConfig, path: &FreightRoute, #[comptime] traversal: Traversal) -> u32 {
    let mut cursor = path.destination;
    let mut previous = u32::new(comptime!(i64::from(u32::MAX)));
    let mut first = u32::new(comptime!(i64::from(u32::MAX)));
    let mut pending = u32::new(comptime!(i64::from(u32::MAX)));
    let mut fee = 0u32;
    let mut toll = 0u32;
    let mut access_owner = u32::new(comptime!(i64::from(u32::MAX)));
    let mut access_region = u32::new(comptime!(i64::from(u32::MAX)));
    let mut access_units = 0u32;
    let rate = config.railway.companies[path.carrier as usize * COMPANY_WORDS + 1];
    let mut steps = 0u32;
    while cursor != path.origin && steps < config.railway.nodes {
        let edge = config.railway.routes[(((path.carrier * config.railway.nodes + path.origin)
            * config.railway.nodes
            + cursor)
            * ROUTE_WORDS as u32
            + 1) as usize];
        if edge < config.railway.segment_count {
            let base = edge as usize * SEGMENT_WORDS;
            let owner = config.railway.segments[base + 2];
            if owner != access_owner {
                toll += access_fare(
                    config,
                    access_owner,
                    path.carrier,
                    access_units,
                    comptime!(traversal),
                );
                access_owner = owner;
                access_region = u32::MAX;
                access_units = 0;
            }
            let reverse = config.railway.segments[base + 1] == cursor;
            let next = select(
                reverse,
                config.railway.segments[base],
                config.railway.segments[base + 1],
            );
            let start = config.railway.segments[base + 5];
            let count = config.railway.segments[base + 6];
            for index in 0..count {
                let position = select(reverse, count - index - 1, index);
                let record = config.railway.segment_regions[(start + position) as usize];
                let region = config.railway.regions[record as usize * 2 + 1];
                access_units +=
                    u32::cast_from(access_region != u32::MAX && access_region != region);
                access_region = region;
                first = select(first == u32::MAX, record, first);
                let entered = region != previous && pending < config.railway.region_count;
                if entered {
                    fee += rate;
                    if comptime!(traversal == Traversal::Request) {
                        config.railway.requests[pending as usize].fetch_add(1);
                    }
                }
                pending = record;
                previous = region;
            }
            cursor = next;
        }
        steps += 1;
    }
    if fee == 0 && first < config.railway.region_count {
        fee = rate;
        if comptime!(traversal == Traversal::Request) {
            config.railway.requests[first as usize].fetch_add(1);
        }
    }
    fee + toll
        + access_fare(
            config,
            access_owner,
            path.carrier,
            access_units,
            comptime!(traversal),
        )
}

#[cube]
fn record_use(
    state: &mut [u32],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
    actor: u32,
    units: Ticks,
) {
    increase(
        state,
        config,
        control,
        USED_FIELDS as u32 + comptime!(Good::Railway as u32) - 1,
        actor,
        units,
    );
    increase(
        state,
        config,
        control,
        comptime!(Field::Used as u32),
        actor,
        units,
    );
}

#[cube]
fn pay(
    state: &mut [u32],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
    company: u32,
    amount: Ticks,
) {
    let firm = config.railway.companies[company as usize * COMPANY_WORDS];
    increase(
        state,
        config,
        control,
        comptime!(Field::Cash as u32),
        firm,
        amount,
    );
    increase(
        state,
        config,
        control,
        comptime!(Field::RailRevenue as u32),
        firm,
        amount,
    );
}

#[cube]
fn purchase(
    state: &mut [u32],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
    sender: u32,
    path: &FreightRoute,
    quantity: Ticks,
) {
    let mut total = zero();
    for record in 0..config.railway.region_count {
        let visits = config.railway.requests[record as usize].load();
        if visits > 0 {
            let units = multiply(control, quantity, visits);
            let stock = regional_value(
                state,
                config,
                record,
                comptime!(RailCapacityField::Available),
            );
            regional_set(
                state,
                config,
                record,
                minus(stock, units),
                comptime!(RailCapacityField::Available),
            );
            let owner = config.railway.regions[record as usize * 2];
            let firm = config.railway.companies[owner as usize * COMPANY_WORDS];
            let inventory = value(state, config, comptime!(Field::Railway as u32), firm);
            set(
                state,
                config,
                comptime!(Field::Railway as u32),
                firm,
                minus(inventory, units),
            );
            increase(
                state,
                config,
                control,
                comptime!(Field::Sales as u32),
                firm,
                units,
            );
            total = add_checked(control, total, units);
        }
    }
    for owner in 0..config.railway.company_count {
        let accesses = config.railway.access_requests[owner as usize].load();
        let units = multiply(control, quantity, accesses);
        let toll = multiply(
            control,
            units,
            config.railway.companies[owner as usize * COMPANY_WORDS + 2],
        );
        pay(state, config, control, owner, toll);
    }
    let freight = multiply(
        control,
        total,
        config.railway.companies[path.carrier as usize * COMPANY_WORDS + 1],
    );
    pay(state, config, control, path.carrier, freight);
    record_use(state, config, control, sender, total);
}

#[cube]
fn prepare_request(config: &EconomyConfig, path: &FreightRoute) {
    for record in 0..config.railway.region_count {
        config.railway.requests[record as usize].store(0);
    }
    for company in 0..config.railway.company_count {
        config.railway.access_requests[company as usize].store(0);
    }
    traverse(config, path, comptime!(Traversal::Request));
}

#[derive(Clone, Copy)]
enum DemandSignal {
    Record,
    Ignore,
}

#[cube]
fn capacity(
    state: &mut [u32],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
    quantity: Ticks,
    #[comptime] signal: DemandSignal,
) -> bool {
    let mut available = true;
    for record in 0..config.railway.region_count {
        let visits = config.railway.requests[record as usize].load();
        if visits > 0 {
            let required = multiply(control, quantity, visits);
            let stock = regional_value(
                state,
                config,
                record,
                comptime!(RailCapacityField::Available),
            );
            if comptime!(matches!(signal, DemandSignal::Record)) {
                let level =
                    regional_number(state, config, record, comptime!(RailCapacityField::Level));
                let used = minus(times32(level, config.railway.units_per_level), stock);
                let wanted = maximum(
                    regional_value(state, config, record, comptime!(RailCapacityField::Demand)),
                    add_checked(control, used, required),
                );
                regional_set(
                    state,
                    config,
                    record,
                    wanted,
                    comptime!(RailCapacityField::Demand),
                );
            }
            available = available && !lt(stock, required);
        }
    }
    available
}

#[cube]
pub(super) fn dispatch(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    proposals: &mut [u32],
    scratch: &mut [u32],
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
    id: u32,
) {
    if id == 0 {
        let count = parameters.count;
        let active = markets::active(control, parameters);
        let mut event = Array::<u32>::new(EVENT_WORDS);
        for index in 0..active {
            let id = scratch[(parameters.sort_source + count + index) as usize];
            let buyer = id + parameters.offset;
            let seller = proposals[buyer as usize * ProposalLayout::WORDS + ProposalLayout::ACTOR];
            let quantity = ticks(
                proposals[buyer as usize * ProposalLayout::WORDS + ProposalLayout::QUANTITY_LOW],
                proposals[buyer as usize * ProposalLayout::WORDS + ProposalLayout::QUANTITY_HIGH],
            );
            let mut selected = animal_route(actors, config, buyer, seller);
            construct_route(
                state,
                actors,
                config,
                proposals,
                &mut event[..],
                control,
                parameters,
                &selected,
                id,
                comptime!(EventStorage::Local),
            );
            let eligible = scheduler::settle(
                state,
                &event[..],
                control,
                0,
                0,
                1,
                comptime!(Posting::CheckAll),
            );
            if eligible.contains(comptime!(EventStatus::Accepted)) {
                let preferred = route(actors, config, buyer, seller);
                if preferred.carrier < config.railway.company_count {
                    prepare_request(config, &preferred);
                    capacity(
                        state,
                        config,
                        control,
                        quantity,
                        comptime!(DemandSignal::Record),
                    );
                }
                for company in 0..config.railway.company_count {
                    let path = candidate(actors, config, buyer, seller, company);
                    let funded = !lt(
                        value(state, config, comptime!(Field::Cash as u32), seller),
                        multiply(control, quantity, path.fee),
                    );
                    if path.carrier < config.railway.company_count
                        && path.preferred_to(&selected)
                        && funded
                    {
                        prepare_request(config, &path);
                        if capacity(
                            state,
                            config,
                            control,
                            quantity,
                            comptime!(DemandSignal::Ignore),
                        ) {
                            selected.replace(&path);
                        }
                    }
                }
            }
            if selected.carrier < config.railway.company_count {
                prepare_request(config, &selected);
                capacity(
                    state,
                    config,
                    control,
                    quantity,
                    comptime!(DemandSignal::Record),
                );
                construct_route(
                    state,
                    actors,
                    config,
                    proposals,
                    &mut event[..],
                    control,
                    parameters,
                    &selected,
                    id,
                    comptime!(EventStorage::Local),
                );
            }
            let status = apply(state, &event[..], control, 0);
            if selected.carrier < config.railway.company_count
                && status.contains(comptime!(EventStatus::Accepted))
            {
                purchase(state, config, control, seller, &selected, quantity);
            }
            status.store(scratch, 4 * count + id);
        }
        control[PROGRESS_WORD].fetch_max(active);
        control[COMPLETED_WORD].fetch_add(active);
    }
}

#[cube]
fn close(state: &mut [u32], config: &EconomyConfig, control: &[Atomic<u32>]) {
    let company = ABSOLUTE_POS as u32;
    if company < config.railway.company_count {
        let firm = config.railway.companies[company as usize * COMPANY_WORDS];
        let status = number(
            state,
            config,
            comptime!(Field::EnterpriseStatus as u32),
            firm,
        );
        if status >= comptime!(crate::domain::enterprise::EnterpriseState::Liquidating as u32) {
            let expired = value(state, config, comptime!(Field::Railway as u32), firm);
            record_use(state, config, control, firm, expired);
            set32(state, config, comptime!(Field::Railway as u32), firm, 0);
            set32(
                state,
                config,
                comptime!(Field::RailOperating as u32),
                firm,
                0,
            );
            for region in 0..config.railway.region_count {
                if config.railway.regions[region as usize * 2] == company {
                    regional_set(
                        state,
                        config,
                        region,
                        zero(),
                        comptime!(RailCapacityField::Available),
                    );
                }
            }
        }
    }
}

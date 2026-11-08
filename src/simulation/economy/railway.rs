use super::*;
use crate::domain::railway::RailCapacityField;
use crate::simulation::{Device, configuration::Configuration};
use burn_wgpu::CubeTensor;

#[cfg(test)]
mod tests;

const COMPANY_WORDS: usize = 4;
const SEGMENT_WORDS: usize = 7;
const ROUTE_WORDS: usize = 6;
const CAPACITY_WORDS: usize = 5;
const SELECTION_WORDS: usize = 4;
const REJECTION_WORDS: usize = 7;
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
    pub access: Box<[u32]>,
    pub settlements: Box<[Atomic<u32>]>,
    pub companies: Box<[u32]>,
    pub station_companies: Box<[u32]>,
    pub segments: Box<[u32]>,
    pub materials: Box<[u32]>,
    pub routes: Box<[u32]>,
    pub quotes: Box<[u32]>,
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
    access: CubeTensor,
    settlements: CubeTensor,
    companies: CubeTensor,
    station_companies: CubeTensor,
    segments: CubeTensor,
    materials: CubeTensor,
    routes: CubeTensor,
    quotes: CubeTensor,
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
        let sites = state
            .actors
            .firm_indices()
            .map(|index| (state.actors.at(index).identity, index as u32))
            .collect::<std::collections::BTreeMap<_, _>>();
        let companies = config
            .companies
            .iter()
            .flat_map(|company| {
                [
                    *sites.get(&company.firm).expect("railway firm"),
                    company.freight_rate,
                    company.access_fee,
                    *sites
                        .get(&state.company_identity(company.firm))
                        .expect("company founding site"),
                ]
            })
            .collect::<Vec<_>>();
        let mut station_companies = vec![std::collections::BTreeSet::new(); stations.len()];
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
                let from = stations
                    .iter()
                    .position(|position| *position == segment.from)
                    .expect("station");
                let to = stations
                    .iter()
                    .position(|position| *position == segment.to)
                    .expect("station");
                station_companies[from].insert(company as u32);
                station_companies[to].insert(company as u32);
                [
                    from as u32,
                    to as u32,
                    company as u32,
                    segment.from.distance(segment.to),
                    ordinal,
                    offset as u32,
                    geography.segments[index].len() as u32,
                ]
            })
            .collect::<Vec<_>>();
        let mut offsets = vec![stations.len() as u32 + 1];
        let mut carriers = Vec::new();
        for companies in station_companies {
            carriers.extend(companies);
            offsets.push((stations.len() + 1 + carriers.len()) as u32);
        }
        offsets.extend(carriers);
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
            access: device.empty(state.actors.len() * 2),
            settlements: device.tensor(&vec![0; state.actors.len() * (REJECTION_WORDS + 1) + 1]),
            companies: device.tensor(&companies),
            station_companies: device.tensor(&offsets),
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
            quotes: device.empty((stations.len() * stations.len() * scenario.ring() * 2).max(1)),
            regions: device.tensor(
                &geography
                    .regions
                    .iter()
                    .flat_map(|region| [region.company as u32, region.region as u32])
                    .collect::<Vec<_>>(),
            ),
            segment_regions: device.tensor(&segment_regions),
            requests: device.tensor(&vec![
                0;
                2 * geography.regions.len()
                    + 2
                    + config.companies.len()
                        * stations.len()
                        * stations.len()
                        * CAPACITY_WORDS
                    + stations.len()
                        * stations.len()
                        * scenario.ring()
                        * SELECTION_WORDS
            ]),
            access_requests: device.tensor(&vec![0; 2 * config.companies.len() + 1]),
            region_count: geography.regions.len() as u32,
            region_base: (BASE_FIELDS + scenario.ring() * Good::COUNT) as u32,
            region_blocks: geography.blocks(scenario.actors()) as u32,
            initial_actors: scenario.actors() as u32,
            units_per_level: config.units_per_level,
        }
    }

    #[cfg(test)]
    pub fn serial_settlements(&self, device: &Device) {
        let mut settlements = vec![0; self.settlements.meta.num_elements()];
        settlements[self.access.meta.num_elements() / 2] = Settlement::Serial as u32;
        device.write(&self.settlements, &settlements);
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
            access: self.access.clone().into_buffer_arg(),
            settlements: self.settlements.clone().into_buffer_arg(),
            companies: self.companies.clone().into_buffer_arg(),
            station_companies: self.station_companies.clone().into_buffer_arg(),
            segments: self.segments.clone().into_buffer_arg(),
            materials: self.materials.clone().into_buffer_arg(),
            routes: self.routes.clone().into_buffer_arg(),
            quotes: self.quotes.clone().into_buffer_arg(),
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
    Access,
    Routes,
    Quotes,
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
            RailStage::Access => access(&input.actors, &mut input.config),
            RailStage::Routes => routes(&mut input.state, &mut input.config),
            RailStage::Quotes => quotes(&mut input.config),
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
            RailStage::Access => config.actors,
            RailStage::Routes => config.railway.nodes * config.railway.company_count,
            RailStage::Quotes => config.railway.nodes * config.railway.nodes * config.ring,
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
    if comptime!(config.railway.enabled == Feature::Enabled)
        && kind(actors, firm).bits() == comptime!(Good::Railway as u32)
    {
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
            let cache = 1 + (invocation * nodes + node) as usize * CAPACITY_WORDS;
            config.railway.requests[(2 * config.railway.region_count + 1) as usize + (cache)]
                .store(0);
            let base = (row + node * ROUTE_WORDS as u32) as usize;
            config.railway.routes[base] = select(node == origin, 0, UNREACHABLE);
            config.railway.routes[base + 1] = u32::MAX;
            config.railway.routes[base + 2] = u32::MAX;
            config.railway.routes[base + 3] = 0;
            config.railway.routes[base + 4] = 0;
            config.railway.routes[base + 5] = select(node == origin, 0, UNREACHABLE);
        }
        let mut remaining = nodes;
        while remaining > 0 {
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
            remaining = select(nearest < nodes, remaining - 1, 0);
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
fn access(actors: &[u32], config: &mut EconomyConfig) {
    let actor = ABSOLUTE_POS as u32;
    if actor == 0 {
        config.railway.requests[(2 * config.railway.region_count + 1) as usize].store(1);
    }
    if actor < config.actors {
        let node = station(actors, &*config, actor);
        let distance = access_distance(actors, &*config, actor, node);
        config.railway.access[actor as usize * 2] = node;
        config.railway.access[actor as usize * 2 + 1] = distance;
    }
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
        let origin = config.railway.access[seller as usize * 2];
        let destination = config.railway.access[buyer as usize * 2];
        let base = (((company * config.railway.nodes + origin) * config.railway.nodes
            + destination)
            * ROUTE_WORDS as u32) as usize;
        let rail_distance = config.railway.routes[base];
        let carrier = config.railway.routes[base + 2];
        let access = config.railway.access[seller as usize * 2 + 1]
            + config.railway.access[buyer as usize * 2 + 1];
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
fn quotes(config: &mut EconomyConfig) {
    let index = ABSOLUTE_POS as u32;
    let nodes = config.railway.nodes;
    if index < nodes * nodes * config.ring {
        let limit = index % config.ring;
        let pair = index / config.ring;
        let mut selected = FreightRoute {
            days: u32::MAX,
            fee: u32::MAX,
            origin: pair / nodes,
            destination: pair % nodes,
            carrier: u32::MAX,
            foreign_distance: u32::MAX,
        };
        let start = config.railway.station_companies[selected.origin as usize];
        let end = config.railway.station_companies[selected.origin as usize + 1];
        let mut maximum_fee = 0;
        for entry in start..end {
            let company = config.railway.station_companies[entry as usize];
            let base = ((company * nodes * nodes + pair) * ROUTE_WORDS as u32) as usize;
            let path = FreightRoute {
                days: max(
                    ceildiv(
                        config.railway.routes[base],
                        config.shipment_speed * config.railway.speed_multiplier,
                    ),
                    1,
                ),
                fee: config.railway.routes[base + 3],
                origin: selected.origin,
                destination: selected.destination,
                carrier: config.railway.routes[base + 2],
                foreign_distance: config.railway.routes[base + 5],
            };
            if path.carrier < config.railway.company_count && path.days <= limit {
                maximum_fee = max(maximum_fee, path.fee);
                if path.preferred_to(&selected) {
                    selected.replace(&path);
                }
            }
        }
        config.railway.quotes[index as usize] = selected.carrier;
        config.railway.quotes[(nodes * nodes * config.ring + index) as usize] = maximum_fee;
        let cache = selection_cache(&*config, index);
        config.railway.requests[cache].store(0);
    }
}

#[cube]
fn selection_cache(config: &EconomyConfig, quote: u32) -> usize {
    (2 * config.railway.region_count + 2) as usize
        + (config.railway.company_count * config.railway.nodes * config.railway.nodes) as usize
            * CAPACITY_WORDS
        + quote as usize * SELECTION_WORDS
}

#[cube]
fn available_route(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
    buyer: u32,
    seller: u32,
    quantity: Ticks,
    preferred: &FreightRoute,
) -> FreightRoute {
    let mut selected = animal_route(actors, config, buyer, seller);
    let access = ceildiv(
        config.railway.access[seller as usize * 2 + 1]
            + config.railway.access[buyer as usize * 2 + 1],
        config.shipment_speed,
    );
    let limit = min(max(selected.days, access + 1) - access - 1, config.ring - 1);
    let quote =
        (preferred.origin * config.railway.nodes + preferred.destination) * config.ring + limit;
    let maximum_fee = config.railway.quotes
        [(config.railway.nodes * config.railway.nodes * config.ring + quote) as usize];
    let cash = value(state, config, comptime!(Field::Cash as u32), seller);
    let funded = !lt(cash, multiply(control, quantity, maximum_fee));
    let cache = selection_cache(config, quote);
    let epoch = config.railway.requests[(2 * config.railway.region_count + 1) as usize].load();
    let recorded = config.railway.requests[cache].load();
    let prior_quantity = ticks(
        config.railway.requests[cache + 1].load(),
        config.railway.requests[cache + 2].load(),
    );
    let prior_company = config.railway.requests[cache + 3].load();
    let exhausted = recorded > 0
        && prior_company >= config.railway.company_count
        && !lt(quantity, prior_quantity);
    let current = recorded == epoch && equal(quantity, prior_quantity);
    if funded && (current || exhausted) {
        let path = candidate(actors, config, buyer, seller, prior_company);
        selected.replace(&path);
    } else {
        let start = config.railway.station_companies[preferred.origin as usize];
        let end = config.railway.station_companies[preferred.origin as usize + 1];
        for entry in start..end {
            let company = config.railway.station_companies[entry as usize];
            let path = candidate(actors, config, buyer, seller, company);
            let affordable = !lt(cash, multiply(control, quantity, path.fee));
            if path.carrier < config.railway.company_count
                && path.preferred_to(&selected)
                && affordable
            {
                if route_capacity(
                    state,
                    config,
                    control,
                    &path,
                    quantity,
                    comptime!(DemandSignal::Ignore),
                ) {
                    selected.replace(&path);
                }
            }
        }
        if funded {
            config.railway.requests[cache].store(epoch);
            config.railway.requests[cache + 1].store(quantity.extract(LOW_WORD));
            config.railway.requests[cache + 2].store(quantity.extract(HIGH_WORD));
            config.railway.requests[cache + 3].store(selected.carrier);
        }
    }
    selected
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
        let origin = config.railway.access[seller as usize * 2];
        let destination = config.railway.access[buyer as usize * 2];
        let access = ceildiv(
            config.railway.access[seller as usize * 2 + 1]
                + config.railway.access[buyer as usize * 2 + 1],
            config.shipment_speed,
        );
        let limit = min(max(result.days, access + 1) - access - 1, config.ring - 1);
        let company = config.railway.quotes
            [((origin * config.railway.nodes + destination) * config.ring + limit) as usize];
        let path = candidate(actors, config, buyer, seller, company);
        result.replace(&path);
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
        if comptime!(traversal == Traversal::Request) && units > 0 {
            let previous = config.railway.access_requests[owner as usize].fetch_add(units);
            if previous == 0 {
                let count = config.railway.company_count;
                let index = config.railway.access_requests[(2 * count) as usize].fetch_add(1);
                config.railway.access_requests[(count + index) as usize].store(owner);
            }
        }
    }
    fee
}

#[cube]
fn request_region(config: &EconomyConfig, region: u32) {
    let previous = config.railway.requests[region as usize].fetch_add(1);
    if previous == 0 {
        let count = config.railway.region_count;
        let index = config.railway.requests[(2 * count) as usize].fetch_add(1);
        config.railway.requests[(count + index) as usize].store(region);
    }
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
                        request_region(config, pending);
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
            request_region(config, first);
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
    let regions = config.railway.region_count;
    for index in 0..config.railway.requests[(2 * regions) as usize].load() {
        let record = config.railway.requests[(regions + index) as usize].load();
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
    let companies = config.railway.company_count;
    for index in 0..config.railway.access_requests[(2 * companies) as usize].load() {
        let owner = config.railway.access_requests[(companies + index) as usize].load();
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
    config.railway.requests[(2 * config.railway.region_count + 1) as usize].fetch_add(1);
}

#[cube]
fn prepare_request(config: &EconomyConfig, path: &FreightRoute) {
    let regions = config.railway.region_count;
    for index in 0..config.railway.requests[(2 * regions) as usize].load() {
        let record = config.railway.requests[(regions + index) as usize].load();
        config.railway.requests[record as usize].store(0);
    }
    config.railway.requests[(2 * regions) as usize].store(0);
    let companies = config.railway.company_count;
    for index in 0..config.railway.access_requests[(2 * companies) as usize].load() {
        let company = config.railway.access_requests[(companies + index) as usize].load();
        config.railway.access_requests[company as usize].store(0);
    }
    config.railway.access_requests[(2 * companies) as usize].store(0);
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
) -> Ticks {
    let mut available = ticks(u32::MAX, u32::MAX);
    let regions = config.railway.region_count;
    for index in 0..config.railway.requests[(2 * regions) as usize].load() {
        let record = config.railway.requests[(regions + index) as usize].load();
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
            available = minimum(available, divide(stock, visits));
        }
    }
    available
}

#[cube]
fn route_capacity(
    state: &mut [u32],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
    path: &FreightRoute,
    quantity: Ticks,
    #[comptime] signal: DemandSignal,
) -> bool {
    let base = (2 * config.railway.region_count + 1) as usize;
    let index = base
        + 1
        + ((path.carrier * config.railway.nodes + path.origin) * config.railway.nodes
            + path.destination) as usize
            * CAPACITY_WORDS;
    let epoch = config.railway.requests[base].load();
    let current = config.railway.requests[index].load() == epoch;
    let previous = ticks(
        config.railway.requests[index + 3].load(),
        config.railway.requests[index + 4].load(),
    );
    if !current || comptime!(matches!(signal, DemandSignal::Record)) && lt(previous, quantity) {
        prepare_request(config, path);
        let available = capacity(state, config, control, quantity, comptime!(signal));
        let mut demand = select(current, previous, zero());
        if comptime!(matches!(signal, DemandSignal::Record)) {
            demand = maximum(demand, quantity);
        }
        config.railway.requests[index].store(epoch);
        config.railway.requests[index + 1].store(available.extract(LOW_WORD));
        config.railway.requests[index + 2].store(available.extract(HIGH_WORD));
        config.railway.requests[index + 3].store(demand.extract(LOW_WORD));
        config.railway.requests[index + 4].store(demand.extract(HIGH_WORD));
    }
    let available = ticks(
        config.railway.requests[index + 1].load(),
        config.railway.requests[index + 2].load(),
    );
    !lt(available, quantity)
}

#[derive(Clone, Copy)]
#[repr(u32)]
enum Settlement {
    Absent,
    Parallel,
    Serial,
    Unbatched,
}

#[cube]
pub(super) fn reset_settlements(config: &EconomyConfig, id: u32) {
    if comptime!(config.railway.enabled == Feature::Enabled) && id <= config.actors {
        config.railway.settlements[id as usize].store(comptime!(Settlement::Absent as u32));
        if id < config.actors {
            config.railway.settlements[rejection_record(config, id)].store(u32::MAX);
        }
    }
}

#[cube]
pub(super) fn classify(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    proposals: &[u32],
    scratch: &[u32],
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
    index: u32,
) {
    if index < markets::active(control, parameters) {
        let buyer = scratch[(parameters.sort_source + parameters.count + index) as usize]
            + parameters.offset;
        let seller = proposals[buyer as usize * ProposalLayout::WORDS + ProposalLayout::ACTOR];
        let path = route(actors, config, buyer, seller);
        let good = comptime!(parameters.market.good().expect("goods market") as u32);
        let available = positive(
            value(state, config, good, seller),
            household_production::reserve(state, actors, config, seller, good),
        );
        let quantity = ticks(
            proposals[buyer as usize * ProposalLayout::WORDS + ProposalLayout::QUANTITY_LOW],
            proposals[buyer as usize * ProposalLayout::WORDS + ProposalLayout::QUANTITY_HIGH],
        );
        let settlement = select(
            path.carrier < config.railway.company_count && !lt(available, quantity),
            comptime!(Settlement::Serial as u32),
            comptime!(Settlement::Parallel as u32),
        );
        config.railway.settlements[seller as usize].fetch_max(settlement);
        if equal(quantity, zero()) || quantity.extract(HIGH_WORD) > 0 {
            config.railway.settlements[seller as usize]
                .fetch_max(comptime!(Settlement::Unbatched as u32));
        }
        if kind(actors, buyer).bits() == comptime!(Good::Railway as u32)
            || kind(actors, seller).bits() == comptime!(Good::Railway as u32)
        {
            config.railway.settlements[seller as usize]
                .fetch_max(comptime!(Settlement::Serial as u32));
        }
    }
}

#[cube]
pub(super) fn dependencies(
    config: &EconomyConfig,
    proposals: &[u32],
    scratch: &[u32],
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
    index: u32,
) {
    if index < markets::active(control, parameters) {
        let buyer = scratch[(parameters.sort_source + parameters.count + index) as usize]
            + parameters.offset;
        if config.railway.settlements[buyer as usize].load() != comptime!(Settlement::Absent as u32)
        {
            let seller = proposals[buyer as usize * ProposalLayout::WORDS + ProposalLayout::ACTOR];
            config.railway.settlements[buyer as usize]
                .fetch_max(comptime!(Settlement::Serial as u32));
            config.railway.settlements[seller as usize]
                .fetch_max(comptime!(Settlement::Serial as u32));
        }
    }
}

#[cube]
pub(super) fn parallel_settlement(
    config: &EconomyConfig,
    parameters: &EconomyParameters,
    seller: u32,
) -> bool {
    let mut parallel = true;
    if comptime!(config.railway.enabled == Feature::Enabled && parameters.market.good().is_some())
        && seller < config.actors
    {
        parallel = config.railway.settlements[config.actors as usize].load()
            != comptime!(Settlement::Serial as u32)
            && config.railway.settlements[seller as usize].load()
                < comptime!(Settlement::Serial as u32);
    }
    parallel
}

#[cube]
fn rejection_record(config: &EconomyConfig, seller: u32) -> usize {
    config.actors as usize + 1 + seller as usize * REJECTION_WORDS
}

#[cube]
fn reject_remainder(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    parameters: &EconomyParameters,
    seller: u32,
    index: u32,
) {
    let base = rejection_record(config, seller);
    let good = comptime!(parameters.market.good().expect("goods market") as u32);
    let cash = value(state, config, comptime!(Field::Cash as u32), seller);
    let sales = value(state, config, comptime!(Field::Sales as u32), seller);
    let reserve = household_production::reserve(state, actors, config, seller, good);
    config.railway.settlements[base].store(index);
    config.railway.settlements[base + 1].store(cash.extract(LOW_WORD));
    config.railway.settlements[base + 2].store(cash.extract(HIGH_WORD));
    config.railway.settlements[base + 3].store(sales.extract(LOW_WORD));
    config.railway.settlements[base + 4].store(sales.extract(HIGH_WORD));
    config.railway.settlements[base + 5].store(reserve.extract(LOW_WORD));
    config.railway.settlements[base + 6].store(reserve.extract(HIGH_WORD));
}

#[cube]
pub(super) fn finalize_rejection(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
    proposals: &[u32],
    scratch: &mut [u32],
    buyer: u32,
    seller: u32,
    index: u32,
) {
    let base = rejection_record(config, seller);
    if config.railway.settlements[config.actors as usize].load()
        != comptime!(Settlement::Serial as u32)
        && index >= config.railway.settlements[base].load()
    {
        let quantity = ticks(
            proposals[buyer as usize * ProposalLayout::WORDS + ProposalLayout::QUANTITY_LOW],
            proposals[buyer as usize * ProposalLayout::WORDS + ProposalLayout::QUANTITY_HIGH],
        );
        let good = comptime!(parameters.market.good().expect("goods market") as u32);
        let path = animal_route(actors, config, buyer, seller);
        let arrival = BASE_FIELDS as u32
            + (good - 1) * config.ring
            + (parameters.day + path.days) % config.ring;
        let cost = multiply(
            control,
            quantity,
            number(state, config, comptime!(Field::Price as u32), seller),
        );
        let cash = ticks(
            config.railway.settlements[base + 1].load(),
            config.railway.settlements[base + 2].load(),
        );
        let sales = ticks(
            config.railway.settlements[base + 3].load(),
            config.railway.settlements[base + 4].load(),
        );
        let reserve = ticks(
            config.railway.settlements[base + 5].load(),
            config.railway.settlements[base + 6].load(),
        );
        add_checked(control, reserve, quantity);
        let faults = plus(cash, cost)
            .faults
            .union(plus(sales, quantity).faults)
            .union(plus(value(state, config, arrival, buyer), quantity).faults)
            .union(
                plus(
                    value(state, config, TRANSIT_FIELDS as u32 + good - 1, buyer),
                    quantity,
                )
                .faults,
            )
            .union(
                plus(
                    value(state, config, comptime!(Field::Transit as u32), buyer),
                    quantity,
                )
                .faults,
            )
            .union(
                plus(
                    value(state, config, comptime!(Field::Purchased as u32), buyer),
                    quantity,
                )
                .faults,
            );
        let mut status = KernelEventStatus::new(comptime!(EventStatus::Rejected));
        if !faults.is_empty() {
            status.set(comptime!(EventStatus::Faulted));
            faults.record(control);
        }
        status.store(scratch, 4 * parameters.count + buyer - parameters.offset);
    }
}

#[derive(CubeType)]
struct GoodsSettlement {
    buyer_cash: Ticks,
    seller_cash: Ticks,
    stock: Ticks,
    arrival: Ticks,
    transit_good: Ticks,
    transit: Ticks,
    sales: Ticks,
    purchased: Ticks,
    arrival_field: u32,
    status: KernelEventStatus,
}

#[cube]
impl GoodsSettlement {
    fn new(
        state: &mut [u32],
        actors: &[u32],
        config: &EconomyConfig,
        control: &[Atomic<u32>],
        parameters: &EconomyParameters,
        buyer: u32,
        seller: u32,
        quantity: Ticks,
    ) -> Self {
        let good = comptime!(parameters.market.good().expect("goods market") as u32);
        let cost = multiply(
            control,
            quantity,
            number(state, config, comptime!(Field::Price as u32), seller),
        );
        let cash = value(state, config, comptime!(Field::Cash as u32), buyer);
        let seller_cash = plus(
            value(state, config, comptime!(Field::Cash as u32), seller),
            cost,
        );
        let stock = value(state, config, good, seller);
        let required = add_checked(
            control,
            household_production::reserve(state, actors, config, seller, good),
            quantity,
        );
        let path = animal_route(actors, config, buyer, seller);
        let arrival_field = BASE_FIELDS as u32
            + (good - 1) * config.ring
            + (parameters.day + path.days) % config.ring;
        let arrival = plus(value(state, config, arrival_field, buyer), quantity);
        let transit_good = plus(
            value(state, config, TRANSIT_FIELDS as u32 + good - 1, buyer),
            quantity,
        );
        let transit = plus(
            value(state, config, comptime!(Field::Transit as u32), buyer),
            quantity,
        );
        let sales = plus(
            value(state, config, comptime!(Field::Sales as u32), seller),
            quantity,
        );
        let purchased = plus(
            value(state, config, comptime!(Field::Purchased as u32), buyer),
            quantity,
        );
        let valid = !lt(cash, cost) && !lt(stock, quantity) && !lt(stock, required);
        let faults = seller_cash
            .faults
            .union(arrival.faults)
            .union(transit_good.faults)
            .union(transit.faults)
            .union(sales.faults)
            .union(purchased.faults);
        let mut status = KernelEventStatus::new(comptime!(EventStatus::Rejected));
        if valid {
            status.set(comptime!(EventStatus::Accepted));
        }
        if !faults.is_empty() {
            status.set(comptime!(EventStatus::Faulted));
            faults.record(control);
        }
        GoodsSettlement {
            buyer_cash: minus(cash, cost),
            seller_cash: seller_cash.value,
            stock: minus(stock, quantity),
            arrival: arrival.value,
            transit_good: transit_good.value,
            transit: transit.value,
            sales: sales.value,
            purchased: purchased.value,
            arrival_field,
            status,
        }
    }

    fn reroute(
        &mut self,
        state: &mut [u32],
        config: &EconomyConfig,
        control: &[Atomic<u32>],
        parameters: &EconomyParameters,
        buyer: u32,
        quantity: Ticks,
        path: &FreightRoute,
    ) {
        let good = comptime!(parameters.market.good().expect("goods market") as u32);
        self.seller_cash = minus(self.seller_cash, multiply(control, quantity, path.fee));
        self.arrival_field = BASE_FIELDS as u32
            + (good - 1) * config.ring
            + (parameters.day + path.days) % config.ring;
        let arrival = plus(value(state, config, self.arrival_field, buyer), quantity);
        self.arrival = arrival.value;
        if !arrival.faults.is_empty() {
            self.status.set(comptime!(EventStatus::Faulted));
            arrival.faults.record(control);
        }
    }

    fn apply(
        &self,
        state: &mut [u32],
        config: &EconomyConfig,
        parameters: &EconomyParameters,
        buyer: u32,
        seller: u32,
    ) {
        if self.status.contains(comptime!(EventStatus::Accepted)) {
            let good = comptime!(parameters.market.good().expect("goods market") as u32);
            set(
                state,
                config,
                comptime!(Field::Cash as u32),
                buyer,
                self.buyer_cash,
            );
            set(
                state,
                config,
                comptime!(Field::Cash as u32),
                seller,
                self.seller_cash,
            );
            set(state, config, good, seller, self.stock);
            set(state, config, self.arrival_field, buyer, self.arrival);
            set(
                state,
                config,
                TRANSIT_FIELDS as u32 + good - 1,
                buyer,
                self.transit_good,
            );
            set(
                state,
                config,
                comptime!(Field::Transit as u32),
                buyer,
                self.transit,
            );
            set(
                state,
                config,
                comptime!(Field::Sales as u32),
                seller,
                self.sales,
            );
            set(
                state,
                config,
                comptime!(Field::Purchased as u32),
                buyer,
                self.purchased,
            );
            let price = value(state, config, comptime!(Field::Price as u32), seller);
            for input in 0..2u32 {
                if number(
                    state,
                    config,
                    comptime!(Field::FirstInput as u32) + input,
                    buyer,
                ) == good
                {
                    set(
                        state,
                        config,
                        comptime!(Field::FirstInputCost as u32) + input,
                        buyer,
                        price,
                    );
                }
            }
            if number(
                state,
                config,
                comptime!(Field::EnterpriseModel as u32),
                buyer,
            ) == 1
            {
                set(
                    state,
                    config,
                    comptime!(Field::ReplacementCost as u32),
                    buyer,
                    price,
                );
            }
        }
    }
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
        let mut index = 0;
        let mut completed = 0;
        while index < active {
            let seller = scratch[(parameters.sort_source + index) as usize];
            if parallel_settlement(config, parameters, seller) {
                index = scratch[(7 * count + seller) as usize];
            } else {
                let good = comptime!(parameters.market.good().expect("goods market") as u32);
                let reserve = household_production::reserve(state, actors, config, seller, good);
                let exhausted = !lt(reserve, value(state, config, good, seller))
                    && reserve.extract(HIGH_WORD) == 0
                    && config.railway.settlements[seller as usize].load()
                        != comptime!(Settlement::Unbatched as u32)
                    && config.railway.settlements[config.actors as usize].load()
                        != comptime!(Settlement::Serial as u32);
                if exhausted {
                    reject_remainder(state, actors, config, parameters, seller, index);
                    let end = scratch[(7 * count + seller) as usize];
                    completed += end - index;
                    index = end;
                } else {
                    let id = scratch[(parameters.sort_source + count + index) as usize];
                    let buyer = id + parameters.offset;
                    let seller =
                        proposals[buyer as usize * ProposalLayout::WORDS + ProposalLayout::ACTOR];
                    let quantity = ticks(
                        proposals
                            [buyer as usize * ProposalLayout::WORDS + ProposalLayout::QUANTITY_LOW],
                        proposals[buyer as usize * ProposalLayout::WORDS
                            + ProposalLayout::QUANTITY_HIGH],
                    );
                    let mut selected = animal_route(actors, config, buyer, seller);
                    let mut settlement = GoodsSettlement::new(
                        state, actors, config, control, parameters, buyer, seller, quantity,
                    );
                    if settlement.status.contains(comptime!(EventStatus::Accepted)) {
                        let preferred = route(actors, config, buyer, seller);
                        if preferred.carrier < config.railway.company_count {
                            route_capacity(
                                state,
                                config,
                                control,
                                &preferred,
                                quantity,
                                comptime!(DemandSignal::Record),
                            );
                            let path = available_route(
                                state, actors, config, control, buyer, seller, quantity, &preferred,
                            );
                            selected.replace(&path);
                        }
                    }
                    if selected.carrier < config.railway.company_count {
                        route_capacity(
                            state,
                            config,
                            control,
                            &selected,
                            quantity,
                            comptime!(DemandSignal::Record),
                        );
                        prepare_request(config, &selected);
                        settlement.reroute(
                            state, config, control, parameters, buyer, quantity, &selected,
                        );
                    }
                    settlement.apply(state, config, parameters, buyer, seller);
                    if selected.carrier < config.railway.company_count
                        && settlement.status.contains(comptime!(EventStatus::Accepted))
                    {
                        purchase(state, config, control, seller, &selected, quantity);
                    }
                    settlement.status.store(scratch, 4 * count + id);
                    index += 1;
                    completed += 1;
                }
            }
        }
        control[PROGRESS_WORD].fetch_max(active);
        control[COMPLETED_WORD].fetch_add(completed);
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

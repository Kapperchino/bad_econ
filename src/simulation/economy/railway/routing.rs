use super::*;

#[derive(CubeType, Clone, Copy)]
pub(in crate::simulation::economy) struct FreightRoute {
    pub days: u32,
    pub fee: u32,
    pub origin: u32,
    pub destination: u32,
    pub carrier: u32,
    pub foreign_distance: u32,
}

#[cube]
pub(super) fn station(actors: &ActorTable<'_>, config: &EconomyConfig, actor: u32) -> u32 {
    let x = actors.x(actor);
    let y = actors.y(actor);
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
pub(super) fn access(actors: &ActorTable<'_>, config: &mut EconomyConfig) {
    let actor = ABSOLUTE_POS as u32;
    if actor < config.actors && config.railway.nodes > 0 {
        let x = actors.x(actor);
        let y = actors.y(actor);
        let position = config.railway.access.len() / 2 + actor as usize * 2;
        if config.railway.access[position] != x + 1 || config.railway.access[position + 1] != y {
            let node = station(actors, &*config, actor);
            config.railway.access[actor as usize * 2] = node;
            config.railway.access[actor as usize * 2 + 1] =
                access_distance(actors, &*config, actor, node);
            config.railway.access[position] = x + 1;
            config.railway.access[position + 1] = y;
        }
    }
}

#[cube]
pub(super) fn access_distance(
    actors: &ActorTable<'_>,
    config: &EconomyConfig,
    actor: u32,
    node: u32,
) -> u32 {
    let x = actors.x(actor);
    let y = actors.y(actor);
    let nx = config.railway.stations[node as usize * 2];
    let ny = config.railway.stations[node as usize * 2 + 1];
    max(x, nx) - min(x, nx) + max(y, ny) - min(y, ny)
}

#[cube]
pub(in crate::simulation::economy) fn animal_route(
    actors: &ActorTable<'_>,
    config: &EconomyConfig,
    buyer: u32,
    seller: u32,
) -> FreightRoute {
    FreightRoute {
        days: max(
            ceildiv(actors.distance(buyer, seller), config.shipment_speed),
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
pub(super) fn candidate(
    actors: &ActorTable<'_>,
    config: &EconomyConfig,
    buyer: u32,
    seller: u32,
    company: u32,
) -> FreightRoute {
    FreightRequest::new(actors, config, buyer, seller).candidate(config, company)
}

#[derive(CubeType)]
pub(super) struct FreightRequest {
    pub(super) animal_days: u32,
    origin: u32,
    destination: u32,
    access_days: u32,
}

#[cube]
impl FreightRequest {
    pub(super) fn new(
        actors: &ActorTable<'_>,
        config: &EconomyConfig,
        buyer: u32,
        seller: u32,
    ) -> Self {
        let animal_days = max(
            ceildiv(actors.distance(buyer, seller), config.shipment_speed),
            1,
        );
        let mut origin = 0u32;
        let mut destination = 0u32;
        let mut access_days = 0u32;
        if comptime!(config.railway.routes_enabled == Feature::Enabled) && config.railway.nodes > 0
        {
            origin = config.railway.access[seller as usize * 2];
            destination = config.railway.access[buyer as usize * 2];
            access_days = ceildiv(
                config.railway.access[seller as usize * 2 + 1]
                    + config.railway.access[buyer as usize * 2 + 1],
                config.shipment_speed,
            );
        }
        FreightRequest {
            animal_days,
            origin,
            destination,
            access_days,
        }
    }

    pub(super) fn animal(&self) -> FreightRoute {
        FreightRoute {
            days: self.animal_days,
            fee: 0,
            origin: 0,
            destination: 0,
            carrier: u32::MAX,
            foreign_distance: u32::MAX,
        }
    }

    pub(super) fn candidate(&self, config: &EconomyConfig, company: u32) -> FreightRoute {
        let mut result = self.animal();
        if comptime!(config.railway.routes_enabled == Feature::Enabled)
            && config.railway.nodes > 0
            && company < config.railway.company_count
        {
            let base = route_cell(config, company, self.origin, self.destination);
            let carrier = config.railway.routes[base + 2];
            let days = self.access_days
                + max(
                    ceildiv(
                        config.railway.routes[base],
                        config.shipment_speed * config.railway.speed_multiplier,
                    ),
                    1,
                );
            let faster = carrier < config.railway.company_count
                && days < self.animal_days
                && config.railway.routes[base + 3] < cartage::rate(config, self.animal_days);
            result.days = select(faster, days, self.animal_days);
            result.fee = select(faster, config.railway.routes[base + 3], 0);
            result.origin = self.origin;
            result.destination = self.destination;
            result.carrier = select(faster, carrier, u32::MAX);
            result.foreign_distance = select(faster, config.railway.routes[base + 5], u32::MAX);
        }
        result
    }

    pub(super) fn quoted_carrier(&self, config: &EconomyConfig) -> u32 {
        let mut company = u32::new(comptime!(i64::from(u32::MAX)));
        if comptime!(config.railway.routes_enabled == Feature::Enabled) && config.railway.nodes > 0
        {
            let limit = min(
                max(self.animal_days, self.access_days + 1) - self.access_days - 1,
                config.ring - 1,
            );
            company = config.railway.quotes[((self.origin * config.railway.nodes
                + self.destination)
                * config.ring
                + limit) as usize];
        }
        company
    }

    fn preferred(&self, config: &EconomyConfig) -> FreightRoute {
        self.candidate(config, self.quoted_carrier(config))
    }
}

#[cube]
pub(super) fn inbound(config: &mut EconomyConfig) {
    let destination = ABSOLUTE_POS as u32;
    let nodes = config.railway.nodes;
    let table = nodes * nodes * config.ring;
    let cache = config.railway.company_count as usize * COMPANY_WORDS;
    if destination < nodes
        && config.railway.companies[cache + 1] != comptime!(RouteCacheState::Clean as u32)
    {
        let mut origin = 0u32;
        let mut connected = false;
        while origin < nodes && !connected {
            let last = (origin * nodes + destination + 1) * config.ring - 1;
            connected = config.railway.quotes[last as usize] < config.railway.company_count;
            origin += 1;
        }
        config.railway.quotes[(table + destination) as usize] = u32::cast_from(connected);
    }
}

#[cube]
pub(in crate::simulation::economy) fn serves_buyer(config: &EconomyConfig, buyer: u32) -> bool {
    let mut available = false;
    if comptime!(config.railway.routes_enabled == Feature::Enabled) && config.railway.nodes > 0 {
        available = true;
        if config.railway.route_index != 0 {
            let node = config.railway.access[buyer as usize * 2];
            let table = config.railway.nodes * config.railway.nodes * config.ring;
            available = config.railway.quotes[(table + node) as usize] != 0;
        }
    }
    available
}

#[cube]
pub(super) fn quotes(config: &mut EconomyConfig) {
    let index = ABSOLUTE_POS as u32;
    let nodes = config.railway.nodes;
    let cache = config.railway.company_count as usize * COMPANY_WORDS;
    let dirty = config.railway.companies[cache + 1] != comptime!(RouteCacheState::Clean as u32);
    if index < nodes * nodes * config.ring && dirty {
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
        for entry in start..end {
            let company = config.railway.station_companies[entry as usize];
            let base = route_cell(&*config, company, selected.origin, selected.destination);
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
                if path.preferred_to(&selected) {
                    selected.replace(&path);
                }
            }
        }
        config.railway.quotes[index as usize] = selected.carrier;
    }
}

#[cube]
pub(in crate::simulation::economy) fn route(
    actors: &ActorTable<'_>,
    config: &EconomyConfig,
    buyer: u32,
    seller: u32,
) -> FreightRoute {
    FreightRequest::new(actors, config, buyer, seller).preferred(config)
}

#[derive(CubeType, Clone, Copy)]
pub(in crate::simulation::economy) struct DeliveryRequest {
    pub buyer: u32,
    pub seller: u32,
    pub trip: u32,
    pub buyer_access: u32,
}

#[cube]
pub(in crate::simulation::economy) fn may_book(
    config: &EconomyConfig,
    buyer: u32,
    seller: u32,
    trip: u32,
    buyer_access: u32,
) -> bool {
    let animal_days = max(ceildiv(trip, config.shipment_speed), 1);
    let request = FreightRequest {
        animal_days,
        origin: config.railway.access[seller as usize * 2],
        destination: config.railway.access[buyer as usize * 2],
        access_days: ceildiv(
            config.railway.access[seller as usize * 2 + 1] + buyer_access,
            config.shipment_speed,
        ),
    };
    animal_days > config.ring || request.quoted_carrier(config) < config.railway.company_count
}

#[cube]
pub(in crate::simulation::economy) fn delivery_days(
    config: &EconomyConfig,
    request: DeliveryRequest,
) -> u32 {
    let mut days = max(ceildiv(request.trip, config.shipment_speed), 1);
    if comptime!(config.railway.routes_enabled == Feature::Enabled) && config.railway.nodes > 0 {
        let origin = config.railway.access[request.seller as usize * 2];
        let destination = config.railway.access[request.buyer as usize * 2];
        let access = ceildiv(
            config.railway.access[request.seller as usize * 2 + 1] + request.buyer_access,
            config.shipment_speed,
        );
        let limit = min(max(days, access + 1) - access - 1, config.ring - 1);
        let company = config.railway.quotes
            [((origin * config.railway.nodes + destination) * config.ring + limit) as usize];
        if company < config.railway.company_count {
            let base = route_cell(config, company, origin, destination);
            let carrier = config.railway.routes[base + 2];
            let candidate = access
                + max(
                    ceildiv(
                        config.railway.routes[base],
                        config.shipment_speed * config.railway.speed_multiplier,
                    ),
                    1,
                );
            days = select(
                carrier < config.railway.company_count
                    && candidate < days
                    && config.railway.routes[base + 3] < cartage::rate(config, days),
                candidate,
                days,
            );
        }
    }
    days
}

#[cube]
pub(super) fn access_fare(config: &EconomyConfig, owner: u32, carrier: u32, regions: u32) -> u32 {
    let mut fee = 0;
    if owner < config.railway.company_count {
        let foreign = config.railway.companies[owner as usize * COMPANY_WORDS + 3]
            != config.railway.companies[carrier as usize * COMPANY_WORDS + 3];
        let units = select(foreign, max(regions, 1), 0);
        fee = units * config.railway.companies[owner as usize * COMPANY_WORDS + 2];
    }
    fee
}

#[derive(CubeType)]
pub(super) struct RouteTraversal {
    pub(super) fee: u32,
    pub(super) visits: u32,
    pub(super) segments: u32,
}

#[cube]
pub(super) fn traverse(config: &EconomyConfig, path: &FreightRoute) -> RouteTraversal {
    let mut cursor = path.destination;
    let mut previous = u32::new(comptime!(i64::from(u32::MAX)));
    let mut first = u32::new(comptime!(i64::from(u32::MAX)));
    let mut pending = u32::new(comptime!(i64::from(u32::MAX)));
    let mut fee = 0u32;
    let mut visits = 0u32;
    let mut toll = 0u32;
    let mut access_owner = u32::new(comptime!(i64::from(u32::MAX)));
    let mut access_region = u32::new(comptime!(i64::from(u32::MAX)));
    let mut access_units = 0u32;
    let rate = config.railway.companies[path.carrier as usize * COMPANY_WORDS + 1];
    let mut steps = 0u32;
    while cursor != path.origin && steps < config.railway.nodes {
        let edge = config.railway.routes[route_cell(config, path.carrier, path.origin, cursor) + 1];
        if edge < config.railway.segment_count {
            let base = edge as usize * SEGMENT_WORDS;
            let owner = config.railway.segments[base + 2];
            if owner != access_owner {
                toll += access_fare(config, access_owner, path.carrier, access_units);
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
                let region = config.railway.regions[record as usize * REGION_WORDS + 1];
                access_units +=
                    u32::cast_from(access_region != u32::MAX && access_region != region);
                access_region = region;
                first = select(first == u32::MAX, record, first);
                let entered = region != previous && pending < config.railway.region_count;
                visits += u32::cast_from(entered);
                if entered {
                    fee += rate;
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
    }
    RouteTraversal {
        fee: fee + toll + access_fare(config, access_owner, path.carrier, access_units),
        visits: max(visits, u32::cast_from(first < config.railway.region_count)),
        segments: steps,
    }
}

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

#[derive(CubeType)]
pub(in crate::simulation::economy) struct FreightRequest {
    pub(super) animal_days: u32,
    origin: u32,
    destination: u32,
    access_days: u32,
}

#[cube]
impl FreightRequest {
    pub(in crate::simulation::economy) fn new(
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
        FreightRoute::road(self.animal_days)
    }

    pub(super) fn candidate(&self, config: &EconomyConfig, company: u32) -> FreightRoute {
        let mut result = self.animal();
        if comptime!(config.railway.routes_enabled == Feature::Enabled)
            && config.railway.nodes > 0
            && company < config.railway.company_count
        {
            let base = config
                .railway
                .route_cell(company, self.origin, self.destination);
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

    pub(in crate::simulation::economy) fn preferred(&self, config: &EconomyConfig) -> FreightRoute {
        self.candidate(config, self.quoted_carrier(config))
    }
}

#[derive(CubeType, Clone, Copy)]
pub(in crate::simulation::economy) struct DeliveryRequest {
    pub buyer: u32,
    pub seller: u32,
    pub trip: u32,
    pub buyer_access: u32,
}

#[derive(CubeType)]
pub(super) struct RouteTraversal {
    pub(super) fee: u32,
    pub(super) visits: u32,
    pub(super) segments: u32,
}

#[derive(CubeType)]
pub(super) struct RouteCursor {
    node: u32,
    origin: u32,
    carrier: u32,
    steps: u32,
    pub(super) owner: u32,
    start: u32,
    pub(super) length: u32,
    reverse: bool,
}

#[cube]
impl RouteCursor {
    pub(super) fn new(path: &FreightRoute) -> Self {
        RouteCursor {
            node: path.destination,
            origin: path.origin,
            carrier: path.carrier,
            steps: 0,
            owner: u32::MAX,
            start: 0,
            length: 0,
            reverse: false,
        }
    }

    pub(super) fn has_next(&self, network: &RailNetwork) -> bool {
        self.node != self.origin && self.steps < network.nodes
    }

    pub(super) fn advance(&mut self, network: &RailNetwork) -> bool {
        let edge = network.routes[network.route_cell(self.carrier, self.origin, self.node) + 1];
        self.steps += 1;
        let present = edge < network.segment_count;
        if present {
            let base = edge as usize * SEGMENT_WORDS;
            self.reverse = network.segments[base + 1] == self.node;
            self.node = select(
                self.reverse,
                network.segments[base],
                network.segments[base + 1],
            );
            self.owner = network.segments[base + 2];
            self.start = network.segments[base + 5];
            self.length = network.segments[base + 6];
        }
        present
    }

    pub(super) fn record(&self, network: &RailNetwork, entry: u32) -> u32 {
        let offset = select(self.reverse, self.length - entry - 1, entry);
        network.segment_regions[(self.start + offset) as usize]
    }
}

#[derive(CubeType)]
pub(super) struct TrackAccess {
    pub(super) owner: u32,
    region: u32,
    pub(super) units: u32,
}

#[cube]
impl TrackAccess {
    pub(super) fn new() -> Self {
        TrackAccess {
            owner: u32::MAX,
            region: u32::MAX,
            units: 0,
        }
    }

    pub(super) fn start_on(&mut self, owner: u32) {
        self.owner = owner;
        self.region = u32::MAX;
        self.units = 0;
    }

    pub(super) fn visit(&mut self, region: u32) {
        self.units += u32::cast_from(self.region != u32::MAX && self.region != region);
        self.region = region;
    }
}

#[cube]
impl FreightRoute {
    pub(in crate::simulation::economy) fn by_road(
        actors: &ActorTable<'_>,
        config: &EconomyConfig,
        buyer: u32,
        seller: u32,
    ) -> Self {
        FreightRoute::road(max(
            ceildiv(actors.distance(buyer, seller), config.shipment_speed),
            1,
        ))
    }

    pub(super) fn traverse(&self, network: &RailNetwork) -> RouteTraversal {
        let mut cursor = RouteCursor::new(self);
        let mut previous = u32::new(comptime!(i64::from(u32::MAX)));
        let mut first = u32::new(comptime!(i64::from(u32::MAX)));
        let mut pending = u32::new(comptime!(i64::from(u32::MAX)));
        let mut fee = 0u32;
        let mut visits = 0u32;
        let mut toll = 0u32;
        let mut access = TrackAccess::new();
        let rate = network.companies[self.carrier as usize * COMPANY_WORDS + 1];
        while cursor.has_next(network) {
            if cursor.advance(network) {
                if cursor.owner != access.owner {
                    toll += network.access_fare(access.owner, self.carrier, access.units);
                    access.start_on(cursor.owner);
                }
                for index in 0..cursor.length {
                    let record = cursor.record(network, index);
                    let region = network.regions[record as usize * REGION_WORDS + 1];
                    access.visit(region);
                    first = select(first == u32::MAX, record, first);
                    let entered = region != previous && pending < network.region_count;
                    visits += u32::cast_from(entered);
                    if entered {
                        fee += rate;
                    }
                    pending = record;
                    previous = region;
                }
            }
        }
        if fee == 0 && first < network.region_count {
            fee = rate;
        }
        RouteTraversal {
            fee: fee + toll + network.access_fare(access.owner, self.carrier, access.units),
            visits: max(visits, u32::cast_from(first < network.region_count)),
            segments: cursor.steps,
        }
    }
}

#[cube]
impl FreightRoute {
    fn road(days: u32) -> Self {
        FreightRoute {
            days,
            fee: 0,
            origin: 0,
            destination: 0,
            carrier: u32::MAX,
            foreign_distance: u32::MAX,
        }
    }
}

#[cube]
impl DeliveryRequest {
    fn road_days(&self, config: &EconomyConfig) -> u32 {
        max(ceildiv(self.trip, config.shipment_speed), 1)
    }

    fn freight(&self, config: &EconomyConfig, road_days: u32) -> FreightRequest {
        FreightRequest {
            animal_days: road_days,
            origin: config.railway.access[self.seller as usize * 2],
            destination: config.railway.access[self.buyer as usize * 2],
            access_days: ceildiv(
                config.railway.access[self.seller as usize * 2 + 1] + self.buyer_access,
                config.shipment_speed,
            ),
        }
    }

    pub(in crate::simulation::economy) fn may_book(&self, config: &EconomyConfig) -> bool {
        let days = self.road_days(config);
        let freight = self.freight(config, days);
        days > config.ring || freight.quoted_carrier(config) < config.railway.company_count
    }

    pub(in crate::simulation::economy) fn days(&self, config: &EconomyConfig) -> u32 {
        let mut days = self.road_days(config);
        if comptime!(config.railway.routes_enabled == Feature::Enabled) && config.railway.nodes > 0
        {
            days = self.freight(config, days).preferred(config).days;
        }
        days
    }
}

#[cube]
impl RailNetwork {
    pub(super) fn nearest_station(&self, actors: &ActorTable<'_>, actor: u32) -> u32 {
        let mut chosen = 0;
        let mut distance = u32::new(comptime!(i64::from(u32::MAX)));
        for node in 0..self.nodes {
            let trip = self.station_distance(actors, actor, node);
            chosen = select(trip < distance, node, chosen);
            distance = min(distance, trip);
        }
        chosen
    }

    pub(super) fn refresh_access(&mut self, actors: &ActorTable<'_>, actor_count: u32) {
        let actor = ABSOLUTE_POS as u32;
        if actor < actor_count && self.nodes > 0 {
            let x = actors.x(actor);
            let y = actors.y(actor);
            let position = self.access.len() / 2 + actor as usize * 2;
            if self.access[position] != x + 1 || self.access[position + 1] != y {
                let node = self.nearest_station(actors, actor);
                self.access[actor as usize * 2] = node;
                self.access[actor as usize * 2 + 1] = self.station_distance(actors, actor, node);
                self.access[position] = x + 1;
                self.access[position + 1] = y;
            }
        }
    }

    pub(super) fn station_distance(&self, actors: &ActorTable<'_>, actor: u32, node: u32) -> u32 {
        let x = actors.x(actor);
        let y = actors.y(actor);
        let nx = self.stations[node as usize * 2];
        let ny = self.stations[node as usize * 2 + 1];
        max(x, nx) - min(x, nx) + max(y, ny) - min(y, ny)
    }

    pub(super) fn refresh_inbound(&mut self, ring: u32) {
        let destination = ABSOLUTE_POS as u32;
        let nodes = self.nodes;
        let table = nodes * nodes * ring;
        let cache = self.company_count as usize * COMPANY_WORDS;
        if destination < nodes
            && self.companies[cache + 1] != comptime!(RouteCacheState::Clean as u32)
        {
            let mut origin = 0u32;
            let mut connected = false;
            while origin < nodes && !connected {
                let last = (origin * nodes + destination + 1) * ring - 1;
                connected = self.quotes[last as usize] < self.company_count;
                origin += 1;
            }
            self.quotes[(table + destination) as usize] = u32::cast_from(connected);
        }
    }

    pub(in crate::simulation::economy) fn serves_buyer(&self, buyer: u32, ring: u32) -> bool {
        let mut available = false;
        if comptime!(self.routes_enabled == Feature::Enabled) && self.nodes > 0 {
            available = true;
            if self.route_index != 0 {
                let node = self.access[buyer as usize * 2];
                let table = self.nodes * self.nodes * ring;
                available = self.quotes[(table + node) as usize] != 0;
            }
        }
        available
    }

    pub(super) fn refresh_quotes(&mut self, ring: u32, shipment_speed: u32) {
        let index = ABSOLUTE_POS as u32;
        let nodes = self.nodes;
        let cache = self.company_count as usize * COMPANY_WORDS;
        let dirty = self.companies[cache + 1] != comptime!(RouteCacheState::Clean as u32);
        if index < nodes * nodes * ring && dirty {
            let limit = index % ring;
            let pair = index / ring;
            let mut selected = FreightRoute {
                days: u32::MAX,
                fee: u32::MAX,
                origin: pair / nodes,
                destination: pair % nodes,
                carrier: u32::MAX,
                foreign_distance: u32::MAX,
            };
            let start = self.station_companies[selected.origin as usize];
            let end = self.station_companies[selected.origin as usize + 1];
            for entry in start..end {
                let company = self.station_companies[entry as usize];
                let base = self.route_cell(company, selected.origin, selected.destination);
                let path = FreightRoute {
                    days: max(
                        ceildiv(self.routes[base], shipment_speed * self.speed_multiplier),
                        1,
                    ),
                    fee: self.routes[base + 3],
                    origin: selected.origin,
                    destination: selected.destination,
                    carrier: self.routes[base + 2],
                    foreign_distance: self.routes[base + 5],
                };
                if path.carrier < self.company_count
                    && path.days <= limit
                    && path.preferred_to(&selected)
                {
                    selected.replace(&path);
                }
            }
            self.quotes[index as usize] = selected.carrier;
        }
    }

    pub(super) fn access_fare(&self, owner: u32, carrier: u32, regions: u32) -> u32 {
        let mut fee = 0;
        if owner < self.company_count {
            let foreign = self.companies[owner as usize * COMPANY_WORDS + 3]
                != self.companies[carrier as usize * COMPANY_WORDS + 3];
            let units = select(foreign, max(regions, 1), 0);
            fee = units * self.companies[owner as usize * COMPANY_WORDS + 2];
        }
        fee
    }
}

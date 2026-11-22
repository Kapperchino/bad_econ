use super::*;

#[cube]
pub(super) fn dirty_carriers(levels: &LedgerColumn<'_>, railway: &mut RailNetwork) {
    let companies = railway.company_count;
    let base = companies as usize * COMPANY_WORDS;
    let carrier = CUBE_POS as u32;
    let mut affected = railway.companies[base] == comptime!(RouteCacheState::Unknown as u32);
    let mut owner = UNIT_POS;
    while owner < companies && !affected {
        let firm = railway.companies[owner as usize * COMPANY_WORDS];
        let operating = levels.number(firm);
        let cached = railway.companies[base + 2 + owner as usize];
        affected = cached != operating
            && railway.access_rights[(carrier * companies + owner) as usize] > 0;
        owner += CUBE_DIM;
    }
    let changed = parallel::reduce::<u32>(u32::cast_from(affected), comptime!(Reduction::Maximum));
    if UNIT_POS == 0 && carrier < companies {
        railway.companies[base + 2 + companies as usize + carrier as usize] = select(
            changed > 0,
            comptime!(RouteCacheState::Dirty as u32),
            comptime!(RouteCacheState::Clean as u32),
        );
    }
}

#[cube]
pub(super) fn prepare_routes(levels: &LedgerColumn<'_>, railway: &mut RailNetwork) {
    let companies = railway.company_count;
    let base = companies as usize * COMPANY_WORDS;
    let mut changed = 0u32;
    let mut owner = UNIT_POS;
    while owner < companies {
        let firm = railway.companies[owner as usize * COMPANY_WORDS];
        railway.companies[base + 2 + owner as usize] = levels.number(firm);
        changed |= u32::cast_from(
            railway.companies[base + 2 + companies as usize + owner as usize]
                == comptime!(RouteCacheState::Dirty as u32),
        );
        owner += CUBE_DIM;
    }
    let any = parallel::reduce::<u32>(changed, comptime!(Reduction::Maximum));
    if UNIT_POS == 0 {
        railway.companies[base] = select(
            any > 0,
            comptime!(RouteCacheState::Dirty as u32),
            comptime!(RouteCacheState::Clean as u32),
        );
        if any > 0 {
            railway.companies[base + 1] = comptime!(RouteCacheState::Dirty as u32);
        }
    }
}

#[cube]
pub(super) fn complete_quotes(config: &mut EconomyConfig) {
    if ABSOLUTE_POS == 0 {
        let base = config.railway.company_count as usize * COMPANY_WORDS;
        config.railway.companies[base + 1] = comptime!(RouteCacheState::Clean as u32);
    }
}

#[cube]
pub(super) fn route_cell(
    config: &EconomyConfig,
    carrier: u32,
    origin: u32,
    destination: u32,
) -> usize {
    let nodes = config.railway.nodes;
    let mut row = carrier * nodes + origin;
    if config.railway.route_index != 0 {
        row = config.railway.station_companies[(config.railway.route_index + row) as usize];
    }
    (row * nodes + destination) as usize * ROUTE_WORDS
}

#[cube]
pub(super) fn routes(levels: &LedgerColumn<'_>, config: &mut EconomyConfig) {
    let invocation = ABSOLUTE_POS as u32;
    let nodes = config.railway.nodes;
    let mut pair = invocation;
    if config.railway.route_index != 0 && invocation < config.railway.route_rows {
        pair = config.railway.station_companies[(config.railway.route_index
            + nodes * config.railway.company_count
            + invocation) as usize];
    }
    let carrier = pair / max(nodes, 1);
    let cache = config.railway.company_count as usize * COMPANY_WORDS;
    let cached_carrier = min(carrier, config.railway.company_count - 1);
    let dirty = config.railway.companies
        [cache + 2 + config.railway.company_count as usize + cached_carrier as usize]
        == comptime!(RouteCacheState::Dirty as u32);
    if invocation < config.railway.route_rows && dirty {
        let origin = pair % nodes;
        let row = invocation * nodes * ROUTE_WORDS as u32;
        let incident_start = config.railway.station_companies[nodes as usize];
        for node in 0..nodes {
            let base = (row + node * ROUTE_WORDS as u32) as usize;
            config.railway.routes[base] = select(node == origin, 0, UNREACHABLE);
            config.railway.routes[base + 1] = u32::MAX;
            config.railway.routes[base + 2] = u32::MAX;
            config.railway.routes[base + 3] = 0;
            config.railway.routes[base + 4] = 0;
            config.railway.routes[base + 5] = select(node == origin, 0, UNREACHABLE);
        }
        let mut remaining = u32::cast_from(nodes);
        while remaining > 0 {
            let mut nearest = u32::cast_from(nodes);
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
                let start = config.railway.station_companies[(incident_start + nearest) as usize];
                let end = config.railway.station_companies[(incident_start + nearest + 1) as usize];
                for entry in start..end {
                    let edge = config.railway.station_companies[entry as usize];
                    let base = edge as usize * SEGMENT_WORDS;
                    let from = config.railway.segments[base];
                    let to = config.railway.segments[base + 1];
                    let owner = config.railway.segments[base + 2];
                    let firm = config.railway.companies[owner as usize * COMPANY_WORDS];
                    let operating = levels.number(firm);
                    let connected = config.railway.access_rights
                        [(carrier * config.railway.company_count + owner) as usize]
                        > 0
                        && (from == nearest || to == nearest)
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
                let traversal = traverse(
                    &*config,
                    &FreightRoute {
                        days: 0,
                        fee: 0,
                        origin,
                        destination: node,
                        carrier,
                        foreign_distance: 0,
                    },
                );
                config.railway.routes[base + 3] = traversal.fee;
                config.railway.routes[base + 4] =
                    max(min(traversal.visits, max(traversal.segments, 1)), 1);
            }
        }
    }
}

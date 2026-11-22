use super::*;

#[cube]
impl RailNetwork {
    pub(super) fn mark_dirty_carriers(&mut self, levels: &LedgerColumn<'_>) {
        let companies = self.company_count;
        let base = companies as usize * COMPANY_WORDS;
        let carrier = CUBE_POS as u32;
        let mut affected = self.companies[base] == comptime!(RouteCacheState::Unknown as u32);
        let mut owner = UNIT_POS;
        while owner < companies && !affected {
            let firm = self.companies[owner as usize * COMPANY_WORDS];
            let operating = levels.number(firm);
            let cached = self.companies[base + 2 + owner as usize];
            affected = cached != operating
                && self.access_rights[(carrier * companies + owner) as usize] > 0;
            owner += CUBE_DIM;
        }
        let changed =
            parallel::reduce::<u32>(u32::cast_from(affected), comptime!(Reduction::Maximum));
        if UNIT_POS == 0 && carrier < companies {
            self.companies[base + 2 + companies as usize + carrier as usize] = select(
                changed > 0,
                comptime!(RouteCacheState::Dirty as u32),
                comptime!(RouteCacheState::Clean as u32),
            );
        }
    }

    pub(super) fn prepare_routes(&mut self, levels: &LedgerColumn<'_>) {
        let companies = self.company_count;
        let base = companies as usize * COMPANY_WORDS;
        let mut changed = 0u32;
        let mut owner = UNIT_POS;
        while owner < companies {
            let firm = self.companies[owner as usize * COMPANY_WORDS];
            self.companies[base + 2 + owner as usize] = levels.number(firm);
            changed |= u32::cast_from(
                self.companies[base + 2 + companies as usize + owner as usize]
                    == comptime!(RouteCacheState::Dirty as u32),
            );
            owner += CUBE_DIM;
        }
        let any = parallel::reduce::<u32>(changed, comptime!(Reduction::Maximum));
        if UNIT_POS == 0 {
            self.companies[base] = select(
                any > 0,
                comptime!(RouteCacheState::Dirty as u32),
                comptime!(RouteCacheState::Clean as u32),
            );
            if any > 0 {
                self.companies[base + 1] = comptime!(RouteCacheState::Dirty as u32);
            }
        }
    }

    pub(super) fn complete_quotes(&mut self) {
        if ABSOLUTE_POS == 0 {
            let base = self.company_count as usize * COMPANY_WORDS;
            self.companies[base + 1] = comptime!(RouteCacheState::Clean as u32);
        }
    }

    pub(super) fn route_cell(&self, carrier: u32, origin: u32, destination: u32) -> usize {
        let nodes = self.nodes;
        let mut row = carrier * nodes + origin;
        if self.route_index != 0 {
            row = self.station_companies[(self.route_index + row) as usize];
        }
        (row * nodes + destination) as usize * ROUTE_WORDS
    }

    pub(super) fn rebuild_routes(&mut self, levels: &LedgerColumn<'_>) {
        let invocation = ABSOLUTE_POS as u32;
        let nodes = self.nodes;
        let mut pair = invocation;
        if self.route_index != 0 && invocation < self.route_rows {
            pair = self.station_companies
                [(self.route_index + nodes * self.company_count + invocation) as usize];
        }
        let carrier = pair / max(nodes, 1);
        let cache = self.company_count as usize * COMPANY_WORDS;
        let cached_carrier = min(carrier, self.company_count - 1);
        let dirty = self.companies
            [cache + 2 + self.company_count as usize + cached_carrier as usize]
            == comptime!(RouteCacheState::Dirty as u32);
        if invocation < self.route_rows && dirty {
            let origin = pair % nodes;
            let row = invocation * nodes * ROUTE_WORDS as u32;
            let incident_start = self.station_companies[nodes as usize];
            for node in 0..nodes {
                let base = (row + node * ROUTE_WORDS as u32) as usize;
                self.routes[base] = select(node == origin, 0, UNREACHABLE);
                self.routes[base + 1] = u32::MAX;
                self.routes[base + 2] = u32::MAX;
                self.routes[base + 3] = 0;
                self.routes[base + 4] = 0;
                self.routes[base + 5] = select(node == origin, 0, UNREACHABLE);
            }
            let mut remaining = u32::cast_from(nodes);
            while remaining > 0 {
                let mut nearest = u32::cast_from(nodes);
                let mut shortest = u32::new(comptime!(i64::from(UNREACHABLE)));
                let mut least_foreign = u32::new(comptime!(i64::from(UNREACHABLE)));
                for node in 0..nodes {
                    let base = (row + node * ROUTE_WORDS as u32) as usize;
                    let foreign = self.routes[base + 5];
                    let distance = self.routes[base];
                    let improve = self.routes[base + 4] == 0
                        && (foreign < least_foreign
                            || foreign == least_foreign && distance < shortest);
                    nearest = select(improve, node, nearest);
                    shortest = select(improve, distance, shortest);
                    least_foreign = select(improve, foreign, least_foreign);
                }
                remaining = select(nearest < nodes, remaining - 1, 0);
                if nearest < nodes {
                    self.routes[(row + nearest * ROUTE_WORDS as u32 + 4) as usize] = 1;
                    let start = self.station_companies[(incident_start + nearest) as usize];
                    let end = self.station_companies[(incident_start + nearest + 1) as usize];
                    for entry in start..end {
                        let edge = self.station_companies[entry as usize];
                        let base = edge as usize * SEGMENT_WORDS;
                        let from = self.segments[base];
                        let to = self.segments[base + 1];
                        let owner = self.segments[base + 2];
                        let firm = self.companies[owner as usize * COMPANY_WORDS];
                        let operating = levels.number(firm);
                        let connected =
                            self.access_rights[(carrier * self.company_count + owner) as usize] > 0
                                && (from == nearest || to == nearest)
                                && self.segments[base + 4] < operating
                                && (nearest != origin || owner == carrier);
                        let target = select(from == nearest, to, from);
                        let destination = (row + target * ROUTE_WORDS as u32) as usize;
                        let distance = shortest + self.segments[base + 3];
                        let foreign_owner = self.companies[owner as usize * COMPANY_WORDS + 3]
                            != self.companies[carrier as usize * COMPANY_WORDS + 3];
                        let foreign =
                            least_foreign + select(foreign_owner, self.segments[base + 3], 0);
                        let improve = connected
                            && (foreign < self.routes[destination + 5]
                                || foreign == self.routes[destination + 5]
                                    && distance < self.routes[destination]);
                        if improve {
                            self.routes[destination] = distance;
                            self.routes[destination + 1] = edge;
                            self.routes[destination + 2] = carrier;
                            self.routes[destination + 5] = foreign;
                        }
                    }
                }
            }
            for node in 0..nodes {
                let base = (row + node * ROUTE_WORDS as u32) as usize;
                if self.routes[base + 2] < self.company_count {
                    let traversal = FreightRoute {
                        days: 0,
                        fee: 0,
                        origin,
                        destination: node,
                        carrier,
                        foreign_distance: 0,
                    }
                    .traverse(&*self);
                    self.routes[base + 3] = traversal.fee;
                    self.routes[base + 4] =
                        max(min(traversal.visits, max(traversal.segments, 1)), 1);
                }
            }
        }
    }
}

mod topology;

use super::*;
use topology::NetworkTopology;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RouteCacheState {
    Unknown,
    Dirty,
    Clean,
}

#[derive(CubeType, CubeLaunch)]
pub struct RailNetwork {
    #[cube(comptime)]
    pub enabled: Feature,
    pub route_nodes: u32,
    pub nodes: u32,
    pub route_index: u32,
    pub route_rows: u32,
    pub company_count: u32,
    pub segment_count: u32,
    pub speed_multiplier: u32,
    pub length_unit: u32,
    pub stations: Box<[u32]>,
    pub access: Box<[u32]>,
    pub companies: Box<[u32]>,
    pub station_companies: Box<[u32]>,
    pub access_rights: Box<[u32]>,
    pub segments: Box<[u32]>,
    pub materials: Box<[u32]>,
    pub routes: Box<[u32]>,
    pub quotes: Box<[u32]>,
    pub regions: Box<[u32]>,
    pub segment_regions: Box<[u32]>,
    pub region_count: u32,
    pub region_base: u32,
    pub region_blocks: u32,
    pub initial_actors: u32,
    pub units_per_level: u32,
}

pub struct Network {
    availability: RouteAvailability,
    pub(super) freight: settlement::Buffers,
    pub(super) firms: u32,
    pub(super) nodes: u32,
    pub(super) route_index: u32,
    pub(super) route_rows: u32,
    pub(super) company_count: u32,
    pub(super) segment_count: u32,
    speed_multiplier: u32,
    pub(super) length_unit: u32,
    pub(super) stations: CubeTensor,
    pub(super) access: CubeTensor,
    pub(super) companies: CubeTensor,
    company_records: CompanyRecords,
    pub(super) station_companies: CubeTensor,
    pub(super) access_rights: CubeTensor,
    pub(super) segments: CubeTensor,
    pub(super) materials: CubeTensor,
    pub(super) routes: CubeTensor,
    pub(super) quotes: CubeTensor,
    pub(super) regions: CubeTensor,
    pub(super) segment_regions: CubeTensor,
    pub(super) region_count: u32,
    region_base: u32,
    region_blocks: u32,
    initial_actors: u32,
    units_per_level: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::simulation) enum RouteAvailability {
    Available,
    Dormant,
}

struct CompanyBuffers {
    companies: CubeTensor,
    access_rights: CubeTensor,
    records: CompanyRecords,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CompanyRecord {
    firm: u32,
    freight_rate: u32,
    access_fee: u32,
    founder: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CompanyRecords {
    companies: Vec<CompanyRecord>,
    identities: Vec<u64>,
}

impl CompanyRecords {
    fn new(scenario: &crate::domain::Scenario, state: &crate::domain::State) -> Self {
        let config = &scenario.config().railway;
        let company_identities = state.company_founders();
        let identities = config
            .companies
            .iter()
            .map(|company| {
                company_identities
                    .get(&company.firm)
                    .copied()
                    .unwrap_or(company.firm)
            })
            .collect::<Vec<_>>();
        let mut fallback = std::collections::BTreeMap::new();
        let mut site = |identity| {
            scenario
                .index_of(identity)
                .filter(|&row| state.actors.identities().get(row) == Some(&identity))
                .map(|row| row as u32)
                .unwrap_or_else(|| {
                    if fallback.is_empty() {
                        fallback.extend(
                            state
                                .actors
                                .firm_indices()
                                .map(|row| (state.actors.at(row).identity, row as u32)),
                        );
                    }
                    fallback[&identity]
                })
        };
        let companies = config
            .companies
            .iter()
            .zip(&identities)
            .map(|(company, identity)| CompanyRecord {
                firm: site(company.firm),
                freight_rate: company.freight_rate,
                access_fee: company.access_fee,
                founder: site(*identity),
            })
            .collect::<Vec<_>>();
        Self {
            companies,
            identities,
        }
    }
}

impl CompanyBuffers {
    fn new(device: &Device, scenario: &crate::domain::Scenario, records: CompanyRecords) -> Self {
        let config = &scenario.config().railway;
        let agreements = config
            .sharing_agreements
            .iter()
            .map(|agreement| (agreement.owner, agreement.carrier))
            .collect::<std::collections::BTreeSet<_>>();
        let identities = &records.identities;
        let agreements = &agreements;
        let access_rights = config
            .companies
            .iter()
            .enumerate()
            .flat_map(|(carrier_index, carrier)| {
                config
                    .companies
                    .iter()
                    .enumerate()
                    .map(move |(owner_index, owner)| {
                        u32::from(
                            owner.sharing == crate::domain::railway::RailSharing::Shared
                                || identities[owner_index] == identities[carrier_index]
                                || agreements.contains(&(owner.firm, carrier.firm)),
                        )
                    })
            })
            .collect::<Vec<_>>();
        Self {
            companies: device.tensor(
                &records
                    .companies
                    .iter()
                    .flat_map(|company| {
                        [
                            company.firm,
                            company.freight_rate,
                            company.access_fee,
                            company.founder,
                        ]
                    })
                    .chain([
                        RouteCacheState::Unknown as u32,
                        RouteCacheState::Unknown as u32,
                    ])
                    .chain(std::iter::repeat_n(0, records.companies.len()))
                    .chain(std::iter::repeat_n(
                        RouteCacheState::Unknown as u32,
                        records.companies.len(),
                    ))
                    .collect::<Vec<_>>(),
            ),
            access_rights: device.tensor(&access_rights),
            records,
        }
    }
}

impl Network {
    fn station_access(device: &Device, actors: usize) -> CubeTensor {
        let capacity = crate::simulation::registry::entries::state_capacity(actors);
        device.tensor(&vec![0; 4 * capacity])
    }

    pub(in crate::simulation) fn operating_counts(
        &self,
        device: &Device,
    ) -> anyhow::Result<Vec<u32>> {
        device.read_range(
            &self.companies,
            self.company_count as usize * COMPANY_WORDS + 2,
            self.company_count as usize,
        )
    }

    pub(in crate::simulation) fn route_availability(&self) -> RouteAvailability {
        self.availability
    }

    pub(in crate::simulation) fn replace_route_availability(
        &mut self,
        availability: RouteAvailability,
    ) -> RouteAvailability {
        std::mem::replace(&mut self.availability, availability)
    }

    pub(in crate::simulation) fn graph_buffers(&self) -> impl Iterator<Item = &CubeTensor> + Clone {
        [
            &self.stations,
            &self.access,
            &self.companies,
            &self.station_companies,
            &self.access_rights,
            &self.segments,
            &self.materials,
            &self.routes,
            &self.quotes,
            &self.regions,
            &self.segment_regions,
        ]
        .into_iter()
        .chain(self.freight.graph_buffers())
    }

    pub fn new(
        device: &Device,
        scenario: &crate::domain::Scenario,
        state: &crate::domain::State,
    ) -> Self {
        let stations = scenario.config().railway.stations();
        let storage = scenario.config().railway.route_storage(stations.len());
        Self::with_stations(device, scenario, state, &stations, storage)
    }

    pub(super) fn with_stations(
        device: &Device,
        scenario: &crate::domain::Scenario,
        state: &crate::domain::State,
        stations: &[crate::domain::Position],
        storage: RouteStorage,
    ) -> Self {
        let config = &scenario.config().railway;
        let geography = scenario.rail_geography();
        let dynamic = CompanyBuffers::new(device, scenario, CompanyRecords::new(scenario, state));
        let topology = NetworkTopology::new(config, geography, stations);
        let tables = topology.route_tables(device, config.companies.len(), storage);
        Self {
            availability: RouteAvailability::Available,
            firms: state.actors.firms() as u32,
            freight: settlement::Buffers::new(
                device,
                match config.companies.is_empty() {
                    true => 0,
                    false => state.actors.len(),
                },
                geography.regions.len(),
            ),
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
            access: Self::station_access(device, state.actors.len()),
            companies: dynamic.companies,
            company_records: dynamic.records,
            station_companies: device.tensor(&tables.offsets),
            access_rights: dynamic.access_rights,
            segments: device.tensor(&topology.segments),
            materials: device.tensor(
                &config
                    .construction
                    .quantities()
                    .into_iter()
                    .chain(config.maintenance.quantities())
                    .collect::<Vec<_>>(),
            ),
            routes: tables.routes,
            route_index: tables.index as u32,
            route_rows: tables.rows as u32,
            quotes: device.empty(
                (stations.len() * stations.len() * scenario.ring()
                    + usize::from(tables.index != 0) * stations.len())
                .max(1),
            ),
            regions: device.tensor(&topology.regions),
            segment_regions: device.tensor(&topology.segment_regions),
            region_count: geography.regions.len() as u32,
            region_base: (BASE_FIELDS + 2 * scenario.ring() * Good::COUNT) as u32,
            region_blocks: geography.blocks(scenario.actors()) as u32,
            initial_actors: scenario.actors() as u32,
            units_per_level: config.units_per_level,
        }
    }

    pub fn updated(
        &self,
        device: &Device,
        scenario: &crate::domain::Scenario,
        state: &crate::domain::State,
    ) -> Self {
        let records = CompanyRecords::new(scenario, state);
        let dynamic = match records == self.company_records {
            true => CompanyBuffers {
                companies: self.companies.clone(),
                access_rights: self.access_rights.clone(),
                records,
            },
            false => CompanyBuffers::new(device, scenario, records),
        };
        let access = match self.access.meta.num_elements() >= state.actors.len() * 4 {
            true => self.access.clone(),
            false => Self::station_access(device, state.actors.len()),
        };
        Self {
            availability: RouteAvailability::Available,
            firms: state.actors.firms() as u32,
            freight: self.freight.updated(
                device,
                match self.company_count {
                    0 => 0,
                    _ => state.actors.len(),
                },
                self.region_count as usize,
            ),
            nodes: self.nodes,
            route_index: self.route_index,
            route_rows: self.route_rows,
            company_count: self.company_count,
            segment_count: self.segment_count,
            speed_multiplier: self.speed_multiplier,
            length_unit: self.length_unit,
            stations: self.stations.clone(),
            access,
            companies: dynamic.companies,
            company_records: dynamic.records,
            station_companies: self.station_companies.clone(),
            access_rights: dynamic.access_rights,
            segments: self.segments.clone(),
            materials: self.materials.clone(),
            routes: self.routes.clone(),
            quotes: self.quotes.clone(),
            regions: self.regions.clone(),
            segment_regions: self.segment_regions.clone(),
            region_count: self.region_count,
            region_base: self.region_base,
            region_blocks: self.region_blocks,
            initial_actors: self.initial_actors,
            units_per_level: self.units_per_level,
        }
    }

    pub fn launch(&self) -> RailNetworkLaunch {
        RailNetworkLaunch {
            enabled: match self.company_count {
                0 => Feature::Disabled,
                _ => Feature::Enabled,
            },
            route_nodes: match self.availability {
                RouteAvailability::Available => self.nodes,
                RouteAvailability::Dormant => 0,
            },
            nodes: self.nodes,
            route_index: self.route_index,
            route_rows: self.route_rows,
            company_count: self.company_count,
            segment_count: self.segment_count,
            speed_multiplier: self.speed_multiplier,
            length_unit: self.length_unit,
            stations: self.stations.buffer_arg(),
            access: self.access.buffer_arg(),
            companies: self.companies.buffer_arg(),
            station_companies: self.station_companies.buffer_arg(),
            access_rights: self.access_rights.buffer_arg(),
            segments: self.segments.buffer_arg(),
            materials: self.materials.buffer_arg(),
            routes: self.routes.buffer_arg(),
            quotes: self.quotes.buffer_arg(),
            regions: self.regions.buffer_arg(),
            segment_regions: self.segment_regions.buffer_arg(),
            region_count: self.region_count,
            region_base: self.region_base,
            region_blocks: self.region_blocks,
            initial_actors: self.initial_actors,
            units_per_level: self.units_per_level,
        }
    }
}

use super::*;
use crate::domain::railway::{RAIL_ROUTE_WORDS as ROUTE_WORDS, RailCapacityField, RouteStorage};
use crate::simulation::{Device, configuration::Configuration};
use burn_wgpu::CubeTensor;

#[cfg(test)]
mod tests;

pub(in crate::simulation) mod settlement;

const COMPANY_WORDS: usize = 4;
const SEGMENT_WORDS: usize = 7;
const REGION_WORDS: usize = 3;
const UNREACHABLE: u32 = u32::MAX / 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RouteCacheState {
    Unknown,
    Dirty,
    Clean,
}

#[derive(CubeType, CubeLaunch)]
pub struct RailNetwork {
    #[cube(comptime)]
    pub enabled: Feature,
    #[cube(comptime)]
    pub routes_enabled: Feature,
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
    freight: settlement::Buffers,
    firms: u32,
    nodes: u32,
    route_index: u32,
    route_rows: u32,
    company_count: u32,
    segment_count: u32,
    speed_multiplier: u32,
    length_unit: u32,
    stations: CubeTensor,
    access: CubeTensor,
    companies: CubeTensor,
    company_records: CompanyRecords,
    station_companies: CubeTensor,
    access_rights: CubeTensor,
    segments: CubeTensor,
    materials: CubeTensor,
    routes: CubeTensor,
    quotes: CubeTensor,
    regions: CubeTensor,
    segment_regions: CubeTensor,
    region_count: u32,
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

    fn with_stations(
        device: &Device,
        scenario: &crate::domain::Scenario,
        state: &crate::domain::State,
        stations: &[crate::domain::Position],
        storage: RouteStorage,
    ) -> Self {
        let config = &scenario.config().railway;
        let geography = scenario.rail_geography();
        let dynamic = CompanyBuffers::new(device, scenario, CompanyRecords::new(scenario, state));
        let mut segment_regions = Vec::new();
        let mut station_companies = vec![std::collections::BTreeSet::new(); stations.len()];
        let mut incident_segments = vec![Vec::new(); stations.len()];
        let mut ordinals = vec![0; config.companies.len()];
        let mut first_segments = vec![u32::MAX; geography.regions.len()];
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
                for &region in &geography.segments[index] {
                    first_segments[region] = first_segments[region].min(ordinal);
                }
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
                incident_segments[from].push(index as u32);
                incident_segments[to].push(index as u32);
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
        let carriers = station_companies
            .iter()
            .map(|companies| companies.len())
            .sum::<usize>();
        let mut offsets =
            Vec::with_capacity(2 * (stations.len() + 1) + carriers + 2 * config.segments.len());
        offsets.push(stations.len() as u32 + 1);
        offsets.extend(
            station_companies
                .iter()
                .scan(stations.len() + 1, |offset, companies| {
                    *offset += companies.len();
                    Some(*offset as u32)
                }),
        );
        offsets.extend(station_companies.iter().flatten().copied());
        let incident_start = offsets.len() + stations.len() + 1;
        offsets.push(incident_start as u32);
        offsets.extend(
            incident_segments
                .iter()
                .scan(incident_start, |offset, segments| {
                    *offset += segments.len();
                    Some(*offset as u32)
                }),
        );
        offsets.extend(incident_segments.into_iter().flatten());
        let dense = stations.len() * config.companies.len();
        let (route_index, route_rows, routes) = match storage {
            RouteStorage::Indexed => {
                let rows = carriers;
                let index = offsets.len();
                let stations_count = stations.len() as u32;
                let origins = station_companies
                    .iter()
                    .enumerate()
                    .flat_map(|(origin, owners)| {
                        owners
                            .iter()
                            .map(move |&company| company * stations_count + origin as u32)
                    })
                    .collect::<Vec<_>>();
                let mut lookup = vec![rows as u32; dense];
                for (row, &pair) in origins.iter().enumerate() {
                    lookup[pair as usize] = row as u32;
                }
                offsets.extend(lookup);
                offsets.extend(origins);
                let empty = [UNREACHABLE, u32::MAX, u32::MAX, 0, 0, UNREACHABLE];
                (
                    index,
                    rows,
                    device.tensor(&empty.repeat((rows + 1) * stations.len())),
                )
            }
            RouteStorage::Dense => (
                0,
                dense,
                device.empty((dense * stations.len() * ROUTE_WORDS).max(1)),
            ),
        };
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
            station_companies: device.tensor(&offsets),
            access_rights: dynamic.access_rights,
            segments: device.tensor(&segments),
            materials: device.tensor(
                &config
                    .construction
                    .quantities()
                    .into_iter()
                    .chain(config.maintenance.quantities())
                    .collect::<Vec<_>>(),
            ),
            routes,
            route_index: route_index as u32,
            route_rows: route_rows as u32,
            quotes: device.empty(
                (stations.len() * stations.len() * scenario.ring()
                    + usize::from(route_index != 0) * stations.len())
                .max(1),
            ),
            regions: device.tensor(
                &geography
                    .regions
                    .iter()
                    .enumerate()
                    .flat_map(|(index, region)| {
                        [
                            region.company as u32,
                            region.region as u32,
                            first_segments[index],
                        ]
                    })
                    .collect::<Vec<_>>(),
            ),
            segment_regions: device.tensor(&segment_regions),
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
            routes_enabled: match (self.company_count, self.availability) {
                (0, _) | (_, RouteAvailability::Dormant) => Feature::Disabled,
                _ => Feature::Enabled,
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

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum RailStage {
    Infrastructure,
    DormantDemand,
    Access,
    Routes,
    Quotes,
    Close,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum RailPass {
    Infrastructure,
    DormantDemand,
    Access,
    DirtyCarriers,
    PrepareRoutes,
    Routes,
    Quotes,
    Inbound,
    CompleteQuotes,
    Close,
}

#[derive(CubeType, CubeLaunch)]
struct RailInput {
    state: Box<[u32]>,
    actors: Box<[u32]>,
    config: EconomyConfig,
    control: Box<[Atomic<u32>]>,
    #[cube(comptime)]
    stage: RailPass,
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
        let levels = LedgerColumn::new(&input.state, &input.config, Field::RailOperating);
        match comptime!(input.stage) {
            RailPass::Infrastructure => infrastructure(
                &mut EconomyView::new(&mut input.state, &input.actors, &input.config),
                &input.control,
            ),
            RailPass::DormantDemand => {
                dormant_demand(&mut Ledger::new(&mut input.state, &input.config))
            }
            RailPass::Access => access(&ActorTable::new(&input.actors), &mut input.config),
            RailPass::DirtyCarriers => dirty_carriers(&levels, &mut input.config.railway),
            RailPass::PrepareRoutes => prepare_routes(&levels, &mut input.config.railway),
            RailPass::Routes => routes(&levels, &mut input.config),
            RailPass::Quotes => quotes(&mut input.config),
            RailPass::Inbound => inbound(&mut input.config),
            RailPass::CompleteQuotes => complete_quotes(&mut input.config),
            RailPass::Close => close(
                &mut Ledger::new(&mut input.state, &input.config),
                &input.control,
            ),
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
        let passes: &[RailPass] = match stage {
            RailStage::Infrastructure => &[RailPass::Infrastructure],
            RailStage::DormantDemand => &[RailPass::DormantDemand],
            RailStage::Access => &[RailPass::Access],
            RailStage::Routes => &[
                RailPass::DirtyCarriers,
                RailPass::PrepareRoutes,
                RailPass::Routes,
            ],
            RailStage::Quotes if config.railway.route_index != 0 => &[
                RailPass::Quotes,
                RailPass::Inbound,
                RailPass::CompleteQuotes,
            ],
            RailStage::Quotes => &[RailPass::Quotes, RailPass::CompleteQuotes],
            RailStage::Close => &[RailPass::Close],
        };
        for &stage in passes {
            let count = match stage {
                RailPass::Infrastructure => config.railway.company_count * device.workgroup,
                RailPass::Close => config.railway.company_count,
                RailPass::DormantDemand => config.railway.region_count,
                RailPass::Access => config.actors,
                RailPass::DirtyCarriers => config.railway.company_count * device.workgroup,
                RailPass::Routes => config.railway.route_rows,
                RailPass::Inbound => config.railway.nodes,
                RailPass::Quotes => config.railway.nodes * config.railway.nodes * config.ring,
                RailPass::PrepareRoutes | RailPass::CompleteQuotes => 1,
            };
            RailSystem::launch_elements(
                device,
                count as usize,
                RailInputLaunch {
                    state: state.buffer_arg(),
                    actors: actors.buffer_arg(),
                    config: config.launch(),
                    control: control.buffer_arg(),
                    stage,
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
fn supplied(ledger: &Ledger<'_>, firm: u32, units: u32, offset: u32) -> bool {
    let config = ledger.config;
    let mut funded = true;
    for index in 0..3u32 {
        let required = times32(units, config.railway.materials[(offset + index) as usize]);
        funded = funded && !lt(ledger.value_at(material(index), firm), required);
    }
    funded
}

#[cube]
fn consume(ledger: &mut Ledger<'_>, control: &[Atomic<u32>], firm: u32, units: u32, offset: u32) {
    let config = ledger.config;
    for index in 0..3u32 {
        let good = material(index);
        let required = times32(units, config.railway.materials[(offset + index) as usize]);
        let remaining = minus(ledger.value_at(good, firm), required);
        ledger.set_at(good, firm, remaining);
        ledger.increase_at(control, USED_FIELDS as u32 + good - 1, firm, required);
        ledger.increase(control, Field::Used, firm, required);
    }
}

#[cube]
pub(super) fn offer_crew_jobs(economy: &mut EconomyView<'_>, control: &[Atomic<u32>], firm: u32) {
    let config = economy.ledger.config;
    let built = economy.ledger.number(Field::RailBuilt, firm);
    let mut segments = 0;
    for index in 0..config.railway.segment_count {
        let company = config.railway.segments[index as usize * SEGMENT_WORDS + 2];
        segments +=
            u32::cast_from(config.railway.companies[company as usize * COMPANY_WORDS] == firm);
    }
    let minutes = economy.ledger.number(Field::LaborMinutes, firm);
    let cart_crews = 1 + ceildiv(
        economy.ledger.number(Field::CartageMinutes, firm),
        max(minutes, 1),
    );
    let requested = cart_crews
        + ceildiv(built, comptime!(crate::domain::railway::SEGMENTS_PER_CREW))
        + u32::cast_from(segments > built);
    let limit = economy.actors.capacity(firm);
    let budget = employment::ShiftBudget::new(&economy.ledger, control, firm);
    let funded = divide(budget.cash, max(budget.pay, 1));
    let operating = economy.ledger.number(Field::EnterpriseStatus, firm)
        == comptime!(crate::domain::enterprise::EnterpriseState::Operating as u32);
    let jobs = minimum(
        funded,
        ticks(select(operating, min(requested, limit), 0), 0),
    );
    economy.ledger.set(Field::Plan, firm, jobs);
    economy.ledger.set(Field::Slots, firm, jobs);
}

#[cube]
fn infrastructure(economy: &mut EconomyView<'_>, control: &[Atomic<u32>]) {
    let config = economy.ledger.config;
    let company = CUBE_POS as u32;
    if company < config.railway.company_count {
        let firm = config.railway.companies[company as usize * COMPANY_WORDS];
        let built = economy.ledger.number(Field::RailBuilt, firm);
        let mut units = 0u32;
        let mut next = 0u32;
        let mut edge = UNIT_POS;
        while edge < config.railway.segment_count {
            let base = edge as usize * SEGMENT_WORDS;
            let own = config.railway.segments[base + 2] == company;
            let ordinal = config.railway.segments[base + 4];
            let length = ceildiv(
                config.railway.segments[base + 3],
                config.railway.length_unit,
            );
            units += select(own && ordinal < built, length, 0);
            next += select(own && ordinal == built, length, 0);
            edge += CUBE_DIM;
        }
        let mut region = UNIT_POS;
        while region < config.railway.region_count {
            if config.railway.regions[region as usize * REGION_WORDS] == company {
                let level =
                    regional_number(&economy.ledger, region, comptime!(RailCapacityField::Level));
                units += max(level, 1) - 1;
            }
            region += CUBE_DIM;
        }
        units = parallel::reduce::<u32>(units, comptime!(Reduction::Sum));
        next = parallel::reduce::<u32>(next, comptime!(Reduction::Sum));
        let mut built_count = Shared::<u32>::new();
        let mut service = Shared::<u32>::new();
        let mut budget = Shared::<u32>::new();
        if UNIT_POS == 0 {
            let active = economy.ledger.number(Field::EnterpriseStatus, firm)
                == comptime!(crate::domain::enterprise::EnterpriseState::Operating as u32);
            let crew = economy.ledger.number(Field::Plan, firm)
                - economy.ledger.number(Field::Slots, firm);
            let upkeep_crew = ceildiv(built, comptime!(crate::domain::railway::SEGMENTS_PER_CREW));
            let maintained =
                active && crew >= max(upkeep_crew, 1) && supplied(&economy.ledger, firm, units, 3);
            if maintained {
                consume(&mut economy.ledger, control, firm, units, 3);
            }
            let build = maintained
                && crew > upkeep_crew
                && next > 0
                && supplied(&economy.ledger, firm, next, 0);
            if build {
                consume(&mut economy.ledger, control, firm, next, 0);
            }
            let rail_crews = select(maintained, upkeep_crew, 0) + u32::cast_from(build);
            let shift = economy.ledger.number(Field::LaborMinutes, firm);
            economy
                .ledger
                .set_number(Field::CartageCapacity, firm, (crew - rail_crews) * shift);
            *built_count = built + u32::cast_from(build);
            *service = u32::cast_from(maintained);
            let mut affordable = ticks(config.railway.region_count, 0);
            for index in 0..3u32 {
                affordable = minimum(
                    affordable,
                    divide(
                        economy.ledger.value_at(material(index), firm),
                        config.railway.materials[index as usize],
                    ),
                );
            }
            *budget = affordable.extract(LOW_WORD);
            economy
                .ledger
                .set_number(Field::RailBuilt, firm, *built_count);
            economy.ledger.set_number(
                Field::RailOperating,
                firm,
                select(maintained, *built_count, 0),
            );
            let expired = economy.ledger.value(Field::Railway, firm);
            record_use(&mut economy.ledger, control, firm, expired);
        }
        sync_storage();
        let expanding = enterprises::expands(&*economy, firm);
        let maintained = *service != 0;
        let mut remaining = *budget;
        let mut upgrades = 0u32;
        let mut capacity = 0u64;
        let mut cursor = 0u32;
        while cursor < config.railway.region_count {
            let region = cursor + UNIT_POS;
            let mut own = false;
            let mut level = 0u32;
            let mut requested = 0u32;
            if region < config.railway.region_count {
                let base = region as usize * REGION_WORDS;
                own = config.railway.regions[base] == company;
                if own {
                    level = max(
                        regional_number(
                            &economy.ledger,
                            region,
                            comptime!(RailCapacityField::Level),
                        ),
                        u32::cast_from(config.railway.regions[base + 2] < *built_count),
                    );
                    let demand = regional_value(
                        &economy.ledger,
                        region,
                        comptime!(RailCapacityField::Demand),
                    );
                    requested = u32::cast_from(
                        maintained
                            && expanding
                            && level > 0
                            && level < 1000
                            && lt(times32(level, config.railway.units_per_level), demand),
                    );
                }
            }
            let prefix = parallel::prefix_sum::<u32>(requested);
            level += u32::cast_from(requested > 0 && prefix.inclusive <= remaining);
            let applied = min(remaining, prefix.total);
            remaining -= applied;
            upgrades += applied;
            if own {
                let units = level * config.railway.units_per_level;
                regional_set(
                    &mut economy.ledger,
                    region,
                    ticks(level, 0),
                    comptime!(RailCapacityField::Level),
                );
                regional_set(
                    &mut economy.ledger,
                    region,
                    ticks(select(maintained, units, 0), 0),
                    comptime!(RailCapacityField::Available),
                );
                capacity += u64::cast_from(units);
            }
            cursor += CUBE_DIM;
        }
        let total = parallel::sum_wide(capacity);
        if UNIT_POS == 0 {
            consume(&mut economy.ledger, control, firm, upgrades, 0);
            let capacity = ticks(
                u32::cast_from(total.value),
                u32::cast_from(total.value >> 32),
            );
            let production = select(maintained, capacity, zero());
            economy.ledger.set(Field::RailCapacity, firm, capacity);
            economy.ledger.set(Field::Railway, firm, production);
            economy
                .ledger
                .increase(control, Field::Produced, firm, production);
            economy
                .ledger
                .increase(control, Field::UnscaledProduced, firm, production);
            economy.ledger.set_number(Field::RailRevenue, firm, 0);
        }
    }
}

#[cube]
pub(super) fn demand(economy: &mut EconomyView<'_>, firm: u32, good: u32) -> Ticks {
    let config = economy.ledger.config;
    let mut wanted = zero();
    let requested_material = good == comptime!(Good::Coal as u32)
        || good == comptime!(Good::Steel as u32)
        || good == comptime!(Good::Locomotives as u32);
    if comptime!(config.railway.enabled == Feature::Enabled)
        && requested_material
        && economy.actors.kind(firm).bits() == comptime!(Good::Railway as u32)
    {
        for company in 0..config.railway.company_count {
            if config.railway.companies[company as usize * COMPANY_WORDS] == firm {
                let built = economy.ledger.number(Field::RailBuilt, firm);
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
                let expanding = enterprises::expands(&*economy, firm);
                for region in 0..config.railway.region_count {
                    if config.railway.regions[region as usize * REGION_WORDS] == company {
                        let level = regional_number(
                            &economy.ledger,
                            region,
                            comptime!(RailCapacityField::Level),
                        );
                        let demand = regional_value(
                            &economy.ledger,
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
fn dormant_demand(ledger: &mut Ledger<'_>) {
    let config = ledger.config;
    let region = ABSOLUTE_POS as u32;
    if region < config.railway.region_count {
        let units = times32(
            regional_number(&*ledger, region, comptime!(RailCapacityField::Level)),
            config.railway.units_per_level,
        );
        let available = regional_value(&*ledger, region, comptime!(RailCapacityField::Available));
        let demand = regional_value(&*ledger, region, comptime!(RailCapacityField::Demand));
        regional_set(
            ledger,
            region,
            maximum(demand, minus(units, available)),
            comptime!(RailCapacityField::Demand),
        );
    }
}

#[cube]
fn dirty_carriers(levels: &LedgerColumn<'_>, railway: &mut RailNetwork) {
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
fn prepare_routes(levels: &LedgerColumn<'_>, railway: &mut RailNetwork) {
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
fn complete_quotes(config: &mut EconomyConfig) {
    if ABSOLUTE_POS == 0 {
        let base = config.railway.company_count as usize * COMPANY_WORDS;
        config.railway.companies[base + 1] = comptime!(RouteCacheState::Clean as u32);
    }
}

#[cube]
fn route_cell(config: &EconomyConfig, carrier: u32, origin: u32, destination: u32) -> usize {
    let nodes = config.railway.nodes;
    let mut row = carrier * nodes + origin;
    if config.railway.route_index != 0 {
        row = config.railway.station_companies[(config.railway.route_index + row) as usize];
    }
    (row * nodes + destination) as usize * ROUTE_WORDS
}

#[cube]
fn routes(levels: &LedgerColumn<'_>, config: &mut EconomyConfig) {
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
fn station(actors: &ActorTable<'_>, config: &EconomyConfig, actor: u32) -> u32 {
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
fn access(actors: &ActorTable<'_>, config: &mut EconomyConfig) {
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
fn access_distance(actors: &ActorTable<'_>, config: &EconomyConfig, actor: u32, node: u32) -> u32 {
    let x = actors.x(actor);
    let y = actors.y(actor);
    let nx = config.railway.stations[node as usize * 2];
    let ny = config.railway.stations[node as usize * 2 + 1];
    max(x, nx) - min(x, nx) + max(y, ny) - min(y, ny)
}

#[cube]
pub(super) fn animal_route(
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
fn candidate(
    actors: &ActorTable<'_>,
    config: &EconomyConfig,
    buyer: u32,
    seller: u32,
    company: u32,
) -> FreightRoute {
    FreightRequest::new(actors, config, buyer, seller).candidate(config, company)
}

#[derive(CubeType)]
struct FreightRequest {
    animal_days: u32,
    origin: u32,
    destination: u32,
    access_days: u32,
}

#[cube]
impl FreightRequest {
    fn new(actors: &ActorTable<'_>, config: &EconomyConfig, buyer: u32, seller: u32) -> Self {
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

    fn animal(&self) -> FreightRoute {
        FreightRoute {
            days: self.animal_days,
            fee: 0,
            origin: 0,
            destination: 0,
            carrier: u32::MAX,
            foreign_distance: u32::MAX,
        }
    }

    fn candidate(&self, config: &EconomyConfig, company: u32) -> FreightRoute {
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

    fn quoted_carrier(&self, config: &EconomyConfig) -> u32 {
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
fn inbound(config: &mut EconomyConfig) {
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
pub(super) fn serves_buyer(config: &EconomyConfig, buyer: u32) -> bool {
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
fn quotes(config: &mut EconomyConfig) {
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
pub(super) fn route(
    actors: &ActorTable<'_>,
    config: &EconomyConfig,
    buyer: u32,
    seller: u32,
) -> FreightRoute {
    FreightRequest::new(actors, config, buyer, seller).preferred(config)
}

#[derive(CubeType, Clone, Copy)]
pub(super) struct DeliveryRequest {
    pub buyer: u32,
    pub seller: u32,
    pub trip: u32,
    pub buyer_access: u32,
}

#[cube]
pub(super) fn may_book(
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
pub(super) fn delivery_days(config: &EconomyConfig, request: DeliveryRequest) -> u32 {
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
fn regional_field(
    config: &EconomyConfig,
    region: u32,
    #[comptime] field: RailCapacityField,
) -> u32 {
    config.railway.region_base
        + comptime!(field as u32) * config.railway.region_blocks
        + region / config.railway.initial_actors
}

#[cube]
fn regional_value(ledger: &Ledger<'_>, region: u32, #[comptime] field: RailCapacityField) -> Ticks {
    let config = ledger.config;
    ledger.value_at(
        regional_field(config, region, comptime!(field)),
        region % config.railway.initial_actors,
    )
}

#[cube]
fn regional_number(ledger: &Ledger<'_>, region: u32, #[comptime] field: RailCapacityField) -> u32 {
    regional_value(ledger, region, comptime!(field)).extract(LOW_WORD)
}

#[cube]
fn regional_set(
    ledger: &mut Ledger<'_>,
    region: u32,
    amount: Ticks,
    #[comptime] field: RailCapacityField,
) {
    let config = ledger.config;
    ledger.set_at(
        regional_field(config, region, comptime!(field)),
        region % config.railway.initial_actors,
        amount,
    );
}

#[cube]
fn access_fare(config: &EconomyConfig, owner: u32, carrier: u32, regions: u32) -> u32 {
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
struct RouteTraversal {
    fee: u32,
    visits: u32,
    segments: u32,
}

#[cube]
fn traverse(config: &EconomyConfig, path: &FreightRoute) -> RouteTraversal {
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

#[cube]
fn record_use(ledger: &mut Ledger<'_>, control: &[Atomic<u32>], actor: u32, units: Ticks) {
    ledger.increase_at(
        control,
        USED_FIELDS as u32 + comptime!(Good::Railway as u32) - 1,
        actor,
        units,
    );
    ledger.increase(control, Field::Used, actor, units);
}

#[cube]
fn close(ledger: &mut Ledger<'_>, control: &[Atomic<u32>]) {
    let config = ledger.config;
    let company = ABSOLUTE_POS as u32;
    if company < config.railway.company_count {
        let firm = config.railway.companies[company as usize * COMPANY_WORDS];
        let status = ledger.number(Field::EnterpriseStatus, firm);
        if status >= comptime!(crate::domain::enterprise::EnterpriseState::Liquidating as u32) {
            let expired = ledger.value(Field::Railway, firm);
            record_use(ledger, control, firm, expired);
            ledger.set_number(Field::Railway, firm, 0);
            ledger.set_number(Field::RailOperating, firm, 0);
            for region in 0..config.railway.region_count {
                if config.railway.regions[region as usize * REGION_WORDS] == company {
                    regional_set(
                        ledger,
                        region,
                        zero(),
                        comptime!(RailCapacityField::Available),
                    );
                }
            }
        }
    }
}

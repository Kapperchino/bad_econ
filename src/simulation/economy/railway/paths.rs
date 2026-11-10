use super::*;

#[derive(CubeType, CubeLaunch)]
struct PathInput {
    config: EconomyConfig,
}

struct PathSystem;
struct PreparePaths;

impl System for PathSystem {
    type Input = PathInput;
    type Passes = gpu_passes![PreparePaths];
}

pub(super) fn launch(device: &Device, config: &Configuration) {
    let entries =
        config.railway.station_companies.meta.num_elements() - config.railway.nodes as usize - 1;
    if config.railway.request_storage != RequestStorage::Global
        && config.railway.path_cache == Feature::Enabled
        && entries > 0
    {
        PathSystem::launch(
            device,
            device.count(entries * device.workgroup as usize),
            CubeDim::new_1d(device.workgroup),
            PathInputLaunch {
                config: config.launch(),
            },
        );
    }
}

#[cube]
impl GpuPass<PathInput> for PreparePaths {
    type Output = ();

    fn execute(mut input: PathInput) {
        let config = &mut input.config;
        let entry = config.railway.nodes + 1 + CUBE_POS as u32;
        if entry < config.railway.station_companies.len() as u32 {
            let mut requests = RequestLists::new(comptime!(RequestStorage::Shared));
            requests.initialize(&*config);
            if UNIT_POS == 0 {
                let mut origin = 0u32;
                let mut end = config.railway.nodes;
                while origin < end {
                    let middle = (origin + end) / 2;
                    if config.railway.station_companies[(middle + 1) as usize] <= entry {
                        origin = middle + 1;
                    } else {
                        end = middle;
                    }
                }
                let carrier = config.railway.station_companies[entry as usize];
                for destination in 0..config.railway.nodes {
                    let route = ((carrier * config.railway.nodes + origin) * config.railway.nodes
                        + destination) as usize
                        * ROUTE_WORDS;
                    if config.railway.routes[route + 2] < config.railway.company_count {
                        let path = FreightRoute {
                            days: 0,
                            fee: config.railway.routes[route + 3],
                            origin,
                            destination,
                            carrier,
                            foreign_distance: config.railway.routes[route + 5],
                        };
                        prepare_request_with_lists(&*config, &path, &mut requests);
                        let region_count = requests.region_count(&*config);
                        let access_count = requests.access_count(&*config);
                        if region_count + access_count <= REQUEST_PATH_ENTRIES as u32 {
                            let row = CUBE_POS as u32 * config.railway.nodes + destination;
                            let base = payload_base(&*config, row);
                            config.railway.routes[base] = region_count;
                            config.railway.routes[base + 1] = access_count;
                            for index in 0..region_count {
                                let request = requests.region_entry(&*config, index);
                                config.railway.routes[base + 2 + 2 * index as usize] =
                                    request.record;
                                config.railway.routes[base + 3 + 2 * index as usize] =
                                    request.visits;
                            }
                            for index in 0..access_count {
                                let request = requests.access_entry(&*config, index);
                                let pair = base + 2 + 2 * (region_count + index) as usize;
                                config.railway.routes[pair] = request.owner;
                                config.railway.routes[pair + 1] = request.units;
                            }
                            config.railway.routes[route + 4] = row + 1;
                        }
                    }
                }
            }
        }
    }
}

#[cube]
fn payload_base(config: &EconomyConfig, row: u32) -> usize {
    (config.railway.company_count * config.railway.nodes * config.railway.nodes) as usize
        * ROUTE_WORDS
        + row as usize * REQUEST_PATH_WORDS
}

#[cube]
pub(super) fn restore(
    config: &EconomyConfig,
    path: &FreightRoute,
    requests: &mut RequestLists,
) -> bool {
    let route = ((path.carrier * config.railway.nodes + path.origin) * config.railway.nodes
        + path.destination) as usize
        * ROUTE_WORDS;
    let row = config.railway.routes[route + 4];
    if row > 0 {
        let base = payload_base(config, row - 1);
        if !requests.bind_path(base) {
            let region_count = config.railway.routes[base];
            let access_count = config.railway.routes[base + 1];
            let regions = config.railway.region_count;
            let companies = config.railway.company_count;
            for index in 0..region_count {
                let pair = base + 2 + 2 * index as usize;
                let record = config.railway.routes[pair];
                requests.set_region(config, record as usize, config.railway.routes[pair + 1]);
                requests.set_region(config, (regions + index) as usize, record);
            }
            requests.set_region(config, (2 * regions) as usize, region_count);
            for index in 0..access_count {
                let pair = base + 2 + 2 * (region_count + index) as usize;
                let owner = config.railway.routes[pair];
                requests.set_access(config, owner as usize, config.railway.routes[pair + 1]);
                requests.set_access(config, (companies + index) as usize, owner);
            }
            requests.set_access(config, (2 * companies) as usize, access_count);
        }
    }
    row > 0
}

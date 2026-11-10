use super::*;

const SHARED_REGIONS: usize = 2048;
const SHARED_COMPANIES: usize = 256;
const SHARED_WORDS: usize = 2 * (SHARED_REGIONS + SHARED_COMPANIES) + 2 + PREPARED_REQUEST_WORDS;
const SHARED_MEMORY_BUDGET: usize = 32 * 1024;
const CACHED_FIELDS: usize = 3;
const CACHED_SCRATCH_WORDS: usize = settlement::SHARED_WORDS + planning::SHARED_WORDS + 16;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum RequestStorage {
    Global,
    Shared,
    Cached { regions: usize },
}

impl RequestStorage {
    pub(super) fn new(device: &Device, regions: usize, companies: usize) -> Self {
        match regions <= SHARED_REGIONS
            && companies <= SHARED_COMPANIES
            && device.client.properties().hardware.max_shared_memory_size >= SHARED_MEMORY_BUDGET
        {
            true => Self::Shared,
            false => Self::Global,
        }
    }

    pub(super) fn cached(self, device: &Device, regions: usize, path_words: usize) -> Self {
        let words = CACHED_FIELDS * regions + PREPARED_REQUEST_WORDS + CACHED_SCRATCH_WORDS;
        match self == Self::Shared
            && path_words > 0
            && words * size_of::<u32>()
                <= device.client.properties().hardware.max_shared_memory_size
        {
            true => Self::Cached { regions },
            false => self,
        }
    }
}

#[derive(Clone, Copy)]
#[repr(usize)]
pub(super) enum RegionField {
    Available,
    Demand,
}

#[derive(CubeType)]
pub(super) struct RequestLists {
    words: Shared<[u32]>,
    cached_base: usize,
    #[cube(comptime)]
    storage: RequestStorage,
}

#[derive(CubeType)]
pub(super) struct RegionRequest {
    pub record: u32,
    pub visits: u32,
}

#[derive(CubeType)]
pub(super) struct AccessRequest {
    pub owner: u32,
    pub units: u32,
}

#[cube]
impl RequestLists {
    pub(super) fn new(#[comptime] storage: RequestStorage) -> Self {
        let words = Shared::<[u32]>::new_slice(comptime!(match storage {
            RequestStorage::Global => 1,
            RequestStorage::Shared => SHARED_WORDS,
            RequestStorage::Cached { regions } => PREPARED_REQUEST_WORDS + CACHED_FIELDS * regions,
        }));
        RequestLists {
            words,
            cached_base: 0,
            storage,
        }
    }

    pub(super) fn initialize(&mut self, config: &EconomyConfig) {
        if comptime!(self.storage == RequestStorage::Shared) {
            let count = 2 * (config.railway.region_count + config.railway.company_count)
                + 2
                + PREPARED_REQUEST_WORDS as u32;
            let mut index = UNIT_POS;
            while index < count {
                self.words[index as usize] = 0;
                index += CUBE_DIM;
            }
            sync_cube();
        }
        if comptime!(matches!(self.storage, RequestStorage::Cached { .. })) {
            if UNIT_POS == 0 {
                for index in 0..PREPARED_REQUEST_WORDS {
                    self.words[index] = 0;
                }
            }
            sync_cube();
        }
    }

    pub(super) fn refresh(&mut self, state: &mut [u32], config: &EconomyConfig) {
        if comptime!(matches!(self.storage, RequestStorage::Cached { .. })) {
            sync_storage();
            if UNIT_POS == 0 {
                self.invalidate(config);
            }
            let mut record = UNIT_POS;
            while record < config.railway.region_count {
                let available = regional_value(
                    state,
                    config,
                    record,
                    comptime!(RailCapacityField::Available),
                );
                let demand =
                    regional_value(state, config, record, comptime!(RailCapacityField::Demand));
                let level =
                    regional_number(state, config, record, comptime!(RailCapacityField::Level));
                let base = PREPARED_REQUEST_WORDS + record as usize;
                self.words[base] = select(
                    available.extract(HIGH_WORD) == 0,
                    available.extract(LOW_WORD),
                    u32::MAX,
                );
                self.words[base + config.railway.region_count as usize] = select(
                    demand.extract(HIGH_WORD) == 0,
                    demand.extract(LOW_WORD),
                    u32::MAX,
                );
                self.words[base + 2 * config.railway.region_count as usize] = level;
                record += CUBE_DIM;
            }
            sync_cube();
        }
    }

    pub(super) fn capacity_value(
        &self,
        state: &mut [u32],
        config: &EconomyConfig,
        record: u32,
        #[comptime] field: RegionField,
    ) -> Ticks {
        let mut cached = u32::new(comptime!(i64::from(u32::MAX)));
        if comptime!(matches!(self.storage, RequestStorage::Cached { .. })) {
            cached = self.words[PREPARED_REQUEST_WORDS
                + comptime!(field as usize) * config.railway.region_count as usize
                + record as usize];
        }
        let mut amount = ticks(cached, 0);
        if cached == u32::MAX {
            amount = regional_value(
                state,
                config,
                record,
                comptime!(match field {
                    RegionField::Available => RailCapacityField::Available,
                    RegionField::Demand => RailCapacityField::Demand,
                }),
            );
        }
        amount
    }

    pub(super) fn set_capacity_value(
        &mut self,
        state: &mut [u32],
        config: &EconomyConfig,
        record: u32,
        amount: Ticks,
        #[comptime] field: RegionField,
    ) {
        let mut changed = true;
        if comptime!(matches!(self.storage, RequestStorage::Cached { .. })) {
            let index = PREPARED_REQUEST_WORDS
                + comptime!(field as usize) * config.railway.region_count as usize
                + record as usize;
            let previous = self.words[index];
            changed = previous == u32::MAX
                || amount.extract(HIGH_WORD) > 0
                || previous != amount.extract(LOW_WORD);
            self.words[index] = select(
                amount.extract(HIGH_WORD) == 0,
                amount.extract(LOW_WORD),
                u32::MAX,
            );
        }
        if changed {
            regional_set(
                state,
                config,
                record,
                amount,
                comptime!(match field {
                    RegionField::Available => RailCapacityField::Available,
                    RegionField::Demand => RailCapacityField::Demand,
                }),
            );
        }
    }

    pub(super) fn capacity_level(
        &self,
        state: &mut [u32],
        config: &EconomyConfig,
        record: u32,
    ) -> u32 {
        match comptime!(self.storage) {
            RequestStorage::Cached { .. } => {
                self.words[PREPARED_REQUEST_WORDS
                    + 2 * config.railway.region_count as usize
                    + record as usize]
            }
            _ => regional_number(state, config, record, comptime!(RailCapacityField::Level)),
        }
    }

    pub(super) fn clear_path(&mut self, config: &EconomyConfig) {
        self.cached_base = 0;
        if comptime!(matches!(self.storage, RequestStorage::Cached { .. })) {
            config.railway.access_requests[(2 * config.railway.company_count + 1) as usize]
                .store(0);
        }
    }

    pub(super) fn invalidate(&mut self, config: &EconomyConfig) {
        self.cached_base = 0;
        self.set_prepared_word(config, comptime!(PreparedRequestWord::Carrier), 0);
    }

    pub(super) fn prepared_word(
        &self,
        config: &EconomyConfig,
        #[comptime] word: PreparedRequestWord,
    ) -> u32 {
        match comptime!(self.storage) {
            RequestStorage::Cached { .. } => self.words[comptime!(word as usize)],
            _ => self.access(
                config,
                (2 * config.railway.company_count + 1) as usize + comptime!(word as usize),
            ),
        }
    }

    pub(super) fn set_prepared_word(
        &mut self,
        config: &EconomyConfig,
        #[comptime] word: PreparedRequestWord,
        value: u32,
    ) {
        match comptime!(self.storage) {
            RequestStorage::Cached { .. } => self.words[comptime!(word as usize)] = value,
            _ => self.set_access(
                config,
                (2 * config.railway.company_count + 1) as usize + comptime!(word as usize),
                value,
            ),
        }
    }

    pub(super) fn restore_cached_path(
        &mut self,
        config: &EconomyConfig,
        path: &FreightRoute,
    ) -> bool {
        self.cached_base = 0;
        let mut restored = false;
        if comptime!(
            matches!(self.storage, RequestStorage::Cached { .. })
                && config.railway.path_cache == Feature::Enabled
                && config.railway.request_storage != RequestStorage::Global
        ) {
            restored = paths::restore(config, path, self);
        }
        restored
    }

    pub(super) fn bind_path(&mut self, base: usize) -> bool {
        let mut bound = false;
        if comptime!(self.storage != RequestStorage::Global) {
            self.cached_base = base;
            bound = true;
        }
        bound
    }

    pub(super) fn region_count(&self, config: &EconomyConfig) -> u32 {
        match self.cached_base as u32 {
            0 => self.region(config, (2 * config.railway.region_count) as usize),
            _ => config.railway.routes[self.cached_base],
        }
    }

    pub(super) fn region_entry(&self, config: &EconomyConfig, index: u32) -> RegionRequest {
        let mut request = RegionRequest {
            record: 0,
            visits: 0,
        };
        if self.cached_base > 0 {
            let pair = self.cached_base + 2 + 2 * index as usize;
            request.record = config.railway.routes[pair];
            request.visits = config.railway.routes[pair + 1];
        } else {
            request.record = self.region(config, (config.railway.region_count + index) as usize);
            request.visits = self.region(config, request.record as usize);
        }
        request
    }

    pub(super) fn access_count(&self, config: &EconomyConfig) -> u32 {
        match self.cached_base as u32 {
            0 => self.access(config, (2 * config.railway.company_count) as usize),
            _ => config.railway.routes[self.cached_base + 1],
        }
    }

    pub(super) fn access_entry(&self, config: &EconomyConfig, index: u32) -> AccessRequest {
        let mut request = AccessRequest { owner: 0, units: 0 };
        if self.cached_base > 0 {
            let regions = config.railway.routes[self.cached_base];
            let pair = self.cached_base + 2 + 2 * (regions + index) as usize;
            request.owner = config.railway.routes[pair];
            request.units = config.railway.routes[pair + 1];
        } else {
            request.owner = self.access(config, (config.railway.company_count + index) as usize);
            request.units = self.access(config, request.owner as usize);
        }
        request
    }

    pub(super) fn region(&self, config: &EconomyConfig, index: usize) -> u32 {
        match comptime!(self.storage) {
            RequestStorage::Global | RequestStorage::Cached { .. } => {
                config.railway.requests[index].load()
            }
            RequestStorage::Shared => self.words[index],
        }
    }

    pub(super) fn set_region(&mut self, config: &EconomyConfig, index: usize, value: u32) {
        match comptime!(self.storage) {
            RequestStorage::Global | RequestStorage::Cached { .. } => {
                config.railway.requests[index].store(value)
            }
            RequestStorage::Shared => self.words[index] = value,
        }
    }

    pub(super) fn add_region(&mut self, config: &EconomyConfig, index: usize, value: u32) -> u32 {
        match comptime!(self.storage) {
            RequestStorage::Global | RequestStorage::Cached { .. } => {
                config.railway.requests[index].fetch_add(value)
            }
            RequestStorage::Shared => {
                let previous = self.words[index];
                self.words[index] = previous + value;
                previous
            }
        }
    }

    pub(super) fn access(&self, config: &EconomyConfig, index: usize) -> u32 {
        match comptime!(self.storage) {
            RequestStorage::Global | RequestStorage::Cached { .. } => {
                config.railway.access_requests[index].load()
            }
            RequestStorage::Shared => {
                self.words[(2 * config.railway.region_count + 1) as usize + index]
            }
        }
    }

    pub(super) fn set_access(&mut self, config: &EconomyConfig, index: usize, value: u32) {
        match comptime!(self.storage) {
            RequestStorage::Global | RequestStorage::Cached { .. } => {
                config.railway.access_requests[index].store(value)
            }
            RequestStorage::Shared => {
                self.words[(2 * config.railway.region_count + 1) as usize + index] = value;
            }
        }
    }

    pub(super) fn add_access(&mut self, config: &EconomyConfig, index: usize, value: u32) -> u32 {
        match comptime!(self.storage) {
            RequestStorage::Global | RequestStorage::Cached { .. } => {
                config.railway.access_requests[index].fetch_add(value)
            }
            RequestStorage::Shared => {
                let index = (2 * config.railway.region_count + 1) as usize + index;
                let previous = self.words[index];
                self.words[index] = previous + value;
                previous
            }
        }
    }
}

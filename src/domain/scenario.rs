mod config;
mod generation;
mod opening;
mod setup;

use crate::{
    arithmetic::{DrawKey, hex},
    domain::{
        Actor, ActorRows, BASE_FIELDS, Economy, Field, Good, InitialTotals, Market, Population,
        Position, ProductionMethod, ProductionSite, ResourceRegion, State, SupplierCacheLayout,
        Technique, calendar, cohorts, employment, enterprise, finance, food, industry, landlords,
        prices, railway, research, resources, supply,
    },
};
use anyhow::{Result, anyhow};
pub use config::{Experiment, SCENARIO_VERSION, ScenarioConfig};
pub(super) use generation::{generated_firm_identity, generated_good};
use sha2::{Digest, Sha256};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Scenario {
    config: ScenarioConfig,
    totals: InitialTotals,
    resources: resources::ResourceGeography,
    rail_geography: railway::RailGeography,
    production: supply::StartingProduction,
    starting_prices: prices::StartingPrices,
    retailers: std::collections::BTreeMap<u64, enterprise::Distributor>,
    grain_markets: setup::GrainMarkets,
    initial_actors: std::sync::Arc<ActorRows>,
    initial_employment: employment::InitialEmployment,
    cohorts: Option<std::sync::Arc<cohorts::CohortLayout>>,
}
impl Scenario {
    fn agrarian_inputs(&self) -> std::collections::BTreeSet<Good> {
        let mut inputs =
            std::collections::BTreeSet::from([Good::Bread, Good::Tools, Good::Wood, Good::Coal]);
        if let enterprise::HouseholdProduction::Available { .. } = self.config.household_production
        {
            inputs.extend([Good::Grain, Good::Flour]);
        }
        for firm in self.initial_actors.firm_indices() {
            let good = Good::from_index(self.actor(firm).kind).expect("validated firm");
            inputs.extend(good.inputs().into_iter().flatten());
        }
        if let Some(research) = &self.config.research {
            for lab in &research.labs {
                let firm = self.index_of(lab.owner).expect("validated lab");
                let good = Good::from_index(self.actor(firm).kind).expect("validated firm");
                let technique = self.technique(firm);
                inputs.extend(lab.methods.iter().flat_map(|method| {
                    method.recipe(good, technique).inputs.into_iter().flatten()
                }));
            }
        }
        inputs.extend(self.config.distributors.iter().map(|store| store.good));
        if !self.config.railway.companies.is_empty() {
            inputs.extend([Good::Coal, Good::Steel, Good::Locomotives]);
        }
        if self.config.property.is_some() || self.config.rent.is_some() {
            inputs.insert(Good::Construction);
        }
        inputs
    }

    pub fn goods_markets(&self) -> Vec<Market> {
        let mut markets: Vec<_> = match self.config.economy {
            Economy::Industrial => Good::ALL.into_iter().map(Market::Input).collect(),
            Economy::Agrarian => self
                .agrarian_inputs()
                .into_iter()
                .map(Market::Input)
                .collect(),
        };
        markets.extend([
            Market::Household,
            Market::Consumer(Good::Coal),
            Market::Consumer(Good::Wood),
        ]);
        if self.config.economy == Economy::Industrial {
            markets.extend(
                Good::ALL
                    .into_iter()
                    .filter(|good| good.consumption_interval() > 1)
                    .map(Market::Consumer),
            );
        }
        markets
    }
    pub fn baseline(households: u32, seed: u64, experiment: Experiment) -> Result<Self> {
        Self::new(ScenarioConfig::baseline(households, seed, experiment)?)
    }
    pub(super) fn cohorts(&self) -> Option<&cohorts::CohortLayout> {
        self.cohorts.as_deref()
    }

    pub(super) fn initial_employment(&self) -> &employment::InitialEmployment {
        &self.initial_employment
    }

    pub(super) fn distributor(&self, firm: u64) -> Option<&enterprise::Distributor> {
        self.retailers.get(&firm)
    }

    pub fn config(&self) -> &ScenarioConfig {
        &self.config
    }
    pub fn weather(&self, day: u64) -> calendar::Weather {
        let weather = calendar::Weather::on_day(day);
        let harvest = match self.config.household_production {
            enterprise::HouseholdProduction::Available {
                harvest_start,
                harvest_days,
                ..
            } => calendar::HarvestSeason::in_window(weather.date, harvest_start, harvest_days),
            enterprise::HouseholdProduction::Unavailable => weather.harvest,
        };
        calendar::Weather { harvest, ..weather }
    }
    pub(crate) fn initial_price(&self, good: Good) -> u32 {
        self.starting_prices.good(good)
    }
    pub fn opening_totals(&self) -> &InitialTotals {
        &self.totals
    }
    pub fn resource_regions(&self) -> &[ResourceRegion] {
        self.resources.regions()
    }
    pub fn has_resource(&self, good: Good, position: Position) -> bool {
        self.resources.has_resource(good, position)
    }
    pub fn production_site(&self, good: Good, position: Position) -> ProductionSite {
        self.resources.production_site(good, position)
    }
    pub fn geographic_capacity(&self, actor: &Actor) -> u32 {
        match actor.kind {
            0 => actor.capacity,
            _ => self.firm_geographic_capacity(actor),
        }
    }
    fn firm_geographic_capacity(&self, actor: &Actor) -> u32 {
        match self.retailers.contains_key(&actor.identity) {
            true => actor.capacity,
            false => match self.production_site(
                Good::from_index(actor.kind).expect("validated firm good"),
                actor.position,
            ) {
                ProductionSite::ResourceMissing => 0,
                ProductionSite::UnrestrictedManufacturing | ProductionSite::ResourceAvailable => {
                    actor.capacity
                }
            },
        }
    }
    pub(super) fn operating_capacity(&self, state: &State, index: usize) -> u32 {
        let actor = state.actors.at(index);
        match state.get(Field::EnterpriseModel, index) {
            model if model == enterprise::EnterpriseModel::Distributor as u64 => actor.capacity,
            _ => self.geographic_capacity(&actor),
        }
    }
    pub(crate) fn aggregate_demography(&self) -> bool {
        self.config.people.is_some()
    }
    pub fn population_rows(&self) -> usize {
        self.cohorts
            .as_ref()
            .map_or(self.config.population_size() as usize, |layout| {
                layout.rows.len()
            })
    }
    pub fn actors(&self) -> usize {
        self.population_rows() + self.config.firms as usize
    }
    pub fn ring(&self) -> usize {
        self.config
            .search_radius
            .div_ceil(self.config.shipment_speed)
            .max(1) as usize
            + 1
    }
    pub fn fields(&self) -> usize {
        BASE_FIELDS
            + super::RetailAccount::FIELDS
            + 2 * self.ring() * Good::COUNT
            + self.rail_geography.blocks(self.actors()) * railway::RailCapacityField::COUNT
    }
    pub fn rail_geography(&self) -> &railway::RailGeography {
        &self.rail_geography
    }
    pub fn state_len(&self) -> usize {
        self.state_len_for(self.actors())
    }
    pub(super) fn state_len_for(&self, actors: usize) -> usize {
        actors * self.fields()
            + self
                .config
                .finance
                .as_ref()
                .map_or(0, finance::FinanceConfig::state_len)
    }
    pub fn memory_estimate(&self) -> u64 {
        let actors = self.actors() as u64;
        let batch = u64::from(self.config.batch_size).min(actors);
        self.state_len() as u64 * 8 * 2
            + (self.state_len() as u64 * 2).div_ceil(512) * 8
            + self
                .config
                .finance
                .as_ref()
                .map_or(0, finance::FinanceConfig::memory_estimate)
            + self
                .config
                .research
                .as_ref()
                .map_or(0, research::ResearchConfig::memory_estimate)
            + actors * 128
            + (actors.div_ceil(128) + 1) * 4
            + if self.aggregate_demography() {
                actors * (16 * 8 + 4)
            } else {
                self.config.demography.people.len().max(
                    self.config.population_size() as usize
                        * self.config.demography.initial_household_members as usize,
                ) as u64
                    * 164
            }
            + actors * 80
            + actors * SupplierCacheLayout::new(&self.initial_actors).extra_words as u64 * 4
            + actors * 4
            + 12
            + match self.config.railway.companies.is_empty() {
                true => 0,
                false => {
                    let companies = self.config.railway.companies.len() as u64;
                    let stations = self.config.railway.stations().len() as u64;
                    let growth = (actors / 64).next_power_of_two().clamp(64, 16_384);
                    let freight_actors = (actors + growth).next_multiple_of(growth);
                    ((freight_actors + self.rail_geography.regions.len() as u64)
                        * (railway::FREIGHT_RESOURCE_WORDS + 1) as u64
                        + freight_actors
                            * (railway::FREIGHT_ACCOUNT_WORDS
                                + railway::FREIGHT_BOOKING_WORDS
                                + railway::FREIGHT_QUEUE_WORDS
                                + 4
                                + 2 * (railway::FREIGHT_CLAIM_WORDS
                                    + railway::FREIGHT_CLAIM_SORT_WORDS))
                                as u64
                        + match self.config.railway.route_storage(stations as usize) {
                            railway::RouteStorage::Indexed => {
                                let rows = self.config.railway.route_origins();
                                (rows as u64 + 1) * stations * railway::RAIL_ROUTE_WORDS as u64
                                    + companies * stations
                                    + rows as u64
                                    + stations
                            }
                            railway::RouteStorage::Dense => {
                                companies * stations.pow(2) * railway::RAIL_ROUTE_WORDS as u64
                            }
                        }
                        + stations.pow(2) * self.ring() as u64
                        + companies.pow(2)
                        + 2 * companies
                        + 2
                        + stations
                        + 1
                        + 2 * self.config.railway.segments.len() as u64
                        + railway::FREIGHT_DISPATCH_WORDS as u64
                        + railway::FREIGHT_PROGRESS_WORDS as u64)
                        * 4
                }
            }
            + batch * (16 * 8 * 4 + 16 * 4 * 3 + 28)
            + 128 * 1024 * 1024
    }
    pub fn fingerprint(&self) -> Result<String> {
        Ok(hex(&Sha256::digest(serde_json::to_vec(&self.config)?)))
    }
    pub fn index_of(&self, identity: u64) -> Option<usize> {
        self.member_row(identity)
            .filter(|&row| self.identity(row) == Some(identity))
    }
    pub(super) fn member_row(&self, identity: u64) -> Option<usize> {
        match &self.config.population {
            Some(population) => population
                .index_of(identity)
                .and_then(|source| self.member_source(source)),
            None => (1..=u64::from(self.config.population_size() + self.config.firms))
                .contains(&identity)
                .then(|| self.member_source((identity - 1) as usize))
                .flatten(),
        }
    }
    pub fn identity(&self, index: usize) -> Option<u64> {
        self.initial_actors.identities().get(index).copied()
    }
    pub fn next_identity(&self) -> u64 {
        self.config
            .population
            .as_ref()
            .map(Population::next_identity)
            .unwrap_or(u64::from(self.config.population_size() + self.config.firms) + 1)
    }
    fn source(&self, row: usize) -> usize {
        match &self.cohorts {
            Some(layout) => layout.source(row),
            None => row,
        }
    }
    pub(super) fn row(&self, source: usize) -> usize {
        self.member_source(source).expect("registered source")
    }
    fn member_source(&self, source: usize) -> Option<usize> {
        match &self.cohorts {
            Some(layout) => layout.row(source),
            None => Some(source),
        }
    }
    pub fn technique(&self, index: usize) -> Technique {
        match (
            &self.config.population,
            index.checked_sub(self.population_rows()),
        ) {
            (Some(population), Some(firm))
                if population.firms[firm].good != Good::Railway
                    || population.firms[firm].technique.labor_minutes > 0 =>
            {
                population.firms[firm].technique
            }
            _ if self.actor(index).kind == Good::Railway as u32 => Technique {
                labor_minutes: Technique::railway().labor_minutes.min(self.config.work_cap),
                ..Technique::railway()
            },
            _ if self.actor(index).kind == Good::Housing as u32 => {
                Technique::housing(landlords::DEFAULT_HOUSING_UNITS).expect("housing recipe")
            }
            _ => food::technique(
                &self.config,
                Good::from_index(self.actor(index).kind).unwrap_or(Good::Bread),
            ),
        }
    }
    pub fn production_recipe(
        &self,
        index: usize,
        method: Option<ProductionMethod>,
    ) -> Result<industry::Recipe> {
        let actor = self.actor(index);
        let good = Good::from_index(actor.kind)?;
        let technique = self.technique(index);
        match method {
            None => Ok(industry::Recipe {
                technique,
                inputs: good.inputs(),
            }),
            Some(method) => match self.config.research.as_ref().is_some_and(|research| {
                research
                    .labs
                    .iter()
                    .any(|lab| lab.owner == actor.identity && lab.methods.contains(&method))
            }) {
                true => Ok(food::upgraded_recipe(&self.config, method, good, technique)),
                false => Err(anyhow!("production method is not configured for this firm")),
            },
        }
    }
    pub(super) fn actor(&self, index: usize) -> Actor {
        match self.initial_actors.get(index) {
            Some(actor) => actor,
            None => self.generate_actor(index),
        }
    }
    fn generate_actor(&self, row: usize) -> Actor {
        let mut actor = self.source_actor(self.source(row));
        actor.owner = self.row(actor.owner as usize) as u32;
        if let Some(cohort) = self
            .cohorts
            .as_ref()
            .and_then(|layout| layout.rows.get(row))
        {
            actor.capacity = cohort.count;
        }
        actor
    }
    pub(super) fn source_actor(&self, index: usize) -> Actor {
        match &self.config.population {
            Some(population) => population.actor(index),
            None => {
                let config = &self.config;
                let identity = index as u64 + 1;
                let kind = match index < config.population_size() as usize {
                    true => 0,
                    false => {
                        generated_good(config, index - config.population_size() as usize) as u32
                    }
                };
                let position = match config.experiment {
                    Experiment::Concentrated => Position {
                        x: config.map_size / 2,
                        y: config.map_size / 2,
                    },
                    _ => {
                        let draw = DrawKey {
                            seed: config.seed,
                            day: 0,
                            phase: 0,
                            actor: identity,
                            purpose: 0,
                            index: 0,
                        }
                        .draw();
                        Position {
                            x: draw as u32 % config.map_size,
                            y: (draw >> 32) as u32 % config.map_size,
                        }
                    }
                };
                let retailer = self.retailers.get(&identity);
                let position = match (kind, retailer, self.grain_markets.sites.get(&index)) {
                    (0, _, _) => match (
                        config.household_production,
                        employment::InitialSettlement::new(&config).worker(index),
                    ) {
                        (enterprise::HouseholdProduction::Available { .. }, None)
                            if !self.grain_markets.villages.is_empty() =>
                        {
                            let village = (u64::from(position.x) * u64::from(config.map_size)
                                + u64::from(position.y))
                                % self.grain_markets.villages.len() as u64;
                            self.source_actor(self.grain_markets.villages[village as usize])
                                .position
                        }
                        (enterprise::HouseholdProduction::Available { .. }, None) => self
                            .resources
                            .place(Good::Grain, identity, config, position),
                        _ => match self.initial_employment.employer(index) {
                            Some(employer) => {
                                let workplace = self.source_actor(employer).position;
                                Position {
                                    x: (workplace.x + position.x % 16).min(config.map_size - 1),
                                    y: (workplace.y + position.y % 16).min(config.map_size - 1),
                                }
                            }
                            None => position,
                        },
                    },
                    (_, _, Some(site)) => self.source_actor(*site).position,
                    (_, Some(store), _)
                        if config.version >= 5
                            && config.economy == Economy::Industrial
                            && store.good == Good::Grain =>
                    {
                        self.resources
                            .place(Good::Grain, identity, config, position)
                    }
                    (_, Some(_), _) => position,
                    _ => self.resources.place(
                        Good::from_index(kind).expect("generated firm good"),
                        identity,
                        config,
                        position,
                    ),
                };
                Actor {
                    identity,
                    position,
                    kind,
                    owner: index as u32 % config.population_size(),
                    capacity: match (kind, retailer) {
                        (0, _) => 1,
                        (_, Some(store)) => self.production.retail_capacity(config, store),
                        _ => self.production.capacities[kind as usize - 1],
                    },
                }
            }
        }
    }
}

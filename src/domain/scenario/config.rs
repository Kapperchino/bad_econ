use crate::domain::{
    BehaviorPolicy, CohortSizes, Economy, Good, HouseholdLiquidity, HouseholdPolicy, PayrollPolicy,
    Population, RentPolicy, ResourceMap, bargaining, company, demography, enterprise, finance,
    food, property, railway, research, supply,
};
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};

pub const SCENARIO_VERSION: u32 = 5;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, clap::ValueEnum)]
pub enum Experiment {
    Baseline,
    Shortage,
    Transport,
    Concentrated,
}

macro_rules! scenario_config {
    ($($field:tt)*) => {
        #[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
        #[serde(try_from = "ScenarioDocument")]
        pub struct ScenarioConfig { $($field)* }

        #[derive(Deserialize)]
        #[serde(remote = "ScenarioConfig", deny_unknown_fields)]
        struct CurrentScenario { $($field)* }
    };
}

scenario_config! {
    #[serde(default)]
    pub economy: Economy,
    pub version: u32,
    pub seed: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub people: Option<u32>,
    #[serde(default, skip_serializing_if = "CohortSizes::is_default")]
    pub cohort_sizes: CohortSizes,
    #[serde(default, skip_serializing_if = "is_zero_count")]
    pub households: u32,
    pub firms: u32,
    pub map_size: u32,
    #[serde(default)]
    pub resources: ResourceMap,
    pub search_radius: u32,
    pub commute_speed: u32,
    pub shipment_speed: u32,
    #[serde(default)]
    pub railway: railway::RailwayConfig,
    pub household_cash: u64,
    #[serde(default, skip_serializing_if = "HouseholdLiquidity::is_uniform")]
    pub household_liquidity: HouseholdLiquidity,
    #[serde(
        default,
        skip_serializing_if = "enterprise::EntrepreneurshipPolicy::is_default"
    )]
    pub entrepreneurship: enterprise::EntrepreneurshipPolicy,
    #[serde(default)]
    pub expansion: enterprise::ExpansionPolicy,
    pub firm_cash: u64,
    pub buffer_days: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub producer_buffer_days: Option<u32>,
    pub household_need: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub food: Option<food::FoodPolicy>,
    pub capacity: u32,
    pub packet_output: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub durable_packet_output: Option<u32>,
    pub packet_minutes: u32,
    pub work_cap: u32,
    pub reserved_minutes: u32,
    pub wage: u32,
    pub prices: [u32; 3],
    pub reserve: u64,
    pub distribution_rate: u32,
    pub forecast_weight: u32,
    pub coverage: u32,
    pub minimum_price: u32,
    pub maximum_price: u32,
    pub delivery_penalty: u32,
    pub batch_size: u32,
    pub experiment: Experiment,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub population: Option<Population>,
    #[serde(default, skip_serializing_if = "BehaviorPolicy::is_default")]
    pub behavior: BehaviorPolicy,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub payroll: Option<PayrollPolicy>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rent: Option<RentPolicy>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub household_policy: Option<HouseholdPolicy>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finance: Option<finance::FinanceConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub research: Option<research::ResearchConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bargaining: Option<bargaining::BargainingPolicy>,
    #[serde(default)]
    pub demography: demography::DemographicPolicy,
    #[serde(default)]
    pub household_production: enterprise::HouseholdProduction,
    #[serde(default)]
    pub distributors: Vec<enterprise::Distributor>,
    #[serde(default)]
    pub ownership: Vec<enterprise::Ownership>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub companies: Vec<company::Company>,
    #[serde(default)]
    pub land_access: Option<enterprise::LandAccess>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub property: Option<property::PropertyConfig>,
    #[serde(default)]
    pub working_conditions: Vec<enterprise::FirmConditions>,
}

fn is_zero_count(count: &u32) -> bool {
    *count == 0
}

impl ScenarioConfig {
    pub fn population_size(&self) -> u32 {
        self.people.unwrap_or(self.households)
    }
    pub(super) const HOUSEHOLDS: std::ops::RangeInclusive<u32> = 1..=1_000_000;

    pub(super) fn normalized(self) -> Result<Self> {
        let mut config = self;
        if config.people.is_some() {
            config.households = 0;
            config.demography.initial_household_members = 1;
        }
        config.minimum_price = 1;
        config.food = config.food.map(food::FoodPolicy::new).transpose()?;
        config.property = config
            .property
            .take()
            .map(|property| property::PropertyConfig::new(property, config.map_size))
            .transpose()?;
        config.entrepreneurship = enterprise::EntrepreneurshipPolicy::new(config.entrepreneurship)?;
        config.expansion = enterprise::ExpansionPolicy::new(config.expansion)?;
        config.companies = config
            .companies
            .into_iter()
            .map(company::Company::new)
            .collect::<Result<_>>()?;
        config.ownership = config
            .ownership
            .into_iter()
            .map(enterprise::Ownership::new)
            .collect::<Result<Vec<_>>>()?;
        config.distributors = config
            .distributors
            .into_iter()
            .map(enterprise::Distributor::new)
            .collect::<Result<Vec<_>>>()?;
        config.land_access = config
            .land_access
            .map(enterprise::LandAccess::new)
            .transpose()?;
        for setting in &mut config.working_conditions {
            setting.conditions = enterprise::WorkingConditions::new(setting.conditions)?;
        }
        config.demography = demography::DemographicPolicy::new(config.demography)?;
        config.household_production =
            enterprise::HouseholdProduction::new(config.household_production)?;
        config.bargaining = config
            .bargaining
            .take()
            .map(bargaining::BargainingPolicy::new)
            .transpose()?;
        config.household_policy = config
            .household_policy
            .take()
            .map(|policy| HouseholdPolicy::new(policy, config.reserved_minutes))
            .transpose()?;
        config.payroll = config.payroll.take().map(PayrollPolicy::new).transpose()?;
        config.rent = config.rent.map(RentPolicy::new).transpose()?;
        config.behavior = BehaviorPolicy::new(config.behavior)?;
        config.population = config
            .population
            .take()
            .map(|population| Population::new(population, &config))
            .transpose()?;
        let bounds = [
            (2..=SCENARIO_VERSION).contains(&config.version),
            match config.people {
                Some(people) => (1..=1_000_000_000).contains(&people),
                None => ScenarioConfig::HOUSEHOLDS.contains(&config.households),
            },
            (3..=20_000).contains(&config.firms),
            (1..=100_000).contains(&config.map_size),
            (1..=config.map_size).contains(&config.search_radius),
            config.commute_speed > 0,
            (1..=1_000_000).contains(&config.shipment_speed),
            config.household_need > 0,
            config.household_need <= 100,
            (1..=10_000).contains(&config.capacity),
            (1..=100).contains(&config.packet_output),
            config
                .durable_packet_output
                .is_none_or(|output| (1..=100).contains(&output)),
            (1..=1440).contains(&config.packet_minutes),
            config.packet_minutes <= config.work_cap,
            config.work_cap <= 1440,
            config.reserved_minutes <= 1440,
            (1..=10_000).contains(&config.wage),
            config.minimum_price > 0,
            config.maximum_price <= 100_000,
            config.minimum_price <= config.maximum_price,
            config
                .prices
                .iter()
                .all(|p| (config.minimum_price..=config.maximum_price).contains(p)),
            config.forecast_weight <= 1000,
            config.distribution_rate <= 1000,
            (1..=30).contains(&config.coverage),
            config.buffer_days <= 30,
            config.producer_buffer_days.is_none_or(|days| days <= 30),
            config.delivery_penalty <= 10_000,
            (1..=1_048_576).contains(&config.batch_size),
        ];
        match bounds.into_iter().all(std::convert::identity) {
            true => Ok(config),
            false => Err(anyhow!("scenario parameter outside DeterministicV1 bounds")),
        }
    }

    pub(super) fn baseline(households: u32, seed: u64, experiment: Experiment) -> Result<Self> {
        match (1..=1_000_000_000).contains(&households) {
            true => (),
            false => Err(anyhow!("scenario parameter outside DeterministicV1 bounds"))?,
        }
        let firms = (households / 100).clamp(3, 20_000);
        let map_size = match households {
            0..=10_000 => 1000,
            10_001..=100_000 => 3000,
            _ => 10_000,
        };
        let mut config = Self {
            economy: Economy::Agrarian,
            version: SCENARIO_VERSION,
            seed,
            people: None,
            cohort_sizes: CohortSizes::default(),
            households,
            firms,
            map_size,
            resources: ResourceMap::Generated,
            search_radius: 1000,
            commute_speed: 50,
            shipment_speed: 1000,
            railway: railway::RailwayConfig::default(),
            household_cash: 5000,
            household_liquidity: HouseholdLiquidity::Uniform,
            entrepreneurship: enterprise::EntrepreneurshipPolicy::default(),
            expansion: enterprise::ExpansionPolicy::default(),
            firm_cash: 200_000,
            buffer_days: 4,
            producer_buffer_days: None,
            household_need: 1,
            food: None,
            capacity: 100,
            packet_output: 3,
            durable_packet_output: None,
            packet_minutes: 360,
            work_cap: 480,
            reserved_minutes: 960,
            wage: 1,
            prices: [150, 300, 450],
            reserve: 100_000,
            distribution_rate: 100,
            forecast_weight: 250,
            coverage: 2,
            minimum_price: 1,
            maximum_price: 10_000,
            delivery_penalty: 5,
            batch_size: 65_536,
            experiment,
            population: None,
            behavior: BehaviorPolicy::default(),
            payroll: None,
            rent: None,
            household_policy: None,
            finance: None,
            research: None,
            bargaining: None,
            demography: demography::DemographicPolicy::default(),
            household_production: enterprise::HouseholdProduction::Unavailable,
            distributors: Vec::new(),
            ownership: Vec::new(),
            companies: Vec::new(),
            land_access: None,
            property: None,
            working_conditions: Vec::new(),
        };
        config.capacity = (supply::StartingDemand::new(&config)
            .production(Good::Bread)
            .div_ceil(u64::from(config.firms / 3) * u64::from(config.packet_output))
            as u32)
            .clamp(config.capacity, 10_000);
        match experiment {
            Experiment::Shortage => config.buffer_days = 0,
            Experiment::Transport => config.shipment_speed = 100,
            Experiment::Baseline | Experiment::Concentrated => (),
        }
        Ok(config)
    }
}

#[derive(Deserialize)]
#[serde(transparent)]
struct ScenarioDocument(serde_json::Value);

impl TryFrom<ScenarioDocument> for ScenarioConfig {
    type Error = anyhow::Error;
    fn try_from(document: ScenarioDocument) -> Result<Self> {
        let mut value = document.0;
        match value.get("version").and_then(serde_json::Value::as_u64) {
            Some(1) => {
                value["version"] = 2.into();
                if let Some(behavior) = value
                    .get_mut("behavior")
                    .and_then(serde_json::Value::as_object_mut)
                {
                    behavior.remove("health_recovery");
                    behavior.remove("hunger_damage");
                }
                if let Some(policy) = value
                    .get_mut("household_policy")
                    .and_then(serde_json::Value::as_object_mut)
                {
                    policy.remove("recovery");
                    policy.remove("deficit_damage");
                    policy.remove("fatigue_damage");
                }
                if let Some(households) = value
                    .get_mut("population")
                    .and_then(|population| population.get_mut("households"))
                    .and_then(serde_json::Value::as_array_mut)
                {
                    for household in households {
                        if let Some(household) = household.as_object_mut() {
                            household.remove("health");
                        }
                    }
                }
            }
            Some(version) if (2..=u64::from(SCENARIO_VERSION)).contains(&version) => (),
            _ => Err(anyhow!(
                "unsupported scenario format; expected version 1 through {SCENARIO_VERSION}"
            ))?,
        }
        let current = CurrentScenario::deserialize(value)?;
        match (current.people, current.households) {
            (Some(_), 1..) => Err(anyhow!("choose one population count: people or households"))?,
            (None, _) if current.cohort_sizes != CohortSizes::default() => {
                Err(anyhow!("cohort sizes require a people count"))?
            }
            _ => (),
        }
        Ok(current)
    }
}

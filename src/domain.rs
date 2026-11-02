use crate::{
    arithmetic::{DrawKey, PROFILE, add, hex, table_hash},
    ledger::Coordinate,
};
use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub mod bargaining;
pub mod demography;
pub mod enterprise;
pub mod finance;
mod households;
pub mod industry;
mod payroll;
pub mod research;
pub use households::HouseholdPolicy;
pub use industry::{Economy, ProductionMethod, Technology};
mod population;
mod resources;
mod supply;
pub use payroll::{MAX_TAX_BRACKETS, PAYROLL_SCALE, PayrollPolicy, TaxBracket};
pub use population::{
    BehaviorPolicy, FirmSeed, HouseholdSeed, InitialTotals, Population, Technique,
};
pub use resources::{ProductionSite, ResourceMap, ResourceRegion};

pub const SCENARIO_VERSION: u32 = 2;
pub const STATE_HASH_VERSION: &str = "bad-econ/state/DeterministicV1/v7";
pub const USED_FIELDS: usize = Field::PendingCapacity as usize + 1;
pub const TRANSIT_FIELDS: usize = USED_FIELDS + Good::COUNT;
pub const CONSUMPTION_FIELDS: usize = TRANSIT_FIELDS + Good::COUNT;
pub const PRODUCTION_FIELDS: usize = CONSUMPTION_FIELDS + Good::COUNT;
pub const BASE_FIELDS: usize = PRODUCTION_FIELDS + Good::COUNT;

#[repr(u32)]
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum Field {
    Cash,
    Grain,
    Flour,
    Bread,
    Iron,
    Coal,
    Steel,
    Flax,
    Linen,
    Cloth,
    Coats,
    Wood,
    Tools,
    Furniture,
    Cotton,
    Dye,
    Glass,
    Escrow,
    Slots,
    Wage,
    Price,
    Demand,
    Pipeline,
    Time,
    Fatigue,
    Employer,
    Due,
    Employed,
    Transit,
    Sales,
    Unfilled,
    Produced,
    Used,
    Consumed,
    Shortage,
    Plan,
    Purchased,
    Hours,
    GrossIncome,
    PayrollReserve,
    TaxTreasury,
    ChargeTreasury,
    ResearchMinutes,
    LaborMinutes,
    InputTicks,
    OutputTicks,
    FirstInput,
    SecondInput,
    ActiveMethod,
    FoodNeed,
    FoodFraction,
    LivingMembers,
    Worker,
    EquipmentCondition,
    SafetyProvision,
    HomeMinutes,
    LandPlots,
    CropWork,
    HarvestYear,
    EnterpriseStatus,
    InvestmentGood,
    InvestmentFirm,
    ConstructionRemaining,
    IdleDays,
    InstalledCapital,
    EnterpriseModel,
    StockCoverage,
    RetailMargin,
    ReplacementCost,
    RetailOpen,
    SplitPerson,
    MoveTarget,
    MoveDue,
    MoveX,
    MoveY,
    ParentHousehold,
    EstateSettled,
    BasicLaborMinutes,
    BasicInputTicks,
    BasicOutputTicks,
    BasicFirstInput,
    BasicSecondInput,
    FoodAccess,
    HomeTime,
    FirstInputCost,
    SecondInputCost,
    OperatingReserve,
    HouseholdBudget,
    EquityInvested,
    PreviousX,
    PreviousY,
    PendingCapacity,
}
impl Field {
    pub fn inventory(good: Good) -> Self {
        match good {
            Good::Grain => Self::Grain,
            Good::Flour => Self::Flour,
            Good::Bread => Self::Bread,
            Good::Iron => Self::Iron,
            Good::Coal => Self::Coal,
            Good::Steel => Self::Steel,
            Good::Flax => Self::Flax,
            Good::Linen => Self::Linen,
            Good::Cloth => Self::Cloth,
            Good::Coats => Self::Coats,
            Good::Wood => Self::Wood,
            Good::Tools => Self::Tools,
            Good::Furniture => Self::Furniture,
            Good::Cotton => Self::Cotton,
            Good::Dye => Self::Dye,
            Good::Glass => Self::Glass,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum GoodKind {
    Natural,
    Manufactured,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum Good {
    Grain = 1,
    Flour = 2,
    Bread = 3,
    Iron = 4,
    Coal = 5,
    Steel = 6,
    Flax = 7,
    Linen = 8,
    Cloth = 9,
    Coats = 10,
    Wood = 11,
    Tools = 12,
    Furniture = 13,
    Cotton = 14,
    Dye = 15,
    Glass = 16,
}
impl Good {
    pub const COUNT: usize = 16;
    pub const ALL: [Self; Self::COUNT] = [
        Self::Grain,
        Self::Flour,
        Self::Bread,
        Self::Iron,
        Self::Coal,
        Self::Steel,
        Self::Flax,
        Self::Linen,
        Self::Cloth,
        Self::Coats,
        Self::Wood,
        Self::Tools,
        Self::Furniture,
        Self::Cotton,
        Self::Dye,
        Self::Glass,
    ];
    pub fn kind(self) -> GoodKind {
        match self {
            Self::Grain
            | Self::Iron
            | Self::Coal
            | Self::Flax
            | Self::Wood
            | Self::Cotton
            | Self::Dye => GoodKind::Natural,
            Self::Flour
            | Self::Bread
            | Self::Steel
            | Self::Linen
            | Self::Cloth
            | Self::Coats
            | Self::Tools
            | Self::Furniture
            | Self::Glass => GoodKind::Manufactured,
        }
    }
    pub fn inputs(self) -> [Option<Self>; 2] {
        match self {
            Self::Flour => [Some(Self::Grain), None],
            Self::Bread => [Some(Self::Flour), None],
            Self::Steel => [Some(Self::Iron), Some(Self::Coal)],
            Self::Linen => [Some(Self::Flax), None],
            Self::Cloth => [Some(Self::Cotton), Some(Self::Dye)],
            Self::Coats => [Some(Self::Cloth), Some(Self::Linen)],
            Self::Tools => [Some(Self::Iron), Some(Self::Wood)],
            Self::Furniture => [Some(Self::Wood), Some(Self::Tools)],
            Self::Glass => [Some(Self::Coal), None],
            _ => [None, None],
        }
    }
    pub fn consumption_interval(self) -> u32 {
        match self {
            Self::Bread => 1,
            Self::Linen | Self::Cloth => 20,
            Self::Coats => 40,
            Self::Furniture | Self::Glass => 60,
            _ => 0,
        }
    }
    pub fn initial_price(self) -> u32 {
        [
            150, 300, 450, 160, 100, 350, 100, 220, 260, 600, 100, 320, 500, 100, 120, 250,
        ][self as usize - 1]
    }
    pub fn food<T: Default + Copy>(values: [T; 3]) -> [T; Self::COUNT] {
        let mut goods = [T::default(); Self::COUNT];
        goods[..3].copy_from_slice(&values);
        goods
    }
    pub fn from_index(index: u32) -> Result<Self> {
        match index {
            1 => Ok(Self::Grain),
            2 => Ok(Self::Flour),
            3 => Ok(Self::Bread),
            4 => Ok(Self::Iron),
            5 => Ok(Self::Coal),
            6 => Ok(Self::Steel),
            7 => Ok(Self::Flax),
            8 => Ok(Self::Linen),
            9 => Ok(Self::Cloth),
            10 => Ok(Self::Coats),
            11 => Ok(Self::Wood),
            12 => Ok(Self::Tools),
            13 => Ok(Self::Furniture),
            14 => Ok(Self::Cotton),
            15 => Ok(Self::Dye),
            16 => Ok(Self::Glass),

            _ => Err(anyhow!("invalid good")),
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum Experiment {
    Baseline,
    Shortage,
    Transport,
    Concentrated,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(try_from = "ScenarioDocument", deny_unknown_fields)]
pub struct ScenarioConfig {
    #[serde(default)]
    pub economy: Economy,
    pub version: u32,
    pub seed: u64,
    pub households: u32,
    pub firms: u32,
    pub map_size: u32,
    #[serde(default)]
    pub resources: ResourceMap,
    pub search_radius: u32,
    pub commute_speed: u32,
    pub shipment_speed: u32,
    pub household_cash: u64,
    pub firm_cash: u64,
    pub buffer_days: u32,
    pub household_need: u32,
    pub capacity: u32,
    pub packet_output: u32,
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
    #[serde(default)]
    pub land_access: Option<enterprise::LandAccess>,
    #[serde(default)]
    pub working_conditions: Vec<enterprise::FirmConditions>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Scenario {
    config: ScenarioConfig,
    totals: InitialTotals,
    resources: resources::ResourceGeography,
    capacities: [u32; Good::COUNT],
    starting_stocks: [u64; Good::COUNT],
    retailers: std::collections::BTreeMap<u64, enterprise::Distributor>,
    retail_demand: [u64; Good::COUNT],
    initial_actors: std::sync::Arc<ActorRows>,
}
impl Scenario {
    pub fn industrial(households: u32, seed: u64, experiment: Experiment) -> Result<Self> {
        let mut config = Self::baseline(households, seed, experiment)?.config;
        config.economy = Economy::Industrial;
        config.demography.initial_household_members = 3;
        config.household_production = enterprise::HouseholdProduction::industrial();
        config.firms = config.firms.max(3 * Good::COUNT as u32);
        config.packet_minutes = 240;
        config.packet_output = 20;
        config.prices = [15, 30, 45];
        config.distributors = (0..config.firms)
            .filter(|&firm| generated_good(&config, firm as usize) == Good::Bread && firm % 3 == 0)
            .map(|firm| enterprise::Distributor {
                firm: u64::from(households + firm + 1),
                good: Good::Bread,
                handling_minutes: 60,
                units_per_shift: 24,
                reserve_days: 3,
                margin_per_mille: 250,
            })
            .collect();
        let labs = (0..config.firms.min(households))
            .filter(|&firm| {
                !config
                    .distributors
                    .iter()
                    .any(|store| store.firm == u64::from(households + firm + 1))
            })
            .map(|firm| research::ResearchLab {
                owner: u64::from(households + firm + 1),
                researcher: u64::from(firm + 1),
                discovered: 0,
                mastered: 0,
                permitted: 31,
                adopted: 0,
                consent: research::ResearchConsent::Granted,
                production_requirements: 0,
                adoption: research::AdoptionPolicy::MasterAndAdoptPermitted,
                methods: ProductionMethod::for_good(generated_good(&config, firm as usize)),
                projects: Technology::ALL
                    .into_iter()
                    .enumerate()
                    .map(|(index, technology)| research::ResearchProject {
                        identity: 10_000_000 + u64::from(firm) * 5 + index as u64,
                        prerequisites: technology.prerequisites(),
                        outcomes: technology as u32,
                        inputs: [0; Good::COUNT],
                        labor_minutes: 30,
                        minimum_effort: 300,
                        pioneer_effort: 600,
                        capacity: 1000,
                        relatedness: 500,
                        shape: 1,
                        exposure: 0,
                        exposure_signal: 50,
                    })
                    .collect(),
            })
            .collect();
        config.research = Some(research::ResearchConfig {
            labs,
            ..Default::default()
        });
        Self::new(config)
    }

    pub fn goods_markets(&self) -> Vec<Market> {
        let mut inputs = std::collections::BTreeSet::new();
        for firm in self.config.households as usize..self.actors() {
            let good = Good::from_index(self.actor(firm).kind).expect("validated firm");
            inputs.extend(good.inputs().into_iter().flatten().map(|good| good as u32));
        }
        if let Some(research) = &self.config.research {
            for lab in &research.labs {
                let firm = self.index_of(lab.owner).expect("validated lab");
                let good = Good::from_index(self.actor(firm).kind).expect("validated firm");
                for method in &lab.methods {
                    inputs.extend(
                        method
                            .recipe(good, self.technique(firm))
                            .inputs
                            .into_iter()
                            .flatten()
                            .map(|good| good as u32),
                    );
                }
            }
        }
        inputs.extend(
            self.config
                .distributors
                .iter()
                .map(|store| store.good as u32),
        );
        inputs.insert(Good::Bread as u32);
        inputs.insert(Good::Tools as u32);
        inputs.insert(Good::Wood as u32);
        if self.config.economy == Economy::Industrial {
            inputs.extend(Good::ALL.map(|good| good as u32));
        }
        inputs
            .into_iter()
            .map(|index| Market::Input(Good::from_index(index).expect("catalogue good")))
            .chain(std::iter::once(Market::Household))
            .chain(
                Good::ALL
                    .into_iter()
                    .filter(|good| {
                        self.config.economy == Economy::Industrial
                            && good.consumption_interval() > 1
                    })
                    .map(Market::Consumer),
            )
            .collect()
    }
    pub fn new(mut config: ScenarioConfig) -> Result<Self> {
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
        config.behavior = BehaviorPolicy::new(config.behavior)?;
        config.population = config
            .population
            .take()
            .map(|population| Population::new(population, &config))
            .transpose()?;
        let bounds = [
            config.version == SCENARIO_VERSION,
            (1..=1_000_000).contains(&config.households),
            (3..=20_000).contains(&config.firms),
            (1..=100_000).contains(&config.map_size),
            (1..=config.map_size).contains(&config.search_radius),
            config.commute_speed > 0,
            config.shipment_speed > 0,
            config.household_need > 0,
            config.household_need <= 100,
            (1..=10_000).contains(&config.capacity),
            (1..=100).contains(&config.packet_output),
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
            config.delivery_penalty <= 10_000,
            (1..=1_048_576).contains(&config.batch_size),
        ];
        match bounds.into_iter().all(std::convert::identity) {
            false => Err(anyhow!("scenario parameter outside DeterministicV1 bounds")),
            true => {
                let retailers = config
                    .distributors
                    .iter()
                    .cloned()
                    .map(|store| (store.firm, store))
                    .collect::<std::collections::BTreeMap<_, _>>();
                match retailers.len() == config.distributors.len() {
                    true => (),
                    false => Err(anyhow!("distributor identities must be unique"))?,
                }
                let mut firm_counts = [0u32; Good::COUNT];
                for firm in 0..config.firms as usize {
                    if !retailers.contains_key(&(u64::from(config.households) + firm as u64 + 1)) {
                        firm_counts[generated_good(&config, firm) as usize - 1] += 1;
                    }
                }
                let demand = supply::StartingDemand::new(&config);
                let daily = std::array::from_fn::<_, { Good::COUNT }, _>(|good| {
                    demand.daily[good]
                        .div_ceil(u64::from(firm_counts[good].max(1)))
                        .max(1)
                });
                let capacities = std::array::from_fn(|good| match config.economy {
                    Economy::Agrarian => config.capacity,
                    Economy::Industrial => (daily[good].div_ceil(config.packet_output.into())
                        as u32)
                        .max(1)
                        .min(config.capacity),
                });
                let starting_stocks = std::array::from_fn(|good| match config.economy {
                    Economy::Agrarian => {
                        u64::from(config.capacity)
                            * u64::from(config.packet_output)
                            * u64::from(config.buffer_days)
                    }
                    Economy::Industrial => daily[good] * u64::from(config.buffer_days),
                });
                let totals = match &config.population {
                    Some(population) => population.totals()?,
                    None => {
                        let cash = u128::from(config.household_cash)
                            * u128::from(config.households)
                            + u128::from(config.firm_cash) * u128::from(config.firms);

                        let mut goods = [0; Good::COUNT];
                        goods[Good::Bread as usize - 1] = u64::from(config.households)
                            * u64::from(config.household_need)
                            * u64::from(config.demography.initial_household_members)
                            * u64::from(config.buffer_days);
                        if let enterprise::HouseholdProduction::Available {
                            annual_yield_per_plot,
                            ..
                        } = config.household_production
                        {
                            let farms = u64::from(config.households.div_ceil(2));
                            goods[Good::Grain as usize - 1] = farms
                                * u64::from(annual_yield_per_plot)
                                * u64::from(
                                    config.demography.initial_household_members.div_ceil(2),
                                );
                            goods[Good::Tools as usize - 1] = farms;
                        }
                        for firm in 0..config.firms as usize {
                            let good = generated_good(&config, firm);
                            let retailer =
                                retailers.get(&(u64::from(config.households) + firm as u64 + 1));
                            let stock = match retailer {
                                Some(store) => {
                                    daily[good as usize - 1]
                                        * u64::from(config.buffer_days.min(store.reserve_days))
                                }
                                None => starting_stocks[good as usize - 1],
                            };
                            goods[good as usize - 1] = add(goods[good as usize - 1], stock)?;
                            for input in good
                                .inputs()
                                .into_iter()
                                .flatten()
                                .filter(|_| retailer.is_none())
                            {
                                goods[input as usize - 1] = add(goods[input as usize - 1], stock)?;
                            }
                        }
                        InitialTotals {
                            cash: u64::try_from(cash)
                                .context("initial cash exceeds tick capacity")?,
                            goods,
                        }
                    }
                };
                let resources = resources::ResourceGeography::new(&config)?;
                let mut scenario = Self {
                    config,
                    totals,
                    resources,
                    capacities,
                    starting_stocks,
                    retailers,
                    retail_demand: daily,
                    initial_actors: std::sync::Arc::default(),
                };
                scenario.initial_actors = std::sync::Arc::new(
                    (0..scenario.actors())
                        .map(|index| scenario.actor(index))
                        .collect(),
                );
                if scenario.config.finance.is_some() || scenario.config.research.is_some() {
                    let actors = (0..scenario.actors())
                        .map(|index| scenario.actor(index))
                        .collect::<ActorRows>();
                    scenario.config.research = scenario
                        .config
                        .research
                        .take()
                        .map(|config| research::ResearchConfig::new(config, &actors))
                        .transpose()?;
                    let identities = actors
                        .iter()
                        .map(|actor| actor.identity)
                        .collect::<Vec<_>>();
                    scenario.config.finance = scenario
                        .config
                        .finance
                        .take()
                        .map(|config| {
                            finance::FinanceConfig::new(
                                config,
                                &identities,
                                &identities[..scenario.config.households as usize],
                            )
                        })
                        .transpose()?;
                    let financial_cash = scenario
                        .config
                        .finance
                        .as_ref()
                        .map(finance::FinanceConfig::initial_cash)
                        .transpose()?
                        .unwrap_or(0);
                    scenario.totals.cash = add(scenario.totals.cash, financial_cash)?;
                }
                let ring = scenario.ring();
                let side = scenario
                    .config
                    .map_size
                    .div_ceil(scenario.config.search_radius);
                match side <= 256 && ring <= 65 {
                    true => Ok(scenario),
                    false => Err(anyhow!("scenario exceeds transit or spatial capacity")),
                }
            }
        }
    }
    pub fn baseline(households: u32, seed: u64, experiment: Experiment) -> Result<Self> {
        let firms = (households / 100).max(3);
        let map_size = match households {
            0..=10_000 => 1000,
            10_001..=100_000 => 3000,
            _ => 10_000,
        };
        let mut config = ScenarioConfig {
            economy: Economy::Agrarian,
            version: SCENARIO_VERSION,
            seed,
            households,
            firms,
            map_size,
            resources: ResourceMap::Generated,
            search_radius: 1000,
            commute_speed: 50,
            shipment_speed: 1000,
            household_cash: 5000,
            firm_cash: 200_000,
            buffer_days: 4,
            household_need: 1,
            capacity: 100,
            packet_output: 3,
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
            household_policy: None,
            finance: None,
            research: None,
            bargaining: None,
            demography: demography::DemographicPolicy::default(),
            household_production: enterprise::HouseholdProduction::Unavailable,
            distributors: Vec::new(),
            ownership: Vec::new(),
            land_access: None,
            working_conditions: Vec::new(),
        };
        match experiment {
            Experiment::Shortage => config.buffer_days = 0,
            Experiment::Transport => config.shipment_speed = 100,
            Experiment::Baseline | Experiment::Concentrated => (),
        }
        Self::new(config)
    }
    pub fn config(&self) -> &ScenarioConfig {
        &self.config
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
        match (actor.kind, self.retailers.contains_key(&actor.identity)) {
            (0, _) | (_, true) => actor.capacity,
            _ => match self.production_site(
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
    pub fn actors(&self) -> usize {
        (self.config.households + self.config.firms) as usize
    }
    pub fn ring(&self) -> usize {
        self.config
            .search_radius
            .div_ceil(self.config.shipment_speed)
            .max(1) as usize
            + 1
    }
    pub fn fields(&self) -> usize {
        BASE_FIELDS + self.ring() * Good::COUNT
    }
    pub fn state_len(&self) -> usize {
        self.actors() * self.fields()
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
            + actors * 72
            + (actors.div_ceil(128) + 1) * 4
            + self.config.demography.people.len().max(
                self.config.households as usize
                    * self.config.demography.initial_household_members as usize,
            ) as u64
                * 144
            + actors * 24
            + batch * (16 * 8 * 4 + 16 * 4 * 3 + 28)
            + 128 * 1024 * 1024
    }
    pub fn fingerprint(&self) -> Result<String> {
        Ok(hex(&Sha256::digest(serde_json::to_vec(&self.config)?)))
    }
    pub fn index_of(&self, identity: u64) -> Option<usize> {
        match &self.config.population {
            Some(population) => population.index_of(identity),
            None => (1..=self.actors() as u64)
                .contains(&identity)
                .then(|| (identity - 1) as usize),
        }
    }
    pub fn identity(&self, index: usize) -> Option<u64> {
        match &self.config.population {
            Some(population) => population.identity(index),
            None => (index < self.actors()).then_some(index as u64 + 1),
        }
    }
    pub fn next_identity(&self) -> u64 {
        self.config
            .population
            .as_ref()
            .map(Population::next_identity)
            .unwrap_or(self.actors() as u64 + 1)
    }
    pub fn technique(&self, index: usize) -> Technique {
        match (
            &self.config.population,
            index.checked_sub(self.config.households as usize),
        ) {
            (Some(population), Some(firm)) => population.firms[firm].technique,
            _ => Technique {
                labor_minutes: self.config.packet_minutes,
                input_ticks: match index.checked_sub(self.config.households as usize) {
                    Some(_)
                        if Good::from_index(self.actor(index).kind)
                            .expect("firm good")
                            .inputs()[0]
                            .is_none() =>
                    {
                        0
                    }
                    _ => self.config.packet_output,
                },
                output_ticks: self.config.packet_output,
            },
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
                true => Ok(method.recipe(good, technique)),
                false => Err(anyhow!("production method is not configured for this firm")),
            },
        }
    }
    pub fn initial(&self) -> Result<State> {
        let mut state = match &self.config.population {
            Some(population) => population.initial(self),
            None => self.generated(),
        }?;
        if let Some(finance) = &self.config.finance {
            state.values.extend(finance.initial_values());
            state.initial_cash = add(state.initial_cash, finance.initial_cash()?)?;
        }
        for actor in self.config.households as usize..self.actors() {
            let good = Good::from_index(state.actors.at(actor).kind)?;
            state.set_recipe(
                actor,
                industry::Recipe {
                    technique: self.technique(actor),
                    inputs: good.inputs(),
                },
            );
        }
        state.ownership = state
            .actors
            .iter()
            .filter(|actor| actor.kind > 0)
            .map(|actor| enterprise::Ownership {
                firm: actor.identity,
                shares: vec![enterprise::OwnershipShare {
                    household: state.actors.at(actor.owner as usize).identity,
                    units: 1_000_000,
                }],
            })
            .collect();
        for ownership in &self.config.ownership {
            let index = state
                .ownership
                .iter()
                .position(|entry| entry.firm == ownership.firm)
                .context("unknown ownership firm")?;
            for share in &ownership.shares {
                match self
                    .index_of(share.household)
                    .is_some_and(|index| state.actors.at(index).kind == 0)
                {
                    true => (),
                    false => Err(anyhow!("ownership references a missing household"))?,
                }
            }
            state.ownership[index] = ownership.clone();
        }
        for actor in self.config.households as usize..self.actors() {
            state.set(
                Field::EnterpriseStatus,
                actor,
                enterprise::EnterpriseState::Operating as u64,
            );
            let owner = state.actors.at(actor).owner as usize;
            state.set(Field::InvestmentFirm, owner, actor as u64 + 1);
        }
        for distributor in &self.config.distributors {
            let firm = self
                .index_of(distributor.firm)
                .context("unknown distributor identity")?;
            match state.actors.at(firm).kind == distributor.good as u32 {
                true => (),
                false => Err(anyhow!("distributor inventory good must match its firm"))?,
            }
            state.set(
                Field::EnterpriseModel,
                firm,
                enterprise::EnterpriseModel::Distributor as u64,
            );
            state.set(Field::StockCoverage, firm, distributor.reserve_days.into());
            state.set(
                Field::RetailMargin,
                firm,
                distributor.margin_per_mille.into(),
            );
            state.set(Field::ReplacementCost, firm, state.get(Field::Price, firm));
            state.set(
                Field::Demand,
                firm,
                u64::from(state.actors.at(firm).capacity) * u64::from(distributor.units_per_shift),
            );
            state.set_recipe(
                firm,
                industry::Recipe {
                    technique: Technique::new(
                        distributor.handling_minutes,
                        0,
                        distributor.units_per_shift,
                    )?,
                    inputs: [None, None],
                },
            );
        }
        for actor in self.config.households as usize..self.actors() {
            state.retain_basic_recipe(actor);
            for (field, input) in [
                (Field::FirstInputCost, Field::FirstInput),
                (Field::SecondInputCost, Field::SecondInput),
            ] {
                let good = state.get(input, actor);
                if good > 0 {
                    let price = match good {
                        1..=3 => self.config.prices[good as usize - 1],
                        _ => Good::from_index(good as u32)?.initial_price(),
                    };
                    state.set(field, actor, price.into());
                }
            }
        }
        if let Some(land) = &self.config.land_access {
            for index in 0..state.actors.len() {
                state.set(Field::LandPlots, index, 0);
            }
            for grant in &land.allocations {
                let index = self
                    .index_of(grant.household)
                    .filter(|&index| state.actors.at(index).kind == 0)
                    .context("land grant references a missing household")?;
                state.set(Field::LandPlots, index, grant.plots.into());
            }
        }
        state.people = self.config.demography.initial(&state)?;
        state.initial_population = state.people.len() as u64;
        for actor in 0..state.actors.len() {
            state.set(Field::EquipmentCondition, actor, 1000);
            if state.actors.at(actor).kind == 0 {
                state.set(Field::Price, actor, self.config.prices[2].into());
            }
        }
        for setting in &self.config.working_conditions {
            let firm = self
                .index_of(setting.firm)
                .filter(|&index| state.actors.at(index).kind > 0)
                .context("working conditions reference a missing firm")?;
            state.set(
                Field::EquipmentCondition,
                firm,
                setting.conditions.equipment_condition.into(),
            );
            state.set(
                Field::SafetyProvision,
                firm,
                setting.conditions.safety_provision.into(),
            );
        }
        state.next_identity = state
            .people
            .iter()
            .map(|person| person.identity)
            .max()
            .unwrap_or(0)
            .max(state.next_identity - 1)
            .checked_add(1)
            .context("person identity capacity exhausted")?;
        Ok(state)
    }
    fn actor(&self, index: usize) -> Actor {
        match self.initial_actors.get(index) {
            Some(actor) => actor,
            None => self.generate_actor(index),
        }
    }
    fn generate_actor(&self, index: usize) -> Actor {
        match &self.config.population {
            Some(population) => population.actor(index),
            None => {
                let config = &self.config;
                let identity = index as u64 + 1;
                let kind = match index < config.households as usize {
                    true => 0,
                    false => generated_good(config, index - config.households as usize) as u32,
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
                let position = match (kind, retailer) {
                    (0, _) => match (config.household_production, index % 2) {
                        (enterprise::HouseholdProduction::Available { .. }, 0) => self
                            .resources
                            .place(Good::Grain, identity, config, position),
                        _ => position,
                    },
                    (_, Some(_)) => position,
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
                    owner: index as u32 % config.households,
                    capacity: match (kind, retailer) {
                        (0, _) => 1,
                        (_, Some(store)) => self.retail_demand[kind as usize - 1]
                            .div_ceil(u64::from(store.units_per_shift))
                            .max(1)
                            .min(u64::from(config.capacity))
                            as u32,
                        _ => self.capacities[kind as usize - 1],
                    },
                }
            }
        }
    }
    fn generated(&self) -> Result<State> {
        let actors = self.actors();
        let config = &self.config;
        let properties = (0..actors)
            .map(|index| self.actor(index))
            .collect::<ActorRows>();
        let mut state = State {
            day: 0,
            next_identity: actors as u64 + 1,
            actors: properties,
            values: vec![0; self.fields() * actors],
            initial_cash: 0,
            initial_goods: [0; Good::COUNT],
            research: None,
            people: Vec::new(),
            initial_population: 0,
            ownership: Vec::new(),
        };
        (0..actors).try_for_each(|actor| -> Result<()> {
            let kind = state.actors.at(actor).kind;
            let retailer = self.retailers.get(&state.actors.at(actor).identity);
            let cash = match kind {
                0 => config.household_cash,
                _ => config.firm_cash,
            };
            state.set(Field::Cash, actor, cash);
            if let (
                0,
                enterprise::HouseholdProduction::Available {
                    annual_yield_per_plot,
                    ..
                },
                0,
            ) = (kind, config.household_production, actor % 2)
            {
                let plots = u64::from(config.demography.initial_household_members.div_ceil(2));
                let harvest = u64::from(annual_yield_per_plot) * plots;
                state.set(Field::Grain, actor, harvest);
                state.set(Field::Tools, actor, 1);
                state.set(Field::LandPlots, actor, plots);
                state.initial_goods[Good::Grain as usize - 1] =
                    add(state.initial_goods[Good::Grain as usize - 1], harvest)?;
                state.initial_goods[Good::Tools as usize - 1] =
                    add(state.initial_goods[Good::Tools as usize - 1], 1)?;
            }
            state.set(
                Field::EnterpriseStatus,
                actor,
                enterprise::EnterpriseState::Operating as u64,
            );
            if kind == 0 {
                state.set(Field::Price, actor, config.prices[2].into());
            }

            state.set(Field::Wage, actor, u64::from(config.wage));
            let stock = match (kind, retailer) {
                (0, _) => {
                    u64::from(config.household_need)
                        * u64::from(config.demography.initial_household_members)
                        * u64::from(config.buffer_days)
                }
                (_, Some(store)) => {
                    self.retail_demand[kind as usize - 1]
                        * u64::from(config.buffer_days.min(store.reserve_days))
                }
                _ => self.starting_stocks[kind as usize - 1],
            };
            let output = match kind {
                0 => Good::Bread,
                _ => Good::from_index(kind)?,
            };
            state.set(Field::inventory(output), actor, stock);
            state.initial_goods[output as usize - 1] =
                add(state.initial_goods[output as usize - 1], stock)?;
            for input in output
                .inputs()
                .into_iter()
                .flatten()
                .filter(|_| kind != 0 && retailer.is_none())
            {
                state.set(Field::inventory(input), actor, stock);
                state.initial_goods[input as usize - 1] =
                    add(state.initial_goods[input as usize - 1], stock)?;
            }
            match kind {
                0 => (),
                _ => {
                    state.set(
                        Field::Price,
                        actor,
                        u64::from(match kind {
                            1..=3 => config.prices[kind as usize - 1],
                            _ => output
                                .initial_price()
                                .clamp(config.minimum_price, config.maximum_price),
                        }),
                    );
                    state.set(
                        Field::Demand,
                        actor,
                        u64::from(state.actors.at(actor).capacity)
                            * u64::from(config.packet_output),
                    );
                }
            }
            state.initial_cash = add(state.initial_cash, cash)?;
            Ok(())
        })?;
        Ok(state)
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Position {
    pub x: u32,
    pub y: u32,
}
impl Position {
    pub fn distance(self, other: Self) -> u32 {
        self.x.abs_diff(other.x) + self.y.abs_diff(other.y)
    }
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Actor {
    pub identity: u64,
    pub position: Position,
    pub kind: u32,
    pub owner: u32,
    pub capacity: u32,
}
#[derive(Clone, Debug, Default, Serialize, PartialEq, Eq)]
pub struct ActorRows {
    identity: Vec<u64>,
    x: Vec<u32>,
    y: Vec<u32>,
    kind: Vec<u32>,
    owner: Vec<u32>,
    capacity: Vec<u32>,
}
impl ActorRows {
    pub fn len(&self) -> usize {
        self.identity.len()
    }
    pub fn is_empty(&self) -> bool {
        self.identity.is_empty()
    }
    pub fn at(&self, index: usize) -> Actor {
        Actor {
            identity: self.identity[index],
            position: Position {
                x: self.x[index],
                y: self.y[index],
            },
            kind: self.kind[index],
            owner: self.owner[index],
            capacity: self.capacity[index],
        }
    }
    pub fn get(&self, index: usize) -> Option<Actor> {
        (index < self.len()).then(|| self.at(index))
    }
    pub fn set(&mut self, index: usize, actor: Actor) {
        self.identity[index] = actor.identity;
        self.x[index] = actor.position.x;
        self.y[index] = actor.position.y;
        self.kind[index] = actor.kind;
        self.owner[index] = actor.owner;
        self.capacity[index] = actor.capacity;
    }
    pub fn iter(&self) -> impl ExactSizeIterator<Item = Actor> + '_ {
        (0..self.len()).map(|index| self.at(index))
    }
}
impl FromIterator<Actor> for ActorRows {
    fn from_iter<T: IntoIterator<Item = Actor>>(actors: T) -> Self {
        actors.into_iter().fold(Self::default(), |mut rows, actor| {
            rows.identity.push(actor.identity);
            rows.x.push(actor.position.x);
            rows.y.push(actor.position.y);
            rows.kind.push(actor.kind);
            rows.owner.push(actor.owner);
            rows.capacity.push(actor.capacity);
            rows
        })
    }
}
impl<'de> Deserialize<'de> for ActorRows {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Rows {
            identity: Vec<u64>,
            x: Vec<u32>,
            y: Vec<u32>,
            kind: Vec<u32>,
            owner: Vec<u32>,
            capacity: Vec<u32>,
        }
        let rows = Rows::deserialize(deserializer)?;
        match [
            rows.x.len(),
            rows.y.len(),
            rows.kind.len(),
            rows.owner.len(),
            rows.capacity.len(),
        ]
        .into_iter()
        .all(|len| len == rows.identity.len())
        {
            true => Ok(Self {
                identity: rows.identity,
                x: rows.x,
                y: rows.y,
                kind: rows.kind,
                owner: rows.owner,
                capacity: rows.capacity,
            }),
            false => Err(serde::de::Error::custom(
                "actor rows have different lengths",
            )),
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct State {
    pub day: u64,
    pub next_identity: u64,
    pub actors: ActorRows,
    pub values: Vec<u64>,
    pub initial_cash: u64,
    pub initial_goods: [u64; Good::COUNT],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub research: Option<research::ResearchSnapshot>,
    pub people: Vec<demography::Person>,
    pub initial_population: u64,
    pub ownership: Vec<enterprise::Ownership>,
}
impl State {
    pub fn retain_basic_recipe(&mut self, actor: usize) {
        for (basic, current) in [
            (Field::BasicLaborMinutes, Field::LaborMinutes),
            (Field::BasicInputTicks, Field::InputTicks),
            (Field::BasicOutputTicks, Field::OutputTicks),
            (Field::BasicFirstInput, Field::FirstInput),
            (Field::BasicSecondInput, Field::SecondInput),
        ] {
            self.set(basic, actor, self.get(current, actor));
        }
    }
    pub fn set_recipe(&mut self, actor: usize, recipe: industry::Recipe) {
        self.set(
            Field::LaborMinutes,
            actor,
            recipe.technique.labor_minutes.into(),
        );
        self.set(
            Field::InputTicks,
            actor,
            recipe.technique.input_ticks.into(),
        );
        self.set(
            Field::OutputTicks,
            actor,
            recipe.technique.output_ticks.into(),
        );
        self.set(
            Field::FirstInput,
            actor,
            recipe.inputs[0].map_or(0, |good| good as u64),
        );
        self.set(
            Field::SecondInput,
            actor,
            recipe.inputs[1].map_or(0, |good| good as u64),
        );
    }

    pub fn shipment_coordinate(
        &self,
        good: Good,
        slot: usize,
        ring: usize,
        actor: usize,
    ) -> Coordinate {
        Coordinate(
            ((BASE_FIELDS + (good as usize - 1) * ring + slot) * self.actors.len() + actor) as u32,
        )
    }
    pub fn coordinate(&self, field: Field, actor: usize) -> Coordinate {
        Coordinate((field as usize * self.actors.len() + actor) as u32)
    }
    pub fn get(&self, field: Field, actor: usize) -> u64 {
        self.values[self.coordinate(field, actor).0 as usize]
    }
    pub fn set(&mut self, field: Field, actor: usize, value: u64) {
        let coordinate = self.coordinate(field, actor);
        self.values[coordinate.0 as usize] = value;
    }
    pub fn ring_coordinate(&self, slot: usize, actor: usize) -> Coordinate {
        Coordinate(((BASE_FIELDS + slot) * self.actors.len() + actor) as u32)
    }
    pub fn digest(&self) -> String {
        let mut hash = Sha256::new();
        hash.update(STATE_HASH_VERSION.as_bytes());
        hash.update([0]);
        hash.update(self.day.to_le_bytes());
        hash.update(self.next_identity.to_le_bytes());
        hash.update(self.initial_cash.to_le_bytes());
        hash.update(self.initial_population.to_le_bytes());
        self.initial_goods
            .iter()
            .for_each(|value| hash.update(value.to_le_bytes()));
        hash.update((self.actors.len() as u64).to_le_bytes());
        self.actors.iter().for_each(|actor| {
            hash.update(actor.identity.to_le_bytes());
            hash.update(actor.position.x.to_le_bytes());
            hash.update(actor.position.y.to_le_bytes());
            hash.update(actor.kind.to_le_bytes());
            hash.update(actor.owner.to_le_bytes());
            hash.update(actor.capacity.to_le_bytes());
        });
        hash.update(serde_json::to_vec(&self.ownership).expect("serializable ownership registry"));
        hash.update(serde_json::to_vec(&self.people).expect("serializable person registry"));
        hash.update((self.values.len() as u64).to_le_bytes());
        self.values
            .iter()
            .for_each(|value| hash.update(value.to_le_bytes()));
        match &self.research {
            None => hash.update([0]),
            Some(research) => {
                hash.update([1]);
                hash.update((research.profile.len() as u64).to_le_bytes());
                hash.update(research.profile.as_bytes());
                hash.update((research.table_hash.len() as u64).to_le_bytes());
                hash.update(research.table_hash.as_bytes());
                hash.update((research.configuration_hash.len() as u64).to_le_bytes());
                hash.update(research.configuration_hash.as_bytes());
                hash.update((research.words.len() as u64).to_le_bytes());
                research
                    .words
                    .iter()
                    .for_each(|word| hash.update(word.to_le_bytes()));
                research
                    .summary
                    .consumed
                    .iter()
                    .chain([
                        &research.summary.attempts,
                        &research.summary.discoveries,
                        &research.summary.remaining_service,
                        &research.summary.natural_stock,
                        &research.summary.striking,
                    ])
                    .for_each(|value| hash.update(value.to_le_bytes()));
            }
        }
        hex(&hash.finalize())
    }
}

#[enumflags2::bitflags]
#[repr(u32)]
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum Phase {
    Opening,
    Observation,
    Proposal,
    Commitment,
    WorkPayment,
    ProvisionPurchase,
    PhysicalClose,
    AccountsDecisions,
    Audit,
}
impl Phase {
    pub fn next(self) -> Self {
        match self {
            Self::Opening => Self::Observation,
            Self::Observation => Self::Proposal,
            Self::Proposal => Self::Commitment,
            Self::Commitment => Self::WorkPayment,
            Self::WorkPayment => Self::ProvisionPurchase,
            Self::ProvisionPurchase => Self::PhysicalClose,
            Self::PhysicalClose => Self::AccountsDecisions,
            Self::AccountsDecisions => Self::Audit,
            Self::Audit => Self::Opening,
        }
    }
    pub const ALL: [Self; 9] = [
        Self::Opening,
        Self::Observation,
        Self::Proposal,
        Self::Commitment,
        Self::WorkPayment,
        Self::ProvisionPurchase,
        Self::PhysicalClose,
        Self::AccountsDecisions,
        Self::Audit,
    ];
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(try_from = "CheckpointDocument", deny_unknown_fields)]
pub struct Checkpoint {
    version: u32,
    profile: String,
    table_hash: String,
    scenario: ScenarioConfig,
    state: State,
    checksum: String,
}
impl Checkpoint {
    pub fn new(scenario: &Scenario, state: &State) -> Result<Self> {
        let _ = ObservationSnapshot::from_state(scenario, state)?;
        let checksum = Self::checksum(scenario.config(), state)?;
        Ok(Self {
            version: 6,
            profile: PROFILE.into(),
            table_hash: table_hash(),
            scenario: scenario.config.clone(),
            state: state.clone(),
            checksum,
        })
    }
    fn checksum(config: &ScenarioConfig, state: &State) -> Result<String> {
        Ok(hex(&Sha256::digest(serde_json::to_vec(&(config, state))?)))
    }
    pub fn restore(&self) -> Result<Restored> {
        let scenario = Scenario::new(self.scenario.clone())?;
        let valid = [
            self.version == 6,
            self.profile == PROFILE,
            self.table_hash == table_hash(),
            self.state.values.len()
                == self.state.actors.len() * scenario.fields()
                    + scenario
                        .config
                        .finance
                        .as_ref()
                        .map_or(0, finance::FinanceConfig::state_len),
            self.state.actors.len() >= scenario.actors(),
            self.checksum == Self::checksum(&self.scenario, &self.state)?,
        ];
        match valid.into_iter().all(std::convert::identity) {
            true => {
                let _ = ObservationSnapshot::from_state(&scenario, &self.state)?;
                Ok(Restored {
                    scenario,
                    state: self.state.clone(),
                })
            }
            false => Err(anyhow!(
                "checkpoint version, shape, profile, table hash, or checksum mismatch"
            )),
        }
    }
}
pub struct Restored {
    pub scenario: Scenario,
    pub state: State,
}

#[derive(Clone, Debug)]
pub struct InspectionRequest {
    identity: u64,
}
impl InspectionRequest {
    pub fn from_registry(identity: u64, identities: &[u64]) -> Result<Self> {
        match identities.contains(&identity) {
            true => Ok(Self { identity }),
            false => Err(anyhow!("unknown actor identity")),
        }
    }
    pub fn new(identity: u64, scenario: &Scenario) -> Result<Self> {
        match scenario.index_of(identity).is_some() {
            true => Ok(Self { identity }),
            false => Err(anyhow!("unknown actor identity")),
        }
    }
    pub fn identity(&self) -> u64 {
        self.identity
    }
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Inspection {
    pub production_method: Option<ProductionMethod>,
    pub labor_minutes: u64,
    pub output_ticks: u64,
    pub actor: Actor,
    pub day: u64,
    pub cash: u64,
    pub gross_income: u64,
    pub payroll_reserve: u64,
    pub income_tax: u64,
    pub employee_charges: u64,
    pub inventory: [u64; Good::COUNT],
    pub transit: u64,
    pub fatigue: u64,
    pub employer: u64,
    pub owner_identity: u64,
    pub available_minutes: u64,
    pub living_members: u64,
    pub equipment_condition: u64,
    pub safety_provision: u64,
    pub enterprise_model: enterprise::EnterpriseModel,
    pub enterprise_state: Option<enterprise::EnterpriseState>,
    pub construction_minutes_remaining: u64,
    pub pending_capacity: u64,
    pub wage: u64,
    pub price: u64,
    pub planned_packets: u64,
    pub open_slots: u64,
}
impl Inspection {
    pub fn from_state(state: &State, request: &InspectionRequest) -> Result<Self> {
        let index = state
            .actors
            .iter()
            .position(|actor| actor.identity == request.identity)
            .context("unknown actor identity")?;
        Ok(Self {
            actor: state.actors.at(index),
            production_method: ProductionMethod::from_index(state.get(Field::ActiveMethod, index))?,
            labor_minutes: state.get(Field::LaborMinutes, index),
            output_ticks: state.get(Field::OutputTicks, index),
            day: state.day,
            cash: state.get(Field::Cash, index),
            gross_income: state.get(Field::GrossIncome, index),
            payroll_reserve: state.get(Field::PayrollReserve, index),
            income_tax: state.get(Field::TaxTreasury, index),
            employee_charges: state.get(Field::ChargeTreasury, index),
            inventory: Good::ALL.map(|good| state.get(Field::inventory(good), index)),
            transit: state.get(Field::Transit, index),
            fatigue: state.get(Field::Fatigue, index),
            employer: state
                .get(Field::Employer, index)
                .checked_sub(1)
                .and_then(|employer| state.actors.get(employer as usize))
                .map(|actor| actor.identity)
                .unwrap_or(0),
            owner_identity: state
                .actors
                .at(state.actors.at(index).owner as usize)
                .identity,
            available_minutes: state.get(Field::Time, index),
            living_members: state.get(Field::LivingMembers, index),
            equipment_condition: state.get(Field::EquipmentCondition, index),
            safety_provision: state.get(Field::SafetyProvision, index),
            enterprise_model: match state.get(Field::EnterpriseModel, index) {
                1 => enterprise::EnterpriseModel::Distributor,
                _ => enterprise::EnterpriseModel::Producer,
            },
            enterprise_state: match state.actors.at(index).kind {
                0 => None,
                _ => Some(enterprise::EnterpriseState::from_index(
                    state.get(Field::EnterpriseStatus, index),
                )?),
            },
            construction_minutes_remaining: state.get(Field::ConstructionRemaining, index),
            pending_capacity: state.get(Field::PendingCapacity, index),
            wage: state.get(Field::Wage, index),
            price: state.get(Field::Price, index),
            planned_packets: state.get(Field::Plan, index),
            open_slots: state.get(Field::Slots, index),
        })
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct QuoteDistribution {
    pub count: u64,
    pub minimum: u64,
    pub maximum: u64,
    pub total: u64,
    pub squares: u64,
}
impl QuoteDistribution {
    pub fn include(&mut self, quote: u64) -> Result<()> {
        self.minimum = match self.count {
            0 => quote,
            _ => self.minimum.min(quote),
        };
        self.maximum = self.maximum.max(quote);
        self.count = add(self.count, 1)?;
        self.total = add(self.total, quote)?;
        self.squares = add(
            self.squares,
            quote.checked_mul(quote).context("quote square overflow")?,
        )?;
        Ok(())
    }
    pub fn mean(&self) -> f64 {
        self.total as f64 / self.count.max(1) as f64
    }
    pub fn deviation(&self) -> f64 {
        let numerator = u128::from(self.count) * u128::from(self.squares)
            - u128::from(self.total) * u128::from(self.total);
        (numerator as f64).sqrt() / self.count.max(1) as f64
    }
}
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct MarketObservation {
    pub wages: QuoteDistribution,
    pub prices: QuoteDistribution,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ObservationSnapshot {
    pub day: u64,
    pub cash: u64,
    pub payroll_reserves: u64,
    pub income_tax: u64,
    pub employee_charges: u64,
    pub financial_cash: u64,
    pub loan_principal: u64,
    pub loan_interest: u64,
    pub active_loans: u64,
    pub overdue_loans: u64,
    pub research: research::ResearchSummary,
    pub employed: u64,
    pub households: u64,
    pub firms: u64,
    pub firm_entries: u64,
    pub firm_closures: u64,
    pub distributors: u64,
    pub household_production: [u64; Good::COUNT],
    pub shortages: u64,
    pub population: demography::PopulationObservation,
    pub inventories: [u64; Good::COUNT],
    pub transit: [u64; Good::COUNT],
    pub produced: [u64; Good::COUNT],
    pub used: [u64; Good::COUNT],
    pub consumed: u64,
    pub consumption: [u64; Good::COUNT],
    pub minimum_wage: u64,
    pub maximum_wage: u64,
    pub minimum_price: u64,
    pub maximum_price: u64,
    pub markets: [MarketObservation; Good::COUNT],
    pub wealth_bins: Vec<u64>,
    pub regions: Vec<u64>,
}
impl ObservationSnapshot {
    pub fn from_state(scenario: &Scenario, state: &State) -> Result<Self> {
        let admitted = AuditedState::new(scenario, state)?;
        let state = admitted.state;
        let mut snapshot = Self {
            day: state.day,
            research: state
                .research
                .as_ref()
                .map(|research| research.summary.clone())
                .unwrap_or_default(),
            households: state.actors.iter().filter(|actor| actor.kind == 0).count() as u64,
            population: demography::PopulationObservation::from_state(
                state,
                &scenario.config.demography,
            ),
            minimum_wage: u64::MAX,
            minimum_price: u64::MAX,
            wealth_bins: vec![0; 64],
            regions: vec![0; 256],
            ..Self::default()
        };
        state
            .actors
            .iter()
            .enumerate()
            .try_for_each(|(i, actor)| -> Result<()> {
                snapshot.cash = add(
                    snapshot.cash,
                    add(state.get(Field::Cash, i), state.get(Field::Escrow, i))?,
                )?;
                let reserve = state.get(Field::PayrollReserve, i);
                let tax = state.get(Field::TaxTreasury, i);
                let charge = state.get(Field::ChargeTreasury, i);
                snapshot.payroll_reserves = add(snapshot.payroll_reserves, reserve)?;
                snapshot.income_tax = add(snapshot.income_tax, tax)?;
                snapshot.employee_charges = add(snapshot.employee_charges, charge)?;
                snapshot.cash = add(snapshot.cash, add(reserve, add(tax, charge)?)?)?;
                (0..Good::COUNT).try_for_each(|good| -> Result<()> {
                    snapshot.inventories[good] = add(
                        snapshot.inventories[good],
                        state.values[(good + 1) * state.actors.len() + i],
                    )?;
                    Ok(())
                })?;
                for good in 0..Good::COUNT {
                    let deliveries = (0..scenario.ring()).try_fold(0, |total, slot| {
                        add(
                            total,
                            state.values[state
                                .shipment_coordinate(Good::ALL[good], slot, scenario.ring(), i)
                                .0 as usize],
                        )
                    })?;
                    match deliveries
                        == state.values[(TRANSIT_FIELDS + good) * state.actors.len() + i]
                    {
                        true => (),
                        false => Err(anyhow!(
                            "shipment ownership mismatch for actor {} and {:?}",
                            actor.identity,
                            Good::ALL[good]
                        ))?,
                    }
                    snapshot.transit[good] = add(
                        snapshot.transit[good],
                        state.values[(TRANSIT_FIELDS + good) * state.actors.len() + i],
                    )?;
                    let household_output =
                        state.values[(PRODUCTION_FIELDS + good) * state.actors.len() + i];
                    snapshot.household_production[good] =
                        add(snapshot.household_production[good], household_output)?;
                    snapshot.produced[good] = add(snapshot.produced[good], household_output)?;
                    snapshot.used[good] = add(
                        snapshot.used[good],
                        state.values[(USED_FIELDS + good) * state.actors.len() + i],
                    )?;
                    snapshot.consumption[good] = add(
                        snapshot.consumption[good],
                        state.values[(CONSUMPTION_FIELDS + good) * state.actors.len() + i],
                    )?;
                }
                match actor.kind {
                    0 => {
                        snapshot.employed += state.get(Field::Employed, i);
                        snapshot.shortages =
                            add(snapshot.shortages, state.get(Field::Shortage, i))?;
                        snapshot.consumed = add(snapshot.consumed, state.get(Field::Consumed, i))?;
                        snapshot.wealth_bins
                            [(64 - state.get(Field::Cash, i).leading_zeros() as usize).min(63)] +=
                            1;
                        let x = actor.position.x * 16 / scenario.config.map_size;
                        let y = actor.position.y * 16 / scenario.config.map_size;
                        snapshot.regions[(y * 16 + x) as usize] += state.get(Field::Shortage, i);
                    }
                    kind => {
                        snapshot.firms += 1;
                        snapshot.firm_entries += u64::from(i >= scenario.actors());
                        snapshot.firm_closures += u64::from(
                            state.get(Field::EnterpriseStatus, i)
                                == enterprise::EnterpriseState::Closed as u64,
                        );
                        snapshot.distributors += u64::from(
                            state.get(Field::EnterpriseModel, i)
                                == enterprise::EnterpriseModel::Distributor as u64,
                        );
                        let j = kind as usize - 1;
                        snapshot.markets[j]
                            .wages
                            .include(state.get(Field::Wage, i))?;
                        snapshot.markets[j]
                            .prices
                            .include(state.get(Field::Price, i))?;
                        snapshot.produced[j] =
                            add(snapshot.produced[j], state.get(Field::Produced, i))?;
                        snapshot.inventories[j] =
                            add(snapshot.inventories[j], state.get(Field::Pipeline, i))?;
                        snapshot.minimum_wage =
                            snapshot.minimum_wage.min(state.get(Field::Wage, i));
                        snapshot.maximum_wage =
                            snapshot.maximum_wage.max(state.get(Field::Wage, i));
                        snapshot.minimum_price =
                            snapshot.minimum_price.min(state.get(Field::Price, i));
                        snapshot.maximum_price =
                            snapshot.maximum_price.max(state.get(Field::Price, i));
                    }
                }
                let in_ring = (0..scenario.ring() * Good::COUNT).try_fold(0, |total, slot| {
                    add(
                        total,
                        state.values[(BASE_FIELDS + slot) * state.actors.len() + i],
                    )
                })?;
                match in_ring == state.get(Field::Transit, i) {
                    true => Ok(()),
                    false => Err(anyhow!(
                        "transit ownership audit failed for {}",
                        actor.identity
                    )),
                }
            })?;
        if let Some(finance) = &scenario.config.finance {
            let view = finance::FinanceStateView::new(
                finance,
                finance.layout(state.actors.len() * scenario.fields())?,
                &state.values,
                state.day,
            )?;
            snapshot.financial_cash = view.cash;
            snapshot.loan_principal = view.principal;
            snapshot.loan_interest = view.interest_due;
            snapshot.active_loans = view.active_loans;
            snapshot.overdue_loans = view.overdue_loans;
            snapshot.cash = add(snapshot.cash, view.cash)?;
        }
        snapshot.audit(state.initial_cash, state.initial_goods)?;
        Ok(snapshot)
    }
    pub fn audit(&self, initial_cash: u64, initial_goods: [u64; Good::COUNT]) -> Result<()> {
        match self.cash == initial_cash {
            true => Ok(()),
            false => Err(anyhow!("cash conservation failed")),
        }?;
        (0..Good::COUNT).try_for_each(|i| {
            let sources = u128::from(initial_goods[i]) + u128::from(self.produced[i]);
            let uses = u128::from(self.inventories[i])
                + u128::from(self.transit[i])
                + u128::from(self.used[i])
                + u128::from(self.research.consumed[i])
                + u128::from(self.consumption[i]);
            match sources == uses {
                true => Ok(()),
                false => Err(anyhow!(
                    "goods conservation failed for good {}: {sources} != {uses}",
                    i + 1
                )),
            }
        })
    }
}

pub(crate) struct AuditedState<'a> {
    state: &'a State,
}
impl<'a> AuditedState<'a> {
    pub(crate) fn new(scenario: &Scenario, state: &'a State) -> Result<Self> {
        let c = scenario.config();
        let _ = demography::PersonRegistry::new(state)?;
        let _ = enterprise::OwnershipRegistry::new(state)?;
        let shape = state.actors.len() >= scenario.actors()
            && state.values.len()
                == state.actors.len() * scenario.fields()
                    + c.finance
                        .as_ref()
                        .map_or(0, finance::FinanceConfig::state_len);
        match shape {
            true => Ok(()),
            false => Err(anyhow!("state shape does not match scenario")),
        }?;
        let ownership = state
            .ownership
            .iter()
            .map(|entry| (entry.firm, entry))
            .collect::<std::collections::HashMap<_, _>>();
        let identities = state.actors.iter().enumerate().find_map(|(index, actor)| {
            let original = match index < scenario.actors() {
                true => {
                    actor.identity == scenario.actor(index).identity
                        && actor.kind == scenario.actor(index).kind
                }
                false => true,
            };
            let capacity = index >= scenario.actors()
                || actor.capacity == scenario.actor(index).capacity
                || state.get(Field::InstalledCapital, index) > 0
                || state.get(Field::EnterpriseStatus, index)
                    == enterprise::EnterpriseState::Closed as u64;
            let position = index >= scenario.actors()
                || actor.position == scenario.actor(index).position
                || (u64::from(actor.position.x) == state.get(Field::MoveX, index)
                    && u64::from(actor.position.y) == state.get(Field::MoveY, index))
                || (state.get(Field::MoveDue, index) > 0
                    && u64::from(actor.position.x) == state.get(Field::PreviousX, index)
                    && u64::from(actor.position.y) == state.get(Field::PreviousY, index));
            let owner = actor.kind == 0
                || ownership.get(&actor.identity).is_some_and(|ownership| {
                    ownership.shares.iter().any(|share| {
                        state
                            .actors
                            .get(actor.owner as usize)
                            .is_some_and(|owner| owner.identity == share.household)
                    })
                });
            let admitted = original
                && capacity
                && position
                && owner
                && actor.kind <= Good::COUNT as u32
                && actor.owner < state.actors.len() as u32
                && actor.position.x < c.map_size
                && actor.position.y < c.map_size;
            (!admitted).then_some((actor, original, capacity, position, owner))
        });
        match identities {
            None => Ok(()),
            Some((actor, original, capacity, position, owner)) => Err(anyhow!(
                "invalid actor registry: {actor:?}; identity={original}, capacity={capacity}, position={position}, ownership={owner}"
            )),
        }?;
        let resources = (0..state.actors.len()).all(|i| {
            let limits = state.get(Field::EquipmentCondition, i) <= 1000
                && state.get(Field::SafetyProvision, i) <= 900
                && state.get(Field::EnterpriseStatus, i)
                    <= enterprise::EnterpriseState::Closed as u64
                && state.get(Field::EnterpriseModel, i)
                    <= enterprise::EnterpriseModel::Distributor as u64
                && state.get(Field::FoodAccess, i)
                    <= enterprise::FoodAccess::MatchingFailure as u64
                && state.get(Field::Fatigue, i) <= 1000
                && state.get(Field::Time, i) <= 1440
                && state.get(Field::Hours, i) <= 1440
                && state.get(Field::ResearchMinutes, i) <= 1440
                && state.get(Field::Employed, i) <= 1
                && state.get(Field::Employer, i) <= state.actors.len() as u64;
            let closed = [Field::Escrow, Field::Due, Field::Pipeline]
                .into_iter()
                .all(|field| state.get(field, i) == 0);
            let actor = &state.actors.at(i);
            let recipe = match (actor.kind, state.get(Field::EnterpriseModel, i)) {
                (_, 1) => {
                    state.get(Field::FirstInput, i) == 0
                        && state.get(Field::SecondInput, i) == 0
                        && Technique::new(
                            state.get(Field::LaborMinutes, i) as u32,
                            0,
                            state.get(Field::OutputTicks, i) as u32,
                        )
                        .is_ok()
                }
                (0, _) => [
                    Field::LaborMinutes,
                    Field::InputTicks,
                    Field::OutputTicks,
                    Field::FirstInput,
                    Field::SecondInput,
                    Field::ActiveMethod,
                ]
                .into_iter()
                .all(|field| state.get(field, i) == 0),
                _ => ProductionMethod::from_index(state.get(Field::ActiveMethod, i))
                    .and_then(|method| match i < scenario.actors() {
                        true => scenario.production_recipe(i, method),
                        false => Ok(industry::Recipe {
                            technique: Technique::new(
                                state.get(Field::LaborMinutes, i) as u32,
                                state.get(Field::InputTicks, i) as u32,
                                state.get(Field::OutputTicks, i) as u32,
                            )?,
                            inputs: Good::from_index(actor.kind)?.inputs(),
                        }),
                    })
                    .is_ok_and(|recipe| {
                        state.get(Field::LaborMinutes, i)
                            == u64::from(recipe.technique.labor_minutes)
                            && state.get(Field::InputTicks, i)
                                == u64::from(recipe.technique.input_ticks)
                            && state.get(Field::OutputTicks, i)
                                == u64::from(recipe.technique.output_ticks)
                            && state.get(Field::FirstInput, i)
                                == recipe.inputs[0].map_or(0, |good| good as u64)
                            && state.get(Field::SecondInput, i)
                                == recipe.inputs[1].map_or(0, |good| good as u64)
                    }),
            };
            let used = (0..Good::COUNT)
                .map(|good| u128::from(state.values[(USED_FIELDS + good) * state.actors.len() + i]))
                .sum::<u128>();
            let accounting = used == u128::from(state.get(Field::Used, i))
                && state.values
                    [(CONSUMPTION_FIELDS + Good::Bread as usize - 1) * state.actors.len() + i]
                    == state.get(Field::Consumed, i);
            let daily_budget = u128::from(state.get(Field::HouseholdBudget, i));
            let labor = match (
                actor.kind,
                state.get(Field::Employed, i),
                state.get(Field::Employer, i),
            ) {
                (0, 0, 0) => {
                    state.get(Field::Hours, i) == 0
                        && u128::from(state.get(Field::Time, i))
                            + u128::from(state.get(Field::HomeMinutes, i))
                            + u128::from(state.get(Field::ResearchMinutes, i))
                            <= daily_budget
                }
                (0, 1, employer) if (1..=state.actors.len() as u64).contains(&employer) => {
                    let employer = (employer - 1) as usize;
                    let employer_capacity =
                        scenario.geographic_capacity(&state.actors.at(employer));
                    let labor_minutes = state.get(Field::LaborMinutes, employer);
                    let commute = actor
                        .position
                        .distance(state.actors.at(employer).position)
                        .div_ceil(c.commute_speed)
                        * 2;
                    employer_capacity > 0
                        && state.actors.at(employer).kind > 0
                        && state.get(Field::Hours, i) == labor_minutes
                        && u128::from(state.get(Field::Hours, i))
                            + u128::from(state.get(Field::Time, i))
                            + u128::from(state.get(Field::HomeMinutes, i))
                            + u128::from(state.get(Field::ResearchMinutes, i))
                            + u128::from(commute)
                            <= daily_budget
                }
                (1..=16, 0, 0) => state.get(Field::Hours, i) == 0 && state.get(Field::Time, i) == 0,
                _ => false,
            };
            let capacity = scenario.geographic_capacity(actor);
            let planning = state.get(Field::Slots, i) <= state.get(Field::Plan, i)
                && state.get(Field::Plan, i) <= u64::from(capacity);
            let production = match capacity {
                0 => state.get(Field::Produced, i) == 0,
                _ => true,
            };
            let role = match actor.kind {
                0 => [Field::Slots, Field::Plan, Field::Produced]
                    .into_iter()
                    .all(|field| state.get(field, i) == 0),
                _ => [Field::Consumed, Field::Shortage]
                    .into_iter()
                    .all(|field| state.get(field, i) == 0),
            };
            let cleared_slot = ((state.day % scenario.ring() as u64 + scenario.ring() as u64 - 1)
                % scenario.ring() as u64) as usize;
            let delivery_window = Good::ALL.into_iter().all(|good| {
                state.values[state
                    .shipment_coordinate(good, cleared_slot, scenario.ring(), i)
                    .0 as usize]
                    == 0
            });
            let quotes = match actor.kind {
                0 => true,
                _ => {
                    (1..=10_000).contains(&state.get(Field::Wage, i))
                        && (u64::from(c.minimum_price)..=u64::from(c.maximum_price))
                            .contains(&state.get(Field::Price, i))
                }
            };
            let payroll = match (&c.payroll, actor.kind) {
                (Some(_), 0) => state.get(Field::PayrollReserve, i) <= 2,
                (None, 0) => [
                    Field::PayrollReserve,
                    Field::TaxTreasury,
                    Field::ChargeTreasury,
                ]
                .into_iter()
                .all(|field| state.get(field, i) == 0),
                _ => [
                    Field::GrossIncome,
                    Field::PayrollReserve,
                    Field::TaxTreasury,
                    Field::ChargeTreasury,
                ]
                .into_iter()
                .all(|field| state.get(field, i) == 0),
            };
            limits
                && closed
                && labor
                && planning
                && production
                && role
                && delivery_window
                && quotes
                && payroll
                && recipe
                && accounting
        });
        let land_limit = match &c.land_access {
            Some(land) => u64::from(land.total_plots),
            None => {
                u64::from(c.households.div_ceil(2))
                    * u64::from(c.demography.initial_household_members.div_ceil(2))
            }
        };
        let land = (0..state.actors.len())
            .map(|index| u128::from(state.get(Field::LandPlots, index)))
            .sum::<u128>();
        let valid = resources
            && land <= u128::from(land_limit)
            && state.initial_population
                == match c.demography.people.is_empty() {
                    true => {
                        u64::from(c.households) * u64::from(c.demography.initial_household_members)
                    }
                    false => c.demography.people.len() as u64,
                }
            && match (&c.research, &state.research) {
                (None, None) => true,
                (Some(_), None) => state.day == 0,
                (Some(_), Some(snapshot)) => {
                    snapshot.profile == research::RESEARCH_PROFILE && !snapshot.words.is_empty()
                }
                (None, Some(_)) => false,
            }
            && state
                .actors
                .iter()
                .map(|actor| actor.identity)
                .chain(state.people.iter().map(|person| person.identity))
                .all(|identity| identity < state.next_identity)
            && state.day <= u64::from(u32::MAX) - scenario.ring() as u64
            && state.initial_cash == scenario.totals.cash
            && state.initial_goods == scenario.totals.goods;
        match valid {
            true => Ok(Self { state }),
            false => Err(anyhow!(
                "invalid identity, policy bounds, opening totals, date, or unsettled day"
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Market {
    Labor,
    Input(Good),
    Household,
    Consumer(Good),
    Payroll,
    Distribution,
}
impl Market {
    pub fn code(self) -> u32 {
        match self {
            Self::Labor => 0,
            Self::Input(good) => good as u32,
            Self::Consumer(good) => Good::COUNT as u32 + good as u32,
            Self::Household => 33,
            Self::Payroll => 34,
            Self::Distribution => 35,
        }
    }
}
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PhaseEvidence {
    pub decisions: Vec<crate::ledger::Decision>,
    pub frontier_depth: usize,
    pub events: usize,
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
                value["version"] = SCENARIO_VERSION.into();
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
            Some(2) => (),
            _ => Err(anyhow!(
                "unsupported scenario format; expected version 1 or 2"
            ))?,
        }
        let current: CurrentScenario = serde_json::from_value(value)?;
        Ok(Self {
            economy: current.economy,
            version: current.version,
            seed: current.seed,
            households: current.households,
            firms: current.firms,
            map_size: current.map_size,
            resources: current.resources,
            search_radius: current.search_radius,
            commute_speed: current.commute_speed,
            shipment_speed: current.shipment_speed,
            household_cash: current.household_cash,
            firm_cash: current.firm_cash,
            buffer_days: current.buffer_days,
            household_need: current.household_need,
            capacity: current.capacity,
            packet_output: current.packet_output,
            packet_minutes: current.packet_minutes,
            work_cap: current.work_cap,
            reserved_minutes: current.reserved_minutes,
            wage: current.wage,
            prices: current.prices,
            reserve: current.reserve,
            distribution_rate: current.distribution_rate,
            forecast_weight: current.forecast_weight,
            coverage: current.coverage,
            minimum_price: current.minimum_price,
            maximum_price: current.maximum_price,
            delivery_penalty: current.delivery_penalty,
            batch_size: current.batch_size,
            experiment: current.experiment,
            population: current.population,
            behavior: current.behavior,
            payroll: current.payroll,
            household_policy: current.household_policy,
            finance: current.finance,
            research: current.research,
            bargaining: current.bargaining,
            demography: current.demography,
            household_production: current.household_production,
            distributors: current.distributors,
            ownership: current.ownership,
            land_access: current.land_access,
            working_conditions: current.working_conditions,
        })
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CurrentScenario {
    #[serde(default)]
    pub economy: Economy,
    pub version: u32,
    pub seed: u64,
    pub households: u32,
    pub firms: u32,
    pub map_size: u32,
    #[serde(default)]
    pub resources: ResourceMap,
    pub search_radius: u32,
    pub commute_speed: u32,
    pub shipment_speed: u32,
    pub household_cash: u64,
    pub firm_cash: u64,
    pub buffer_days: u32,
    pub household_need: u32,
    pub capacity: u32,
    pub packet_output: u32,
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
    #[serde(default)]
    pub land_access: Option<enterprise::LandAccess>,
    #[serde(default)]
    pub working_conditions: Vec<enterprise::FirmConditions>,
}

fn generated_good(config: &ScenarioConfig, firm: usize) -> Good {
    match config.economy {
        Economy::Agrarian => Good::ALL[firm % 3],
        Economy::Industrial => match firm {
            0..16 => Good::ALL[firm],
            _ => match (firm - 16) % 20 {
                0..=3 => Good::Grain,
                4..=7 => Good::Flour,
                8..=11 => Good::Bread,
                slot => Good::ALL[3 + ((firm - 16) / 20 * 8 + slot - 12) % 13],
            },
        },
    }
}

#[derive(Deserialize)]
#[serde(transparent)]
struct CheckpointDocument(serde_json::Value);

impl TryFrom<CheckpointDocument> for Checkpoint {
    type Error = anyhow::Error;
    fn try_from(document: CheckpointDocument) -> Result<Self> {
        match document
            .0
            .get("version")
            .and_then(serde_json::Value::as_u64)
        {
            Some(6) => (),
            _ => Err(anyhow!(
                "unsupported checkpoint format: format 6 is required; regenerate older checkpoints"
            ))?,
        }
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Current {
            version: u32,
            profile: String,
            table_hash: String,
            scenario: ScenarioConfig,
            state: State,
            checksum: String,
        }
        let current: Current = serde_json::from_value(document.0)?;
        Ok(Self {
            version: current.version,
            profile: current.profile,
            table_hash: current.table_hash,
            scenario: current.scenario,
            state: current.state,
            checksum: current.checksum,
        })
    }
}

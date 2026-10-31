use crate::{
    arithmetic::{DrawKey, PROFILE, add, hex, table_hash},
    ledger::Coordinate,
};
use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub mod bargaining;
pub mod finance;
mod households;
mod payroll;
pub mod research;
pub use households::HouseholdPolicy;
mod population;
pub use payroll::{MAX_TAX_BRACKETS, PAYROLL_SCALE, PayrollPolicy, TaxBracket};
pub use population::{
    BehaviorPolicy, FirmSeed, HouseholdSeed, InitialTotals, Population, Technique,
};

pub const SCENARIO_VERSION: u32 = 1;
pub const STATE_HASH_VERSION: &str = "bad-econ/state/DeterministicV1/v3";
pub const BASE_FIELDS: usize = 31;
pub const MEMORY_BUDGET: u64 = 4 * 1024 * 1024 * 1024;

#[repr(u32)]
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum Field {
    Cash,
    Grain,
    Flour,
    Bread,
    Escrow,
    Slots,
    Wage,
    Price,
    Demand,
    Pipeline,
    Time,
    Health,
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
}
impl Field {
    pub fn inventory(good: Good) -> Self {
        match good {
            Good::Grain => Self::Grain,
            Good::Flour => Self::Flour,
            Good::Bread => Self::Bread,
        }
    }
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum Good {
    Grain = 1,
    Flour = 2,
    Bread = 3,
}
impl Good {
    pub const COUNT: usize = 3;
    pub fn from_index(index: u32) -> Result<Self> {
        match index {
            1 => Ok(Self::Grain),
            2 => Ok(Self::Flour),
            3 => Ok(Self::Bread),
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
#[serde(deny_unknown_fields)]
pub struct ScenarioConfig {
    pub version: u32,
    pub seed: u64,
    pub households: u32,
    pub firms: u32,
    pub map_size: u32,
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
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Scenario {
    config: ScenarioConfig,
    totals: InitialTotals,
}
impl Scenario {
    pub fn new(mut config: ScenarioConfig) -> Result<Self> {
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
                let totals = match &config.population {
                    Some(population) => population.totals()?,
                    None => {
                        let cash = u128::from(config.household_cash)
                            * u128::from(config.households)
                            + u128::from(config.firm_cash) * u128::from(config.firms);
                        let stock = u64::from(config.capacity)
                            * u64::from(config.packet_output)
                            * u64::from(config.buffer_days);
                        let grain = u64::from(config.firms.div_ceil(3));
                        let flour = u64::from((config.firms + 1) / 3);
                        let bread = u64::from(config.firms / 3);
                        InitialTotals {
                            cash: u64::try_from(cash)
                                .context("initial cash exceeds tick capacity")?,
                            goods: [
                                stock * (grain + flour),
                                stock * (flour + bread),
                                stock * bread
                                    + u64::from(config.households)
                                        * u64::from(config.household_need)
                                        * u64::from(config.buffer_days),
                            ],
                        }
                    }
                };
                let mut scenario = Self { config, totals };
                if scenario.config.finance.is_some() || scenario.config.research.is_some() {
                    let actors = (0..scenario.actors())
                        .map(|index| scenario.actor(index))
                        .collect::<Vec<_>>();
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
                match side <= 256 && ring <= 65 && scenario.memory_estimate() <= MEMORY_BUDGET {
                    true => Ok(scenario),
                    false => Err(anyhow!(
                        "scenario exceeds transit, cash, or 4 GiB memory capacity"
                    )),
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
            version: 1,
            seed,
            households,
            firms,
            map_size,
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
        BASE_FIELDS + self.ring()
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
            + actors * 64
            + batch * (12 * 8 * 4 + 12 * 4 * 3 + 28)
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
                    Some(firm) if firm % Good::COUNT == 0 => 0,
                    _ => self.config.packet_output,
                },
                output_ticks: self.config.packet_output,
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
        Ok(state)
    }
    fn actor(&self, index: usize) -> Actor {
        match &self.config.population {
            Some(population) => population.actor(index),
            None => {
                let config = &self.config;
                let identity = index as u64 + 1;
                let kind = match index < config.households as usize {
                    true => 0,
                    false => 1 + (index as u32 - config.households) % Good::COUNT as u32,
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
                Actor {
                    identity,
                    position,
                    kind,
                    owner: index as u32 % config.households,
                    capacity: config.capacity,
                }
            }
        }
    }
    fn generated(&self) -> Result<State> {
        let actors = self.actors();
        let config = &self.config;
        let properties = (0..actors)
            .map(|index| self.actor(index))
            .collect::<Vec<_>>();
        let mut state = State {
            day: 0,
            next_identity: actors as u64 + 1,
            actors: properties,
            values: vec![0; self.fields() * actors],
            initial_cash: 0,
            initial_goods: [0; 3],
            research: None,
        };
        (0..actors).try_for_each(|actor| -> Result<()> {
            let kind = state.actors[actor].kind;
            let cash = match kind {
                0 => config.household_cash,
                _ => config.firm_cash,
            };
            state.set(Field::Cash, actor, cash);
            state.set(Field::Health, actor, 1000);
            state.set(Field::Wage, actor, u64::from(config.wage));
            let stock = match kind {
                0 => u64::from(config.household_need) * u64::from(config.buffer_days),
                _ => {
                    u64::from(config.capacity)
                        * u64::from(config.packet_output)
                        * u64::from(config.buffer_days)
                }
            };
            let output = match kind {
                0 => Good::Bread,
                _ => Good::from_index(kind)?,
            };
            state.set(Field::inventory(output), actor, stock);
            state.initial_goods[output as usize - 1] =
                add(state.initial_goods[output as usize - 1], stock)?;
            match kind {
                2 | 3 => {
                    let input = Good::from_index(kind - 1)?;
                    state.set(Field::inventory(input), actor, stock);
                    state.initial_goods[input as usize - 1] =
                        add(state.initial_goods[input as usize - 1], stock)?;
                }
                _ => (),
            }
            match kind {
                0 => (),
                _ => {
                    state.set(
                        Field::Price,
                        actor,
                        u64::from(config.prices[kind as usize - 1]),
                    );
                    state.set(
                        Field::Demand,
                        actor,
                        u64::from(config.capacity) * u64::from(config.packet_output),
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
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct State {
    pub day: u64,
    pub next_identity: u64,
    pub actors: Vec<Actor>,
    pub values: Vec<u64>,
    pub initial_cash: u64,
    pub initial_goods: [u64; 3],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub research: Option<research::ResearchSnapshot>,
}
impl State {
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
#[serde(deny_unknown_fields)]
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
            version: 2,
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
            self.version == 2,
            self.profile == PROFILE,
            self.table_hash == table_hash(),
            self.state.values.len() == scenario.state_len(),
            self.state.actors.len() == scenario.actors(),
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
    pub actor: Actor,
    pub day: u64,
    pub cash: u64,
    pub gross_income: u64,
    pub payroll_reserve: u64,
    pub income_tax: u64,
    pub employee_charges: u64,
    pub inventory: [u64; 3],
    pub transit: u64,
    pub health: u64,
    pub fatigue: u64,
    pub employer: u64,
    pub owner_identity: u64,
    pub available_minutes: u64,
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
            actor: state.actors[index].clone(),
            day: state.day,
            cash: state.get(Field::Cash, index),
            gross_income: state.get(Field::GrossIncome, index),
            payroll_reserve: state.get(Field::PayrollReserve, index),
            income_tax: state.get(Field::TaxTreasury, index),
            employee_charges: state.get(Field::ChargeTreasury, index),
            inventory: [
                state.get(Field::Grain, index),
                state.get(Field::Flour, index),
                state.get(Field::Bread, index),
            ],
            transit: state.get(Field::Transit, index),
            health: state.get(Field::Health, index),
            fatigue: state.get(Field::Fatigue, index),
            employer: state
                .get(Field::Employer, index)
                .checked_sub(1)
                .and_then(|employer| state.actors.get(employer as usize))
                .map(|actor| actor.identity)
                .unwrap_or(0),
            owner_identity: state.actors[state.actors[index].owner as usize].identity,
            available_minutes: state.get(Field::Time, index),
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
    pub shortages: u64,
    pub inventories: [u64; 3],
    pub transit: [u64; 3],
    pub produced: [u64; 3],
    pub used: [u64; 3],
    pub consumed: u64,
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
            households: u64::from(scenario.config.households),
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
                (0..3).try_for_each(|good| -> Result<()> {
                    snapshot.inventories[good] = add(
                        snapshot.inventories[good],
                        state.values[(good + 1) * state.actors.len() + i],
                    )?;
                    Ok(())
                })?;
                let input = match actor.kind {
                    0 => 3,
                    1 => 0,
                    other => other - 1,
                };
                match input {
                    0 => (),
                    _ => {
                        let j = input as usize - 1;
                        snapshot.transit[j] =
                            add(snapshot.transit[j], state.get(Field::Transit, i))?;
                        snapshot.used[j] = add(snapshot.used[j], state.get(Field::Used, i))?;
                    }
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
                let in_ring = (0..scenario.ring()).try_fold(0, |total, slot| {
                    add(
                        total,
                        state.values[state.ring_coordinate(slot, i).0 as usize],
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
                finance.layout(scenario.actors() * scenario.fields())?,
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
    pub fn audit(&self, initial_cash: u64, initial_goods: [u64; 3]) -> Result<()> {
        match self.cash == initial_cash {
            true => Ok(()),
            false => Err(anyhow!("cash conservation failed")),
        }?;
        (0..3).try_for_each(|i| {
            let sources = u128::from(initial_goods[i]) + u128::from(self.produced[i]);
            let uses = u128::from(self.inventories[i])
                + u128::from(self.transit[i])
                + u128::from(self.used[i])
                + u128::from(self.research.consumed[i])
                + match i {
                    2 => u128::from(self.consumed),
                    _ => 0,
                };
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

struct AuditedState<'a> {
    state: &'a State,
}
impl<'a> AuditedState<'a> {
    fn new(scenario: &Scenario, state: &'a State) -> Result<Self> {
        let c = scenario.config();
        let shape =
            state.actors.len() == scenario.actors() && state.values.len() == scenario.state_len();
        match shape {
            true => Ok(()),
            false => Err(anyhow!("state shape does not match scenario")),
        }?;
        let identities = state
            .actors
            .iter()
            .enumerate()
            .all(|(index, actor)| *actor == scenario.actor(index));
        match identities {
            true => Ok(()),
            false => Err(anyhow!("actor metadata does not match the fixed scenario")),
        }?;
        let resources = (0..scenario.actors()).all(|i| {
            let limits = state.get(Field::Health, i) <= 1000
                && state.get(Field::Fatigue, i) <= 1000
                && state.get(Field::Time, i) <= 1440
                && state.get(Field::Hours, i) <= 1440
                && state.get(Field::ResearchMinutes, i) <= 1440
                && state.get(Field::Employed, i) <= 1
                && state.get(Field::Employer, i) <= scenario.actors() as u64;
            let closed = [Field::Escrow, Field::Due, Field::Pipeline]
                .into_iter()
                .all(|field| state.get(field, i) == 0);
            let actor = &state.actors[i];
            let daily_budget = u128::from(c.work_cap.min(1440 - c.reserved_minutes));
            let labor = match (
                actor.kind,
                state.get(Field::Employed, i),
                state.get(Field::Employer, i),
            ) {
                (0, 0, 0) => {
                    state.get(Field::Hours, i) == 0
                        && u128::from(state.get(Field::Time, i))
                            + u128::from(state.get(Field::ResearchMinutes, i))
                            <= daily_budget
                }
                (0, 1, employer)
                    if (u64::from(c.households) + 1..=scenario.actors() as u64)
                        .contains(&employer) =>
                {
                    let employer = (employer - 1) as usize;
                    let technique = scenario.technique(employer);
                    let commute = actor
                        .position
                        .distance(state.actors[employer].position)
                        .div_ceil(c.commute_speed)
                        * 2;
                    state.get(Field::Hours, i) == u64::from(technique.labor_minutes)
                        && u128::from(state.get(Field::Hours, i))
                            + u128::from(state.get(Field::Time, i))
                            + u128::from(state.get(Field::ResearchMinutes, i))
                            + u128::from(commute)
                            <= daily_budget
                }
                (1..=3, 0, 0) => state.get(Field::Hours, i) == 0 && state.get(Field::Time, i) == 0,
                _ => false,
            };
            let planning = state.get(Field::Slots, i) <= state.get(Field::Plan, i)
                && state.get(Field::Plan, i) <= u64::from(actor.capacity);
            let role = match actor.kind {
                0 => [Field::Slots, Field::Plan, Field::Produced, Field::Used]
                    .into_iter()
                    .all(|field| state.get(field, i) == 0),
                1 => [
                    Field::Transit,
                    Field::Used,
                    Field::Consumed,
                    Field::Shortage,
                ]
                .into_iter()
                .all(|field| state.get(field, i) == 0),
                _ => [Field::Consumed, Field::Shortage]
                    .into_iter()
                    .all(|field| state.get(field, i) == 0),
            };
            let cleared_slot = ((state.day % scenario.ring() as u64 + scenario.ring() as u64 - 1)
                % scenario.ring() as u64) as usize;
            let delivery_window =
                state.values[state.ring_coordinate(cleared_slot, i).0 as usize] == 0;
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
                _ => [
                    Field::GrossIncome,
                    Field::PayrollReserve,
                    Field::TaxTreasury,
                    Field::ChargeTreasury,
                ]
                .into_iter()
                .all(|field| state.get(field, i) == 0),
            };
            limits && closed && labor && planning && role && delivery_window && quotes && payroll
        });
        let valid = resources
            && match (&c.research, &state.research) {
                (None, None) => true,
                (Some(_), None) => state.day == 0,
                (Some(_), Some(snapshot)) => {
                    snapshot.profile == research::RESEARCH_PROFILE && !snapshot.words.is_empty()
                }
                (None, Some(_)) => false,
            }
            && state.next_identity == scenario.next_identity()
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
    Payroll,
    Distribution,
}
impl Market {
    pub fn code(self) -> u32 {
        match self {
            Self::Labor => 0,
            Self::Input(good) => good as u32,
            Self::Household => 4,
            Self::Payroll => 5,
            Self::Distribution => 6,
        }
    }
}
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PhaseEvidence {
    pub decisions: Vec<crate::ledger::Decision>,
    pub frontier_depth: usize,
    pub events: usize,
}

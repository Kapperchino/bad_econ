use crate::{
    arithmetic::{DrawKey, PROFILE, add, hex, table_hash},
    ledger::Coordinate,
};
use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const SCENARIO_VERSION: u32 = 1;
pub const BASE_FIELDS: usize = 26;
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
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Scenario {
    config: ScenarioConfig,
}
impl Scenario {
    pub fn new(config: ScenarioConfig) -> Result<Self> {
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
                let scenario = Self { config };
                let ring = scenario.ring();
                let side = scenario
                    .config
                    .map_size
                    .div_ceil(scenario.config.search_radius);
                let cash = u128::from(scenario.config.household_cash)
                    * u128::from(scenario.config.households)
                    + u128::from(scenario.config.firm_cash) * u128::from(scenario.config.firms);
                match side <= 256
                    && ring <= 65
                    && cash <= u128::from(u64::MAX)
                    && scenario.memory_estimate() <= MEMORY_BUDGET
                {
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
    pub fn memory_estimate(&self) -> u64 {
        let actors = self.actors() as u64;
        let batch = u64::from(self.config.batch_size).min(actors);
        actors * (self.fields() as u64 * 8 * 2 + 64)
            + batch * (12 * 8 * 4 + 12 * 4 * 3 + 28)
            + 128 * 1024 * 1024
    }
    pub fn fingerprint(&self) -> Result<String> {
        Ok(hex(&Sha256::digest(serde_json::to_vec(&self.config)?)))
    }
    pub fn initial(&self) -> Result<State> {
        let actors = self.actors();
        let config = &self.config;
        let properties = (0..actors)
            .map(|index| {
                let identity = index as u64 + 1;
                let draw = DrawKey {
                    seed: config.seed,
                    day: 0,
                    phase: 0,
                    actor: identity,
                    purpose: 0,
                    index: 0,
                }
                .draw();
                let kind = match index < config.households as usize {
                    true => 0,
                    false => 1 + (index as u32 - config.households) % 3,
                };
                let position = match (config.experiment, kind) {
                    (Experiment::Concentrated, _) => Position {
                        x: config.map_size / 2,
                        y: config.map_size / 2,
                    },
                    _ => Position {
                        x: draw as u32 % config.map_size,
                        y: (draw >> 32) as u32 % config.map_size,
                    },
                };
                Actor {
                    identity,
                    position,
                    kind,
                    owner: (index as u32 % config.households),
                    capacity: config.capacity,
                }
            })
            .collect::<Vec<_>>();
        let mut state = State {
            day: 0,
            next_identity: actors as u64 + 1,
            actors: properties,
            values: vec![0; self.fields() * actors],
            initial_cash: 0,
            initial_goods: [0; 3],
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
        hash.update(self.day.to_le_bytes());
        hash.update(self.next_identity.to_le_bytes());
        self.values
            .iter()
            .for_each(|value| hash.update(value.to_le_bytes()));
        hex(&hash.finalize())
    }
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
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
            version: 1,
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
            self.version == 1,
            self.profile == PROFILE,
            self.table_hash == table_hash(),
            self.state.values.len() == scenario.actors() * scenario.fields(),
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
        match (1..=scenario.actors() as u64).contains(&identity) {
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
    pub inventory: [u64; 3],
    pub transit: u64,
    pub health: u64,
    pub fatigue: u64,
    pub employer: u64,
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
            inventory: [
                state.get(Field::Grain, index),
                state.get(Field::Flour, index),
                state.get(Field::Bread, index),
            ],
            transit: state.get(Field::Transit, index),
            health: state.get(Field::Health, index),
            fatigue: state.get(Field::Fatigue, index),
            employer: state.get(Field::Employer, index),
            available_minutes: state.get(Field::Time, index),
            wage: state.get(Field::Wage, index),
            price: state.get(Field::Price, index),
            planned_packets: state.get(Field::Plan, index),
            open_slots: state.get(Field::Slots, index),
        })
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ObservationSnapshot {
    pub day: u64,
    pub cash: u64,
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
    pub wealth_bins: Vec<u64>,
    pub regions: Vec<u64>,
}
impl ObservationSnapshot {
    pub fn from_state(scenario: &Scenario, state: &State) -> Result<Self> {
        let admitted = AuditedState::new(scenario, state)?;
        let state = admitted.state;
        let mut snapshot = Self {
            day: state.day,
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
        let shape = state.actors.len() == scenario.actors()
            && state.values.len() == scenario.actors() * scenario.fields();
        match shape {
            true => Ok(()),
            false => Err(anyhow!("state shape does not match scenario")),
        }?;
        let identities = state.actors.iter().enumerate().all(|(i, actor)| {
            let kind = match i < c.households as usize {
                true => 0,
                false => 1 + (i as u32 - c.households) % 3,
            };
            actor.identity == i as u64 + 1
                && actor.kind == kind
                && actor.position.x < c.map_size
                && actor.position.y < c.map_size
                && actor.owner < c.households
                && (1..=10_000).contains(&actor.capacity)
        });
        let resources = (0..scenario.actors()).all(|i| {
            let limits = state.get(Field::Health, i) <= 1000
                && state.get(Field::Fatigue, i) <= 1000
                && state.get(Field::Time, i) <= 1440
                && state.get(Field::Hours, i) <= 1440
                && state.get(Field::Employed, i) <= 1
                && state.get(Field::Employer, i) <= scenario.actors() as u64;
            let closed = [Field::Escrow, Field::Due, Field::Pipeline]
                .into_iter()
                .all(|field| state.get(field, i) == 0);
            let quotes = match state.actors[i].kind {
                0 => true,
                _ => {
                    (1..=10_000).contains(&state.get(Field::Wage, i))
                        && (u64::from(c.minimum_price)..=u64::from(c.maximum_price))
                            .contains(&state.get(Field::Price, i))
                }
            };
            limits && closed && quotes
        });
        let cash = u128::from(c.household_cash) * u128::from(c.households)
            + u128::from(c.firm_cash) * u128::from(c.firms);
        let stock = u64::from(c.capacity) * u64::from(c.packet_output) * u64::from(c.buffer_days);
        let grain_firms = u64::from(c.firms.div_ceil(3));
        let flour_firms = u64::from((c.firms + 1) / 3);
        let bread_firms = u64::from(c.firms / 3);
        let goods = [
            stock * (grain_firms + flour_firms),
            stock * (flour_firms + bread_firms),
            stock * bread_firms
                + u64::from(c.households) * u64::from(c.household_need) * u64::from(c.buffer_days),
        ];
        let valid = identities
            && resources
            && state.next_identity == scenario.actors() as u64 + 1
            && state.day <= u64::from(u32::MAX) - scenario.ring() as u64
            && u128::from(state.initial_cash) == cash
            && state.initial_goods == goods;
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

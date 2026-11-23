use super::{ActorRows, Good, company, demography, enterprise, industry, research};
use crate::{arithmetic::hex, ledger::Coordinate};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const STATE_HASH_VERSION: &str = "bad-econ/state/CubekV2/v32";
pub const USED_FIELDS: usize = Field::CartageCapacity as usize + 1;
pub const TRANSIT_FIELDS: usize = USED_FIELDS + Good::COUNT;
pub const CONSUMPTION_FIELDS: usize = TRANSIT_FIELDS + Good::COUNT;
pub const PRODUCTION_FIELDS: usize = CONSUMPTION_FIELDS + Good::COUNT;
#[repr(u32)]
#[derive(Clone, Copy)]
pub enum RetailAccount {
    Quote,
    Cost,
    Demand,
    Orders,
}

impl RetailAccount {
    pub const FIELDS: usize = 4 * Good::COUNT + 3;
    pub const PRIOR_SALES: usize = 3;
    pub const PRIOR_UNFILLED: usize = 2;
    pub const REPLACEMENT_EXPENSE: usize = 1;
}

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
    #[serde(alias = "Coats")]
    Clothes,
    Wood,
    Tools,
    Furniture,
    Cotton,
    Dye,
    Glass,
    Construction,
    Housing,
    Locomotives,
    Railway,
    LuxuryClothes,
    LuxuryFurniture,
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
    Job,
    RentTreasury,
    RentArrears,
    PropertyRegion,
    PropertyUse,
    OccupiedPlots,
    RequestedPlots,
    ReservedPlots,
    DisplacedPlots,
    ProjectStatus,
    HousingStatus,
    RentOverdueDays,
    WorkersPerPlot,
    UnscaledProduced,
    ScaleRemainder,
    RentPending,
    Landlord,
    RentalIncome,
    HousingAvailable,
    MaintenanceDays,
    BusinessOpeningCash,
    BusinessClosingCash,
    BusinessOpeningRent,
    BusinessRevenue,
    RailBuilt,
    RailOperating,
    FreightEscrow,
    RailRevenue,
    RailCapacity,
    HeatingNeed,
    HeatingServed,
    NutritionNeed,
    MillingRemainder,
    BakingRemainder,
    FarmHectares,
    HarvestedThisYear,
    OpeningGrossIncome,
    PriceQuote,
    IncomeForecast,
    FoodPrice,
    JobDaysRemaining,
    EligibleWorkers,
    CohortDelegate,
    Infants,
    Children,
    YoungAdults,
    OlderAdults,
    Seniors,
    CohortBirths,
    BackgroundDeaths,
    OccupationalDeaths,
    FamineDeaths,
    ColdDeaths,
    CohortFamineExposure,
    CohortColdExposure,
    FemaleInfants,
    FemaleChildren,
    FemaleYoungAdults,
    FemaleOlderAdults,
    FemaleSeniors,
    RecoveringMothers,
    GrainReserve,
    GrainQuote,
    GrainSales,
    ExpectedGrainSales,
    RoadFreight,
    CartageMinutes,
    CartageIncome,
    CartagePayable,
    CartageCapacity,
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
            Good::Clothes => Self::Clothes,
            Good::Wood => Self::Wood,
            Good::Tools => Self::Tools,
            Good::Furniture => Self::Furniture,
            Good::Cotton => Self::Cotton,
            Good::Dye => Self::Dye,
            Good::Glass => Self::Glass,
            Good::Construction => Self::Construction,
            Good::Housing => Self::Housing,
            Good::Locomotives => Self::Locomotives,
            Good::Railway => Self::Railway,
            Good::LuxuryClothes => Self::LuxuryClothes,
            Good::LuxuryFurniture => Self::LuxuryFurniture,
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
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub companies: Vec<company::Company>,
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
    pub fn cartage_coordinate(
        &self,
        good: Good,
        days: usize,
        ring: usize,
        actor: usize,
    ) -> Coordinate {
        Coordinate(
            ((BASE_FIELDS + (Good::COUNT + good as usize - 1) * ring + days) * self.actors.len()
                + actor) as u32,
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
        hash.update(serde_json::to_vec(&self.companies).expect("serializable company registry"));
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

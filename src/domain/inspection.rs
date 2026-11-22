use super::{
    Actor, ActorRows, Field, Good, HousingState, ProductionMethod, Scenario, State, calendar,
    diagnostics, enterprise, industry, landlords, property, railway,
};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug)]
pub struct InspectionRequest {
    identity: u64,
}
impl InspectionRequest {
    pub fn from_registry(identity: u64, identities: &[u64]) -> Result<Self> {
        Self::find(identity, identities).context("unknown actor identity")
    }
    pub(crate) fn find(identity: u64, identities: &[u64]) -> Option<Self> {
        identities.contains(&identity).then_some(Self { identity })
    }
    pub fn new(identity: u64, scenario: &Scenario) -> Result<Self> {
        scenario
            .index_of(identity)
            .map(|_| Self { identity })
            .context("unknown actor identity")
    }
    pub(crate) fn row(&self, actors: &ActorRows) -> Result<usize> {
        actors
            .index_of(self.identity)
            .context("unknown actor identity")
    }

    pub fn identity(&self) -> u64 {
        self.identity
    }
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Inspection {
    pub railway_business: Option<railway::RailwayBusiness>,
    pub business_account: Option<diagnostics::BusinessAccount>,
    pub production_method: Option<ProductionMethod>,
    pub labor_minutes: u64,
    pub output_ticks: u64,
    pub actor: Actor,
    pub day: u64,
    #[serde(default)]
    pub weather: calendar::Weather,
    pub cash: u64,
    pub gross_income: u64,
    pub payroll_reserve: u64,
    pub income_tax: u64,
    pub employee_charges: u64,
    pub rent_collected: u64,
    pub rent_arrears: u64,
    pub landlord_identity: Option<u64>,
    pub rental_business: Option<landlords::RentalBusiness>,
    pub housing_state: Option<HousingState>,
    pub rent_overdue_days: u64,
    pub inventory: [u64; Good::COUNT],
    pub transit: u64,
    pub fatigue: u64,
    pub employer: u64,
    pub contract_employer: Option<u64>,
    #[serde(default)]
    pub job_days_remaining: u64,
    pub owner_identity: u64,
    pub company_identity: u64,
    pub company_sites: Vec<u64>,
    pub workers_per_plot: u64,
    pub scale_bonus_per_mille: u32,
    pub available_minutes: u64,
    pub living_members: u64,
    #[serde(default)]
    pub heating_need: u64,
    #[serde(default)]
    pub heating_served: u64,
    pub equipment_condition: u64,
    pub safety_provision: u64,
    pub enterprise_model: enterprise::EnterpriseModel,
    pub business_scale: Option<enterprise::BusinessScale>,
    pub enterprise_state: Option<enterprise::EnterpriseState>,
    pub construction_minutes_remaining: u64,
    pub property_region: u64,
    pub occupied_plots: u64,
    pub reserved_plots: u64,
    pub subsistence_plots: u64,
    pub displaced_plots: u64,
    pub project_status: property::ProjectStatus,
    pub pending_capacity: u64,
    pub wage: u64,
    pub price: u64,
    pub planned_packets: u64,
    pub open_slots: u64,
}
impl Inspection {
    pub fn from_state(
        scenario: &Scenario,
        state: &State,
        request: &InspectionRequest,
    ) -> Result<Self> {
        let index = request.row(&state.actors)?;
        Self::from_fields(
            scenario,
            state,
            state.day,
            state.actors.at(index),
            |field| state.get(field, index),
        )
    }

    pub(crate) fn from_fields(
        scenario: &Scenario,
        state: &State,
        day: u64,
        actor: Actor,
        value: impl Fn(Field) -> u64,
    ) -> Result<Self> {
        let contract_employer = state.actors.assigned_identity(value(Field::Job));
        Ok(Inspection {
            railway_business: (actor.kind == Good::Railway as u32).then(|| {
                railway::RailwayBusiness {
                    built_segments: value(Field::RailBuilt),
                    operating_segments: value(Field::RailOperating),
                    daily_receipts: value(Field::RailRevenue),
                    daily_capacity: value(Field::RailCapacity),
                    available_units: value(Field::Railway),
                }
            }),
            business_account: (actor.kind > 0).then(|| diagnostics::BusinessAccount {
                opening_cash: value(Field::BusinessOpeningCash),
                closing_cash: value(Field::BusinessClosingCash),
                revenue: value(Field::BusinessRevenue),
            }),
            production_method: ProductionMethod::from_index(value(Field::ActiveMethod))?,
            labor_minutes: value(Field::LaborMinutes),
            output_ticks: value(Field::OutputTicks),
            day,
            weather: scenario.weather(day),
            cash: value(Field::Cash),
            gross_income: value(Field::GrossIncome),
            payroll_reserve: value(Field::PayrollReserve),
            income_tax: value(Field::TaxTreasury),
            employee_charges: value(Field::ChargeTreasury),
            rent_collected: value(Field::RentTreasury),
            rent_arrears: value(Field::RentArrears),
            landlord_identity: state.actors.assigned_identity(value(Field::Landlord)),
            rental_business: (actor.kind == Good::Housing as u32).then(|| {
                landlords::RentalBusiness {
                    homes: value(Field::OccupiedPlots) * value(Field::OutputTicks),
                    income: value(Field::RentalIncome),
                }
            }),
            housing_state: match actor.kind {
                0 => Some(HousingState::from_index(value(Field::HousingStatus))?),
                _ => None,
            },
            rent_overdue_days: value(Field::RentOverdueDays),
            inventory: Good::ALL.map(|good| value(Field::inventory(good))),
            transit: value(Field::Transit),
            fatigue: value(Field::Fatigue),
            job_days_remaining: value(Field::JobDaysRemaining),
            contract_employer,
            employer: state
                .actors
                .assigned_identity(value(Field::Employer))
                .or(contract_employer)
                .unwrap_or(0),
            owner_identity: state.actors.at(actor.owner as usize).identity,
            available_minutes: value(Field::Time),
            living_members: value(Field::LivingMembers),
            heating_need: value(Field::HeatingNeed),
            heating_served: value(Field::HeatingServed),
            equipment_condition: value(Field::EquipmentCondition),
            safety_provision: value(Field::SafetyProvision),
            business_scale: (actor.kind > 0).then(|| {
                enterprise::BusinessScale::for_business(
                    &actor,
                    value(Field::OccupiedPlots) * value(Field::OutputTicks),
                )
            }),
            enterprise_model: match value(Field::EnterpriseModel) {
                1 => enterprise::EnterpriseModel::Distributor,
                _ => enterprise::EnterpriseModel::Producer,
            },
            enterprise_state: match actor.kind {
                0 => None,
                _ => Some(enterprise::EnterpriseState::from_index(value(
                    Field::EnterpriseStatus,
                ))?),
            },
            company_identity: match actor.kind {
                0 => 0,
                _ => state.company_identity(actor.identity),
            },
            company_sites: match actor.kind {
                0 => Vec::new(),
                _ => state.company_sites(actor.identity),
            },
            workers_per_plot: value(Field::WorkersPerPlot),
            scale_bonus_per_mille: match actor.kind {
                0 => 0,
                _ => {
                    industry::ProductionScale::for_plots(value(Field::OccupiedPlots))
                        .bonus_per_mille
                }
            },
            construction_minutes_remaining: value(Field::ConstructionRemaining),
            property_region: value(Field::PropertyRegion),
            occupied_plots: value(Field::OccupiedPlots),
            reserved_plots: value(Field::ReservedPlots),
            subsistence_plots: value(Field::LandPlots),
            displaced_plots: value(Field::DisplacedPlots),
            project_status: property::ProjectStatus::from_index(value(Field::ProjectStatus))?,
            pending_capacity: value(Field::PendingCapacity),
            wage: value(Field::Wage),
            price: value(Field::Price),
            planned_packets: value(Field::Plan),
            open_slots: value(Field::Slots),
            actor,
        })
    }
}

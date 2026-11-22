use super::{
    Actor, Field, Good, HousingState, ProductionMethod, Scenario, State, calendar, diagnostics,
    enterprise, industry, landlords, property, railway,
};
use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};

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
        let index = state
            .actors
            .iter()
            .position(|actor| actor.identity == request.identity)
            .context("unknown actor identity")?;
        let contract_employer = state.actors.assigned_identity(state.get(Field::Job, index));
        Ok(Self {
            actor: state.actors.at(index),
            railway_business: (state.actors.at(index).kind == Good::Railway as u32).then(|| {
                railway::RailwayBusiness {
                    built_segments: state.get(Field::RailBuilt, index),
                    operating_segments: state.get(Field::RailOperating, index),
                    daily_receipts: state.get(Field::RailRevenue, index),
                    daily_capacity: state.get(Field::RailCapacity, index),
                    available_units: state.get(Field::Railway, index),
                }
            }),
            business_account: (state.actors.at(index).kind > 0).then(|| {
                diagnostics::BusinessAccount {
                    opening_cash: state.get(Field::BusinessOpeningCash, index),
                    closing_cash: state.get(Field::BusinessClosingCash, index),
                    revenue: state.get(Field::BusinessRevenue, index),
                }
            }),
            production_method: ProductionMethod::from_index(state.get(Field::ActiveMethod, index))?,
            labor_minutes: state.get(Field::LaborMinutes, index),
            output_ticks: state.get(Field::OutputTicks, index),
            day: state.day,
            weather: scenario.weather(state.day),
            cash: state.get(Field::Cash, index),
            gross_income: state.get(Field::GrossIncome, index),
            payroll_reserve: state.get(Field::PayrollReserve, index),
            income_tax: state.get(Field::TaxTreasury, index),
            employee_charges: state.get(Field::ChargeTreasury, index),
            rent_collected: state.get(Field::RentTreasury, index),
            rent_arrears: state.get(Field::RentArrears, index),
            landlord_identity: state
                .get(Field::Landlord, index)
                .checked_sub(1)
                .and_then(|landlord| state.actors.get(landlord as usize))
                .map(|actor| actor.identity),
            rental_business: (state.actors.at(index).kind == Good::Housing as u32).then(|| {
                landlords::RentalBusiness {
                    homes: state.get(Field::OccupiedPlots, index)
                        * state.get(Field::OutputTicks, index),
                    income: state.get(Field::RentalIncome, index),
                }
            }),
            housing_state: match state.actors.at(index).kind {
                0 => Some(HousingState::from_index(
                    state.get(Field::HousingStatus, index),
                )?),
                _ => None,
            },
            rent_overdue_days: state.get(Field::RentOverdueDays, index),
            inventory: Good::ALL.map(|good| state.get(Field::inventory(good), index)),
            transit: state.get(Field::Transit, index),
            fatigue: state.get(Field::Fatigue, index),
            job_days_remaining: state.get(Field::JobDaysRemaining, index),
            contract_employer,
            employer: state
                .actors
                .assigned_identity(state.get(Field::Employer, index))
                .or(contract_employer)
                .unwrap_or(0),
            owner_identity: state
                .actors
                .at(state.actors.at(index).owner as usize)
                .identity,
            available_minutes: state.get(Field::Time, index),
            living_members: state.get(Field::LivingMembers, index),
            heating_need: state.get(Field::HeatingNeed, index),
            heating_served: state.get(Field::HeatingServed, index),
            equipment_condition: state.get(Field::EquipmentCondition, index),
            safety_provision: state.get(Field::SafetyProvision, index),
            business_scale: (state.actors.at(index).kind > 0).then(|| {
                enterprise::BusinessScale::for_business(
                    &state.actors.at(index),
                    state.get(Field::OccupiedPlots, index) * state.get(Field::OutputTicks, index),
                )
            }),
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
            company_identity: match state.actors.at(index).kind {
                0 => 0,
                _ => state.company_identity(request.identity),
            },
            company_sites: match state.actors.at(index).kind {
                0 => Vec::new(),
                _ => state.company_sites(request.identity),
            },
            workers_per_plot: state.get(Field::WorkersPerPlot, index),
            scale_bonus_per_mille: match state.actors.at(index).kind {
                0 => 0,
                _ => {
                    industry::ProductionScale::for_plots(state.get(Field::OccupiedPlots, index))
                        .bonus_per_mille
                }
            },
            construction_minutes_remaining: state.get(Field::ConstructionRemaining, index),
            property_region: state.get(Field::PropertyRegion, index),
            occupied_plots: state.get(Field::OccupiedPlots, index),
            reserved_plots: state.get(Field::ReservedPlots, index),
            subsistence_plots: state.get(Field::LandPlots, index),
            displaced_plots: state.get(Field::DisplacedPlots, index),
            project_status: property::ProjectStatus::from_index(
                state.get(Field::ProjectStatus, index),
            )?,
            pending_capacity: state.get(Field::PendingCapacity, index),
            wage: state.get(Field::Wage, index),
            price: state.get(Field::Price, index),
            planned_packets: state.get(Field::Plan, index),
            open_slots: state.get(Field::Slots, index),
        })
    }
}

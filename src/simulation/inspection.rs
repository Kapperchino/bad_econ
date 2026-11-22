use super::{
    Simulation, decode,
    economy::EconomyStage,
    layout::{TICK_BYTES, TICK_WORDS},
};
use crate::domain::*;
use anyhow::{Context, Result};

impl Simulation {
    pub fn property_snapshot(
        &mut self,
    ) -> Result<Option<crate::domain::property::PropertySnapshot>> {
        match self.scenario.config().property.is_some() {
            false => Ok(None),
            true => {
                let state = self.download()?;
                crate::domain::property::PropertySnapshot::from_state(&self.scenario, &state)
            }
        }
    }

    pub fn inspect(&mut self, request: InspectionRequest) -> Result<Inspection> {
        let index = self
            .metadata
            .actors
            .identities()
            .iter()
            .position(|&identity| identity == request.identity())
            .context("unknown actor identity")?;
        let parameters = self.parameters(Phase::Audit, Market::Labor, index, 1)?;
        self.economy(
            EconomyStage::Inspect,
            self.scenario.fields(),
            &parameters,
            &self.summary,
        );
        let values = decode(&self.device.read_range(
            &self.summary,
            0,
            self.scenario.fields() * TICK_WORDS,
        )?);
        self.readback_bytes += (values.len() * TICK_BYTES) as u64;
        let actor = self.metadata.actors.at(index);
        let value = |field: Field| values[field as usize];
        let contract_employer = self.metadata.actors.assigned_identity(value(Field::Job));
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
            day: self.day,
            weather: self.scenario.weather(self.day),
            cash: value(Field::Cash),
            gross_income: value(Field::GrossIncome),
            payroll_reserve: value(Field::PayrollReserve),
            income_tax: value(Field::TaxTreasury),
            employee_charges: value(Field::ChargeTreasury),
            rent_collected: value(Field::RentTreasury),
            rent_arrears: value(Field::RentArrears),
            landlord_identity: value(Field::Landlord)
                .checked_sub(1)
                .and_then(|landlord| self.metadata.actors.get(landlord as usize))
                .map(|actor| actor.identity),
            rental_business: (actor.kind == Good::Housing as u32).then(|| {
                crate::domain::landlords::RentalBusiness {
                    homes: value(Field::OccupiedPlots) * value(Field::OutputTicks),
                    income: value(Field::RentalIncome),
                }
            }),
            housing_state: match actor.kind {
                0 => Some(crate::domain::HousingState::from_index(value(
                    Field::HousingStatus,
                ))?),
                _ => None,
            },
            rent_overdue_days: value(Field::RentOverdueDays),
            inventory: values[Field::Grain as usize..Field::Escrow as usize].try_into()?,
            transit: value(Field::Transit),
            fatigue: value(Field::Fatigue),
            job_days_remaining: value(Field::JobDaysRemaining),
            contract_employer,
            employer: self
                .metadata
                .actors
                .assigned_identity(value(Field::Employer))
                .or(contract_employer)
                .unwrap_or(0),
            owner_identity: self.metadata.actors.at(actor.owner as usize).identity,
            available_minutes: value(Field::Time),
            living_members: value(Field::LivingMembers),
            heating_need: value(Field::HeatingNeed),
            heating_served: value(Field::HeatingServed),
            equipment_condition: value(Field::EquipmentCondition),
            safety_provision: value(Field::SafetyProvision),
            business_scale: (actor.kind > 0).then(|| {
                crate::domain::enterprise::BusinessScale::for_business(
                    &actor,
                    value(Field::OccupiedPlots) * value(Field::OutputTicks),
                )
            }),
            enterprise_model: match value(Field::EnterpriseModel) {
                1 => crate::domain::enterprise::EnterpriseModel::Distributor,
                _ => crate::domain::enterprise::EnterpriseModel::Producer,
            },
            enterprise_state: match actor.kind {
                0 => None,
                _ => Some(crate::domain::enterprise::EnterpriseState::from_index(
                    value(Field::EnterpriseStatus),
                )?),
            },
            company_identity: match actor.kind {
                0 => 0,
                _ => self.metadata.company_identity(request.identity()),
            },
            company_sites: match actor.kind {
                0 => Vec::new(),
                _ => self.metadata.company_sites(request.identity()),
            },
            workers_per_plot: value(Field::WorkersPerPlot),
            scale_bonus_per_mille: match actor.kind {
                0 => 0,
                _ => {
                    crate::domain::industry::ProductionScale::for_plots(value(Field::OccupiedPlots))
                        .bonus_per_mille
                }
            },
            construction_minutes_remaining: value(Field::ConstructionRemaining),
            property_region: value(Field::PropertyRegion),
            occupied_plots: value(Field::OccupiedPlots),
            reserved_plots: value(Field::ReservedPlots),
            subsistence_plots: value(Field::LandPlots),
            displaced_plots: value(Field::DisplacedPlots),
            project_status: crate::domain::property::ProjectStatus::from_index(value(
                Field::ProjectStatus,
            ))?,
            pending_capacity: value(Field::PendingCapacity),
            wage: value(Field::Wage),
            price: value(Field::Price),
            planned_packets: value(Field::Plan),
            open_slots: value(Field::Slots),
            actor,
        })
    }
}

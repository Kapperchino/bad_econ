use super::layout::WORD_BITS;
use super::negotiation::BargainingConfig;
use crate::{
    arithmetic::PRICE_TABLE,
    domain::{Good, Scenario, State},
};
use cubecl::prelude::*;

#[derive(Clone, Copy, Debug, Default, Hash, PartialEq, Eq)]
pub enum Feature {
    #[default]
    Disabled,
    Enabled,
}
impl<T> From<Option<&T>> for Feature {
    fn from(value: Option<&T>) -> Self {
        match value {
            None => Self::Disabled,
            Some(_) => Self::Enabled,
        }
    }
}

#[derive(CubeType, CubeLaunch, Clone, Copy, Default)]
pub struct TaxBracket {
    pub low: u32,
    pub high: u32,
    pub rate: u32,
}
impl TaxBracket {
    pub fn launch(self) -> TaxBracketLaunch {
        TaxBracketLaunch {
            low: self.low,
            high: self.high,
            rate: self.rate,
        }
    }
}
#[derive(CubeType, CubeLaunch, Clone, Copy, Default)]
pub struct HomeProduction {
    pub milling_minutes: u32,
    pub baking_minutes: u32,
    pub cultivation_minutes: u32,
    pub harvest_start: u32,
    pub harvest_days: u32,
    pub annual_yield_per_plot: u32,
}
impl HomeProduction {
    fn new(value: crate::domain::enterprise::HouseholdProduction) -> Self {
        match value {
            crate::domain::enterprise::HouseholdProduction::Unavailable => Self {
                harvest_start: crate::domain::calendar::HARVEST_START,
                harvest_days: crate::domain::calendar::HARVEST_DAYS,
                ..Self::default()
            },
            crate::domain::enterprise::HouseholdProduction::Available {
                milling_minutes,
                baking_minutes,
                cultivation_minutes,
                harvest_start,
                harvest_days,
                annual_yield_per_plot,
            } => Self {
                milling_minutes,
                baking_minutes,
                cultivation_minutes,
                harvest_start,
                harvest_days,
                annual_yield_per_plot,
            },
        }
    }
    pub fn launch(self) -> HomeProductionLaunch {
        HomeProductionLaunch {
            milling_minutes: self.milling_minutes,
            baking_minutes: self.baking_minutes,
            cultivation_minutes: self.cultivation_minutes,
            harvest_start: self.harvest_start,
            harvest_days: self.harvest_days,
            annual_yield_per_plot: self.annual_yield_per_plot,
        }
    }
}
#[derive(CubeType, CubeLaunch)]
pub struct EconomyConfig {
    pub heating_per_person: u32,
    pub actors: u32,
    pub fields: u32,
    pub ring: u32,
    pub search_radius: u32,
    pub map_size: u32,
    pub commute_speed: u32,
    pub shipment_speed: u32,
    pub railway: super::economy::railway::RailNetwork,
    pub household_need: u32,
    #[cube(comptime)]
    pub economy: crate::domain::Economy,
    pub home: HomeProduction,
    pub consumption_intervals: Box<[u32]>,
    pub work_cap: u32,
    pub reserved_minutes: u32,
    pub forecast_weight: u32,
    pub coverage: u32,
    pub distribution_rate: u32,
    pub minimum_price: u32,
    pub maximum_price: u32,
    pub delivery_penalty: u32,
    pub reserve_low: u32,
    pub reserve_high: u32,
    pub spatial_side: u32,
    pub work_fatigue: u32,
    pub rest_recovery: u32,
    pub wage_increment: u32,
    pub starting_wage: u32,
    pub price_response: u32,
    #[cube(comptime)]
    pub payroll: Feature,
    pub tax_base_rate: u32,
    pub employee_charge: u32,
    pub household_rent: u32,
    pub firm_rent: u32,
    #[cube(comptime)]
    pub household_policy: Feature,
    pub fatigue_response: u32,
    pub rest_minutes: u32,
    pub bargaining: BargainingConfig,
    pub price_table: Box<[u32]>,
    #[cube(comptime)]
    pub home_mode: Feature,
    pub tax_brackets: Sequence<TaxBracket>,
    #[cube(comptime)]
    pub property_mode: Feature,
    pub property_regions: Box<[u32]>,
    pub grain_farms: Box<[u32]>,
    pub rental_accounts: Box<[u32]>,
    pub rental_businesses: u32,
    pub small_business_expansion: u32,
    pub industrial_expansion: u32,
}
pub struct Configuration {
    pub heating_per_person: u32,
    rentals: crate::domain::landlords::RentalRegistry,
    pub actors: u32,
    pub fields: u32,
    pub ring: u32,
    pub search_radius: u32,
    pub map_size: u32,
    pub commute_speed: u32,
    pub shipment_speed: u32,
    pub railway: super::economy::railway::Network,
    pub household_need: u32,
    pub economy: crate::domain::Economy,
    pub home: HomeProduction,
    pub consumption_intervals: burn_wgpu::CubeTensor,
    pub work_cap: u32,
    pub reserved_minutes: u32,
    pub forecast_weight: u32,
    pub coverage: u32,
    pub distribution_rate: u32,
    pub minimum_price: u32,
    pub maximum_price: u32,
    pub delivery_penalty: u32,
    pub reserve_low: u32,
    pub reserve_high: u32,
    pub spatial_side: u32,
    pub work_fatigue: u32,
    pub rest_recovery: u32,
    pub wage_increment: u32,
    pub starting_wage: u32,
    pub price_response: u32,
    pub payroll: Feature,
    pub tax_base_rate: u32,
    pub employee_charge: u32,
    pub household_rent: u32,
    pub firm_rent: u32,
    pub household_policy: Feature,
    pub fatigue_response: u32,
    pub rest_minutes: u32,
    pub bargaining: BargainingConfig,
    pub price_table: burn_wgpu::CubeTensor,
    pub home_mode: Feature,
    pub tax_brackets: Vec<TaxBracket>,
    pub property_mode: Feature,
    pub property_regions: burn_wgpu::CubeTensor,
    pub grain_farms: burn_wgpu::CubeTensor,
    pub rental_accounts: burn_wgpu::CubeTensor,
    pub rental_businesses: u32,
    pub small_business_expansion: u32,
    pub industrial_expansion: u32,
}
impl Configuration {
    pub fn new(device: &super::Device, scenario: &Scenario, state: &State) -> Self {
        let rentals = crate::domain::landlords::RentalRegistry::new(scenario, state);
        Self::with_rentals(device, scenario, state, rentals)
    }

    pub fn updated(
        &self,
        device: &super::Device,
        scenario: &Scenario,
        previous: &crate::domain::ActorRows,
        state: &State,
    ) -> Self {
        let rentals = self.rentals.updated(scenario, previous, state);
        Self::with_rentals(device, scenario, state, rentals)
    }

    pub fn update_rentals(
        &mut self,
        device: &super::Device,
        scenario: &Scenario,
        state: &State,
        changes: impl IntoIterator<Item = usize>,
    ) {
        match self.rentals.refresh(scenario, state, changes) {
            crate::domain::landlords::RentalRefresh::Current => (),
            crate::domain::landlords::RentalRefresh::Changed => {
                self.rental_accounts = device.tensor(&self.rentals.words());
                self.rental_businesses = self.rentals.accounts.len() as u32;
            }
        }
    }

    fn with_rentals(
        device: &super::Device,
        scenario: &Scenario,
        state: &State,
        rentals: crate::domain::landlords::RentalRegistry,
    ) -> Self {
        let config = scenario.config();
        Self {
            heating_per_person: crate::domain::calendar::Weather::on_day(state.day)
                .heating_units_per_person,
            small_business_expansion: config.expansion.small_business_per_mille,
            industrial_expansion: config.expansion.industrial_per_mille,
            rental_accounts: device.tensor(&rentals.words()),
            rental_businesses: rentals.accounts.len() as u32,
            rentals,
            actors: state.actors.len() as u32,
            fields: scenario.fields() as u32,
            ring: scenario.ring() as u32,
            search_radius: config.search_radius,
            map_size: config.map_size,
            commute_speed: config.commute_speed,
            shipment_speed: config.shipment_speed,
            railway: super::economy::railway::Network::new(device, scenario, state),
            household_need: config.household_need,
            economy: config.economy,
            home: HomeProduction::new(config.household_production),
            property_mode: Feature::from(config.property.as_ref()),
            grain_farms: device.tensor(
                &state
                    .actors
                    .iter()
                    .enumerate()
                    .filter(|(_, actor)| actor.kind == Good::Grain as u32)
                    .map(|(index, _)| index as u32)
                    .chain([u32::MAX])
                    .collect::<Vec<_>>(),
            ),
            property_regions: device.tensor(&config.property.as_ref().map_or_else(
                || vec![0; 6],
                |property| {
                    property
                        .regions
                        .iter()
                        .flat_map(|region| {
                            [
                                region.lower.x,
                                region.lower.y,
                                region.upper.x,
                                region.upper.y,
                                region.plots,
                                state
                                    .actors
                                    .iter()
                                    .position(|actor| actor.identity == region.landowner)
                                    .expect("validated landowner")
                                    as u32,
                            ]
                        })
                        .collect()
                },
            )),
            home_mode: match config.household_production {
                crate::domain::enterprise::HouseholdProduction::Unavailable => Feature::Disabled,
                crate::domain::enterprise::HouseholdProduction::Available { .. } => {
                    Feature::Enabled
                }
            },
            consumption_intervals: device.tensor(&Good::ALL.map(Good::consumption_interval)),
            work_cap: config.work_cap,
            reserved_minutes: config.reserved_minutes,
            forecast_weight: config.forecast_weight,
            coverage: config.coverage,
            distribution_rate: config.distribution_rate,
            minimum_price: config.minimum_price,
            maximum_price: config.maximum_price,
            delivery_penalty: config.delivery_penalty,
            reserve_low: config.reserve as u32,
            reserve_high: (config.reserve >> WORD_BITS) as u32,
            spatial_side: config.map_size.div_ceil(config.search_radius),
            work_fatigue: config.behavior.work_fatigue,
            rest_recovery: config.behavior.rest_recovery,
            wage_increment: config.behavior.wage_increment,
            starting_wage: config.wage,
            price_response: config.behavior.price_response,
            household_policy: Feature::from(config.household_policy.as_ref()),
            fatigue_response: config
                .household_policy
                .as_ref()
                .map_or(0, |policy| policy.fatigue_response),
            rest_minutes: config
                .household_policy
                .as_ref()
                .map_or(0, |policy| policy.rest_minutes),
            bargaining: BargainingConfig::new(config.bargaining.as_ref()),
            payroll: Feature::from(config.payroll.as_ref()),
            tax_base_rate: config.payroll.as_ref().map_or(0, |policy| policy.base_rate),
            household_rent: config.rent.map_or(0, |policy| policy.household_daily),
            firm_rent: config.rent.map_or(0, |policy| policy.firm_daily),
            employee_charge: config
                .payroll
                .as_ref()
                .map_or(0, |policy| policy.employee_charge),
            tax_brackets: config
                .payroll
                .as_ref()
                .map(|policy| {
                    policy
                        .brackets
                        .iter()
                        .map(|bracket| TaxBracket {
                            low: bracket.threshold as u32,
                            high: (bracket.threshold >> WORD_BITS) as u32,
                            rate: bracket.marginal_rate,
                        })
                        .collect()
                })
                .unwrap_or_default(),
            price_table: device.tensor(&PRICE_TABLE),
        }
    }
    pub fn launch(&self) -> EconomyConfigLaunch {
        EconomyConfigLaunch {
            heating_per_person: self.heating_per_person,
            small_business_expansion: self.small_business_expansion,
            industrial_expansion: self.industrial_expansion,
            rental_accounts: self.rental_accounts.clone().into_buffer_arg(),
            rental_businesses: self.rental_businesses,
            property_mode: self.property_mode,
            property_regions: self.property_regions.clone().into_buffer_arg(),
            grain_farms: self.grain_farms.clone().into_buffer_arg(),
            actors: self.actors,
            fields: self.fields,
            ring: self.ring,
            search_radius: self.search_radius,
            map_size: self.map_size,
            commute_speed: self.commute_speed,
            shipment_speed: self.shipment_speed,
            railway: self.railway.launch(),
            household_need: self.household_need,
            economy: self.economy,
            home: self.home.launch(),
            consumption_intervals: self.consumption_intervals.clone().into_buffer_arg(),
            work_cap: self.work_cap,
            reserved_minutes: self.reserved_minutes,
            forecast_weight: self.forecast_weight,
            coverage: self.coverage,
            distribution_rate: self.distribution_rate,
            minimum_price: self.minimum_price,
            maximum_price: self.maximum_price,
            delivery_penalty: self.delivery_penalty,
            reserve_low: self.reserve_low,
            reserve_high: self.reserve_high,
            spatial_side: self.spatial_side,
            work_fatigue: self.work_fatigue,
            rest_recovery: self.rest_recovery,
            wage_increment: self.wage_increment,
            starting_wage: self.starting_wage,
            price_response: self.price_response,
            payroll: self.payroll,
            tax_base_rate: self.tax_base_rate,
            employee_charge: self.employee_charge,
            household_rent: self.household_rent,
            firm_rent: self.firm_rent,
            household_policy: self.household_policy,
            fatigue_response: self.fatigue_response,
            rest_minutes: self.rest_minutes,
            bargaining: self.bargaining.launch(),
            price_table: self.price_table.clone().into_buffer_arg(),
            home_mode: self.home_mode,
            tax_brackets: self
                .tax_brackets
                .iter()
                .map(|bracket| bracket.launch())
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{Actor, Experiment, Position};
    use anyhow::Result;

    enum RentalChange {
        Migration,
        Ownership,
        Entry,
        Housing,
        BuildingMigration,
        Capacity,
    }

    #[test]
    fn cached_rental_configuration_matches_rebuilt_assignments_on_gpu() -> Result<()> {
        let scenario = Scenario::industrial(128, 42, Experiment::Baseline)?;
        let device = crate::simulation::tests::device(64)?;
        let mut state = scenario.initial()?;
        let mut cached = Configuration::new(&device, &scenario, &state);
        for change in [
            RentalChange::Migration,
            RentalChange::Ownership,
            RentalChange::Entry,
            RentalChange::Housing,
            RentalChange::BuildingMigration,
            RentalChange::Capacity,
        ] {
            let previous = state.actors.clone();
            let housing = state
                .actors
                .iter()
                .position(|actor| actor.kind == Good::Housing as u32)
                .unwrap();
            let changed = match change {
                RentalChange::Migration => {
                    let mut actor = state.actors.at(2);
                    actor.position = Position { x: 1, y: 1 };
                    state.actors.set(2, actor);
                    2
                }
                RentalChange::Ownership => {
                    let firm = state
                        .actors
                        .iter()
                        .position(|actor| actor.kind == Good::Bread as u32)
                        .unwrap();
                    let mut actor = state.actors.at(firm);
                    actor.owner = 2;
                    state.actors.set(firm, actor);
                    firm
                }
                RentalChange::Entry | RentalChange::Housing => {
                    let kind = match change {
                        RentalChange::Entry => Good::Bread,
                        _ => Good::Housing,
                    };
                    state.actors.extend([Actor {
                        identity: state.next_identity,
                        position: Position { x: 2, y: 2 },
                        kind: kind as u32,
                        owner: 3,
                        capacity: 1,
                    }]);
                    state.next_identity += 1;
                    state.actors.len() - 1
                }
                RentalChange::BuildingMigration => {
                    let mut actor = state.actors.at(housing);
                    actor.position = Position { x: 1, y: 2 };
                    state.actors.set(housing, actor);
                    housing
                }
                RentalChange::Capacity => {
                    let mut actor = state.actors.at(housing);
                    actor.capacity += 1;
                    state.actors.set(housing, actor);
                    housing
                }
            };
            match previous.len() == state.actors.len() {
                true => cached.update_rentals(&device, &scenario, &state, [changed]),
                false => cached = cached.updated(&device, &scenario, &previous, &state),
            }
            let rebuilt = Configuration::new(&device, &scenario, &state);
            assert_eq!(cached.rental_businesses, rebuilt.rental_businesses);
            assert_eq!(
                device.read(&cached.rental_accounts)?,
                device.read(&rebuilt.rental_accounts)?
            );
        }
        Ok(())
    }
}

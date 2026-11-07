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
            crate::domain::enterprise::HouseholdProduction::Unavailable => Self::default(),
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
    pub actors: u32,
    pub fields: u32,
    pub ring: u32,
    pub search_radius: u32,
    pub map_size: u32,
    pub commute_speed: u32,
    pub shipment_speed: u32,
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
}
pub struct Configuration {
    pub actors: u32,
    pub fields: u32,
    pub ring: u32,
    pub search_radius: u32,
    pub map_size: u32,
    pub commute_speed: u32,
    pub shipment_speed: u32,
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
}
impl Configuration {
    pub fn new(device: &super::Device, scenario: &Scenario, state: &State) -> Self {
        let config = scenario.config();
        Self {
            actors: state.actors.len() as u32,
            fields: scenario.fields() as u32,
            ring: scenario.ring() as u32,
            search_radius: config.search_radius,
            map_size: config.map_size,
            commute_speed: config.commute_speed,
            shipment_speed: config.shipment_speed,
            household_need: config.household_need,
            economy: config.economy,
            home: HomeProduction::new(config.household_production),
            property_mode: Feature::from(config.property.as_ref()),
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
            property_mode: self.property_mode,
            property_regions: self.property_regions.clone().into_buffer_arg(),
            actors: self.actors,
            fields: self.fields,
            ring: self.ring,
            search_radius: self.search_radius,
            map_size: self.map_size,
            commute_speed: self.commute_speed,
            shipment_speed: self.shipment_speed,
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
